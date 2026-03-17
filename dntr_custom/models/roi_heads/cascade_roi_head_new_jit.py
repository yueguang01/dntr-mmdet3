import torch
import torch.nn as nn
import time
import numpy as np
try:
    from numba import jit
except Exception:
    def jit(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def _decorator(func):
            return func

        return _decorator

from dntr_custom.core import (bbox2result, bbox2roi, bbox_mapping,
                              build_assigner, build_sampler,
                              merge_aug_bboxes, merge_aug_masks,
                              multiclass_nms)
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin
from mmdet.models.utils import unpack_gt_instances
from mmengine.structures import InstanceData

######### t2t model ##########
from .t2t_models.t2t_vit_woshuffle import T2T_module
####### evit topk model ######
import math
from functools import partial
from .evit.helpers import complement_idx
##############################


######### t2t model ##########
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # print(self.fc1.grad())
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., keep_rate=1.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.keep_rate = keep_rate
        assert 0 < keep_rate <= 1, "keep_rate must > 0 and <= 1, got {0}".format(keep_rate)

    def forward(self, x, keep_rate=None, tokens=None):
        if keep_rate is None:
            keep_rate = self.keep_rate
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        left_tokens = N - 1
        if self.keep_rate < 1 and keep_rate < 1 or tokens is not None:  # double check the keep rate
            left_tokens = math.ceil(keep_rate * (N - 1))
            if tokens is not None:
                left_tokens = tokens
            if left_tokens == N - 1:
                return x, None, None, None, left_tokens
            assert left_tokens >= 1
            cls_attn = attn[:, :, 0, 1:]  # [B, H, N-1]
            cls_attn = cls_attn.mean(dim=1)  # [B, N-1]
            _, idx = torch.topk(cls_attn, left_tokens, dim=1, largest=True, sorted=True)  # [B, left_tokens]
            # cls_idx = torch.zeros(B, 1, dtype=idx.dtype, device=idx.device)
            # index = torch.cat([cls_idx, idx + 1], dim=1)
            index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C]

            return x, index, idx, cls_attn, left_tokens

        return  x, None, None, None, left_tokens

## no topK , only reture atten score
class Token_Pair_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x, tokens=None):
        B, N, C = x.shape
        # print(B,N,C)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        # print(q.shape)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn[:,:,0,1] = 0  # cls see box as 0
        attn[:,:,1,0] = 0  # box see cls as 0
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        ##### for mask #####
        # with torch.no_grad():
        # attn[:,:,0,1] = 0  # cls not see box
        # attn[:,:,1,0] = 0  # box not see cls
        # attn[:,:,0,1].detach()
        # attn[:,:,1,0].detach()
        ####################


        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        cls_attn = attn[:, :, 0, 2:]  # [B, H, N-1]
        cls_attn = cls_attn.mean(dim=1)  # [B, N-1]

        box_attn = attn[:, :, 1, 2:]
        box_attn = box_attn.mean(dim=1)
        # print(x.shape)

        return  x, cls_attn, box_attn

## pair the class and bbox token
## return cls + cls_g token and bbox + bbox_g token
# @jit()
def gpu_pair(token_compare, index, B, N):
    # id = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    idx = np.zeros([B,N-2],dtype=np.int64)
    for b in range(B):
      box_token_cnt = 0
      cls_token_cnt = 15
      for j in index[b]:
          if token_compare[b,j]:
              idx[b,box_token_cnt] = int(j)
              box_token_cnt += 1
          else:
              idx[b,cls_token_cnt] = int(j)
              cls_token_cnt -= 1
    return idx

class Token_pair_block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Token_Pair_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.norm2 = norm_layer(dim)
        # mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.mlp_hidden_dim = mlp_hidden_dim

    def forward(self, general_token , cls_token , box_token, keep_rate=None, tokens=None, get_idx=False):

        ## cat two tokens
        cat_token = torch.cat((box_token, general_token), dim=1)
        cat_token = torch.cat((cls_token, cat_token), dim=1)


        B, N, C = cat_token.shape

        ## calculate the attention score
        cat_token_temp, cls_attn, box_attn = self.attn(self.norm1(cat_token))
        ## skip connection
        cat_token = cat_token + self.drop_path(cat_token_temp)


        ## seperate the token
        general_token = cat_token[:,2:]
        cls_token = cat_token[:,0:1]
        box_token = cat_token[:,1:2]

        ## sum the attention scores to decide order
        sum_attn = cls_attn + box_attn

        ## get the order index
        _ , index = torch.sort(sum_attn, descending=True)

        ## compare which is more important
        token_compare = torch.gt(box_attn,cls_attn) # True if (box_attn>cls_attn)

        ## to CPU accerate
        index = index.cpu().numpy()
        token_compare = token_compare.cpu().numpy()



        ## get pair index
        idx = gpu_pair(token_compare,index,B,N)

        idx = torch.from_numpy(idx).to(torch.device('cuda:0'))
        idx = idx.unsqueeze(-1).expand(-1, -1, C)

        ## using idx to choose token
        cat_token = torch.gather(general_token,dim=1,index=idx)
        cls_task_token = torch.cat((cls_token,cat_token[:,0:8,:]),dim=1)
        box_task_token = torch.cat((box_token,cat_token[:,8:16,:]),dim=1)


        return cls_task_token , box_task_token


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_rate=0.,
                 fuse_token=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop, keep_rate=keep_rate)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.keep_rate = keep_rate
        self.mlp_hidden_dim = mlp_hidden_dim
        self.fuse_token = fuse_token

    def forward(self, x, keep_rate=None, tokens=None, get_idx=False):
        if keep_rate is None:
            keep_rate = self.keep_rate  # this is for inference, use the default keep rate
        B, N, C = x.shape

        tmp, index, idx, cls_attn, left_tokens = self.attn(self.norm1(x), keep_rate, tokens)
        x = x + self.drop_path(tmp)

        if index is not None:
            # B, N, C = x.shape
            non_cls = x[:, 1:]
            x_others = torch.gather(non_cls, dim=1, index=index)  # [B, left_tokens, C]

            if self.fuse_token:
                compl = complement_idx(idx, N - 1)  # [B, N-1-left_tokens]
                non_topk = torch.gather(non_cls, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))  # [B, N-1-left_tokens, C]

                non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)  # [B, N-1-left_tokens]
                extra_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)  # [B, 1, C]
                x = torch.cat([x[:, 0:1], x_others, extra_token], dim=1)
            else:
                x = torch.cat([x[:, 0:1], x_others], dim=1)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        n_tokens = x.shape[1] - 1
        if get_idx and index is not None:
            return x, n_tokens, idx
        return x, n_tokens, None
##################################

@HEADS.register_module()
class Cascade_t2t_new_jit_mask_RoIHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Cascade roi head including one bbox head and one mask head.

    https://arxiv.org/abs/1712.00726
    """

    def __init__(self,
                 num_stages,
                 stage_loss_weights,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert shared_head is None, \
            'Shared head is not supported in Cascade RCNN anymore'

        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        super(Cascade_t2t_new_jit_mask_RoIHead, self).__init__(
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            shared_head=shared_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        ####### t2t_module #######
        # self.token_to_token = T2T_module(
        #         img_size=7, tokens_type='performer', in_chans=256, embed_dim=128, token_dim=100)
        self.keep_rate = 0.5
        # t2t_token should be checked by feature map dim
        num_head = 16
        # evit_token = math.ceil(self.keep_rate*t2t_token) + 2
        self.in_chans = 256
        self.token_dim = 100
        self.embed_dim = 128
        self.bbox_token = [nn.Parameter(torch.zeros(1, self.in_chans * 3 * 3)).to(torch.device('cuda:0')) for _ in range(self.num_stages)]
        self.cls_token = [nn.Parameter(torch.zeros(1, self.in_chans * 3 * 3)).to(torch.device('cuda:0')) for _ in range(self.num_stages)]
        self.token_to_token = nn.ModuleList([T2T_module(
                                img_size=7, tokens_type='transformer', in_chans=self.in_chans, embed_dim=self.embed_dim, token_dim=self.token_dim,mask=True) for _ in range(self.num_stages)])
        self.t2t_bbox_head = nn.ModuleList([nn.Linear(9*128, 4) for _ in range(self.num_stages)])
        # norm_layer = partial(nn.LayerNorm, eps=1e-6)


        ## do the bipartite token pair ##
        self.token_pair = nn.ModuleList([Token_pair_block(dim=self.embed_dim, num_heads=num_head)  for _ in range(self.num_stages)]) ## 8

        # self.blk = Block(keep_rate=0.7, fuse_token=True)
        # self.cls_blk = ModuleList([Block(dim=self.embed_dim, num_heads=t2t_token, keep_rate=self.keep_rate, fuse_token=True) for _ in range(self.num_stages)])
        # self.bbox_blk = ModuleList([Block(dim=self.embed_dim, num_heads=t2t_token, keep_rate=self.keep_rate, fuse_token=True) for _ in range(self.num_stages)])
        # self.norm = norm_layer(self.embed_dim)

        ###### aitod dataset #####
        # Output dim: num_classes + 1 (background). For AI-TOD: 8 classes + 1 = 9
        self.t2t_cls_head = nn.ModuleList([nn.Linear(9*128, 9) for _ in range(self.num_stages)])
        ##### visdrone dataset ###
        # self.t2t_cls_head = ModuleList([nn.Linear(128, 11) for _ in range(self.num_stages)])
        ##########################

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize box head and box roi extractor.

        Args:
            bbox_roi_extractor (dict): Config of box roi extractor.
            bbox_head (dict): Config of box in box head.
        """
        self.bbox_roi_extractor = nn.ModuleList()
        self.bbox_head = nn.ModuleList()
        if not isinstance(bbox_roi_extractor, list):
            bbox_roi_extractor = [
                bbox_roi_extractor for _ in range(self.num_stages)
            ]
        if not isinstance(bbox_head, list):
            bbox_head = [bbox_head for _ in range(self.num_stages)]
        assert len(bbox_roi_extractor) == len(bbox_head) == self.num_stages
        for roi_extractor, head in zip(bbox_roi_extractor, bbox_head):
            self.bbox_roi_extractor.append(build_roi_extractor(roi_extractor))
            self.bbox_head.append(build_head(head))

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize mask head and mask roi extractor.

        Args:
            mask_roi_extractor (dict): Config of mask roi extractor.
            mask_head (dict): Config of mask in mask head.
        """
        self.mask_head = nn.ModuleList()
        if not isinstance(mask_head, list):
            mask_head = [mask_head for _ in range(self.num_stages)]
        assert len(mask_head) == self.num_stages
        for head in mask_head:
            self.mask_head.append(build_head(head))
        if mask_roi_extractor is not None:
            self.share_roi_extractor = False
            self.mask_roi_extractor = nn.ModuleList()
            if not isinstance(mask_roi_extractor, list):
                mask_roi_extractor = [
                    mask_roi_extractor for _ in range(self.num_stages)
                ]
            assert len(mask_roi_extractor) == self.num_stages
            for roi_extractor in mask_roi_extractor:
                self.mask_roi_extractor.append(
                    build_roi_extractor(roi_extractor))
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor

    def init_assigner_sampler(self):
        """Initialize assigner and sampler for each stage."""
        self.bbox_assigner = []
        self.bbox_sampler = []
        if self.train_cfg is not None:
            for idx, rcnn_train_cfg in enumerate(self.train_cfg):
                self.bbox_assigner.append(
                    build_assigner(rcnn_train_cfg.assigner))
                self.current_stage = idx
                self.bbox_sampler.append(
                    build_sampler(rcnn_train_cfg.sampler, context=self))

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            for i in range(self.num_stages):
                bbox_results = self._bbox_forward(i, x, rois)
                outs = outs + (bbox_results['cls_score'],
                               bbox_results['bbox_pred'])
        # mask heads
        if self.with_mask:
            mask_rois = rois[:100]
            for i in range(self.num_stages):
                mask_results = self._mask_forward(i, x, mask_rois)
                outs = outs + (mask_results['mask_pred'], )
        return outs

    def _bbox_forward(self, stage, x, rois):
        """Box head forward function used in both training and testing."""
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)
        # do not support caffe_c4 model anymore
        # cls_score, bbox_pred = bbox_head(bbox_feats)

        ####### t2t_module #######
        token_to_token = self.token_to_token[stage]
        # bbox_blk = self.bbox_blk[stage]
        t2t_bbox_head = self.t2t_bbox_head[stage]

        # cls_blk = self.cls_blk[stage]
        t2t_cls_head = self.t2t_cls_head[stage]

        cls_token = self.cls_token[stage].expand(bbox_feats.shape[0], -1, -1)
        bbox_token = self.bbox_token[stage].expand(bbox_feats.shape[0], -1, -1)
        token_pair = self.token_pair[stage]

        t2t_feats, cls_token, bbox_token = token_to_token(bbox_feats, cls_token=cls_token, bbox_token=bbox_token, random_shuffle_forward=True)

        ######### token pairing #############
        t2t_feats_cls, t2t_feats_bbox = token_pair(general_token=t2t_feats, cls_token=cls_token, box_token=bbox_token)

        # print('cls_token:',t2t_feats_cls.size())
        ########## evit topk ##########
        # t2t_feats_cls = torch.cat((cls_token, t2t_feats), dim=1)
        # t2t_feats_cls, cls_n_token, cls_token_idx = cls_blk(t2t_feats_cls)
        ################################
        # t2t_feats_cls = self.norm(t2t_feats_cls)
        # t2t_feats_cls = t2t_feats_cls[:, 0]
        t2t_feats_cls = torch.flatten(t2t_feats_cls, start_dim=1)
        cls_score = t2t_cls_head(t2t_feats_cls)

        ########## evit topk ##########
        # t2t_feats_bbox = torch.cat((bbox_token, t2t_feats), dim=1)
        # t2t_feats_bbox, bbox_n_token, bbox_token_idx = bbox_blk(t2t_feats_bbox)
        ################################
        # t2t_feats_bbox = self.norm(t2t_feats_bbox)
        # t2t_feats_bbox = t2t_feats_bbox[:, 0]
        t2t_feats_bbox = torch.flatten(t2t_feats_bbox, start_dim=1)
        bbox_pred = t2t_bbox_head(t2t_feats_bbox)

        # print("cls_score:", cls_score.size())
        # bbox_pred = self.t2t_bbox_head(t2t_feats)
        ##########################

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, stage, x, sampling_results, gt_bboxes,
                            gt_labels, rcnn_train_cfg):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(stage, x, rois)
        bbox_targets = self.bbox_head[stage].get_targets(
            sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg)
        loss_bbox = self.bbox_head[stage].loss(bbox_results['cls_score'],
                                               bbox_results['bbox_pred'], rois,
                                               *bbox_targets)

        bbox_results.update(
            loss_bbox=loss_bbox, rois=rois, bbox_targets=bbox_targets)
        return bbox_results

    def _mask_forward(self, stage, x, rois):
        """Mask head forward function used in both training and testing."""
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        mask_feats = mask_roi_extractor(x[:mask_roi_extractor.num_inputs],
                                        rois)
        # do not support caffe_c4 model anymore
        mask_pred = mask_head(mask_feats)

        mask_results = dict(mask_pred=mask_pred)
        return mask_results

    def _mask_forward_train(self,
                            stage,
                            x,
                            sampling_results,
                            gt_masks,
                            rcnn_train_cfg,
                            bbox_feats=None):
        """Run forward function and calculate loss for mask head in
        training."""
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        mask_results = self._mask_forward(stage, x, pos_rois)

        mask_targets = self.mask_head[stage].get_targets(
            sampling_results, gt_masks, rcnn_train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head[stage].loss(mask_results['mask_pred'],
                                               mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask)
        return mask_results

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()
        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg[i]
            lw = self.stage_loss_weights[i]

            # assign gts and sample proposals
            sampling_results = []
            if self.with_bbox or self.with_mask:
                bbox_assigner = self.bbox_assigner[i]
                bbox_sampler = self.bbox_sampler[i]
                num_imgs = len(img_metas)
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]

                for j in range(num_imgs):
                    pred_instances = InstanceData(priors=proposal_list[j])
                    gt_instances = InstanceData(bboxes=gt_bboxes[j], labels=gt_labels[j])
                    gt_instances_ignore = (
                        InstanceData(bboxes=gt_bboxes_ignore[j])
                        if gt_bboxes_ignore[j] is not None else None
                    )

                    assign_result = bbox_assigner.assign(
                        pred_instances,
                        gt_instances,
                        gt_instances_ignore,
                    )
                    sampling_result = bbox_sampler.sample(
                        assign_result,
                        pred_instances,
                        gt_instances,
                        feats=[lvl_feat[j][None] for lvl_feat in x])
                    sampling_results.append(sampling_result)

            # bbox head forward and loss
            bbox_results = self._bbox_forward_train(i, x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    rcnn_train_cfg)

            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{i}.{name}'] = (
                    value * lw if 'loss' in name else value)

            # mask head forward and loss
            if self.with_mask:
                mask_results = self._mask_forward_train(
                    i, x, sampling_results, gt_masks, rcnn_train_cfg,
                    bbox_results['bbox_feats'])
                for name, value in mask_results['loss_mask'].items():
                    losses[f's{i}.{name}'] = (
                        value * lw if 'loss' in name else value)

            # refine bboxes
            if i < self.num_stages - 1:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                # bbox_targets is a tuple
                roi_labels = bbox_results['bbox_targets'][0]
                with torch.no_grad():
                    roi_labels = torch.where(
                        roi_labels == self.bbox_head[i].num_classes,
                        bbox_results['cls_score'][:, :-1].argmax(1),
                        roi_labels)
                    proposal_list = self.bbox_head[i].refine_bboxes(
                        bbox_results['rois'], roi_labels,
                        bbox_results['bbox_pred'], pos_is_gts, img_metas)
        # print(losses)
        return losses

    def loss(self, x, rpn_results_list, batch_data_samples):
        """MMDet3 training interface adapter.

        The original DNTR implementation uses MMDet2-style forward_train(...).
        This method converts MMDet3 inputs and dispatches to forward_train.
        """
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas = \
            unpack_gt_instances(batch_data_samples)

        proposal_list = [results.bboxes for results in rpn_results_list]
        gt_bboxes = [gt_instances.bboxes for gt_instances in batch_gt_instances]
        gt_labels = [gt_instances.labels for gt_instances in batch_gt_instances]

        gt_bboxes_ignore = []
        for ignored in batch_gt_instances_ignore:
            if ignored is None:
                gt_bboxes_ignore.append(None)
            else:
                gt_bboxes_ignore.append(ignored.bboxes)

        gt_masks = [getattr(gt_instances, 'masks', None) for gt_instances in batch_gt_instances]

        return self.forward_train(
            x=x,
            img_metas=batch_img_metas,
            proposal_list=proposal_list,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels,
            gt_bboxes_ignore=gt_bboxes_ignore,
            gt_masks=gt_masks,
        )

    def simple_test(self, x, proposal_list, img_metas, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        num_imgs = len(proposal_list)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg

        rois = bbox2roi(proposal_list)
        for i in range(self.num_stages):
            bbox_results = self._bbox_forward(i, x, rois)

            # split batch bbox prediction back to each image
            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']
            num_proposals_per_img = tuple(
                len(proposals) for proposals in proposal_list)
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head[i].bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
            ms_scores.append(cls_score)

            if i < self.num_stages - 1:
                bbox_label = [s[:, :-1].argmax(dim=1) for s in cls_score]
                rois = torch.cat([
                    # MMDet3: regress_by_class expects [N, 4] coords (no batch_idx).
                    # Strip batch_idx (col 0), regress, then re-prepend batch_idx.
                    torch.cat([
                        rois[j][:, :1],
                        self.bbox_head[i].regress_by_class(
                            rois[j][:, 1:], bbox_label[j], bbox_pred[j], img_metas[j]),
                    ], dim=1)
                    for j in range(num_imgs)
                ])

        # average scores of each image by stages
        cls_score = [
            sum([score[i] for score in ms_scores]) / float(len(ms_scores))
            for i in range(num_imgs)
        ]

        # apply bbox post-processing to each image
        det_bboxes = []
        det_labels = []

        # predict_by_feat processes all images at once
        result_instances = self.bbox_head[-1].predict_by_feat(
            rois=rois,
            cls_scores=cls_score,
            bbox_preds=bbox_pred,
            batch_img_metas=img_metas,
            rcnn_test_cfg=rcnn_test_cfg,
            rescale=rescale,
        )
        for inst in result_instances:
            if len(inst) > 0:
                # inst.bboxes: [K, 4], inst.scores: [K], inst.labels: [K]
                det_bboxes.append(
                    torch.cat([inst.bboxes, inst.scores.unsqueeze(1)], dim=1))
                det_labels.append(inst.labels)
            else:
                det_bboxes.append(
                    torch.zeros((0, 5), device=rois[0].device))
                det_labels.append(
                    torch.zeros((0,), dtype=torch.long, device=rois[0].device))

        if torch.onnx.is_in_onnx_export():
            return det_bboxes, det_labels
        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head[-1].num_classes)
            for i in range(num_imgs)
        ]
        ms_bbox_result['ensemble'] = bbox_results

        if self.with_mask:
            if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[[] for _ in range(mask_classes)]
                                for _ in range(num_imgs)]
            else:
                if rescale and not isinstance(scale_factors[0], float):
                    scale_factors = [
                        torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                        for scale_factor in scale_factors
                    ]
                _bboxes = [
                    det_bboxes[i][:, :4] *
                    scale_factors[i] if rescale else det_bboxes[i][:, :4]
                    for i in range(len(det_bboxes))
                ]
                mask_rois = bbox2roi(_bboxes)
                num_mask_rois_per_img = tuple(
                    _bbox.size(0) for _bbox in _bboxes)
                aug_masks = []
                for i in range(self.num_stages):
                    mask_results = self._mask_forward(i, x, mask_rois)
                    mask_pred = mask_results['mask_pred']
                    # split batch mask prediction back to each image
                    mask_pred = mask_pred.split(num_mask_rois_per_img, 0)
                    aug_masks.append(
                        [m.sigmoid().cpu().numpy() for m in mask_pred])

                # apply mask post-processing to each image individually
                segm_results = []
                for i in range(num_imgs):
                    if det_bboxes[i].shape[0] == 0:
                        segm_results.append(
                            [[]
                             for _ in range(self.mask_head[-1].num_classes)])
                    else:
                        aug_mask = [mask[i] for mask in aug_masks]
                        merged_masks = merge_aug_masks(
                            aug_mask, [[img_metas[i]]] * self.num_stages,
                            rcnn_test_cfg)
                        segm_result = self.mask_head[-1].get_seg_masks(
                            merged_masks, _bboxes[i], det_labels[i],
                            rcnn_test_cfg, ori_shapes[i], scale_factors[i],
                            rescale)
                        segm_results.append(segm_result)
            ms_segm_result['ensemble'] = segm_results

        if self.with_mask:
            results = list(
                zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble']))
        else:
            results = ms_bbox_result['ensemble']

        return results

    # ------------------------------------------------------------------
    # MMDet3 interface: predict() wraps the MMDet2 simple_test() logic
    # ------------------------------------------------------------------
    def predict(self, x, rpn_results_list, batch_data_samples, rescale=True):
        """MMDet3 test interface — adapts to simple_test internally."""
        import numpy as np
        from mmdet.structures import DetDataSample
        from mmengine.structures import InstanceData

        # 1. Convert rpn_results_list (InstanceData) → MMDet2 proposal_list
        proposal_list = []
        for r in rpn_results_list:
            bboxes = r.bboxes  # [N, 4]
            # Only pass 4-col bboxes: MMDet3's bbox2roi keeps all columns,
            # so passing [N, 5] would produce [M, 6] ROIs which RoIAlign rejects.
            proposal_list.append(bboxes[:, :4])

        # 2. Convert batch_data_samples → MMDet2 img_metas list
        img_metas = []
        for ds in batch_data_samples:
            meta = ds.metainfo
            scale_factor = meta.get('scale_factor', (1.0, 1.0))
            if isinstance(scale_factor, (int, float)):
                scale_factor = (scale_factor, scale_factor)
            elif isinstance(scale_factor, np.ndarray):
                scale_factor = tuple(scale_factor.tolist())
            img_metas.append({
                'img_shape': meta.get('img_shape'),
                'ori_shape': meta.get('ori_shape'),
                'scale_factor': scale_factor,
                'pad_shape': meta.get('pad_shape', meta.get('img_shape')),
            })

        # 3. Run MMDet2-style inference
        bbox_results = self.simple_test(x, proposal_list, img_metas, rescale=rescale)

        # 4. Convert MMDet2 bbox_results → list of DetDataSample
        result_list = []
        for i, (ds, bbox_result) in enumerate(zip(batch_data_samples, bbox_results)):
            if isinstance(bbox_result, tuple):
                # (bbox_result_list, segm_result_list) when with_mask=True
                bbox_res, _ = bbox_result
            else:
                bbox_res = bbox_result  # list of arrays per class

            # bbox_res: list[ndarray], one per class, each shape [K, 5]
            bboxes_list, scores_list, labels_list = [], [], []
            for cls_idx, cls_bboxes in enumerate(bbox_res):
                if len(cls_bboxes) > 0:
                    bboxes_list.append(torch.as_tensor(cls_bboxes[:, :4], dtype=torch.float32))
                    scores_list.append(torch.as_tensor(cls_bboxes[:, 4], dtype=torch.float32))
                    labels_list.append(torch.full((len(cls_bboxes),), cls_idx, dtype=torch.long))

            pred_instances = InstanceData()
            if bboxes_list:
                pred_instances.bboxes = torch.cat(bboxes_list, dim=0)
                pred_instances.scores = torch.cat(scores_list, dim=0)
                pred_instances.labels = torch.cat(labels_list, dim=0)
            else:
                pred_instances.bboxes = torch.zeros((0, 4), dtype=torch.float32)
                pred_instances.scores = torch.zeros((0,), dtype=torch.float32)
                pred_instances.labels = torch.zeros((0,), dtype=torch.long)

            # Return InstanceData directly; TwoStageDetector.add_pred_to_datasample
            # will assign it to data_sample.pred_instances.
            result_list.append(pred_instances)

        return result_list

    def aug_test(self, features, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        rcnn_test_cfg = self.test_cfg
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(features, img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            flip_direction = img_meta[0]['flip_direction']

            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip, flip_direction)
            # "ms" in variable names means multi-stage
            ms_scores = []

            rois = bbox2roi([proposals])
            for i in range(self.num_stages):
                bbox_results = self._bbox_forward(i, x, rois)
                ms_scores.append(bbox_results['cls_score'])

                if i < self.num_stages - 1:
                    bbox_label = bbox_results['cls_score'][:, :-1].argmax(
                        dim=1)
                    rois = self.bbox_head[i].regress_by_class(
                        rois, bbox_label, bbox_results['bbox_pred'],
                        img_meta[0])

            cls_score = sum(ms_scores) / float(len(ms_scores))
            bboxes, scores = self.bbox_head[-1].get_bboxes(
                rois,
                cls_score,
                bbox_results['bbox_pred'],
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                                rcnn_test_cfg.score_thr,
                                                rcnn_test_cfg.nms,
                                                rcnn_test_cfg.max_per_img)

        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head[-1].num_classes)

        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                segm_result = [[]
                               for _ in range(self.mask_head[-1].num_classes)]
            else:
                aug_masks = []
                aug_img_metas = []
                for x, img_meta in zip(features, img_metas):
                    img_shape = img_meta[0]['img_shape']
                    scale_factor = img_meta[0]['scale_factor']
                    flip = img_meta[0]['flip']
                    flip_direction = img_meta[0]['flip_direction']
                    _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
                                           scale_factor, flip, flip_direction)
                    mask_rois = bbox2roi([_bboxes])
                    for i in range(self.num_stages):
                        mask_results = self._mask_forward(i, x, mask_rois)
                        aug_masks.append(
                            mask_results['mask_pred'].sigmoid().cpu().numpy())
                        aug_img_metas.append(img_meta)
                merged_masks = merge_aug_masks(aug_masks, aug_img_metas,
                                               self.test_cfg)

                ori_shape = img_metas[0][0]['ori_shape']
                segm_result = self.mask_head[-1].get_seg_masks(
                    merged_masks,
                    det_bboxes,
                    det_labels,
                    rcnn_test_cfg,
                    ori_shape,
                    scale_factor=1.0,
                    rescale=False)
            return [(bbox_result, segm_result)]
        else:
            return [bbox_result]