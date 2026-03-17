import torch

from mmdet.models.task_modules.assigners.assign_result import AssignResult
from mmdet.models.task_modules.assigners.base_assigner import BaseAssigner
from mmdet.registry import TASK_UTILS


@TASK_UTILS.register_module()
class RankingAssigner(BaseAssigner):
    """Assign top-k proposals to each GT based on a ranking metric.

    This module preserves the behavior of the original DNTR assigner while
    using MMDet3 registries and task-module APIs.
    """

    def __init__(
        self,
        ignore_iof_thr=-1,
        ignore_wrt_candidates=True,
        gpu_assign_thr=-1,
        iou_calculator=None,
        assign_metric='iou',
        topk=1,
    ):
        if iou_calculator is None:
            iou_calculator = dict(type='BboxOverlaps2D')
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr
        self.iou_calculator = TASK_UTILS.build(iou_calculator)
        self.assign_metric = assign_metric
        self.topk = topk

    def assign(self, pred_instances, gt_instances, gt_instances_ignore=None, **kwargs):
        # MMDet3: inputs are InstanceData objects; extract tensors
        # pred_instances.priors holds anchor boxes (from anchor_head)
        bboxes = pred_instances.priors if hasattr(pred_instances, 'priors') else pred_instances.bboxes
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels if hasattr(gt_instances, 'labels') else None
        gt_bboxes_ignore = (
            gt_instances_ignore.bboxes
            if gt_instances_ignore is not None and len(gt_instances_ignore) > 0
            else None
        )

        assign_on_cpu = (
            (self.gpu_assign_thr > 0) and (gt_bboxes.shape[0] > self.gpu_assign_thr)
        )

        if assign_on_cpu:
            device = bboxes.device
            bboxes = bboxes.cpu()
            gt_bboxes = gt_bboxes.cpu()
            if gt_bboxes_ignore is not None:
                gt_bboxes_ignore = gt_bboxes_ignore.cpu()
            if gt_labels is not None:
                gt_labels = gt_labels.cpu()

        overlaps = self.iou_calculator(gt_bboxes, bboxes, mode=self.assign_metric)

        if (
            self.ignore_iof_thr > 0
            and gt_bboxes_ignore is not None
            and gt_bboxes_ignore.numel() > 0
            and bboxes.numel() > 0
        ):
            if self.ignore_wrt_candidates:
                ignore_overlaps = self.iou_calculator(
                    bboxes, gt_bboxes_ignore, mode='iof'
                )
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = self.iou_calculator(
                    gt_bboxes_ignore, bboxes, mode='iof'
                )
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

        assign_result = self.assign_wrt_ranking(overlaps, gt_labels)

        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            if assign_result.labels is not None:
                assign_result.labels = assign_result.labels.to(device)
        return assign_result

    def assign_wrt_ranking(self, overlaps, gt_labels=None):
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)
        assigned_gt_inds = overlaps.new_full((num_bboxes,), -1, dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            max_overlaps = overlaps.new_zeros((num_bboxes,))
            if num_gts == 0:
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes,), -1, dtype=torch.long)
            return AssignResult(num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        max_overlaps, _ = overlaps.max(dim=0)
        gt_max_overlaps, _ = overlaps.topk(self.topk, dim=1, largest=True, sorted=True)

        assigned_gt_inds[(max_overlaps >= 0) & (max_overlaps < 0.3)] = 0

        for i in range(num_gts):
            for j in range(self.topk):
                max_overlap_inds = overlaps[i, :] == gt_max_overlaps[i, j]
                assigned_gt_inds[max_overlap_inds] = i + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
            pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)
