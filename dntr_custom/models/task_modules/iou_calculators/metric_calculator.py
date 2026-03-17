import math
import torch

from mmdet.registry import TASK_UTILS


@TASK_UTILS.register_module()
class BboxDistanceMetric:
    """Overlap and distance metrics used by DNTR ranking assignment."""

    def __init__(self, constant=12.7):
        self.constant = constant

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        assert bboxes1.size(-1) in [0, 4, 5]
        assert bboxes2.size(-1) in [0, 4, 5]
        if bboxes2.size(-1) == 5:
            bboxes2 = bboxes2[..., :4]
        if bboxes1.size(-1) == 5:
            bboxes1 = bboxes1[..., :4]
        return bbox_overlaps(
            bboxes1,
            bboxes2,
            mode=mode,
            is_aligned=is_aligned,
            constant=self.constant,
        )

    def __repr__(self):
        return f'{self.__class__.__name__}()'


def bbox_overlaps(
    bboxes1,
    bboxes2,
    mode='iou',
    is_aligned=False,
    eps=1e-6,
    constant=12.7,
    weight=2,
):
    assert mode in ['iou', 'iof', 'giou', 'normalized_giou', 'ciou', 'diou', 'nwd', 'dotd']
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # The original implementation only supports non-aligned pairwise mode.
    if is_aligned:
        raise NotImplementedError('BboxDistanceMetric does not support is_aligned=True')

    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]
    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)

    if rows * cols == 0:
        return bboxes1.new(batch_shape + (rows, cols))

    # Normalize malformed boxes to avoid negative width/height causing NaN cascades.
    b1_x1 = torch.minimum(bboxes1[..., 0], bboxes1[..., 2])
    b1_y1 = torch.minimum(bboxes1[..., 1], bboxes1[..., 3])
    b1_x2 = torch.maximum(bboxes1[..., 0], bboxes1[..., 2])
    b1_y2 = torch.maximum(bboxes1[..., 1], bboxes1[..., 3])
    b2_x1 = torch.minimum(bboxes2[..., 0], bboxes2[..., 2])
    b2_y1 = torch.minimum(bboxes2[..., 1], bboxes2[..., 3])
    b2_x2 = torch.maximum(bboxes2[..., 0], bboxes2[..., 2])
    b2_y2 = torch.maximum(bboxes2[..., 1], bboxes2[..., 3])
    bboxes1 = torch.stack([b1_x1, b1_y1, b1_x2, b1_y2], dim=-1)
    bboxes2 = torch.stack([b2_x1, b2_y1, b2_x2, b2_y2], dim=-1)

    area1 = ((bboxes1[..., 2] - bboxes1[..., 0]).clamp(min=0)
             * (bboxes1[..., 3] - bboxes1[..., 1]).clamp(min=0))
    area2 = ((bboxes2[..., 2] - bboxes2[..., 0]).clamp(min=0)
             * (bboxes2[..., 3] - bboxes2[..., 1]).clamp(min=0))

    lt = torch.max(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
    rb = torch.min(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])

    wh = (rb - lt).clamp(min=0)
    overlap = wh[..., 0] * wh[..., 1]

    union = area1[..., None] + area2[..., None, :] - overlap + eps

    if mode in ['giou', 'normalized_giou', 'ciou', 'diou']:
        enclosed_lt = torch.min(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
        enclosed_rb = torch.max(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])

    eps_t = union.new_tensor([eps])
    union = torch.max(union, eps_t)
    ious = overlap / union

    if mode in ['iou', 'iof']:
        return torch.nan_to_num(ious, nan=0.0, posinf=1.0, neginf=0.0)

    if mode in ['giou', 'normalized_giou', 'ciou', 'diou']:
        enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
        enclose_area = torch.max(enclose_area, eps_t)
        gious = ious - (enclose_area - union) / enclose_area

    if mode == 'giou':
        return torch.nan_to_num(gious, nan=0.0, posinf=1.0, neginf=-1.0)

    if mode == 'normalized_giou':
        out = (1 + gious) / 2
        return torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)

    if mode == 'diou':
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]
        center_distance = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + eps

        enclosed_diagonal_distances = (
            enclose_wh[..., 0] * enclose_wh[..., 0] + enclose_wh[..., 1] * enclose_wh[..., 1]
        )
        dious = ious - center_distance / torch.max(enclosed_diagonal_distances, eps_t)
        out = torch.clamp(dious, min=-1.0, max=1.0)
        return torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)

    if mode == 'ciou':
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]
        center_distance = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + eps

        enclosed_diagonal_distances = (
            enclose_wh[..., 0] * enclose_wh[..., 0] + enclose_wh[..., 1] * enclose_wh[..., 1]
        )

        w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0] + eps
        h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1] + eps
        w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0] + eps
        h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1] + eps

        factor = 4 / math.pi ** 2
        v = factor * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
        cious = ious - (
            center_distance / torch.max(enclosed_diagonal_distances, eps_t)
            + v ** 2 / torch.max(1 - ious + v, eps_t)
        )
        out = torch.clamp(cious, min=-1.0, max=1.0)
        return torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)

    if mode == 'nwd':
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]
        center_distance = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + eps

        w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0] + eps
        h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1] + eps
        w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0] + eps
        h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1] + eps

        wh_distance = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / (weight ** 2)
        wassersteins = torch.sqrt(torch.clamp(center_distance + wh_distance, min=0))
        out = torch.exp(-wassersteins / max(constant, eps))
        return torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)

    # mode == 'dotd'
    center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
    center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
    whs = center1[..., :2] - center2[..., :2]
    center_distance = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + eps
    distance = torch.sqrt(torch.clamp(center_distance, min=0))
    out = torch.exp(-distance / max(constant, eps))
    return torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
