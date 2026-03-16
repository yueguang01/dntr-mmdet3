from mmdet.registry import MODELS

BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS


def build_backbone(cfg):
    return MODELS.build(cfg)


def build_head(cfg):
    return MODELS.build(cfg)


def build_roi_extractor(cfg):
    return MODELS.build(cfg)


def build_shared_head(cfg):
    return MODELS.build(cfg)
