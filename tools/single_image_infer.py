import argparse
import os
import sys

import mmcv
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Register custom modules before model build.
import dntr_custom.models.task_modules.assigners.ranking_assigner
import dntr_custom.models.task_modules.iou_calculators.metric_calculator
import dntr_custom.models.roi_heads.cascade_roi_head_new_jit

from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmengine.dataset import Compose
from mmdet.registry import MODELS, TRANSFORMS
from mmdet.models import DetDataPreprocessor
from mmdet.visualization import DetLocalVisualizer

AITOD_CLASSES = (
    "airplane", "bridge", "storage-tank", "ship",
    "swimming-pool", "vehicle", "person", "wind-mill",
)


def parse_args():
    parser = argparse.ArgumentParser(description="DNTR MMDet3 single image inference")
    parser.add_argument("image", help="Input image path")
    parser.add_argument("config", help="Model config path")
    parser.add_argument("checkpoint", help="Model checkpoint path")
    parser.add_argument("--out-file", default="pred_result.png")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-thr", type=float, default=0.3)
    return parser.parse_args()


def build_model(config_path, checkpoint_path, device):
    cfg = Config.fromfile(config_path)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, checkpoint_path, map_location="cpu", strict=False)
    model.dataset_meta = {"classes": AITOD_CLASSES}
    model.data_preprocessor = DetDataPreprocessor(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
    )
    model.to(device)
    model.eval()
    return model


def run_inference(model, image_path):
    pipeline_cfg = [
        dict(type="LoadImageFromFile", backend_args=None),
        dict(type="Resize", scale=(1333, 800), keep_ratio=True),
        dict(type="PackDetInputs"),
    ]
    pipeline = Compose([TRANSFORMS.build(t) for t in pipeline_cfg])
    raw = pipeline(dict(img_path=image_path))
    batch = {
        "inputs": [raw["inputs"]],
        "data_samples": [raw["data_samples"]],
    }
    with torch.no_grad():
        batch = model.data_preprocessor(batch, False)
        results = model(**batch, mode="predict")
    return results[0]


def main():
    args = parse_args()
    model = build_model(args.config, args.checkpoint, args.device)
    result = run_inference(model, args.image)

    img = mmcv.imread(args.image, channel_order="rgb")
    visualizer = DetLocalVisualizer()
    visualizer.dataset_meta = {"classes": AITOD_CLASSES}
    visualizer.add_datasample(
        "prediction", img, data_sample=result,
        draw_gt=False, show=False,
        pred_score_thr=args.score_thr,
        out_file=args.out_file,
    )
    print(f"Saved: {args.out_file}")


if __name__ == "__main__":
    main()
