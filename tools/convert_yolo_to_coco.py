import argparse
import json
import os
from pathlib import Path
from PIL import Image

CLASSES = [
    'airplane', 'bridge', 'storage-tank', 'ship',
    'swimming-pool', 'vehicle', 'person', 'wind-mill'
]


def yolo_to_xywh(cx, cy, w, h, img_w, img_h):
    bw = w * img_w
    bh = h * img_h
    x = (cx * img_w) - bw / 2.0
    y = (cy * img_h) - bh / 2.0
    return x, y, bw, bh


def convert_split(root, split, start_image_id=1, start_ann_id=1):
    images_dir = root / split / 'images'
    labels_dir = root / split / 'labels'

    image_files = sorted(
        [p for p in images_dir.glob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}]
    )

    images = []
    annotations = []
    image_id = start_image_id
    ann_id = start_ann_id

    for img_path in image_files:
        with Image.open(img_path) as img:
            width, height = img.size

        rel_file_name = f'{split}/images/{img_path.name}'
        images.append(
            dict(id=image_id, file_name=rel_file_name, width=width, height=height)
        )

        label_path = labels_dir / f'{img_path.stem}.txt'
        if label_path.exists():
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        continue

                    cls_id = int(float(parts[0]))
                    cx, cy, w, h = map(float, parts[1:])
                    x, y, bw, bh = yolo_to_xywh(cx, cy, w, h, width, height)

                    # Clamp boxes to image boundary.
                    x = max(0.0, min(x, width - 1.0))
                    y = max(0.0, min(y, height - 1.0))
                    bw = max(0.0, min(bw, width - x))
                    bh = max(0.0, min(bh, height - y))

                    if bw <= 0 or bh <= 0:
                        continue

                    annotations.append(
                        dict(
                            id=ann_id,
                            image_id=image_id,
                            category_id=cls_id + 1,  # COCO category_id starts from 1
                            bbox=[x, y, bw, bh],
                            area=bw * bh,
                            iscrowd=0,
                            segmentation=[],
                        )
                    )
                    ann_id += 1

        image_id += 1

    categories = [
        dict(id=i + 1, name=name, supercategory='object')
        for i, name in enumerate(CLASSES)
    ]

    coco = dict(
        images=images,
        annotations=annotations,
        categories=categories,
    )

    return coco, image_id, ann_id


def main():
    parser = argparse.ArgumentParser(description='Convert YOLO labels to COCO json for AI-TOD folder layout')
    parser.add_argument(
        '--root',
        default=os.environ.get('DNTR_DATA_ROOT', '/data/AI-TOD'),
        help='Dataset root with train/ and val/ folders',
    )
    parser.add_argument('--out-dir', default=None, help='Output annotations directory (default: <root>/annotations)')
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / 'annotations'
    out_dir.mkdir(parents=True, exist_ok=True)

    train_coco, next_image_id, next_ann_id = convert_split(root, 'train', 1, 1)
    val_coco, _, _ = convert_split(root, 'val', next_image_id, next_ann_id)

    train_json = out_dir / 'instances_train.json'
    val_json = out_dir / 'instances_val.json'

    with open(train_json, 'w', encoding='utf-8') as f:
        json.dump(train_coco, f)
    with open(val_json, 'w', encoding='utf-8') as f:
        json.dump(val_coco, f)

    print(f'Saved: {train_json}')
    print(f'Saved: {val_json}')
    print(f"Train images: {len(train_coco['images'])}, anns: {len(train_coco['annotations'])}")
    print(f"Val images: {len(val_coco['images'])}, anns: {len(val_coco['annotations'])}")


if __name__ == '__main__':
    main()
