"""
GC10-DET (Supervisely JSON) -> YOLO format
10 classes steel surface defect
"""
import json
import os
import shutil
import random
from pathlib import Path

RAW_ANN = Path(r"C:\dev\active\yolo26-industrial-vision\datasets\gc10-raw\GC10-DET\ann")
RAW_IMG = Path(r"C:\dev\active\yolo26-industrial-vision\datasets\gc10-raw\GC10-DET\images")
OUT = Path(r"C:\dev\active\yolo26-industrial-vision\datasets\gc10-det")

CLASS_NAMES = [
    'crease', 'crescent_gap', 'inclusion', 'oil_spot', 'punching_hole',
    'rolled_pit', 'silk_spot', 'waist_folding', 'water_spot', 'welding_line'
]
CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}
TRAIN_RATIO = 0.8
random.seed(42)


def convert():
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'val']:
        (OUT / 'images' / split).mkdir(parents=True, exist_ok=True)
        (OUT / 'labels' / split).mkdir(parents=True, exist_ok=True)

    samples = []
    for ann_file in sorted(RAW_ANN.glob("*.json")):
        img_name = ann_file.stem  # e.g. img_01_xxx.jpg
        img_path = RAW_IMG / img_name
        if not img_path.exists():
            continue
        samples.append((img_path, ann_file))

    print(f"Found {len(samples)} samples")

    random.shuffle(samples)
    n_train = int(len(samples) * TRAIN_RATIO)
    splits = {'train': samples[:n_train], 'val': samples[n_train:]}

    total_objs = 0
    small_objs = 0

    for split, split_samples in splits.items():
        count = 0
        for img_path, ann_file in split_samples:
            with open(ann_file, encoding='utf-8') as f:
                data = json.load(f)

            img_w = data['size']['width']
            img_h = data['size']['height']

            labels = []
            for obj in data.get('objects', []):
                cls_title = obj['classTitle']
                if cls_title == 'waist folding':
                    cls_title = 'waist_folding'
                if cls_title not in CLASS_MAP:
                    continue

                cls_id = CLASS_MAP[cls_title]
                pts = obj['points']['exterior']
                x1, y1 = pts[0]
                x2, y2 = pts[1]

                cx = (x1 + x2) / 2 / img_w
                cy = (y1 + y2) / 2 / img_h
                w = abs(x2 - x1) / img_w
                h = abs(y2 - y1) / img_h

                cx = max(0, min(1, cx))
                cy = max(0, min(1, cy))
                w = max(0, min(1, w))
                h = max(0, min(1, h))

                if w > 0 and h > 0:
                    labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    total_objs += 1
                    if abs(x2 - x1) < 32 or abs(y2 - y1) < 32:
                        small_objs += 1

            # Copy image + write label (even if no labels, still copy for background)
            stem = img_path.stem.replace('.jpg', '').replace('.png', '')
            dst_img = OUT / 'images' / split / img_path.name
            shutil.copy2(img_path, dst_img)

            lbl_name = stem + '.txt'
            if labels:
                (OUT / 'labels' / split / lbl_name).write_text("\n".join(labels))
            count += 1

        print(f"  {split}: {count} images")

    yaml = f"""path: {str(OUT).replace(chr(92), '/')}
train: images/train
val: images/val

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    (OUT / 'data.yaml').write_text(yaml, encoding='utf-8')

    print(f"\nTotal: {total_objs} objects, {len(CLASS_NAMES)} classes")
    print(f"Small objects (<32px): {small_objs} ({small_objs/max(total_objs,1)*100:.1f}%)")
    print(f"data.yaml: {OUT / 'data.yaml'}")


if __name__ == "__main__":
    convert()
