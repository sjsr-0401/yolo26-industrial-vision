"""
DeepPCB -> YOLO format 변환
DeepPCB format: x1,y1,x2,y2,type (per line)
Types: 1-open, 2-short, 3-mousebite, 4-spur, 5-copper, 6-pinhole
Image size: 640x640
"""

import os
import shutil
import random
from pathlib import Path

RAW = Path(r"C:\dev\active\yolo26-industrial-vision\datasets\deeppcb-raw\DeepPCB\PCBData")
OUT = Path(r"C:\dev\active\yolo26-industrial-vision\datasets\deeppcb")

CLASS_NAMES = ['open', 'short', 'mousebite', 'spur', 'copper', 'pinhole']
IMG_W, IMG_H = 640, 640
TRAIN_RATIO = 0.8

random.seed(42)


def convert():
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'val']:
        (OUT / 'images' / split).mkdir(parents=True, exist_ok=True)
        (OUT / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Collect all test images + annotations
    # Structure: group*/XXXXX/XXXXX_test.jpg (images)
    #            group*/XXXXX_not/XXXXX.txt   (annotations)
    samples = []
    for group_dir in sorted(RAW.iterdir()):
        if not group_dir.is_dir() or group_dir.name.endswith('.txt'):
            continue
        for sub_dir in sorted(group_dir.iterdir()):
            if not sub_dir.is_dir():
                continue
            if sub_dir.name.endswith('_not'):
                # This is annotation dir, find matching image dir
                img_dir_name = sub_dir.name.replace('_not', '')
                img_dir = group_dir / img_dir_name
                if not img_dir.exists():
                    continue
                for ann_file in sorted(sub_dir.glob("*.txt")):
                    stem = ann_file.stem
                    test_img = img_dir / f"{stem}_test.jpg"
                    if not test_img.exists():
                        continue
                    samples.append((test_img, ann_file))

    print(f"Found {len(samples)} samples")

    # Shuffle and split
    random.shuffle(samples)
    n_train = int(len(samples) * TRAIN_RATIO)
    splits = {
        'train': samples[:n_train],
        'val': samples[n_train:]
    }

    total = 0
    total_objs = 0
    small_objs = 0

    for split, split_samples in splits.items():
        for test_img, ann_file in split_samples:
            # Read annotation
            labels = []
            with open(ann_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        parts = line.strip().split(',')
                    if len(parts) < 5:
                        continue

                    x1 = float(parts[0])
                    y1 = float(parts[1])
                    x2 = float(parts[2])
                    y2 = float(parts[3])
                    cls_type = int(parts[4])

                    if cls_type < 1 or cls_type > 6:
                        continue

                    cls_id = cls_type - 1  # 0-indexed

                    # Convert to YOLO format
                    cx = (x1 + x2) / 2 / IMG_W
                    cy = (y1 + y2) / 2 / IMG_H
                    w = (x2 - x1) / IMG_W
                    h = (y2 - y1) / IMG_H

                    cx = max(0, min(1, cx))
                    cy = max(0, min(1, cy))
                    w = max(0, min(1, w))
                    h = max(0, min(1, h))

                    if w > 0 and h > 0:
                        labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                        total_objs += 1
                        # Small object check
                        if (x2 - x1) < 32 or (y2 - y1) < 32:
                            small_objs += 1

            if labels:
                # Copy image
                dst_img = OUT / 'images' / split / test_img.name
                shutil.copy2(test_img, dst_img)

                # Write label
                lbl_name = test_img.stem.replace('_test', '') + '.txt'
                (OUT / 'labels' / split / lbl_name).write_text("\n".join(labels))
                total += 1

        print(f"  {split}: {len(split_samples)} -> {total} images written")

    # Create data.yaml
    yaml = f"""path: {str(OUT).replace(chr(92), '/')}
train: images/train
val: images/val

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    (OUT / 'data.yaml').write_text(yaml, encoding='utf-8')

    print(f"\nTotal: {total} images, {total_objs} objects")
    print(f"Small objects (<32px): {small_objs} ({small_objs/max(total_objs,1)*100:.1f}%)")
    print(f"data.yaml: {OUT / 'data.yaml'}")


if __name__ == "__main__":
    convert()
