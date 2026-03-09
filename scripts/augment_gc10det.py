"""
GC10-DET 데이터 증강 파이프라인
- Albumentations 오프라인 증강
- 소수 클래스 오버샘플링 (crease 56 → 500+, rolled_pit 76 → 500+)
- SAM 기반 Copy-Paste (결함 마스크 추출 → 배경 합성)

목표: 1,835장 → ~5,000장 (클래스 균형 맞춤)
"""
import os
import sys
import random
import shutil
import json
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

try:
    import albumentations as A
except ImportError:
    os.system(f"{sys.executable} -m pip install albumentations -q")
    import albumentations as A

random.seed(42)
np.random.seed(42)

# ============================================================
# Configuration
# ============================================================
PROJECT = Path(r"C:\dev\active\yolo26-industrial-vision")
SRC_IMG = PROJECT / "datasets" / "gc10-det" / "images" / "train"
SRC_LBL = PROJECT / "datasets" / "gc10-det" / "labels" / "train"
VAL_IMG = PROJECT / "datasets" / "gc10-det" / "images" / "val"
VAL_LBL = PROJECT / "datasets" / "gc10-det" / "labels" / "val"

# Output: augmented dataset
OUT = PROJECT / "datasets" / "gc10-det-aug"
OUT_TRAIN_IMG = OUT / "images" / "train"
OUT_TRAIN_LBL = OUT / "labels" / "train"
OUT_VAL_IMG = OUT / "images" / "val"
OUT_VAL_LBL = OUT / "labels" / "val"

CLASS_NAMES = [
    'crease', 'crescent_gap', 'inclusion', 'oil_spot', 'punching_hole',
    'rolled_pit', 'silk_spot', 'waist_folding', 'water_spot', 'welding_line'
]

# Current distribution (train):
# crease:56, crescent_gap:212, inclusion:292, oil_spot:445,
# punching_hole:265, rolled_pit:76, silk_spot:695,
# waist_folding:122, water_spot:283, welding_line:423
#
# Target: each class ~400-500 instances minimum
# Oversample multiplier per class
TARGET_MIN = 400
CURRENT_COUNTS = {
    0: 56,   # crease → 7x
    1: 212,  # crescent_gap → 2x
    2: 292,  # inclusion → 2x
    3: 445,  # oil_spot → 1x (enough)
    4: 265,  # punching_hole → 2x
    5: 76,   # rolled_pit → 6x
    6: 695,  # silk_spot → 1x (enough)
    7: 122,  # waist_folding → 4x
    8: 283,  # water_spot → 2x
    9: 423,  # welding_line → 1x (enough)
}


# ============================================================
# Albumentations Pipeline
# ============================================================
def get_augmentation_pipeline(level="medium"):
    """
    References:
    - CLAHE: Zuiderveld, Graphics Gems IV, 1994
    - ElasticTransform: Simard et al., ICDAR 2003
    - GridDistortion: simulates real-world surface deformation
    - CoarseDropout: DeVries & Taylor, "Improved Regularization", 2017
    """
    if level == "light":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
        ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.3, label_fields=['class_ids']))

    elif level == "medium":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.Affine(shift_limit=0.05, scale_limit=0.15, rotate_limit=10, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.3),
            A.GaussNoise(std_range=(0.03, 0.1), p=0.3),
            A.Blur(blur_limit=3, p=0.1),
        ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.3, label_fields=['class_ids']))

    else:  # heavy
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(shift_limit=0.08, scale_limit=0.2, rotate_limit=15, p=0.6),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.4),
            A.GaussNoise(std_range=(0.05, 0.15), p=0.4),
            A.ElasticTransform(alpha=60, sigma=60 * 0.05, p=0.2),
            A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.2),
            A.CoarseDropout(num_holes_range=(1, 6), hole_height_range=(8, 32),
                           hole_width_range=(8, 32), p=0.2),
            A.Blur(blur_limit=5, p=0.15),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.3, label_fields=['class_ids']))


# ============================================================
# Bbox Copy-Paste (simplified SAM alternative)
# ============================================================
def bbox_copy_paste(src_img, src_bboxes, src_classes, dst_img, dst_bboxes, dst_classes):
    """
    Extract defect regions from src and paste onto dst.
    Simplified version of Ghiasi et al. (CVPR 2021) Copy-Paste.
    Uses bbox regions with alpha blending for natural edges.
    """
    h, w = dst_img.shape[:2]
    result_img = dst_img.copy()
    result_bboxes = list(dst_bboxes)
    result_classes = list(dst_classes)

    for bbox, cls_id in zip(src_bboxes, src_classes):
        cx, cy, bw, bh = bbox
        # Source region in pixels
        sh, sw = src_img.shape[:2]
        sx1 = max(0, int((cx - bw/2) * sw))
        sy1 = max(0, int((cy - bh/2) * sh))
        sx2 = min(sw, int((cx + bw/2) * sw))
        sy2 = min(sh, int((cy + bh/2) * sh))

        if sx2 - sx1 < 4 or sy2 - sy1 < 4:
            continue

        patch = src_img[sy1:sy2, sx1:sx2].copy()
        patch_h, patch_w = patch.shape[:2]

        # Random position on dst
        max_dx = max(0, w - patch_w)
        max_dy = max(0, h - patch_h)
        if max_dx == 0 or max_dy == 0:
            continue

        dx = random.randint(0, max_dx)
        dy = random.randint(0, max_dy)

        # Alpha blend edges (feathering)
        mask = np.ones((patch_h, patch_w), dtype=np.float32)
        feather = min(8, patch_h // 4, patch_w // 4)
        if feather > 1:
            for i in range(feather):
                alpha = (i + 1) / feather
                mask[i, :] *= alpha
                mask[-(i+1), :] *= alpha
                mask[:, i] *= alpha
                mask[:, -(i+1)] *= alpha

        mask3 = np.stack([mask]*3, axis=-1)
        roi = result_img[dy:dy+patch_h, dx:dx+patch_w]
        blended = (patch * mask3 + roi * (1 - mask3)).astype(np.uint8)
        result_img[dy:dy+patch_h, dx:dx+patch_w] = blended

        # New bbox in YOLO format
        new_cx = (dx + patch_w / 2) / w
        new_cy = (dy + patch_h / 2) / h
        new_bw = patch_w / w
        new_bh = patch_h / h
        result_bboxes.append((new_cx, new_cy, new_bw, new_bh))
        result_classes.append(cls_id)

    return result_img, result_bboxes, result_classes


# ============================================================
# Utils
# ============================================================
def read_labels(label_path):
    """Read YOLO format labels."""
    bboxes = []
    class_ids = []
    if not label_path.exists():
        return bboxes, class_ids
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(float(parts[0]))
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                bboxes.append([cx, cy, w, h])
                class_ids.append(cls)
    return bboxes, class_ids


def write_labels(label_path, bboxes, class_ids):
    """Write YOLO format labels."""
    with open(label_path, 'w') as f:
        for bbox, cls in zip(bboxes, class_ids):
            f.write(f"{cls} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")


def get_class_image_map(img_dir, lbl_dir):
    """Map class_id -> list of (img_path, label_path)."""
    class_map = defaultdict(list)
    for img_path in sorted(img_dir.glob("*")):
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
            continue
        lbl_path = lbl_dir / (img_path.stem + '.txt')
        bboxes, class_ids = read_labels(lbl_path)
        for cls in set(class_ids):
            class_map[cls].append((img_path, lbl_path))
    return class_map


# ============================================================
# Main Pipeline
# ============================================================
def main():
    print("=" * 60)
    print("  GC10-DET Data Augmentation Pipeline")
    print("=" * 60)

    # Create output directories
    for d in [OUT_TRAIN_IMG, OUT_TRAIN_LBL, OUT_VAL_IMG, OUT_VAL_LBL]:
        d.mkdir(parents=True, exist_ok=True)

    # Step 0: Copy validation set (unchanged)
    print("\n[Step 0] Copying validation set (unchanged)...")
    val_count = 0
    for img_path in VAL_IMG.glob("*"):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            shutil.copy2(img_path, OUT_VAL_IMG / img_path.name)
            lbl_src = VAL_LBL / (img_path.stem + '.txt')
            if lbl_src.exists():
                shutil.copy2(lbl_src, OUT_VAL_LBL / lbl_src.name)
            val_count += 1
    print(f"  Copied {val_count} val images")

    # Step 1: Copy all original training images
    print("\n[Step 1] Copying original training images...")
    orig_count = 0
    for img_path in sorted(SRC_IMG.glob("*")):
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
            continue
        shutil.copy2(img_path, OUT_TRAIN_IMG / img_path.name)
        lbl_src = SRC_LBL / (img_path.stem + '.txt')
        if lbl_src.exists():
            shutil.copy2(lbl_src, OUT_TRAIN_LBL / lbl_src.name)
        orig_count += 1
    print(f"  Copied {orig_count} original images")

    # Step 2: Class-aware oversampling with augmentation
    print("\n[Step 2] Class-aware oversampling + Albumentations...")
    class_map = get_class_image_map(SRC_IMG, SRC_LBL)

    aug_medium = get_augmentation_pipeline("medium")
    aug_heavy = get_augmentation_pipeline("heavy")

    aug_count = 0
    new_class_counts = dict(CURRENT_COUNTS)

    for cls_id in range(len(CLASS_NAMES)):
        current = CURRENT_COUNTS.get(cls_id, 0)
        if current >= TARGET_MIN:
            print(f"  {CLASS_NAMES[cls_id]}: {current} (enough, skip)")
            continue

        needed = TARGET_MIN - current
        sources = class_map.get(cls_id, [])
        if not sources:
            print(f"  {CLASS_NAMES[cls_id]}: no source images, skip")
            continue

        multiplier = max(1, needed // len(sources) + 1)
        print(f"  {CLASS_NAMES[cls_id]}: {current} -> target {TARGET_MIN} "
              f"(need {needed}, {len(sources)} sources, ~{multiplier}x each)")

        generated = 0
        for round_idx in range(multiplier + 2):  # extra rounds to ensure target
            if generated >= needed:
                break

            for img_path, lbl_path in sources:
                if generated >= needed:
                    break

                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                bboxes, class_ids = read_labels(lbl_path)
                if not bboxes:
                    continue

                # Alternate between medium and heavy augmentation
                aug = aug_heavy if round_idx % 2 == 0 else aug_medium

                try:
                    result = aug(
                        image=img,
                        bboxes=bboxes,
                        class_ids=class_ids
                    )
                except Exception:
                    continue

                if not result['bboxes']:
                    continue

                # Check if target class survived augmentation
                if cls_id not in result['class_ids']:
                    continue

                # Save
                aug_name = f"aug_{CLASS_NAMES[cls_id]}_{aug_count:05d}"
                cv2.imwrite(str(OUT_TRAIN_IMG / f"{aug_name}.jpg"), result['image'])
                write_labels(
                    OUT_TRAIN_LBL / f"{aug_name}.txt",
                    result['bboxes'],
                    result['class_ids']
                )

                aug_count += 1
                generated += 1

        new_class_counts[cls_id] = current + generated
        print(f"    -> Generated {generated} augmented images")

    # Step 3: Copy-Paste for extreme minority classes
    print("\n[Step 3] Copy-Paste augmentation for minority classes...")
    minority_classes = [0, 5, 7]  # crease, rolled_pit, waist_folding

    # Collect all training images as potential backgrounds
    all_train_imgs = sorted(OUT_TRAIN_IMG.glob("*.jpg"))[:500]  # limit for speed

    cp_count = 0
    for cls_id in minority_classes:
        sources = class_map.get(cls_id, [])
        if not sources:
            continue

        target_extra = max(0, 150 - (new_class_counts.get(cls_id, 0) - CURRENT_COUNTS.get(cls_id, 0)))
        if target_extra <= 0:
            continue

        print(f"  Copy-Paste {CLASS_NAMES[cls_id]}: generating {target_extra} more...")
        generated = 0

        for i in range(target_extra * 3):  # try more, some may fail
            if generated >= target_extra:
                break

            # Random source (with target class) and destination
            src_path, src_lbl = random.choice(sources)
            dst_path = random.choice(all_train_imgs)
            dst_lbl_path = OUT_TRAIN_LBL / (dst_path.stem + '.txt')

            src_img = cv2.imread(str(src_path))
            dst_img = cv2.imread(str(dst_path))
            if src_img is None or dst_img is None:
                continue

            src_bboxes, src_classes = read_labels(src_lbl)
            dst_bboxes, dst_classes = read_labels(dst_lbl_path)

            # Only paste bboxes of target class
            target_bboxes = [b for b, c in zip(src_bboxes, src_classes) if c == cls_id]
            target_classes = [cls_id] * len(target_bboxes)

            if not target_bboxes:
                continue

            try:
                result_img, result_bboxes, result_classes = bbox_copy_paste(
                    src_img, target_bboxes, target_classes,
                    dst_img, dst_bboxes, dst_classes
                )
            except Exception:
                continue

            # Save
            cp_name = f"cp_{CLASS_NAMES[cls_id]}_{cp_count:05d}"
            cv2.imwrite(str(OUT_TRAIN_IMG / f"{cp_name}.jpg"), result_img)
            write_labels(
                OUT_TRAIN_LBL / f"{cp_name}.txt",
                result_bboxes,
                result_classes
            )
            cp_count += 1
            generated += 1

        new_class_counts[cls_id] = new_class_counts.get(cls_id, 0) + generated
        print(f"    -> Generated {generated} copy-paste images")

    # Step 4: Global augmentation pass (all classes)
    print("\n[Step 4] Global augmentation pass (diversity boost)...")
    aug_global = get_augmentation_pipeline("light")
    global_count = 0
    global_target = 1000  # add ~1000 more diverse images

    all_orig = [(p, SRC_LBL / (p.stem + '.txt')) for p in sorted(SRC_IMG.glob("*"))
                if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
    random.shuffle(all_orig)

    for img_path, lbl_path in all_orig:
        if global_count >= global_target:
            break

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        bboxes, class_ids = read_labels(lbl_path)
        if not bboxes:
            continue

        try:
            result = aug_global(image=img, bboxes=bboxes, class_ids=class_ids)
        except Exception:
            continue

        if not result['bboxes']:
            continue

        gname = f"global_{global_count:05d}"
        cv2.imwrite(str(OUT_TRAIN_IMG / f"{gname}.jpg"), result['image'])
        write_labels(OUT_TRAIN_LBL / f"{gname}.txt", result['bboxes'], result['class_ids'])
        global_count += 1

    print(f"  Generated {global_count} global augmented images")

    # Summary
    print("\n" + "=" * 60)
    print("  AUGMENTATION COMPLETE")
    print("=" * 60)

    total_train = len(list(OUT_TRAIN_IMG.glob("*")))
    total_val = len(list(OUT_VAL_IMG.glob("*")))
    print(f"\n  Original:  {orig_count} train / {val_count} val")
    print(f"  Augmented: {total_train} train / {total_val} val")
    print(f"  Added:     {total_train - orig_count} images")
    print(f"    - Class oversampling: {aug_count}")
    print(f"    - Copy-paste: {cp_count}")
    print(f"    - Global diversity: {global_count}")

    # Class distribution after augmentation
    print(f"\n  Class distribution (train instances):")
    final_counts = defaultdict(int)
    for lbl_path in OUT_TRAIN_LBL.glob("*.txt"):
        _, class_ids = read_labels(lbl_path)
        for c in class_ids:
            final_counts[c] += 1

    for i, name in enumerate(CLASS_NAMES):
        before = CURRENT_COUNTS.get(i, 0)
        after = final_counts.get(i, 0)
        ratio = after / before if before > 0 else 0
        bar = "#" * min(50, after // 20)
        print(f"    {name:<16} {before:>4} -> {after:>5} ({ratio:.1f}x) {bar}")

    # Create data.yaml
    yaml_content = f"""path: {OUT}
train: images/train
val: images/val

nc: 10
names: {CLASS_NAMES}
"""
    with open(OUT / "data.yaml", "w") as f:
        f.write(yaml_content)
    print(f"\n  data.yaml: {OUT / 'data.yaml'}")

    # Save stats
    stats = {
        "original_train": orig_count,
        "augmented_train": total_train,
        "val": total_val,
        "class_oversampling": aug_count,
        "copy_paste": cp_count,
        "global_diversity": global_count,
        "class_distribution": {CLASS_NAMES[i]: final_counts.get(i, 0) for i in range(10)},
    }
    with open(OUT / "augmentation_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats: {OUT / 'augmentation_stats.json'}")


if __name__ == "__main__":
    main()
