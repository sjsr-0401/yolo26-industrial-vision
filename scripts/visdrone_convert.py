"""
VisDrone annotation → YOLO format 변환
VisDrone format: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
Categories: 0-ignored, 1-pedestrian, 2-people, 3-bicycle, 4-car, 5-van, 6-truck, 7-tricycle, 8-awning-tricycle, 9-bus, 10-motor, 11-others
"""

import os
from pathlib import Path

VD_BASE = Path(r"C:\Users\admin\.openclaw\workspace\datasets\VisDrone")
VD_VAL = VD_BASE / "VisDrone2019-DET-val"

# YOLO output
OUT = VD_BASE
CLASSES = {
    1: 0,   # pedestrian
    2: 1,   # people  
    3: 2,   # bicycle
    4: 3,   # car
    5: 4,   # van
    6: 5,   # truck
    7: 6,   # tricycle
    8: 7,   # awning-tricycle
    9: 8,   # bus
    10: 9,  # motor
}

CLASS_NAMES = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']


def convert_visdrone_to_yolo():
    """VisDrone annotations → YOLO txt"""
    ann_dir = VD_VAL / "annotations"
    img_dir = VD_VAL / "images"
    
    # Create YOLO structure
    out_img = OUT / "images" / "val"
    out_lbl = OUT / "labels" / "val"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    
    converted = 0
    
    for ann_file in sorted(ann_dir.glob("*.txt")):
        stem = ann_file.stem
        
        # Find corresponding image
        img_path = None
        for ext in [".jpg", ".png", ".jpeg"]:
            p = img_dir / (stem + ext)
            if p.exists():
                img_path = p
                break
        
        if img_path is None:
            continue
        
        # Get image size
        from PIL import Image
        with Image.open(img_path) as im:
            w, h = im.size
        
        # Parse annotations
        labels = []
        with open(ann_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 8:
                    continue
                
                bbox_left = float(parts[0])
                bbox_top = float(parts[1])
                bbox_w = float(parts[2])
                bbox_h = float(parts[3])
                # score = float(parts[4])
                category = int(parts[5])
                # truncation = int(parts[6])
                # occlusion = int(parts[7])
                
                if category not in CLASSES:
                    continue
                
                cls_id = CLASSES[category]
                
                # Convert to YOLO format (center x, center y, w, h - normalized)
                cx = (bbox_left + bbox_w / 2) / w
                cy = (bbox_top + bbox_h / 2) / h
                nw = bbox_w / w
                nh = bbox_h / h
                
                # Clamp
                cx = max(0, min(1, cx))
                cy = max(0, min(1, cy))
                nw = max(0, min(1, nw))
                nh = max(0, min(1, nh))
                
                if nw > 0 and nh > 0:
                    labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        
        if labels:
            # Symlink or copy image
            dst_img = out_img / img_path.name
            if not dst_img.exists():
                import shutil
                shutil.copy2(img_path, dst_img)
            
            # Write labels
            (out_lbl / (stem + ".txt")).write_text("\n".join(labels))
            converted += 1
    
    print(f"Converted {converted} images")
    
    # Create data.yaml
    yaml = f"""path: {str(OUT).replace(chr(92), '/')}
train: images/val
val: images/val

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    (OUT / "data.yaml").write_text(yaml, encoding='utf-8')
    print(f"data.yaml created at {OUT / 'data.yaml'}")
    
    # Stats: small object ratio
    total_objs = 0
    small_objs = 0  # < 32x32 pixels
    
    for lbl_file in (out_lbl).glob("*.txt"):
        with open(lbl_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    total_objs += 1
                    bw, bh = float(parts[3]), float(parts[4])
                    # Approximate pixel size (assuming ~1920x1080)
                    pw = bw * 1920
                    ph = bh * 1080
                    if pw < 32 or ph < 32:
                        small_objs += 1
    
    print(f"Total objects: {total_objs}")
    print(f"Small objects (<32px): {small_objs} ({small_objs/total_objs*100:.1f}%)")


if __name__ == "__main__":
    print("Converting VisDrone to YOLO format...")
    convert_visdrone_to_yolo()
