"""
데이터셋 다운로드 — gdown (Google Drive) + 직접 URL
확실히 작동하는 소스만 사용
"""

import os
import shutil
import random
import glob
from pathlib import Path

BASE = Path(r"C:\dev\active\yolo26-industrial-vision\datasets")
random.seed(42)


def download_neu_det():
    """
    NEU Surface Defect Database
    - 6 classes: crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches
    - 1800 images total (300 per class), 200x200
    - Direct from Google Drive (from Surface-Defect-Detection repo)
    """
    print("\n[1/3] NEU-DET (Steel Surface Defect)")
    dest = BASE / "neu-det"
    
    if (dest / "train" / "images").exists():
        n = len(list((dest / "train" / "images").glob("*")))
        if n > 100:
            print(f"  Already done ({n} train images)")
            return True
    
    import gdown
    
    # NEU-DET YOLO format from Google Drive
    # This is a well-known public dataset
    gdrive_url = "https://drive.google.com/uc?id=1qrdZlaDi272eA79bSneUwnO6pVbsRl83"
    zip_path = BASE / "NEU-DET.zip"
    
    print("  Downloading from Google Drive...")
    try:
        gdown.download(gdrive_url, str(zip_path), quiet=False)
    except Exception as e:
        print(f"  GDrive failed: {e}")
        # Fallback: download original NEU-CLS and convert
        return download_neu_det_original()
    
    if zip_path.exists() and zip_path.stat().st_size > 1000:
        import zipfile
        print("  Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(BASE / "neu-raw")
        zip_path.unlink()
        
        # Organize
        raw = BASE / "neu-raw"
        _organize_neu(raw, dest)
        return True
    
    return download_neu_det_original()


def download_neu_det_original():
    """Fallback: Download original NEU-CLS from direct link and convert to YOLO"""
    print("  Trying original NEU source...")
    
    import gdown
    
    # Original NEU-CLS dataset
    url = "https://drive.google.com/uc?id=1L1kpT4MVKSz-n7W8EV08VOXNcLmJMqIG"
    zip_path = BASE / "NEU-CLS.zip"
    
    try:
        gdown.download(url, str(zip_path), quiet=False, fuzzy=True)
    except:
        pass
    
    if not zip_path.exists() or zip_path.stat().st_size < 1000:
        print("  All downloads failed. Will generate from scratch.")
        return generate_neu_det_synthetic()
    
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(BASE / "neu-cls-raw")
    zip_path.unlink()
    
    # Convert CLS to detection format
    _convert_neu_cls_to_yolo(BASE / "neu-cls-raw", BASE / "neu-det")
    return True


def _organize_neu(raw_dir, dest):
    """Organize NEU-DET into YOLO train/val structure"""
    # Find where images and labels are
    all_imgs = list(raw_dir.rglob("*.jpg")) + list(raw_dir.rglob("*.png")) + list(raw_dir.rglob("*.bmp"))
    all_lbls = list(raw_dir.rglob("*.txt"))
    all_xmls = list(raw_dir.rglob("*.xml"))
    
    print(f"  Found: {len(all_imgs)} images, {len(all_lbls)} txt, {len(all_xmls)} xml")
    
    # Check if already in YOLO format
    for d in [raw_dir] + list(raw_dir.iterdir()):
        if d.is_dir() and (d / "train" / "images").exists():
            shutil.copytree(str(d), str(dest), dirs_exist_ok=True)
            shutil.rmtree(raw_dir, ignore_errors=True)
            return
    
    # If we have images + txt labels, split into train/val
    if all_imgs and all_lbls:
        _split_to_yolo(all_imgs, all_lbls, dest)
    elif all_imgs and all_xmls:
        # Convert XML (Pascal VOC) to YOLO format
        _convert_voc_to_yolo(all_imgs, all_xmls, dest, 
                            ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"])
    else:
        print("  Cannot organize - unknown format")
    
    shutil.rmtree(raw_dir, ignore_errors=True)


def _split_to_yolo(imgs, lbls, dest, train_ratio=0.8):
    """Split images+labels into train/val YOLO structure"""
    lbl_map = {Path(l).stem: l for l in lbls}
    
    pairs = [(img, lbl_map[img.stem]) for img in imgs if img.stem in lbl_map]
    random.shuffle(pairs)
    
    split = int(len(pairs) * train_ratio)
    
    for split_name, subset in [("train", pairs[:split]), ("val", pairs[split:])]:
        (dest / split_name / "images").mkdir(parents=True, exist_ok=True)
        (dest / split_name / "labels").mkdir(parents=True, exist_ok=True)
        
        for img, lbl in subset:
            shutil.copy2(img, dest / split_name / "images" / img.name)
            shutil.copy2(lbl, dest / split_name / "labels" / Path(lbl).name)
    
    # Create data.yaml
    yaml = f"""path: {str(dest).replace(chr(92), '/')}
train: train/images
val: val/images

nc: 6
names: ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
"""
    (dest / "data.yaml").write_text(yaml, encoding='utf-8')
    print(f"  Split: {split} train, {len(pairs)-split} val")


def _convert_voc_to_yolo(imgs, xmls, dest, class_names, train_ratio=0.8):
    """Convert Pascal VOC XML to YOLO format"""
    import xml.etree.ElementTree as ET
    
    xml_map = {Path(x).stem: x for x in xmls}
    pairs = []
    
    for img in imgs:
        if img.stem not in xml_map:
            continue
        
        tree = ET.parse(xml_map[img.stem])
        root = tree.getroot()
        
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        
        labels = []
        for obj in root.findall('object'):
            cls_name = obj.find('name').text
            if cls_name not in class_names:
                continue
            cls_id = class_names.index(cls_name)
            
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # Convert to YOLO format (center x, center y, width, height - normalized)
            cx = (xmin + xmax) / 2 / w
            cy = (ymin + ymax) / 2 / h
            bw = (xmax - xmin) / w
            bh = (ymax - ymin) / h
            
            labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        
        if labels:
            pairs.append((img, "\n".join(labels)))
    
    random.shuffle(pairs)
    split = int(len(pairs) * train_ratio)
    
    for split_name, subset in [("train", pairs[:split]), ("val", pairs[split:])]:
        (dest / split_name / "images").mkdir(parents=True, exist_ok=True)
        (dest / split_name / "labels").mkdir(parents=True, exist_ok=True)
        
        for img_path, lbl_text in subset:
            shutil.copy2(img_path, dest / split_name / "images" / img_path.name)
            lbl_path = dest / split_name / "labels" / (img_path.stem + ".txt")
            lbl_path.write_text(lbl_text, encoding='utf-8')
    
    yaml = f"""path: {str(dest).replace(chr(92), '/')}
train: train/images
val: val/images

nc: {len(class_names)}
names: {class_names}
"""
    (dest / "data.yaml").write_text(yaml, encoding='utf-8')
    print(f"  Converted VOC->YOLO: {split} train, {len(pairs)-split} val")


def _convert_neu_cls_to_yolo(raw_dir, dest):
    """Convert NEU-CLS (classification images) to detection format
    Each image is 200x200 with defect covering most of the image
    We treat the entire image as a bounding box
    """
    classes = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
    
    all_pairs = []
    for cls_idx, cls_name in enumerate(classes):
        # Find images for this class
        for ext in ['*.bmp', '*.jpg', '*.png']:
            for img in raw_dir.rglob(ext):
                if cls_name.lower().replace('_', '') in img.stem.lower().replace('_', '') or \
                   cls_name.lower() in str(img.parent).lower():
                    # Full-image bounding box (centered, 90% coverage)
                    label = f"{cls_idx} 0.5 0.5 0.9 0.9"
                    all_pairs.append((img, label, cls_name))
    
    if not all_pairs:
        # Try different naming convention
        for img in raw_dir.rglob("*.*"):
            if img.suffix.lower() in ['.bmp', '.jpg', '.png']:
                for cls_idx, cls_name in enumerate(classes):
                    short = cls_name[:2].upper()
                    if img.stem.startswith(short) or img.stem.startswith(cls_name):
                        label = f"{cls_idx} 0.5 0.5 0.9 0.9"
                        all_pairs.append((img, label, cls_name))
                        break
    
    print(f"  Found {len(all_pairs)} classified images")
    
    if not all_pairs:
        print("  No images matched class patterns")
        return
    
    random.shuffle(all_pairs)
    split = int(len(all_pairs) * 0.8)
    
    for split_name, subset in [("train", all_pairs[:split]), ("val", all_pairs[split:])]:
        (dest / split_name / "images").mkdir(parents=True, exist_ok=True)
        (dest / split_name / "labels").mkdir(parents=True, exist_ok=True)
        
        for img, lbl, _ in subset:
            shutil.copy2(img, dest / split_name / "images" / img.name)
            (dest / split_name / "labels" / (img.stem + ".txt")).write_text(lbl)
    
    yaml = f"""path: {str(dest).replace(chr(92), '/')}
train: train/images
val: val/images

nc: 6
names: {classes}
"""
    (dest / "data.yaml").write_text(yaml, encoding='utf-8')
    print(f"  Split: {split} train, {len(all_pairs)-split} val")
    shutil.rmtree(raw_dir, ignore_errors=True)


def generate_neu_det_synthetic():
    """Generate synthetic steel defect images using OpenCV for demo purposes"""
    print("  Generating synthetic steel defect dataset...")
    import cv2
    import numpy as np
    
    dest = BASE / "neu-det"
    classes = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
    
    for split_name, count in [("train", 200), ("val", 50)]:
        (dest / split_name / "images").mkdir(parents=True, exist_ok=True)
        (dest / split_name / "labels").mkdir(parents=True, exist_ok=True)
        
        for i in range(count):
            cls_idx = i % 6
            cls_name = classes[cls_idx]
            
            # Create steel-like background
            img = np.random.randint(100, 160, (200, 200, 3), dtype=np.uint8)
            img = cv2.GaussianBlur(img, (5, 5), 2)
            
            # Add defect-like features based on class
            cx, cy = random.randint(40, 160), random.randint(40, 160)
            w, h = random.randint(30, 80), random.randint(30, 80)
            
            if cls_name == "crazing":
                for _ in range(5):
                    x1, y1 = cx + random.randint(-30, 30), cy + random.randint(-30, 30)
                    x2, y2 = x1 + random.randint(-20, 20), y1 + random.randint(-20, 20)
                    cv2.line(img, (x1, y1), (x2, y2), (60, 60, 60), 1)
            elif cls_name == "inclusion":
                cv2.ellipse(img, (cx, cy), (w//3, h//3), random.randint(0, 180), 0, 360, (40, 40, 40), -1)
            elif cls_name == "patches":
                cv2.rectangle(img, (cx-w//2, cy-h//2), (cx+w//2, cy+h//2), (180, 180, 180), -1)
            elif cls_name == "pitted_surface":
                for _ in range(8):
                    px, py = cx + random.randint(-30, 30), cy + random.randint(-30, 30)
                    r = random.randint(2, 6)
                    cv2.circle(img, (px, py), r, (50, 50, 50), -1)
            elif cls_name == "rolled-in_scale":
                for _ in range(3):
                    y = cy + random.randint(-20, 20)
                    cv2.line(img, (cx-40, y), (cx+40, y), (70, 70, 70), random.randint(1, 3))
            elif cls_name == "scratches":
                x1, y1 = cx - random.randint(20, 40), cy - random.randint(20, 40)
                x2, y2 = cx + random.randint(20, 40), cy + random.randint(20, 40)
                cv2.line(img, (x1, y1), (x2, y2), (50, 50, 50), random.randint(1, 2))
            
            fname = f"{cls_name}_{i:04d}"
            cv2.imwrite(str(dest / split_name / "images" / f"{fname}.jpg"), img)
            
            # YOLO label
            ncx, ncy = cx/200, cy/200
            nw, nh = w/200, h/200
            (dest / split_name / "labels" / f"{fname}.txt").write_text(
                f"{cls_idx} {ncx:.4f} {ncy:.4f} {nw:.4f} {nh:.4f}"
            )
    
    yaml = f"""path: {str(dest).replace(chr(92), '/')}
train: train/images
val: val/images

nc: 6
names: {classes}
"""
    (dest / "data.yaml").write_text(yaml, encoding='utf-8')
    print(f"  Generated: 200 train + 50 val synthetic images")
    return True


def download_hardhat():
    """Safety Helmet dataset from a known working source"""
    print("\n[3/3] Safety Helmet Detection")
    dest = BASE / "safety-helmet"
    
    if (dest / "train" / "images").exists():
        n = len(list((dest / "train" / "images").glob("*")))
        if n > 50:
            print(f"  Already done ({n} train images)")
            return True
    
    import gdown
    
    # Hard hat dataset (YOLO format) from Google Drive
    # Multiple sources to try
    urls = [
        "https://drive.google.com/uc?id=1vXGUBuGMs5XK1QYPfjXPqBNfbTtUfzYU",
        "https://drive.google.com/uc?id=1N1wPPKNm6iGYiQaEjOWdXrWHBWS_H_6K",
    ]
    
    zip_path = BASE / "hardhat.zip"
    
    for url in urls:
        try:
            gdown.download(url, str(zip_path), quiet=False, fuzzy=True)
            if zip_path.exists() and zip_path.stat().st_size > 10000:
                break
        except:
            continue
    
    if zip_path.exists() and zip_path.stat().st_size > 10000:
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(BASE / "hardhat-raw")
        zip_path.unlink()
        _organize_neu(BASE / "hardhat-raw", dest)
        return True
    
    print("  Download failed, generating synthetic safety dataset...")
    return generate_safety_synthetic()


def generate_safety_synthetic():
    """Generate synthetic safety helmet detection data"""
    print("  Generating synthetic safety helmet dataset...")
    import cv2
    import numpy as np
    
    dest = BASE / "safety-helmet"
    classes = ["helmet", "no_helmet", "person"]
    
    for split_name, count in [("train", 200), ("val", 50)]:
        (dest / split_name / "images").mkdir(parents=True, exist_ok=True)
        (dest / split_name / "labels").mkdir(parents=True, exist_ok=True)
        
        for i in range(count):
            # Construction site-like background
            img = np.random.randint(140, 200, (416, 416, 3), dtype=np.uint8)
            img = cv2.GaussianBlur(img, (7, 7), 3)
            
            labels = []
            num_people = random.randint(1, 3)
            
            for p in range(num_people):
                px = random.randint(60, 356)
                py = random.randint(100, 360)
                pw, ph = random.randint(40, 80), random.randint(80, 150)
                
                # Draw person
                cv2.rectangle(img, (px-pw//2, py-ph//2), (px+pw//2, py+ph//2), 
                            (random.randint(40, 100), random.randint(40, 100), random.randint(100, 200)), -1)
                
                # Person label
                labels.append(f"2 {px/416:.4f} {py/416:.4f} {pw/416:.4f} {ph/416:.4f}")
                
                # Head area
                hx, hy = px, py - ph//2 + 15
                hw, hh = 25, 25
                
                has_helmet = random.random() > 0.4
                if has_helmet:
                    cv2.ellipse(img, (hx, hy), (hw//2, hh//2), 0, 0, 360, (0, 200, 255), -1)
                    labels.append(f"0 {hx/416:.4f} {hy/416:.4f} {hw/416:.4f} {hh/416:.4f}")
                else:
                    cv2.circle(img, (hx, hy), 10, (200, 180, 160), -1)
                    labels.append(f"1 {hx/416:.4f} {hy/416:.4f} {hw/416:.4f} {hh/416:.4f}")
            
            fname = f"safety_{i:04d}"
            cv2.imwrite(str(dest / split_name / "images" / f"{fname}.jpg"), img)
            (dest / split_name / "labels" / f"{fname}.txt").write_text("\n".join(labels))
    
    yaml = f"""path: {str(dest).replace(chr(92), '/')}
train: train/images
val: val/images

nc: 3
names: {classes}
"""
    (dest / "data.yaml").write_text(yaml, encoding='utf-8')
    print(f"  Generated: 200 train + 50 val synthetic images")
    return True


def download_pcb():
    """PCB Defect Dataset"""
    print("\n[2/3] PCB Defect Detection")
    dest = BASE / "pcb-defect"
    
    if (dest / "train" / "images").exists():
        n = len(list((dest / "train" / "images").glob("*")))
        if n > 50:
            print(f"  Already done ({n} train images)")
            return True
    
    import gdown
    
    # PCB defect datasets on Google Drive
    urls = [
        "https://drive.google.com/uc?id=17-EvYI-dR2W62A6JI8EVbsPr47c-vQSi",
    ]
    
    zip_path = BASE / "pcb.zip"
    for url in urls:
        try:
            gdown.download(url, str(zip_path), quiet=False, fuzzy=True)
            if zip_path.exists() and zip_path.stat().st_size > 10000:
                break
        except:
            continue
    
    if zip_path.exists() and zip_path.stat().st_size > 10000:
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(BASE / "pcb-raw")
        zip_path.unlink()
        # Check format and organize
        _organize_pcb(BASE / "pcb-raw", dest)
        return True
    
    print("  Download failed, generating synthetic PCB data...")
    return generate_pcb_synthetic()


def generate_pcb_synthetic():
    """Generate synthetic PCB defect images"""
    print("  Generating synthetic PCB defect dataset...")
    import cv2
    import numpy as np
    
    dest = BASE / "pcb-defect"
    classes = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]
    
    for split_name, count in [("train", 200), ("val", 50)]:
        (dest / split_name / "images").mkdir(parents=True, exist_ok=True)
        (dest / split_name / "labels").mkdir(parents=True, exist_ok=True)
        
        for i in range(count):
            cls_idx = i % 6
            
            # PCB-like green background
            img = np.zeros((640, 640, 3), dtype=np.uint8)
            img[:] = (40, 100, 40)  # PCB green
            
            # Add traces (copper lines)
            for _ in range(random.randint(5, 15)):
                x1, y1 = random.randint(0, 640), random.randint(0, 640)
                x2, y2 = x1 + random.choice([-1,0,1])*random.randint(50,200), y1 + random.choice([-1,0,1])*random.randint(50,200)
                cv2.line(img, (x1, y1), (x2, y2), (60, 180, 200), random.randint(2, 5))
            
            # Add pads
            for _ in range(random.randint(3, 8)):
                px, py = random.randint(50, 590), random.randint(50, 590)
                cv2.circle(img, (px, py), random.randint(8, 15), (80, 200, 220), -1)
                cv2.circle(img, (px, py), random.randint(3, 5), (40, 100, 40), -1)
            
            # Add defect
            cx, cy = random.randint(100, 540), random.randint(100, 540)
            w, h = random.randint(30, 80), random.randint(30, 80)
            
            if cls_idx == 0:  # missing_hole
                cv2.circle(img, (cx, cy), 12, (80, 200, 220), -1)  # pad without hole
            elif cls_idx == 1:  # mouse_bite
                cv2.circle(img, (cx, cy), 15, (80, 200, 220), -1)
                cv2.circle(img, (cx+8, cy+8), 8, (40, 100, 40), -1)
            elif cls_idx == 2:  # open_circuit
                cv2.line(img, (cx-30, cy), (cx-5, cy), (60, 180, 200), 3)
                cv2.line(img, (cx+5, cy), (cx+30, cy), (60, 180, 200), 3)
            elif cls_idx == 3:  # short
                cv2.line(img, (cx, cy-20), (cx, cy+20), (60, 180, 200), 3)
                cv2.line(img, (cx-20, cy), (cx+20, cy), (60, 180, 200), 3)
                cv2.rectangle(img, (cx-10, cy-10), (cx+10, cy+10), (60, 180, 200), -1)
            elif cls_idx == 4:  # spur
                cv2.line(img, (cx-20, cy), (cx+20, cy), (60, 180, 200), 3)
                cv2.line(img, (cx, cy), (cx+15, cy-15), (60, 180, 200), 2)
            elif cls_idx == 5:  # spurious_copper
                pts = np.array([(cx+random.randint(-15,15), cy+random.randint(-15,15)) for _ in range(5)], np.int32)
                cv2.fillPoly(img, [pts], (60, 180, 200))
            
            fname = f"pcb_{cls_idx}_{i:04d}"
            cv2.imwrite(str(dest / split_name / "images" / f"{fname}.jpg"), img)
            (dest / split_name / "labels" / f"{fname}.txt").write_text(
                f"{cls_idx} {cx/640:.4f} {cy/640:.4f} {w/640:.4f} {h/640:.4f}"
            )
    
    yaml = f"""path: {str(dest).replace(chr(92), '/')}
train: train/images
val: val/images

nc: 6
names: {classes}
"""
    (dest / "data.yaml").write_text(yaml, encoding='utf-8')
    print(f"  Generated: 200 train + 50 val synthetic images")
    return True


def _organize_pcb(raw_dir, dest):
    """Organize PCB data"""
    _organize_neu(raw_dir, dest)  # Same logic


if __name__ == "__main__":
    print("=" * 60)
    print("  Dataset Download (Final)")
    print("=" * 60)
    
    BASE.mkdir(parents=True, exist_ok=True)
    
    download_neu_det()
    download_pcb()
    download_hardhat()
    
    print("\n" + "=" * 60)
    print("  Dataset Summary")
    print("=" * 60)
    
    for d in sorted(BASE.iterdir()):
        if d.is_dir() and not d.name.endswith("-raw"):
            train_imgs = list((d / "train" / "images").glob("*")) if (d / "train" / "images").exists() else []
            val_imgs = list((d / "val" / "images").glob("*")) if (d / "val" / "images").exists() else []
            yaml_exists = (d / "data.yaml").exists()
            print(f"  {d.name}: {len(train_imgs)} train / {len(val_imgs)} val | yaml: {yaml_exists}")
