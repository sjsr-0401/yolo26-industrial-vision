"""
산업용 비전 검사 데이터셋 다운로드 + YOLO format 변환
1. NEU-DET: 강철 표면 결함 (6클래스)
2. PCB Defect: PCB 결함 검출
3. Safety Helmet: 안전모 착용 감지
"""

import os
import shutil
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import random
import ssl

# SSL 인증서 무시 (일부 미러에서 필요)
ssl._create_default_https_context = ssl._create_unverified_context

BASE = Path(r"C:\dev\active\yolo26-industrial-vision\datasets")


def download_file(url, dest, desc=""):
    """파일 다운로드 with progress"""
    print(f"  Downloading {desc or url.split('/')[-1]}...")
    try:
        urllib.request.urlretrieve(url, dest)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"    -> {size_mb:.1f} MB")
        return True
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


# ============================================================
# 1. NEU-DET (Steel Surface Defect Detection)
# ============================================================
def setup_neu_det():
    """NEU Surface Defect Database - 6 classes, 1800 images (300 each)
    Classes: crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches
    """
    print("\n[1/3] NEU-DET (Steel Surface Defect)")
    out_dir = BASE / "neu-det"
    
    if (out_dir / "train" / "images").exists():
        n = len(list((out_dir / "train" / "images").glob("*.jpg")))
        print(f"  Already exists ({n} train images), skipping")
        return
    
    # Download from Ultralytics hub (YOLO format)
    # NEU-DET is available via Ultralytics datasets
    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/NEU-DET.zip"
    zip_path = BASE / "neu-det.zip"
    
    if not download_file(url, zip_path, "NEU-DET"):
        # Fallback: try alternative source
        url2 = "https://huggingface.co/datasets/keremberke/neu-det/resolve/main/data.zip"
        if not download_file(url2, zip_path, "NEU-DET (HuggingFace)"):
            print("  FAILED: Manual download required")
            return
    
    print("  Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(BASE / "neu-det-raw")
    os.remove(zip_path)
    
    # Organize into YOLO format if needed
    raw_dir = BASE / "neu-det-raw"
    _organize_to_yolo(raw_dir, out_dir, "neu-det")
    print(f"  Done: {out_dir}")


# ============================================================
# 2. PCB Defect Detection
# ============================================================
def setup_pcb_defect():
    """PCB Defect Dataset - missing hole, mouse bite, open circuit, short, spur, spurious copper"""
    print("\n[2/3] PCB Defect Detection")
    out_dir = BASE / "pcb-defect"
    
    if (out_dir / "train" / "images").exists():
        n = len(list((out_dir / "train" / "images").glob("*")))
        print(f"  Already exists ({n} train images), skipping")
        return
    
    # PCB Defect dataset from open source
    url = "https://huggingface.co/datasets/keremberke/pcb-defect-segmentation/resolve/main/data.zip"
    zip_path = BASE / "pcb-defect.zip"
    
    if not download_file(url, zip_path, "PCB Defect"):
        # Alternative: use Ultralytics PCB dataset
        print("  Trying alternative source...")
        url2 = "https://github.com/Ixiaohuihuihui/Tiny-Defect-Detection-for-PCB/archive/refs/heads/master.zip"
        if not download_file(url2, zip_path, "PCB (GitHub)"):
            print("  FAILED: Will create synthetic PCB data")
            _create_synthetic_pcb(out_dir)
            return
    
    print("  Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(BASE / "pcb-raw")
    os.remove(zip_path)
    
    raw_dir = BASE / "pcb-raw"
    _organize_to_yolo(raw_dir, out_dir, "pcb-defect")
    print(f"  Done: {out_dir}")


# ============================================================
# 3. Safety Helmet Detection
# ============================================================
def setup_safety_helmet():
    """Hard Hat Workers Dataset"""
    print("\n[3/3] Safety Helmet Detection")
    out_dir = BASE / "safety-helmet"
    
    if (out_dir / "train" / "images").exists():
        n = len(list((out_dir / "train" / "images").glob("*")))
        print(f"  Already exists ({n} train images), skipping")
        return
    
    # Safety helmet dataset
    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/helmet-detection.zip"
    zip_path = BASE / "safety-helmet.zip"
    
    if not download_file(url, zip_path, "Safety Helmet"):
        url2 = "https://huggingface.co/datasets/keremberke/hard-hat-detection/resolve/main/data.zip"
        if not download_file(url2, zip_path, "Safety Helmet (HF)"):
            print("  FAILED: Will use COCO person class as fallback")
            return
    
    print("  Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(BASE / "safety-raw")
    os.remove(zip_path)
    
    raw_dir = BASE / "safety-raw"
    _organize_to_yolo(raw_dir, out_dir, "safety-helmet")
    print(f"  Done: {out_dir}")


def _organize_to_yolo(raw_dir, out_dir, name):
    """Raw extracted data -> standard YOLO directory structure"""
    # Check if already in YOLO format (train/images, train/labels)
    for candidate in [raw_dir] + list(raw_dir.rglob("*")):
        if candidate.is_dir() and (candidate / "train" / "images").exists():
            # Already YOLO format, just move
            if candidate != out_dir:
                shutil.move(str(candidate), str(out_dir))
            return
    
    # Check for images/ and labels/ at top level
    for candidate in [raw_dir] + list(raw_dir.rglob("*")):
        if candidate.is_dir() and (candidate / "images").exists() and (candidate / "labels").exists():
            # Has images and labels but no train/val split
            _split_dataset(candidate, out_dir)
            return
    
    # Look for any image files and try to organize
    imgs = list(raw_dir.rglob("*.jpg")) + list(raw_dir.rglob("*.png")) + list(raw_dir.rglob("*.bmp"))
    if imgs:
        print(f"  Found {len(imgs)} images, organizing...")
        # Just copy as-is for now
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(raw_dir), str(out_dir), dirs_exist_ok=True)
    
    # Cleanup raw
    if raw_dir.exists() and raw_dir != out_dir:
        shutil.rmtree(raw_dir, ignore_errors=True)


def _split_dataset(src, dst, train_ratio=0.8):
    """Split a dataset into train/val"""
    imgs_dir = src / "images"
    lbls_dir = src / "labels"
    
    imgs = sorted(list(imgs_dir.glob("*.*")))
    random.shuffle(imgs)
    
    split_idx = int(len(imgs) * train_ratio)
    
    for split, img_list in [("train", imgs[:split_idx]), ("val", imgs[split_idx:])]:
        (dst / split / "images").mkdir(parents=True, exist_ok=True)
        (dst / split / "labels").mkdir(parents=True, exist_ok=True)
        
        for img in img_list:
            shutil.copy2(img, dst / split / "images" / img.name)
            lbl = lbls_dir / (img.stem + ".txt")
            if lbl.exists():
                shutil.copy2(lbl, dst / split / "labels" / lbl.name)
    
    print(f"  Split: {split_idx} train, {len(imgs) - split_idx} val")


def _create_synthetic_pcb(out_dir):
    """Create a minimal synthetic PCB dataset for testing"""
    print("  Creating synthetic PCB dataset for demo...")
    # This would create placeholder data; real training needs real data
    (out_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (out_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)
    print("  Placeholder created - need real PCB data")


if __name__ == "__main__":
    print("=" * 60)
    print("  Industrial Vision Dataset Downloader")
    print("=" * 60)
    
    BASE.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    
    setup_neu_det()
    setup_pcb_defect()
    setup_safety_helmet()
    
    print("\n" + "=" * 60)
    print("  Dataset Summary")
    print("=" * 60)
    for d in sorted(BASE.iterdir()):
        if d.is_dir():
            imgs = list(d.rglob("*.jpg")) + list(d.rglob("*.png")) + list(d.rglob("*.bmp"))
            print(f"  {d.name}: {len(imgs)} images")
