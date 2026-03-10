"""
YOLO26 Medium - GC10-DET Steel Defect Detection (Kaggle T4 GPU)
Upload dataset + run training + download results
"""

import os
import json
import shutil
import subprocess
import sys
import time

def run(cmd):
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0 and result.stderr:
        print(result.stderr[-1000:])
    return result.returncode

# === Config ===
KAGGLE_USER = "sjsr0401"
DATASET_SLUG = f"{KAGGLE_USER}/gc10-det-yolo"
NOTEBOOK_SLUG = f"{KAGGLE_USER}/yolo26m-gc10det-v3"

# === Step 1: Create/update dataset (if needed) ===
DATASET_DIR = r"C:\dev\active\yolo26-industrial-vision\datasets\gc10-det"

def upload_dataset():
    """Upload GC10-DET dataset to Kaggle"""
    staging = os.path.join(os.environ["TEMP"], "kaggle-gc10det")
    if os.path.exists(staging):
        shutil.rmtree(staging)
    os.makedirs(staging)
    
    # Create dataset metadata
    metadata = {
        "title": "GC10-DET YOLO Format",
        "id": DATASET_SLUG,
        "licenses": [{"name": "CC0-1.0"}]
    }
    with open(os.path.join(staging, "dataset-metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Copy dataset files
    for split in ["train", "val"]:
        for subdir in ["images", "labels"]:
            src = os.path.join(DATASET_DIR, split, subdir)
            dst = os.path.join(staging, split, subdir)
            if os.path.exists(src):
                shutil.copytree(src, dst)
    
    # Copy data.yaml
    yaml_src = os.path.join(DATASET_DIR, "data.yaml")
    if os.path.exists(yaml_src):
        shutil.copy2(yaml_src, staging)
    
    print(f"Dataset staged: {staging}")
    
    # Check if dataset exists
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    
    try:
        api.dataset_status(DATASET_SLUG)
        print("Dataset exists, updating...")
        run(f'python -c "from kaggle.api.kaggle_api_extended import KaggleApi; api = KaggleApi(); api.authenticate(); api.dataset_create_version(r\'{staging}\', \'update\')"')
    except Exception:
        print("Creating new dataset...")
        run(f'python -c "from kaggle.api.kaggle_api_extended import KaggleApi; api = KaggleApi(); api.authenticate(); api.dataset_create_new(r\'{staging}\', public=False)"')
    
    shutil.rmtree(staging, ignore_errors=True)

# === Step 2: Create Kaggle notebook ===
NOTEBOOK_CONTENT = '''#!/usr/bin/env python
# YOLO26 Medium - GC10-DET v3

!pip install -q ultralytics

import os
from pathlib import Path

# Setup dataset path
DATASET_PATH = "/kaggle/input/gc10-det-yolo"
WORK_DIR = "/kaggle/working"

# Create data.yaml pointing to Kaggle paths
data_yaml = f"""
path: {DATASET_PATH}
train: train/images
val: val/images

nc: 10
names: ['punching', 'welding_line', 'crescent_gap', 'water_spot', 'oil_spot',
        'silk_spot', 'inclusion', 'rolled_pit', 'crease', 'waist_folding']
"""

yaml_path = os.path.join(WORK_DIR, "gc10det.yaml")
with open(yaml_path, "w") as f:
    f.write(data_yaml)

# Train
from ultralytics import YOLO

model = YOLO("yolo26m.pt")
results = model.train(
    data=yaml_path,
    epochs=200,
    imgsz=1024,
    batch=2,
    patience=30,
    device=0,
    workers=2,
    project=WORK_DIR,
    name="yolo26m_gc10det_v3",
    exist_ok=True,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    cos_lr=True,
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.3,
    degrees=15.0,
    translate=0.15,
    scale=0.4,
    fliplr=0.5,
    mosaic=0.8,
    mixup=0.1,
    copy_paste=0.05,
)

# Export ONNX
best_path = os.path.join(WORK_DIR, "yolo26m_gc10det_v3", "weights", "best.pt")
best_model = YOLO(best_path)
best_model.export(format="onnx", imgsz=1024, simplify=True)

print("Training complete! Download results from Output tab.")
'''

def create_notebook():
    """Create and push Kaggle notebook"""
    nb_dir = os.path.join(os.environ["TEMP"], "kaggle-nb-medium")
    if os.path.exists(nb_dir):
        shutil.rmtree(nb_dir)
    os.makedirs(nb_dir)
    
    # Write script
    with open(os.path.join(nb_dir, "yolo26m-gc10det-v3.py"), "w") as f:
        f.write(NOTEBOOK_CONTENT)
    
    # Kernel metadata
    meta = {
        "id": NOTEBOOK_SLUG,
        "title": "yolo26m-gc10det-v3",
        "code_file": "yolo26m-gc10det-v3.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [DATASET_SLUG],
        "competition_sources": [],
        "kernel_sources": []
    }
    with open(os.path.join(nb_dir, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"Notebook staged: {nb_dir}")
    
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    api.kernels_push(nb_dir)
    print(f"Notebook pushed: https://www.kaggle.com/code/{NOTEBOOK_SLUG}")
    
    shutil.rmtree(nb_dir, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        action = sys.argv[1]
    else:
        action = "all"
    
    if action in ("dataset", "all"):
        print("=== Uploading dataset ===")
        upload_dataset()
    
    if action in ("notebook", "all"):
        print("\n=== Creating notebook ===")
        create_notebook()
    
    print("\nDone! Check https://www.kaggle.com/code/" + NOTEBOOK_SLUG)
