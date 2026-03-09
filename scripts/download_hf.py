"""
HuggingFace에서 산업용 비전 데이터셋 다운로드
keremberke의 YOLO-format 데이터셋 사용 (검증된 소스)
"""

import os
import json
import shutil
import zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files

BASE = Path(r"C:\dev\active\yolo26-industrial-vision\datasets")


def download_hf_dataset(repo_id, dest_name, class_names):
    """HuggingFace datasets repo에서 YOLO format 데이터 다운로드"""
    dest = BASE / dest_name
    
    if (dest / "train" / "images").exists():
        n = len(list((dest / "train" / "images").glob("*")))
        if n > 50:
            print(f"  Already done ({n} train images)")
            return True
    
    print(f"  Repo: {repo_id}")
    
    try:
        # List files in repo
        files = list_repo_files(repo_id, repo_type="dataset")
        print(f"  Files in repo: {len(files)}")
        
        # Look for zip/tar files or direct image files
        zip_files = [f for f in files if f.endswith('.zip')]
        data_files = [f for f in files if f.endswith('.zip') or 'data' in f.lower()]
        
        if zip_files:
            for zf in zip_files:
                print(f"  Downloading: {zf}")
                local = hf_hub_download(repo_id, zf, repo_type="dataset")
                print(f"  Extracting...")
                with zipfile.ZipFile(local, 'r') as z:
                    z.extractall(str(dest))
                print(f"  -> {dest}")
        else:
            # Download all files
            print(f"  Downloading all {len(files)} files...")
            for f in files[:500]:  # limit
                try:
                    hf_hub_download(repo_id, f, repo_type="dataset", 
                                   local_dir=str(dest))
                except:
                    pass
        
        # Verify structure
        _verify_and_fix_structure(dest, class_names)
        return True
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def _verify_and_fix_structure(dest, class_names):
    """YOLO 디렉토리 구조 확인 및 수정"""
    # Check for train/images
    if (dest / "train" / "images").exists():
        n_train = len(list((dest / "train" / "images").glob("*")))
        n_val = len(list((dest / "val" / "images").glob("*"))) if (dest / "val" / "images").exists() else 0
        print(f"  Structure OK: {n_train} train / {n_val} val")
    else:
        # Look for images in subdirectories
        for sub in dest.rglob("*"):
            if sub.is_dir() and sub.name == "images" and (sub.parent / "labels").exists():
                parent_name = sub.parent.name
                if parent_name in ["train", "valid", "val", "test"]:
                    continue
                # Found images+labels pair
                print(f"  Found data at: {sub.parent}")
    
    # Create data.yaml if missing
    if not (dest / "data.yaml").exists():
        yaml = f"""path: {str(dest).replace(chr(92), '/')}
train: train/images
val: val/images

nc: {len(class_names)}
names: {class_names}
"""
        # Check for 'valid' instead of 'val'
        if (dest / "valid").exists() and not (dest / "val").exists():
            yaml = yaml.replace("val/images", "valid/images")
        
        (dest / "data.yaml").write_text(yaml, encoding='utf-8')
        print(f"  Created data.yaml")


def main():
    print("=" * 60)
    print("  HuggingFace Dataset Download")
    print("=" * 60)
    
    BASE.mkdir(parents=True, exist_ok=True)
    
    # Clean up synthetic data
    for d in ["neu-det", "pcb-defect", "safety-helmet"]:
        p = BASE / d
        if p.exists():
            # Check if it's synthetic (small files)
            imgs = list(p.rglob("*.jpg"))
            if imgs and all(i.stat().st_size < 20000 for i in imgs[:5]):
                print(f"  Removing synthetic {d}...")
                shutil.rmtree(p)
    
    # 1. NEU Surface Defect (Steel)
    print("\n[1/3] NEU-DET (Steel Surface Defect)")
    datasets_to_try = [
        "keremberke/neu-surface-defect-detection",
        "Francesco/neu-surface-defect-detection",
    ]
    
    for repo in datasets_to_try:
        ok = download_hf_dataset(
            repo, "neu-det",
            ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
        )
        if ok:
            break
    
    # 2. PCB Defect
    print("\n[2/3] PCB Defect Detection")
    pcb_repos = [
        "keremberke/pcb-defect-segmentation",
        "keremberke/pcb-defect-detection",
    ]
    
    for repo in pcb_repos:
        ok = download_hf_dataset(
            repo, "pcb-defect",
            ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']
        )
        if ok:
            break
    
    # 3. Hard Hat / Safety
    print("\n[3/3] Safety Helmet Detection")
    safety_repos = [
        "keremberke/hard-hat-detection",
        "keremberke/construction-safety-detection",
    ]
    
    for repo in safety_repos:
        ok = download_hf_dataset(
            repo, "safety-helmet",
            ['helmet', 'head', 'person']
        )
        if ok:
            break
    
    # Summary
    print("\n" + "=" * 60)
    print("  Final Dataset Summary")
    print("=" * 60)
    
    for d in sorted(BASE.iterdir()):
        if d.is_dir():
            for split in ["train", "valid", "val"]:
                imgs_dir = d / split / "images"
                if imgs_dir.exists():
                    imgs = list(imgs_dir.glob("*"))
                    if imgs:
                        avg_size = sum(i.stat().st_size for i in imgs[:10]) / min(len(imgs), 10) / 1024
                        print(f"  {d.name}/{split}: {len(imgs)} images (avg {avg_size:.0f}KB)")
            yaml = d / "data.yaml"
            if yaml.exists():
                print(f"    data.yaml: YES")


if __name__ == "__main__":
    main()
