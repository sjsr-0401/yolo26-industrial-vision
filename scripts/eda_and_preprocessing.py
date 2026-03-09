"""
EDA (Exploratory Data Analysis) + Preprocessing Analysis
- 데이터셋별 클래스 분포, bbox 크기 분포, 이미지 해상도 분석
- CLAHE 전처리 before/after 시각화
- 출력: figures/ 폴더에 시각화 저장
"""

import os, sys, glob, json, random
import numpy as np
import cv2
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

BASE = Path(r"C:\dev\active\yolo26-industrial-vision")
FIG_DIR = BASE / "figures"
FIG_DIR.mkdir(exist_ok=True)

DATASETS = {
    "NEU-DET": {
        "images": BASE / "datasets/neu-det/train/images",
        "labels": BASE / "datasets/neu-det/train/labels",
        "classes": ["crazing","inclusion","patches","pitted_surface","rolled-in_scale","scratches"],
        "split": "",
        "native_res": "200x200",
    },
    "DeepPCB": {
        "images": BASE / "datasets/deeppcb/images/train",
        "labels": BASE / "datasets/deeppcb/labels/train",
        "classes": ["open","short","mousebite","spur","spurious_copper","pin_hole"],
        "split": "",
        "native_res": "640x640",
    },
    "GC10-DET": {
        "images": BASE / "datasets/gc10-det/images/train",
        "labels": BASE / "datasets/gc10-det/labels/train",
        "classes": ["punching","weld_line","crescent_gap","water_spot","oil_spot",
                     "silk_spot","inclusion","rolled_pit","crease","waist_folding"],
        "split": "",
        "native_res": "~2048x1000",
    },
}

def load_labels(label_dir, split="train"):
    """Load all YOLO format labels"""
    label_path = Path(label_dir) / split if split else Path(label_dir)
    if not label_path.exists():
        label_path = Path(label_dir)
    
    all_labels = []
    for txt in sorted(label_path.glob("*.txt")):
        with open(txt) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    all_labels.append({"file": txt.stem, "cls": cls, "x": x, "y": y, "w": w, "h": h})
    return all_labels

def get_image_sizes(img_dir, split="train"):
    """Get image dimensions"""
    img_path = Path(img_dir) / split if split else Path(img_dir)
    if not img_path.exists():
        img_path = Path(img_dir)
    
    sizes = []
    for ext in ["*.jpg", "*.png", "*.bmp", "*.jpeg"]:
        for f in sorted(img_path.glob(ext))[:200]:  # sample 200
            img = cv2.imread(str(f))
            if img is not None:
                h, w = img.shape[:2]
                sizes.append((w, h))
    return sizes

def plot_class_distribution(all_data):
    """Plot class distribution for all datasets"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Class Distribution per Dataset", fontsize=16, fontweight='bold')
    
    for idx, (name, info) in enumerate(DATASETS.items()):
        labels = all_data[name]["labels"]
        cls_counts = Counter(l["cls"] for l in labels)
        classes = info["classes"]
        counts = [cls_counts.get(i, 0) for i in range(len(classes))]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
        bars = axes[idx].barh(range(len(classes)), counts, color=colors, edgecolor='black', linewidth=0.5)
        axes[idx].set_yticks(range(len(classes)))
        axes[idx].set_yticklabels(classes, fontsize=9)
        axes[idx].set_xlabel("Instance Count")
        axes[idx].set_title(f"{name}\n(total: {sum(counts)})", fontweight='bold')
        
        # Add count labels
        for bar, count in zip(bars, counts):
            axes[idx].text(bar.get_width() + max(counts)*0.02, bar.get_y() + bar.get_height()/2,
                          str(count), va='center', fontsize=9)
        
        # Imbalance ratio
        if min(counts) > 0:
            ratio = max(counts) / min(counts)
            axes[idx].text(0.95, 0.05, f"Imbalance: {ratio:.1f}x",
                          transform=axes[idx].transAxes, ha='right', fontsize=10,
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_class_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] 01_class_distribution.png")

def plot_bbox_size_distribution(all_data):
    """Plot bbox size (relative area) distribution"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Bounding Box Size Distribution (Relative Area)", fontsize=16, fontweight='bold')
    
    size_bins = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    size_labels_text = ["<0.1%", "0.1-0.5%", "0.5-1%", "1-5%", "5-10%", "10-50%", ">50%"]
    
    for idx, (name, info) in enumerate(DATASETS.items()):
        labels = all_data[name]["labels"]
        areas = [l["w"] * l["h"] for l in labels]
        
        # Histogram
        axes[idx].hist(areas, bins=50, color='steelblue', edgecolor='black', linewidth=0.5, alpha=0.8)
        axes[idx].set_xlabel("Relative Area (w*h)")
        axes[idx].set_ylabel("Count")
        axes[idx].set_title(f"{name}", fontweight='bold')
        axes[idx].axvline(x=0.001, color='red', linestyle='--', alpha=0.7, label='Small (<0.1%)')
        
        # Stats
        small_pct = sum(1 for a in areas if a < 0.001) / len(areas) * 100 if areas else 0
        med = np.median(areas) if areas else 0
        axes[idx].text(0.95, 0.95, f"Small obj: {small_pct:.1f}%\nMedian: {med:.4f}",
                      transform=axes[idx].transAxes, ha='right', va='top', fontsize=10,
                      bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        axes[idx].legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_bbox_size_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] 02_bbox_size_distribution.png")

def plot_bbox_wh_scatter(all_data):
    """Plot bbox width vs height scatter"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Bounding Box Width vs Height (Normalized)", fontsize=16, fontweight='bold')
    
    for idx, (name, info) in enumerate(DATASETS.items()):
        labels = all_data[name]["labels"]
        ws = [l["w"] for l in labels]
        hs = [l["h"] for l in labels]
        
        axes[idx].scatter(ws, hs, s=5, alpha=0.3, c='steelblue')
        axes[idx].set_xlabel("Width (normalized)")
        axes[idx].set_ylabel("Height (normalized)")
        axes[idx].set_title(f"{name} ({len(labels)} objects)", fontweight='bold')
        axes[idx].set_xlim(0, 1)
        axes[idx].set_ylim(0, 1)
        axes[idx].plot([0, 1], [0, 1], 'r--', alpha=0.3, label='aspect=1:1')
        axes[idx].legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_bbox_wh_scatter.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] 03_bbox_wh_scatter.png")

def apply_clahe_comparison(all_data):
    """Apply CLAHE and show before/after comparison"""
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle("CLAHE Preprocessing: Before vs After\n(Zuiderveld, 1994 - Contrast Limited Adaptive Histogram Equalization)",
                 fontsize=14, fontweight='bold')
    
    for idx, (name, info) in enumerate(DATASETS.items()):
        img_dir = Path(info["images"])
        if info["split"]:
            img_dir = img_dir / info["split"]
        
        # Get a sample image
        imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.bmp"))
        if not imgs:
            continue
        
        random.seed(42)
        sample = random.choice(imgs)
        img = cv2.imread(str(sample))
        if img is None:
            continue
        
        # Original
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # CLAHE on L channel (LAB color space)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Histograms
        gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_clahe = cv2.cvtColor(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2GRAY)
        
        axes[idx, 0].imshow(img_rgb)
        axes[idx, 0].set_title(f"{name} - Original", fontsize=11)
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(img_clahe)
        axes[idx, 1].set_title(f"{name} - CLAHE Applied", fontsize=11)
        axes[idx, 1].axis('off')
        
        axes[idx, 2].hist(gray_orig.ravel(), 256, [0, 256], color='gray', alpha=0.7)
        axes[idx, 2].set_title("Histogram (Original)", fontsize=10)
        axes[idx, 2].set_xlim(0, 256)
        
        axes[idx, 3].hist(gray_clahe.ravel(), 256, [0, 256], color='steelblue', alpha=0.7)
        axes[idx, 3].set_title("Histogram (CLAHE)", fontsize=10)
        axes[idx, 3].set_xlim(0, 256)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_clahe_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] 04_clahe_comparison.png")

def plot_image_resolution(all_data):
    """Plot native image resolutions"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Image Resolution Distribution", fontsize=16, fontweight='bold')
    
    for idx, (name, info) in enumerate(DATASETS.items()):
        sizes = all_data[name]["sizes"]
        if not sizes:
            continue
        ws, hs = zip(*sizes)
        
        axes[idx].scatter(ws, hs, s=20, alpha=0.5, c='steelblue')
        axes[idx].set_xlabel("Width (px)")
        axes[idx].set_ylabel("Height (px)")
        axes[idx].set_title(f"{name}\nNative: {info['native_res']}", fontweight='bold')
        
        mean_w, mean_h = np.mean(ws), np.mean(hs)
        axes[idx].axvline(x=mean_w, color='red', linestyle='--', alpha=0.5)
        axes[idx].axhline(y=mean_h, color='red', linestyle='--', alpha=0.5)
        axes[idx].text(0.95, 0.95, f"Mean: {mean_w:.0f}x{mean_h:.0f}",
                      transform=axes[idx].transAxes, ha='right', va='top', fontsize=10,
                      bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_image_resolution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] 05_image_resolution.png")

def plot_augmentation_effects():
    """Visualize augmentation techniques: Mosaic, Copy-Paste, CutMix"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Data Augmentation Techniques (Applied in Training)\nMosaic (YOLOv4, 2020) | Copy-Paste (CVPR 2021) | CutMix (ICCV 2019)",
                 fontsize=14, fontweight='bold')
    
    # Load sample images from NEU-DET
    neu_dir = BASE / "datasets/neu-det/train/images"
    imgs = sorted(neu_dir.glob("*.jpg")) + sorted(neu_dir.glob("*.bmp"))
    
    if len(imgs) < 4:
        print("  [SKIP] Not enough images for augmentation demo")
        return
    
    random.seed(42)
    samples = random.sample(imgs, 4)
    loaded = []
    for s in samples:
        im = cv2.imread(str(s))
        if im is not None:
            loaded.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    
    if len(loaded) < 4:
        return
    
    # Resize all to same size
    target_size = (200, 200)
    loaded = [cv2.resize(img, target_size) for img in loaded]
    
    # Row 1: Original images
    for i in range(3):
        axes[0, i].imshow(loaded[i])
        axes[0, i].set_title(f"Original Sample {i+1}", fontsize=11)
        axes[0, i].axis('off')
    
    # Row 2: Augmentation demos
    
    # Mosaic (4 images combined)
    h, w = target_size
    mosaic = np.zeros((h*2, w*2, 3), dtype=np.uint8)
    mosaic[:h, :w] = loaded[0]
    mosaic[:h, w:w*2] = loaded[1]
    mosaic[h:h*2, :w] = loaded[2]
    mosaic[h:h*2, w:w*2] = loaded[3]
    mosaic = cv2.resize(mosaic, target_size)
    axes[1, 0].imshow(mosaic)
    axes[1, 0].set_title("Mosaic\n(Bochkovskiy, YOLOv4, 2020)", fontsize=11)
    axes[1, 0].axis('off')
    
    # CutMix simulation
    cutmix = loaded[0].copy()
    cx, cy = 100, 100
    cw, ch = 80, 80
    cutmix[cy:cy+ch, cx:cx+cw] = loaded[1][cy:cy+ch, cx:cx+cw]
    cv2.rectangle(cutmix, (cx, cy), (cx+cw, cy+ch), (255, 0, 0), 2)
    axes[1, 1].imshow(cutmix)
    axes[1, 1].set_title("CutMix\n(Yun et al., ICCV 2019)", fontsize=11)
    axes[1, 1].axis('off')
    
    # Copy-Paste simulation
    cp = loaded[2].copy()
    patch = loaded[3][50:120, 50:150]
    patch_resized = cv2.resize(patch, (80, 56))
    cp[20:76, 110:190] = patch_resized
    cv2.rectangle(cp, (110, 20), (190, 76), (0, 255, 0), 2)
    axes[1, 2].imshow(cp)
    axes[1, 2].set_title("Copy-Paste\n(Ghiasi et al., CVPR 2021)", fontsize=11)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_augmentation_techniques.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] 06_augmentation_techniques.png")

def generate_eda_summary(all_data):
    """Generate EDA summary JSON for PDF generation"""
    summary = {}
    for name, info in DATASETS.items():
        labels = all_data[name]["labels"]
        sizes = all_data[name]["sizes"]
        cls_counts = Counter(l["cls"] for l in labels)
        areas = [l["w"] * l["h"] for l in labels]
        
        classes = info["classes"]
        counts = [cls_counts.get(i, 0) for i in range(len(classes))]
        
        summary[name] = {
            "total_images": len(set(l["file"] for l in labels)),
            "total_instances": len(labels),
            "classes": len(classes),
            "class_names": classes,
            "class_counts": counts,
            "imbalance_ratio": round(max(counts) / max(min(counts), 1), 1),
            "native_res": info["native_res"],
            "mean_resolution": f"{np.mean([s[0] for s in sizes]):.0f}x{np.mean([s[1] for s in sizes]):.0f}" if sizes else "N/A",
            "small_object_pct": round(sum(1 for a in areas if a < 0.001) / max(len(areas), 1) * 100, 1),
            "median_area": round(np.median(areas), 5) if areas else 0,
            "mean_area": round(np.mean(areas), 5) if areas else 0,
        }
    
    with open(FIG_DIR / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("  [OK] eda_summary.json")
    return summary

def main():
    print("=" * 60)
    print("EDA + Preprocessing Analysis")
    print("=" * 60)
    
    all_data = {}
    
    for name, info in DATASETS.items():
        print(f"\n[{name}] Loading...")
        labels = load_labels(info["labels"], info["split"])
        sizes = get_image_sizes(info["images"], info["split"])
        all_data[name] = {"labels": labels, "sizes": sizes}
        print(f"  Labels: {len(labels)}, Images sampled: {len(sizes)}")
    
    print("\n--- Generating Visualizations ---")
    plot_class_distribution(all_data)
    plot_bbox_size_distribution(all_data)
    plot_bbox_wh_scatter(all_data)
    plot_image_resolution(all_data)
    apply_clahe_comparison(all_data)
    plot_augmentation_effects()
    summary = generate_eda_summary(all_data)
    
    print("\n--- EDA Summary ---")
    for name, s in summary.items():
        print(f"\n{name}:")
        print(f"  Images: {s['total_images']}, Instances: {s['total_instances']}")
        print(f"  Classes: {s['classes']}, Imbalance: {s['imbalance_ratio']}x")
        print(f"  Small objects: {s['small_object_pct']}%")
        print(f"  Native resolution: {s['native_res']}")
    
    print(f"\nAll figures saved to: {FIG_DIR}")
    print("Done!")

if __name__ == "__main__":
    main()
