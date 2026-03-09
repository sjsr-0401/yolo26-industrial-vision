# YOLO26 Industrial Defect Detection

End-to-end pipeline for industrial surface defect detection using **YOLO26** (Ultralytics).  
From dataset preparation to ONNX deployment with a C# WPF inspection app.

> **Key Finding**: Data augmentation (+3.6% mAP) outperformed model scaling (+2.7% mAP) on small datasets, demonstrating that engineering judgment matters more than brute-force scaling.

---

## Results Summary

| Model | Dataset | Images | imgsz | mAP50 | mAP50-95 | ONNX | GPU (ms) | FPS |
|-------|---------|--------|-------|-------|----------|------|----------|-----|
| YOLO26n | NEU-DET | 1,800 | 640 | 0.726 | — | 5.2MB | — | — |
| YOLO26n | DeepPCB | 1,500 | 640 | **0.984** | — | 5.2MB | — | — |
| YOLO26n | GC10-DET (baseline) | 1,835 | 1024 | 0.635 | 0.298 | 9.6MB | — | — |
| YOLO26n | GC10-DET v3 | 1,835 | 1024 | 0.701 | 0.370 | 9.6MB | 19.0 | 53 |
| YOLO26s | GC10-DET v3 | 1,835 | 1024 | 0.720 | 0.364 | 36.7MB | 33.1 | 30 |
| **YOLO26n** | **GC10-DET v4 (aug)** | **4,329** | **1024** | **0.726** | **0.394** | **9.6MB** | **11.5** | **87** |

### Key Insights

- **Nano v4 (augmented) > Small v3**: Nano with augmented data (0.726) beat Small with original data (0.720), using 4× fewer parameters
- **Augmentation > Scaling**: +3.6% from data augmentation vs +2.7% from Nano→Small scaling
- **Class imbalance fix**: 12.4× → 2.4× imbalance ratio (5× improvement via Albumentations pipeline)

---

## Datasets

| Dataset | Domain | Classes | Train | Val | Resolution | Source |
|---------|--------|---------|-------|-----|------------|--------|
| [NEU-DET](http://faculty.neu.edu.cn/songkechen/en/) | Steel Surface | 6 | 1,200 | 600 | 200×200 | Northeastern University |
| [DeepPCB](https://github.com/tangsanli5201/DeepPCB) | PCB Defects | 6 | 1,000 | 500 | 640×640 | Peking University |
| [GC10-DET](https://github.com/lvxiaoming2019/GC10-DET) | Steel Strip | 10 | 1,835 | 459 | Mixed | Collected from real production |

---

## Project Structure

```
yolo26-industrial-vision/
├── datasets/                    # YOLO-format datasets
│   ├── gc10-det/               # Original (1,835 train / 459 val)
│   └── gc10-det-aug/           # Augmented (4,329 train / 459 val)
├── notebooks/
│   └── colab_gc10det_medium.ipynb  # Colab T4 training notebook
├── results/
│   ├── yolo26n_gc10det_v3/     # Nano v3 (mAP50=0.701)
│   ├── yolo26n_gc10det_v4_aug/ # Nano v4 augmented (mAP50=0.726)
│   ├── yolo26s_gc10det_v3/     # Small v3 (mAP50=0.720)
│   ├── yolo26n_neudet_v3/      # NEU-DET (mAP50=0.726)
│   ├── yolo26n_deeppcb/        # DeepPCB (mAP50=0.984)
│   └── benchmark_results.json  # ONNX inference benchmark
├── scripts/
│   ├── augment_gc10det.py      # Albumentations augmentation pipeline
│   ├── benchmark_inference.py  # ONNX Runtime GPU+CPU benchmark
│   ├── train_improved_v3.py    # v3 training with tuned hyperparams
│   ├── train_augmented.py      # v4 augmented data training
│   ├── eda_and_preprocessing.py # EDA + CLAHE analysis
│   └── generate_*.py           # PDF report generators
└── src/
    └── InspectView/            # C# WPF ONNX inspection app
```

---

## Improvement Strategy

### v3: Hyperparameter Tuning (baseline → +10.4%)
Based on dataset analysis and peer-reviewed research:

| Technique | Reference | Effect |
|-----------|-----------|--------|
| `imgsz=1024` | Match native resolution | Better small object detection |
| `copy_paste=0.2` | [Ghiasi et al., CVPR 2021](https://arxiv.org/abs/2012.07177) | Augment minority classes |
| `mixup=0.1` | [Zhang et al., ICLR 2018](https://arxiv.org/abs/1710.09412) | Regularization |
| `scale=0.7` | Ultralytics best practice | Scale variation robustness |
| `patience=30` | Early stopping | Prevent overfitting |

### v4: Data Augmentation (+3.6%)
Offline augmentation using [Albumentations 2.0](https://albumentations.ai/):

- **Class-aware oversampling**: Minority classes (crease 56→550, rolled_pit 76→718)
- **Augmentation pipeline**: HorizontalFlip + RandomBrightnessContrast + CLAHE + GaussNoise + Rotate + RandomResizedCrop
- **Class imbalance**: 12.4× → 2.4× (5× improvement)
- **Total images**: 1,835 → 4,329 (+136%)

---

## ONNX Inference Benchmark

**Device**: NVIDIA RTX 3060 12GB / Intel i5 / Windows 10  
**Runtime**: ONNX Runtime 1.24 (CUDA EP + CPU EP)

| Model | ONNX Size | GPU (ms) | GPU FPS | CPU (ms) | CPU FPS |
|-------|-----------|----------|---------|----------|---------|
| YOLO26n v3 | 9.6 MB | 19.0 | 53 | 102.5 | 10 |
| YOLO26n v4 | 9.6 MB | 11.5 | 87 | 99.2 | 10 |
| YOLO26s v3 | 36.7 MB | 33.1 | 30 | 296.3 | 3 |

---

## InspectView — C# WPF Inspection App

Desktop application for ONNX-based industrial defect inspection.

- **Framework**: .NET 8 / WPF / MVVM (CommunityToolkit.Mvvm)
- **Inference**: ONNX Runtime + OpenCvSharp4
- **Features**: Drag-and-drop images, Pass/Fail judgment, Detection Log, CSV export
- **Test Result**: 459 val images → Pass 54, Fail 405, Avg 112.9ms/image

---

## Quick Start

### Training
```bash
pip install ultralytics==8.4.21

# Train YOLO26n on GC10-DET
python scripts/train_improved_v3.py

# Data augmentation + retrain
python scripts/augment_gc10det.py
python scripts/train_augmented.py
```

### ONNX Export & Benchmark
```bash
# Export to ONNX
yolo export model=results/yolo26n_gc10det_v4_aug/weights/best.pt format=onnx imgsz=1024

# Benchmark
python scripts/benchmark_inference.py
```

### InspectView (C#)
```bash
cd src/InspectView/InspectView
dotnet build
dotnet run
```

---

## Reports

| Report | Description |
|--------|-------------|
| `YOLO26_Baseline_Analysis.pdf` | Baseline training analysis across 3 datasets |
| `YOLO26_Improvement_Report.pdf` | v3 improvement results with paper references |
| `YOLO26_Pipeline_Analysis.pdf` | EDA + preprocessing + augmentation analysis |
| `YOLO26_Engineering_Report.pdf` | Nano vs Small comparison + deployment guide |

---

## Tech Stack

- **Training**: Python 3.13, Ultralytics 8.4.21, PyTorch 2.6.0+cu124
- **Augmentation**: Albumentations 2.0.8
- **Inference**: ONNX Runtime 1.24 (GPU + CPU)
- **App**: C# / .NET 8 / WPF / OpenCvSharp4
- **GPU**: NVIDIA RTX 3060 12GB

---

## Author

**Kim Seongjin** ([@sjsr-0401](https://github.com/sjsr-0401))  
Electronic Engineering graduate | Industrial Vision & Equipment Control SW

---

## License

MIT
