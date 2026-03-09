"""
Baseline 분석 PDF 생성
- 6개 모델 결과 비교 차트
- 클래스별 성능 분석
- 개선 근거 정리
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import csv
import json
import os
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

# 한글 폰트
font_path = r"C:\Windows\Fonts\malgun.ttf"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

RESULTS = Path(r"C:\dev\active\yolo26-industrial-vision\results")
OUT_PDF = Path(r"C:\Users\admin\Desktop\YOLO26_Baseline_Analysis.pdf")

# Colors
C_BLUE = '#2196F3'
C_ORANGE = '#FF9800'
C_GREEN = '#4CAF50'
C_RED = '#F44336'
C_PURPLE = '#9C27B0'
C_TEAL = '#009688'
COLORS = [C_BLUE, C_ORANGE, C_GREEN, C_RED, C_PURPLE, C_TEAL]


def read_results(name):
    csv_path = RESULTS / name / "results.csv"
    if not csv_path.exists():
        return None
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def get_final(rows):
    if not rows:
        return {}
    last = rows[-1]
    return {
        'epochs': len(rows),
        'P': float(last.get('metrics/precision(B)', 0)),
        'R': float(last.get('metrics/recall(B)', 0)),
        'mAP50': float(last.get('metrics/mAP50(B)', 0)),
        'mAP50_95': float(last.get('metrics/mAP50-95(B)', 0)),
    }


def get_history(rows, key):
    vals = []
    for r in rows:
        try:
            vals.append(float(r.get(key, 0)))
        except:
            vals.append(0)
    return vals


MODELS = {
    'yolo26n_steel-defect': {'label': 'YOLO26n / NEU-DET', 'dataset': 'NEU-DET'},
    'yolov8n_steel-defect': {'label': 'YOLOv8n / NEU-DET', 'dataset': 'NEU-DET'},
    'yolo26n_deeppcb': {'label': 'YOLO26n / DeepPCB', 'dataset': 'DeepPCB'},
    'yolo26n_pcb-defect': {'label': 'YOLO26n / PCB-Defect', 'dataset': 'PCB-Defect'},
    'yolov8n_pcb-defect': {'label': 'YOLOv8n / PCB-Defect', 'dataset': 'PCB-Defect'},
    'yolo26n_gc10det': {'label': 'YOLO26n / GC10-DET', 'dataset': 'GC10-DET'},
}


def main():
    all_results = {}
    all_history = {}
    for name, info in MODELS.items():
        rows = read_results(name)
        if rows:
            all_results[name] = get_final(rows)
            all_results[name]['label'] = info['label']
            all_results[name]['dataset'] = info['dataset']
            all_history[name] = rows

    with PdfPages(str(OUT_PDF)) as pdf:
        # === Page 1: Title + Summary Table ===
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis('off')

        ax.text(0.5, 0.92, 'YOLO26 Industrial Defect Detection', fontsize=24, fontweight='bold',
                ha='center', va='top', transform=ax.transAxes)
        ax.text(0.5, 0.86, 'Baseline Analysis Report', fontsize=18, color='#666',
                ha='center', va='top', transform=ax.transAxes)
        ax.text(0.5, 0.80, '2026-03-07 | Kim Seongjin', fontsize=12, color='#999',
                ha='center', va='top', transform=ax.transAxes)

        # Summary table
        cols = ['Model / Dataset', 'Epochs', 'Precision', 'Recall', 'F1', 'mAP50', 'mAP50-95', 'Grade']
        rows_data = []
        for name in MODELS:
            r = all_results.get(name, {})
            if not r:
                continue
            p, rc = r.get('P', 0), r.get('R', 0)
            f1 = 2 * p * rc / (p + rc) if (p + rc) > 0 else 0
            m50 = r.get('mAP50', 0)
            m95 = r.get('mAP50_95', 0)

            if m50 >= 0.90:
                grade = 'Excellent'
            elif m50 >= 0.80:
                grade = 'Good'
            elif m50 >= 0.70:
                grade = 'Fair'
            else:
                grade = 'Needs Work'

            rows_data.append([
                r['label'], str(r['epochs']),
                f"{p:.3f}", f"{rc:.3f}", f"{f1:.3f}",
                f"{m50:.3f}", f"{m95:.3f}", grade
            ])

        table = ax.table(cellText=rows_data, colLabels=cols,
                         loc='center', cellLoc='center',
                         bbox=[0.02, 0.10, 0.96, 0.60])
        table.auto_set_font_size(False)
        table.set_fontsize(9)

        # Color header
        for j in range(len(cols)):
            table[0, j].set_facecolor('#1a1a2e')
            table[0, j].set_text_props(color='white', fontweight='bold')

        # Color grades
        last_col = len(cols) - 1
        for i, row in enumerate(rows_data, 1):
            grade = row[-1]
            if grade == 'Excellent':
                table[i, last_col].set_facecolor('#C8E6C9')
            elif grade == 'Good':
                table[i, last_col].set_facecolor('#DCEDC8')
            elif grade == 'Fair':
                table[i, last_col].set_facecolor('#FFF9C4')
            else:
                table[i, last_col].set_facecolor('#FFCDD2')

        ax.text(0.5, 0.05, 'DeepPCB: Excellent (mAP50=0.984) | NEU-DET: Fair | GC10-DET: Needs Work',
                fontsize=11, ha='center', va='bottom', transform=ax.transAxes,
                style='italic', color='#333')

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # === Page 2: Bar Chart Comparison ===
        fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))

        labels = [all_results[n]['label'] for n in MODELS if n in all_results]
        metrics_names = ['Precision', 'Recall', 'mAP50', 'mAP50-95']
        metrics_keys = ['P', 'R', 'mAP50', 'mAP50_95']

        x = np.arange(len(labels))
        width = 0.18

        ax1 = axes[0]
        for i, (mname, mkey) in enumerate(zip(metrics_names, metrics_keys)):
            vals = [all_results[n].get(mkey, 0) for n in MODELS if n in all_results]
            ax1.bar(x + i * width, vals, width, label=mname, color=COLORS[i], alpha=0.85)

        ax1.set_ylabel('Score')
        ax1.set_title('Baseline Performance Comparison', fontweight='bold')
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels(labels, rotation=30, ha='right', fontsize=7)
        ax1.legend(fontsize=8)
        ax1.set_ylim(0, 1.1)
        ax1.axhline(y=0.85, color='red', linestyle='--', alpha=0.5, label='Target 0.85')
        ax1.grid(axis='y', alpha=0.3)

        # Radar chart - per dataset best
        ax2 = axes[1]
        ax2.remove()
        ax2 = fig.add_subplot(122, projection='polar')

        categories = ['Precision', 'Recall', 'mAP50', 'mAP50-95', 'F1']
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        datasets_to_plot = {
            'NEU-DET': 'yolo26n_steel-defect',
            'DeepPCB': 'yolo26n_deeppcb',
            'GC10-DET': 'yolo26n_gc10det',
        }

        for idx, (ds_name, model_name) in enumerate(datasets_to_plot.items()):
            r = all_results.get(model_name, {})
            if not r:
                continue
            p, rc = r.get('P', 0), r.get('R', 0)
            f1 = 2 * p * rc / (p + rc) if (p + rc) > 0 else 0
            values = [p, rc, r.get('mAP50', 0), r.get('mAP50_95', 0), f1]
            values += values[:1]
            ax2.plot(angles, values, 'o-', linewidth=2, label=ds_name, color=COLORS[idx])
            ax2.fill(angles, values, alpha=0.1, color=COLORS[idx])

        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(categories, fontsize=8)
        ax2.set_ylim(0, 1)
        ax2.set_title('YOLO26n per Dataset', fontweight='bold', pad=20)
        ax2.legend(loc='lower right', fontsize=8)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # === Page 3: Training Curves ===
        fig, axes = plt.subplots(2, 3, figsize=(11.69, 8.27))

        curve_models = ['yolo26n_steel-defect', 'yolo26n_deeppcb', 'yolo26n_gc10det',
                        'yolov8n_steel-defect', 'yolov8n_pcb-defect', 'yolo26n_pcb-defect']

        for idx, name in enumerate(curve_models):
            ax = axes[idx // 3][idx % 3]
            rows = all_history.get(name)
            if not rows:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                continue

            map50 = get_history(rows, 'metrics/mAP50(B)')
            map95 = get_history(rows, 'metrics/mAP50-95(B)')
            epochs = list(range(1, len(map50) + 1))

            ax.plot(epochs, map50, color=C_BLUE, linewidth=1.5, label='mAP50')
            ax.plot(epochs, map95, color=C_ORANGE, linewidth=1.5, label='mAP50-95')
            ax.axhline(y=0.85, color='red', linestyle='--', alpha=0.4)
            ax.set_title(MODELS.get(name, {}).get('label', name), fontsize=9, fontweight='bold')
            ax.set_xlabel('Epoch', fontsize=7)
            ax.set_ylabel('mAP', fontsize=7)
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=6)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=7)

        fig.suptitle('Training Curves (mAP50 / mAP50-95)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # === Page 4: Loss Curves ===
        fig, axes = plt.subplots(2, 3, figsize=(11.69, 8.27))

        for idx, name in enumerate(curve_models):
            ax = axes[idx // 3][idx % 3]
            rows = all_history.get(name)
            if not rows:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                continue

            box_loss = get_history(rows, 'train/box_loss')
            cls_loss = get_history(rows, 'train/cls_loss')
            epochs = list(range(1, len(box_loss) + 1))

            ax.plot(epochs, box_loss, color=C_RED, linewidth=1.2, label='Box Loss')
            ax.plot(epochs, cls_loss, color=C_PURPLE, linewidth=1.2, label='Cls Loss')
            ax.set_title(MODELS.get(name, {}).get('label', name), fontsize=9, fontweight='bold')
            ax.set_xlabel('Epoch', fontsize=7)
            ax.set_ylabel('Loss', fontsize=7)
            ax.legend(fontsize=6)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=7)

        fig.suptitle('Training Loss Curves', fontsize=14, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # === Page 5: Problem Diagnosis ===
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis('off')

        ax.text(0.5, 0.95, 'Problem Diagnosis & Improvement Strategy',
                fontsize=20, fontweight='bold', ha='center', va='top', transform=ax.transAxes)

        problems = [
            ("1. Model Capacity (Nano = 2.5M params)",
             "Nano has 4x fewer params than Small.\n"
             "Complex 10-class GC10-DET requires more capacity.\n"
             "Expected: +4~5% mAP50 with Small model."),

            ("2. Resolution Mismatch",
             "GC10-DET: 2048x1000 -> 640 = 3.2x downscale -> small defects lost.\n"
             "NEU-DET: 200x200 -> 640 = upscale with no real detail gain.\n"
             "Fix: imgsz=1280 for GC10-DET. Ultralytics docs recommend matching input scale."),

            ("3. Insufficient Epochs",
             "80 epochs + patience 20 may early-stop before convergence.\n"
             "Ultralytics recommends 300 epochs for best results.\n"
             "Fix: epochs=200, patience=30."),

            ("4. Augmentation Gaps",
             "Missing: copy_paste (CVPR 2021, +1-2% mAP for small objects).\n"
             "Missing: scale=0.9, degrees=10 for robustness.\n"
             "Fix: copy_paste=0.3, scale=0.9, degrees=10."),

            ("5. DeepPCB already excellent (mAP50=0.984)",
             "No improvement needed. Clean data + matched resolution (640x640).\n"
             "This proves the pipeline works well when data quality is good."),
        ]

        y = 0.85
        for title, desc in problems:
            ax.text(0.05, y, title, fontsize=12, fontweight='bold',
                    transform=ax.transAxes, color='#1a1a2e')
            ax.text(0.08, y - 0.03, desc, fontsize=9, transform=ax.transAxes,
                    color='#444', family='monospace', linespacing=1.6)
            y -= 0.18

        # Improvement config box
        ax.text(0.5, 0.08, 'Improved Config: imgsz=1280 | epochs=200 | copy_paste=0.3 | scale=0.9 | Target: P/R 0.85+',
                fontsize=11, ha='center', va='bottom', transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#E3F2FD', edgecolor='#2196F3'))

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    print(f"PDF saved: {OUT_PDF}")
    print(f"Size: {OUT_PDF.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
