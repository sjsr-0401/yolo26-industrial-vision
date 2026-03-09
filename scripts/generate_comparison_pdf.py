"""
Baseline vs Improved 비교 PDF 생성
학습 완료 후 실행
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import csv
import os
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

font_path = r"C:\Windows\Fonts\malgun.ttf"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

RESULTS = Path(r"C:\dev\active\yolo26-industrial-vision\results")
OUT_PDF = Path(r"C:\Users\admin\Desktop\YOLO26_Improvement_Report.pdf")

C_BASE = '#F44336'
C_IMPROVED = '#4CAF50'
C_BLUE = '#2196F3'
C_ORANGE = '#FF9800'
C_PURPLE = '#9C27B0'


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


def get_best(rows, key='metrics/mAP50(B)'):
    """Get best epoch metrics"""
    best_val = 0
    best_row = rows[-1] if rows else {}
    for r in rows:
        try:
            v = float(r.get(key, 0))
            if v > best_val:
                best_val = v
                best_row = r
        except:
            pass
    return {
        'P': float(best_row.get('metrics/precision(B)', 0)),
        'R': float(best_row.get('metrics/recall(B)', 0)),
        'mAP50': float(best_row.get('metrics/mAP50(B)', 0)),
        'mAP50_95': float(best_row.get('metrics/mAP50-95(B)', 0)),
    }


def get_history(rows, key):
    return [float(r.get(key, 0)) for r in rows]


COMPARISONS = [
    {
        'dataset': 'NEU-DET',
        'baseline': 'yolo26n_steel-defect',
        'improved': 'yolo26n_neudet_v2',
    },
    {
        'dataset': 'GC10-DET',
        'baseline': 'yolo26n_gc10det',
        'improved': 'yolo26n_gc10det_v2',
    },
]


def main():
    with PdfPages(str(OUT_PDF)) as pdf:
        # === Page 1: Title ===
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis('off')
        ax.text(0.5, 0.65, 'YOLO26 Industrial Defect Detection', fontsize=26, fontweight='bold',
                ha='center', va='center', transform=ax.transAxes)
        ax.text(0.5, 0.55, 'Baseline vs Improved — Comparison Report', fontsize=18, color='#555',
                ha='center', va='center', transform=ax.transAxes)
        ax.text(0.5, 0.45, '2026-03-07 | Kim Seongjin', fontsize=12, color='#999',
                ha='center', va='center', transform=ax.transAxes)

        # Config comparison box
        config_text = (
            "Baseline: imgsz=640, epochs=80, patience=20, scale=0.5, no copy_paste\n"
            "Improved: imgsz=1280, epochs=200, patience=30, scale=0.9, copy_paste=0.3, degrees=10"
        )
        ax.text(0.5, 0.30, config_text, fontsize=11, ha='center', va='center',
                transform=ax.transAxes, family='monospace',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='#E8EAF6', edgecolor='#3F51B5'))

        pdf.savefig(fig)
        plt.close()

        # === Page 2: Summary Table ===
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis('off')
        ax.text(0.5, 0.95, 'Performance Summary', fontsize=20, fontweight='bold',
                ha='center', va='top', transform=ax.transAxes)

        cols = ['Dataset', 'Version', 'Epochs', 'P', 'R', 'F1', 'mAP50', 'mAP50-95', 'Delta mAP50']
        rows_data = []

        for comp in COMPARISONS:
            base_rows = read_results(comp['baseline'])
            imp_rows = read_results(comp['improved'])

            base = get_best(base_rows) if base_rows else {}
            imp = get_best(imp_rows) if imp_rows else {}

            base_f1 = 2 * base.get('P', 0) * base.get('R', 0) / max(base.get('P', 0) + base.get('R', 0), 1e-6)
            imp_f1 = 2 * imp.get('P', 0) * imp.get('R', 0) / max(imp.get('P', 0) + imp.get('R', 0), 1e-6)

            delta = imp.get('mAP50', 0) - base.get('mAP50', 0)

            rows_data.append([
                comp['dataset'], 'Baseline',
                str(len(base_rows) if base_rows else 0),
                f"{base.get('P', 0):.3f}", f"{base.get('R', 0):.3f}", f"{base_f1:.3f}",
                f"{base.get('mAP50', 0):.3f}", f"{base.get('mAP50_95', 0):.3f}", '—'
            ])
            rows_data.append([
                '', 'Improved',
                str(len(imp_rows) if imp_rows else 0),
                f"{imp.get('P', 0):.3f}", f"{imp.get('R', 0):.3f}", f"{imp_f1:.3f}",
                f"{imp.get('mAP50', 0):.3f}", f"{imp.get('mAP50_95', 0):.3f}",
                f"+{delta:.3f}" if delta > 0 else f"{delta:.3f}"
            ])

        # DeepPCB (no improvement needed)
        deep_rows = read_results('yolo26n_deeppcb')
        if deep_rows:
            deep = get_best(deep_rows)
            deep_f1 = 2 * deep['P'] * deep['R'] / max(deep['P'] + deep['R'], 1e-6)
            rows_data.append([
                'DeepPCB', 'Baseline Only',
                str(len(deep_rows)),
                f"{deep['P']:.3f}", f"{deep['R']:.3f}", f"{deep_f1:.3f}",
                f"{deep['mAP50']:.3f}", f"{deep['mAP50_95']:.3f}", 'N/A (excellent)'
            ])

        table = ax.table(cellText=rows_data, colLabels=cols,
                         loc='center', cellLoc='center',
                         bbox=[0.02, 0.15, 0.96, 0.70])
        table.auto_set_font_size(False)
        table.set_fontsize(9)

        for j in range(len(cols)):
            table[0, j].set_facecolor('#1a1a2e')
            table[0, j].set_text_props(color='white', fontweight='bold')

        # Color improved rows
        for i, row in enumerate(rows_data, 1):
            if row[1] == 'Improved':
                for j in range(len(cols)):
                    table[i, j].set_facecolor('#E8F5E9')
            elif row[1] == 'Baseline Only':
                for j in range(len(cols)):
                    table[i, j].set_facecolor('#E3F2FD')

        pdf.savefig(fig)
        plt.close()

        # === Page 3+: Per-dataset comparison ===
        for comp in COMPARISONS:
            base_rows = read_results(comp['baseline'])
            imp_rows = read_results(comp['improved'])
            if not base_rows or not imp_rows:
                continue

            fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
            fig.suptitle(f"{comp['dataset']} — Baseline vs Improved", fontsize=16, fontweight='bold')

            # mAP50 curve
            ax = axes[0][0]
            base_map = get_history(base_rows, 'metrics/mAP50(B)')
            imp_map = get_history(imp_rows, 'metrics/mAP50(B)')
            ax.plot(range(1, len(base_map)+1), base_map, color=C_BASE, linewidth=1.5, label='Baseline (640)')
            ax.plot(range(1, len(imp_map)+1), imp_map, color=C_IMPROVED, linewidth=1.5, label='Improved (1280)')
            ax.axhline(y=0.85, color='gray', linestyle='--', alpha=0.5, label='Target 0.85')
            ax.set_title('mAP50', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('mAP50')
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

            # mAP50-95 curve
            ax = axes[0][1]
            base_map95 = get_history(base_rows, 'metrics/mAP50-95(B)')
            imp_map95 = get_history(imp_rows, 'metrics/mAP50-95(B)')
            ax.plot(range(1, len(base_map95)+1), base_map95, color=C_BASE, linewidth=1.5, label='Baseline')
            ax.plot(range(1, len(imp_map95)+1), imp_map95, color=C_IMPROVED, linewidth=1.5, label='Improved')
            ax.axhline(y=0.55, color='gray', linestyle='--', alpha=0.5, label='Target 0.55')
            ax.set_title('mAP50-95', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('mAP50-95')
            ax.set_ylim(0, 0.8)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

            # P/R comparison bar
            ax = axes[1][0]
            base_best = get_best(base_rows)
            imp_best = get_best(imp_rows)
            metrics = ['Precision', 'Recall', 'mAP50', 'mAP50-95']
            base_vals = [base_best['P'], base_best['R'], base_best['mAP50'], base_best['mAP50_95']]
            imp_vals = [imp_best['P'], imp_best['R'], imp_best['mAP50'], imp_best['mAP50_95']]

            x = np.arange(len(metrics))
            width = 0.35
            ax.bar(x - width/2, base_vals, width, label='Baseline', color=C_BASE, alpha=0.8)
            ax.bar(x + width/2, imp_vals, width, label='Improved', color=C_IMPROVED, alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(metrics)
            ax.set_ylim(0, 1.1)
            ax.set_title('Metrics Comparison', fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(axis='y', alpha=0.3)

            # Delta annotation
            for i, (b, im) in enumerate(zip(base_vals, imp_vals)):
                delta = im - b
                color = C_IMPROVED if delta > 0 else C_BASE
                ax.annotate(f"{delta:+.3f}", xy=(i + width/2, im + 0.02),
                            ha='center', fontsize=8, color=color, fontweight='bold')

            # Loss comparison
            ax = axes[1][1]
            base_box = get_history(base_rows, 'train/box_loss')
            imp_box = get_history(imp_rows, 'train/box_loss')
            base_cls = get_history(base_rows, 'train/cls_loss')
            imp_cls = get_history(imp_rows, 'train/cls_loss')

            ax.plot(range(1, len(base_box)+1), base_box, color=C_BASE, linewidth=1, linestyle='--', label='Base Box', alpha=0.7)
            ax.plot(range(1, len(imp_box)+1), imp_box, color=C_IMPROVED, linewidth=1, linestyle='--', label='Imp Box', alpha=0.7)
            ax.plot(range(1, len(base_cls)+1), base_cls, color=C_ORANGE, linewidth=1, label='Base Cls', alpha=0.7)
            ax.plot(range(1, len(imp_cls)+1), imp_cls, color=C_BLUE, linewidth=1, label='Imp Cls', alpha=0.7)
            ax.set_title('Training Loss', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

        # === Final Page: Conclusions ===
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis('off')
        ax.text(0.5, 0.92, 'Conclusions & Key Findings', fontsize=20, fontweight='bold',
                ha='center', va='top', transform=ax.transAxes)

        conclusions = []
        for comp in COMPARISONS:
            base_rows = read_results(comp['baseline'])
            imp_rows = read_results(comp['improved'])
            if base_rows and imp_rows:
                b = get_best(base_rows)
                im = get_best(imp_rows)
                delta = im['mAP50'] - b['mAP50']
                delta95 = im['mAP50_95'] - b['mAP50_95']
                conclusions.append(
                    f"{comp['dataset']}: mAP50 {b['mAP50']:.3f} -> {im['mAP50']:.3f} ({delta:+.3f}), "
                    f"mAP50-95 {b['mAP50_95']:.3f} -> {im['mAP50_95']:.3f} ({delta95:+.3f})"
                )

        y = 0.80
        for i, line in enumerate(conclusions):
            improved = '+' in line.split('(')[-1] if '(' in line else False
            color = '#2E7D32' if '+' in line.split('(')[1] else '#C62828'
            ax.text(0.08, y, f"{i+1}. {line}", fontsize=11, transform=ax.transAxes, color='#333')
            y -= 0.06

        # Key improvements text
        improvements_text = """
Key Improvements Applied:
1. imgsz: 640 -> 1280 (higher resolution for small defect detection)
2. epochs: 80 -> 200 (more training time for convergence)
3. copy_paste: 0 -> 0.3 (augmentation for small objects, CVPR 2021)
4. scale: 0.5 -> 0.9 (stronger scale augmentation)
5. degrees: 0 -> 10 (rotation augmentation)
6. close_mosaic: 10 -> 20 (longer fine-tuning without mosaic)

References:
- Simple Copy-Paste, Ghiasi et al., CVPR 2021
- NEU Surface Defect Database, Song & Yan, Applied Surface Science, 2013
- GC10-DET, Lv et al., Acta Optica Sinica, 2020
- YOLO26 (Ultralytics, 2026) — NMS-free end-to-end detection
"""
        ax.text(0.08, y - 0.05, improvements_text, fontsize=9, transform=ax.transAxes,
                family='monospace', color='#444', linespacing=1.5,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5', edgecolor='#999'))

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    print(f"PDF saved: {OUT_PDF}")
    print(f"Size: {OUT_PDF.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
