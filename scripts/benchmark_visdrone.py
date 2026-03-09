"""
VisDrone 소형 객체 벤치마크: YOLO26 vs YOLOv8 (Nano + Small)
Pretrained COCO → VisDrone val 평가 (파인튜닝 없이 아키텍처 비교)
"""

import json
import time
import torch
from pathlib import Path
from ultralytics import YOLO

VD_YAML = r"C:\Users\admin\.openclaw\workspace\datasets\VisDrone\data.yaml"
RESULTS = Path(r"C:\dev\active\yolo26-industrial-vision\results")

MODELS = [
    ("yolo26n.pt", "YOLO26-Nano", "yolo26n"),
    ("yolov8n.pt", "YOLOv8-Nano", "yolov8n"),
    ("yolo26s.pt", "YOLO26-Small", "yolo26s"),
    ("yolov8s.pt", "YOLOv8-Small", "yolov8s"),
]


def benchmark_model(weights, label, key):
    """단일 모델 VisDrone 평가"""
    print(f"\n{'='*60}")
    print(f"  {label} ({weights})")
    print(f"{'='*60}")
    
    model = YOLO(weights)
    
    # Evaluate on VisDrone
    results = model.val(
        data=VD_YAML,
        device=0,
        imgsz=640,
        verbose=True,
        save_json=False,
        plots=False,
    )
    
    # Speed benchmark (GPU)
    dummy = torch.randn(1, 3, 640, 640).cuda()
    
    # Warmup
    for _ in range(10):
        model.predict(source=dummy, verbose=False, device=0)
    
    # Measure GPU
    times_gpu = []
    for _ in range(50):
        t0 = time.time()
        model.predict(source=dummy, verbose=False, device=0)
        times_gpu.append((time.time() - t0) * 1000)
    
    gpu_ms = sum(times_gpu) / len(times_gpu)
    
    # Measure CPU
    model_cpu = YOLO(weights)
    dummy_cpu = torch.randn(1, 3, 640, 640)
    
    for _ in range(3):
        model_cpu.predict(source=dummy_cpu, verbose=False, device='cpu')
    
    times_cpu = []
    for _ in range(10):
        t0 = time.time()
        model_cpu.predict(source=dummy_cpu, verbose=False, device='cpu')
        times_cpu.append((time.time() - t0) * 1000)
    
    cpu_ms = sum(times_cpu) / len(times_cpu)
    
    metrics = {
        "model": key,
        "label": label,
        "weights": weights,
        "dataset": "VisDrone (small objects)",
        "mAP50": round(float(results.box.map50), 4),
        "mAP50_95": round(float(results.box.map), 4),
        "precision": round(float(results.box.mp), 4),
        "recall": round(float(results.box.mr), 4),
        "gpu_ms": round(gpu_ms, 1),
        "gpu_fps": round(1000 / gpu_ms, 1),
        "cpu_ms": round(cpu_ms, 1),
        "cpu_fps": round(1000 / cpu_ms, 1),
        "params_M": round(sum(p.numel() for p in model.model.parameters()) / 1e6, 2),
    }
    
    print(f"\n  mAP50:    {metrics['mAP50']:.4f}")
    print(f"  mAP50-95: {metrics['mAP50_95']:.4f}")
    print(f"  P/R:      {metrics['precision']:.4f} / {metrics['recall']:.4f}")
    print(f"  GPU:      {metrics['gpu_ms']:.1f}ms ({metrics['gpu_fps']:.0f} FPS)")
    print(f"  CPU:      {metrics['cpu_ms']:.1f}ms ({metrics['cpu_fps']:.0f} FPS)")
    print(f"  Params:   {metrics['params_M']:.2f}M")
    
    return metrics


def main():
    print("=" * 60)
    print("  VisDrone Small Object Benchmark")
    print("  YOLO26 vs YOLOv8 (Nano + Small)")
    print("  60% objects < 32px — true small object test")
    print("=" * 60)
    
    all_metrics = []
    
    for weights, label, key in MODELS:
        try:
            m = benchmark_model(weights, label, key)
            all_metrics.append(m)
        except Exception as e:
            print(f"  ERROR: {label}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save
    out_path = RESULTS / "visdrone_benchmark.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    
    # Print comparison
    print(f"\n{'='*90}")
    print(f"  VISDRONE BENCHMARK — Small Object Detection")
    print(f"{'='*90}")
    print(f"  {'Model':<16} {'Params':>8} {'mAP50':>8} {'mAP50-95':>10} {'GPU(ms)':>8} {'CPU(ms)':>8} {'CPU FPS':>8}")
    print(f"  {'-'*82}")
    for m in all_metrics:
        print(f"  {m['label']:<16} {m['params_M']:>7.2f}M {m['mAP50']:>8.4f} {m['mAP50_95']:>10.4f} {m['gpu_ms']:>8.1f} {m['cpu_ms']:>8.1f} {m['cpu_fps']:>8.1f}")
    
    # YOLO26 vs YOLOv8 speedup
    print(f"\n  === Speed Comparison ===")
    for size in ['n', 's']:
        y26 = next((m for m in all_metrics if m['model'] == f'yolo26{size}'), None)
        y8 = next((m for m in all_metrics if m['model'] == f'yolov8{size}'), None)
        if y26 and y8:
            cpu_speedup = (y8['cpu_ms'] - y26['cpu_ms']) / y8['cpu_ms'] * 100
            print(f"  {size.upper()}: CPU speedup = {cpu_speedup:+.1f}% ({y8['cpu_ms']:.0f}ms → {y26['cpu_ms']:.0f}ms)")
            map_diff = y26['mAP50'] - y8['mAP50']
            print(f"  {size.upper()}: mAP50 diff  = {map_diff:+.4f}")
    
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
