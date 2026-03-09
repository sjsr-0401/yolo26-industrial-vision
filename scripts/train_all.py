"""
YOLO26 vs YOLOv8 학습 파이프라인 — NEU-DET + PCB
RTX 3060 12GB 최적화
"""

import json
import time
import shutil
from pathlib import Path
from ultralytics import YOLO

PROJECT = Path(r"C:\dev\active\yolo26-industrial-vision")
RESULTS = PROJECT / "results"
MODELS = PROJECT / "models"

SCENARIOS = [
    {
        "key": "steel-defect",
        "name": "Steel Surface Defect (NEU-DET)",
        "data": str(PROJECT / "datasets" / "neu-det" / "data.yaml"),
        "epochs": 80,
        "imgsz": 640,
    },
    {
        "key": "pcb-defect", 
        "name": "PCB Defect Detection",
        "data": str(PROJECT / "datasets" / "pcb-defect" / "data.yaml"),
        "epochs": 80,
        "imgsz": 640,
    },
]

MODEL_CONFIGS = [
    {"key": "yolo26n", "weights": "yolo26n.pt", "label": "YOLO26-Nano"},
    {"key": "yolov8n", "weights": "yolov8n.pt", "label": "YOLOv8-Nano"},
]


def train_one(model_cfg, scenario):
    """단일 모델+시나리오 학습"""
    run_name = f"{model_cfg['key']}_{scenario['key']}"
    best = RESULTS / run_name / "weights" / "best.pt"
    
    if best.exists():
        print(f"\n  SKIP (already trained): {run_name}")
        return best
    
    print(f"\n{'='*60}")
    print(f"  {model_cfg['label']} x {scenario['name']}")
    print(f"{'='*60}")
    
    model = YOLO(model_cfg["weights"])
    
    t0 = time.time()
    model.train(
        data=scenario["data"],
        epochs=scenario["epochs"],
        imgsz=scenario["imgsz"],
        batch=-1,          # auto batch (GPU memory)
        device=0,
        project=str(RESULTS),
        name=run_name,
        exist_ok=True,
        patience=20,       # early stopping
        save=True,
        plots=True,
        verbose=True,
        workers=4,
        amp=True,          # mixed precision
        optimizer="auto",  # YOLO26: MuSGD / YOLOv8: AdamW
        lr0=0.01,
        cos_lr=True,       # cosine annealing
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.5,        # 산업 이미지 특성: 상하 대칭
        mosaic=1.0,
        mixup=0.1,
    )
    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed/60:.1f} min")
    
    return best


def evaluate_one(best_path, scenario, model_cfg):
    """모델 평가"""
    if not best_path.exists():
        return None
    
    model = YOLO(str(best_path))
    results = model.val(data=scenario["data"], device=0, verbose=False)
    
    # Speed benchmark
    import torch
    model_infer = YOLO(str(best_path))
    
    # Warmup
    dummy = torch.randn(1, 3, 640, 640).cuda()
    for _ in range(5):
        model_infer.predict(source=dummy, verbose=False, device=0)
    
    # Measure
    times = []
    for _ in range(20):
        t0 = time.time()
        model_infer.predict(source=dummy, verbose=False, device=0)
        times.append((time.time() - t0) * 1000)
    
    avg_ms = sum(times) / len(times)
    
    metrics = {
        "model": model_cfg["key"],
        "label": model_cfg["label"],
        "scenario": scenario["key"],
        "scenario_name": scenario["name"],
        "mAP50": round(float(results.box.map50), 4),
        "mAP50_95": round(float(results.box.map), 4),
        "precision": round(float(results.box.mp), 4),
        "recall": round(float(results.box.mr), 4),
        "avg_inference_ms": round(avg_ms, 1),
        "fps": round(1000 / avg_ms, 1),
    }
    
    return metrics


def export_best_onnx(best_path, run_name):
    """Best 모델 ONNX export"""
    if not best_path.exists():
        return
    
    model = YOLO(str(best_path))
    onnx_path = model.export(format="onnx", imgsz=640, simplify=True)
    
    dest = MODELS / f"{run_name}.onnx"
    MODELS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_path, dest)
    print(f"  ONNX: {dest} ({dest.stat().st_size/1024/1024:.1f}MB)")


def main():
    print("=" * 60)
    print("  YOLO26 vs YOLOv8 Industrial Vision Training")
    print("  GPU: RTX 3060 12GB | Ultralytics 8.4.21")
    print("=" * 60)
    
    RESULTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    
    all_metrics = []
    
    for scenario in SCENARIOS:
        for model_cfg in MODEL_CONFIGS:
            try:
                # Train
                best = train_one(model_cfg, scenario)
                
                # Evaluate
                if best and best.exists():
                    m = evaluate_one(best, scenario, model_cfg)
                    if m:
                        all_metrics.append(m)
                        print(f"\n  {m['label']} | {m['scenario']}")
                        print(f"    mAP50={m['mAP50']:.4f} | mAP50-95={m['mAP50_95']:.4f}")
                        print(f"    P={m['precision']:.4f} | R={m['recall']:.4f}")
                        print(f"    Speed={m['avg_inference_ms']:.1f}ms ({m['fps']:.0f} FPS)")
                    
                    # Export YOLO26 only
                    if "yolo26" in model_cfg["key"]:
                        run_name = f"{model_cfg['key']}_{scenario['key']}"
                        export_best_onnx(best, run_name)
                        
            except Exception as e:
                print(f"\n  ERROR: {model_cfg['key']}/{scenario['key']}: {e}")
                import traceback
                traceback.print_exc()
    
    # Save comparison
    summary = RESULTS / "comparison.json"
    with open(summary, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    
    # Print comparison table
    print(f"\n{'='*80}")
    print(f"  COMPARISON TABLE")
    print(f"{'='*80}")
    print(f"  {'Model':<16} {'Dataset':<20} {'mAP50':>8} {'mAP50-95':>10} {'P':>8} {'R':>8} {'FPS':>6}")
    print(f"  {'-'*76}")
    for m in all_metrics:
        print(f"  {m['label']:<16} {m['scenario']:<20} {m['mAP50']:>8.4f} {m['mAP50_95']:>10.4f} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['fps']:>6.0f}")
    
    print(f"\n  Results saved: {summary}")


if __name__ == "__main__":
    main()
