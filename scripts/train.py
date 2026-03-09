"""
YOLO26 vs YOLOv8 산업용 비전 파인튜닝 파이프라인
- 각 시나리오별 YOLO26n/YOLOv8n 학습
- 성능 비교 (mAP, 속도)
- 결과 저장
"""

import json
import time
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(r"C:\dev\active\yolo26-industrial-vision")
DATASETS = PROJECT_ROOT / "datasets"
RESULTS = PROJECT_ROOT / "results"
MODELS = PROJECT_ROOT / "models"

# 시나리오 설정
SCENARIOS = {
    "steel-defect": {
        "name": "Steel Surface Defect Detection",
        "name_kr": "강철 표면 결함 검출",
        "dataset_dir": DATASETS / "neu-det",
        "classes": ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"],
        "epochs": 50,
        "imgsz": 640,
    },
    "pcb-defect": {
        "name": "PCB Defect Detection", 
        "name_kr": "PCB 결함 검출",
        "dataset_dir": DATASETS / "pcb-defect",
        "classes": ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"],
        "epochs": 50,
        "imgsz": 640,
    },
    "safety-helmet": {
        "name": "Safety Helmet Detection",
        "name_kr": "안전모 착용 감지",
        "dataset_dir": DATASETS / "safety-helmet",
        "classes": ["helmet", "head", "person"],
        "epochs": 50,
        "imgsz": 640,
    },
}

# 모델 설정
MODELS_CONFIG = {
    "yolo26n": {"weights": "yolo26n.pt", "label": "YOLO26-Nano"},
    "yolov8n": {"weights": "yolov8n.pt", "label": "YOLOv8-Nano"},
}


def find_data_yaml(dataset_dir: Path) -> str:
    """데이터셋 YAML 파일 찾기"""
    # 가능한 이름들
    for name in ["data.yaml", "dataset.yaml", f"{dataset_dir.name}.yaml"]:
        p = dataset_dir / name
        if p.exists():
            return str(p)
    
    # 하위 디렉토리에서 찾기
    for yaml_file in dataset_dir.rglob("*.yaml"):
        return str(yaml_file)
    
    return None


def create_data_yaml(scenario_key: str, scenario: dict) -> str:
    """YOLO data.yaml 생성"""
    dataset_dir = scenario["dataset_dir"]
    yaml_path = dataset_dir / "data.yaml"
    
    # 이미 존재하면 반환
    existing = find_data_yaml(dataset_dir)
    if existing:
        print(f"  Using existing: {existing}")
        return existing
    
    # 생성
    content = f"""# {scenario['name']} Dataset
path: {dataset_dir}
train: train/images
val: val/images

nc: {len(scenario['classes'])}
names: {scenario['classes']}
"""
    yaml_path.write_text(content, encoding='utf-8')
    print(f"  Created: {yaml_path}")
    return str(yaml_path)


def train_model(model_key: str, scenario_key: str, data_yaml: str, scenario: dict):
    """단일 모델 학습"""
    model_cfg = MODELS_CONFIG[model_key]
    run_name = f"{model_key}_{scenario_key}"
    
    print(f"\n{'='*60}")
    print(f"  Training: {model_cfg['label']} on {scenario['name_kr']}")
    print(f"{'='*60}")
    
    # 이미 학습된 모델 체크
    best_path = RESULTS / run_name / "weights" / "best.pt"
    if best_path.exists():
        print(f"  Already trained: {best_path}")
        return best_path
    
    model = YOLO(model_cfg["weights"])
    
    start = time.time()
    results = model.train(
        data=data_yaml,
        epochs=scenario["epochs"],
        imgsz=scenario["imgsz"],
        batch=-1,  # auto batch size
        device=0,  # GPU
        project=str(RESULTS),
        name=run_name,
        exist_ok=True,
        patience=15,  # early stopping
        save=True,
        plots=True,
        verbose=True,
    )
    elapsed = time.time() - start
    
    print(f"\n  Training complete: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    return RESULTS / run_name / "weights" / "best.pt"


def evaluate_model(model_path: Path, data_yaml: str, model_key: str, scenario_key: str):
    """모델 평가 + 결과 저장"""
    print(f"\n  Evaluating: {model_path.name}")
    
    model = YOLO(str(model_path))
    
    # Validation
    results = model.val(data=data_yaml, device=0, verbose=False)
    
    # Speed test (100 images)
    start = time.time()
    model.predict(
        source=str(Path(data_yaml).parent / "val" / "images"),
        device=0,
        verbose=False,
        save=False,
        max_det=100,
    )
    speed = (time.time() - start)
    
    metrics = {
        "model": model_key,
        "scenario": scenario_key,
        "mAP50": float(results.box.map50),
        "mAP50-95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "inference_time_total": round(speed, 2),
    }
    
    print(f"    mAP50: {metrics['mAP50']:.4f}")
    print(f"    mAP50-95: {metrics['mAP50-95']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall: {metrics['recall']:.4f}")
    
    return metrics


def export_onnx(model_path: Path, scenario_key: str, model_key: str):
    """ONNX export (C# SmartDetector 연동용)"""
    print(f"\n  Exporting ONNX: {model_key}_{scenario_key}")
    model = YOLO(str(model_path))
    onnx_path = model.export(format="onnx", imgsz=640, simplify=True)
    
    # Copy to models directory
    dest = MODELS / f"{model_key}_{scenario_key}.onnx"
    import shutil
    shutil.copy2(onnx_path, dest)
    print(f"    -> {dest}")
    return dest


def main():
    print("=" * 60)
    print("  YOLO26 Industrial Vision Training Pipeline")
    print("=" * 60)
    
    RESULTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    
    all_metrics = []
    
    for scenario_key, scenario in SCENARIOS.items():
        dataset_dir = scenario["dataset_dir"]
        
        # 데이터셋 확인
        if not dataset_dir.exists():
            print(f"\n  SKIP: {scenario['name_kr']} - dataset not found")
            continue
        
        # data.yaml 확인/생성
        data_yaml = create_data_yaml(scenario_key, scenario)
        if not data_yaml:
            print(f"  SKIP: No data.yaml for {scenario_key}")
            continue
        
        print(f"\n  Dataset: {data_yaml}")
        
        for model_key in MODELS_CONFIG:
            try:
                # 학습
                best_path = train_model(model_key, scenario_key, data_yaml, scenario)
                
                if best_path and best_path.exists():
                    # 평가
                    metrics = evaluate_model(best_path, data_yaml, model_key, scenario_key)
                    all_metrics.append(metrics)
                    
                    # YOLO26만 ONNX export
                    if "yolo26" in model_key:
                        export_onnx(best_path, scenario_key, model_key)
                        
            except Exception as e:
                print(f"  ERROR: {model_key}/{scenario_key}: {e}")
                import traceback
                traceback.print_exc()
    
    # 결과 요약
    summary_path = RESULTS / "comparison_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"  Results Summary")
    print(f"{'='*60}")
    print(f"  Saved to: {summary_path}")
    
    # 비교표 출력
    if all_metrics:
        print(f"\n  {'Model':<15} {'Scenario':<20} {'mAP50':<10} {'mAP50-95':<10} {'P':<10} {'R':<10}")
        print(f"  {'-'*75}")
        for m in all_metrics:
            print(f"  {m['model']:<15} {m['scenario']:<20} {m['mAP50']:<10.4f} {m['mAP50-95']:<10.4f} {m['precision']:<10.4f} {m['recall']:<10.4f}")


if __name__ == "__main__":
    main()
