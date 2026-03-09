"""
Phase 2 학습: YOLO26n x DeepPCB + GC10-DET
"""
from ultralytics import YOLO

RESULTS = r"C:\dev\active\yolo26-industrial-vision\results"

TASKS = [
    {
        "model": "yolo26n.pt",
        "data": r"C:\dev\active\yolo26-industrial-vision\datasets\deeppcb\data.yaml",
        "name": "yolo26n_deeppcb",
        "epochs": 100,
    },
    {
        "model": "yolo26n.pt",
        "data": r"C:\dev\active\yolo26-industrial-vision\datasets\gc10-det\data.yaml",
        "name": "yolo26n_gc10det",
        "epochs": 80,
    },
]


def main():
    for task in TASKS:
        print("=" * 60)
        print(f"  {task['name']}")
        print("=" * 60)

        model = YOLO(task["model"])
        model.train(
            data=task["data"],
            epochs=task["epochs"],
            imgsz=640,
            batch=-1,
            device=0,
            project=RESULTS,
            name=task["name"],
            exist_ok=True,
            patience=20,
            cos_lr=True,
            flipud=0.5,
            mosaic=1.0,
            mixup=0.1,
            amp=True,
            plots=True,
            verbose=True,
        )

        # Export ONNX
        best = f"{RESULTS}/{task['name']}/weights/best.pt"
        try:
            m = YOLO(best)
            m.export(format="onnx", simplify=True, opset=17)
            print(f"ONNX exported: {best.replace('.pt', '.onnx')}")
        except Exception as e:
            print(f"ONNX export error: {e}")

        print(f"{task['name']} done\n")


if __name__ == "__main__":
    main()
