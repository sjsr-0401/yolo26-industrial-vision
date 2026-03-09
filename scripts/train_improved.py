"""
Improved Training — Round 1
Target: NEU-DET + GC10-DET (DeepPCB already excellent)
Changes: imgsz=1280, epochs=200, copy_paste=0.3, scale=0.9, degrees=10
"""
from ultralytics import YOLO

RESULTS = r"C:\dev\active\yolo26-industrial-vision\results"

TASKS = [
    {
        "model": "yolo26n.pt",
        "data": r"C:\dev\active\yolo26-industrial-vision\datasets\neu-det\data.yaml",
        "name": "yolo26n_neudet_v2",
        "epochs": 200,
    },
    {
        "model": "yolo26n.pt",
        "data": r"C:\dev\active\yolo26-industrial-vision\datasets\gc10-det\data.yaml",
        "name": "yolo26n_gc10det_v2",
        "epochs": 200,
    },
]


def main():
    for task in TASKS:
        print("=" * 60)
        print(f"  IMPROVED: {task['name']}")
        print("=" * 60)

        model = YOLO(task["model"])
        model.train(
            data=task["data"],
            epochs=task["epochs"],
            imgsz=1280,
            batch=-1,
            device=0,
            project=RESULTS,
            name=task["name"],
            exist_ok=True,
            patience=30,
            cos_lr=True,
            # Augmentation improvements
            copy_paste=0.3,
            scale=0.9,
            degrees=10,
            flipud=0.5,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.15,
            close_mosaic=20,
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

    print("=" * 60)
    print("ALL IMPROVED TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
