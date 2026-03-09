"""
Improved Training v3 — 데이터셋별 맞춤 설정
NEU-DET: imgsz=640 유지 (원본 200x200, 업스케일 무의미), epochs 200, augmentation 조절
GC10-DET: imgsz=1024 (2048->1024 적정), batch=2, epochs 200
"""
from ultralytics import YOLO

RESULTS = r"C:\dev\active\yolo26-industrial-vision\results"

TASKS = [
    {
        "model": "yolo26n.pt",
        "data": r"C:\dev\active\yolo26-industrial-vision\datasets\neu-det\data.yaml",
        "name": "yolo26n_neudet_v3",
        "epochs": 200,
        "imgsz": 640,
        # NEU-DET: 200x200 원본, 가벼운 augmentation
        "copy_paste": 0.0,
        "scale": 0.5,
        "degrees": 0,
        "mixup": 0.1,
        "close_mosaic": 15,
        "batch": -1,
    },
    {
        "model": "yolo26n.pt",
        "data": r"C:\dev\active\yolo26-industrial-vision\datasets\gc10-det\data.yaml",
        "name": "yolo26n_gc10det_v3",
        "epochs": 200,
        "imgsz": 1024,
        # GC10-DET: 2048x1000 원본, 1024가 적정
        "copy_paste": 0.2,
        "scale": 0.7,
        "degrees": 5,
        "mixup": 0.1,
        "close_mosaic": 15,
        "batch": 4,
    },
]


def main():
    for task in TASKS:
        print("=" * 60)
        print(f"  v3: {task['name']} (imgsz={task['imgsz']})")
        print("=" * 60)

        model = YOLO(task["model"])
        model.train(
            data=task["data"],
            epochs=task["epochs"],
            imgsz=task["imgsz"],
            batch=task["batch"],
            device=0,
            project=RESULTS,
            name=task["name"],
            exist_ok=True,
            patience=30,
            cos_lr=True,
            copy_paste=task["copy_paste"],
            scale=task["scale"],
            degrees=task["degrees"],
            flipud=0.5,
            fliplr=0.5,
            mosaic=1.0,
            mixup=task["mixup"],
            close_mosaic=task["close_mosaic"],
            amp=True,
            plots=True,
            verbose=True,
        )

        # Export ONNX
        best = f"{RESULTS}/{task['name']}/weights/best.pt"
        try:
            m = YOLO(best)
            m.export(format="onnx", simplify=True, opset=17)
            print(f"ONNX exported")
        except Exception as e:
            print(f"ONNX export error: {e}")

        print(f"{task['name']} done\n")

    print("ALL v3 TRAINING COMPLETE")


if __name__ == "__main__":
    main()
