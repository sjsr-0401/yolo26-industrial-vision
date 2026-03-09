"""
YOLO26n v4 — 증강 데이터셋(gc10-det-aug) 학습
Nano먼저, 효과 확인 후 Small 결정
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from ultralytics import YOLO

RESULTS = r"C:\dev\active\yolo26-industrial-vision\results"
DATA = r"C:\dev\active\yolo26-industrial-vision\datasets\gc10-det-aug\data.yaml"

def main():
    print("=" * 60)
    print("  YOLO26n v4 — Augmented GC10-DET (4,329 train)")
    print("=" * 60)

    model = YOLO("yolo26n.pt")
    model.train(
        data=DATA,
        epochs=200,
        imgsz=1024,
        batch=4,
        device=0,
        project=RESULTS,
        name="yolo26n_gc10det_v4_aug",
        exist_ok=True,
        patience=30,
        cos_lr=True,
        copy_paste=0.1,      # 이미 오프라인 증강했으므로 줄임
        scale=0.5,
        degrees=5,
        mixup=0.05,
        close_mosaic=15,
        pretrained=True,
        optimizer="auto",
        workers=0,
        verbose=True,
    )

    # Export ONNX
    best = YOLO(os.path.join(RESULTS, "yolo26n_gc10det_v4_aug", "weights", "best.pt"))
    best.export(format="onnx", imgsz=1024)
    print("ONNX exported!")

    # Write status
    with open(os.path.join(RESULTS, "train_v4_status.txt"), "w") as f:
        f.write("DONE")
    print("Training complete!")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
