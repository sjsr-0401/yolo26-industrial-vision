"""
YOLO26 Small model training on GC10-DET (v3 config)
Compare with Nano baseline and Nano v3
"""
import multiprocessing
multiprocessing.freeze_support()

from ultralytics import YOLO
import torch

if __name__ == "__main__":
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # GC10-DET with Small model
    model = YOLO("yolo26s.pt")
    results = model.train(
        data=r"C:\dev\active\yolo26-industrial-vision\datasets\gc10-det\data.yaml",
        epochs=200,
        patience=30,
        imgsz=1024,
        batch=4,
        project=r"C:\dev\active\yolo26-industrial-vision\results",
        name="yolo26s_gc10det_v3",
        device=0,
        workers=0,
        cos_lr=True,
        copy_paste=0.2,
        scale=0.7,
        degrees=5,
        mixup=0.1,
        flipud=0.3,
        mosaic=1.0,
        close_mosaic=15,
        amp=True,
        verbose=True,
        exist_ok=True,
    )

    # Export ONNX
    print("\n--- Exporting ONNX ---")
    best_model = YOLO(r"C:\dev\active\yolo26-industrial-vision\results\yolo26s_gc10det_v3\weights\best.pt")
    best_model.export(format="onnx", imgsz=1024, simplify=True)
    print("ONNX export complete!")
