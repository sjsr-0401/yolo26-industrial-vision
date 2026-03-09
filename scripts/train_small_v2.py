import multiprocessing
multiprocessing.freeze_support()

if __name__ == '__main__':
    from ultralytics import YOLO
    import torch
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    
    model = YOLO('yolo26s.pt')
    results = model.train(
        data=r'C:\dev\active\yolo26-industrial-vision\datasets\gc10-det\data.yaml',
        epochs=200,
        patience=30,
        imgsz=1024,
        batch=2,
        project=r'C:\dev\active\yolo26-industrial-vision\results',
        name='yolo26s_gc10det_v3',
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
    print('Training complete!')
    best = YOLO(r'C:\dev\active\yolo26-industrial-vision\results\yolo26s_gc10det_v3\weights\best.pt')
    best.export(format='onnx', imgsz=1024, simplify=True)
    print('ONNX export done!')
