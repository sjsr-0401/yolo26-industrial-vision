"""Small 모델 학습 — 완전 독립 실행. 모든 출력 파일로."""

import multiprocessing
multiprocessing.freeze_support()

if __name__ == '__main__':
    import sys, os, time, traceback
    
    log_path = r"C:\dev\active\yolo26-industrial-vision\results\train_small_standalone.log"
    status_path = r"C:\dev\active\yolo26-industrial-vision\results\train_status.txt"
    
    log = open(log_path, "w", encoding="utf-8")
    sys.stdout = log
    sys.stderr = log
    
    def write_status(msg):
        with open(status_path, "w", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}\n")
    
    write_status("STARTING")
    
    try:
        import torch
        from ultralytics import YOLO
        
        write_status(f"GPU: {torch.cuda.get_device_name(0)}, loading model...")
        
        model = YOLO('yolo26s.pt')
        write_status("Model loaded, starting training...")
        
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
        
        write_status("Training complete! Exporting ONNX...")
        
        best = YOLO(r'C:\dev\active\yolo26-industrial-vision\results\yolo26s_gc10det_v3\weights\best.pt')
        best.export(format='onnx', imgsz=1024, simplify=True)
        
        write_status("DONE - ONNX exported")
        
    except Exception as e:
        write_status(f"EXCEPTION: {type(e).__name__}: {e}")
        traceback.print_exc(file=log)
    finally:
        log.flush()
        log.close()
