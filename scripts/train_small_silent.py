"""Small 모델 학습 — stdout/stderr를 파일로 완전 리다이렉트.
콘솔 출력 최소화로 파이프 버퍼 문제 회피."""

import multiprocessing
multiprocessing.freeze_support()

if __name__ == '__main__':
    import sys, os
    
    log_path = r"C:\dev\active\yolo26-industrial-vision\results\train_small.log"
    
    # stdout/stderr를 파일로 완전 리다이렉트
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = log_file
    sys.stderr = log_file
    
    # 원본 stdout 보존 (최소 상태 출력용)
    real_stdout = open("CON", "w")
    
    try:
        import torch
        from ultralytics import YOLO
        
        real_stdout.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
        real_stdout.write("Starting YOLO26s GC10-DET training...\n")
        real_stdout.write(f"Log: {log_path}\n")
        real_stdout.flush()
        
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
        
        real_stdout.write("Training complete! Exporting ONNX...\n")
        real_stdout.flush()
        
        best = YOLO(r'C:\dev\active\yolo26-industrial-vision\results\yolo26s_gc10det_v3\weights\best.pt')
        best.export(format='onnx', imgsz=1024, simplify=True)
        
        real_stdout.write("ONNX export done!\n")
        real_stdout.flush()
        
    except Exception as e:
        import traceback
        real_stdout.write(f"\nEXCEPTION: {type(e).__name__}: {e}\n")
        traceback.print_exc(file=real_stdout)
        real_stdout.flush()
        traceback.print_exc(file=log_file)
    finally:
        log_file.flush()
        log_file.close()
        real_stdout.write("Process ended.\n")
        real_stdout.flush()
