"""Small 모델 학습 — 크래시 원인 분석용.
stderr/stdout 전부 파일로 리다이렉트 + 시스템 RAM 모니터링."""

import multiprocessing
multiprocessing.freeze_support()

if __name__ == '__main__':
    import sys, os, threading, time, psutil, traceback

    log_path = r"C:\dev\active\yolo26-industrial-vision\results\train_small_debug.log"
    
    class Logger:
        def __init__(self, filepath):
            self.terminal = sys.stdout
            self.log = open(filepath, "w", encoding="utf-8")
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    sys.stdout = Logger(log_path)
    sys.stderr = sys.stdout

    # RAM 모니터링 스레드
    stop_monitor = threading.Event()
    def monitor_ram():
        while not stop_monitor.is_set():
            mem = psutil.virtual_memory()
            if mem.percent > 85:
                print(f"\n[RAM WARNING] {mem.percent}% used, {mem.available/1024**3:.1f}GB free")
            time.sleep(10)
    
    t = threading.Thread(target=monitor_ram, daemon=True)
    t.start()

    try:
        import torch
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"System RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
        print(f"Available RAM: {psutil.virtual_memory().available / 1024**3:.1f} GB")
        
        from ultralytics import YOLO
        model = YOLO('yolo26s.pt')
        
        print("\n=== Starting training ===")
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
        print("\n=== Training complete! ===")
        
        best = YOLO(r'C:\dev\active\yolo26-industrial-vision\results\yolo26s_gc10det_v3\weights\best.pt')
        best.export(format='onnx', imgsz=1024, simplify=True)
        print('ONNX export done!')
        
    except Exception as e:
        print(f"\n\n=== EXCEPTION CAUGHT ===")
        print(f"Type: {type(e).__name__}")
        print(f"Message: {e}")
        traceback.print_exc()
    finally:
        stop_monitor.set()
        print(f"\n=== Process ended ===")
        mem = psutil.virtual_memory()
        print(f"Final RAM: {mem.percent}% used, {mem.available/1024**3:.1f}GB free")
