"""EP1 batch 250~260 근처에서 크래시하는 이미지 찾기.
batch=2이므로 이미지 인덱스 500~520 근처.
각 이미지를 개별 로드+전처리해서 문제 이미지 특정."""

import multiprocessing
multiprocessing.freeze_support()

if __name__ == '__main__':
    import os, sys, traceback
    from pathlib import Path
    from ultralytics import YOLO
    import torch
    import cv2
    import numpy as np

    data_dir = Path(r"C:\dev\active\yolo26-industrial-vision\datasets\gc10-det")
    train_imgs = sorted((data_dir / "images" / "train").glob("*"))
    train_labels = data_dir / "labels" / "train"

    print(f"Total train images: {len(train_imgs)}")

    # 1) 기본 이미지 검사: 읽을 수 있는지, 크기, 채널
    bad_images = []
    for i, img_path in enumerate(train_imgs):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[BAD] #{i} {img_path.name} - cv2.imread returned None")
                bad_images.append((i, img_path.name, "imread_none"))
                continue
            h, w = img.shape[:2]
            if h == 0 or w == 0:
                print(f"[BAD] #{i} {img_path.name} - zero dimension: {w}x{h}")
                bad_images.append((i, img_path.name, "zero_dim"))
                continue
            if len(img.shape) < 3:
                print(f"[WARN] #{i} {img_path.name} - grayscale {img.shape}")
            # 라벨 파일 확인
            lbl_path = train_labels / (img_path.stem + ".txt")
            if lbl_path.exists():
                with open(lbl_path, 'r') as f:
                    lines = f.readlines()
                for li, line in enumerate(lines):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        print(f"[BAD] #{i} {img_path.name} label line {li}: '{line.strip()}'")
                        bad_images.append((i, img_path.name, f"bad_label_line_{li}"))
                    else:
                        cls = int(parts[0])
                        coords = [float(x) for x in parts[1:5]]
                        if any(c < 0 or c > 1.0 for c in coords):
                            print(f"[BAD] #{i} {img_path.name} label line {li}: coords out of [0,1] -> {coords}")
                            bad_images.append((i, img_path.name, f"oob_coords_line_{li}"))
                        if coords[2] <= 0 or coords[3] <= 0:
                            print(f"[BAD] #{i} {img_path.name} label line {li}: zero w/h -> {coords}")
                            bad_images.append((i, img_path.name, f"zero_wh_line_{li}"))
        except Exception as e:
            print(f"[ERROR] #{i} {img_path.name} - {e}")
            bad_images.append((i, img_path.name, str(e)))

    print(f"\n=== Scan complete ===")
    print(f"Total bad/suspicious: {len(bad_images)}")
    for idx, name, reason in bad_images:
        print(f"  #{idx} {name}: {reason}")

    # 2) 크래시 구간 (batch 248~270, batch_size=2 -> img 496~540) 집중 검사
    print(f"\n=== Crash zone analysis (img 490-550) ===")
    for i in range(max(0, 490), min(len(train_imgs), 550)):
        img_path = train_imgs[i]
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  #{i} {img_path.name}: CANNOT READ")
            continue
        h, w, c = img.shape
        filesize = os.path.getsize(img_path)
        # 리사이즈 테스트
        try:
            resized = cv2.resize(img, (1024, 1024))
            tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0).cuda()
            del tensor
            torch.cuda.empty_cache()
            status = "OK"
        except Exception as e:
            status = f"FAIL: {e}"
        print(f"  #{i} {img_path.name}: {w}x{h}x{c}, {filesize/1024:.0f}KB -> {status}")

    # 3) 전체 데이터셋에서 corrupt 이미지 비율
    print(f"\n=== Summary ===")
    print(f"Total images: {len(train_imgs)}")
    print(f"Bad images: {len(bad_images)}")
    if bad_images:
        print("Action needed: remove or fix these images before training")
