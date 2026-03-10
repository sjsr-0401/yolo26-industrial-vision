"""
InspectView v4 ONNX Validation Test
v4 (augmented) 모델로 GC10-DET val 이미지 전체 테스트
결과: Pass/Fail 카운트, 평균 추론 시간, 클래스별 검출 통계
"""
import os
import sys
import time
import json
import csv
from pathlib import Path
from collections import Counter

# CUDA DLL path fix
try:
    import torch
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.isdir(torch_lib):
        os.add_dll_directory(torch_lib)
        os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    os.system(f"{sys.executable} -m pip install onnxruntime-gpu -q")
    import onnxruntime as ort

try:
    import cv2
except ImportError:
    os.system(f"{sys.executable} -m pip install opencv-python -q")
    import cv2

# Config
PROJECT = Path(r"C:\dev\active\yolo26-industrial-vision")
VAL_IMAGES = PROJECT / "datasets" / "gc10-det" / "val" / "images"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
IMGSZ = 1024

CLASS_NAMES = [
    'punching', 'welding_line', 'crescent_gap', 'water_spot', 'oil_spot',
    'silk_spot', 'inclusion', 'rolled_pit', 'crease', 'waist_folding'
]

MODELS = {
    "Nano v3": PROJECT / "results" / "yolo26n_gc10det_v3" / "weights" / "best.onnx",
    "Nano v4 (aug)": PROJECT / "results" / "yolo26n_gc10det_v4_aug" / "weights" / "best.onnx",
    "Small v3": PROJECT / "results" / "yolo26s_gc10det_v3" / "weights" / "best.onnx",
}


def preprocess(img_path, imgsz):
    img = cv2.imread(str(img_path))
    if img is None:
        return None, None
    h, w = img.shape[:2]
    # Letterbox
    scale = min(imgsz / h, imgsz / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    dy, dx = (imgsz - nh) // 2, (imgsz - nw) // 2
    canvas[dy:dy+nh, dx:dx+nw] = resized
    blob = canvas.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
    return blob, (h, w, scale, dx, dy)


def postprocess(output, meta, conf_thresh, iou_thresh):
    """Process YOLO output: [1, num_classes+4, num_boxes]"""
    preds = output[0]  # shape: (1, 14, N) for 10 classes + 4 box coords
    if preds.ndim == 3:
        preds = preds[0]  # (14, N)
    
    # Transpose to (N, 14) if needed
    if preds.shape[0] < preds.shape[1]:
        preds = preds.T  # (N, 14)
    
    boxes = preds[:, :4]  # cx, cy, w, h
    scores = preds[:, 4:]  # class scores
    
    max_scores = scores.max(axis=1)
    mask = max_scores > conf_thresh
    
    if not mask.any():
        return []
    
    boxes = boxes[mask]
    scores = scores[mask]
    max_scores = max_scores[mask]
    class_ids = scores.argmax(axis=1)
    
    # Convert cx,cy,w,h to x1,y1,x2,y2
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    
    # NMS per class
    results = []
    for cls_id in np.unique(class_ids):
        cls_mask = class_ids == cls_id
        cls_boxes = np.stack([x1[cls_mask], y1[cls_mask], x2[cls_mask], y2[cls_mask]], axis=1)
        cls_scores = max_scores[cls_mask]
        
        indices = cv2.dnn.NMSBoxes(
            cls_boxes.tolist(),
            cls_scores.tolist(),
            conf_thresh, iou_thresh
        )
        if len(indices) > 0:
            for i in indices.flatten():
                results.append({
                    'class_id': int(cls_id),
                    'class_name': CLASS_NAMES[int(cls_id)],
                    'confidence': float(cls_scores[i]),
                    'box': cls_boxes[i].tolist()
                })
    
    return results


def test_model(model_name, model_path, images):
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"Model: {model_path}")
    print(f"{'='*60}")
    
    if not model_path.exists():
        print(f"  SKIP: Model not found")
        return None
    
    # Create session with GPU
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    try:
        sess = ort.InferenceSession(str(model_path), providers=providers)
        ep = sess.get_providers()[0]
    except Exception:
        sess = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        ep = 'CPUExecutionProvider'
    
    print(f"  Provider: {ep}")
    input_name = sess.get_inputs()[0].name
    
    pass_count = 0
    fail_count = 0
    error_count = 0
    total_time = 0
    class_counter = Counter()
    inference_times = []
    
    for i, img_path in enumerate(images):
        blob, meta = preprocess(img_path, IMGSZ)
        if blob is None:
            error_count += 1
            continue
        
        t0 = time.perf_counter()
        outputs = sess.run(None, {input_name: blob})
        t1 = time.perf_counter()
        
        elapsed_ms = (t1 - t0) * 1000
        inference_times.append(elapsed_ms)
        total_time += elapsed_ms
        
        detections = postprocess(outputs[0], meta, CONF_THRESHOLD, IOU_THRESHOLD)
        
        if len(detections) > 0:
            fail_count += 1
            for d in detections:
                class_counter[d['class_name']] += 1
        else:
            pass_count += 1
        
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(images)}] Pass={pass_count} Fail={fail_count} Avg={total_time/(i+1):.1f}ms")
    
    avg_ms = total_time / len(images) if images else 0
    
    result = {
        'model': model_name,
        'total_images': len(images),
        'pass': pass_count,
        'fail': fail_count,
        'errors': error_count,
        'avg_inference_ms': round(avg_ms, 1),
        'median_inference_ms': round(np.median(inference_times), 1) if inference_times else 0,
        'p95_inference_ms': round(np.percentile(inference_times, 95), 1) if inference_times else 0,
        'provider': ep,
        'class_detections': dict(class_counter.most_common()),
        'total_detections': sum(class_counter.values()),
    }
    
    print(f"\n  Results:")
    print(f"    Pass: {pass_count} | Fail: {fail_count} | Errors: {error_count}")
    print(f"    Avg: {avg_ms:.1f}ms | Median: {result['median_inference_ms']}ms | P95: {result['p95_inference_ms']}ms")
    print(f"    Total detections: {result['total_detections']}")
    print(f"    Class breakdown:")
    for cls, cnt in class_counter.most_common():
        print(f"      {cls}: {cnt}")
    
    return result


def main():
    # Collect val images
    images = sorted(VAL_IMAGES.glob("*.jpg")) + sorted(VAL_IMAGES.glob("*.png"))
    if not images:
        # Try alternate path
        alt = PROJECT / "datasets" / "gc10-det" / "images" / "val"
        images = sorted(alt.glob("*.jpg")) + sorted(alt.glob("*.png"))
    
    print(f"Val images: {len(images)}")
    
    all_results = {}
    for name, path in MODELS.items():
        result = test_model(name, path, images)
        if result:
            all_results[name] = result
    
    # Save results
    output = PROJECT / "results" / "inspectview_test_results.json"
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {output}")
    
    # Print comparison table
    print(f"\n{'='*70}")
    print(f"{'Model':<20} {'Pass':>6} {'Fail':>6} {'Avg(ms)':>8} {'Detections':>11}")
    print(f"{'-'*70}")
    for name, r in all_results.items():
        print(f"{name:<20} {r['pass']:>6} {r['fail']:>6} {r['avg_inference_ms']:>8.1f} {r['total_detections']:>11}")


if __name__ == '__main__':
    main()
