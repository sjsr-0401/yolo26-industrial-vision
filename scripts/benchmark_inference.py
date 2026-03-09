"""
ONNX Runtime Inference Benchmark
Nano vs Small (vs Medium when available)
GPU (CUDA) + CPU 속도 비교
"""
import time
import os
import sys
import json
import numpy as np
from pathlib import Path

# CUDA/cuDNN DLL path fix (Windows) — PyTorch bundles CUDA 12 + cuDNN 9
try:
    import torch
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.isdir(torch_lib):
        os.add_dll_directory(torch_lib)
        os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass

try:
    import onnxruntime as ort
except ImportError:
    print("Installing onnxruntime-gpu...")
    os.system(f"{sys.executable} -m pip install onnxruntime-gpu -q")
    import onnxruntime as ort

try:
    import cv2
except ImportError:
    os.system(f"{sys.executable} -m pip install opencv-python -q")
    import cv2

# ============================================================
# Configuration
# ============================================================
PROJECT = Path(r"C:\dev\active\yolo26-industrial-vision")
VAL_DIR = PROJECT / "datasets" / "gc10-det" / "images" / "val"
RESULTS_DIR = PROJECT / "results"
OUTPUT_JSON = RESULTS_DIR / "benchmark_results.json"

MODELS = {
    "YOLO26n (Nano)": RESULTS_DIR / "yolo26n_gc10det_v3" / "weights" / "best.onnx",
    "YOLO26s (Small)": RESULTS_DIR / "yolo26s_gc10det_v3" / "weights" / "best.onnx",
    # Medium - Colab 학습 완료 후 추가
    # "YOLO26m (Medium)": RESULTS_DIR / "yolo26m_gc10det_v3" / "weights" / "best.onnx",
}

IMGSZ = 1024
WARMUP = 10       # warmup iterations
NUM_IMAGES = 100  # benchmark iteration count
CONF_THRESHOLD = 0.25

CLASS_NAMES = [
    'crease', 'crescent_gap', 'inclusion', 'oil_spot', 'punching_hole',
    'rolled_pit', 'silk_spot', 'waist_folding', 'water_spot', 'welding_line'
]


def preprocess(img_path, imgsz):
    """Load and preprocess image for ONNX inference."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None, None
    orig_h, orig_w = img.shape[:2]

    # Resize with letterbox (maintain aspect ratio)
    scale = min(imgsz / orig_h, imgsz / orig_w)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to square
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    dx, dy = (imgsz - new_w) // 2, (imgsz - new_h) // 2
    canvas[dy:dy+new_h, dx:dx+new_w] = resized

    # HWC -> CHW, BGR -> RGB, normalize
    blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = np.expand_dims(blob, 0)  # (1, 3, H, W)

    return blob, img


def postprocess(output, conf_threshold=0.25):
    """Parse YOLO26 NMS-free output: (1, 300, 6) -> detections."""
    # output shape: (1, 300, 6) = [x1, y1, x2, y2, conf, class_id]
    preds = output[0]  # (300, 6)
    mask = preds[:, 4] > conf_threshold
    dets = preds[mask]
    return dets


def benchmark_model(onnx_path, provider, images, imgsz, warmup=10, n_iter=100):
    """Run inference benchmark for a single model + provider."""
    providers = [provider]
    try:
        session = ort.InferenceSession(str(onnx_path), providers=providers)
    except Exception as e:
        return {"error": str(e)}

    actual_provider = session.get_providers()[0]
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_shape = session.get_outputs()[0].shape

    # Warmup
    blob, _ = preprocess(images[0], imgsz)
    if blob is None:
        return {"error": "Failed to load warmup image"}

    for _ in range(warmup):
        session.run(None, {input_name: blob})

    # Benchmark
    times = []
    total_detections = 0
    for i in range(min(n_iter, len(images))):
        blob, _ = preprocess(images[i % len(images)], imgsz)
        if blob is None:
            continue

        t0 = time.perf_counter()
        outputs = session.run(None, {input_name: blob})
        t1 = time.perf_counter()

        times.append((t1 - t0) * 1000)  # ms
        dets = postprocess(outputs[0], CONF_THRESHOLD)
        total_detections += len(dets)

    times_arr = np.array(times)
    return {
        "provider": actual_provider,
        "input_shape": str(input_shape),
        "output_shape": str(output_shape),
        "warmup": warmup,
        "iterations": len(times),
        "mean_ms": round(float(times_arr.mean()), 2),
        "median_ms": round(float(np.median(times_arr)), 2),
        "min_ms": round(float(times_arr.min()), 2),
        "max_ms": round(float(times_arr.max()), 2),
        "std_ms": round(float(times_arr.std()), 2),
        "fps": round(1000.0 / float(times_arr.mean()), 1),
        "p95_ms": round(float(np.percentile(times_arr, 95)), 2),
        "p99_ms": round(float(np.percentile(times_arr, 99)), 2),
        "avg_detections": round(total_detections / len(times), 1),
    }


def main():
    print("=" * 70)
    print("  YOLO26 ONNX Runtime Inference Benchmark")
    print("  GPU: NVIDIA GeForce RTX 3060 12GB")
    print(f"  ONNX Runtime: {ort.__version__}")
    print(f"  Available providers: {ort.get_available_providers()}")
    print("=" * 70)

    # Load image paths
    images = sorted(VAL_DIR.glob("*.jpg"))[:NUM_IMAGES]
    if not images:
        images = sorted(VAL_DIR.glob("*.png"))[:NUM_IMAGES]
    print(f"\nBenchmark images: {len(images)}")

    providers_to_test = []
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers_to_test.append(("GPU (CUDA)", "CUDAExecutionProvider"))
    providers_to_test.append(("CPU", "CPUExecutionProvider"))

    all_results = {}

    for model_name, onnx_path in MODELS.items():
        if not onnx_path.exists():
            print(f"\n[SKIP] {model_name}: {onnx_path} not found")
            continue

        onnx_size = onnx_path.stat().st_size / (1024 * 1024)
        print(f"\n{'='*60}")
        print(f"  {model_name} ({onnx_size:.1f} MB)")
        print(f"  {onnx_path}")
        print(f"{'='*60}")

        model_results = {"onnx_size_mb": round(onnx_size, 1)}

        for prov_name, prov_id in providers_to_test:
            print(f"\n  [{prov_name}] Benchmarking ({WARMUP} warmup + {NUM_IMAGES} iterations)...")
            result = benchmark_model(
                onnx_path, prov_id, images, IMGSZ,
                warmup=WARMUP, n_iter=NUM_IMAGES
            )

            if "error" in result:
                print(f"    ERROR: {result['error']}")
            else:
                print(f"    Mean:   {result['mean_ms']:.1f} ms ({result['fps']:.0f} FPS)")
                print(f"    Median: {result['median_ms']:.1f} ms")
                print(f"    Min:    {result['min_ms']:.1f} ms | Max: {result['max_ms']:.1f} ms")
                print(f"    P95:    {result['p95_ms']:.1f} ms | P99: {result['p99_ms']:.1f} ms")
                print(f"    Avg detections: {result['avg_detections']}")

            model_results[prov_name] = result

        all_results[model_name] = model_results

    # Summary Table
    print(f"\n{'='*70}")
    print("  SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'ONNX':>8} {'GPU ms':>8} {'GPU FPS':>9} {'CPU ms':>8} {'CPU FPS':>9}")
    print("-" * 70)

    for model_name, data in all_results.items():
        onnx_size = data.get("onnx_size_mb", "?")
        gpu = data.get("GPU (CUDA)", {})
        cpu = data.get("CPU", {})
        gpu_ms = f"{gpu.get('mean_ms', '?')}" if 'mean_ms' in gpu else "N/A"
        gpu_fps = f"{gpu.get('fps', '?')}" if 'fps' in gpu else "N/A"
        cpu_ms = f"{cpu.get('mean_ms', '?')}" if 'mean_ms' in cpu else "N/A"
        cpu_fps = f"{cpu.get('fps', '?')}" if 'fps' in cpu else "N/A"
        print(f"{model_name:<20} {onnx_size:>7} MB {gpu_ms:>7} {gpu_fps:>8} {cpu_ms:>8} {cpu_fps:>8}")

    # Save results
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
