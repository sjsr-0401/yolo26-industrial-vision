"""
YOLO26 산업용 비전 검사 Gradio 데모
- 강철 표면 결함 검출
- PCB 결함 검출
- 모델 비교 (YOLO26 vs YOLOv8)
"""

import gradio as gr
import json
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

PROJECT = Path(r"C:\dev\active\yolo26-industrial-vision")
RESULTS = PROJECT / "results"

# 모델 로드 (lazy)
_models = {}

def get_model(scenario: str, model_type: str = "yolo26n") -> YOLO:
    """모델 캐시 로드"""
    key = f"{model_type}_{scenario}"
    if key not in _models:
        best_path = RESULTS / key / "weights" / "best.pt"
        if best_path.exists():
            _models[key] = YOLO(str(best_path))
        else:
            return None
    return _models[key]


def detect(image, scenario, model_type, confidence):
    """이미지에서 결함 검출"""
    if image is None:
        return None, "이미지를 업로드해주세요."
    
    model = get_model(scenario, model_type)
    if model is None:
        return image, f"모델을 찾을 수 없습니다: {model_type}_{scenario}"
    
    # Run inference
    results = model.predict(
        source=image,
        conf=confidence,
        device=0,
        verbose=False,
    )
    
    # Annotate
    annotated = results[0].plot(
        line_width=2,
        font_size=14,
    )
    
    # Summary
    boxes = results[0].boxes
    n_detections = len(boxes)
    
    if n_detections == 0:
        summary = "검출된 결함 없음"
    else:
        class_names = results[0].names
        counts = {}
        for cls_id in boxes.cls.cpu().numpy():
            name = class_names[int(cls_id)]
            counts[name] = counts.get(name, 0) + 1
        
        summary = f"총 {n_detections}개 결함 검출:\n"
        for name, count in sorted(counts.items(), key=lambda x: -x[1]):
            summary += f"  - {name}: {count}개\n"
        
        # Confidence stats
        confs = boxes.conf.cpu().numpy()
        summary += f"\n평균 신뢰도: {confs.mean():.2%}"
        summary += f"\n최고: {confs.max():.2%} | 최저: {confs.min():.2%}"
    
    return annotated, summary


def compare_models(image, scenario, confidence):
    """YOLO26 vs YOLOv8 비교"""
    if image is None:
        return None, None, "이미지를 업로드해주세요."
    
    results_text = ""
    outputs = []
    
    for model_type, label in [("yolo26n", "YOLO26-Nano"), ("yolov8n", "YOLOv8-Nano")]:
        model = get_model(scenario, model_type)
        if model is None:
            outputs.append(image)
            results_text += f"{label}: 모델 없음\n"
            continue
        
        import time
        t0 = time.time()
        results = model.predict(source=image, conf=confidence, device=0, verbose=False)
        elapsed = (time.time() - t0) * 1000
        
        annotated = results[0].plot(line_width=2, font_size=12)
        outputs.append(annotated)
        
        boxes = results[0].boxes
        n = len(boxes)
        avg_conf = float(boxes.conf.mean()) if n > 0 else 0
        
        results_text += f"[{label}]\n"
        results_text += f"  검출: {n}개 | 평균 신뢰도: {avg_conf:.2%} | 추론: {elapsed:.0f}ms\n\n"
    
    return outputs[0] if outputs else None, outputs[1] if len(outputs) > 1 else None, results_text


def load_comparison_data():
    """학습 결과 비교 데이터 로드"""
    comp_file = RESULTS / "comparison.json"
    if not comp_file.exists():
        return "학습 결과 파일이 없습니다."
    
    with open(comp_file, encoding="utf-8") as f:
        data = json.load(f)
    
    text = "=" * 60 + "\n"
    text += "  YOLO26 vs YOLOv8 성능 비교\n"
    text += "=" * 60 + "\n\n"
    
    for d in data:
        text += f"[{d['label']}] {d['scenario_name']}\n"
        text += f"  mAP@50:    {d['mAP50']:.4f}\n"
        text += f"  mAP@50-95: {d['mAP50_95']:.4f}\n"
        text += f"  Precision: {d['precision']:.4f}\n"
        text += f"  Recall:    {d['recall']:.4f}\n"
        text += f"  Speed:     {d['avg_inference_ms']:.1f}ms ({d['fps']:.0f} FPS)\n\n"
    
    return text


def get_sample_images(scenario):
    """샘플 이미지 경로 반환"""
    val_dir = PROJECT / "datasets" / scenario / "val" / "images"
    if val_dir.exists():
        imgs = sorted(val_dir.glob("*"))[:6]
        return [str(i) for i in imgs]
    return []


# ── Gradio UI ──
with gr.Blocks(
    title="YOLO26 Industrial Vision Inspector",
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="gray"),
    css="""
    .gradio-container { max-width: 1200px; margin: auto; }
    .header { text-align: center; padding: 20px; }
    """
) as app:
    
    gr.Markdown("""
    # 🏭 YOLO26 Industrial Vision Inspector
    ### 산업용 비전 검사 — 강철 표면 결함 / PCB 결함 검출
    **YOLO26** (2026 최신, NMS-free, CPU 43% 빠름) vs **YOLOv8** 성능 비교
    """)
    
    with gr.Tabs():
        # Tab 1: Single Detection
        with gr.Tab("🔍 결함 검출"):
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(type="numpy", label="검사 이미지")
                    scenario_select = gr.Radio(
                        choices=[
                            ("강철 표면 결함 (NEU-DET)", "steel-defect"),
                            ("PCB 결함", "pcb-defect"),
                        ],
                        value="steel-defect",
                        label="검사 시나리오"
                    )
                    model_select = gr.Radio(
                        choices=[
                            ("YOLO26-Nano", "yolo26n"),
                            ("YOLOv8-Nano", "yolov8n"),
                        ],
                        value="yolo26n",
                        label="모델"
                    )
                    conf_slider = gr.Slider(0.1, 0.9, 0.25, step=0.05, label="신뢰도 임계값")
                    detect_btn = gr.Button("검출 실행", variant="primary")
                
                with gr.Column():
                    img_output = gr.Image(label="검출 결과")
                    summary_output = gr.Textbox(label="검출 요약", lines=8)
            
            detect_btn.click(
                detect,
                inputs=[img_input, scenario_select, model_select, conf_slider],
                outputs=[img_output, summary_output]
            )
            
            # Sample images
            gr.Markdown("### 샘플 이미지")
            with gr.Row():
                steel_examples = gr.Examples(
                    examples=get_sample_images("neu-det"),
                    inputs=img_input,
                    label="강철 표면"
                )
        
        # Tab 2: Model Comparison
        with gr.Tab("⚖️ 모델 비교"):
            with gr.Row():
                with gr.Column():
                    comp_img = gr.Image(type="numpy", label="검사 이미지")
                    comp_scenario = gr.Radio(
                        choices=[
                            ("강철 표면 결함", "steel-defect"),
                            ("PCB 결함", "pcb-defect"),
                        ],
                        value="steel-defect",
                        label="시나리오"
                    )
                    comp_conf = gr.Slider(0.1, 0.9, 0.25, step=0.05, label="신뢰도")
                    comp_btn = gr.Button("비교 실행", variant="primary")
                
            with gr.Row():
                yolo26_out = gr.Image(label="YOLO26-Nano")
                yolov8_out = gr.Image(label="YOLOv8-Nano")
            
            comp_text = gr.Textbox(label="비교 결과", lines=8)
            
            comp_btn.click(
                compare_models,
                inputs=[comp_img, comp_scenario, comp_conf],
                outputs=[yolo26_out, yolov8_out, comp_text]
            )
        
        # Tab 3: Training Results
        with gr.Tab("📊 학습 결과"):
            gr.Markdown("### YOLO26 vs YOLOv8 성능 비교표")
            results_text = gr.Textbox(
                value=load_comparison_data,
                label="학습 결과",
                lines=25,
                interactive=False
            )
            refresh_btn = gr.Button("새로고침")
            refresh_btn.click(load_comparison_data, outputs=results_text)
    
    gr.Markdown("""
    ---
    **Tech Stack**: YOLO26 · YOLOv8 · Ultralytics 8.4.21 · PyTorch 2.6 · RTX 3060 12GB  
    **GitHub**: [sjsr-0401/yolo26-industrial-vision](https://github.com/sjsr-0401/yolo26-industrial-vision)  
    **Author**: Kim Seongjin (김성진)
    """)


if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
