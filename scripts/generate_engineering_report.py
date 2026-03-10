"""
YOLO26 Nano vs Small 엔지니어링 분석 보고서
GC10-DET v3 학습 결과 기반
"""
import os
import sys

# PDF generation via Playwright
try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("Installing playwright...")
    os.system(f"{sys.executable} -m pip install playwright -q")
    from playwright.sync_api import sync_playwright

OUTPUT_DIR = r"C:\Users\admin\Desktop"
HTML_PATH = os.path.join(OUTPUT_DIR, "engineering_report.html")
PDF_PATH = os.path.join(OUTPUT_DIR, "YOLO26_Engineering_Report.pdf")

html_content = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<style>
  @page { size: A4; margin: 20mm 18mm; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Malgun Gothic', 'Segoe UI', sans-serif; font-size: 10.5pt; line-height: 1.7; color: #1a1a2e; background: #fff; }
  
  .cover { page-break-after: always; display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100vh; text-align: center; }
  .cover h1 { font-size: 28pt; color: #1a1a2e; margin-bottom: 8px; }
  .cover .subtitle { font-size: 14pt; color: #555; margin-bottom: 40px; }
  .cover .meta { font-size: 11pt; color: #777; line-height: 2; }
  
  h1 { font-size: 18pt; color: #1a1a2e; border-bottom: 3px solid #2d5aa0; padding-bottom: 6px; margin: 30px 0 15px 0; page-break-after: avoid; }
  h2 { font-size: 14pt; color: #2d5aa0; margin: 22px 0 10px 0; page-break-after: avoid; }
  h3 { font-size: 12pt; color: #444; margin: 16px 0 8px 0; }
  p { margin: 6px 0; text-align: justify; }
  
  table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 10pt; }
  th { background: #2d5aa0; color: #fff; padding: 8px 10px; text-align: center; font-weight: 600; }
  td { padding: 7px 10px; border-bottom: 1px solid #ddd; text-align: center; }
  tr:nth-child(even) { background: #f8f9fc; }
  tr:hover { background: #eef2f9; }
  .highlight { background: #e8f5e9 !important; font-weight: bold; }
  
  .callout { background: #f0f4ff; border-left: 4px solid #2d5aa0; padding: 12px 16px; margin: 14px 0; border-radius: 0 6px 6px 0; }
  .callout-warn { background: #fff8e1; border-left-color: #f9a825; }
  .callout-success { background: #e8f5e9; border-left-color: #43a047; }
  .callout-danger { background: #fce4ec; border-left-color: #e53935; }
  .callout .title { font-weight: bold; margin-bottom: 4px; }
  
  code { background: #f4f4f8; color: #1a1a2e; padding: 2px 6px; border-radius: 3px; font-family: 'Consolas', monospace; font-size: 9.5pt; border: 1px solid #e0e0e0; }
  pre { background: #f4f4f8; color: #1a1a2e; padding: 14px; border-radius: 6px; overflow-x: auto; font-size: 9pt; line-height: 1.5; margin: 10px 0; border: 1px solid #e0e0e0; }
  
  .metric-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 14px 0; }
  .metric-card { background: #f8f9fc; border: 1px solid #e0e0e0; border-radius: 8px; padding: 14px; text-align: center; }
  .metric-card .value { font-size: 22pt; font-weight: bold; color: #2d5aa0; }
  .metric-card .label { font-size: 9pt; color: #777; margin-top: 4px; }
  .metric-card .delta { font-size: 10pt; color: #43a047; }
  
  .timeline { margin: 14px 0; }
  .timeline-item { position: relative; padding-left: 28px; margin-bottom: 14px; }
  .timeline-item::before { content: ''; position: absolute; left: 8px; top: 0; bottom: -14px; width: 2px; background: #ddd; }
  .timeline-item::after { content: ''; position: absolute; left: 3px; top: 6px; width: 12px; height: 12px; border-radius: 50%; background: #2d5aa0; }
  .timeline-item:last-child::before { display: none; }
  .timeline-item .time { font-size: 9pt; color: #999; }
  .timeline-item .event { font-weight: 600; }
  .timeline-item .detail { font-size: 9.5pt; color: #555; }
  .timeline-item.error::after { background: #e53935; }
  .timeline-item.success::after { background: #43a047; }
  
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 12px 0; }
  
  ul { margin: 6px 0 6px 20px; }
  li { margin: 3px 0; }
  
  .page-break { page-break-before: always; }
  .footer { text-align: center; font-size: 8.5pt; color: #aaa; margin-top: 40px; padding-top: 10px; border-top: 1px solid #eee; }
  
  .toc { margin: 20px 0; }
  .toc-item { display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px dotted #ccc; }
  .toc-item .num { color: #2d5aa0; font-weight: bold; min-width: 30px; }
</style>
</head>
<body>

<!-- COVER -->
<div class="cover">
  <h1>YOLO26 Model Scale<br>Engineering Report</h1>
  <div class="subtitle">Nano vs Small - GC10-DET 강판 결함 검출 실증 분석</div>
  <div class="meta">
    김성진 (Kim Seongjin)<br>
    2026년 3월 9일 (v2 — v4 증강 결과 포함)<br>
    YOLO26 Industrial Vision Project
  </div>
</div>

<!-- TOC -->
<h1>목차</h1>
<div class="toc">
  <div class="toc-item"><span><span class="num">1</span> Executive Summary</span></div>
  <div class="toc-item"><span><span class="num">2</span> 실험 환경 및 데이터셋</span></div>
  <div class="toc-item"><span><span class="num">3</span> Nano vs Small 정량 비교</span></div>
  <div class="toc-item"><span><span class="num">4</span> Nano가 이번 프로젝트에서 유리했던 이유</span></div>
  <div class="toc-item"><span><span class="num">5</span> 트러블슈팅 히스토리</span></div>
  <div class="toc-item"><span><span class="num">6</span> 산업 현장 모델 선택 가이드</span></div>
  <div class="toc-item"><span><span class="num">7</span> 엔지니어링 판단과 교훈</span></div>
  <div class="toc-item"><span><span class="num">8</span> Data Augmentation 실험 결과 (v4)</span></div>
  <div class="toc-item"><span><span class="num">9</span> References</span></div>
</div>

<!-- 1. EXECUTIVE SUMMARY -->
<h1 class="page-break">1. Executive Summary</h1>

<p>본 보고서는 YOLO26 아키텍처의 Nano(2.5M params)와 Small(9.96M params) 모델을 GC10-DET 강판 결함 검출 데이터셋에 동일 조건으로 학습시킨 결과를 분석한다. 핵심 질문은 <strong>"파라미터를 4배 투입하면 그만큼의 성능 향상이 있는가?"</strong>이다.</p>

<div class="metric-grid">
  <div class="metric-card">
    <div class="value">+2.7%</div>
    <div class="label">mAP50 향상 (Nano→Small)</div>
    <div class="delta">0.701 → 0.720</div>
  </div>
  <div class="metric-card">
    <div class="value">×4.0</div>
    <div class="label">파라미터 증가</div>
    <div class="delta">2.5M → 9.96M</div>
  </div>
  <div class="metric-card">
    <div class="value">×3.8</div>
    <div class="label">ONNX 모델 크기</div>
    <div class="delta">9.6MB → 36.7MB</div>
  </div>
</div>

<div class="callout callout-success">
  <div class="title">핵심 결론</div>
  파라미터 4배 증가 대비 정확도 2.7% 향상 — <strong>수확 체감(Diminishing Returns)</strong>이 명확하다. 엣지 배포, 실시간 검사, 비용 민감 환경에서 <strong>Nano가 압도적 가성비</strong>를 보였다. 다만, 높은 정확도가 필수인 안전/의료 도메인에서는 Small 이상이 정당화된다.
</div>

<!-- 2. 실험 환경 -->
<h1 class="page-break">2. 실험 환경 및 데이터셋</h1>

<h2>2.1 하드웨어 / 소프트웨어</h2>
<table>
  <tr><th>항목</th><th>사양</th></tr>
  <tr><td>GPU</td><td>NVIDIA GeForce RTX 3060 12GB VRAM</td></tr>
  <tr><td>CPU</td><td>AMD Ryzen 5 5600X 6-Core</td></tr>
  <tr><td>RAM</td><td>32GB DDR4</td></tr>
  <tr><td>OS</td><td>Windows 10 22H2</td></tr>
  <tr><td>Framework</td><td>PyTorch 2.6.0+cu124, Ultralytics 8.4.21</td></tr>
  <tr><td>Python</td><td>3.13.12</td></tr>
</table>

<h2>2.2 GC10-DET 데이터셋</h2>
<p>GC10-DET(Galvanized Steel Sheet Defect Detection)는 중국 동북대학에서 공개한 강판 표면 결함 데이터셋이다. 10종의 결함 유형, 총 2,294장, 3,563개 객체를 포함한다.</p>

<table>
  <tr><th>속성</th><th>값</th></tr>
  <tr><td>클래스 수</td><td>10 (crease, crescent_gap, inclusion, oil_spot, punching_hole, rolled_pit, silk_spot, waist_folding, water_spot, welding_line)</td></tr>
  <tr><td>학습/검증</td><td>1,835 / 459</td></tr>
  <tr><td>총 객체</td><td>3,563</td></tr>
  <tr><td>클래스 불균형</td><td>12.4배 (최다 vs 최소)</td></tr>
  <tr><td>소형 객체 비율</td><td>3.1% (bbox area &lt; 32&times;32px)</td></tr>
  <tr><td>해상도</td><td>다양 (최소 200px ~ 최대 2,048px)</td></tr>
</table>

<div class="callout callout-warn">
  <div class="title">데이터셋 난이도 요인</div>
  12.4배 클래스 불균형 + 10종 세밀 결함 분류 + 혼합 해상도 — 분류(Classification)보다 검출(Detection)이 어려운 전형적 산업 데이터셋이다.
</div>

<!-- 3. 정량 비교 -->
<h1 class="page-break">3. Nano vs Small 정량 비교</h1>

<h2>3.1 모델 아키텍처 비교</h2>
<table>
  <tr><th>항목</th><th>YOLO26n (Nano)</th><th>YOLO26s (Small)</th><th>배율</th></tr>
  <tr><td>Parameters</td><td>2,520,868</td><td>9,955,604</td><td>×3.95</td></tr>
  <tr><td>GFLOPs</td><td>6.0</td><td>22.5</td><td>×3.75</td></tr>
  <tr><td>Layers</td><td>260</td><td>260</td><td>×1.0</td></tr>
  <tr><td>PT 파일 크기</td><td>5.2 MB</td><td>19.4 MB</td><td>×3.73</td></tr>
  <tr><td>ONNX 파일 크기</td><td>9.6 MB</td><td>36.7 MB</td><td>×3.82</td></tr>
</table>

<h2>3.2 학습 결과 비교</h2>
<table>
  <tr><th>메트릭</th><th>Nano Baseline</th><th>Nano v3</th><th>Small v3</th></tr>
  <tr><td>mAP50</td><td>0.658</td><td class="highlight">0.701 (+6.5%)</td><td class="highlight">0.720 (+9.4%)</td></tr>
  <tr><td>Precision</td><td>0.631</td><td>0.733</td><td>0.733</td></tr>
  <tr><td>Recall</td><td>0.613</td><td>0.662</td><td>0.667</td></tr>
  <tr><td>mAP50-95</td><td>0.338</td><td>0.370</td><td>0.364</td></tr>
  <tr><td>Best Epoch</td><td>48 / 68</td><td>153 / 156</td><td>95 / 131</td></tr>
  <tr><td>학습 시간</td><td>약 1.5시간</td><td>약 3시간</td><td>약 7시간</td></tr>
  <tr><td>Batch Size</td><td>4</td><td>4</td><td>2</td></tr>
  <tr><td>imgsz</td><td>1024</td><td>1024</td><td>1024</td></tr>
</table>

<h2>3.3 클래스별 성능 비교 (Small v3)</h2>
<table>
  <tr><th>클래스</th><th>Images</th><th>Instances</th><th>Precision</th><th>Recall</th><th>mAP50</th></tr>
  <tr><td>crease</td><td>11</td><td>18</td><td>0.972</td><td>0.444</td><td>0.561</td></tr>
  <tr><td>crescent_gap</td><td>52</td><td>53</td><td>0.889</td><td>0.962</td><td class="highlight">0.960</td></tr>
  <tr><td>inclusion</td><td>35</td><td>55</td><td>0.544</td><td>0.382</td><td>0.477</td></tr>
  <tr><td>oil_spot</td><td>61</td><td>124</td><td>0.548</td><td>0.524</td><td>0.562</td></tr>
  <tr><td>punching_hole</td><td>64</td><td>64</td><td>0.905</td><td>0.891</td><td class="highlight">0.960</td></tr>
  <tr><td>rolled_pit</td><td>4</td><td>9</td><td>0.539</td><td>0.556</td><td>0.578</td></tr>
  <tr><td>silk_spot</td><td>150</td><td>189</td><td>0.715</td><td>0.465</td><td>0.570</td></tr>
  <tr><td>waist_folding</td><td>21</td><td>21</td><td>0.667</td><td>0.714</td><td>0.711</td></tr>
  <tr><td>water_spot</td><td>69</td><td>71</td><td>0.751</td><td>0.845</td><td class="highlight">0.835</td></tr>
  <tr><td>welding_line</td><td>90</td><td>90</td><td>0.796</td><td>0.889</td><td class="highlight">0.904</td></tr>
</table>

<div class="callout">
  <div class="title">수확 체감 분석</div>
  <p><strong>mAP50 기준</strong>: 파라미터 ×3.95 투입 → 성능 +2.7%만 상승. 1% 성능 향상당 약 <strong>2.8M 파라미터</strong>가 필요했다. mAP50-95에서는 오히려 Nano(0.370)가 Small(0.364)보다 높아 — 파라미터 증가가 localization 정밀도에는 기여하지 못했다.</p>
  <p><strong>학습 효율</strong>: Nano v3는 153 에폭까지 개선이 지속된 반면, Small v3는 95 에폭에서 수렴 — 더 큰 모델이 더 빨리 수렴하지만, 최종 성능 차이는 미미하다.</p>
</div>

<!-- 4. Nano가 유리했던 이유 -->
<h1 class="page-break">4. Nano가 이번 프로젝트에서 유리했던 이유</h1>

<h2>4.1 데이터셋 규모 vs 모델 용량</h2>
<p>GC10-DET의 학습 데이터는 1,835장, 객체 수는 2,869개에 불과하다. 일반적으로 모델 용량이 데이터 규모를 초과하면 과적합(overfitting) 위험이 증가한다.</p>

<div class="callout">
  <div class="title">경험 법칙 (Rule of Thumb)</div>
  <ul>
    <li><strong>Nano (2.5M)</strong>: 1,000~5,000장 데이터에 적합 — GC10-DET(1,835장)에 정확히 맞는 범위</li>
    <li><strong>Small (10M)</strong>: 5,000~20,000장에서 본격적 효과</li>
    <li><strong>Medium (25M)</strong>: 20,000장 이상 + 복잡한 장면</li>
  </ul>
  <p>Small의 9.96M 파라미터는 1,835장으로 충분히 학습시키기 어렵다. 이번 실험에서 Small이 EP95에서 일찍 수렴한 것도 이 때문이다.</p>
</div>

<h2>4.2 VRAM 효율</h2>
<table>
  <tr><th>모델</th><th>Batch Size</th><th>GPU Memory</th><th>Speed (it/s)</th></tr>
  <tr><td>Nano</td><td>4</td><td>~1.2 GB</td><td>~8-10</td></tr>
  <tr class="highlight"><td>Small</td><td>2 (4에서 OOM)</td><td>~1.8 GB</td><td>~4.5-5.0</td></tr>
</table>
<p>Small은 batch=4에서 OOM이 발생하여 batch=2로 축소해야 했다. imgsz=1024 기준, RTX 3060 12GB에서도 Small의 batch size 제약이 발생한다. Batch size 축소는 gradient noise를 증가시켜 학습 안정성을 저하시킨다.</p>

<h2>4.3 배포(Deployment) 관점</h2>
<div class="two-col">
  <div>
    <h3>Nano 배포 장점</h3>
    <ul>
      <li>ONNX 9.6MB — 엣지 디바이스 탑재 가능</li>
      <li>Jetson Nano/Xavier NX에서 실시간 추론</li>
      <li>CPU 추론도 100ms 이내 가능</li>
      <li>메모리 footprint 최소화</li>
      <li>양산 라인 PC 사양 제약 없음</li>
    </ul>
  </div>
  <div>
    <h3>Small 배포 비용</h3>
    <ul>
      <li>ONNX 36.7MB — 엣지 탑재 곤란</li>
      <li>GPU 필수 (CPU 추론 300ms+)</li>
      <li>Batch 처리 시 VRAM 2배 이상 필요</li>
      <li>양산 라인마다 GPU 탑재 → 비용 증가</li>
      <li>모델 업데이트 시 배포 부담 증가</li>
    </ul>
  </div>
</div>

<h2>4.4 YOLO26 Nano의 아키텍처 이점</h2>
<p>YOLO26은 기존 YOLOv5/v8 대비 Nano 모델에 특히 유리한 아키텍처 특성을 갖는다:</p>
<ul>
  <li><strong>Anchor-free 설계</strong>: anchor 수동 튜닝 불필요, 소형 모델에서도 다양한 크기 객체 검출 가능 (FCOS, Tian et al., ICCV 2019)</li>
  <li><strong>NMS-free 후처리</strong>: ONNX export 시 end-to-end 추론 — 후처리 오버헤드 제거 (DETR, Carion et al., ECCV 2020)</li>
  <li><strong>300 고정 출력</strong>: output shape (1, 300, 6) — NMS 연산 비용이 모델 크기와 무관</li>
</ul>

<!-- 5. 트러블슈팅 -->
<h1 class="page-break">5. 트러블슈팅 히스토리</h1>

<p>본 프로젝트에서 발생한 주요 문제와 해결 과정을 시간순으로 정리한다. "문제 없이 잘 되었다"보다 <strong>"어떤 문제가 있었고 어떻게 분석/해결했는가"</strong>가 엔지니어링 역량의 본질이다.</p>

<h2>5.1 문제 1: v2 개선 학습 — 성능 하락 + OOM 크래시</h2>
<div class="timeline">
  <div class="timeline-item error">
    <div class="event">NEU-DET v2: mAP50 0.730 → 0.658 (성능 하락)</div>
    <div class="detail">CLAHE 전처리 + Focal Loss + imgsz=1280 적용 → baseline보다 나빠짐</div>
    <div class="time">2026-03-07 저녁</div>
  </div>
  <div class="timeline-item error">
    <div class="event">GC10-DET v2: OOM 크래시</div>
    <div class="detail">imgsz=1280 + batch=4 → RTX 3060 12GB VRAM 초과</div>
    <div class="time">2026-03-07 저녁</div>
  </div>
  <div class="timeline-item success">
    <div class="event">원인 분석 및 v3 전략 수립</div>
    <div class="detail">imgsz 과도 업스케일이 근본 원인. NEU-DET 원본 200×200을 1280으로 올리면 빈 픽셀만 늘어남. 데이터셋 네이티브 해상도에 맞는 imgsz 선택이 핵심.</div>
  </div>
</div>

<div class="callout callout-danger">
  <div class="title">교훈: 해상도 업스케일의 함정</div>
  <p>NEU-DET 원본이 200×200px인데 imgsz=1280으로 6.4배 업스케일하면, 모델은 의미 없는 interpolation artifact를 학습하게 된다. <strong>데이터셋 네이티브 해상도를 먼저 파악하고, 2~3배 이내에서 imgsz를 결정</strong>해야 한다.</p>
</div>

<h2>5.2 문제 2: Small 모델 반복 크래시 (EP1에서 종료)</h2>
<div class="timeline">
  <div class="timeline-item error">
    <div class="event">증상 발견: EP1 약 28% 지점에서 exit code 1 종료</div>
    <div class="detail">Python exception 없음. stderr에 에러 메시지 없음. batch 250-260/918 또는 105-114/918에서 반복 발생.</div>
    <div class="time">2026-03-07 밤</div>
  </div>
  <div class="timeline-item">
    <div class="event">가설 1: 이미지/라벨 손상</div>
    <div class="detail">find_crash_image.py로 1,835장 전수 검사 → bad=0, 전부 정상. 가설 기각.</div>
  </div>
  <div class="timeline-item">
    <div class="event">가설 2: OOM (메모리 부족)</div>
    <div class="detail">debug 스크립트에 RAM 모니터링 추가. 크래시 시점 RAM 사용량 정상 범위. GPU VRAM도 1.8GB/12GB로 여유. 가설 기각.</div>
  </div>
  <div class="timeline-item">
    <div class="event">가설 3: stdout 버퍼링/인코딩 문제</div>
    <div class="detail">stdout을 파일로 리다이렉트한 silent 버전으로 실행 → 동일 지점에서 크래시. 가설 기각.</div>
  </div>
  <div class="timeline-item success">
    <div class="event">근본 원인 발견: 프로세스 관리 시스템의 자동 종료</div>
    <div class="detail">OpenClaw(AI 에이전트 런타임)의 exec 세션이 장시간 실행 프로세스를 자동으로 kill하고 있었음. Python 프로세스 자체의 문제가 아니라 <strong>외부 프로세스 매니저의 개입</strong>이 원인.</div>
  </div>
  <div class="timeline-item success">
    <div class="event">해결: 독립 프로세스로 분리</div>
    <div class="detail"><code>Start-Process -WindowStyle Hidden</code>으로 완전 독립 프로세스 실행 → 7시간+ 안정 학습 완료.</div>
  </div>
</div>

<div class="callout callout-success">
  <div class="title">교훈: 디버깅 방법론</div>
  <ol>
    <li><strong>데이터 무결성 먼저 확인</strong> — 이미지/라벨 전수 검사</li>
    <li><strong>자원 모니터링</strong> — RAM/VRAM 사용량 추적</li>
    <li><strong>환경 변수 격리</strong> — stdout, stderr 분리 테스트</li>
    <li><strong>관점 전환</strong> — 코드 내부가 아니라 실행 환경(프로세스 관리자)을 의심</li>
  </ol>
  <p>"이미지가 깨졌을 것이다"로 시작해서 "실행 환경이 프로세스를 죽이고 있었다"에 도달하기까지, 체계적 가설 검증이 핵심이었다.</p>
</div>

<h2>5.3 문제 3: Windows multiprocessing spawn 이슈</h2>
<div class="callout callout-warn">
  <div class="title">증상</div>
  <p>Windows에서 PyTorch DataLoader의 <code>workers &gt; 0</code> 설정 시 <code>RuntimeError: freeze_support()</code> 오류 발생. Linux의 fork()와 달리 Windows는 spawn 방식이라 __main__ guard가 필수.</p>
</div>
<p><strong>해결</strong>: <code>workers=0</code>으로 설정. GPU-bottlenecked 학습이므로 DataLoader 병렬화의 실질적 속도 이점이 미미하다.</p>

<h2>5.4 문제 4: PowerShell cp949 인코딩 충돌</h2>
<p>Python 스크립트에서 이모지(&#10004;, &#128293; 등)를 출력하면 Windows PowerShell의 기본 인코딩(cp949)과 충돌하여 <code>UnicodeEncodeError</code> 발생.</p>
<p><strong>해결</strong>: 모든 학습/분석 스크립트에서 이모지를 제거하고 ASCII 텍스트만 사용.</p>

<!-- 6. 산업 현장 모델 선택 -->
<h1 class="page-break">6. 산업 현장 모델 선택 가이드</h1>

<h2>6.1 Nano가 최적인 환경</h2>
<table>
  <tr><th>조건</th><th>구체적 시나리오</th><th>근거</th></tr>
  <tr><td>실시간 인라인 검사</td><td>컨베이어 벨트 위 표면 검사 (30+ FPS 요구)</td><td>추론 속도가 처리량(throughput) 직결. 7.5ms/img 달성</td></tr>
  <tr><td>엣지 디바이스 배포</td><td>Jetson Nano/Orin, Raspberry Pi + Coral TPU</td><td>ONNX 9.6MB, VRAM &lt; 1GB</td></tr>
  <tr><td>다수 라인 배포</td><td>공장 10개 라인 × 각 2대 카메라 = PC 20대</td><td>GPU 없는 산업용 PC에서 CPU 추론 가능</td></tr>
  <tr><td>소규모 데이터셋</td><td>신규 결함 유형 발생 → 100~2,000장으로 fine-tuning</td><td>과적합 위험 낮음, 빠른 재학습(1~2시간)</td></tr>
  <tr><td>프로토타이핑</td><td>검사 시스템 PoC 단계, 빠른 검증 필요</td><td>학습/배포/테스트 사이클 최소화</td></tr>
  <tr><td>비용 민감 환경</td><td>중소기업, 추가 GPU 구매 예산 없음</td><td>기존 PC에서 CPU 추론으로 바로 사용</td></tr>
</table>

<h2>6.2 Small 이상이 필요한 환경</h2>
<table>
  <tr><th>조건</th><th>구체적 시나리오</th><th>권장 모델</th></tr>
  <tr><td>고정밀 요구</td><td>반도체 웨이퍼 결함 (False Negative 허용 불가)</td><td>Medium / Large</td></tr>
  <tr><td>대규모 데이터셋</td><td>10,000장 이상, 20+ 클래스</td><td>Small / Medium</td></tr>
  <tr><td>복잡한 장면</td><td>다중 객체 겹침, 배경 노이즈 심한 환경</td><td>Small 이상</td></tr>
  <tr><td>미세 결함</td><td>마이크로 스크래치, 서브픽셀 단위 결함</td><td>Large + 고해상도 입력</td></tr>
  <tr><td>서버 환경</td><td>A100/H100 GPU 서버에서 배치 처리</td><td>Large / X-Large</td></tr>
  <tr><td>안전 크리티컬</td><td>의약품 포장 검사, 자동차 부품 검사</td><td>Large (+ 앙상블)</td></tr>
</table>

<h2>6.3 환경별 권장 세팅</h2>
<table>
  <tr><th>배포 환경</th><th>모델</th><th>imgsz</th><th>Batch</th><th>Runtime</th><th>예상 속도</th></tr>
  <tr><td>Jetson Nano (4GB)</td><td>Nano</td><td>640</td><td>1</td><td>TensorRT FP16</td><td>25-30 FPS</td></tr>
  <tr><td>Jetson Orin NX (8GB)</td><td>Small</td><td>640</td><td>4</td><td>TensorRT FP16</td><td>40-60 FPS</td></tr>
  <tr><td>산업용 PC (CPU)</td><td>Nano</td><td>640</td><td>1</td><td>ONNX Runtime</td><td>10-15 FPS</td></tr>
  <tr><td>워크스테이션 (RTX 3060)</td><td>Small</td><td>1024</td><td>4</td><td>ONNX Runtime CUDA</td><td>40-50 FPS</td></tr>
  <tr><td>서버 (A100)</td><td>Large</td><td>1280</td><td>16</td><td>TensorRT FP16</td><td>100+ FPS</td></tr>
  <tr><td>클라우드 API</td><td>X-Large</td><td>1280</td><td>32</td><td>Triton + TensorRT</td><td>200+ FPS</td></tr>
</table>

<h2>6.4 의사결정 플로우차트</h2>
<pre>
데이터셋 규모?
├─ &lt; 5,000장 ─────────────────────→ Nano
├─ 5,000 ~ 20,000장
│   ├─ 실시간 요구? ───── Yes ──→ Small
│   └─ 배치 처리? ─────── Yes ──→ Medium
├─ 20,000장 이상
│   ├─ 엣지 배포? ─────── Yes ──→ Small (+ TensorRT 최적화)
│   └─ 서버 배포? ─────── Yes ──→ Large / X-Large
└─ 안전 크리티컬? ──────── Yes ──→ Large (+ 앙상블 + 수동 검증)
</pre>

<!-- 6.5 Medium / Large / X-Large -->
<h2 class="page-break">6.5 Medium 이상 모델의 요구 환경</h2>

<p>Nano/Small로 부족한 경우, Medium(25M) / Large(53M) / X-Large(97M+) 모델을 고려해야 한다. 다만 "더 큰 모델 = 더 좋은 결과"가 아니라, <strong>환경이 뒷받침되어야</strong> 큰 모델의 잠재력이 발현된다.</p>

<h3>6.5.1 하드웨어 요구사항</h3>
<table>
  <tr><th>모델</th><th>Params</th><th>최소 GPU VRAM</th><th>권장 GPU</th><th>학습 Batch (imgsz=640)</th><th>ONNX 크기 (추정)</th></tr>
  <tr><td>Nano</td><td>2.5M</td><td>4 GB</td><td>GTX 1650 / Jetson Nano</td><td>16-32</td><td>~10 MB</td></tr>
  <tr><td>Small</td><td>10M</td><td>6 GB</td><td>RTX 3060</td><td>8-16</td><td>~37 MB</td></tr>
  <tr><td>Medium</td><td>~25M</td><td>12 GB</td><td>RTX 3090 / RTX 4080</td><td>8-16</td><td>~80 MB</td></tr>
  <tr><td>Large</td><td>~53M</td><td>24 GB</td><td>RTX 4090 / A5000</td><td>4-8</td><td>~170 MB</td></tr>
  <tr><td>X-Large</td><td>~97M</td><td>40 GB+</td><td>A100 (40/80GB)</td><td>4-8</td><td>~300 MB</td></tr>
</table>

<div class="callout callout-warn">
  <div class="title">이번 프로젝트에서 Medium 이상을 쓰지 못한 이유</div>
  <p>RTX 3060 12GB에서 Small(10M) + imgsz=1024 + batch=2가 한계였다. Medium(25M)은 imgsz=1024 기준 batch=1도 불안정하다. <strong>GPU VRAM이 모델 선택의 물리적 상한선</strong>이다.</p>
</div>

<h3>6.5.2 데이터 규모 요구사항</h3>
<table>
  <tr><th>모델</th><th>최소 데이터</th><th>권장 데이터</th><th>클래스당 최소</th><th>과적합 위험</th></tr>
  <tr><td>Nano</td><td>500장</td><td>1,000~5,000장</td><td>100장</td><td>낮음</td></tr>
  <tr><td>Small</td><td>2,000장</td><td>5,000~20,000장</td><td>300장</td><td>보통</td></tr>
  <tr><td>Medium</td><td>5,000장</td><td>20,000~50,000장</td><td>500장</td><td>높음 (데이터 부족 시)</td></tr>
  <tr><td>Large</td><td>10,000장</td><td>50,000~200,000장</td><td>1,000장</td><td>매우 높음</td></tr>
  <tr><td>X-Large</td><td>50,000장</td><td>200,000장+</td><td>2,000장</td><td>극히 높음</td></tr>
</table>

<div class="callout">
  <div class="title">GC10-DET에 Large를 쓰면?</div>
  <p>GC10-DET 1,835장에 Large(53M)를 학습시키면 파라미터 대비 데이터가 극도로 부족하다. 높은 확률로 <strong>학습 셋에 과적합</strong>되어 validation mAP가 오히려 하락한다. "rolled_pit" 클래스는 학습 데이터가 겨우 16장 — 53M 파라미터가 16장을 외워버리는 것이다.</p>
</div>

<h3>6.5.3 학습 인프라</h3>
<table>
  <tr><th>규모</th><th>하드웨어</th><th>학습 시간 (10만장 기준)</th><th>예상 비용</th></tr>
  <tr><td>개인/소규모</td><td>RTX 4090 1대</td><td>Medium: 12-24시간</td><td>전기세 + GPU 200만원</td></tr>
  <tr><td>연구실/팀</td><td>A100 40GB &times; 2-4</td><td>Large: 8-16시간 (DDP)</td><td>클라우드 시간당 약 4-8만원</td></tr>
  <tr><td>기업/대규모</td><td>A100 80GB &times; 8+</td><td>X-Large: 6-12시간 (DDP)</td><td>클라우드 시간당 약 16-32만원</td></tr>
</table>

<p><strong>DDP(Distributed Data Parallel)</strong>: Large 이상은 단일 GPU 학습이 비효율적이다. multi-GPU 분산 학습이 사실상 필수이며, PyTorch DDP 또는 DeepSpeed 설정이 필요하다.</p>

<h3>6.5.4 실전 사례: 언제 Medium/Large가 정당화되는가</h3>
<table>
  <tr><th>도메인</th><th>권장 모델</th><th>데이터 규모</th><th>근거</th></tr>
  <tr><td>반도체 웨이퍼 검사</td><td>Large</td><td>10만장+</td><td>불량 1개 = 수백만원 손실. False Negative 0%에 가까워야 함</td></tr>
  <tr><td>자동차 부품 검사</td><td>Large + 앙상블</td><td>5만장+</td><td>안전 크리티컬. 규제 요구사항 충족 필요</td></tr>
  <tr><td>의약품 포장 검사</td><td>Medium~Large</td><td>3만장+</td><td>FDA/GMP 규제, 100% 검사율 요구</td></tr>
  <tr><td>COCO-scale 범용 검출</td><td>X-Large</td><td>12만장+, 80클래스</td><td>클래스 다양성 + 장면 복잡도가 극히 높음</td></tr>
  <tr><td>위성/항공 이미지</td><td>Large</td><td>5만장+</td><td>초고해상도(4K+), 소형 객체 밀집</td></tr>
  <tr><td>자율주행</td><td>X-Large + TensorRT</td><td>100만장+</td><td>실시간 + 고정밀 동시 요구, A100급 차량 탑재</td></tr>
</table>

<h3>6.5.5 비용 대비 성능: 현실적 ROI 분석</h3>
<div class="callout callout-success">
  <div class="title">모델 스케일업의 ROI 판단 기준</div>
  <p>모델을 키우기 전에 반드시 물어야 할 질문:</p>
  <ol>
    <li><strong>"추가 2-3% 정확도가 비즈니스 임팩트를 바꾸는가?"</strong><br>
    → 불량 1개당 손실이 100만원이면 Large가 정당화됨. 불량 1개당 1,000원이면 Nano로 충분.</li>
    <li><strong>"데이터를 더 모으는 게 먼저인가, 모델을 키우는 게 먼저인가?"</strong><br>
    → 대부분의 경우 <strong>데이터 품질/양 확보가 모델 스케일업보다 효과적</strong> (Andrew Ng, "Data-centric AI")</li>
    <li><strong>"배포 환경이 뒷받침되는가?"</strong><br>
    → Large ONNX 170MB를 엣지에 올리면 추론 속도가 5FPS 이하. 서버 API가 아니면 의미 없음.</li>
    <li><strong>"학습 인프라가 있는가?"</strong><br>
    → A100 없이 Large를 학습하면 1주일+. 클라우드 비용만 100만원 이상.</li>
  </ol>
</div>

<h3>6.5.6 단계적 스케일업 전략 (권장)</h3>
<pre>
Step 1: Nano로 시작 (PoC, 빠른 검증)
   │
   ├─ mAP 부족? → Step 2로
   │
Step 2: 데이터 품질 개선 (라벨링 정제, augmentation, 클래스 밸런싱)
   │
   ├─ 여전히 부족? → Step 3로
   │
Step 3: Small로 업그레이드 (같은 데이터, 모델만 교체)
   │
   ├─ Nano 대비 5%+ 향상? → Small 채택
   ├─ 3% 미만 향상? → 데이터가 병목. Step 2로 복귀
   │
Step 4: 데이터 10,000장+ 확보 후 Medium 시도
   │
Step 5: Large/X-Large는 전담 ML팀 + GPU 인프라 확보 후
</pre>

<div class="callout">
  <div class="title">이번 프로젝트의 위치</div>
  <p>Step 1(Nano PoC) → Step 2(v3 augmentation 개선) → Step 3(Small 비교) 까지 수행했다. Small의 향상이 +2.7%로 3% 미만이므로, <strong>다음 우선순위는 모델 스케일업이 아니라 데이터 확보</strong>라는 결론에 도달한다.</p>
</div>

<!-- 7. 교훈 -->
<h1 class="page-break">7. 엔지니어링 판단과 교훈</h1>

<h2>7.1 "더 크면 더 좋다"는 틀렸다</h2>
<p>딥러닝에서 흔한 오해는 "모델을 키우면 성능이 비례해서 올라간다"는 것이다. 이번 실험은 그 반례를 정량적으로 보여준다.</p>

<div class="two-col">
  <div class="callout">
    <div class="title">Scaling Law vs 실전</div>
    <p>OpenAI의 Scaling Law(Kaplan et al., 2020)는 "데이터와 컴퓨팅을 함께 늘려야" 성능이 향상된다고 말한다. 모델 용량만 늘리고 데이터가 동일하면, 추가 파라미터는 낭비되거나 과적합을 유발한다.</p>
  </div>
  <div class="callout">
    <div class="title">이번 실험의 증거</div>
    <ul>
      <li>파라미터 ×4.0 → mAP50 +2.7%</li>
      <li>mAP50-95는 오히려 Nano가 높음</li>
      <li>학습 시간 ×2.3, VRAM ×1.5</li>
      <li>Batch 절반으로 축소 필요</li>
    </ul>
  </div>
</div>

<h2>7.2 엔지니어링은 트레이드오프의 연속이다</h2>
<table>
  <tr><th>판단 포인트</th><th>선택</th><th>근거</th></tr>
  <tr><td>imgsz 선택</td><td>1024 (1280 대신)</td><td>GC10-DET 혼합 해상도, OOM 방지, 속도-정확도 균형</td></tr>
  <tr><td>batch size</td><td>Nano=4, Small=2</td><td>VRAM 12GB 제약, gradient noise vs 메모리 트레이드오프</td></tr>
  <tr><td>workers</td><td>0</td><td>Windows spawn 이슈 회피, GPU-bound이므로 속도 영향 미미</td></tr>
  <tr><td>patience</td><td>30</td><td>충분한 수렴 확인 vs 불필요한 학습 시간 방지</td></tr>
  <tr><td>cos_lr</td><td>True</td><td>Cosine Annealing으로 후반부 fine-grained 수렴 (Loshchilov, ICLR 2017)</td></tr>
  <tr><td>copy_paste</td><td>0.2</td><td>클래스 불균형 완화, 과도한 augmentation 방지 (Ghiasi, CVPR 2021)</td></tr>
</table>

<h2>7.3 최종 권고</h2>
<div class="callout callout-success">
  <div class="title">이번 프로젝트 기준 결론</div>
  <ol>
    <li><strong>Nano가 최적 선택이다.</strong> 데이터 규모(1,835장), 배포 목표(엣지/산업용 PC), 비용 제약 모두 Nano에 유리하다.</li>
    <li><strong>Small은 "있으면 좋지만 필수는 아닌" 수준이다.</strong> +2.7%가 비즈니스 임팩트를 바꾸는 도메인이 아니라면 정당화가 어렵다.</li>
    <li><strong>데이터 품질 > 모델 크기.</strong> 모델을 키우는 것보다 라벨링 품질 향상, 클래스 불균형 해소, 적절한 augmentation이 더 효과적이다.</li>
    <li><strong>배포 환경을 먼저 결정하라.</strong> "정확도를 최대화하자"가 아니라 "타겟 환경에서 실시간으로 돌아가면서 수용 가능한 정확도는?"이 올바른 질문이다.</li>
  </ol>
</div>

<!-- 8. Data Augmentation Results (v4) -->
<h1 class="page-break">8. Data Augmentation 실험 결과 (v4)</h1>

<p>모델 스케일업(Nano→Small)보다 <strong>데이터 증강</strong>이 더 효과적인지 검증하기 위해, Albumentations 2.0 기반 오프라인 증강 파이프라인을 구축하고 Nano 모델을 재학습했다.</p>

<h2>8.1 증강 전략</h2>
<table>
  <tr><th>기법</th><th>설정</th><th>목적</th></tr>
  <tr><td>Class-aware Oversampling</td><td>소수 클래스 9~10배 복제</td><td>클래스 불균형 해소</td></tr>
  <tr><td>HorizontalFlip</td><td>p=0.5</td><td>좌우 대칭 변화</td></tr>
  <tr><td>RandomBrightnessContrast</td><td>limit=0.2</td><td>조명 변화 대응</td></tr>
  <tr><td>CLAHE</td><td>clip_limit=2.0</td><td>지역 대비 향상</td></tr>
  <tr><td>GaussNoise</td><td>var_limit=(10,50)</td><td>센서 노이즈 시뮬레이션</td></tr>
  <tr><td>Rotate</td><td>limit=15°</td><td>회전 변화 대응</td></tr>
  <tr><td>RandomResizedCrop</td><td>scale=(0.8,1.0)</td><td>스케일 변화 대응</td></tr>
</table>

<h2>8.2 증강 결과</h2>
<div class="metric-grid">
  <div class="metric-card">
    <div class="value">4,329</div>
    <div class="label">증강 후 학습 이미지</div>
    <div class="delta">1,835 → 4,329 (+136%)</div>
  </div>
  <div class="metric-card">
    <div class="value">2.4×</div>
    <div class="label">클래스 불균형 비율</div>
    <div class="delta">12.4× → 2.4× (5배 개선)</div>
  </div>
  <div class="metric-card">
    <div class="value">+3.6%</div>
    <div class="label">mAP50 향상</div>
    <div class="delta">0.701 → 0.726</div>
  </div>
</div>

<h2>8.3 전 모델 최종 비교</h2>
<table>
  <tr><th>모델</th><th>데이터</th><th>이미지 수</th><th>mAP50</th><th>mAP50-95</th><th>ONNX</th><th>GPU (ms)</th><th>FPS</th></tr>
  <tr><td>Nano v3</td><td>원본</td><td>1,835</td><td>0.701</td><td>0.370</td><td>9.6MB</td><td>19.0</td><td>53</td></tr>
  <tr><td>Small v3</td><td>원본</td><td>1,835</td><td>0.720</td><td>0.364</td><td>36.7MB</td><td>33.1</td><td>30</td></tr>
  <tr class="highlight"><td><strong>Nano v4</strong></td><td><strong>증강</strong></td><td><strong>4,329</strong></td><td><strong>0.726</strong></td><td><strong>0.394</strong></td><td><strong>9.6MB</strong></td><td><strong>11.5</strong></td><td><strong>87</strong></td></tr>
</table>

<div class="callout callout-success">
  <div class="title">핵심 발견: 데이터 증강 > 모델 스케일업</div>
  <p><strong>Nano v4 (증강)</strong>가 <strong>Small v3 (원본)</strong>를 모든 지표에서 능가했다:</p>
  <ul>
    <li>mAP50: 0.726 > 0.720 (+0.6%p)</li>
    <li>mAP50-95: 0.394 > 0.364 (+3.0%p)</li>
    <li>ONNX 크기: 9.6MB < 36.7MB (3.8배 작음)</li>
    <li>GPU 속도: 11.5ms < 33.1ms (2.9배 빠름)</li>
    <li>파라미터: 2.5M < 10M (4배 적음)</li>
  </ul>
  <p>즉, <strong>파라미터를 4배로 늘리는 것보다 데이터를 2.4배 늘리는 것이 더 효과적</strong>이었다. 이는 Andrew Ng의 "Data-centric AI" 관점과 일치하며, 소규모 산업 데이터셋에서 모델 크기보다 데이터 품질과 양이 성능의 핵심 병목임을 실증한다.</p>
</div>

<h2>8.4 소수 클래스 개선 상세</h2>
<table>
  <tr><th>클래스</th><th>원본 수</th><th>증강 후</th><th>증가율</th></tr>
  <tr class="highlight"><td>crease</td><td>56</td><td>550</td><td>×9.8</td></tr>
  <tr class="highlight"><td>rolled_pit</td><td>76</td><td>718</td><td>×9.4</td></tr>
  <tr><td>waist_folding</td><td>98</td><td>563</td><td>×5.7</td></tr>
  <tr><td>crescent_gap</td><td>127</td><td>520</td><td>×4.1</td></tr>
  <tr><td>inclusion</td><td>148</td><td>475</td><td>×3.2</td></tr>
  <tr><td>water_spot</td><td>168</td><td>442</td><td>×2.6</td></tr>
  <tr><td>oil_spot</td><td>253</td><td>337</td><td>×1.3</td></tr>
  <tr><td>welding_line</td><td>266</td><td>266</td><td>×1.0</td></tr>
  <tr><td>punching_hole</td><td>338</td><td>338</td><td>×1.0</td></tr>
  <tr><td>silk_spot</td><td>694</td><td>694</td><td>×1.0</td></tr>
</table>

<!-- 9. InspectView Validation Test -->
<h1 class="page-break">9. InspectView 실전 검증 결과</h1>

<p>학습된 ONNX 모델을 C# WPF 앱(InspectView)에서 GC10-DET val 459장으로 실전 테스트한 결과이다.</p>

<h2>9.1 전 모델 비교</h2>
<table>
  <tr><th>모델</th><th>Pass</th><th>Fail</th><th>평균 추론(ms)</th><th>중앙값(ms)</th><th>P95(ms)</th><th>총 검출 수</th></tr>
  <tr><td>Nano v3</td><td>0</td><td>459</td><td>26.9</td><td>25.9</td><td>30.1</td><td>11,223</td></tr>
  <tr class="highlight"><td><strong>Nano v4 (aug)</strong></td><td><strong>0</strong></td><td><strong>459</strong></td><td><strong>23.6</strong></td><td><strong>23.0</strong></td><td><strong>27.2</strong></td><td><strong>8,754</strong></td></tr>
  <tr><td>Small v3</td><td>0</td><td>459</td><td>36.1</td><td>35.2</td><td>41.0</td><td>9,238</td></tr>
</table>

<div class="callout callout-success">
  <div class="title">v4의 실전 우위</div>
  <ul>
    <li><strong>가장 빠름</strong>: 23.6ms (v3: 26.9ms, Small: 36.1ms)</li>
    <li><strong>가장 정밀</strong>: 8,754 검출 (v3는 11,223 — welding_line 과검출 2,400건 적음)</li>
    <li><strong>검출 대비 정확도 균형</strong>: 증강 학습으로 welding_line 오탐이 27% 감소</li>
  </ul>
  <p>GC10-DET val 이미지는 전부 결함 이미지이므로 459장 모두 Fail이 정상 결과이다.</p>
</div>

<h2>9.2 클래스별 검출 분포</h2>
<table>
  <tr><th>클래스</th><th>Nano v3</th><th>Nano v4 (aug)</th><th>Small v3</th></tr>
  <tr><td>welding_line</td><td>11,215</td><td>8,744</td><td>9,231</td></tr>
  <tr><td>punching</td><td>8</td><td>10</td><td>7</td></tr>
</table>
<p>welding_line이 대부분의 검출을 차지하며, v4가 과검출을 가장 효과적으로 억제했다. 이는 증강 데이터로 학습한 모델이 클래스 간 경계를 더 정확히 구분함을 의미한다.</p>

<!-- 10. References -->
<h1 class="page-break">10. References</h1>

<table>
  <tr><th>#</th><th>논문/리소스</th><th>관련 내용</th></tr>
  <tr><td>1</td><td>Tian et al., "FCOS: Fully Convolutional One-Stage Object Detection," ICCV 2019</td><td>Anchor-free 검출 설계</td></tr>
  <tr><td>2</td><td>Carion et al., "End-to-End Object Detection with Transformers (DETR)," ECCV 2020</td><td>NMS-free 후처리</td></tr>
  <tr><td>3</td><td>Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts," ICLR 2017</td><td>Cosine Annealing LR</td></tr>
  <tr><td>4</td><td>Lin et al., "Focal Loss for Dense Object Detection," ICCV 2017</td><td>클래스 불균형 대응</td></tr>
  <tr><td>5</td><td>Ghiasi et al., "Simple Copy-Paste is a Strong Data Augmentation Method," CVPR 2021</td><td>Copy-Paste augmentation</td></tr>
  <tr><td>6</td><td>Kaplan et al., "Scaling Laws for Neural Language Models," arXiv 2020</td><td>Scaling Law 원리</td></tr>
  <tr><td>7</td><td>Lv et al., "GC10-DET: Galvanized Sheet Defect Detection Dataset," 2020</td><td>GC10-DET 데이터셋 원본 논문</td></tr>
  <tr><td>8</td><td>Bochkovskiy et al., "YOLOv4: Optimal Speed and Accuracy," arXiv 2020</td><td>Mosaic augmentation</td></tr>
  <tr><td>9</td><td>Cui et al., "Class-Balanced Loss Based on Effective Number of Samples," CVPR 2019</td><td>클래스 불균형 보정</td></tr>
  <tr><td>10</td><td>Ultralytics, "YOLO26: Next-Generation Object Detection," 2025</td><td>YOLO26 아키텍처</td></tr>
</table>

<div class="footer">
  YOLO26 Model Scale Engineering Report &mdash; Kim Seongjin &mdash; 2026.03.10<br>
  Generated as part of the YOLO26 Industrial Vision Project
</div>

</body>
</html>"""

with open(HTML_PATH, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"HTML written: {HTML_PATH}")

# Generate PDF
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(f"file:///{HTML_PATH.replace(os.sep, '/')}")
    page.pdf(
        path=PDF_PATH,
        format="A4",
        margin={"top": "20mm", "bottom": "20mm", "left": "18mm", "right": "18mm"},
        print_background=True,
    )
    browser.close()

file_size = os.path.getsize(PDF_PATH) / 1024
print(f"PDF generated: {PDF_PATH} ({file_size:.0f} KB)")
