# 프로젝트 계획서 v2
# YOLO26 산업용 결함 검출 — End-to-End ONNX C# 배포 파이프라인

> "YOLOv5에서 겪었던 anchor 설정, NMS 구현, 소형 객체 문제를
> YOLO26이 어떻게 해결하는지 3개 산업 데이터셋으로 실증하고,
> NMS-free ONNX를 C# WPF 앱으로 배포하는 End-to-End 프로젝트"

---

## 왜 이 프로젝트인가? (스토리)

성진이(김성진)는 웨이퍼 엣지 결함 검출 R&D에서 YOLOv5를 직접 사용했다.
그때 겪었던 실무 문제:

1. **Anchor 설정** — k-means로 anchor 크기를 데이터셋마다 재계산
2. **NMS 구현** — C# 배포 시 NMS 후처리 코드를 직접 작성
3. **소형 객체** — P2 헤드를 수동으로 추가해야 작은 결함이 잡힘
4. **배포 복잡도** — ONNX export 후에도 전처리/후처리 코드가 많음

→ YOLO26은 이 4가지를 **아키텍처 수준에서 해결**했다.
→ 이걸 **실제 산업 데이터**로 증명하고 **C# 앱으로 배포**까지 보여주는 프로젝트.

---

## 데이터셋 (3개 — 다른 산업 도메인, 저명한 출처)

### 1. NEU-DET — 강철 표면 결함
| 항목 | 내용 |
|------|------|
| 출처 | Northeastern University (동북대학교, 중국) |
| 논문 | Song & Yan, "A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects" (Applied Surface Science, 2013) |
| 규모 | 1,800장 (200×200), 6 classes × 300장 |
| 클래스 | crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches |
| 인용 | 산업 결함 검출 분야 **표준 벤치마크** — Google Scholar 1,500+ 인용 |
| 특징 | 열연 강판 표면, 클래스당 균등 분포 |

### 2. DeepPCB — PCB 결함 검출
| 항목 | 내용 |
|------|------|
| 출처 | Peking University (북경대학교) |
| 논문 | Tang et al., "Online PCB Defect Detector on a New PCB Defect Dataset" (arXiv, 2019) |
| 규모 | 1,500 이미지쌍 (640×640), 6 classes |
| 클래스 | open, short, mousebite, spur, pinhole, spurious_copper |
| 인용 | PCB 결함 검출 분야 대표 벤치마크, GitHub 700+ stars |
| 특징 | 48px/mm 해상도, **결함 크기가 매우 작음** → 소형 객체 검출 핵심 테스트 |
| 포맷 | `x1,y1,x2,y2,type` bbox annotation 제공 → YOLO 변환 용이 |

### 3. Severstal Steel Defect Detection — 대규모 제철소 데이터
| 항목 | 내용 |
|------|------|
| 출처 | Severstal (세베르스탈, 러시아 최대 철강기업) |
| 플랫폼 | **Kaggle Competition** (2019), 1,136팀 참가 |
| 규모 | 12,568장 (256×1600), 4 classes |
| 클래스 | ClassId 1~4 (표면 결함 유형) |
| 인용 | Kaggle **금메달 대회**, 산업 AI 분야 가장 유명한 벤치마크 중 하나 |
| 특징 | **실제 제철소 생산라인 데이터**, RLE segmentation mask → bbox 변환 필요 |
| 데이터 | `kaggle competitions download severstal-steel-defect-detection` |

### 왜 이 3개?

```
NEU-DET     — 학술 표준 (논문 인용 최다)
DeepPCB     — 소형 객체 (YOLO26 ProgLoss+STAL 장점 실증)
Severstal   — 실전 대규모 (Kaggle 대회, 실제 공장 데이터)
```

3개 도메인이 겹치지 않음: **강철 표면 / PCB 기판 / 제철 생산라인**

---

## Phase 1: 데이터 + 학습 (Python)

### 1-1. 데이터 준비
| 데이터셋 | 다운로드 | 변환 | 예상 시간 |
|---------|---------|------|----------|
| NEU-DET | ✅ 완료 | ✅ YOLO 변환 완료 | — |
| DeepPCB | GitHub clone | `x1,y1,x2,y2,type` → YOLO txt | 20분 |
| Severstal | Kaggle API | RLE mask → bbox → YOLO txt | 40분 |

### 1-2. 학습
| 모델 | 데이터셋 | Epochs | imgsz | 예상 시간 |
|------|---------|--------|-------|----------|
| YOLO26n | NEU-DET | 80 | 640 | ✅ 완료 (mAP50=0.730) |
| YOLO26n | DeepPCB | 100 | 640 | ~40분 |
| YOLO26n | Severstal | 80 | 640 | ~1.5시간 (데이터 많음) |

### 1-3. 학습 참고사항
- pretrained: COCO (`yolo26n.pt`)
- augmentation: mosaic, mixup=0.1, flipud=0.5
- optimizer: auto (AdamW)
- patience: 20 (early stopping)
- cos_lr: True

---

## Phase 2: ONNX Export + 벤치마크

### 2-1. ONNX Export
```python
from ultralytics import YOLO
model = YOLO("best.pt")
model.export(format="onnx", simplify=True, opset=17)
```

**YOLO26 ONNX의 핵심:**
- NMS-free → ONNX 모델 output이 **바로 최종 detection 결과**
- YOLOv5/v8은 ONNX output이 raw boxes → 별도 NMS 필요

### 2-2. 속도 벤치마크 (ONNX Runtime)
| 측정 항목 | 방법 |
|----------|------|
| GPU 추론 (ms) | OnnxRuntime CUDA EP, 100회 평균 |
| CPU 추론 (ms) | OnnxRuntime CPU EP, 50회 평균 |
| End-to-End (ms) | 이미지 로드 → 전처리 → 추론 → 결과 |
| 모델 크기 (MB) | .onnx 파일 |

### 2-3. YOLOv5 vs YOLO26 ONNX 비교
**성진이 경험과 직접 연결:**
```
[YOLOv5 시절 — 성진이가 직접 겪은 것]
ONNX 추론 → Raw Boxes (8400×85) → NMS 직접 구현(C#)
                                    ↑ IoU threshold, conf threshold,
                                      class-specific NMS, max detections...
                                      버그 가능성, 속도 저하

[YOLO26 — 지금]
ONNX 추론 → 최종 Detections (N×6: x1,y1,x2,y2,conf,class)
             ↑ 끝. C# 코드 10줄이면 됨.
```

---

## Phase 3: C# WPF 앱 — "InspectView"

### 3-1. 앱 컨셉
**산업용 결함 검출 데스크톱 도구**
- YOLO26 ONNX 모델로 이미지 결함 검출
- NMS-free 덕분에 추론 코드가 극도로 단순
- 3개 산업 도메인 모델 선택 가능

### 3-2. 기능
| 기능 | 설명 |
|------|------|
| **이미지 검사** | 드래그&드롭 → 결함 검출 → BBox 오버레이 |
| **모델 전환** | Steel / PCB / Severstal 모델 선택 |
| **결과 패널** | 결함 목록, 신뢰도, 추론 시간(ms) |
| **배치 검사** | 폴더 일괄 검사 → CSV 리포트 |
| **Pass/Fail** | 결함 수/신뢰도 기준 합격/불합격 판정 |
| **성능 모니터** | GPU/CPU 추론 시간, FPS, 메모리 사용량 |

### 3-3. 기술 스택
- .NET 8, WPF, MVVM (CommunityToolkit.Mvvm)
- Microsoft.ML.OnnxRuntime.Gpu
- OpenCvSharp4 (이미지 전처리)
- LiveCharts2 (결과 차트)

### 3-4. 핵심 코드 — NMS-free ONNX 추론 (간결함이 포인트)
```csharp
// YOLO26 ONNX 추론 — NMS 코드가 없다!
using var session = new InferenceSession("yolo26n_steel.onnx");
var input = PreprocessImage(image, 640, 640);
var results = session.Run(new[] { NamedOnnxValue.CreateFromTensor("images", input) });
var detections = ParseDetections(results);  // 바로 최종 결과
// ↑ YOLOv5는 여기서 NMS 100줄이 더 필요했음
```

### 3-5. UI 레이아웃
```
┌─────────────────────────────────────────────────┐
│  InspectView — YOLO26 Industrial Defect Inspector │
├───────────┬─────────────────────────────────────┤
│           │                                     │
│ [Steel]   │    ┌─────────────────────────┐      │
│ [PCB]     │    │  검출 결과 이미지         │      │
│ [Severstal│    │  BBox + Label 오버레이    │      │
│           │    └─────────────────────────┘      │
│ ────────  │                                     │
│ Conf: 0.5 ├─────────────────┬───────────────────┤
│ [검사]    │ 결함 목록        │ 추론 정보          │
│ [배치]    │ ☐ Scratch  0.92 │ Model: YOLO26n    │
│ [Export]  │ ☐ Crack    0.87 │ Time:  6.2ms      │
│           │ ☐ Pit      0.73 │ ONNX:  9.4MB      │
│ ────────  │                 │ NMS:   불필요 ✓    │
│ PASS ✓    │ Total: 3 defects│ FPS:   161        │
└───────────┴─────────────────┴───────────────────┘
```

---

## Phase 4: GitHub 레포 구조

```
yolo26-industrial-vision/
│
├── README.md                        # 프로젝트 소개 (한/영)
├── README_EN.md                     # English version
├── LICENSE                          # MIT
├── .gitignore
│
├── training/                        # [Python] 학습 파이프라인
│   ├── requirements.txt
│   ├── scripts/
│   │   ├── download_datasets.py     # 데이터셋 자동 다운로드
│   │   ├── convert_severstal.py     # RLE mask → YOLO bbox
│   │   ├── convert_deeppcb.py       # DeepPCB → YOLO format
│   │   ├── convert_neudet.py        # VOC XML → YOLO
│   │   ├── train.py                 # 통합 학습 스크립트
│   │   └── evaluate.py              # 평가 + 비교표 생성
│   ├── configs/                     # data.yaml 파일들
│   └── results/                     # 학습 결과 (metrics, plots, CSV)
│       ├── comparison.json          # 전체 비교 데이터
│       └── benchmark.json           # ONNX 속도 벤치마크
│
├── export/                          # [Python] ONNX 변환
│   ├── export_onnx.py               # PT → ONNX
│   ├── benchmark_onnx.py            # ONNX Runtime 속도 측정
│   └── models/                      # .onnx 파일 (Git LFS)
│       ├── yolo26n_steel.onnx
│       ├── yolo26n_pcb.onnx
│       └── yolo26n_severstal.onnx
│
├── app/                             # [C#] WPF 데스크톱 앱
│   ├── InspectView.sln
│   └── InspectView/
│       ├── Core/                    # ONNX 추론 엔진
│       │   ├── Yolo26Detector.cs    # NMS-free 추론 (핵심!)
│       │   └── ImagePreprocessor.cs
│       ├── ViewModels/              # MVVM
│       ├── Views/                   # XAML
│       ├── Services/                # CSV 내보내기, 배치 검사
│       └── Assets/                  # ONNX 모델
│
├── demo/                            # [Python] Gradio 웹 데모
│   └── app.py
│
└── docs/
    ├── results.md                   # 전체 실험 결과
    ├── yolov5_vs_yolo26.md          # YOLOv5→YOLO26 진화 비교
    ├── deployment_guide.md          # C# ONNX 배포 가이드
    └── images/                      # 스크린샷, 결과 이미지
```

---

## Phase 5: README + 영상

### README 구성
1. **배너** — InspectView 앱 스크린샷 + 결함 검출 결과 이미지
2. **한 줄 소개** — "YOLO26 → ONNX → C# WPF End-to-End 산업 결함 검출"
3. **YOLOv5 vs YOLO26** — 실무 관점 비교 (성진이 경험 기반)
4. **실험 결과표** — 3개 데이터셋 × mAP, 속도, 크기
5. **아키텍처 다이어그램** — 학습→ONNX→C# 플로우
6. **Quick Start** — Python 학습 / C# 앱 빌드 방법
7. **데이터셋 레퍼런스** — 논문/대회 출처 명시

### 영상 (성진이 나레이션, ~3분)
| 구간 | 내용 | 시간 |
|------|------|------|
| 인트로 | "YOLOv5로 산업 비전을 했었는데..." | 20초 |
| 문제 제기 | anchor, NMS, 소형 객체 문제 설명 | 30초 |
| YOLO26 소개 | NMS-free, ProgLoss 등 핵심 개선점 | 30초 |
| 학습 결과 | 3개 데이터셋 mAP 비교표 | 30초 |
| C# 앱 데모 | InspectView로 이미지 검사 실제 시연 | 40초 |
| ONNX 비교 | YOLOv5 NMS 코드 vs YOLO26 코드 간결함 | 20초 |
| 아웃트로 | GitHub 링크, 기술 스택 요약 | 10초 |

---

## 타임라인

| # | Phase | 작업 | 예상 시간 | 비고 |
|---|-------|------|----------|------|
| 1 | 데이터 | DeepPCB 다운 + YOLO 변환 | 20분 | GitHub clone |
| 2 | 데이터 | Severstal 다운 + RLE→bbox 변환 | 40분 | Kaggle API |
| 3 | 학습 | YOLO26n × DeepPCB (100 epochs) | 40분 | RTX 3060 |
| 4 | 학습 | YOLO26n × Severstal (80 epochs) | 1.5시간 | 12K 이미지 |
| 5 | ONNX | 3개 모델 export + 벤치마크 | 30분 | |
| 6 | C# 앱 | InspectView WPF 앱 개발 | 3~4시간 | MVVM |
| 7 | GitHub | 레포 구조 정리 + push | 30분 | |
| 8 | README | 한/영 README 작성 | 1시간 | |
| 9 | 문서 | 비교표, 배포 가이드 | 30분 | |
| 10 | 영상 | 스크립트 + 편집 | 1시간 | 성진이 녹음 |
| | **합계** | | **~9시간** | |

---

## 레퍼런스

### 데이터셋
1. **NEU Surface Defect Database**
   - Song, K. & Yan, Y. (2013). "A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects." Applied Surface Science, 285, 858-864.
   - http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html

2. **DeepPCB**
   - Tang, S. et al. (2019). "Online PCB Defect Detector On A New PCB Defect Dataset."
   - https://github.com/tangsanli5201/DeepPCB

3. **Severstal: Steel Defect Detection**
   - Kaggle Competition (2019), 1,136 teams, hosted by Severstal
   - https://www.kaggle.com/c/severstal-steel-defect-detection

### 기술
4. **YOLO26** — Ultralytics (2026). NMS-free end-to-end detection.
   - https://docs.ultralytics.com/models/yolo26/

5. **Surface Defect Detection: Dataset & Papers**
   - Charmve (GitHub, 3,500+ stars) — 산업 결함 데이터셋/논문 종합
   - https://github.com/Charmve/Surface-Defect-Detection

### 성진이 선행 경험
6. **웨이퍼 엣지 결함 검출 R&D** — YOLOv5 + OpenCV, Precision 86% / Recall 81%
   - KIPS 학술발표대회 논문 (1저자, 4명 PM)
