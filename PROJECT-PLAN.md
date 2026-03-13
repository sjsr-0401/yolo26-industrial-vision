# YOLO26 산업용 결함 검출 + ONNX C# 배포 파이프라인

## 프로젝트 컨셉

**"최신 YOLO26으로 산업용 결함 검출 모델을 학습하고, ONNX로 C# WPF 앱에 배포하는 End-to-End 파이프라인"**

단순 학습 비교가 아니라, **실제 산업 현장에서 쓸 수 있는 배포 파이프라인**을 보여주는 프로젝트.

---

## Phase 1: 데이터셋 확보 + 학습 (Python)

### 선정 데이터셋 (3개 — 다양한 산업 도메인)

| # | 데이터셋 | 도메인 | 규모 | 출처 | 왜 선택? |
|---|---------|--------|------|------|---------|
| 1 | **Severstal Steel Defect** | 제철/강판 | 12,568장, 4 classes | Kaggle 대회 (2019) | 실제 제철소 데이터, 1,000+ 팀 참가, 산업 벤치마크 |
| 2 | **DeepPCB** | PCB/전자 | 1,500 이미지쌍, 6 classes | AAAI 2019 논문 | PCB 결함 검출 표준, 소형 결함 다수 |
| 3 | **Magnetic Tile Defect** | 자성소재/제조 | 1,344장, 6 classes (5 defect + 1 free) | 중국과학원 (CAS) | 다양한 결함 형태, 소형 결함 포함 |

### 왜 이 3개?

1. **Severstal** — Kaggle 대회 데이터. 규모가 크고(12K+) 실제 공장 데이터. 논문/포트폴리오에서 인용 가치 높음.
2. **DeepPCB** — 반도체/전자 분야 표준. 템플릿 매칭 기반이지만 YOLO detection으로도 활용 가능. 결함이 매우 작아서 YOLO26 ProgLoss+STAL 장점 테스트 가능.
3. **Magnetic Tile** — 제조업 일반 결함. NEU-DET보다 다양한 결함 패턴. 중국과학원 출처로 학술적 신뢰도.

### 기존 데이터 (이미 완료)

- NEU-DET (강철 표면, 1,800장, 6 classes) — 이미 학습 완료
- PCB Defect (Kaggle, 693장, 6 classes) — 이미 학습 완료

### 학습 설정

- 모델: **YOLO26n** (Nano) — 엣지 배포 기준
- Epochs: 80~100
- imgsz: 640
- Pretrained: COCO
- Augmentation: mosaic, mixup, flipud, fliplr

---

## Phase 2: ONNX Export + 벤치마크

### YOLO26 ONNX의 핵심 차별점

YOLO26은 **NMS-free end-to-end** → ONNX 모델 하나로 추론 완결. 별도 NMS 후처리 코드 불필요.

```
[기존 YOLOv8 ONNX 파이프라인]
이미지 → ONNX 추론 → Raw Boxes → NMS(C# 직접 구현) → 최종 결과
                                    ↑ 복잡, 버그 가능

[YOLO26 ONNX 파이프라인]
이미지 → ONNX 추론 → 최종 결과 (NMS 불필요!)
                      ↑ 단순, 빠름, 실수 여지 없음
```

### 벤치마크 항목

| 항목 | 측정 방법 |
|------|----------|
| ONNX 추론 속도 (GPU) | OnnxRuntime CUDA Provider, 100회 평균 |
| ONNX 추론 속도 (CPU) | OnnxRuntime CPU Provider, 50회 평균 |
| End-to-End 레이턴시 | 이미지 로드 → 전처리 → 추론 → 후처리 → 결과 |
| 모델 크기 | .onnx 파일 사이즈 |
| 메모리 사용량 | 추론 시 peak memory |

---

## Phase 3: C# WPF 시각화 앱 — "InspectView"

### 앱 개요

산업용 결함 검출 결과를 시각화하는 **WPF 데스크탑 앱**.

### 핵심 기능

1. **이미지 로드 + 결함 검출**
   - 이미지 파일 드래그&드롭 또는 파일 선택
   - YOLO26 ONNX 모델로 실시간 추론
   - Bounding Box + 클래스명 + 신뢰도 오버레이

2. **모델 선택**
   - 3개 데이터셋 × YOLO26n 모델 중 선택
   - Steel / PCB / Magnetic Tile

3. **결과 패널**
   - 검출된 결함 목록 (클래스, 신뢰도, 좌표)
   - 추론 시간 표시 (ms)
   - 결함 통계 차트

4. **배치 검사 모드**
   - 폴더 내 이미지 일괄 검사
   - 결과 CSV 내보내기
   - Pass/Fail 판정 (결함 개수/신뢰도 기준)

### 기술 스택

- .NET 8 / WPF
- Microsoft.ML.OnnxRuntime (GPU + CPU)
- OpenCvSharp4 (이미지 전처리)
- LiveCharts2 or OxyPlot (차트)

### UI 레이아웃 (초안)

```
┌──────────────────────────────────────────────┐
│  InspectView — YOLO26 Industrial Inspector   │
├──────────┬───────────────────────────────────┤
│          │                                   │
│  모델    │     [검출 결과 이미지]              │
│  선택    │     BBox + Label 오버레이           │
│          │                                   │
│ ○ Steel  │                                   │
│ ○ PCB    │                                   │
│ ○ Tile   │                                   │
│          ├───────────────────────────────────┤
│  설정    │  결함 목록        │  추론 정보      │
│ ───────  │  - Scratch 0.92  │  Model: 26n    │
│ Conf:0.5 │  - Crack   0.87  │  Time: 6.2ms   │
│ [검사]   │  - Pit     0.73  │  FPS: 161      │
│ [배치]   │                  │  Size: 9.4MB   │
└──────────┴──────────────────┴────────────────┘
```

---

## Phase 4: GitHub 레포 구조

```
yolo26-industrial-vision/
├── README.md                    # 프로젝트 소개 (한/영)
├── LICENSE                      # MIT
├── .gitignore
│
├── training/                    # Phase 1: Python 학습
│   ├── scripts/
│   │   ├── download_datasets.py # 데이터셋 자동 다운로드
│   │   ├── convert_to_yolo.py   # 포맷 변환 (VOC/COCO → YOLO)
│   │   ├── train.py             # 학습 스크립트
│   │   └── evaluate.py          # 평가 + 비교표 생성
│   ├── configs/                 # 데이터셋 YAML
│   ├── results/                 # 학습 결과 (metrics, plots)
│   └── requirements.txt
│
├── export/                      # Phase 2: ONNX
│   ├── export_onnx.py           # PT → ONNX 변환
│   ├── benchmark.py             # ONNX 속도 벤치마크
│   └── models/                  # .onnx 파일
│
├── app/                         # Phase 3: C# WPF 앱
│   ├── InspectView.sln
│   ├── InspectView/
│   │   ├── Models/              # ONNX 추론 래퍼
│   │   ├── ViewModels/          # MVVM
│   │   ├── Views/               # XAML
│   │   ├── Services/            # 이미지 처리, 결과 내보내기
│   │   └── Assets/              # ONNX 모델 파일
│   └── InspectView.Tests/       # 단위 테스트
│
├── demo/                        # Gradio 웹 데모
│   └── app.py
│
├── docs/                        # 문서
│   ├── results.md               # 전체 결과 비교표
│   ├── deployment.md            # C# 배포 가이드
│   └── images/                  # 스크린샷, 결과 이미지
│
└── assets/                      # README용 이미지
    ├── banner.png
    ├── app_screenshot.png
    └── architecture.png
```

---

## Phase 5: README + 영상

### README 핵심 구성

1. **배너 이미지** — 앱 스크린샷 + 결함 검출 예시
2. **프로젝트 소개** — 한 문단으로 뭘 하는지
3. **결과 비교표** — 데이터셋별 mAP, 속도, 모델 크기
4. **아키텍처 다이어그램** — 학습→ONNX→C# 앱 플로우
5. **Quick Start** — Python 학습 / C# 앱 실행 방법
6. **데이터셋 출처** — 논문/대회 레퍼런스 명시

### 영상 (성진이 나레이션)

1. 프로젝트 소개 (30초)
2. 학습 결과 비교 (30초)
3. C# 앱 데모 — 이미지 로드 → 검출 → 결과 (60초)
4. ONNX 배포 핵심 설명 (30초)

---

## 타임라인

| Phase | 작업 | 예상 시간 |
|-------|------|----------|
| 1-1 | Severstal 데이터셋 다운 + 변환 | 30분 |
| 1-2 | DeepPCB 다운 + 변환 | 15분 |
| 1-3 | Magnetic Tile 다운 + 변환 | 15분 |
| 1-4 | YOLO26n 학습 ×3 (80 epochs each) | 2~3시간 |
| 2 | ONNX export + 벤치마크 | 30분 |
| 3 | C# WPF 앱 (InspectView) | 3~4시간 |
| 4 | GitHub 정리 + README | 1시간 |
| 5 | 영상 스크립트 + 편집 | 1시간 |
| **합계** | | **약 8~10시간** |

---

## 핵심 메시지 (포트폴리오/영상용)

> "YOLO26 모델을 3개 산업 데이터셋(강철/PCB/자성타일)에 파인튜닝하고,
> NMS-free ONNX로 export해서 C# WPF 앱에 배포했습니다.
> End-to-End 추론 파이프라인으로 별도 후처리 없이 6ms 내 결함 검출이 가능합니다."

---

## 레퍼런스

1. **Severstal Steel Defect Detection** — Kaggle Competition (2019), 1,000+ teams
   - https://www.kaggle.com/c/severstal-steel-defect-detection
2. **DeepPCB** — Tang et al., "Online PCB Defect Detector On A New PCB Defect Dataset" (2019)
   - https://github.com/tangsanli5201/DeepPCB
3. **Magnetic Tile Defect** — Huang et al., Chinese Academy of Sciences
   - https://github.com/abin24/Magnetic-tile-defect-datasets
4. **NEU Surface Defect** — Song & Yan, Northeastern University
   - http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html
5. **YOLO26** — Ultralytics (2026)
   - https://docs.ultralytics.com/models/yolo26/
6. **Surface Defect Detection Dataset Collection** — Charmve (GitHub, 3.5K+ stars)
   - https://github.com/Charmve/Surface-Defect-Detection
