# YOLO26 산업용 결함 검출

**YOLO26** (Ultralytics) 기반 산업용 표면 결함 검출 End-to-End 파이프라인.  
데이터셋 준비부터 ONNX 배포, C# WPF 검사 앱까지.

> **핵심 발견**: 소규모 데이터셋에서 데이터 증강(+3.6% mAP)이 모델 스케일업(+2.7% mAP)보다 효과적이었다. 엔지니어링 판단이 단순 스케일링보다 중요하다.

---

## 결과 요약

| 모델 | 데이터셋 | 이미지 수 | imgsz | mAP50 | ONNX | GPU (ms) | FPS |
|------|---------|----------|-------|-------|------|----------|-----|
| YOLO26n | NEU-DET | 1,800 | 640 | 0.726 | 5.2MB | — | — |
| YOLO26n | DeepPCB | 1,500 | 640 | **0.984** | 5.2MB | — | — |
| YOLO26n | GC10-DET (baseline) | 1,835 | 1024 | 0.635 | 9.6MB | — | — |
| YOLO26n | GC10-DET v3 | 1,835 | 1024 | 0.701 | 9.6MB | 19.0 | 53 |
| YOLO26s | GC10-DET v3 | 1,835 | 1024 | 0.720 | 36.7MB | 33.1 | 30 |
| **YOLO26n** | **GC10-DET v4 (증강)** | **4,329** | **1024** | **0.726** | **9.6MB** | **11.5** | **87** |

### 핵심 인사이트

- **Nano v4 > Small v3**: 증강 데이터를 사용한 Nano(0.726)가 원본 데이터 Small(0.720)을 능가. 파라미터는 4배 적음
- **증강 > 스케일링**: 데이터 증강 +3.6% vs 모델 스케일업 +2.7%
- **클래스 불균형 해소**: 12.4배 → 2.4배 (5배 개선, Albumentations 파이프라인)

---

## 개선 전략

### v3: 하이퍼파라미터 튜닝 (baseline → +10.4%)

| 기법 | 레퍼런스 | 효과 |
|------|---------|------|
| `imgsz=1024` | 원본 해상도 매칭 | 소형 객체 검출 향상 |
| `copy_paste=0.2` | Ghiasi et al., CVPR 2021 | 소수 클래스 증강 |
| `mixup=0.1` | Zhang et al., ICLR 2018 | 정규화 |
| `scale=0.7` | Ultralytics 권장 | 스케일 변화 대응 |

### v4: 데이터 증강 (+3.6%)

Albumentations 2.0 기반 오프라인 증강:
- 소수 클래스 집중 오버샘플링 (crease 56→550, rolled_pit 76→718)
- 증강 파이프라인: Flip + BrightnessContrast + CLAHE + GaussNoise + Rotate + RandomResizedCrop
- 총 이미지: 1,835 → 4,329장 (+136%)

---

## ONNX 추론 벤치마크

**장비**: NVIDIA RTX 3060 12GB / Intel i5 / Windows 10

| 모델 | ONNX 크기 | GPU (ms) | GPU FPS | CPU (ms) | CPU FPS |
|------|----------|----------|---------|----------|---------|
| YOLO26n v3 | 9.6 MB | 19.0 | 53 | 102.5 | 10 |
| YOLO26n v4 | 9.6 MB | 11.5 | 87 | 99.2 | 10 |
| YOLO26s v3 | 36.7 MB | 33.1 | 30 | 296.3 | 3 |

---

## InspectView — C# WPF 검사 앱

ONNX 기반 산업용 결함 검사 데스크톱 앱.

- .NET 8 / WPF / MVVM (CommunityToolkit.Mvvm)
- ONNX Runtime + OpenCvSharp4
- 드래그앤드롭, Pass/Fail 판정, 검출 로그, CSV 내보내기

---

## 기술 스택

- **학습**: Python 3.13, Ultralytics 8.4.21, PyTorch 2.6.0+cu124
- **증강**: Albumentations 2.0.8
- **추론**: ONNX Runtime 1.24 (GPU + CPU)
- **앱**: C# / .NET 8 / WPF / OpenCvSharp4
- **GPU**: NVIDIA RTX 3060 12GB

---

## 작성자

**김성진** ([@sjsr-0401](https://github.com/sjsr-0401))  
전자공학 졸업 | 산업용 비전 & 장비 제어 SW

## 라이선스

MIT
