# YOLO26 산업 결함 검출 — Baseline 분석 및 개선 전략

## 1. Baseline 실험 결과

### 1.1 학습 설정 (공통)
| 항목 | 값 |
|------|-----|
| 모델 | YOLO26n (2.5M params, 5.8 GFLOPs) |
| Pretrained | COCO (yolo26n.pt) |
| imgsz | 640 |
| Optimizer | Auto (AdamW, lr=0.001) |
| Augmentation | mosaic=1.0, mixup=0.1, flipud=0.5, fliplr=0.5 |
| Batch | Auto (-1) |
| AMP | True |
| cos_lr | True |

### 1.2 결과 요약

| 데이터셋 | Epochs | P | R | F1 | mAP50 | mAP50-95 | 평가 |
|---------|--------|---|---|-----|-------|----------|------|
| NEU-DET (강철) | 80 | 0.706 | 0.657 | 0.681 | 0.730 | 0.390 | P/R 모두 0.70 미만 |
| PCB-Defect (Kaggle) | 80 | 0.889 | 0.634 | 0.740 | 0.711 | 0.376 | P 높으나 R 매우 낮음 |
| DeepPCB (진행중) | 100 | — | — | — | — | — | 학습 중 |
| GC10-DET (진행중) | 80 | — | — | — | — | — | 학습 중 |

### 1.3 클래스별 성능 편차 (PCB-Defect)

| 클래스 | P | R | mAP50 | 문제 |
|--------|---|---|-------|------|
| missing_hole | 1.000 | 0.000 | 0.037 | **완전 미검출** |
| mouse_bite | 0.848 | 0.784 | 0.866 | 양호 |
| open_circuit | 0.901 | 0.700 | 0.859 | 양호 |
| short | 0.868 | 0.870 | 0.915 | 좋음 |
| spur | 0.862 | 0.659 | 0.749 | R 낮음 |
| spurious_copper | 0.856 | 0.790 | 0.841 | 양호 |

**핵심 문제**: `missing_hole` 클래스 Recall=0.000. 115개 인스턴스가 있는데 단 하나도 검출 못 함.

---

## 2. 문제 진단

### 2.1 모델 Capacity 부족

**근거**: Ultralytics 공식 문서 및 YOLO 아키텍처 설계 원칙

YOLO26n은 2.5M parameters로, 가장 작은 모델이다. 복잡한 결함 패턴을 구분하기에는 feature representation이 부족할 수 있다.

| 모델 | Params | GFLOPs | COCO mAP50-95 |
|------|--------|--------|---------------|
| YOLO26n | 2.5M | 5.8 | 기본 |
| YOLO26s | 9.5M | 21.4 | +4~5% |
| YOLO26m | 20.2M | 58.5 | +7~8% |

**참고**: Ultralytics 공식 벤치마크에서 n→s로 올리면 COCO mAP50-95가 약 4~5% 상승.

### 2.2 이미지 해상도 미스매치

**근거**: YOLO 공식 문서 "Image Size" 섹션, 소형 객체 검출 논문

| 데이터셋 | 원본 해상도 | 학습 imgsz | 문제 |
|---------|------------|-----------|------|
| NEU-DET | 200×200 | 640 | 3.2배 업스케일 — 의미없는 픽셀 생성 |
| DeepPCB | 640×640 | 640 | 정확히 일치 — 최적 |
| GC10-DET | 2048×1000 | 640 | 3.2배 다운스케일 — 소형 결함 정보 손실 |

NEU-DET은 원본이 200px이라 640으로 키워도 해상도가 올라가지 않는다.
GC10-DET은 반대로 고해상도 이미지를 640으로 축소하면서 작은 결함이 사라질 수 있다.

**해결**: GC10-DET은 imgsz=1280으로 키우면 소형 결함 검출이 개선된다.

### 2.3 학습 Epochs 부족

**근거**: Ultralytics 공식 권장 — "300 epochs for best results"

현재 80 epochs + patience 20은 데이터셋에 따라 수렴 전에 early stop될 수 있다.
특히 작은 데이터셋(NEU-DET 1,800장)은 더 많은 epoch이 필요하다.

### 2.4 클래스 불균형 및 Small Object 문제

**근거**: 
- FCOS (Tian et al., ICCV 2019) — anchor-free 소형 객체 검출
- YOLO26 ProgLoss+STAL — 소형 객체 개선을 위해 도입된 기술

PCB-Defect에서 `missing_hole`이 Recall=0인 것은:
1. 결함 크기가 매우 작아서 640 해상도에서 사라짐
2. 다른 클래스와 시각적으로 유사해서 구분 못 함
3. 데이터 수는 115개로 적지 않으나, 학습 시 augmentation이 불충분

### 2.5 Augmentation 강도 부족

**근거**: Ultralytics docs "Data Augmentation" + copy_paste 논문

현재 augmentation:
- mosaic=1.0 ✓
- mixup=0.1 ✓ (기본)
- flipud=0.5 ✓
- fliplr=0.5 ✓

누락된 효과적인 augmentation:
- **copy_paste=0.3** — 소형 객체를 다른 이미지에 복사. 소형 결함 검출에 매우 효과적
- **scale=0.9** — 스케일 변화 강화 (기본 0.5)
- **degrees=10** — 약간의 회전

---

## 3. 개선 전략

### 3.1 개선 설정

```python
model.train(
    # === 핵심 변경 ===
    imgsz=1280,          # 640 → 1280 (소형 결함 핵심)
    epochs=200,          # 80 → 200 (충분한 수렴)
    patience=30,         # 20 → 30
    
    # === Augmentation 강화 ===
    copy_paste=0.3,      # NEW: 소형 객체 복사 증강
    scale=0.9,           # 0.5 → 0.9 (스케일 변화 강화)
    degrees=10,          # 0 → 10 (회전 추가)
    mosaic=1.0,          # 유지
    mixup=0.15,          # 0.1 → 0.15
    
    # === 기타 ===
    batch=-1,            # auto (1280에선 batch 4~8)
    cos_lr=True,
    amp=True,
    close_mosaic=20,     # 10 → 20 (마지막 20ep에서 mosaic 끔)
)
```

### 3.2 각 변경의 근거

| 변경 | 근거 | 기대 효과 |
|------|------|----------|
| imgsz 1280 | 소형 객체 검출은 해상도에 직접 비례 (YOLO docs) | mAP +5~10%, 특히 소형 결함 |
| epochs 200 | Ultralytics "300 epochs recommended" | 수렴 보장, 특히 작은 데이터셋 |
| copy_paste 0.3 | Simple Copy-Paste (Ghiasi et al., CVPR 2021) | 소형 객체 검출 개선 |
| scale 0.9 | 스케일 변화에 robust한 검출기 학습 | 다양한 크기 결함 대응 |
| close_mosaic 20 | YOLO11 공식 기본값 (마지막 epoch은 원본 이미지로 학습) | 최종 성능 미세 조정 |

### 3.3 목표 수치

| 지표 | Baseline 범위 | 목표 | 근거 |
|------|-------------|------|------|
| Precision | 0.64~0.89 | **0.85+** | 서비스/데모용 시작선 |
| Recall | 0.63~0.70 | **0.85+** | 산업 결함: 놓치면 안 됨 |
| F1 | 0.68~0.74 | **0.85+** | P/R 균형 |
| mAP50 | 0.71~0.74 | **0.85+** | 검출 품질 종합 |
| mAP50-95 | 0.37~0.39 | **0.55+** | 박스 위치 정밀도 |
| 클래스별 최소 R | 0.00 (!!) | **0.50+** | 모든 클래스 검출 가능 |

---

## 4. 레퍼런스

1. **FCOS** — Tian et al., "FCOS: Fully Convolutional One-Stage Object Detection", ICCV 2019
   - Anchor-free detection의 원류. 소형 객체 검출에서 anchor 기반 대비 우수성 입증.

2. **DETR** — Carion et al., "End-to-End Object Detection with Transformers", ECCV 2020
   - NMS-free end-to-end 검출의 원류. YOLO26의 NMS 제거 이론적 기반.

3. **Simple Copy-Paste** — Ghiasi et al., CVPR 2021
   - 소형 객체를 다른 이미지에 복사하는 augmentation. COCO에서 +1~2% mAP 향상.

4. **NEU Surface Defect Database** — Song & Yan, Applied Surface Science, 2013
   - 산업 결함 검출 표준 벤치마크. 인용 1,500+.

5. **DeepPCB** — Tang et al., 2019
   - PCB 결함 검출 표준 데이터셋. 640×640, 6 classes, bbox annotation.

6. **GC10-DET** — Lv et al., Acta Optica Sinica, 2020
   - NEU-DET 상위 호환. 10 classes, 2,294 images.

7. **Ultralytics YOLO26 Documentation**
   - https://docs.ultralytics.com/models/yolo26/

8. **YOLO Architecture Survey** — Terven et al., MLKE 2023
   - YOLOv1→v8 아키텍처 진화 종합 분석. 인용 1,000+.
