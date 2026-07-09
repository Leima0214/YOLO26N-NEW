# Japan7 Baseline — Round 1 (2026-07-07)

**Dataset**: Japan7 (7 classes: D00, D10, D20, D40, D43, D44, D50)
**Protocol**: `configs/mappings/japan7.yaml`
**Training**: epochs=100, imgsz=640, batch=32, seed=42, device=0

## Overall results

| Model | P | R | mAP50 | mAP50-95 | Params | FLOPs | Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8n | 0.647 | 0.606 | 0.642 | **0.353** | 3.007M | 8.1G | 1.48h |
| YOLO11n | 0.639 | 0.616 | 0.642 | 0.349 | 2.584M | 6.3G | 1.82h |
| YOLO26s | 0.678 | 0.591 | 0.630 | 0.347 | 9.468M | 20.5G | 2.68h |
| YOLO26n | 0.644 | 0.597 | 0.623 | 0.341 | **2.376M** | **5.2G** | 2.53h |

## Key findings

1. **YOLOv8n** achieves the highest mAP50-95 (0.353) with modest params.
2. **YOLO11n** ties YOLOv8n on mAP50 (0.642) and has the highest Recall (0.616).
3. **YOLO26n** has the lowest params (2.376M) and FLOPs (5.2G), but trails in mAP.
4. **YOLO26s**'s 4× parameter increase over YOLO26n yields only +0.006 mAP50-95.
5. All models struggle on D00 (mAP50 0.388–0.461) and D10 (0.368–0.411), suggesting these classes need architectural attention.
6. D43/D50 are easy classes across all models (mAP50 0.791–0.839).

## Next steps (Paper 1)

Based on YOLO26n (lowest params, most room for improvement):

- P2 / WFA / shallow detail enhancement ablations
- Focus on D00 (pothole) and D10 (longitudinal crack) — thin, elongated, low-contrast
- These modifications will happen on a separate ablation branch, NOT on `baseline/japan-baseline-engineering`
