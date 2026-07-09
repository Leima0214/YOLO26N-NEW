# Baseline Results — Overall

| Model | Params | FLOPs | P | R | mAP50 | mAP50-95 | Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8n | 3.007M | 8.1G | 0.647 | 0.606 | 0.642 | **0.353** | 1.48h |
| YOLO11n | 2.584M | 6.3G | 0.639 | 0.616 | 0.642 | 0.349 | 1.82h |
| YOLO26s | 9.468M | 20.5G | 0.678 | 0.591 | 0.630 | 0.347 | 2.68h |
| YOLO26n | 2.376M | 5.2G | 0.644 | 0.597 | 0.623 | 0.341 | 2.53h |

Dataset: Japan7 (7 classes). Epochs=100, imgsz=640, batch=32, seed=42.

**Best overall**: YOLOv8n (mAP50-95 = 0.353)
**Lightest**: YOLO26n (2.376M params, 5.2G FLOPs)
**Key observation**: YOLO26s (4× params vs YOLO26n) gains only +0.006 mAP50-95
