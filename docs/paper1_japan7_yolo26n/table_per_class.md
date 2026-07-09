# Baseline Results — Per-Class mAP50-95

| Model | D00 | D10 | D20 | D40 | D43 | D44 | D50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8n | 0.219 | 0.167 | 0.338 | 0.268 | 0.570 | 0.448 | 0.459 |
| YOLO11n | 0.202 | 0.158 | 0.342 | 0.276 | 0.562 | 0.457 | 0.446 |
| YOLO26s | 0.173 | 0.150 | 0.338 | 0.289 | 0.584 | 0.439 | 0.454 |
| YOLO26n | 0.183 | 0.148 | 0.346 | 0.278 | 0.541 | 0.454 | 0.435 |

## Bottleneck classes (D00 + D10)

| Model | D00 mAP50-95 | D10 mAP50-95 | Avg(D00,D10) | vs Best |
| --- | ---: | ---: | ---: | ---: |
| YOLOv8n | **0.219** | **0.167** | **0.193** | — |
| YOLO11n | 0.202 | 0.158 | 0.180 | -0.013 |
| YOLO26n | 0.183 | 0.148 | 0.166 | -0.028 |
| YOLO26s | 0.173 | 0.150 | 0.162 | -0.031 |

## Easy classes (D43, D44, D50)

| Model | D43 mAP50-95 | D44 mAP50-95 | D50 mAP50-95 | Avg |
| --- | ---: | ---: | ---: | ---: |
| YOLO26s | **0.584** | 0.439 | **0.454** | **0.492** |
| YOLOv8n | 0.570 | 0.448 | 0.459 | 0.492 |
| YOLO11n | 0.562 | **0.457** | 0.446 | 0.488 |
| YOLO26n | 0.541 | 0.454 | 0.435 | 0.477 |

## Key insight

D43/D44/D50 (severe damage) are well-detected across all models (mAP50-95 0.435–0.584).
D00/D10 (minor/moderate cracks) are the bottleneck (mAP50-95 0.148–0.219).
The gap between easy and hard classes is 2–3×.
