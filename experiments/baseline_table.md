# Japan7 Baseline Results

Dataset: Japan7  
Classes: D00, D10, D20, D40, D43, D44, D50  
Protocol: epochs=100, imgsz=640, batch=32, device=0, workers=8, seed=42  
Metric source: final validation on best.pt

| Model | Dataset | Epochs | Img | Batch | Params | FLOPs | P | R | mAP50 | mAP50-95 | Run |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| YOLOv8n | Japan7 | 100 | 640 | 32 | 3.007M | 8.1G | 0.647 | 0.606 | 0.642 | 0.353 | yolov8n_japan7_e100_img640_b32_seed422 |
| YOLO11n | Japan7 | 100 | 640 | 32 | 2.584M | 6.3G | 0.639 | 0.616 | 0.642 | 0.349 | yolo11n_japan7_e100_img640_b32_seed42 |
| YOLO26s | Japan7 | 100 | 640 | 32 | 9.468M | 20.5G | 0.678 | 0.591 | 0.630 | 0.347 | yolo26s_japan7_e100_img640_b32_seed42 |
| YOLO26n | Japan7 | 100 | 640 | 32 | 2.376M | 5.2G | 0.644 | 0.597 | 0.623 | 0.341 | yolo26n_japan7_e100_img640_b32_seed42 |

## Preliminary observations

1. YOLOv8n achieved the highest mAP50-95 on Japan7.
2. YOLO11n and YOLOv8n achieved the same mAP50 of 0.642.
3. YOLO26n has the lowest Params and FLOPs, but its detection accuracy is lower than YOLOv8n and YOLO11n.
4. YOLO26s has much larger Params and FLOPs than YOLO26n, but does not bring clear accuracy gains.
5. The main weak categories across models are D00 and D10.
6. The next improvement stage should focus on thin cracks, low-contrast damage, small targets, shallow details, and D00/D10 performance.
