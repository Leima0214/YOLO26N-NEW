# Japan7 Baseline — Overall Metrics

| Model | Weight | Dataset | Epochs | ImgSize | Batch | Params_M | FLOPs_G | Train_Time_h | Precision | Recall | mAP50 | mAP50_95 | Run_Dir | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| YOLOv8n | yolov8n.pt | Japan7 | 100 | 640 | 32 | 3.007 | 8.1 | 1.480 | 0.647 | 0.606 | 0.642 | 0.353 | yolov8n_japan7_e100_img640_b32_seed422 | seed422 |
| YOLO11n | yolo11n.pt | Japan7 | 100 | 640 | 32 | 2.584 | 6.3 | 1.819 | 0.639 | 0.616 | 0.642 | 0.349 | yolo11n_japan7_e100_img640_b32_seed42 | |
| YOLO26s | yolo26s.pt | Japan7 | 100 | 640 | 32 | 9.468 | 20.5 | 2.680 | 0.678 | 0.591 | 0.630 | 0.347 | yolo26s_japan7_e100_img640_b32_seed42 | |
| YOLO26n | yolo26n.pt | Japan7 | 100 | 640 | 32 | 2.376 | 5.2 | 2.533 | 0.644 | 0.597 | 0.623 | 0.341 | yolo26n_japan7_e100_img640_b32_seed42 | |
