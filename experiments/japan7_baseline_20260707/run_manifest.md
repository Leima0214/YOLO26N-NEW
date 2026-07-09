# Run Manifest — Japan7 Baseline Round 1

Valid runs on remote GPU (`/root/YOLO26-probe/runs/detect/`).

| Model | Run Name | Path | Status |
| --- | --- | --- | --- |
| YOLOv8n | yolov8n_japan7_e100_img640_b32_seed422 | `runs/baseline/yolov8n_japan7_e100_img640_b32_seed422` | ✅ valid |
| YOLOv8n | yolov8n_japan7_e100_img640_b32_seed42 | `runs/baseline/yolov8n_japan7_e100_img640_b32_seed42` | ❌ empty dir |
| YOLO11n | yolo11n_japan7_e100_img640_b32_seed42 | `runs/baseline/yolo11n_japan7_e100_img640_b32_seed42` | ✅ valid |
| YOLO26n | yolo26n_japan7_e100_img640_b32_seed42 | `runs/baseline/yolo26n_japan7_e100_img640_b32_seed42` | ✅ valid |
| YOLO26s | yolo26s_japan7_e100_img640_b32_seed42 | `runs/baseline/yolo26s_japan7_e100_img640_b32_seed42` | ✅ valid |

## Missing artifacts

`results.csv` and `args.yaml` from the remote GPU server were not copied to this commit.
They exist at the remote paths above and should be archived before the runs directory is cleaned up.
