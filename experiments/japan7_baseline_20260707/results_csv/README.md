# results.csv — NOT stored in Git (lives on remote GPU server)

To retrieve, run on the remote GPU:

```bash
cd /root/YOLO26-probe
for run in yolov8n_japan7_e100_img640_b32_seed422 yolo11n_japan7_e100_img640_b32_seed42 yolo26n_japan7_e100_img640_b32_seed42 yolo26s_japan7_e100_img640_b32_seed42; do
  cp runs/baseline/$run/results.csv experiments/japan7_baseline_20260707/results_csv/${run}_results.csv
done
```
