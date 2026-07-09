# args.yaml — NOT stored in Git (lives on remote GPU server)

To retrieve, run on the remote GPU:

```bash
cd /root/YOLO26-probe
for run in yolov8n_japan7_e100_img640_b32_seed422 yolo11n_japan7_e100_img640_b32_seed42 yolo26n_japan7_e100_img640_b32_seed42 yolo26s_japan7_e100_img640_b32_seed42; do
  if [ -f runs/baseline/$run/args.yaml ]; then
    cp runs/baseline/$run/args.yaml experiments/japan7_baseline_20260707/args_yaml/${run}_args.yaml
  fi
done
```
