# Commands used for Japan7 baseline training (remote GPU)

## Build datasets
```bash
bash scripts/build_all_derived_datasets.sh
python scripts/check_dataset.py --data configs/japan7_remote.yaml
```

## Train
```bash
# YOLOv8n (seed422 — original seed=42 run was empty)
python scripts/train_baseline_yolov8n.py \
  --data configs/japan7_remote.yaml --epochs 100 --imgsz 640 --batch 32 \
  --device 0 --workers 8 --name yolov8n_japan7_e100_img640_b32_seed422

# YOLO11n
python scripts/train_baseline_yolo11n.py \
  --data configs/japan7_remote.yaml --epochs 100 --imgsz 640 --batch 32 \
  --device 0 --workers 8 --name yolo11n_japan7_e100_img640_b32_seed42

# YOLO26n
python scripts/train_baseline_yolo26n.py \
  --data configs/japan7_remote.yaml --epochs 100 --imgsz 640 --batch 32 \
  --device 0 --workers 8 --name yolo26n_japan7_e100_img640_b32_seed42

# YOLO26s
python scripts/train_baseline_yolo26s.py \
  --data configs/japan7_remote.yaml --epochs 100 --imgsz 640 --batch 32 \
  --device 0 --workers 8 --name yolo26s_japan7_e100_img640_b32_seed42
```
