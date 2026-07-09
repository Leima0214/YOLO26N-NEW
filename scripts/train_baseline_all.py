#!/usr/bin/env python3
"""
Train all 4 baselines sequentially (YOLOv8n → YOLO11n → YOLO26n → YOLO26s).

IMPORTANT: Do NOT run this first on a remote GPU.
Always run smoke test + 3-epoch pilot first:
    1. python scripts/check_dataset.py --data configs/japan7_remote.yaml
    2. python scripts/smoke_test_yolo26n.py --data configs/japan7_remote.yaml --device 0 --batch 8 --workers 4
    3. python scripts/train_baseline_yolo26n.py --data configs/japan7_remote.yaml --epochs 3 --device 0 --workers 8 --name yolo26n_japan7_e3_test

Usage (remote GPU):
    python scripts/train_baseline_all.py --data configs/japan7_remote.yaml --epochs 100 --imgsz 640 --batch 16 --device 0 --workers 8
"""

import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ultralytics import YOLO

BASELINES = [
    ("yolov8n.pt", "yolov8n_japan7_e100_img640_seed42"),
    ("yolo11n.pt", "yolo11n_japan7_e100_img640_seed42"),
    ("yolo26n.pt", "yolo26n_japan7_e100_img640_seed42"),
    ("yolo26s.pt", "yolo26s_japan7_e100_img640_seed42"),
]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="configs/japan7_local.yaml")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--amp", dest="amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--workers", type=int, default=0)
    args = p.parse_args()

    for i, (weights, name) in enumerate(BASELINES, 1):
        print(f"\n{'='*60}")
        print(f"  [{i}/{len(BASELINES)}] {name}  ({weights})")
        print(f"{'='*60}\n")
        YOLO(weights).train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                            device=args.device, workers=args.workers, seed=42,
                             amp=args.amp,
                            project="runs/baseline", name=name)
    print(f"\nAll {len(BASELINES)} baselines complete.")
