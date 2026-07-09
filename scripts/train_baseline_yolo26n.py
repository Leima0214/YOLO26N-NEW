#!/usr/bin/env python3
"""
Train YOLO26n baseline on Japan domain.

Usage (local CPU):
    python scripts/train_baseline_yolo26n.py --data configs/japan7_local.yaml --device cpu

Usage (remote GPU — 3 epoch pilot):
    python scripts/train_baseline_yolo26n.py --data configs/japan7_remote.yaml --epochs 3 --imgsz 640 --batch 16 --device 0 --workers 8 --name yolo26n_japan_e3_test

Usage (remote GPU — full):
    python scripts/train_baseline_yolo26n.py --data configs/japan7_remote.yaml --epochs 100 --imgsz 640 --batch 16 --device 0 --workers 8
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO26n baseline")
    parser.add_argument("--data", default="configs/japan7_local.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", default="runs/baseline")
    parser.add_argument("--name", default="yolo26n_japan7_e100_img640_seed42")
    parser.add_argument("--amp", dest="amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    print(f"Training: {args.name}")
    print(f"  Model:   yolo26n.pt")
    print(f"  Data:    {args.data}")
    print(f"  Epochs:  {args.epochs}")
    print(f"  ImgSz:   {args.imgsz}")
    print(f"  Batch:   {args.batch}")
    print(f"  Device:  {args.device}")
    print()

    model = YOLO("yolo26n.pt")
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=42,
                             amp=args.amp,
        project=args.project,
        name=args.name,
        resume=args.resume,
    )
    print(f"\nTraining complete. Results in {args.project}/{args.name}/")
