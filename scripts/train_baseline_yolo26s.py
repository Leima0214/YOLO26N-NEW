#!/usr/bin/env python3
"""Train YOLO26s baseline. See train_baseline_yolo26n.py for full usage."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ultralytics import YOLO

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
    p.add_argument("--project", default="runs/baseline")
    p.add_argument("--name", default="yolo26s_japan7_e100_img640_seed42")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    print(f"Training: {args.name} | Model: yolo26s.pt | Data: {args.data} | Epochs: {args.epochs} | Device: {args.device}")
    YOLO("yolo26s.pt").train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                             device=args.device, workers=args.workers, seed=42,
                             amp=args.amp, project=args.project,
                             name=args.name, resume=args.resume)
    print(f"Done. Results in {args.project}/{args.name}/")
