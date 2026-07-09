#!/usr/bin/env python3
"""
Smoke test — 1-epoch training to verify the pipeline works end-to-end.

Usage (local CPU):
    python scripts/smoke_test_yolo26n.py --data configs/japan7_local.yaml --device cpu

Usage (remote GPU):
    python scripts/smoke_test_yolo26n.py --data configs/japan7_remote.yaml --device 0 --batch 8 --workers 4
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO26n smoke test (1 epoch)")
    parser.add_argument("--data", default="configs/japan7_local.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--amp", dest="amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    # Determine name from data path
    if "remote" in args.data:
        name = "yolo26n_japan7_smoke_remote"
    else:
        name = "yolo26n_japan7_smoke_local"

    print(f"Smoke test: {name}")
    print(f"  Data:    {args.data}")
    print(f"  Device:  {args.device}")
    print(f"  Batch:   {args.batch}")
    print()

    model = YOLO("yolo26n.pt")
    model.train(
        data=args.data,
        epochs=1,
        imgsz=320,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=42,
                             amp=args.amp,
        project="runs/smoke",
        name=name,
        exist_ok=True,
    )
    print(f"\nSmoke test complete. Results in runs/smoke/{name}/")
