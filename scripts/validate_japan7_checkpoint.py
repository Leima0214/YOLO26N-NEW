"""Validate a Japan7 checkpoint and emit aggregate and per-class metrics as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("weights")
    parser.add_argument("--data", default="configs/japan7_remote.yaml")
    parser.add_argument("--name", required=True)
    parser.add_argument("--zero-gate-layer", type=int)
    parser.add_argument("--overlock-layer", type=int)
    parser.add_argument("--overlock-mode", choices=("full", "local", "overview", "average"))
    args = parser.parse_args()

    model = YOLO(args.weights)
    if args.overlock_mode is not None:
        assert args.overlock_layer is not None, "--overlock-layer is required with --overlock-mode"
        enhance = model.model.model[args.overlock_layer].enhance
        enhance.output_mode = args.overlock_mode
        print(f"overlock_mode=model.{args.overlock_layer}.enhance:{args.overlock_mode}")
    if args.zero_gate_layer is not None:
        gate = model.model.model[args.zero_gate_layer].enhance.scale.gamma
        with torch.no_grad():
            gate.zero_()
        print(f"zeroed_gate=model.{args.zero_gate_layer}.enhance.scale.gamma")

    metrics = model.val(
        data=args.data,
        imgsz=640,
        batch=32,
        device=0,
        workers=8,
        iou=0.7,
        max_det=300,
        plots=False,
        save_json=False,
        project=str((REPO_ROOT / "runs" / "paper1_eval").resolve()),
        name=args.name,
        exist_ok=True,
        verbose=True,
    )

    class_ids = [int(value) for value in metrics.box.ap_class_index]
    per_class = {}
    for position, class_id in enumerate(class_ids):
        per_class[metrics.names[class_id]] = {
            "precision": float(metrics.box.p[position]),
            "recall": float(metrics.box.r[position]),
            "map50": float(metrics.box.ap50[position]),
            "map75": float(metrics.box.all_ap[position, 5]),
            "map50_95": float(metrics.box.maps[position]),
        }

    payload = {
        "aggregate": {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "map50": float(metrics.box.map50),
            "map75": float(metrics.box.map75),
            "map50_95": float(metrics.box.map),
        },
        "per_class": per_class,
    }
    print("METRICS_JSON=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
