#!/usr/bin/env python3
"""Evaluate both YOLO26 heads from the same full LiteRG checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ultralytics  # noqa: E402

ULTRALYTICS_ROOT = Path(ultralytics.__file__).resolve().parent
assert ULTRALYTICS_ROOT.is_relative_to(ROOT), f"Imported ultralytics outside repository: {ULTRALYTICS_ROOT}"

from ultralytics import YOLO  # noqa: E402


def use_one_to_many(model: YOLO) -> None:
    """Remove only the in-memory O2O copies so Detect follows its native O2M+NMS path."""
    head = model.model.model[-1]
    if not hasattr(head, "one2one_cv2") or not hasattr(head, "one2one_cv3"):
        raise RuntimeError("Checkpoint does not have both YOLO26 detection heads.")
    del head.one2one_cv2
    del head.one2one_cv3
    if head.end2end:
        raise RuntimeError("Failed to switch the in-memory Detect head to one-to-many mode.")


def metrics_payload(metrics, names: dict[int, str]) -> dict:
    box = metrics.box
    return {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map75": float(box.map75),
        "map50_95": float(box.map),
        "classes": {
            names[index]: {"map50": float(box.ap50[index]), "map50_95": float(box.maps[index])}
            for index in range(len(box.maps))
        },
        "speed_ms_per_image": {key: float(value) for key, value in metrics.speed.items()},
    }


def evaluate(checkpoint: Path, data: Path, branch: str, args) -> dict:
    model = YOLO(str(checkpoint))
    if getattr(model.model, "lite_rg", None) is None:
        raise RuntimeError("LiteRG is missing after independent checkpoint load.")
    if branch == "one_to_many":
        use_one_to_many(model)
    result = model.val(
        data=str(data),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        iou=args.iou,
        max_det=args.max_det,
        plots=args.plots,
        project=str(args.output / "validation"),
        name=branch,
        exist_ok=True,
        verbose=False,
    )
    return metrics_payload(result, {int(key): value for key, value in model.names.items()})


def check_fused_export(checkpoint: Path, output: Path, device: str, imgsz: int) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="literg-export-", dir=output) as directory:
        copied = Path(directory) / checkpoint.name
        shutil.copy2(checkpoint, copied)
        model = YOLO(str(copied))
        model.fuse()
        exported = Path(
            model.export(format="torchscript", imgsz=imgsz, device=device, optimize=False, verbose=False)
        )
        payload = {"format": "torchscript", "created": exported.is_file(), "size_bytes": exported.stat().st_size}
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--branch", choices=("both", "one_to_one", "one_to_many"), default="both")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--check-export", action="store_true")
    args = parser.parse_args()
    if not args.checkpoint.is_file() or not args.data.is_file():
        raise FileNotFoundError(f"Missing checkpoint or data YAML: {args.checkpoint}, {args.data}")
    args.output.mkdir(parents=True, exist_ok=True)
    branches = ("one_to_one", "one_to_many") if args.branch == "both" else (args.branch,)
    payload = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.data),
        "same_checkpoint_dual_branch": {branch: evaluate(args.checkpoint, args.data, branch, args) for branch in branches},
    }
    if args.check_export:
        payload["fused_export"] = check_fused_export(args.checkpoint, args.output / "export_check", args.device, args.imgsz)
    (args.output / "dual_branch_evaluation.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
