"""Run matched Japan7 B0-B6 ablations for Progressive LiteRG-YOLO26."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import YAML


ROOT = Path(__file__).resolve().parents[1]
BASELINE_MODEL = ROOT / "ultralytics/cfg/models/26/yolo26n.yaml"
LITERG_MODEL = ROOT / "ultralytics/cfg/models/26/yolo26n-literg.yaml"
PROJECT = ROOT / "runs/paper1_literg"
STAGES = {
    "B0": {},
    "B1": {"target_mode": "hard", "use_drg": False, "use_rff": False},
    "B2": {"target_mode": "soft", "use_drg": False, "use_rff": False},
    "B3": {"target_mode": "soft", "use_drg": True, "use_rff": False},
    "B4": {"target_mode": "soft", "use_drg": True, "use_rff": True, "progressive_region": False},
    "B5": {"target_mode": "soft", "use_drg": True, "use_rff": True, "progressive_region": True},
    "B6": {
        "target_mode": "soft",
        "use_drg": True,
        "use_rff": True,
        "progressive_region": True,
        "end2end": False,
    },
    "B7": {"target_mode": "soft", "use_drg": True, "use_rff": True, "progressive_region": True},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="B5", help="Experiment stage; defaults to the full B5 model.")
    parser.add_argument("--data", default="configs/japan7_remote.yaml")
    parser.add_argument("--weights", default="yolo26n.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default=None)
    return parser.parse_args()


def build_model(stage: str) -> YOLO:
    """Build B0 directly or materialize a short-lived YAML for one controlled LiteRG ablation."""
    if stage == "B0":
        return YOLO(str(BASELINE_MODEL))

    config = YAML.load(LITERG_MODEL)
    config["scale"] = "n"
    stage_config = dict(STAGES[stage])
    config["end2end"] = bool(stage_config.pop("end2end", True))
    config["lite_rg"].update(stage_config)
    handle = tempfile.NamedTemporaryFile(prefix="yolo26n-literg-", suffix=".yaml", delete=False)
    temporary_yaml = Path(handle.name)
    handle.close()
    try:
        YAML.save(temporary_yaml, config)
        return YOLO(str(temporary_yaml))
    finally:
        temporary_yaml.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    model = build_model(args.stage)
    model.load(args.weights)
    run_name = args.name or f"literg_{args.stage.lower()}_japan7_{args.epochs}e_seed{args.seed}"
    model.train(
        data=args.data,
        project=str(PROJECT),
        name=run_name,
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        amp=True,
        optimizer="auto",
        cos_lr=False,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        warmup_epochs=3.0,
        weight_decay=0.0005,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=10,
        iou=0.7,
        max_det=300,
    )


if __name__ == "__main__":
    main()
