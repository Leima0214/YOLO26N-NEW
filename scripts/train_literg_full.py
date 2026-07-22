#!/usr/bin/env python3
"""Dedicated, reproducible training entry for the complete LiteRG-YOLO26n model."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ultralytics  # noqa: E402

ULTRALYTICS_ROOT = Path(ultralytics.__file__).resolve().parent
assert ULTRALYTICS_ROOT.is_relative_to(ROOT), f"Imported ultralytics outside repository: {ULTRALYTICS_ROOT}"

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils import YAML  # noqa: E402
from ultralytics.utils.torch_utils import unwrap_model  # noqa: E402


MODEL_YAML = ROOT / "ultralytics/cfg/models/26/yolo26-literg.yaml"
DEFAULT_CONFIG = ROOT / "configs/literg_full_japan7_100e.yaml"
LFS_SIGNATURE = b"version https://git-lfs.github.com/spec/v1"
EXPECTED_LITERG_PARAMETERS = 91_787


def git_output(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8"
    )
    return result.stdout.strip()


def check_weight_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Pretrained YOLO26n weights do not exist: {path}")
    if path.read_bytes()[:128].startswith(LFS_SIGNATURE):
        raise RuntimeError(f"Pretrained file is a Git LFS pointer, not a real checkpoint: {path}")
    if path.stat().st_size < 1_000_000:
        raise RuntimeError(f"Pretrained checkpoint is unexpectedly small ({path.stat().st_size} bytes): {path}")


def full_model() -> YOLO:
    if not MODEL_YAML.is_file():
        raise FileNotFoundError(MODEL_YAML)
    config = YAML.load(MODEL_YAML)
    config["scale"] = "n"
    lite_rg = config["lite_rg"]
    required = {
        "enabled": True,
        "use_drg": True,
        "use_rff": True,
        "target_mode": "soft",
        "progressive_region": True,
    }
    for key, expected in required.items():
        if lite_rg.get(key) != expected:
            raise RuntimeError(f"Full LiteRG requires lite_rg.{key}={expected!r}, got {lite_rg.get(key)!r}")
    handle = tempfile.NamedTemporaryFile(prefix="yolo26n-literg-full-", suffix=".yaml", delete=False)
    temporary_yaml = Path(handle.name)
    handle.close()
    try:
        YAML.save(temporary_yaml, config)
        model = YOLO(str(temporary_yaml))
    finally:
        temporary_yaml.unlink(missing_ok=True)
    if getattr(model.model, "lite_rg", None) is None:
        raise RuntimeError("Full model was constructed without LiteRG.")
    lite_parameters = sum(parameter.numel() for parameter in model.model.lite_rg.parameters())
    if lite_parameters != EXPECTED_LITERG_PARAMETERS:
        raise RuntimeError(f"Expected {EXPECTED_LITERG_PARAMETERS} LiteRG parameters, found {lite_parameters}.")
    if not any(name.startswith("lite_rg.") for name in model.model.state_dict()):
        raise RuntimeError("Full model state_dict has no lite_rg.* keys.")
    return model


def progressive_schedule(epochs: int, gain: float, floor: float) -> list[dict]:
    rows = []
    for epoch in range(1, epochs + 1):
        o2m = max(1 - (epoch - 1) / max(epochs - 1, 1), 0) * 0.7 + 0.1
        rows.append(
            {
                "epoch": epoch,
                "updates_before_epoch": epoch - 1,
                "o2m": o2m,
                "o2o": 1 - o2m,
                "effective_region_lambda": gain * max(o2m, floor),
            }
        )
    return rows


class LiteRGTelemetry:
    def __init__(self, command: list[str], weights: Path, data: Path, diagnostic_log: bool) -> None:
        self.command = command
        self.weights = weights
        self.data = data
        self.diagnostic_log = diagnostic_log

    def pretrain_end(self, trainer) -> None:
        core = unwrap_model(trainer.model)
        lite_rg = core.lite_rg
        config = core.yaml["lite_rg"]
        schedule = progressive_schedule(trainer.epochs, float(config["region_gain"]), float(config["region_floor"]))
        metadata = {
            "git": {
                "commit": git_output("rev-parse", "HEAD"),
                "branch": git_output("branch", "--show-current"),
                "status_porcelain": git_output("status", "--porcelain", "--untracked-files=no").splitlines(),
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "ultralytics_source": str(ULTRALYTICS_ROOT),
            },
            "command": self.command,
            "weights": str(self.weights),
            "data": str(self.data),
            "parameters": {
                "total": sum(parameter.numel() for parameter in core.parameters()),
                "lite_rg": sum(parameter.numel() for parameter in lite_rg.parameters()),
            },
            "initial_scalars": {
                name: float(getattr(lite_rg, name).detach().float().cpu())
                for name in ("gamma3", "gamma4", "eta3", "eta4")
            },
            "lite_rg_config": config,
            "progressive_region_schedule": schedule,
            "optimizer_groups": [
                {
                    "index": index,
                    "param_group": group.get("param_group"),
                    "parameters": sum(parameter.numel() for parameter in group["params"]),
                    "lr": float(group["lr"]),
                    "weight_decay": float(group.get("weight_decay", 0.0)),
                    "use_muon": bool(group.get("use_muon", False)),
                }
                for index, group in enumerate(trainer.optimizer.param_groups)
            ],
        }
        save_dir = Path(trainer.save_dir)
        (save_dir / "literg_run_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        with (save_dir / "progressive_region_schedule.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(schedule[0]))
            writer.writeheader()
            writer.writerows(schedule)

    def fit_epoch_end(self, trainer) -> None:
        if not self.diagnostic_log:
            return
        core = unwrap_model(trainer.model)
        criterion = core.criterion
        o2m_used = max(1 - trainer.epoch / max(trainer.epochs - 1, 1), 0) * 0.7 + 0.1
        effective_lambda_used = float(criterion.region_gain * max(o2m_used, criterion.region_floor))
        row = {
            "epoch": trainer.epoch + 1,
            "criterion_updates_after_epoch": getattr(criterion, "updates", None),
            "o2m_used_for_epoch": o2m_used,
            "o2o_used_for_epoch": 1 - o2m_used,
            "o2m_after_epoch_update": getattr(criterion, "o2m", None),
            "effective_region_lambda_used": effective_lambda_used,
            "logged_train_region_loss_weighted": float(trainer.tloss[-1].detach().cpu()),
            "gamma3": float(core.lite_rg.gamma3.detach().float().cpu()),
            "gamma4": float(core.lite_rg.gamma4.detach().float().cpu()),
            "eta3": float(core.lite_rg.eta3.detach().float().cpu()),
            "eta4": float(core.lite_rg.eta4.detach().float().cpu()),
        }
        row["derived_raw_region_loss"] = row["logged_train_region_loss_weighted"] / max(
            effective_lambda_used, 1e-12
        )
        path = Path(trainer.save_dir) / "literg_training_diagnostics.csv"
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--name")
    parser.add_argument("--no-diagnostic-log", action="store_true")
    args = parser.parse_args()
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    config = YAML.load(args.config)
    config_data = config.pop("data")
    config_weights = config.pop("weights")
    data = (args.data or ROOT / config_data).resolve()
    weights = (args.weights or ROOT / config_weights).resolve()
    project = (ROOT / config.pop("project")).resolve()
    split_status = config.pop("split_status")
    assert not {"data", "weights", "project", "split_status", "resume"} & config.keys()
    if args.device is not None:
        config["device"] = args.device
    if args.name is not None:
        config["name"] = args.name
    check_weight_file(weights)
    if not data.is_file():
        raise FileNotFoundError(data)
    model = full_model()
    transferred = model.load(str(weights))
    telemetry = LiteRGTelemetry(sys.argv, weights, data, not args.no_diagnostic_log)
    model.add_callback("on_pretrain_routine_end", telemetry.pretrain_end)
    model.add_callback("on_fit_epoch_end", telemetry.fit_epoch_end)
    print(f"repository={ROOT}", flush=True)
    print(f"ultralytics_source={ULTRALYTICS_ROOT}", flush=True)
    print(f"git_commit={git_output('rev-parse', 'HEAD')}", flush=True)
    print(f"split_status={split_status}", flush=True)
    print(f"pretrained_load={transferred}", flush=True)
    print(f"lite_rg_parameters={EXPECTED_LITERG_PARAMETERS}", flush=True)
    model.train(data=str(data), project=str(project), resume=False, **config)


if __name__ == "__main__":
    main()
