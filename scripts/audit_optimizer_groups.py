#!/usr/bin/env python3
"""Verify that MuSGD 3x-LR groups follow Detect modules instead of hard-coded layer indices."""

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.engine.trainer import BaseTrainer, get_musgd_high_lr_parameter_names  # noqa: E402
from ultralytics.nn.tasks import DetectionModel  # noqa: E402

CASES = (
    ("ultralytics/cfg/models/26/yolo26.yaml", 23),
    ("ultralytics/cfg/models/26/yolo26-MobileMamba-Backbone.yaml", 18),
)


def audit(model_yaml: str, detect_index: int) -> None:
    model = DetectionModel(model_yaml, nc=7, verbose=False)
    names = set(get_musgd_high_lr_parameter_names(model))
    prefixes = (f"model.{detect_index}.cv3.", f"model.{detect_index}.one2one_cv3.")
    assert names and all(name.startswith(prefixes) for name in names), sorted(names)

    trainer = object.__new__(BaseTrainer)
    trainer.args = SimpleNamespace(lr0=0.01, momentum=0.937, warmup_bias_lr=0.1, warmup_momentum=0.8)
    trainer.data = {"nc": 7}
    optimizer = trainer.build_optimizer(model, name="auto", lr=0.01, momentum=0.937, iterations=3960)
    optimizer_momenta = {group["momentum"] for group in optimizer.param_groups if "momentum" in group}
    parameter_lrs = {id(parameter): group["lr"] for group in optimizer.param_groups for parameter in group["params"]}
    parameters = dict(model.named_parameters())
    base_lr = round(0.002 * 5 / 11, 6)

    assert optimizer_momenta == {0.9}
    assert trainer.args.momentum == 0.937
    assert all(parameter_lrs[id(parameters[name])] == base_lr * 3 for name in names)
    ordinary_cv3 = [name for name in parameters if ".cv3." in name and name not in names]
    assert all(parameter_lrs[id(parameters[name])] == base_lr for name in ordinary_cv3)
    print(
        f"{model_yaml}: Detect=model.{detect_index}, high_lr_parameters={len(names)}, "
        f"ordinary_cv3_parameters={len(ordinary_cv3)}, lr={base_lr * 3:.6f}, "
        f"auto_initial_momentum=0.9, warmup_target_momentum={trainer.args.momentum}"
    )


if __name__ == "__main__":
    for case in CASES:
        audit(*case)
