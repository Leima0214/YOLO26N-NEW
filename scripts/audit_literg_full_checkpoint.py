#!/usr/bin/env python3
"""Audit a full LiteRG checkpoint without updating or rewriting its weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from copy import deepcopy
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ultralytics  # noqa: E402

ULTRALYTICS_ROOT = Path(ultralytics.__file__).resolve().parent
assert ULTRALYTICS_ROOT.is_relative_to(ROOT), f"Imported ultralytics outside repository: {ULTRALYTICS_ROOT}"

from ultralytics import YOLO  # noqa: E402
from ultralytics.models.yolo.detect import DetectionTrainer  # noqa: E402
from ultralytics.utils import YAML  # noqa: E402
from ultralytics.utils.torch_utils import de_parallel  # noqa: E402


SCALARS = ("gamma3", "gamma4", "eta3", "eta4")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def extract_scalars(core: torch.nn.Module) -> dict[str, float]:
    lite_rg = getattr(core, "lite_rg", None)
    if lite_rg is None:
        raise RuntimeError("Checkpoint does not contain an active LiteRG module.")
    return {name: float(getattr(lite_rg, name).detach().float().cpu()) for name in SCALARS}


def decoded(output):
    return output[0] if isinstance(output, tuple) else output


def grad_norm(parameters) -> float | None:
    squared = [parameter.grad.detach().float().square().sum() for parameter in parameters if parameter.grad is not None]
    return float(torch.stack(squared).sum().sqrt().cpu()) if squared else None


def gradient_groups(core: torch.nn.Module) -> dict[str, list[torch.nn.Parameter]]:
    named = list(core.named_parameters())

    def select(predicate):
        return [parameter for name, parameter in named if predicate(name)]

    groups = {
        "prior": select(lambda name: name.startswith("lite_rg.prior.")),
        "drg3": select(lambda name: name.startswith("lite_rg.drg3.")),
        "drg4": select(lambda name: name.startswith("lite_rg.drg4.")),
        "rff3": select(lambda name: name.startswith("lite_rg.rff3.")),
        "rff4": select(lambda name: name.startswith("lite_rg.rff4.")),
        "backbone": select(lambda name: name.startswith("model.") and int(name.split(".")[1]) <= 10),
        "neck": select(lambda name: name.startswith("model.") and 11 <= int(name.split(".")[1]) <= 22),
        "one_to_many_head": select(lambda name: name.startswith("model.23.cv2.") or name.startswith("model.23.cv3.")),
        "one_to_one_head": select(lambda name: name.startswith("model.23.one2one_")),
    }
    for scalar in SCALARS:
        groups[scalar] = select(lambda name, scalar=scalar: name == f"lite_rg.{scalar}")
    return groups


def snapshot_gradients(core: torch.nn.Module, groups: dict[str, list[torch.nn.Parameter]]) -> dict[str, float | None]:
    return {name: grad_norm(parameters) for name, parameters in groups.items()}


def make_trainer(core: torch.nn.Module, checkpoint: Path, data: Path, device: str, batch_size: int):
    trainer = DetectionTrainer(
        overrides={
            "model": str(checkpoint),
            "data": str(data),
            "imgsz": 640,
            "batch": batch_size,
            "device": device,
            "workers": 0,
            "epochs": 30,
            "optimizer": "auto",
            "seed": 42,
            "deterministic": True,
            "plots": False,
        }
    )
    trainer.model = core.to(trainer.device)
    trainer.set_model_attributes()
    trainer.stride = max(int(core.stride.max()), 32)
    loader = trainer.get_dataloader(trainer.data["train"], batch_size=batch_size, rank=-1, mode="train")
    return trainer, next(iter(loader))


def run_gradient_audit(core: torch.nn.Module, trainer, batch: dict) -> dict:
    core.train()
    core.criterion = None
    batch = trainer.preprocess_batch(batch)
    groups = gradient_groups(core)

    def backward(kind: str) -> dict:
        core.zero_grad(set_to_none=True)
        predictions = core(batch["img"])
        criterion = core.criterion or core.init_criterion()
        core.criterion = criterion
        parsed = criterion.one2many.parse_output(predictions)
        if kind == "one_to_one_only":
            objective = criterion.one2one.loss(parsed["one2one"], batch)[0].sum()
        elif kind == "region_only":
            objective = criterion.region_component(parsed, batch, schedule=criterion.o2m)[0].sum()
        elif kind == "full":
            objective = core.loss(batch, predictions)[0].sum()
        else:
            raise ValueError(kind)
        objective.backward()
        return {"objective": float(objective.detach().cpu()), "grad_norms": snapshot_gradients(core, groups)}

    one_to_one = backward("one_to_one_only")
    region = backward("region_only")
    full = backward("full")
    shared_o2o = [one_to_one["grad_norms"][name] for name in ("backbone", "neck", "prior", "drg3", "drg4", "rff3", "rff4")]
    return {
        "batch_size": int(batch["img"].shape[0]),
        "one_to_one_only": one_to_one,
        "region_only": region,
        "full_loss": full,
        "detach_semantics": {
            "shared_feature_grads_are_zero_or_none": all(value in {None, 0.0} for value in shared_o2o),
            "one_to_one_head_has_gradient": (one_to_one["grad_norms"]["one_to_one_head"] or 0.0) > 0,
            "region_updates_prior": (region["grad_norms"]["prior"] or 0.0) > 0,
            "region_updates_backbone": (region["grad_norms"]["backbone"] or 0.0) > 0,
            "full_updates_drg_rff": all((full["grad_norms"][name] or 0.0) > 0 for name in ("drg3", "drg4", "rff3", "rff4")),
        },
    }


def optimizer_audit(core: torch.nn.Module, trainer, epochs: int, dataset_size: int, batch_size: int) -> dict:
    iterations = math.ceil(dataset_size / max(batch_size, trainer.args.nbs)) * epochs
    optimizer = trainer.build_optimizer(
        core,
        name="auto",
        lr=trainer.args.lr0,
        momentum=trainer.args.momentum,
        decay=trainer.args.weight_decay * batch_size * max(round(trainer.args.nbs / batch_size), 1) / trainer.args.nbs,
        iterations=iterations,
    )
    parameter_names = {id(parameter): name for name, parameter in core.named_parameters()}
    categories = {
        "prior": lambda name: name.startswith("lite_rg.prior."),
        "drg3": lambda name: name.startswith("lite_rg.drg3."),
        "drg4": lambda name: name.startswith("lite_rg.drg4."),
        "rff3": lambda name: name.startswith("lite_rg.rff3."),
        "rff4": lambda name: name.startswith("lite_rg.rff4."),
        "gamma_eta": lambda name: name in {f"lite_rg.{scalar}" for scalar in SCALARS},
        "backbone": lambda name: name.startswith("model.") and int(name.split(".")[1]) <= 10,
        "neck": lambda name: name.startswith("model.") and 11 <= int(name.split(".")[1]) <= 22,
        "detect": lambda name: name.startswith("model.23."),
    }
    output = {}
    for category, predicate in categories.items():
        rows = []
        for group_index, group in enumerate(optimizer.param_groups):
            selected = [p for p in group["params"] if predicate(parameter_names[id(p)])]
            if not selected:
                continue
            rows.append(
                {
                    "optimizer_group": group_index,
                    "param_group": group.get("param_group"),
                    "parameter_tensors": len(selected),
                    "parameters": sum(parameter.numel() for parameter in selected),
                    "initial_lr": float(group["lr"]),
                    "theoretical_final_lr": float(group["lr"] * trainer.args.lrf),
                    "weight_decay": float(group.get("weight_decay", 0.0)),
                    "use_muon": bool(group.get("use_muon", False)),
                    "frozen_parameters": sum(parameter.numel() for parameter in selected if not parameter.requires_grad),
                }
            )
        output[category] = rows
    return {"epochs": epochs, "iterations": iterations, "groups": output}


def checkpoint_payload(path: Path) -> dict:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    model = raw.get("ema") or raw.get("model")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
        "epoch": raw.get("epoch"),
        "git": raw.get("git"),
        "date": raw.get("date"),
        "optimizer_present": raw.get("optimizer") is not None,
        "scaler_present": raw.get("scaler") is not None,
        "ema_present": raw.get("ema") is not None,
        "updates": raw.get("updates"),
        "lite_rg_state_keys": sum(name.startswith("lite_rg.") for name in model.state_dict()),
        "scalars": extract_scalars(model),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--last", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--expected-fused-b0", type=int, default=2_376_201)
    args = parser.parse_args()

    for path in (args.checkpoint, args.last, args.data):
        if path is not None and not path.exists():
            raise FileNotFoundError(path)
    yolo = YOLO(str(args.checkpoint))
    core = de_parallel(yolo.model)
    lite_rg = getattr(core, "lite_rg", None)
    if lite_rg is None:
        raise RuntimeError("LiteRG is missing after independent checkpoint load.")
    state = core.state_dict()
    lite_keys = [name for name in state if name.startswith("lite_rg.")]
    head = core.model[-1]
    config = core.yaml.get("lite_rg", {})
    input_tensor = torch.rand(1, 3, 640, 640, device=next(core.parameters()).device)
    core.eval()
    with torch.no_grad():
        unfused_output = core(input_tensor)
    raw = unfused_output[1] if isinstance(unfused_output, tuple) else {}
    region_logits = raw.get("region_logits") if isinstance(raw, dict) else None

    fused = deepcopy(core).eval()
    fused.fuse(verbose=False)
    with torch.no_grad():
        fused_output = fused(input_tensor)
    difference = (decoded(unfused_output).float() - decoded(fused_output).float()).abs()
    fused_state = fused.state_dict()
    lite_parameters = count_parameters(lite_rg)
    fused_parameters = count_parameters(fused)
    payload = {
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "ultralytics_source": str(ULTRALYTICS_ROOT),
        },
        "best": checkpoint_payload(args.checkpoint),
        "last": checkpoint_payload(args.last) if args.last else None,
        "unfused": {
            "parameters": count_parameters(core),
            "lite_rg_parameters": lite_parameters,
            "lite_rg_state_keys": len(lite_keys),
            "required_state_prefixes": {
                prefix: any(name.startswith(f"lite_rg.{prefix}") for name in lite_keys)
                for prefix in ("prior.", "drg3.", "drg4.", "rff3.", "rff4.")
            },
            "scalars": extract_scalars(core),
            "one_to_many_present": getattr(head, "cv2", None) is not None and getattr(head, "cv3", None) is not None,
            "one_to_one_present": hasattr(head, "one2one_cv2") and hasattr(head, "one2one_cv3"),
            "lite_rg_enabled": config.get("enabled") is True,
            "lite_rg_config": config,
            "region_logits_shape": list(region_logits.shape) if region_logits is not None else None,
            "region_logits_finite": bool(torch.isfinite(region_logits).all()) if region_logits is not None else False,
        },
        "fused": {
            "parameters": fused_parameters,
            "lite_rg_parameters": count_parameters(fused.lite_rg),
            "lite_rg_state_keys": sum(name.startswith("lite_rg.") for name in fused_state),
            "lite_rg_present": getattr(fused, "lite_rg", None) is not None,
            "one_to_many_removed": getattr(fused.model[-1], "cv2", None) is None
            and getattr(fused.model[-1], "cv3", None) is None,
            "one_to_one_present": hasattr(fused.model[-1], "one2one_cv2") and hasattr(fused.model[-1], "one2one_cv3"),
            "expected_arithmetic": {
                "fused_b0": args.expected_fused_b0,
                "lite_rg": lite_parameters,
                "sum": args.expected_fused_b0 + lite_parameters,
                "matches": fused_parameters == args.expected_fused_b0 + lite_parameters,
            },
            "prediction_difference": {
                "max_abs": float(difference.max().cpu()),
                "mean_abs": float(difference.mean().cpu()),
                "all_finite": bool(torch.isfinite(decoded(fused_output)).all()),
            },
        },
    }
    if args.data:
        trainer, batch = make_trainer(core, args.checkpoint, args.data, args.device, args.batch)
        payload["gradient_audit"] = run_gradient_audit(core, trainer, batch)
        dataset_size = len(trainer.get_dataloader(trainer.data["train"], 1, -1, "val").dataset)
        payload["optimizer_30e"] = optimizer_audit(core, trainer, 30, dataset_size, 32)
        payload["optimizer_100e"] = optimizer_audit(core, trainer, 100, dataset_size, 32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
