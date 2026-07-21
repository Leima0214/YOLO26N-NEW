"""Shared Japan7 old-split exploratory 100e protocol for the B0/P4 pair."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from ultralytics import YOLO


WEIGHTS = REPO_ROOT / "yolo26n.pt"
KNOWN_AUDIT_STATUS = "FAIL_CONFIRMED_NEAR_DUPLICATE_LEAKAGE"
DEVELOPMENT_SPLIT_STATUS = "KNOWN_NEAR_DUPLICATE_DEVELOPMENT_SPLIT"


class EpochTelemetry:
    def __init__(self, gate_layer: int | None) -> None:
        self.gate_layer = gate_layer
        self.trainer = None
        self.gammas = []
        self.grad_norms: list[float] = []
        self.output_ratios: list[float] = []

    def epoch_start(self, trainer) -> None:
        self.trainer = trainer
        self.grad_norms.clear()
        self.output_ratios.clear()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def train_start(self, trainer) -> None:
        self.trainer = trainer
        if self.gate_layer is None:
            return
        enhance = trainer.model.model[self.gate_layer].enhance
        self.gammas = [parameter for name, parameter in enhance.named_parameters() if name.endswith("scale.gamma")]
        if not self.gammas:
            raise RuntimeError(f"No LayerScale gate at model.{self.gate_layer}.enhance")
        for parameter in self.gammas:
            parameter.register_hook(self._gate_grad_hook)
        enhance.register_forward_hook(self._output_hook)

    def _gate_grad_hook(self, gradient: torch.Tensor) -> None:
        scale = float(self.trainer.scaler.get_scale()) if self.trainer and self.trainer.amp else 1.0
        self.grad_norms.append(float(gradient.detach().float().norm() / scale))

    def _output_hook(self, module, inputs, output) -> None:
        if not module.training or not inputs:
            return
        base = inputs[0].detach().float()
        contribution = output.detach().float() - base
        self.output_ratios.append(float(contribution.norm() / base.norm().clamp_min(1e-12)))

    def fit_epoch_end(self, trainer) -> None:
        values = torch.cat([gamma.detach().float().flatten().cpu() for gamma in self.gammas]) if self.gammas else None
        path = Path(trainer.save_dir) / "telemetry.csv"
        row = {
            "epoch": trainer.epoch + 1,
            "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0,
            "epoch_seconds": float(trainer.epoch_time),
            "iterations_per_second": len(trainer.train_loader) / max(float(trainer.epoch_time), 1e-12),
            "gate_mean": float(values.mean()) if values is not None else "",
            "gate_std": float(values.std(unbiased=False)) if values is not None else "",
            "gate_min": float(values.min()) if values is not None else "",
            "gate_max": float(values.max()) if values is not None else "",
            "gate_grad_norm_mean": sum(self.grad_norms) / len(self.grad_norms) if self.grad_norms else "",
            "enhance_base_output_norm_ratio": sum(self.output_ratios) / len(self.output_ratios) if self.output_ratios else "",
        }
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow(row)


def resolve_split_status(audit_status: str, allow_known_development_split: bool) -> str:
    if audit_status == "PASS":
        return "PASS"
    if audit_status == KNOWN_AUDIT_STATUS and allow_known_development_split:
        return DEVELOPMENT_SPLIT_STATUS
    raise SystemExit(
        f"Training blocked by dataset audit: {audit_status}. "
        "Use --allow-known-development-split only for the documented exploratory old split."
    )


def record_split_scope(trainer, split_status: str, allow_known_development_split: bool, result_scope: str) -> None:
    save_dir = Path(trainer.save_dir)
    args_path = save_dir / "args.yaml"
    with args_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"split_status: {split_status}\n"
            f"allow_known_development_split: {str(allow_known_development_split).lower()}\n"
            f"result_scope: {result_scope}\n"
        )
    (save_dir / f"{result_scope}_scope.md").write_text(
        f"# {result_scope}\n\n"
        f"- split_status={split_status}\n"
        + (
            "- Contains documented train/val near-duplicate scenes.\n"
            "- Use only for paired long-convergence exploration; not a clean test or final paper benchmark.\n"
            if result_scope.startswith("exploratory_oldsplit")
            else "- Scene-grouped split candidate; report separately from the historical development split.\n"
        ),
        encoding="utf-8",
    )


def run(model_yaml: str, run_name: str, gate_layer: int | None = None, dataset: str = "oldsplit") -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-known-development-split", action="store_true")
    args = parser.parse_args()
    if dataset == "oldsplit":
        data = REPO_ROOT / "configs/japan7_remote.yaml"
        audit_path = REPO_ROOT / "reports/dataset_integrity_and_leakage_audit.json"
        project = REPO_ROOT / "runs/paper1/exploratory_oldsplit"
        result_scope = "exploratory_oldsplit_not_final_benchmark"
    elif dataset == "v2":
        data = REPO_ROOT / "configs/japan7_v2_scene_disjoint/dataset.yaml"
        audit_path = REPO_ROOT / "configs/japan7_v2_scene_disjoint/leakage_audit_v2.json"
        project = REPO_ROOT / "runs/paper1/japan7_v2_scene_disjoint"
        result_scope = "japan7_v2_scene_disjoint_candidate"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    split_status = resolve_split_status(audit.get("status", "MISSING_STATUS"), args.allow_known_development_split)
    if not data.is_file() or not WEIGHTS.is_file():
        raise FileNotFoundError(f"Missing data config or weights: {data}, {WEIGHTS}")

    print(f"split_status={split_status}", flush=True)
    print(f"result_scope={result_scope}", flush=True)

    model = YOLO(str(REPO_ROOT / model_yaml))
    model.load(str(WEIGHTS))
    telemetry = EpochTelemetry(gate_layer)
    model.add_callback(
        "on_pretrain_routine_start",
        lambda trainer: record_split_scope(trainer, split_status, args.allow_known_development_split, result_scope),
    )
    model.add_callback("on_train_start", telemetry.train_start)
    model.add_callback("on_train_epoch_start", telemetry.epoch_start)
    model.add_callback("on_fit_epoch_end", telemetry.fit_epoch_end)
    model.train(
        data=str(data),
        project=str(project.resolve()),
        name=run_name,
        epochs=100,
        patience=1_000_000_000,
        imgsz=640,
        batch=32,
        device=0,
        workers=8,
        seed=42,
        deterministic=True,
        amp=True,
        optimizer="auto",
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
        cos_lr=False,
        resume=False,
    )
