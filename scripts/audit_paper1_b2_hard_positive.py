#!/usr/bin/env python3
"""Adversarially audit Paper 1 B2 quality-aware hard-positive weighting without training."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audit_paper1_shape_iou import criterion_losses, synthetic_batch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils.loss import quality_hard_positive_weights

BASELINE_YAML = ROOT / "ultralytics/cfg/models/26/yolo26.yaml"
B2_YAML = ROOT / "ultralytics/cfg/models/26/yolo26n-Paper1-B2-QualityHardPositive.yaml"
CHECKPOINT = ROOT / "yolo26n.pt"
CHECKPOINT_SHA256 = "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"
REPORT_JSON = ROOT / "experiments/module_scan/paper1_b2_adversarial_audit.json"
REPORT_MD = ROOT / "experiments/module_scan/paper1_b2_adversarial_audit.md"

CHECKS: list[str] = []
DETAILS: dict[str, object] = {}


def require(condition, message: str) -> None:
    if isinstance(condition, torch.Tensor):
        condition = bool(condition.detach().all().item())
    if not condition:
        raise AssertionError(message)
    CHECKS.append(message)


def require_raises(error_types, function, message: str) -> None:
    try:
        function()
    except error_types:
        CHECKS.append(message)
        return
    raise AssertionError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def write_report(status: str, error: str = "") -> None:
    report = {
        "status": status,
        "error": error,
        "git_commit": git_commit(),
        "checks_passed": len(CHECKS),
        "checks": CHECKS,
        "details": DETAILS,
        "checkpoint_sha256": sha256(CHECKPOINT) if CHECKPOINT.is_file() else "missing",
        "loss_py_sha256": sha256(ROOT / "ultralytics/utils/loss.py"),
        "b2_yaml_sha256": sha256(B2_YAML),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable",
        "cuda_amp_verified": torch.cuda.is_available(),
        "ddp_verified": False,
    }
    atomic_write(REPORT_JSON, json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Paper 1 B2 Adversarial Audit",
        "",
        f"- status: `{status}`",
        f"- git commit: `{report['git_commit']}`",
        f"- checks passed: `{len(CHECKS)}`",
        f"- checkpoint SHA256: `{report['checkpoint_sha256']}`",
        f"- loss.py SHA256: `{report['loss_py_sha256']}`",
        f"- B2 YAML SHA256: `{report['b2_yaml_sha256']}`",
        f"- torch/CUDA: `{report['torch']}` / `{report['cuda']}`",
        f"- CUDA device: `{report['cuda_device']}`",
        f"- CUDA AMP verified: `{report['cuda_amp_verified']}`",
        "- DDP: `not verified; formal protocol is single GPU`",
        "",
        "## Numerical Details",
        "",
        "```json",
        json.dumps(DETAILS, indent=2, sort_keys=True),
        "```",
        "",
        "## Checks",
        "",
        *[f"- PASS: {item}" for item in CHECKS],
    ]
    if error:
        lines.extend(("", "## Error", "", f"`{error}`"))
    atomic_write(REPORT_MD, "\n".join(lines) + "\n")


def audit_source_and_yaml() -> None:
    for path in (ROOT / "ultralytics/utils/loss.py", ROOT / "scripts/train_module_pilot.py", Path(__file__)):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        forbidden = [
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec"}
        ]
        require(not forbidden, f"{path.name}: no eval or exec calls")
    with BASELINE_YAML.open("r", encoding="utf-8") as handle:
        baseline = yaml.safe_load(handle)
    with B2_YAML.open("r", encoding="utf-8") as handle:
        candidate = yaml.safe_load(handle)
    require(candidate.get("hard_positive_cls_weight") == 0.25, "B2 YAML pins hard-positive weight 0.25")
    candidate.pop("hard_positive_cls_weight")
    require(candidate == baseline, "B2 YAML architecture and all baseline settings are unchanged")


def audit_weight_math(device: str) -> None:
    logits = torch.tensor([-1.3862944, 1.3862944, -1.3862944], device=device)
    pred_scores = torch.zeros(1, 3, 2, device=device)
    pred_scores[0, :, 0] = logits
    pred_scores.requires_grad_(True)
    target_scores = torch.zeros_like(pred_scores)
    target_scores[0, :, 0] = 1.0
    target_boxes = torch.tensor([[[0.0, 0.0, 2.0, 1.0]] * 3], device=device)
    pred_boxes = target_boxes.clone()
    pred_boxes[0, 2] = torch.tensor([0.0, 0.0, 0.5, 1.0], device=device)
    foreground = torch.ones(1, 3, dtype=torch.bool, device=device)
    weights, quality, confidence, boost = quality_hard_positive_weights(
        pred_scores, target_scores, pred_boxes, target_boxes, foreground, 0.25
    )
    positive_weights = weights[0, :, 0]
    require(not weights.requires_grad, f"{device}: adaptive weights are detached")
    require(torch.equal(weights[0, :, 1], torch.ones(3, device=device)), f"{device}: negative classes unchanged")
    require(weights.min() >= 1.0 and weights.max() <= 1.25, f"{device}: weights bounded to [1, 1.25]")
    require(positive_weights[0] > positive_weights[1], f"{device}: lower-confidence match receives more weight")
    require(positive_weights[2] < positive_weights[0], f"{device}: poor localization suppresses the boost")
    require(torch.isfinite(quality).all() and torch.isfinite(confidence).all(), f"{device}: factors are finite")
    loss = (F.binary_cross_entropy_with_logits(pred_scores, target_scores, reduction="none") * weights).sum()
    loss.backward()
    require(torch.isfinite(pred_scores.grad).all(), f"{device}: weighted BCE gradients are finite")
    empty = torch.zeros_like(foreground)
    empty_weights, empty_quality, _, _ = quality_hard_positive_weights(
        pred_scores.detach(), target_scores, pred_boxes, target_boxes, empty, 0.25
    )
    require(torch.equal(empty_weights, torch.ones_like(empty_weights)), f"{device}: empty foreground is baseline BCE")
    require(empty_quality.numel() == 0, f"{device}: empty foreground diagnostics are empty")
    DETAILS[f"{device}_synthetic_probe"] = {
        "quality": [float(value) for value in quality.flatten()],
        "confidence": [float(value) for value in confidence.flatten()],
        "boost": [float(value) for value in boost.flatten()],
        "positive_weights": [float(value) for value in positive_weights],
    }


def audit_invalid_strengths() -> None:
    scores = torch.zeros(1, 1, 1)
    boxes = torch.tensor([[[0.0, 0.0, 1.0, 1.0]]])
    foreground = torch.ones(1, 1, dtype=torch.bool)
    for value in (True, -0.1, 1.1, float("nan"), float("inf")):
        require_raises(
            (TypeError, ValueError),
            lambda value=value: quality_hard_positive_weights(scores, scores, boxes, boxes, foreground, value),
            f"invalid hard-positive strength rejected: {value}",
        )


def assert_architecture_and_transfer(baseline, candidate_yolo) -> None:
    candidate = candidate_yolo.model
    baseline_state, candidate_state = baseline.state_dict(), candidate.state_dict()
    require(baseline_state.keys() == candidate_state.keys(), "B2 state names match baseline")
    require(
        all(baseline_state[key].shape == candidate_state[key].shape for key in baseline_state),
        "B2 state shapes match baseline",
    )
    pretrained = YOLO(CHECKPOINT).model.state_dict()
    candidate_yolo.load(CHECKPOINT)
    candidate_state = candidate.state_dict()
    require(pretrained.keys() == candidate_state.keys(), "B2 checkpoint has 708 matching items")
    require(
        all(torch.equal(value.float(), candidate_state[key].float()) for key, value in pretrained.items()),
        "B2 transferred tensors are bitwise equal",
    )


def audit_zero_case(model, baseline_criterion, zero_criterion, batch, amp: bool, name: str) -> None:
    model.zero_grad(set_to_none=True)
    context = torch.autocast("cuda", dtype=torch.float16) if amp else torch.autocast("cpu", enabled=False)
    with context:
        predictions = model(batch["img"])
        base_o2m, base_o2o, base_total = criterion_losses(baseline_criterion, predictions, batch)
        zero_o2m, zero_o2o, zero_total = criterion_losses(zero_criterion, predictions, batch)
    require(torch.equal(base_o2m, zero_o2m), f"weight0 {name}: one2many exact")
    require(torch.equal(base_o2o, zero_o2o), f"weight0 {name}: one2one exact")
    require(torch.equal(base_total, zero_total), f"weight0 {name}: total exact")
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    base_grads = torch.autograd.grad(base_total.sum(), parameters, retain_graph=True, allow_unused=True)
    zero_grads = torch.autograd.grad(zero_total.sum(), parameters, allow_unused=True)
    require(
        all(
            (left is None and right is None)
            or (left is not None and right is not None and torch.equal(left, right))
            for left, right in zip(base_grads, zero_grads)
        ),
        f"weight0 {name}: every parameter gradient exact",
    )


def audit_zero_equivalence() -> None:
    baseline_yolo = YOLO(BASELINE_YAML, task="detect")
    baseline_yolo.load(CHECKPOINT)
    model = baseline_yolo.model
    model.args = get_cfg()
    model.train()
    baseline_criterion = model.init_criterion()
    zero_yolo = YOLO(B2_YAML, task="detect")
    zero_yolo.load(CHECKPOINT)
    zero_yolo.model.yaml["hard_positive_cls_weight"] = 0.0
    zero_yolo.model.args = get_cfg()
    zero_criterion = zero_yolo.model.init_criterion()
    require(
        zero_criterion.one2many.hard_positive_cls_weight == 0.0
        and zero_criterion.one2one.hard_positive_cls_weight == 0.0,
        "weight0 config reaches both E2E branches",
    )
    for mode in ("single", "empty", "mixed"):
        audit_zero_case(
            model,
            baseline_criterion,
            zero_criterion,
            synthetic_batch(64, "cpu", mode),
            False,
            f"CPU {mode}",
        )
    if torch.cuda.is_available():
        model = model.cuda()
        model.criterion = None
        model.args = get_cfg()
        baseline_criterion = model.init_criterion()
        zero_yolo.model = zero_yolo.model.cuda()
        zero_yolo.model.args = get_cfg()
        zero_criterion = zero_yolo.model.init_criterion()
        audit_zero_case(
            model,
            baseline_criterion,
            zero_criterion,
            synthetic_batch(640, "cuda", "mixed"),
            True,
            "CUDA AMP 640 batch2 mixed",
        )


def audit_candidate(baseline) -> None:
    candidate_yolo = YOLO(B2_YAML, task="detect")
    assert_architecture_and_transfer(baseline, candidate_yolo)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    candidate = candidate_yolo.model.to(device)
    candidate.args = get_cfg()
    candidate.train()
    candidate.criterion = candidate.init_criterion()
    branches = (candidate.criterion.one2many, candidate.criterion.one2one)
    require(all(branch.hard_positive_cls_weight == 0.25 for branch in branches), "B2 weight reaches both E2E branches")
    for branch in branches:
        branch.collect_cls_diagnostics_once = True
    image_size = 640 if device == "cuda" else 64
    batch = synthetic_batch(image_size, device, "mixed")
    with (torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else torch.autocast("cpu", enabled=False)):
        loss, items = candidate.loss(batch)
    loss.sum().backward()
    require(torch.isfinite(items).all(), "B2 mixed-target loss is finite")
    require(
        all(torch.isfinite(parameter.grad).all() for parameter in candidate.parameters() if parameter.grad is not None),
        "B2 mixed-target gradients are finite",
    )
    required = {
        "positive_count",
        "base_bce",
        "added_bce",
        "added_to_base_ratio",
        "quality_mean",
        "correct_confidence_mean",
        "boost_mean",
        "boost_max",
        "base_grad_norm",
        "added_grad_norm",
        "grad_norm_ratio",
    }
    for name, branch in zip(("one2many", "one2one"), branches):
        diagnostics = branch.last_cls_diagnostics
        require(diagnostics is not None and required <= diagnostics.keys(), f"B2 {name} diagnostics captured")
        require(diagnostics["positive_count"] > 0, f"B2 {name} diagnostics contain positives")
        require(0.0 < diagnostics["boost_max"] <= 0.25, f"B2 {name} boost is active and bounded")
        require(
            all(math.isfinite(value) for value in diagnostics.values() if isinstance(value, (int, float))),
            f"B2 {name} diagnostics are finite",
        )
        DETAILS[f"b2_{name}_assigned_positive_probe"] = diagnostics
    candidate.zero_grad(set_to_none=True)
    with (torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else torch.autocast("cpu", enabled=False)):
        empty_loss, empty_items = candidate.loss(synthetic_batch(image_size, device, "empty"))
    empty_loss.sum().backward()
    require(torch.isfinite(empty_items).all(), "B2 empty-target loss is finite")
    require(
        all(torch.isfinite(parameter.grad).all() for parameter in candidate.parameters() if parameter.grad is not None),
        "B2 empty-target gradients are finite",
    )


def run() -> None:
    require(CHECKPOINT.is_file(), "trusted yolo26n checkpoint exists")
    require(sha256(CHECKPOINT) == CHECKPOINT_SHA256, "trusted yolo26n checkpoint SHA256 matches")
    audit_source_and_yaml()
    audit_invalid_strengths()
    audit_weight_math("cpu")
    if torch.cuda.is_available():
        with torch.autocast("cuda", dtype=torch.float16):
            audit_weight_math("cuda")
    audit_zero_equivalence()
    baseline = YOLO(BASELINE_YAML, task="detect").model
    baseline.args = get_cfg()
    baseline_criterion = baseline.init_criterion()
    require(
        baseline_criterion.one2many.hard_positive_cls_weight == 0.0
        and baseline_criterion.one2one.hard_positive_cls_weight == 0.0,
        "baseline remains unweighted BCE in both E2E branches",
    )
    audit_candidate(baseline)


def main() -> None:
    try:
        run()
    except Exception as error:
        write_report("FAIL", f"{type(error).__name__}: {error}")
        raise
    write_report("PASS")
    print(f"PASS: checks={len(CHECKS)} report={REPORT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
