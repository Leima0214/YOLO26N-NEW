#!/usr/bin/env python3
"""Adversarially audit the Paper 1 localization-loss candidates without training."""

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils.loss import (
    BboxLoss,
    bbox_regression_iou_loss,
    bounded_elongation_components,
    bounded_elongation_penalty,
)
from ultralytics.utils.metrics import shape_iou


BASELINE_YAML = ROOT / "ultralytics/cfg/models/26/yolo26.yaml"
SHAPE_IOU_YAML = ROOT / "ultralytics/cfg/models/26/yolo26n-Paper1-ShapeIoU.yaml"
BOUNDED_SHAPE_YAML = ROOT / "ultralytics/cfg/models/26/yolo26n-Paper1-BoundedShapeLoss.yaml"
CHECKPOINT = ROOT / "yolo26n.pt"
CHECKPOINT_SHA256 = "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"
REPORT_JSON = ROOT / "experiments/module_scan/paper1_a2_adversarial_audit.json"
REPORT_MD = ROOT / "experiments/module_scan/paper1_a2_adversarial_audit.md"

CHECKS: list[str] = []
DETAILS: dict[str, object] = {}


def require(condition, message: str) -> None:
    """Record a passed check or raise even when Python runs with -O."""
    if isinstance(condition, torch.Tensor):
        condition = bool(condition.detach().all().item())
    if not condition:
        raise AssertionError(message)
    CHECKS.append(message)


def require_raises(error_types, fn, message: str) -> None:
    """Require a callable to raise one of the expected exceptions."""
    try:
        fn()
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


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def write_report(status: str, error: str = "") -> None:
    report = {
        "status": status,
        "error": error,
        "git_commit": git_commit(),
        "loss_py_sha256": sha256(ROOT / "ultralytics/utils/loss.py"),
        "a2_yaml_sha256": sha256(BOUNDED_SHAPE_YAML),
        "checkpoint_sha256": sha256(CHECKPOINT) if CHECKPOINT.is_file() else "missing",
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable",
        "checks_passed": len(CHECKS),
        "checks": CHECKS,
        "details": DETAILS,
        "ddp_verified": False,
    }
    atomic_write(REPORT_JSON, json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Paper 1 A2 Adversarial Audit",
        "",
        f"- status: `{status}`",
        f"- git commit: `{report['git_commit']}`",
        f"- checks passed: `{len(CHECKS)}`",
        f"- checkpoint SHA256: `{report['checkpoint_sha256']}`",
        f"- loss.py SHA256: `{report['loss_py_sha256']}`",
        f"- A2 YAML SHA256: `{report['a2_yaml_sha256']}`",
        f"- torch/CUDA: `{report['torch']}` / `{report['cuda']}`",
        f"- CUDA device: `{report['cuda_device']}`",
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


def synthetic_batch(
    imgsz: int, device: str, mode: str = "mixed", dtype: torch.dtype = torch.float32
) -> dict[str, torch.Tensor]:
    """Build deterministic single, empty, or two-image mixed-AR targets."""
    images = torch.linspace(0, 1, 2 * 3 * imgsz * imgsz, device=device, dtype=dtype).reshape(2, 3, imgsz, imgsz)
    if mode == "empty":
        return {
            "img": images[:1],
            "batch_idx": torch.empty(0, device=device),
            "cls": torch.empty(0, 1, device=device),
            "bboxes": torch.empty(0, 4, device=device),
        }
    if mode == "single":
        return {
            "img": images[:1],
            "batch_idx": torch.tensor([0], device=device),
            "cls": torch.tensor([[1.0]], device=device),
            "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.1]], device=device),
        }
    aspect_ratios = (1.0, 3.0, 5.0, 10.0, 50.0)
    widths = torch.tensor([0.20, 0.30, 0.35, 0.40, 0.50], device=device)
    heights = widths / torch.tensor(aspect_ratios, device=device)
    centers_x = torch.tensor([0.15, 0.32, 0.50, 0.68, 0.82], device=device)
    bboxes = torch.stack((centers_x, torch.full_like(centers_x, 0.5), widths, heights), dim=1)
    return {
        "img": images,
        "batch_idx": torch.ones(len(aspect_ratios), device=device),
        "cls": torch.tensor([[0.0], [1.0], [1.0], [2.0], [3.0]], device=device),
        "bboxes": bboxes,
    }


def assert_architecture_and_transfer(baseline, candidate_yolo, name: str) -> None:
    candidate = candidate_yolo.model
    baseline_state, candidate_state = baseline.state_dict(), candidate.state_dict()
    require(baseline_state.keys() == candidate_state.keys(), f"{name}: state names match baseline")
    require(
        all(baseline_state[key].shape == candidate_state[key].shape for key in baseline_state),
        f"{name}: state shapes match baseline",
    )
    pretrained = YOLO(CHECKPOINT).model.state_dict()
    candidate_yolo.load(CHECKPOINT)
    candidate_state = candidate.state_dict()
    require(pretrained.keys() == candidate_state.keys(), f"{name}: checkpoint has 708 matching items")
    require(
        all(torch.equal(value.float(), candidate_state[key].float()) for key, value in pretrained.items()),
        f"{name}: all transferred tensors are bitwise equal",
    )


def audit_invalid_inputs() -> None:
    valid = torch.tensor([[0.0, 0.0, 2.0, 1.0]])
    cases = {
        "broadcast shape": lambda: bounded_elongation_penalty(valid.repeat(2, 1), valid),
        "rank/last dimension": lambda: bounded_elongation_penalty(torch.ones(2, 3), torch.ones(2, 3)),
        "integer dtype": lambda: bounded_elongation_penalty(valid.long(), valid.long()),
        "bool dtype": lambda: bounded_elongation_penalty(valid.bool(), valid.bool()),
        "pred NaN": lambda: bounded_elongation_penalty(valid + torch.tensor([[float("nan"), 0, 0, 0]]), valid),
        "pred +Inf": lambda: bounded_elongation_penalty(valid + torch.tensor([[0, 0, float("inf"), 0]]), valid),
        "pred -Inf": lambda: bounded_elongation_penalty(valid + torch.tensor([[float("-inf"), 0, 0, 0]]), valid),
        "target NaN": lambda: bounded_elongation_penalty(valid, valid + torch.tensor([[float("nan"), 0, 0, 0]])),
        "target +Inf": lambda: bounded_elongation_penalty(valid, valid + torch.tensor([[0, 0, float("inf"), 0]])),
        "target -Inf": lambda: bounded_elongation_penalty(valid, valid + torch.tensor([[float("-inf"), 0, 0, 0]])),
        "zero pred width": lambda: bounded_elongation_penalty(torch.tensor([[0.0, 0.0, 0.0, 1.0]]), valid),
        "negative target height": lambda: bounded_elongation_penalty(valid, torch.tensor([[0.0, 1.0, 2.0, 0.0]])),
    }
    for name, call in cases.items():
        require_raises((TypeError, ValueError), call, f"invalid input rejected: {name}")
    mixed_dtype = valid.half().requires_grad_(True)
    mixed_penalty = bounded_elongation_penalty(mixed_dtype, valid.float())
    mixed_penalty.sum().backward()
    require(mixed_penalty.dtype == torch.float32, "mixed floating dtypes promote to float32")
    require(torch.isfinite(mixed_dtype.grad).all(), "mixed floating dtype gradients are finite")
    if torch.cuda.is_available():
        require_raises(
            ValueError,
            lambda: bounded_elongation_penalty(valid.cuda(), valid),
            "invalid input rejected: device mismatch",
        )


def audit_shape_iou_math() -> None:
    for invalid_scale in (-1.0, float("nan"), True):
        require_raises(
            (TypeError, ValueError),
            lambda value=invalid_scale: BboxLoss(shape_iou_scale=value),
            f"invalid Shape-IoU scale rejected: {invalid_scale}",
        )
    xyxy = torch.tensor([[0.0, 0.0, 10.0, 2.0]], requires_grad=True)
    target_xyxy = torch.tensor([[1.0, 0.0, 11.0, 2.0]])
    xywh = torch.tensor([[5.0, 1.0, 10.0, 2.0]])
    target_xywh = torch.tensor([[6.0, 1.0, 10.0, 2.0]])
    score_xyxy = shape_iou(xyxy, target_xyxy, xywh=False, scale=1.0)
    score_xywh = shape_iou(xywh, target_xywh, xywh=True, scale=1.0)
    require(torch.allclose(score_xyxy, score_xywh, atol=1e-6, rtol=1e-6), "Shape-IoU xyxy/xywh match")
    (1.0 - score_xyxy).sum().backward()
    require(xyxy.grad is not None and torch.isfinite(xyxy.grad).all(), "Shape-IoU gradients are finite")


def audit_extreme_penalty() -> None:
    for invalid_weight in (-0.1, 1.1, float("nan"), True):
        require_raises(
            (TypeError, ValueError),
            lambda value=invalid_weight: BboxLoss(elongation_penalty_weight=value),
            f"invalid A2 weight rejected: {invalid_weight}",
        )
    require_raises(
        ValueError,
        lambda: BboxLoss(shape_iou_scale=1.0, elongation_penalty_weight=0.1),
        "A1 and A2 cannot be enabled together",
    )

    device = "cuda"
    require(torch.cuda.is_available(), "CUDA is available for adversarial FP16 audit")
    tested, skipped = 0, 0
    penalties = []
    for aspect_ratio in (1.0, 3.0, 5.0, 10.0, 50.0, 1e3, 1e6):
        for short_edge in (1e-7, 1e-5, 1e-3, 1.0, 1e3):
            long_edge = aspect_ratio * short_edge
            if long_edge * 1.5 >= torch.finfo(torch.float16).max:
                skipped += 4
                continue
            for factor in (0.5, 1.5):
                target_h = torch.tensor([[0.0, 0.0, long_edge, short_edge]], device=device, dtype=torch.float16)
                pred_h = torch.tensor(
                    [[0.0, 0.0, long_edge * factor, short_edge / factor]],
                    device=device,
                    dtype=torch.float16,
                    requires_grad=True,
                )
                target_v = target_h[:, [1, 0, 3, 2]].detach().clone()
                pred_v = pred_h.detach()[:, [1, 0, 3, 2]].clone().requires_grad_(True)
                horizontal = bounded_elongation_penalty(pred_h, target_h)
                vertical = bounded_elongation_penalty(pred_v, target_v)
                require(torch.isfinite(horizontal) and torch.isfinite(vertical), "extreme FP16 penalty is finite")
                require(torch.equal(horizontal, vertical), "extreme FP16 horizontal/vertical symmetry")
                (horizontal.sum() + vertical.sum()).backward()
                require(torch.isfinite(pred_h.grad).all() and torch.isfinite(pred_v.grad).all(), "extreme FP16 gradients finite")
                penalties.append(float(horizontal.item()))
                tested += 2
    require(tested > 0, "extreme FP16 matrix executed")
    DETAILS["fp16_extreme_matrix"] = {
        "tested_orientations": tested,
        "skipped_unrepresentable": skipped,
        "penalty_min": min(penalties),
        "penalty_max": max(penalties),
    }


def criterion_losses(criterion, preds, targets):
    parsed = criterion.one2many.parse_output(preds)
    one2many = criterion.one2many.loss(parsed["one2many"], targets)[0]
    one2one = criterion.one2one.loss(parsed["one2one"], targets)[0]
    total = one2many * criterion.o2m + one2one * criterion.o2o
    return one2many, one2one, total


def require_exact_tensors(left: torch.Tensor, right: torch.Tensor, message: str) -> None:
    require(left.dtype == right.dtype and torch.equal(left, right), message)


def audit_lambda_zero_case(model, baseline_criterion, zero_criterion, targets, amp: bool, name: str) -> None:
    model.zero_grad(set_to_none=True)
    context = torch.autocast("cuda", dtype=torch.float16) if amp else torch.autocast("cpu", enabled=False)
    with context:
        preds = model(targets["img"])
        base_o2m, base_o2o, base_total = criterion_losses(baseline_criterion, preds, targets)
        zero_o2m, zero_o2o, zero_total = criterion_losses(zero_criterion, preds, targets)
    require_exact_tensors(base_o2m, zero_o2m, f"lambda0 {name}: one2many box/cls/dfl exact")
    require_exact_tensors(base_o2o, zero_o2o, f"lambda0 {name}: one2one box/cls/dfl exact")
    require_exact_tensors(base_total, zero_total, f"lambda0 {name}: total loss exact")
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    base_grads = torch.autograd.grad(base_total.sum(), parameters, retain_graph=True, allow_unused=True)
    zero_grads = torch.autograd.grad(zero_total.sum(), parameters, allow_unused=True)
    require(
        all(
            (left is None and right is None)
            or (left is not None and right is not None and torch.equal(left, right))
            for left, right in zip(base_grads, zero_grads)
        ),
        f"lambda0 {name}: every parameter gradient exact",
    )


def audit_lambda_zero_full() -> None:
    baseline_yolo = YOLO(BASELINE_YAML, task="detect")
    baseline_yolo.load(CHECKPOINT)
    model = baseline_yolo.model
    model.args = get_cfg()
    model.train()
    baseline_criterion = model.init_criterion()

    zero_yolo = YOLO(BASELINE_YAML, task="detect")
    zero_yolo.load(CHECKPOINT)
    zero_yolo.model.yaml["elongation_penalty_weight"] = 0.0
    zero_yolo.model.args = get_cfg()
    zero_criterion = zero_yolo.model.init_criterion()
    require(
        all(
            loss.elongation_penalty_weight == 0.0
            for loss in (zero_criterion.one2many.bbox_loss, zero_criterion.one2one.bbox_loss)
        ),
        "lambda0 control explicitly configures both E2E branches",
    )

    audit_lambda_zero_case(model, baseline_criterion, zero_criterion, synthetic_batch(64, "cpu", "single"), False, "CPU single")
    audit_lambda_zero_case(model, baseline_criterion, zero_criterion, synthetic_batch(64, "cpu", "empty"), False, "CPU empty")
    audit_lambda_zero_case(model, baseline_criterion, zero_criterion, synthetic_batch(64, "cpu", "mixed"), False, "CPU mixed")

    model = model.cuda()
    model.criterion = None
    model.args = get_cfg()
    baseline_criterion = model.init_criterion()
    zero_yolo.model = zero_yolo.model.cuda()
    zero_yolo.model.args = get_cfg()
    zero_criterion = zero_yolo.model.init_criterion()
    audit_lambda_zero_case(
        model,
        baseline_criterion,
        zero_criterion,
        synthetic_batch(640, "cuda", "mixed", torch.float32),
        True,
        "CUDA AMP 640 batch2 mixed",
    )


def audit_a2_full_loss(baseline) -> None:
    bounded_yolo = YOLO(BOUNDED_SHAPE_YAML, task="detect")
    assert_architecture_and_transfer(baseline, bounded_yolo, "A2")
    candidate = bounded_yolo.model.cuda()
    candidate.args = get_cfg()
    candidate.train()
    candidate.criterion = candidate.init_criterion()
    losses = (candidate.criterion.one2many.bbox_loss, candidate.criterion.one2one.bbox_loss)
    for bbox_loss in losses:
        bbox_loss.collect_diagnostics_once = True
    mixed = synthetic_batch(640, "cuda", "mixed")
    with torch.autocast("cuda", dtype=torch.float16):
        loss, items = candidate.loss(mixed)
    loss.sum().backward()
    require(torch.isfinite(items), "A2 CUDA AMP 640 batch2 mixed loss finite")
    require(
        all(torch.isfinite(parameter.grad).all() for parameter in candidate.parameters() if parameter.grad is not None),
        "A2 CUDA AMP 640 batch2 mixed gradients finite",
    )
    candidate.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        empty_loss, empty_items = candidate.loss(synthetic_batch(640, "cuda", "empty"))
    empty_loss.sum().backward()
    require(torch.isfinite(empty_items), "A2 empty-target loss finite")
    require(
        all(torch.isfinite(parameter.grad).all() for parameter in candidate.parameters() if parameter.grad is not None),
        "A2 empty-target backward gradients finite",
    )
    require(
        all(loss.shape_iou_scale is None and loss.elongation_penalty_weight == 0.1 for loss in losses),
        "A2 one2many and one2one both use CIoU plus weight 0.1",
    )
    required = {
        "positive_count",
        "base_ciou_mean",
        "bounded_penalty_mean",
        "weighted_penalty_to_ciou",
        "base_ciou_grad_norm",
        "penalty_grad_norm",
        "grad_norm_ratio",
        "penalty_mean_by_ar",
    }
    require(
        all(loss.last_diagnostics and required <= loss.last_diagnostics.keys() for loss in losses),
        "A2 diagnostics captured for both E2E branches",
    )
    for name, loss in zip(("one2many", "one2one"), losses):
        diagnostics = loss.last_diagnostics
        require(diagnostics["positive_count"] > 0, f"A2 {name} diagnostic has assigned positives")
        require(
            all(math.isfinite(value) for value in diagnostics.values() if isinstance(value, (int, float))),
            f"A2 {name} diagnostic values are finite",
        )
        DETAILS[f"a2_{name}_assigned_positive_probe"] = diagnostics


def audit_penalty_contribution() -> None:
    targets = torch.tensor(
        [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 3.0, 1.0], [0.0, 0.0, 5.0, 1.0], [0.0, 0.0, 10.0, 1.0], [0.0, 0.0, 50.0, 1.0]],
        requires_grad=False,
    )
    predictions = (targets * torch.tensor([1.0, 1.0, 0.9, 1.1])).requires_grad_(True)
    aspect_error, gate, penalty = bounded_elongation_components(predictions, targets)
    base = bbox_regression_iou_loss(predictions, targets)
    base_grad = torch.autograd.grad(base.mean(), predictions, retain_graph=True)[0]
    penalty_grad = torch.autograd.grad((0.1 * penalty).mean(), predictions)[0]
    DETAILS["synthetic_penalty_contribution"] = {
        "base_ciou_mean": float(base.mean().item()),
        "aspect_error_mean": float(aspect_error.mean().item()),
        "elongation_gate_mean": float(gate.mean().item()),
        "bounded_penalty_mean": float(penalty.mean().item()),
        "weighted_penalty_mean": float((0.1 * penalty).mean().item()),
        "weighted_penalty_to_ciou": float((0.1 * penalty).mean().item() / max(base.mean().item(), 1e-12)),
        "base_grad_norm": float(base_grad.norm().item()),
        "penalty_grad_norm": float(penalty_grad.norm().item()),
        "grad_norm_ratio": float(penalty_grad.norm().item() / max(base_grad.norm().item(), 1e-12)),
        "gate_by_ar": {str(ar): float(value) for ar, value in zip((1, 3, 5, 10, 50), gate.flatten())},
    }


def run() -> None:
    require(CHECKPOINT.is_file(), "trusted yolo26n checkpoint exists")
    require(sha256(CHECKPOINT) == CHECKPOINT_SHA256, "trusted yolo26n checkpoint SHA256 matches")
    audit_invalid_inputs()
    audit_shape_iou_math()
    audit_extreme_penalty()
    audit_penalty_contribution()

    baseline = YOLO(BASELINE_YAML, task="detect").model
    baseline.args = get_cfg()
    baseline_criterion = baseline.init_criterion()
    baseline_losses = (baseline_criterion.one2many.bbox_loss, baseline_criterion.one2one.bbox_loss)
    require(
        all(loss.shape_iou_scale is None and loss.elongation_penalty_weight == 0.0 for loss in baseline_losses),
        "baseline remains CIoU with zero elongation penalty",
    )
    shape_iou_yolo = YOLO(SHAPE_IOU_YAML, task="detect")
    assert_architecture_and_transfer(baseline, shape_iou_yolo, "A1")
    audit_lambda_zero_full()
    audit_a2_full_loss(baseline)


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
