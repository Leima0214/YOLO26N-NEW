#!/usr/bin/env python3
"""Audit the Paper 1 localization-loss candidates without training."""

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils.loss import BboxLoss, bbox_regression_iou_loss, bounded_elongation_penalty
from ultralytics.utils.metrics import shape_iou


BASELINE_YAML = ROOT / "ultralytics/cfg/models/26/yolo26.yaml"
SHAPE_IOU_YAML = ROOT / "ultralytics/cfg/models/26/yolo26n-Paper1-ShapeIoU.yaml"
BOUNDED_SHAPE_YAML = ROOT / "ultralytics/cfg/models/26/yolo26n-Paper1-BoundedShapeLoss.yaml"
CHECKPOINT = ROOT / "yolo26n.pt"


def batch(empty: bool = False, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Return one synthetic detection batch."""
    if empty:
        return {
            "img": torch.randn(1, 3, 64, 64, device=device),
            "batch_idx": torch.empty(0, device=device),
            "cls": torch.empty(0, 1, device=device),
            "bboxes": torch.empty(0, 4, device=device),
        }
    return {
        "img": torch.randn(1, 3, 64, 64, device=device),
        "batch_idx": torch.tensor([0], device=device),
        "cls": torch.tensor([[1.0]], device=device),
        "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.1]], device=device),
    }


def assert_architecture_and_transfer(baseline, candidate_yolo) -> None:
    """Require baseline-identical tensors and exact checkpoint transfer."""
    candidate = candidate_yolo.model
    assert baseline.state_dict().keys() == candidate.state_dict().keys()
    for key, value in baseline.state_dict().items():
        assert value.shape == candidate.state_dict()[key].shape, key
    pretrained = YOLO(CHECKPOINT).model.state_dict()
    candidate_yolo.load(CHECKPOINT)
    candidate_state = candidate.state_dict()
    assert pretrained.keys() == candidate_state.keys()
    for key, value in pretrained.items():
        assert torch.allclose(value.float(), candidate_state[key].float(), atol=0.0, rtol=0.0), key


def audit_shape_iou_math() -> None:
    """Check the published Shape-IoU candidate's coordinate handling and gradients."""
    for invalid_scale in (-1.0, float("nan")):
        try:
            BboxLoss(shape_iou_scale=invalid_scale)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid Shape-IoU scale accepted: {invalid_scale}")

    xyxy = torch.tensor([[0.0, 0.0, 10.0, 2.0]], requires_grad=True)
    target_xyxy = torch.tensor([[1.0, 0.0, 11.0, 2.0]])
    xywh = torch.tensor([[5.0, 1.0, 10.0, 2.0]])
    target_xywh = torch.tensor([[6.0, 1.0, 10.0, 2.0]])
    score_xyxy = shape_iou(xyxy, target_xyxy, xywh=False, scale=1.0)
    score_xywh = shape_iou(xywh, target_xywh, xywh=True, scale=1.0)
    assert torch.allclose(score_xyxy, score_xywh, atol=1e-6, rtol=1e-6)
    (1.0 - score_xyxy).sum().backward()
    assert xyxy.grad is not None and torch.isfinite(xyxy.grad).all()


def audit_bounded_shape_math() -> None:
    """Check bounds, symmetry, thin boxes, and exact lambda-zero recovery."""
    for invalid_weight in (-0.1, 1.1, float("nan")):
        try:
            BboxLoss(elongation_penalty_weight=invalid_weight)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid elongation weight accepted: {invalid_weight}")
    try:
        BboxLoss(shape_iou_scale=1.0, elongation_penalty_weight=0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("two localization candidates were enabled together")

    horizontal_target = torch.tensor([[0.0, 0.0, 10.0, 1.0]])
    horizontal_pred = torch.tensor([[0.0, 0.0, 9.0, 1.2]], requires_grad=True)
    vertical_target = torch.tensor([[0.0, 0.0, 1.0, 10.0]])
    vertical_pred = torch.tensor([[0.0, 0.0, 1.2, 9.0]])
    horizontal = bounded_elongation_penalty(horizontal_pred, horizontal_target)
    vertical = bounded_elongation_penalty(vertical_pred, vertical_target)
    assert torch.allclose(horizontal, vertical, atol=1e-6, rtol=1e-6)
    assert 0.0 <= horizontal.item() < 1.0
    assert bounded_elongation_penalty(horizontal_target, horizontal_target).item() == 0.0

    square_target = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
    square_pred = torch.tensor([[0.0, 0.0, 3.0, 1.0]])
    assert bounded_elongation_penalty(square_pred, square_target).item() == 0.0

    thin_target = torch.tensor([[0.0, 0.0, 10.0, 1e-4]])
    thin_pred = torch.tensor([[0.0, 0.0, 9.5, 2e-4]], requires_grad=True)
    thin_loss = bounded_elongation_penalty(thin_pred, thin_target)
    thin_loss.sum().backward()
    assert torch.isfinite(thin_loss).all() and torch.isfinite(thin_pred.grad).all()

    baseline_pred = horizontal_pred.detach().clone().requires_grad_(True)
    zero_pred = horizontal_pred.detach().clone().requires_grad_(True)
    baseline_loss = bbox_regression_iou_loss(baseline_pred, horizontal_target)
    zero_loss = bbox_regression_iou_loss(zero_pred, horizontal_target, elongation_penalty_weight=0.0)
    assert torch.equal(baseline_loss, zero_loss)
    baseline_grad = torch.autograd.grad(baseline_loss.sum(), baseline_pred)[0]
    zero_grad = torch.autograd.grad(zero_loss.sum(), zero_pred)[0]
    assert torch.equal(baseline_grad, zero_grad)


def audit_full_loss(candidate) -> bool:
    """Check finite target/empty-target loss, gradients, and CUDA AMP when available."""
    candidate.args = get_cfg()
    candidate.train()
    loss, loss_items = candidate.loss(batch())
    loss.sum().backward()
    assert torch.isfinite(loss_items).all()
    assert all(torch.isfinite(p.grad).all() for p in candidate.parameters() if p.grad is not None)
    candidate.zero_grad(set_to_none=True)
    empty_loss, empty_items = candidate.loss(batch(empty=True))
    assert torch.isfinite(empty_loss).all() and torch.isfinite(empty_items).all()

    if not torch.cuda.is_available():
        return False
    candidate = candidate.cuda()
    candidate.criterion = None
    candidate.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        amp_loss, amp_items = candidate.loss(batch(device="cuda"))
    amp_loss.sum().backward()
    assert torch.isfinite(amp_items).all()
    assert all(torch.isfinite(p.grad).all() for p in candidate.parameters() if p.grad is not None)
    return True


def main() -> None:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"required checkpoint not found: {CHECKPOINT}")
    audit_shape_iou_math()
    audit_bounded_shape_math()

    baseline = YOLO(BASELINE_YAML, task="detect").model
    baseline.args = get_cfg()
    baseline_criterion = baseline.init_criterion()
    baseline_losses = (baseline_criterion.one2many.bbox_loss, baseline_criterion.one2one.bbox_loss)
    assert all(loss.shape_iou_scale is None and loss.elongation_penalty_weight == 0.0 for loss in baseline_losses)

    shape_iou_yolo = YOLO(SHAPE_IOU_YAML, task="detect")
    assert_architecture_and_transfer(baseline, shape_iou_yolo)
    shape_iou_yolo.model.args = get_cfg()
    shape_criterion = shape_iou_yolo.model.init_criterion()
    shape_losses = (shape_criterion.one2many.bbox_loss, shape_criterion.one2one.bbox_loss)
    assert all(loss.shape_iou_scale == 1.0 and loss.elongation_penalty_weight == 0.0 for loss in shape_losses)

    bounded_yolo = YOLO(BOUNDED_SHAPE_YAML, task="detect")
    assert_architecture_and_transfer(baseline, bounded_yolo)
    cuda_amp = audit_full_loss(bounded_yolo.model)
    bounded_criterion = bounded_yolo.model.criterion
    bounded_losses = (bounded_criterion.one2many.bbox_loss, bounded_criterion.one2one.bbox_loss)
    assert all(loss.shape_iou_scale is None and loss.elongation_penalty_weight == 0.1 for loss in bounded_losses)
    print(f"PASS: A1/A2 math, lambda-zero recovery, full/empty loss, 708/708 transfer, cuda_amp={cuda_amp}")


if __name__ == "__main__":
    main()
