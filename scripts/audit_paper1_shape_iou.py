#!/usr/bin/env python3
"""Audit the Paper 1 Shape-IoU candidate without training."""

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.metrics import shape_iou


MODEL_YAML = ROOT / "ultralytics/cfg/models/26/yolo26n-Paper1-ShapeIoU.yaml"
BASELINE_YAML = ROOT / "ultralytics/cfg/models/26/yolo26.yaml"
CHECKPOINT = ROOT / "yolo26n.pt"


def main() -> None:
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

    baseline = YOLO(BASELINE_YAML, task="detect").model
    candidate_yolo = YOLO(MODEL_YAML, task="detect")
    candidate = candidate_yolo.model
    assert baseline.state_dict().keys() == candidate.state_dict().keys()
    for key, value in baseline.state_dict().items():
        assert value.shape == candidate.state_dict()[key].shape, key
    baseline.args = get_cfg()
    baseline_criterion = baseline.init_criterion()
    baseline_losses = (baseline_criterion.one2many.bbox_loss, baseline_criterion.one2one.bbox_loss)
    assert all(loss.shape_iou_scale is None for loss in baseline_losses)

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"required checkpoint not found: {CHECKPOINT}")
    pretrained = YOLO(CHECKPOINT).model.state_dict()
    candidate_yolo.load(CHECKPOINT)
    candidate_state = candidate.state_dict()
    assert pretrained.keys() == candidate_state.keys()
    for key, value in pretrained.items():
        assert torch.allclose(value.float(), candidate_state[key].float(), atol=0.0, rtol=0.0), key

    candidate.args = get_cfg()
    batch = {
        "img": torch.randn(1, 3, 64, 64),
        "batch_idx": torch.tensor([0]),
        "cls": torch.tensor([[1.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.1]]),
    }
    candidate.train()
    loss, loss_items = candidate.loss(batch)
    loss.sum().backward()
    assert torch.isfinite(loss_items).all()
    assert all(torch.isfinite(p.grad).all() for p in candidate.parameters() if p.grad is not None)

    criterion = candidate.criterion
    losses = (criterion.one2many.bbox_loss, criterion.one2one.bbox_loss)
    assert all(isinstance(loss, BboxLoss) and loss.shape_iou_scale == 1.0 for loss in losses)
    print("PASS: Shape-IoU math, full loss/backward, baseline architecture, 708/708 transfer, and E2E selection")


if __name__ == "__main__":
    main()
