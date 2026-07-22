"""Remote-GPU preflight for Progressive LiteRG-YOLO26."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import get_flops, intersect_dicts


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "ultralytics/cfg/models/26/yolo26n.yaml"
CANDIDATE = ROOT / "ultralytics/cfg/models/26/yolo26n-literg.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=str(ROOT / "yolo26n.pt"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)
    device = torch.device(args.device)

    baseline = YOLO(str(BASELINE))
    candidate = YOLO(str(CANDIDATE))
    baseline_state = baseline.model.state_dict()
    candidate_state = candidate.model.state_dict()
    transferred = intersect_dicts(baseline_state, candidate_state)
    missing_baseline = sorted(set(baseline_state) - set(transferred))
    new_items = sorted(set(candidate_state) - set(transferred))
    assert not missing_baseline, f"B0 keys lost or reshaped: {missing_baseline[:10]}"
    assert new_items and all(name.startswith("lite_rg.") for name in new_items), new_items[:10]

    baseline.load(args.weights)
    candidate.load(args.weights)
    shared_pretrained = intersect_dicts(baseline.model.state_dict(), candidate.model.state_dict())
    candidate.model.load_state_dict(shared_pretrained, strict=False)

    japan7 = DetectionModel(str(CANDIDATE), ch=3, nc=7, verbose=False)
    japan7_loaded = intersect_dicts(candidate.model.state_dict(), japan7.state_dict())
    japan7_missing = sorted(set(japan7.state_dict()) - set(japan7_loaded))
    unexpected_japan7 = [
        name
        for name in japan7_missing
        if not name.startswith(("model.23.cv3.", "model.23.one2one_cv3."))
    ]
    assert not unexpected_japan7, unexpected_japan7[:10]

    baseline.model.to(device).eval()
    candidate.model.to(device).eval()
    image = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)
    with torch.no_grad():
        baseline_decoded = baseline.model(image)[0]
        candidate_decoded = candidate.model(image)[0]
    denominator = baseline_decoded.float().square().sum().clamp_min(1e-12)
    zero_init_delta = (
        (baseline_decoded.float() - candidate_decoded.float()).square().sum() / denominator
    ).sqrt().item()
    assert zero_init_delta < 1e-7, zero_init_delta

    candidate.model.train()
    candidate.model.zero_grad(set_to_none=True)
    train_predictions = candidate.model(image)
    assert set(train_predictions) == {"one2many", "one2one", "region_logits"}
    assert all(feature.requires_grad for feature in train_predictions["one2many"]["feats"])
    assert all(not feature.requires_grad for feature in train_predictions["one2one"]["feats"])
    batch = {
        "img": image,
        "batch_idx": torch.tensor([0], device=device),
        "cls": torch.tensor([[0.0]], device=device),
        "bboxes": torch.tensor([[0.5, 0.5, 0.35, 0.06]], device=device),
    }
    loss, loss_items = candidate.model.loss(batch, train_predictions)
    assert loss.shape == loss_items.shape == (4,)
    assert torch.isfinite(loss).all() and torch.isfinite(loss_items).all()
    loss.sum().backward()
    prior_gradient = sum(
        parameter.grad.detach().abs().sum().item()
        for name, parameter in candidate.model.named_parameters()
        if name.startswith("lite_rg.prior.") and parameter.grad is not None
    )
    scale_gradients = {
        name: parameter.grad.detach().abs().sum().item()
        for name, parameter in candidate.model.named_parameters()
        if name.startswith("lite_rg.") and name.rsplit(".", 1)[-1] in {"gamma3", "gamma4", "eta3", "eta4"}
    }
    assert prior_gradient > 0.0, prior_gradient
    assert len(scale_gradients) == 4 and all(torch.isfinite(torch.tensor(v)) for v in scale_gradients.values())

    print(f"baseline_transfer={len(transferred)}/{len(baseline_state)}")
    print(f"literg_new_state_items={len(new_items)}")
    print(f"japan7_transfer={len(japan7_loaded)}/{len(japan7.state_dict())}")
    print(f"japan7_expected_classifier_missing={len(japan7_missing)}")
    print("japan7_unexpected_missing=0")
    print(f"baseline_parameters={sum(p.numel() for p in baseline.model.parameters())}")
    print(f"candidate_parameters={sum(p.numel() for p in candidate.model.parameters())}")
    print(f"literg_parameters={sum(p.numel() for p in candidate.model.lite_rg.parameters())}")
    print(f"baseline_gflops={get_flops(baseline.model, args.imgsz):.3f}")
    print(f"candidate_gflops={get_flops(candidate.model, args.imgsz):.3f}")
    print(f"zero_init_relative_output_delta={zero_init_delta:.10f}")
    print(f"region_logits_shape={tuple(train_predictions['region_logits'].shape)}")
    print(f"loss_items={loss_items.detach().cpu().tolist()}")
    print(f"prior_gradient_abs_sum={prior_gradient:.8f}")
    print(f"scale_gradient_abs_sum={scale_gradients}")
    print("shared_features=one2many_grad one2one_detached")
    print("forward=ok backward=ok soft_region_loss=ok")


if __name__ == "__main__":
    main()
