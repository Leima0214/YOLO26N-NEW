#!/usr/bin/env python3
"""Audit pretrained coverage, step-0 identity, and residual-gate gradients."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.nn.tasks import DetectionModel, load_checkpoint  # noqa: E402
from ultralytics.utils.torch_utils import intersect_dicts  # noqa: E402

YAML = ROOT / "ultralytics/cfg/models/26/yolo26n-Paper1-MobileMamba-P3.yaml"
WEIGHTS = ROOT / "yolo26n.pt"


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from tensors(value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensors(item)


def main() -> None:
    source, _ = load_checkpoint(WEIGHTS, device="cpu")
    candidate = DetectionModel(YAML, nc=80, verbose=False)
    matched = intersect_dicts(source.float().state_dict(), candidate.state_dict())
    candidate.load_state_dict(matched, strict=False)

    parameters = dict(candidate.named_parameters())
    matched_parameters = set(matched) & set(parameters)
    coverage = sum(parameters[name].numel() for name in matched_parameters) / sum(
        parameter.numel() for parameter in parameters.values()
    )
    gate = candidate.model[4].enhance.scale.gamma

    assert len(candidate.model) == 24
    assert coverage >= 0.96, coverage
    assert torch.count_nonzero(gate) == 0

    source.eval()
    candidate.eval()
    sample = torch.randn(1, 3, 128, 128)
    with torch.inference_mode():
        source_outputs = list(tensors(source(sample)))
        candidate_outputs = list(tensors(candidate(sample)))
    assert len(source_outputs) == len(candidate_outputs)
    assert all(torch.equal(left, right) for left, right in zip(source_outputs, candidate_outputs))

    candidate.train()
    loss = sum(output.float().mean() for output in tensors(candidate(sample)))
    loss.backward()
    assert gate.grad is not None and torch.isfinite(gate.grad).all() and torch.count_nonzero(gate.grad)

    print(
        f"PASS layers=24 transferred={len(matched)}/{len(candidate.state_dict())} "
        f"parameter_coverage={coverage:.2%} exact_identity=True gate_gradient=True"
    )


if __name__ == "__main__":
    main()
