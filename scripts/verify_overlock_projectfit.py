"""Verify that a zero-gated ProjectFit model safely extends pretrained YOLO26n."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.torch_utils import intersect_dicts  # noqa: E402


def tensors(value) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from tensors(value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensors(item)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--weights", default="yolo26n.pt")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--max-relative-output-diff", type=float, default=0.0)
    args = parser.parse_args()

    torch.manual_seed(42)
    baseline = YOLO("ultralytics/cfg/models/26/yolo26.yaml")
    baseline.load(args.weights)
    candidate = YOLO(args.model)
    candidate.load(args.weights)

    baseline_state = baseline.model.state_dict()
    candidate_state = candidate.model.state_dict()
    loaded = intersect_dicts(baseline_state, candidate_state)
    new_keys = sorted(set(candidate_state) - set(loaded))
    expected_prefix = f"model.{args.layer}.enhance."
    assert new_keys and all(key.startswith(expected_prefix) for key in new_keys), new_keys[:10]
    assert all(torch.equal(candidate_state[key], value) for key, value in loaded.items())

    baseline.model.eval()
    candidate.model.eval()
    sample = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        baseline_outputs = list(tensors(baseline.model(sample)))
        candidate_outputs = list(tensors(candidate.model(sample)))
    assert len(baseline_outputs) == len(candidate_outputs)
    max_diff = max(
        (left.float() - right.float()).abs().max().item()
        for left, right in zip(baseline_outputs, candidate_outputs, strict=True)
    )
    relative_diff = sum(
        (left.float() - right.float()).abs().mean().item()
        for left, right in zip(baseline_outputs, candidate_outputs, strict=True)
    ) / sum(left.float().abs().mean().item() for left in baseline_outputs)
    assert relative_diff <= args.max_relative_output_diff, relative_diff

    candidate.model.train()
    candidate.model.zero_grad(set_to_none=True)
    first_loss = sum(output.float().square().mean() for output in tensors(candidate.model(sample)))
    first_loss.backward()
    enhance = candidate.model.model[args.layer].enhance
    gammas = [parameter for name, parameter in enhance.named_parameters() if name.endswith("scale.gamma")]
    assert gammas
    zero_gate = all(torch.count_nonzero(gamma.detach()).item() == 0 for gamma in gammas)
    first_gamma_grad = sum(gamma.grad.detach().abs().sum().item() for gamma in gammas)
    branch_parameters = [
        parameter for name, parameter in enhance.named_parameters() if not name.endswith("scale.gamma")
    ]
    first_branch_grad = sum(
        parameter.grad.detach().abs().sum().item() for parameter in branch_parameters if parameter.grad is not None
    )
    assert first_gamma_grad > 0.0
    assert (first_branch_grad == 0.0) if zero_gate else (first_branch_grad > 0.0)
    mix_logit = getattr(enhance, "mix_logit", None)
    first_mix_grad = None if mix_logit is None else mix_logit.grad.detach().abs().item()
    if first_mix_grad is not None:
        assert first_mix_grad > 0.0

    with torch.no_grad():
        for gamma in gammas:
            gamma.add_(1e-3 * gamma.grad.sign())
    candidate.model.zero_grad(set_to_none=True)
    second_loss = sum(output.float().square().mean() for output in tensors(candidate.model(sample)))
    second_loss.backward()
    second_branch_grad = sum(
        parameter.grad.detach().abs().sum().item() for parameter in branch_parameters if parameter.grad is not None
    )
    assert second_branch_grad > 0.0

    print(f"pretrained_items={len(loaded)}/{len(candidate_state)}")
    print(f"new_items={len(new_keys)}")
    print(f"max_output_diff={max_diff}")
    print(f"relative_output_diff={relative_diff:.8f}")
    print(f"first_gamma_grad={first_gamma_grad:.8f}")
    print(f"first_branch_grad={first_branch_grad:.8f}")
    if first_mix_grad is not None:
        print(f"first_mix_grad={first_mix_grad:.8f}")
    print(f"second_branch_grad={second_branch_grad:.8f}")


if __name__ == "__main__":
    main()
