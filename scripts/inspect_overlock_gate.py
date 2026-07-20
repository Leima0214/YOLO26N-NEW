"""Inspect learned LayerScale gates in an OverLoCK checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    model = checkpoint.get("ema") or checkpoint.get("model")
    if model is None:
        raise RuntimeError("Checkpoint contains neither an EMA model nor a training model")

    found = False
    for name, module in model.named_modules():
        gamma = getattr(getattr(module, "scale", None), "gamma", None)
        if gamma is None or "overlock" not in module.__class__.__name__.lower():
            continue
        values = gamma.detach().float().flatten()
        print(
            f"{name}.scale.gamma: count={values.numel()} "
            f"mean={values.mean().item():.8f} "
            f"mean_abs={values.abs().mean().item():.8f} "
            f"min={values.min().item():.8f} "
            f"max={values.max().item():.8f}"
        )
        mix_logit = getattr(module, "mix_logit", None)
        if mix_logit is not None:
            print(
                f"{name}.mix: local={mix_logit.detach().float().sigmoid().item():.8f} "
                f"overview={(-mix_logit.detach().float()).sigmoid().item():.8f}"
            )
        found = True

    if not found:
        raise RuntimeError("No OverLoCK LayerScale gate found")


if __name__ == "__main__":
    main()
