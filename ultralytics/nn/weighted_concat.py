"""Lightweight weighted concatenation without the legacy v9 dependency chain."""

import torch
import torch.nn as nn


class Concat_bifpn(nn.Module):
    """Identity-initialized positive weighted concatenation for BiFPN-style fusion."""

    def __init__(self, num_inputs=2, dimension=1):
        super().__init__()
        if not isinstance(num_inputs, int) or num_inputs < 2:
            raise ValueError(f"Concat_bifpn requires at least two inputs, got {num_inputs}")
        if dimension != 1:
            raise ValueError("Concat_bifpn only supports channel-dimension concatenation")
        self.num_inputs = num_inputs
        self.d = dimension
        self.w = nn.Parameter(torch.zeros(num_inputs, dtype=torch.float32))

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != self.num_inputs:
            count = len(x) if isinstance(x, (list, tuple)) else type(x).__name__
            raise ValueError(f"Concat_bifpn expected {self.num_inputs} tensors, got {count}")
        spatial = x[0].shape[2:]
        if any(tensor.shape[2:] != spatial for tensor in x[1:]):
            raise ValueError(f"Concat_bifpn spatial shapes must match, got {[tuple(t.shape) for t in x]}")
        positive = torch.nn.functional.softplus(self.w)
        weights = positive / positive.mean()
        return torch.cat([weight.to(tensor.dtype) * tensor for weight, tensor in zip(weights, x)], self.d)


__all__ = ("Concat_bifpn",)
