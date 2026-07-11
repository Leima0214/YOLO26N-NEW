"""Space-to-depth downsampling modules."""

import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv


def _pad_even(x):
    pad_h, pad_w = x.shape[-2] % 2, x.shape[-1] % 2
    return F.pad(x, (0, pad_w, 0, pad_h)) if pad_h or pad_w else x


class space_to_depth(nn.Module):
    """Legacy space-to-depth transform with deterministic odd-size padding."""

    def __init__(self, dimension=1):
        super().__init__()
        if dimension != 1:
            raise ValueError("space_to_depth only supports channel dimension 1")

    def forward(self, x):
        return F.pixel_unshuffle(_pad_even(x), 2)


class SPDConv(nn.Module):
    """Single-layer SPD downsampling that preserves downstream YAML indices."""

    def __init__(self, c1, c2, act=True):
        super().__init__()
        self.proj = Conv(4 * c1, c2, 1, 1, act=act)

    def forward(self, x):
        return self.proj(F.pixel_unshuffle(_pad_even(x), 2))


__all__ = ("SPDConv", "space_to_depth")
