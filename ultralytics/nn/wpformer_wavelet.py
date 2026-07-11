"""WPFormer-WCA-inspired wavelet detail refinement for detection features."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def haar_decompose(x):
    """Return orthonormal Haar LL/LH/HL/HH bands, padding odd edges by replication."""
    if x.ndim != 4:
        raise ValueError(f"Haar decomposition expects BCHW input, got {tuple(x.shape)}")
    pad_h, pad_w = x.shape[-2] % 2, x.shape[-1] % 2
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    return (
        (a + b + c + d) * 0.5,
        (-a + b - c + d) * 0.5,
        (-a - b + c + d) * 0.5,
        (a - b - c + d) * 0.5,
    )


def haar_reconstruct(ll, lh, hl, hh):
    """Invert orthonormal Haar bands without allocating a mutable output tensor."""
    shapes = {tuple(band.shape) for band in (ll, lh, hl, hh)}
    if len(shapes) != 1:
        raise ValueError(f"Haar bands must share one shape, got {sorted(shapes)}")
    a = (ll - lh - hl + hh) * 0.5
    b = (ll + lh - hl - hh) * 0.5
    c = (ll - lh + hl - hh) * 0.5
    d = (ll + lh + hl + hh) * 0.5
    top = torch.stack((a, b), dim=-1).flatten(-2)
    bottom = torch.stack((c, d), dim=-1).flatten(-2)
    return torch.stack((top, bottom), dim=-2).flatten(-3, -2)


class WaveletDetailRefinement(nn.Module):
    """Modulate Haar detail bands with local/global context and refine a feature map.

    This adapts the frequency-modulation idea from WPFormer's WCA to a convolutional
    YOLO feature map. A zero-initialized output projection preserves the pretrained
    baseline exactly while still receiving gradients on the first optimization step.
    """

    def __init__(self, c1, c2=None, reduction=8):
        super().__init__()
        c2 = c1 if c2 is None else c2
        if c1 <= 0 or c2 != c1:
            raise ValueError(f"WaveletDetailRefinement must preserve positive channels, got {c1=}, {c2=}")
        if not isinstance(reduction, int) or reduction <= 0:
            raise ValueError(f"reduction must be a positive integer, got {reduction}")
        hidden = max(4, c1 // reduction)
        self.channels = c1
        self.local_context = nn.Sequential(
            nn.Conv2d(c1, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c1, 1),
        )
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c1, 1),
        )
        self.out_proj = nn.Conv2d(c1, c1, 1, bias=False)
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f"Expected BCHW with {self.channels} channels, got {tuple(x.shape)}")
        height, width = x.shape[-2:]
        ll, lh, hl, hh = haar_decompose(x)
        context = ll + lh + hl + hh
        gate = torch.sigmoid(self.local_context(context) + self.global_context(context))
        refined = haar_reconstruct(ll, gate * lh, gate * hl, gate * hh)[..., :height, :width]
        return x + self.out_proj(refined - x)


__all__ = ("WaveletDetailRefinement", "haar_decompose", "haar_reconstruct")
