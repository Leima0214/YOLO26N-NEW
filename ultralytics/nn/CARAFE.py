"""Identity-initialized content-aware feature reassembly."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CARAFE(nn.Module):
    """CARAFE upsampling with nearest-neighbor behavior at initialization."""

    def __init__(self, c1, c2, kernel_size=3, up_factor=2):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"CARAFE kernel_size must be positive and odd, got {kernel_size}")
        if not isinstance(up_factor, int) or up_factor < 2:
            raise ValueError(f"CARAFE up_factor must be an integer >= 2, got {up_factor}")
        hidden = max(1, c1 // 4)
        self.kernel_size = kernel_size
        self.up_factor = up_factor
        self.down = nn.Conv2d(c1, hidden, 1)
        self.encoder = nn.Conv2d(hidden, up_factor**2 * kernel_size**2, kernel_size, padding=kernel_size // 2)
        self.out = nn.Conv2d(c1, c2, 1, bias=False)
        if c1 == c2:
            nn.init.dirac_(self.out.weight)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"CARAFE expects BCHW input, got shape {tuple(x.shape)}")
        n, c, h, w = x.shape
        logits = self.encoder(self.down(x))
        kernels = F.pixel_shuffle(logits, self.up_factor).softmax(dim=1)
        patches = F.unfold(x, self.kernel_size, padding=self.kernel_size // 2)
        patches = patches.reshape(n, c, self.kernel_size**2, h, w)
        patches = patches.repeat_interleave(self.up_factor, 3).repeat_interleave(self.up_factor, 4)
        reassembled = (patches * kernels.unsqueeze(1)).sum(2)
        nearest = F.interpolate(x, scale_factor=self.up_factor, mode="nearest")
        mixed = nearest + torch.tanh(self.gamma) * (reassembled - nearest)
        return self.out(mixed)


__all__ = ("CARAFE",)
