"""Identity-initialized content-aware feature reassembly."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CARAFE(nn.Module):
    """CARAFE upsampling with nearest-neighbor behavior at initialization."""

    def __init__(self, c1, c2, kernel_size=3, up_factor=2, max_workspace_bytes=768 * 1024**2):
        super().__init__()
        if not isinstance(kernel_size, int) or not 1 <= kernel_size <= 7 or kernel_size % 2 == 0:
            raise ValueError(f"CARAFE kernel_size must be an odd integer in [1, 7], got {kernel_size}")
        if not isinstance(up_factor, int) or not 2 <= up_factor <= 4:
            raise ValueError(f"CARAFE up_factor must be an integer in [2, 4], got {up_factor}")
        if not isinstance(max_workspace_bytes, int) or max_workspace_bytes < 1:
            raise ValueError("CARAFE max_workspace_bytes must be a positive integer")
        hidden = max(1, c1 // 4)
        self.kernel_size = kernel_size
        self.up_factor = up_factor
        self.max_workspace_bytes = max_workspace_bytes
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
        kernel_elements = self.kernel_size**2
        scale_elements = self.up_factor**2
        workspace_elements = n * h * w * (
            c * kernel_elements + kernel_elements * scale_elements + 3 * c * scale_elements
        )
        workspace_bytes = workspace_elements * x.element_size()
        if workspace_bytes > self.max_workspace_bytes:
            raise RuntimeError(
                f"CARAFE workspace estimate {workspace_bytes / 1024**2:.1f} MiB exceeds "
                f"limit {self.max_workspace_bytes / 1024**2:.1f} MiB"
            )
        logits = self.encoder(self.down(x))
        kernels = logits.reshape(n, kernel_elements, self.up_factor, self.up_factor, h, w).softmax(dim=1)
        patches = F.unfold(x, self.kernel_size, padding=self.kernel_size // 2)
        patches = patches.reshape(n, c, kernel_elements, h, w)
        reassembled = torch.einsum("nckhw,nkijhw->ncijhw", patches, kernels)
        reassembled = reassembled.permute(0, 1, 4, 2, 5, 3).reshape(
            n, c, h * self.up_factor, w * self.up_factor
        )
        nearest = F.interpolate(x, scale_factor=self.up_factor, mode="nearest")
        mixed = nearest + torch.tanh(self.gamma) * (reassembled - nearest)
        return self.out(mixed)


__all__ = ("CARAFE",)
