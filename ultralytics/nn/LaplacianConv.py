"""Transfer-friendly Laplacian detail convolution."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import autopad
from ultralytics.utils.torch_utils import fuse_conv_and_bn


class LaplacianConv(nn.Module):
    """Standard Conv-BN-activation with a bounded, identity-initialized edge residual."""

    def __init__(
        self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1, act=True, alpha_init=0.0
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            autopad(kernel_size, padding),
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act else nn.Identity()
        kernel = torch.tensor([[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]])
        self.register_buffer("laplacian_kernel", kernel.reshape(1, 1, 3, 3), persistent=False)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def _enhance(self, x):
        channels = x.shape[1]
        kernel = self.laplacian_kernel.to(dtype=x.dtype).expand(channels, 1, 3, 3)
        edge = F.conv2d(x, kernel, padding=1, groups=channels)
        return x + 0.1 * torch.tanh(self.alpha) * edge

    def forward(self, x):
        return self.act(self.bn(self.conv(self._enhance(x))))

    def forward_fuse(self, x):
        return self.act(self.conv(self._enhance(x)))

    def fuse_convs(self):
        """Fuse the transfer-compatible Conv-BN path for deployment."""
        if hasattr(self, "bn"):
            self.conv = fuse_conv_and_bn(self.conv, self.bn)
            delattr(self, "bn")
            self.forward = self.forward_fuse


__all__ = ("LaplacianConv",)
