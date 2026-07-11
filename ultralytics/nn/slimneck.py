"""Minimal GSConv and slim-neck blocks for YOLO YAML models."""

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


class GSConv(nn.Module):
    """GSConv with depthwise feature generation and channel shuffle."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        if c2 % 2:
            raise ValueError(f"GSConv requires an even output channel count, got {c2}")
        hidden = c2 // 2
        self.cv1 = Conv(c1, hidden, k, s, g=g, act=act)
        self.cv2 = Conv(hidden, hidden, 5, 1, g=hidden, act=act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        b, c, h, w = x2.shape
        return x2.reshape(b, 2, c // 2, h, w).permute(0, 2, 1, 3, 4).reshape(b, c, h, w)


class GSBottleneck(nn.Module):
    """Residual bottleneck built from GSConv blocks."""

    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        hidden = int(c2 * e)
        self.main = nn.Sequential(GSConv(c1, hidden), GSConv(hidden, c2, 3, act=False))
        self.shortcut = Conv(c1, c2, 1, act=False)

    def forward(self, x):
        return self.main(x) + self.shortcut(x)


class VoVGSCSP(nn.Module):
    """Slim-neck CSP block using GS bottlenecks."""

    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        hidden = int(c2 * e)
        self.cv1 = Conv(c1, hidden, 1)
        self.cv2 = Conv(c1, hidden, 1)
        self.blocks = nn.Sequential(*(GSBottleneck(hidden, hidden, e=1.0) for _ in range(n)))
        self.cv3 = Conv(2 * hidden, c2, 1)

    def forward(self, x):
        return self.cv3(torch.cat((self.cv2(x), self.blocks(self.cv1(x))), 1))


__all__ = ("GSConv", "VoVGSCSP")
