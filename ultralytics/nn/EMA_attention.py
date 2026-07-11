"""Identity-initialized efficient multi-scale attention."""

import torch
from torch import nn


class EMA_attention(nn.Module):
    """EMA with a zero-initialized residual gate for pretrained fine-tuning."""

    def __init__(self, channels, c2=None, factor=32):
        super().__init__()
        if not isinstance(factor, int) or factor <= 0 or channels % factor:
            raise ValueError(f"EMA factor must be a positive divisor of channels, got channels={channels}, factor={factor}")
        self.groups = factor
        group_channels = channels // factor
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d(1)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(group_channels, group_channels)
        self.conv1x1 = nn.Conv2d(group_channels, group_channels, 1)
        self.conv3x3 = nn.Conv2d(group_channels, group_channels, 3, padding=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.shape
        if c % self.groups:
            raise ValueError(f"EMA input channels {c} are not divisible by factor {self.groups}")
        grouped = x.reshape(b * self.groups, c // self.groups, h, w)
        x_h = self.pool_h(grouped)
        x_w = self.pool_w(grouped).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat((x_h, x_w), dim=2))
        x_h, x_w = torch.split(hw, (h, w), dim=2)
        x1 = self.gn(grouped * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(grouped)
        x11 = self.softmax(self.agp(x1).flatten(2).transpose(1, 2))
        x12 = x2.flatten(2)
        x21 = self.softmax(self.agp(x2).flatten(2).transpose(1, 2))
        x22 = x1.flatten(2)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        attended = (grouped * weights.sigmoid()).reshape_as(x)
        return x + self.gamma * (attended - x)


__all__ = ("EMA_attention",)
