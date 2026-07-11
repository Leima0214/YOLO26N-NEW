"""Identity-initialized convolutional block attention module."""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        if channels <= 0 or reduction <= 0:
            raise ValueError(f"CBAM channels and reduction must be positive, got {channels=}, {reduction=}")
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, x):
        return torch.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        if kernel_size not in {3, 7}:
            raise ValueError(f"CBAM spatial kernel must be 3 or 7, got {kernel_size}")
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        pooled = torch.cat((x.mean(1, keepdim=True), x.amax(1, keepdim=True)), dim=1)
        return torch.sigmoid(self.conv(pooled))


class CBAM(nn.Module):
    def __init__(self, c1, c2=None, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_attention = ChannelAttention(c1, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        attended = x * self.channel_attention(x)
        attended = attended * self.spatial_attention(attended)
        return x + self.gamma * (attended - x)


__all__ = ("CBAM",)
