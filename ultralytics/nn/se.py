"""Identity-initialized squeeze-and-excitation attention."""

import torch
from torch import nn


class SEAttention(nn.Module):
    def __init__(self, channel=512, c2=None, reduction=16):
        super().__init__()
        if channel <= 0 or reduction <= 0:
            raise ValueError(f"SE channel and reduction must be positive, got {channel=}, {reduction=}")
        hidden = max(1, channel // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channel, bias=False),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, _, _ = x.shape
        gate = self.fc(self.avg_pool(x).reshape(b, c)).reshape(b, c, 1, 1)
        attended = x * gate
        return x + self.gamma * (attended - x)


__all__ = ("SEAttention",)
