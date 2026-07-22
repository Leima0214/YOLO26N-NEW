# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""YOLO26-native lightweight region guidance for road-damage detection."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int, maximum: int = 8) -> int:
    """Return the largest small GroupNorm divisor for ``channels``."""
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvGNAct(nn.Sequential):
    """A fusion-safe convolution block that does not depend on batch statistics."""

    def __init__(self, c1: int, c2: int, kernel_size=1, padding=0, groups=1, act=True):
        layers = [
            nn.Conv2d(c1, c2, kernel_size, padding=padding, groups=groups, bias=False),
            nn.GroupNorm(_group_count(c2), c2),
        ]
        if act:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class LiteRegionPriorHead(nn.Module):
    """Predict a high-resolution region logit map from P2 and P3 features."""

    def __init__(self, c2: int, c3: int, hidden: int = 32):
        super().__init__()
        hidden = max(int(hidden), 8)
        self.p2_proj = ConvGNAct(c2, hidden, 1)
        self.p3_proj = ConvGNAct(c3, hidden, 1)
        self.fuse = nn.Sequential(
            ConvGNAct(2 * hidden, 2 * hidden, 3, padding=1, groups=2 * hidden),
            ConvGNAct(2 * hidden, hidden, 1),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
        """Return P2-resolution region logits."""
        p3 = F.interpolate(self.p3_proj(p3), size=p2.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat((self.p2_proj(p2), p3), dim=1))


class DirectionalRegionGate(nn.Module):
    """Extract horizontal and vertical residuals under a learned region prior."""

    def __init__(self, channels: int, kernel_size: int = 9, reduction: int = 4):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"Directional kernel size must be odd, got {kernel_size}.")
        padding = kernel_size // 2
        self.horizontal = nn.Conv2d(
            channels, channels, (1, kernel_size), padding=(0, padding), groups=channels, bias=False
        )
        self.vertical = nn.Conv2d(
            channels, channels, (kernel_size, 1), padding=(padding, 0), groups=channels, bias=False
        )
        self.mix = ConvGNAct(channels, channels, 1)
        hidden = max(channels // reduction, 8)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(nn.Conv2d(3, 1, 7, padding=3, bias=False), nn.Sigmoid())

    def forward(self, feature: torch.Tensor, region_logits: torch.Tensor) -> torch.Tensor:
        """Return a region-conditioned directional residual with the same shape as ``feature``."""
        region = torch.sigmoid(
            F.interpolate(region_logits, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        )
        masked = feature * region
        directional = self.mix(self.horizontal(masked) + self.vertical(masked))
        channel_weight = self.channel_gate(directional)
        spatial_input = torch.cat(
            (directional.mean(1, keepdim=True), directional.amax(1, keepdim=True), region), dim=1
        )
        return directional * channel_weight * self.spatial_gate(spatial_input)


class LiteRFFProjection(nn.Sequential):
    """Depthwise-pointwise residual projection used by Lite-RFF."""

    def __init__(self, c1: int, c2: int):
        super().__init__(
            ConvGNAct(c1, c1, 3, padding=1, groups=c1),
            ConvGNAct(c1, c2, 1, act=False),
        )


class ProgressiveLiteRG(nn.Module):
    """Coordinate P2 prior prediction, P3/P4 DRG, and N3/N4 Lite-RFF guidance."""

    def __init__(
        self,
        p2_channels: int,
        p3_channels: int,
        p4_channels: int,
        n3_channels: int,
        n4_channels: int,
        hidden_channels: int = 32,
        directional_kernel: int = 9,
        use_drg: bool = True,
        use_rff: bool = True,
    ):
        super().__init__()
        if use_rff and not use_drg:
            raise ValueError("Lite-RFF requires DRG residuals; set use_drg=True.")
        self.use_drg = bool(use_drg)
        self.use_rff = bool(use_rff)
        self.prior = LiteRegionPriorHead(p2_channels, p3_channels, hidden_channels)
        self.drg3 = DirectionalRegionGate(p3_channels, directional_kernel) if self.use_drg else None
        self.drg4 = DirectionalRegionGate(p4_channels, directional_kernel) if self.use_drg else None
        self.rff3 = LiteRFFProjection(p3_channels, n3_channels) if self.use_rff else None
        self.rff4 = LiteRFFProjection(p4_channels, n4_channels) if self.use_rff else None

        # Exact B0 detection behavior at initialization; the auxiliary prior still learns immediately.
        self.gamma3 = nn.Parameter(torch.zeros(1)) if self.use_drg else None
        self.gamma4 = nn.Parameter(torch.zeros(1)) if self.use_drg else None
        self.eta3 = nn.Parameter(torch.zeros(1)) if self.use_rff else None
        self.eta4 = nn.Parameter(torch.zeros(1)) if self.use_rff else None

    def guide_backbone(
        self, p2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict the prior and inject directional residuals before the original neck."""
        region_logits = self.prior(p2, p3)
        if not self.use_drg:
            return p3, p4, None, None, region_logits
        residual3 = self.drg3(p3, region_logits)
        residual4 = self.drg4(p4, region_logits)
        return (
            p3 + self.gamma3 * residual3,
            p4 + self.gamma4 * residual4,
            residual3,
            residual4,
            region_logits,
        )

    def fuse_neck3(self, neck3: torch.Tensor, residual3: torch.Tensor) -> torch.Tensor:
        """Apply the P3 Lite-RFF residual after the original neck block."""
        if not self.use_rff:
            return neck3
        return neck3 + self.eta3 * self.rff3(residual3)

    def fuse_neck4(self, neck4: torch.Tensor, residual4: torch.Tensor) -> torch.Tensor:
        """Apply the P4 Lite-RFF residual after the original neck block."""
        if not self.use_rff:
            return neck4
        return neck4 + self.eta4 * self.rff4(residual4)
