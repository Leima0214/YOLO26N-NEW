"""Transfer-friendly frequency-detail convolution."""

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv, autopad


class FDConv(nn.Module):
    """Conv-BN-activation plus a bounded high-frequency residual.

    The standard convolution path and state-dict names match Ultralytics Conv,
    while a zero-initialized scalar makes the initial forward exactly baseline-equivalent.
    """

    def __init__(
        self,
        c1,
        c2,
        k=1,
        s=1,
        p=None,
        g=1,
        d=1,
        act=True,
        cutoff=0.25,
        max_gain=0.1,
    ):
        super().__init__()
        if not 0.0 < float(cutoff) <= 0.5:
            raise ValueError(f"FDConv cutoff must be in (0, 0.5], got {cutoff}")
        if not 0.0 < float(max_gain) <= 1.0:
            raise ValueError(f"FDConv max_gain must be in (0, 1], got {max_gain}")
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.cutoff = float(cutoff)
        self.max_gain = float(max_gain)
        self.gamma = nn.Parameter(torch.zeros(1))

    def _high_frequency(self, x):
        _, _, h, w = x.shape
        work = x.float()
        spectrum = torch.fft.rfft2(work, norm="ortho")
        fy = torch.fft.fftfreq(h, device=x.device, dtype=work.dtype).reshape(h, 1)
        fx = torch.fft.rfftfreq(w, device=x.device, dtype=work.dtype).reshape(1, -1)
        radius = torch.sqrt(fy.square() + fx.square())
        high_pass = 1.0 - torch.exp(-radius.square() / (self.cutoff**2))
        detail = torch.fft.irfft2(spectrum * high_pass, s=(h, w), norm="ortho")
        return detail.to(dtype=x.dtype)

    def forward(self, x):
        base = self.conv(x)
        detail = self.conv(self._high_frequency(x))
        mixed = base + self.max_gain * torch.tanh(self.gamma) * detail
        return self.act(self.bn(mixed))


__all__ = ("FDConv",)
