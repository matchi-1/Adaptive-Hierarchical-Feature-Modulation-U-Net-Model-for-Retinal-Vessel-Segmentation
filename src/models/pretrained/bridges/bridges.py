# bridges.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GrayToRGB(nn.Module):
    """1→3 learnable adapter (safer than repeat(…,3))."""
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(1, 3, kernel_size=1, bias=False)
        with torch.no_grad():
            self.proj.weight[:] = 1/3
    def forward(self, x): return self.proj(x)

class FuseCat1x1(nn.Module):
    """Fuse two tensors by concat+1x1 projection to 'out_ch'."""
    def __init__(self, inA, inB, out_ch):
        super().__init__()
        self.proj = nn.Conv2d(inA + inB, out_ch, kernel_size=1, bias=True)
    def forward(self, A, B):
        if A.shape[-2:] != B.shape[-2:]:
            B = F.interpolate(B, size=A.shape[-2:], mode='bilinear', align_corners=False)
        return self.proj(torch.cat([A, B], dim=1))
