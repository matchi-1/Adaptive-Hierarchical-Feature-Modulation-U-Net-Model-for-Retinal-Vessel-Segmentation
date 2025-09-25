import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d

class DPCN(nn.Module):
    def __init__(self, in_ch, channels=None, iters=3):
        super().__init__()
        self.in_ch = in_ch
        self.channels = channels or in_ch   # default: same channels as input
        self.iters = iters

        # ---- Coupled Linking setup ----
        k = 3  # 3x3 deformable kernel
        off_ch = 2 * k * k  # 18 channels for (dy,dx) per tap

        # Offset predictor: predicts offsets from the previous state Y(n-1)
        self.offset_conv = nn.Conv2d(self.channels, off_ch, kernel_size=3, padding=1)
        nn.init.zeros_(self.offset_conv.weight)  # init near zero so first ≈ plain conv
        nn.init.zeros_(self.offset_conv.bias)

        # Kernel weights for the deformable conv (W(i,j) in the paper)
        weight = torch.empty(self.channels, self.channels, k, k)
        nn.init.kaiming_normal_(weight, nonlinearity="relu")
        self.weight = nn.Parameter(weight)
        self.bias   = nn.Parameter(torch.zeros(self.channels))

        # normalization (helps stability)
        self.norm = nn.BatchNorm2d(self.channels)

    def coupled_linking(self, y_prev):
        """
        Coupled Linking Subsystem:
        L(n) = DefConv(Y(n-1))

        Args:
            y_prev: previous iteration output Y(n-1), shape [N,C,H,W]
        Returns:
            L: locally enhanced feature map, shape [N,C,H,W]
        """
        # 1. predict offsets from Y(n-1)
        offsets = self.offset_conv(y_prev)  # [N,18,H,W]

        # 2. apply deformable conv
        L = deform_conv2d(
            input=y_prev,
            offset=offsets,
            weight=self.weight,
            bias=self.bias,
            stride=1,
            padding=1,
            dilation=1,
            mask=None
        )

        # 3. normalization
        L = self.norm(L)
        return L

    def forward(self, x, fov=None):
        """
        x:   input shallow feature [N,C,H,W]
        fov: optional binary mask [N,1,H,W]
        """
        # for now: just demo coupled linking once
        y_prev = x
        L = self.coupled_linking(y_prev)

        return L
