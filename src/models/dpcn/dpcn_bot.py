# src/models/dpcn.py
# quick dpcn implementation by gpt
from __future__ import annotations
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import deformable conv from torchvision; fallback to standard conv if unavailable
_HAS_TV_DEFORM = False
try:
    from torchvision.ops import deform_conv2d   # weights-first API
    _HAS_TV_DEFORM = True
except Exception:
    _HAS_TV_DEFORM = False


class _DeformableConv2dOrConv2d(nn.Module):
    """
    Small wrapper that performs 3x3 deformable convolution if available,
    otherwise a plain 3x3 Conv2d (same channels, stride=1, padding=1).

    Inputs:
      - x:   (N, C_in, H, W)
      - off: (N, 2*k*k, H, W) offsets for deformable conv (ignored in fallback)

    Output:
      - y:   (N, C_out, H, W)
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, bias: bool = True):
        super().__init__()
        assert kernel_size == 3, "This wrapper assumes k=3 for simplicity."
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size

        if _HAS_TV_DEFORM:
            # We keep the conv weights as learnable params and call torchvision.ops.deform_conv2d in forward
            weight = torch.empty(out_ch, in_ch, kernel_size, kernel_size)
            nn.init.kaiming_normal_(weight, nonlinearity="relu")
            self.weight = nn.Parameter(weight)
            self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None
        else:
            # Fallback to plain Conv2d
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2, bias=bias)

    def forward(self, x: torch.Tensor, off: Optional[torch.Tensor] = None) -> torch.Tensor:
        if _HAS_TV_DEFORM:
            if off is None:
                # if offsets not provided, default to zeros (behaves like plain conv with same padding)
                N, _, H, W = x.shape
                off = x.new_zeros(N, 2 * self.kernel_size * self.kernel_size, H, W)
            # stride=1, padding=1, dilation=1, mask=None (no modulated version here)
            return deform_conv2d(
                input=x,
                offset=off,
                weight=self.weight,
                bias=self.bias,
                stride=1,
                padding=1,
                dilation=1,
                mask=None
            )
        else:
            return self.conv(x)


class DPCNIter(nn.Module):
    """
    One iteration of the Deformable-convolutional Pulse Coupling Network.

    Subsystems (per the paper):
      - Coupled Linking:       L(n) = Conv(Y(n-1))  (deformable if available)
      - Feeding Input:         F(n) = I_OR         (original input / shallow feature)
      - Modulation:            U(n) = β F(n) + (1-β) L(n)
      - Dynamic Threshold:     E(n) = exp(-aE) E(n-1) + (1 - exp(-aE)) * V_E * Y(n-1)
      - Activation:            Y(n) = sigmoid(U(n) - E(n))

    Shapes: all (N, C, H, W)
    """

    def __init__(
        self,
        channels: int,
        beta: float = 0.5,            # trade-off feeding vs linking
        aE: float = 0.5,              # decay rate; higher -> faster decay
        V_E: float = 1.0,             # growth scale from previous Y
        clamp_each_iter: bool = True
    ):
        super().__init__()
        self.channels = channels
        self.beta = nn.Parameter(torch.tensor(float(beta)), requires_grad=False)  # treat β as hyperparam (freeze)
        self.aE = nn.Parameter(torch.tensor(float(aE)), requires_grad=False)
        self.V_E = nn.Parameter(torch.tensor(float(V_E)), requires_grad=False)

        # Offset predictor for k=3 -> 2*k*k = 18 channels
        self.k = 3
        off_ch = 2 * self.k * self.k

        # offsets come from Y(n-1); a small conv preserves spatial size
        self.offset_conv = nn.Conv2d(channels, off_ch, kernel_size=3, padding=1)

        # deformable (or plain) 3x3 conv produces L(n)
        self.link_conv = _DeformableConv2dOrConv2d(channels, channels, kernel_size=3, bias=True) 

        # mild normalization on L(n) to stabilize (optional but helpful)
        self.norm = nn.BatchNorm2d(channels)

        # tiny init so offsets start near zero (behave like plain conv at the beginning)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

    def forward(self, y_prev: torch.Tensor, F: torch.Tensor, E_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          y_prev : Y(n-1)
          F      : I_OR (the “feeding input”—original shallow feature)
          E_prev : E(n-1)

        Returns:
          y      : Y(n)
          E      : E(n)
        """
        # Coupled Linking: L(n) = Conv(Y(n-1))  (deformable conv if offsets provided)
        offsets = self.offset_conv(y_prev)                     # (N, 18, H, W)
        L = self.link_conv(y_prev, offsets)                    # (N, C, H, W)
        L = self.norm(L)

        # Modulation: U(n) = β F + (1-β) L
        beta = torch.clamp(self.beta, 0.0, 1.0)
        U = beta * F + (1.0 - beta) * L

        # Dynamic Threshold:
        # E(n) = exp(-aE) E(n-1) + (1 - exp(-aE)) * V_E * Y(n-1)
        decay = torch.exp(-torch.clamp(self.aE, min=1e-6))
        grow  = (1.0 - decay) * self.V_E
        E = decay * E_prev + grow * y_prev

        # Activation: Y(n) = sigmoid(U - E)
        y = torch.sigmoid(U - E)

        return y, E

"""
    Full DPCN block that runs T iterations on shallow features.

    Usage:
      dpcn = DPCN(in_ch=1, iters=4, beta=0.5, aE=0.5, V_E=1.0)
      y = dpcn(x)  # x: (N, C=1, H, W) -> (N, C=1, H, W)

    Notes:
      - We project input to 'channels' (if needed), run iterations, and optionally project back.
      - By default, keep channels the same for simplicity.
    """
# src/models/dpcn.py
class DPCN(nn.Module):
    def __init__(
        self,
        in_ch: int,
        channels: Optional[int] = None,
        iters: int = 3,
        beta_init: float = 0.5,
        aE: float = 0.5,
        V_E: float = 1.0,   
        project_out: bool = False,
        clamp_each_iter: bool = True,   # <-- add this for parity with VAT
    ):
        super().__init__()
        channels = channels or in_ch
        self.iters = int(iters)
        self.channels = channels
        self.clamp_each_iter = clamp_each_iter

        self.proj_in  = nn.Identity() if in_ch == channels else nn.Conv2d(in_ch, channels, kernel_size=1)
        self.proj_out = nn.Identity() if not project_out else nn.Conv2d(channels, in_ch, kernel_size=1)

        self.cell = DPCNIter(
            channels=channels,
            beta=beta_init,
            aE=aE,
            V_E=V_E,
        )

    def forward(self, x: torch.Tensor, fov: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Feeding input
        F = self.proj_in(x)

        # Init states
        y = torch.sigmoid(F)
        E = torch.zeros_like(F)

        # --- make fov safe (device + dtype) ---
        if fov is not None:
            fov = fov.to(y.device).type_as(y)

        # collect outputs per iteration
        ys = []    # will hold each Y(n) for n in [1..T]

        # Iterate once (no second loop!)
        for _ in range(self.iters):
            #print(f"DPCN ITER: {_+1}/{self.iters}")
            y, E = self.cell(y_prev=y, F=F, E_prev=E)
            # optional FOV clamp each step
            if fov is not None and self.clamp_each_iter:
                y = y * fov
            
            ys.append(y)  # store current output

        #out = self.proj_out(y)

        # If you didn't clamp each iteration, at least clamp final output
        # if fov is not None and not self.clamp_each_iter:
        #     out = out * fov

        ys = torch.stack(ys, dim=1)  
        return ys


