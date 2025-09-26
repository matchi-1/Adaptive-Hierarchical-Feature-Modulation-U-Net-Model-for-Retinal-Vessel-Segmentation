# src/models/dpcn_exp1.py
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d

# Hard-require deformable conv
if not hasattr(torch.ops.torchvision, "deform_conv2d"):
    raise RuntimeError(
        "This build of torchvision lacks deform_conv2d. "
        "Install matching torch/torchvision wheels."
    )

class DPCNIter(nn.Module):
    """
    One iteration of DPCN (paper-faithful math):

      Coupled Linking:   L(n) = DefConv( Y(n-1) )
      Modulation:        U(n) = F(n) * ( 1 + β * L(n) )
      Threshold:         E(n) = exp(-aE)*E(n-1) + V_E * Y(n-1)
                         (you can switch to (1-exp(-aE))*V_E if desired)
      Activation:        Y(n) = sigmoid( U(n) - E(n) )

    Shapes are all [N, C, H, W].
    """

    def __init__(self, channels: int, beta: float, aE: float, V_E: float):
        super().__init__()
        self.channels = channels

        # β, aE, V_E as hyperparams (frozen to match your latest code)
        self.beta = nn.Parameter(torch.tensor(float(beta)), requires_grad=False)
        self.aE  = nn.Parameter(torch.tensor(float(aE)),   requires_grad=False)
        self.V_E = nn.Parameter(torch.tensor(float(V_E)),  requires_grad=False)

        # --- Coupled linking: offsets + deformable conv weights ---
        k = 3
        off_ch = 2 * k * k  # 18 (dy,dx for each of 9 taps)
        self.offset_conv = nn.Conv2d(channels, off_ch, kernel_size=3, padding=1)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

        # deformable conv learnable weight/bias
        w = torch.empty(channels, channels, k, k)  # (C_out, C_in, 3, 3)
        nn.init.kaiming_normal_(w, nonlinearity="relu")
        self.weight = nn.Parameter(w)
        self.bias   = nn.Parameter(torch.zeros(channels))

        # mild norm on L(n)
        self.norm_L = nn.BatchNorm2d(channels)

        # optional norm on U(n) to stabilize multiplicative modulation
        self.norm_U = nn.BatchNorm2d(channels)

    def forward(self, y_prev: torch.Tensor, F: torch.Tensor, E_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1) Coupled linking: L(n) = DefConv( Y(n-1) )
        offsets = self.offset_conv(y_prev)  # [N,18,H,W]
        L = deform_conv2d(
            input=y_prev,
            offset=offsets,
            weight=self.weight,
            bias=self.bias,
            stride=1, padding=1, dilation=1, mask=None
        )
        L = self.norm_L(L)

        # 2) Modulation (paper): U(n) = F(n) * (1 + β * L(n))
        beta = torch.clamp(self.beta, 0.0, 1.0)   # keep sane
        U = F * (1.0 + beta * L)
        U = self.norm_U(U)

        # 3) Threshold: E(n) = exp(-aE) * E(n-1) + V_E * Y(n-1)
        aE = torch.clamp(self.aE, min=1e-6)
        decay = torch.exp(-aE)               # in (0,1)
        #E = decay * E_prev + self.V_E * y_prev
        grow  = (1.0 - decay) * self.V_E   
        E = decay * E_prev + grow * y_prev  

        # 4) Activation: Y(n) = sigmoid( U(n) - E(n) )
        y = torch.sigmoid(U - E)
        return y, E


class DPCN(nn.Module):
    """
    Wrapper that runs T iterations of DPCNIter on shallow features.
    Returns all iteration outputs stacked: [N, T, C, H, W].
    """

    def __init__(
        self,
        in_ch: int,
        channels: Optional[int] = None,
        iters: int = 3,
        beta_init: float = 0.5,
        aE: float = 0.5,
        V_E: float = 1.0,
        clamp_each_iter: bool = True,
        project_out: bool = False,
    ):
        super().__init__()
        channels = channels or in_ch
        self.iters = int(iters)
        self.channels = channels
        self.clamp_each_iter = clamp_each_iter

        # project input to internal channels (F(n) = I_OR after projection)
        self.proj_in  = nn.Identity() if in_ch == channels else nn.Conv2d(in_ch, channels, kernel_size=1)
        # optional projection back
        self.proj_out = nn.Identity() if not project_out else nn.Conv2d(channels, in_ch, kernel_size=1)

        # one iteration cell with paper-faithful math
        self.cell = DPCNIter(
            channels=channels,
            beta=beta_init,
            aE=aE,
            V_E=V_E,
        )

    def forward(self, x: torch.Tensor, fov: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Feeding input F (computed once)
        F = self.proj_in(x)                # [N,C,H,W]

        # Initialize Y(0), E(0)
        y = torch.sigmoid(F)               # reasonable Y0
        E = torch.zeros_like(F)            # E0 = 0

        # --- make fov safe (device + dtype) ---
        if fov is not None:
            fov = fov.to(y.device).type_as(y)

        ys = []
        for _ in range(self.iters):
            print(f"DPCN ITER: {_+1}/{self.iters}")
            # one DPCN iteration
            y, E = self.cell(y_prev=y, F=F, E_prev=E)

            # optional FOV clamp per iter
            if fov is not None and self.clamp_each_iter:
                y = y * fov

            ys.append(y)

        # stack: [N,T,C,H,W]
        ys = torch.stack(ys, dim=1)

        # if only final clamp desired:
        if fov is not None and not self.clamp_each_iter:
            ys[:, -1] = ys[:, -1] * fov

        return ys
