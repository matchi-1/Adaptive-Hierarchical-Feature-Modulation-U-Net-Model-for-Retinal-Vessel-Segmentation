# src/models/dpcn_exp1.py
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


# Hard-require deformable conv (fail fast if build mismatch)
if not hasattr(torch.ops.torchvision, "deform_conv2d"):
    raise RuntimeError(
        "This build of torchvision lacks deform_conv2d. "
        "Install matching torch/torchvision wheels."
    )


class DPCNIter(nn.Module):
    """
    One stable iteration of DPCN with:
      • Mixed linking: L = (1-λ)*ConvDW+PW(Yprev)  +  λ*DefConv(Yprev, offsets)
      • Modulation:    U = F * (1 + β * L)
      • Threshold:     E = exp(-aE)*Eprev + gain * smooth(Yprev) * (Vconf?)
      • Activation:    y_hat = sigmoid(U - E)
      • EMA smoothing: y = (1-η)*Yprev + η*y_hat

    All shapes [N,C,H,W]. Scalars are constrained (β∈(0,1), aE>0, V_E>0, η∈(0,1)).
    Offsets are clamped via tanh and a learnable global scale so early training
    behaves like a regular conv (no wild warping).
    """
    def __init__(
        self,
        channels: int,
        beta: float = 0.3,
        aE: float = 0.4,
        V_E: float = 1.0,
        max_offset: float = 1.0,       # max abs offset in pixels per tap
        threshold_mode: str = "vat",   # "paper", "paper_mod", "vat", "scaled_vat"
        ema_init: float = 0.5          # EMA mixing for y across iterations
    ):
        super().__init__()
        self.channels = channels
        self.threshold_mode = threshold_mode.lower()

        # ---- learnable, constrained scalars ----
        # β in (0,1): how strong the linking modulates F
        self.beta_p = nn.Parameter(torch.tensor(float(beta)).logit())  # invert sigmoid
        # aE>0 (decay), V_E>0 (growth), η∈(0,1) (EMA)
        self.aE_p   = nn.Parameter(torch.tensor(float(aE)).log())      # invert softplus≈exp
        self.V_E_p  = nn.Parameter(torch.tensor(float(V_E)).log())
        self.eta_p  = nn.Parameter(torch.tensor(float(ema_init)).logit())

        # ---- Linking: stable DW+PW conv branch ----
        self.link_dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                                 groups=channels, bias=False)
        self.link_pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.link_dw.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.link_pw.weight, nonlinearity="relu")

        # ---- Linking: deformable conv branch ----
        k = 3
        off_ch = 2 * k * k          # (dy,dx) per tap
        self.offset_conv = nn.Conv2d(channels, off_ch, kernel_size=3, padding=1)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

        w = torch.empty(channels, channels, k, k)
        nn.init.kaiming_normal_(w, nonlinearity="relu")
        self.def_weight = nn.Parameter(w)
        self.def_bias   = nn.Parameter(torch.zeros(channels))

        # learnable global scale for offsets; start small → near-zero offsets initially
        self.offset_scale_p = nn.Parameter(torch.tensor(-2.0))  # sigmoid(-2)~0.12
        self.max_offset = float(max_offset)

        # mix stable vs deformable: λ ∈ (0,1), start biased to stable
        self.lam_p = nn.Parameter(torch.tensor(-1.5))  # sigmoid≈0.18

        # ---- normalizations ----
        self.norm_L_stable = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.norm_L_def    = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.norm_U        = nn.GroupNorm(num_groups=8, num_channels=channels)

        # ---- smoothing for threshold growth ----
        self.avg3 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    def _linking(self, y_prev: torch.Tensor) -> torch.Tensor:
        # stable path
        Ls = self.link_pw(self.link_dw(y_prev))
        Ls = self.norm_L_stable(Ls)

        # deformable path with safe offsets
        raw_off = self.offset_conv(y_prev)
        # clamp offsets ∈ [-max_offset * s, +max_offset * s]
        s = torch.sigmoid(self.offset_scale_p) * self.max_offset
        offs = torch.tanh(raw_off) * s

        Ld = deform_conv2d(
            input=y_prev,
            offset=offs,
            weight=self.def_weight,
            bias=self.def_bias,
            stride=1, padding=1, dilation=1, mask=None
        )
        Ld = self.norm_L_def(Ld)

        lam = torch.sigmoid(self.lam_p)
        return (1.0 - lam) * Ls + lam * Ld

    def _update_threshold(self, E_prev, y_prev, Vconf: Optional[torch.Tensor]) -> torch.Tensor:
        aE  = F.softplus(self.aE_p) + 1e-6
        V_E = F.softplus(self.V_E_p)
        decay = torch.exp(-aE)

        # smooth y to avoid speckle-driven threshold growth
        y_s = self.avg3(y_prev)

        mode = self.threshold_mode
        if mode == "paper":
            grow_term = V_E * y_s
        elif mode == "paper_mod":
            grow_term = (1.0 - decay) * V_E * y_s
        elif mode == "vat":
            if Vconf is None:
                raise ValueError("Vconf is required for 'vat' mode.")
            grow_term = V_E * (y_s * Vconf)
        elif mode == "scaled_vat":
            if Vconf is None:
                raise ValueError("Vconf is required for 'scaled_vat' mode.")
            grow_term = (1.0 - decay) * V_E * (y_s * Vconf)
        else:
            raise ValueError(f"Unknown threshold_mode: {self.threshold_mode}")

        return decay * E_prev + grow_term

    def forward(
        self,
        y_prev: torch.Tensor,
        F_in: torch.Tensor,
        E_prev: torch.Tensor,
        Vconf: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # 1) Coupled linking (stable + deformable with gating)
        L = self._linking(y_prev)

        # 2) Modulation
        beta = torch.sigmoid(self.beta_p)
        U = F_in * (1.0 + beta * L)
        U = self.norm_U(U)

        # 3) Dynamic threshold
        E = self._update_threshold(E_prev, y_prev, Vconf)

        # 4) Activation + EMA smoothing across iterations
        y_hat = torch.sigmoid(U - E)
        eta   = torch.sigmoid(self.eta_p)
        y     = (1.0 - eta) * y_prev + eta * y_hat
        return y, E


class DPCN(nn.Module):
    """
    Stable DPCN wrapper:
      • projects input to C channels once (F)
      • computes a smoothed Vconf once (optional VAT)
      • runs T iterations with EMA smoothing
      • returns stack [N, T, C, H, W]
    """
    def __init__(
        self,
        in_ch: int,
        channels: Optional[int] = None,
        iters: int = 3,
        beta_init: float = 0.3,
        aE: float = 0.4,
        V_E: float = 1.0,
        clamp_each_iter: bool = True,
        project_out: bool = False,
        threshold_mode: str = "scaled_vat",
        max_offset: float = 1.0,
        ema_init: float = 0.5,
    ):
        super().__init__()
        channels = channels or in_ch
        self.iters = int(iters)
        self.channels = channels
        self.clamp_each_iter = clamp_each_iter
        self.threshold_mode = threshold_mode

        # projection in/out
        self.proj_in  = nn.Identity() if in_ch == channels else nn.Conv2d(in_ch, channels, 1)
        self.proj_out = nn.Identity() if not project_out else nn.Conv2d(channels, in_ch, 1)

        # light Vconf branch (smoothed)
        mid = max(8, in_ch)
        self.vconf_branch = nn.Sequential(
            nn.Conv2d(in_ch, mid, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(mid,  mid, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(mid,  1,   kernel_size=1), nn.Sigmoid()
        )
        self.vconf_smooth = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)

        # iteration cell
        self.cell = DPCNIter(
            channels=channels,
            beta=beta_init,
            aE=aE,
            V_E=V_E,
            max_offset=max_offset,
            threshold_mode=threshold_mode,
            ema_init=ema_init,
        )

    def forward(self, x: torch.Tensor, fov: Optional[torch.Tensor] = None) -> torch.Tensor:
        F = self.proj_in(x)                     # [N,C,H,W]
        y = torch.sigmoid(F)                    # Y(0)
        E = torch.zeros_like(F)                 # E(0)

        # Vconf once (smoothed + FOV-gated)
        Vconf = self.vconf_branch(x).type_as(y)
        Vconf = self.vconf_smooth(Vconf)
        if fov is not None:
            fov   = fov.to(y.device).type_as(y)
            Vconf = Vconf * fov

        ys = []
        for _ in range(self.iters):
            y, E = self.cell(
                y_prev=y,
                F_in=F,
                E_prev=E,
                Vconf=Vconf if self.threshold_mode != "paper" and self.threshold_mode != "paper_mod" else None
            )
            if fov is not None and self.clamp_each_iter:
                y = y * fov
            ys.append(y)

        if fov is not None and not self.clamp_each_iter:
            ys[-1] = ys[-1] * fov

        return torch.stack(ys, dim=1)          # [N,T,C,H,W]
