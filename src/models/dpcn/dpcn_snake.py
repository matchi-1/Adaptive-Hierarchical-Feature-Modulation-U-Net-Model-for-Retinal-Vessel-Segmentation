# src/models/dpcn_v2.py
# Improved DPCN / Vessel-Aware Thresholding (VAT) feature enhancer
# - Stable parameterization (no runtime clamps): beta∈(0,1), aE>0, V_E≥0
# - Positive modulation gate: U = F * exp(beta * L)  (no sign flips)
# - Scale-invariant threshold update via (1 - exp(-aE)) factor
# - Optional E smoothing for spatial coherence
# - InstanceNorm on L (batch-size agnostic); BN available via flag
# - deform_conv2d fallback to plain 3x3 conv for portability
# - Aggregation control: "stack" (compat), "last", "mean", "max"
from __future__ import annotations
from typing import Optional, Tuple, Literal
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.ops import deform_conv2d as _deform_conv2d
    _HAS_DEFORM = True
except Exception:
    _deform_conv2d = None
    _HAS_DEFORM = False


def _gaussian3x3(device=None, dtype=None):
    # normalized separable 3x3 Gaussian (approx σ≈0.85)
    k = torch.tensor([[1., 2., 1.],
                      [2., 4., 2.],
                      [1., 2., 1.]], device=device, dtype=dtype)
    k = k / k.sum()
    return k


class DPCNIter(nn.Module):
    r"""
    One iteration of Deformable-Predictive Coupled Neuron with dynamic threshold.

    Subsystems (per iteration n):
      1) Coupled Linking:
         L(n) = DefConv( Y(n-1) )         (deformable 3×3; fallback: plain 3×3)
         L is normalized (InstanceNorm2d by default).

      2) Modulation (positive gate):
         U(n) = F ⊙ exp( β · L(n) ),   β ∈ (0, 1)

      3) Dynamic Threshold (IIR, four modes):
         decay = exp(-aE),  aE > 0, V_E ≥ 0
         Modes for grow_term G(n-1):
           - "paper":      G = V_E · Y(n-1)
           - "paper_mod":  G = (1 - decay) · V_E · Y(n-1)
           - "vat":        G = V_E · (Y(n-1) ⊙ Vconf)
           - "scaled_vat": G = (1 - decay) · V_E · (Y(n-1) ⊙ Vconf)

         E(n) = decay · E(n-1) + G

         (Optional) a small depth-wise 3×3 Gaussian smooth on E(n) improves spatial coherence.

      4) Activation:
         Y(n) = σ( U(n) - E(n) ),  σ = sigmoid

    Shapes: all tensors [N, C, H, W] except Vconf [N, 1, H, W].
    """

    def __init__(
        self,
        channels: int,
        beta_init: float = 0.3,
        aE_init: float = 0.35,   # ≈ ln(2)/2 → half-life ~2 iterations
        V_E_init: float = 1.0,
        threshold_mode: Literal["paper", "paper_mod", "vat", "scaled_vat"] = "scaled_vat",
        norm_on_L: Literal["instance", "batch", "none"] = "instance",
        smooth_E: bool = True,
        use_deformable: Optional[bool] = None,
    ):
        super().__init__()
        self.channels = int(channels)
        self.threshold_mode = threshold_mode.lower()
        self.smooth_E = bool(smooth_E)

        # Decide deformable availability
        if use_deformable is None:
            self.use_deformable = _HAS_DEFORM
        else:
            self.use_deformable = bool(use_deformable and _HAS_DEFORM)
        if not self.use_deformable:
            warnings.warn("[DPCNIter] deform_conv2d not available → using plain 3×3 Conv as fallback.", RuntimeWarning)

        # ---- Learnable scalars with safe re-parameterizations ----
        # raw params (unconstrained); mapped in forward:
        #   beta = sigmoid(b_raw)        ∈ (0,1)
        #   aE   = softplus(a_raw) + ε   > 0
        #   V_E  = softplus(v_raw)       ≥ 0
        self.b_raw = nn.Parameter(torch.tensor(float(beta_init)))
        self.a_raw = nn.Parameter(torch.tensor(float(aE_init)))
        self.v_raw = nn.Parameter(torch.tensor(float(V_E_init)))

        # ---- Offsets for deformable conv (predicted from Y(n-1)) ----
        k = 3
        off_ch = 2 * k * k   # (dy, dx) per tap
        self.offset_conv = nn.Conv2d(channels, off_ch, kernel_size=3, padding=1)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

        # ---- Kernel weights for the (deformable) conv ----
        w = torch.empty(channels, channels, k, k)
        nn.init.kaiming_normal_(w, nonlinearity="relu")
        self.weight = nn.Parameter(w)
        self.bias   = nn.Parameter(torch.zeros(channels))

        # ---- Normalization on L ----
        if norm_on_L == "instance":
            self.norm_L = nn.InstanceNorm2d(channels, affine=True)
        elif norm_on_L == "batch":
            self.norm_L = nn.BatchNorm2d(channels)
        else:
            self.norm_L = nn.Identity()

        # ---- Optional E smoothing (depth-wise Gaussian 3×3, fixed) ----
        if self.smooth_E:
            self.smoothE = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
            with torch.no_grad():
                g = _gaussian3x3()
                self.smoothE.weight.data.zero_()
                for c in range(channels):
                    self.smoothE.weight.data[c, 0, :, :] = g
            for p in self.smoothE.parameters():
                p.requires_grad = False

    # Parameter mappings
    def _beta(self) -> torch.Tensor:
        return torch.sigmoid(self.b_raw)

    def _aE(self) -> torch.Tensor:
        return F.softplus(self.a_raw) + 1e-4

    def _VE(self) -> torch.Tensor:
        return F.softplus(self.v_raw)

    def forward(
        self,
        y_prev: torch.Tensor,
        F_in: torch.Tensor,
        E_prev: torch.Tensor,
        Vconf: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1) Coupled Linking
        if self.use_deformable:
            offsets = self.offset_conv(y_prev)  # [N,18,H,W]
            L = _deform_conv2d(
                input=y_prev,
                offset=offsets,
                weight=self.weight,
                bias=self.bias,
                stride=1, padding=1, dilation=1, mask=None
            )
        else:
            # fallback: plain conv
            L = F.conv2d(y_prev, self.weight, self.bias, stride=1, padding=1, dilation=1)

        L = self.norm_L(L)

        # 2) Positive modulation gate: U = F * exp(beta * L)
        beta = self._beta()
        U = F_in * torch.exp(beta * L)

        # 3) Dynamic Threshold
        aE  = self._aE()
        V_E = self._VE()
        decay = torch.exp(-aE)  # (0,1)

        mode = self.threshold_mode
        if mode == "paper":
            grow_term = V_E * y_prev
        elif mode == "paper_mod":
            grow_term = (1.0 - decay) * V_E * y_prev
        elif mode == "vat":
            if Vconf is None:
                raise ValueError("Vconf is required for 'vat' mode.")
            grow_term = V_E * (y_prev * Vconf)
        elif mode == "scaled_vat":
            if Vconf is None:
                raise ValueError("Vconf is required for 'scaled_vat' mode.")
            grow_term = (1.0 - decay) * V_E * (y_prev * Vconf)
        else:
            raise ValueError(f"Unknown threshold_mode: {mode}")

        E = decay * E_prev + grow_term
        if self.smooth_E:
            E = self.smoothE(E)

        # 4) Activation
        y = torch.sigmoid(U - E)
        return y, E


class DPCN(nn.Module):
    r"""
    DPCN wrapper that runs T iterations on shallow features and returns
    either the stacked outputs (compat) or an aggregation.

    Args:
      in_ch:            input channels
      channels:         internal channels (defaults to in_ch)
      iters:            number of iterations (T)
      beta_init:        initial β for modulation gate (0<β<1 after mapping)
      half_life:        desired half-life (in iterations) of E's memory; aE = ln(2)/half_life
      V_E_init:         initial growth scale V_E (mapped to ≥0)
      threshold_mode:   "paper", "paper_mod", "vat", "scaled_vat"
      clamp_each_iter:  if FOV provided, clamp Y(n) inside FOV each step
      project_out:      optional 1×1 projection back to in_ch (unused if aggregate="stack")
      norm_on_L:        normalization on L: "instance"(default), "batch", "none"
      smooth_E:         enable E smoothing (depth-wise Gaussian 3×3)
      aggregate:        "stack"(default), "last", "mean", "max"
      vconf_from:       "x" (raw input) or "F" (projected features) for confidence branch
      use_deformable:   force enable/disable deformable conv (None=auto)
    """

    def __init__(
        self,
        in_ch: int,
        channels: Optional[int] = None,
        iters: int = 3,
        beta_init: float = 0.3,
        half_life: float = 2.0,
        V_E_init: float = 1.0,
        threshold_mode: Literal["paper", "paper_mod", "vat", "scaled_vat"] = "scaled_vat",
        clamp_each_iter: bool = True,
        project_out: bool = False,
        norm_on_L: Literal["instance", "batch", "none"] = "instance",
        smooth_E: bool = True,
        aggregate: Literal["stack", "last", "mean", "max"] = "stack",
        vconf_from: Literal["x", "F"] = "x",
        use_deformable: Optional[bool] = None,
    ):
        super().__init__()
        channels = channels or in_ch
        self.in_ch = int(in_ch)
        self.channels = int(channels)
        self.iters = int(iters)
        self.clamp_each_iter = bool(clamp_each_iter)
        self.threshold_mode = threshold_mode.lower()
        self.aggregate_mode = aggregate
        self.vconf_from = vconf_from

        # aE from half-life (iterations): aE = ln(2)/h
        if half_life <= 0:
            raise ValueError("half_life must be > 0")
        aE_init = float(math.log(2.0) / half_life)

        # projection in/out
        self.proj_in  = nn.Identity() if in_ch == channels else nn.Conv2d(in_ch, channels, kernel_size=1)
        self.proj_out = nn.Identity() if not project_out else nn.Conv2d(channels, in_ch, kernel_size=1)

        # Vessel-confidence branch (tiny CNN)
        vconf_hidden = max(8, in_ch)
        self.vconf_branch = nn.Sequential(
            nn.Conv2d(in_ch, vconf_hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(vconf_hidden, vconf_hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(vconf_hidden, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # One iteration cell
        self.cell = DPCNIter(
            channels=channels,
            beta_init=beta_init,
            aE_init=aE_init,
            V_E_init=V_E_init,
            threshold_mode=self.threshold_mode,
            norm_on_L=norm_on_L,
            smooth_E=smooth_E,
            use_deformable=use_deformable,
        )

    @torch.no_grad()
    def _make_vconf(self, base: torch.Tensor) -> torch.Tensor:
        Vconf = self.vconf_branch(base)
        return Vconf

    def forward(
        self,
        x: torch.Tensor,
        fov: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
          x:   [N, in_ch, H, W]
          fov: [N, 1, H, W] binary mask of valid retina region (optional)
        Returns:
          If aggregate="stack": [N, T, C, H, W] (compat)
          Else:                  [N, C, H, W]
        """
        N, _, H, W = x.shape

        # Feeding input F (project once)
        F_in = self.proj_in(x)                          # [N,C,H,W]

        # Initial states
        y = torch.sigmoid(F_in)                         # Y(0)
        E = torch.zeros_like(F_in)                      # E(0)=0

        # Vessel confidence (cheap, once)
        if self.vconf_from == "F":
            base_for_vconf = F_in
        else:
            base_for_vconf = x
        Vconf = self._make_vconf(base_for_vconf).to(y.dtype)
        if fov is not None:
            Vconf = Vconf * fov.to(y.dtype)

        # Safe FOV
        if fov is not None:
            fov = fov.to(y.dtype)

        ys = []
        for _ in range(self.iters):
            y, E = self.cell(y_prev=y, F_in=F_in, E_prev=E,
                             Vconf=Vconf if self.threshold_mode not in ("paper", "paper_mod") else None)
            if fov is not None and self.clamp_each_iter:
                y = y * fov
            ys.append(y)

        # If we didn’t clamp each step, clamp last
        if fov is not None and not self.clamp_each_iter:
            ys[-1] = ys[-1] * fov

        # Aggregate
        ys_stack = torch.stack(ys, dim=1)  # [N,T,C,H,W]
        if self.aggregate_mode == "stack":
            return ys_stack
        elif self.aggregate_mode == "last":
            return ys_stack[:, -1]
        elif self.aggregate_mode == "mean":
            return ys_stack.mean(dim=1)
        elif self.aggregate_mode == "max":
            return ys_stack.max(dim=1).values
        else:
            raise ValueError(f"Unknown aggregate mode: {self.aggregate_mode}")
