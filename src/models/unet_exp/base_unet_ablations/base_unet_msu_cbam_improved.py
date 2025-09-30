# src/models/unet_with_msu_cbam_edgegated.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.unet import ConvBlock, EncoderBlock, DecoderBlock
from src.models.blocks.msu import MSU
from src.models.blocks.cbam import CBAM


# ---------- helpers ----------
def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] != ref.shape[-2:]:
        x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
    return x


# ---------- MSU alignment (unchanged) ----------
class AlignMSU(nn.Module):
    """
    Align two feature maps to a chosen 'anchor' resolution/channels, then apply MSU.
    """
    def __init__(self, inA: int, inB: int, out_ch: int, use_bn: bool = True, activation: bool = True):
        super().__init__()
        self.projA = nn.Conv2d(inA, out_ch, kernel_size=1, bias=True)
        self.projB = nn.Conv2d(inB, out_ch, kernel_size=1, bias=True)
        self.msu   = MSU(in_channels=out_ch, out_channels=out_ch, use_bn=use_bn, activation=activation)

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A_ = self.projA(A)
        B_ = self.projB(B)
        if B_.shape[-2:] != A_.shape[-2:]:
            B_ = F.interpolate(B_, size=A_.shape[-2:], mode="bilinear", align_corners=False)
        return self.msu(A_, B_)  # (N, out_ch, H_anchor, W_anchor)


# ---------- Edge-Aware Residual CBAM ----------
class EdgeAwareResidualCBAM(nn.Module):
    """
    Residual CBAM gated by an edge hint (from MSU):
        y = x + alpha * gate(edge_hint) * (CBAM(x) - x)

    - 'edge_hint' can be any tensor; we reduce(abs) across channels and resize to x.
    - Gate is initialized to prefer background (↑CBAM) and suppress on edges (↓CBAM).
    """
    def __init__(self, channels: int, reduction: int = 16, use_spatial: bool = False, alpha_init: float = 0.25):
        super().__init__()
        self.cbam = CBAM(channels, reduction_ratio=reduction, use_spatial=use_spatial)

        # residual strength α ∈ (0,1)
        self._alpha = nn.Parameter(torch.log(torch.tensor(alpha_init / (1 - alpha_init))))  # logit

        # 1x1 map from edge magnitude -> gate ∈ [0,1], init so gate is *smaller* on edges
        self.edge_to_gate = nn.Conv2d(1, 1, kernel_size=1, bias=True)
        with torch.no_grad():
            self.edge_to_gate.weight.fill_(-2.0)  # negative slope: high edge -> low gate
            self.edge_to_gate.bias.fill_(+1.0)    # shift so smooth areas start near ~0.73

    def _edge_from_hint(self, x: torch.Tensor, hint: torch.Tensor | None) -> torch.Tensor:
        """
        Return a normalized edge map in [0,1], shape (B,1,H,W).
        If 'hint' is None, fall back to a weak edge proxy from x.
        """
        if hint is None:
            # weak fallback: channel-abs-mean of x
            e = x.abs().mean(dim=1, keepdim=True)
        else:
            e = hint
            if e.shape[-2:] != x.shape[-2:]:
                e = _resize_like(e, x)
            if e.dim() == 4 and e.size(1) > 1:
                e = e.abs().mean(dim=1, keepdim=True)
            elif e.dim() == 3:
                e = e.unsqueeze(1).abs()
            else:
                e = e.abs()

        # per-sample normalize to [0,1] (robust)
        e = e / (e.mean(dim=(2,3), keepdim=True) + 1e-6)
        e = e.clamp_(0.0, 3.0) / 3.0
        return e

    def forward(self, x: torch.Tensor, edge_hint: torch.Tensor | None = None) -> torch.Tensor:
        y = self.cbam(x)
        edge = self._edge_from_hint(x, edge_hint)              # (B,1,H,W) in [0,1]
        gate = torch.sigmoid(self.edge_to_gate(edge))          # (B,1,H,W)
        alpha = torch.sigmoid(self._alpha)                     # scalar ∈ (0,1)
        return x + alpha * gate * (y - x)


# ---------- Full model: UNet + MSU + Edge-Gated CBAM ----------
class UNetWithMSUSkipsCBAM_EG(nn.Module):
    """
    Same wiring as your UNet+MSU+CBAM, but replace plain CBAM on skips with
    Edge-Aware Residual CBAM that uses MSU maps as *edge hints*:

      d1: hint = MSU(s4, b)
      d2: hint = A34
      d3: hint = P2334
      d4: hint = Qlast

    This keeps CBAM strong in smooth/background (↑SPE) but soft on vessel edges (protects SEN/clDice).
    """
    def __init__(self, in_channels: int = 1,
                 cbam_reduction: int = 16,
                 # spatial CBAM only at the finest level (often best for precision)
                 use_spatial_d1: bool = False,
                 use_spatial_d2: bool = False,
                 use_spatial_d3: bool = False,
                 use_spatial_d4: bool = True,
                 cbam_on_bottleneck: bool = True):
        super().__init__()

        # ---- base UNet (unchanged) ----
        self.e1 = EncoderBlock(in_channels, 64)
        self.e2 = EncoderBlock(64,   128)
        self.e3 = EncoderBlock(128,  256)
        self.e4 = EncoderBlock(256,  512)

        self.bottleneck = ConvBlock(512, 1024)

        self.d1 = DecoderBlock(1024, 512)  # expects skip 512
        self.d2 = DecoderBlock(512,  256)  # expects skip 256
        self.d3 = DecoderBlock(256,  128)  # expects skip 128
        self.d4 = DecoderBlock(128,   64)  # expects skip 64

        self.final = nn.Conv2d(64, 1, kernel_size=1)

        # ---- MSU graph (unchanged) ----
        self.msu_A12   = AlignMSU(inA=64,  inB=128, out_ch=64)
        self.msu_A23   = AlignMSU(inA=128, inB=256, out_ch=128)
        self.msu_A34   = AlignMSU(inA=256, inB=512, out_ch=256)
        self.msu_P1223 = AlignMSU(inA=64,  inB=128, out_ch=64)
        self.msu_P2334 = AlignMSU(inA=128, inB=256, out_ch=128)
        self.msu_Qlast = AlignMSU(inA=64,  inB=128, out_ch=64)

        # (NEW) MSU at E4 scale to provide a hint for d1 (coarsest)
        self.msu_top   = AlignMSU(inA=512, inB=1024, out_ch=512)

        # ---- 1x1 adapters to match decoder skip channels ----
        self.skip_d2_proj = nn.Conv2d(256 + 256, 256, kernel_size=1, bias=True)  # [E3, A34] -> 256
        self.skip_d3_proj = nn.Conv2d(128 * 3,   128, kernel_size=1, bias=True)  # [E2, A23, P2334] -> 128
        self.skip_d4_proj = nn.Conv2d(64  * 4,    64, kernel_size=1, bias=True)  # [E1, A12, P1223, Qlast] -> 64

        # ---- Edge-gated residual CBAM on skip tensors ----
        self.eg_cbam_d1 = EdgeAwareResidualCBAM(512, reduction=cbam_reduction, use_spatial=use_spatial_d1, alpha_init=0.25)
        self.eg_cbam_d2 = EdgeAwareResidualCBAM(256, reduction=cbam_reduction, use_spatial=use_spatial_d2, alpha_init=0.25)
        self.eg_cbam_d3 = EdgeAwareResidualCBAM(128, reduction=cbam_reduction, use_spatial=use_spatial_d3, alpha_init=0.20)
        self.eg_cbam_d4 = EdgeAwareResidualCBAM( 64, reduction=cbam_reduction, use_spatial=use_spatial_d4, alpha_init=0.15)

        # Optional: mild CBAM on bottleneck (channel-only)
        self.cbam_bott = CBAM(1024, reduction_ratio=cbam_reduction, use_spatial=False) \
                         if cbam_on_bottleneck else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ---- encoders ----
        s1, p1 = self.e1(x)    # 64,   H
        s2, p2 = self.e2(p1)   # 128,  H/2
        s3, p3 = self.e3(p2)   # 256,  H/4
        s4, p4 = self.e4(p3)   # 512,  H/8

        # ---- bottleneck (+ optional channel-only CBAM) ----
        b = self.bottleneck(p4)     # 1024, H/8
        b = self.cbam_bott(b)

        # ---- MSU graph (unchanged) ----
        A12   = self.msu_A12(s1, s2)        # (64, H)
        A23   = self.msu_A23(s2, s3)        # (128, H/2)
        A34   = self.msu_A34(s3, s4)        # (256, H/4)
        P1223 = self.msu_P1223(A12, A23)    # (64, H)
        P2334 = self.msu_P2334(A23, A34)    # (128, H/2)
        Qlast = self.msu_Qlast(P1223, P2334)# (64, H)
        TopMSU = self.msu_top(s4, b)        # (512, H/8)  # d1 hint

        # ---- build / refine skips with edge-gated CBAM ----
        # d1 skip: start from E4, edge hint = MSU(s4, b)
        s4_ref = self.eg_cbam_d1(s4, edge_hint=TopMSU)  # (512, H/8)

        # d2 skip: fuse [E3, A34] then refine; edge hint = A34
        skip_d2 = self.skip_d2_proj(torch.cat([s3, A34], dim=1))  # (256, H/4)
        skip_d2 = self.eg_cbam_d2(skip_d2, edge_hint=A34)

        # d3 skip: fuse [E2, A23, P2334]; edge hint = P2334
        skip_d3 = self.skip_d3_proj(torch.cat([s2, A23, P2334], dim=1))  # (128, H/2)
        skip_d3 = self.eg_cbam_d3(skip_d3, edge_hint=P2334)

        # d4 skip: fuse [E1, A12, P1223, Qlast]; edge hint = Qlast
        skip_d4 = self.skip_d4_proj(torch.cat([s1, A12, P1223, Qlast], dim=1))  # (64, H)
        skip_d4 = self.eg_cbam_d4(skip_d4, edge_hint=Qlast)

        # ---- decoders (unchanged) ----
        d1 = self.d1(b,  s4_ref)   # -> 512
        d2 = self.d2(d1, skip_d2)  # -> 256
        d3 = self.d3(d2, skip_d3)  # -> 128
        d4 = self.d4(d3, skip_d4)  # ->  64

        return self.final(d4)      # logits
