# src/models/unet_with_msu_cbam.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.unet import ConvBlock, EncoderBlock, DecoderBlock
from src.models.blocks.msu import MSU
from src.models.blocks.cbam import CBAM


def _resize_like(x, ref):
    """Bilinear-resize x to ref's HxW if needed."""
    if x.shape[-2:] != ref.shape[-2:]:
        x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
    return x


class AlignMSU(nn.Module):
    """
    Align two feature maps to an anchor's channels/size, then apply MSU.
    A -> 1x1->C, B -> 1x1->C, resize B to A, MSU(C->C).
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


class HASSkip(nn.Module):
    """
    HAS-Skip per paper (Eq. 2.30–2.35) with two stabilizers:
      • temperature-softmax over encoders (τ) to prevent peaky selection early
      • residual gate with learnable γ (logit) so HAS starts near 'off' and learns to help

    For each decoder level l (d1..d4):
      - Upsample E1..E4 to level size, project to Cout_l
      - Fagg = sum_i softmax(w_i/τ) * Ei_proj
      - G_l = sigmoid(Wg_l(Fdec_l) + Wx_l(Fagg))
      - FSKIP_l = El_proj + σ(γ_l) * (G_l*El_proj - El_proj)      # residual gate
        (σ(γ_l) ∈ (0,1) is a learnable on-knob)
    """
    def __init__(
        self,
        Cin_list=(64,128,256,512),
        Cout_list=(512,256,128,64),      # expected skip chans for d1..d4
        Cdec_list=(1024,512,256,128),    # decoder context chans for d1..d4 (b,d1,d2,d3)
        softmax_tau: float = 2.0,        # τ=2 keeps early mix broad/stable
        init_gamma: float = -4.0         # γ logit init ~ off (sigmoid(-4)≈0.018)
    ):
        super().__init__()
        self.L = 4
        self.Cin_list  = Cin_list
        self.Cout_list = Cout_list
        self.Cdec_list = Cdec_list
        self.softmax_tau = float(softmax_tau)

        # Projections: for each level l, project each encoder Ei -> Cout_l
        self.proj = nn.ModuleList()
        for l in range(self.L):
            Cout = Cout_list[l]
            self.proj.append(nn.ModuleList([
                nn.Conv2d(Cin_list[0], Cout, 1, bias=True),
                nn.Conv2d(Cin_list[1], Cout, 1, bias=True),
                nn.Conv2d(Cin_list[2], Cout, 1, bias=True),
                nn.Conv2d(Cin_list[3], Cout, 1, bias=True),
            ]))

        # Mixture logits (per level) over the 4 encoders; softmaxed with temperature τ
        self.w_logits = nn.ParameterList([nn.Parameter(torch.zeros(4)) for _ in range(self.L)])

        # Gating convs per level
        self.Wg = nn.ModuleList([nn.Conv2d(Cdec_list[l], Cout_list[l], 1, bias=True) for l in range(self.L)])
        self.Wx = nn.ModuleList([nn.Conv2d(Cout_list[l], Cout_list[l], 1, bias=True) for l in range(self.L)])

        # Residual gate strength per level (logit γ_l) → σ(γ_l) ∈ (0,1)
        self.gamma = nn.Parameter(torch.full((self.L,), float(init_gamma)))

    def forward_level(self, level_idx: int,
                      encs,            # [s1,s2,s3,s4]
                      dec_ctx: torch.Tensor,
                      target_ref: torch.Tensor) -> torch.Tensor:
        l = level_idx
        Cout = self.Cout_list[l]

        # 1) Align/project all encoders to level l size/ch
        Ei_proj = []
        for i in range(4):
            x = encs[i]
            x = _resize_like(x, target_ref)
            x = self.proj[l][i](x)  # -> (B, Cout, H_l, W_l)
            Ei_proj.append(x)

        # 2) Temperature-softmax mixture
        w = torch.softmax(self.w_logits[l] / self.softmax_tau, dim=0)
        Fagg = w[0]*Ei_proj[0] + w[1]*Ei_proj[1] + w[2]*Ei_proj[2] + w[3]*Ei_proj[3]

        # 3) Context-conditioned gate
        dec_ctx = _resize_like(dec_ctx, target_ref)
        G = torch.sigmoid(self.Wg[l](dec_ctx) + self.Wx[l](Fagg))  # (B, Cout, H_l, W_l)

        # 4) Gate current level's encoder (d1->E4, d2->E3, d3->E2, d4->E1)
        El_proj = Ei_proj[3 - l]
        gamma_star = torch.sigmoid(self.gamma[l])                  # in (0,1)
        gated = G * El_proj
        FSKIP_l = El_proj + gamma_star * (gated - El_proj)         # residual gate
        return FSKIP_l


class UNetWithMSUSkipsCBAM(nn.Module):
    """
    Full model:
      • Base UNet (unchanged)
      • MSU graph: A12,A23,A34,P1223,P2334,Qlast + MSU(s4,b) for d1
      • CBAM on bottleneck (optional) and optionally on fused skips
      • HAS-Skip with residual gating + τ-softmax encoder mixing
      • Per-level α/β fusion: FB_l = α_l*FMSU_l + β_l*FSKIP_l (α≈1, β≈0 at init)
    """
    def __init__(self,
                 in_channels: int = 1,
                 cbam_reduction: int = 16,
                 cbam_use_spatial: bool = True,
                 cbam_on_bottleneck: bool = True,
                 cbam_on_fused_skips: bool = False,
                 has_tau: float = 2.0,
                 has_init_gamma: float = -4.0):
        super().__init__()

        # ---- base UNet (do not alter) ----
        self.e1 = EncoderBlock(in_channels, 64)
        self.e2 = EncoderBlock(64,   128)
        self.e3 = EncoderBlock(128,  256)
        self.e4 = EncoderBlock(256,  512)

        self.bottleneck = ConvBlock(512, 1024)

        self.d1 = DecoderBlock(1024, 512)   # expects 512-ch skip
        self.d2 = DecoderBlock(512,  256)   # expects 256-ch skip
        self.d3 = DecoderBlock(256,  128)   # expects 128-ch skip
        self.d4 = DecoderBlock(128,   64)   # expects  64-ch skip
        self.final = nn.Conv2d(64, 1, kernel_size=1)

        # ---- MSU graph (unchanged from your spec) ----
        self.msu_A12   = AlignMSU(inA=64,  inB=128, out_ch=64)
        self.msu_A23   = AlignMSU(inA=128, inB=256, out_ch=128)
        self.msu_A34   = AlignMSU(inA=256, inB=512, out_ch=256)
        self.msu_P1223 = AlignMSU(inA=64,  inB=128, out_ch=64)
        self.msu_P2334 = AlignMSU(inA=128, inB=256, out_ch=128)
        self.msu_Qlast = AlignMSU(inA=64,  inB=128, out_ch=64)

        # MSU at the coarsest level so d1 also has an FMSU
        self.msu_top   = AlignMSU(inA=512, inB=1024, out_ch=512)

        # ---- HAS-Skip (with stabilizers) ----
        self.has = HASSkip(
            Cin_list=(64,128,256,512),
            Cout_list=(512,256,128,64),       # skips for d1..d4
            Cdec_list=(1024,512,256,128),     # decoder contexts (b,d1,d2,d3)
            softmax_tau=has_tau,
            init_gamma=has_init_gamma
        )

        # ---- CBAM ----
        self.cbam_bott = CBAM(1024, reduction_ratio=cbam_reduction,
                              use_spatial=cbam_use_spatial) if cbam_on_bottleneck else nn.Identity()

        self.cbam_on_fused_skips = cbam_on_fused_skips
        if cbam_on_fused_skips:
            self.cbam_skip_d1 = CBAM(512,  reduction_ratio=cbam_reduction, use_spatial=cbam_use_spatial)
            self.cbam_skip_d2 = CBAM(256,  reduction_ratio=cbam_reduction, use_spatial=cbam_use_spatial)
            self.cbam_skip_d3 = CBAM(128,  reduction_ratio=cbam_reduction, use_spatial=cbam_use_spatial)
            self.cbam_skip_d4 = CBAM(64,   reduction_ratio=cbam_reduction, use_spatial=cbam_use_spatial)
        else:
            self.cbam_skip_d1 = self.cbam_skip_d2 = self.cbam_skip_d3 = self.cbam_skip_d4 = nn.Identity()

        # ---- α/β fusion (non-negative via softplus) ----
        # store *log* parameters; at runtime use softplus to keep >=0
        # α starts ~1, β starts ~0
        sp_inv_1 = torch.log(torch.exp(torch.tensor(1.0)) - 1.0)   # softplus^{-1}(1) ≈ 0.541
        self.alpha_log = nn.Parameter(torch.full((4,), float(sp_inv_1)))  # softplus(alpha_log) ≈ 1
        self.beta_log  = nn.Parameter(torch.full((4,), -20.0))            # softplus(beta_log)  ≈ 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ---- encoders ----
        s1, p1 = self.e1(x)    # 64,   H
        s2, p2 = self.e2(p1)   # 128,  H/2
        s3, p3 = self.e3(p2)   # 256,  H/4
        s4, p4 = self.e4(p3)   # 512,  H/8

        # ---- bottleneck (+ optional CBAM) ----
        b = self.bottleneck(p4)      # 1024, H/8
        b = self.cbam_bott(b)

        # ---- MSU graph ----
        A12   = self.msu_A12(s1, s2)         # (64, H)
        A23   = self.msu_A23(s2, s3)         # (128, H/2)
        A34   = self.msu_A34(s3, s4)         # (256, H/4)
        P1223 = self.msu_P1223(A12, A23)     # (64, H)
        P2334 = self.msu_P2334(A23, A34)     # (128, H/2)
        Qlast = self.msu_Qlast(P1223, P2334) # (64, H)

        # FMSU per decoder level
        FMSU_d1 = self.msu_top(s4, b)        # (512, H/8)
        FMSU_d2 = A34                        # (256, H/4)
        FMSU_d3 = P2334                      # (128, H/2)
        FMSU_d4 = Qlast                      # ( 64, H)

        # α/β (non-negative)
        alpha = F.softplus(self.alpha_log)
        beta  = F.softplus(self.beta_log)

        # ---- HAS-Skip & fusion level by level ----
        # d1: context=b at H/8, anchor size = s4
        FSKIP_d1 = self.has.forward_level(0, [s1,s2,s3,s4], b,  s4)
        FB1 = alpha[0]*FMSU_d1 + beta[0]*FSKIP_d1
        FB1 = self.cbam_skip_d1(FB1)
        d1  = self.d1(b, FB1)                # (512, H/4)

        # d2: context=d1 at H/4, anchor size = s3
        FSKIP_d2 = self.has.forward_level(1, [s1,s2,s3,s4], d1, s3)
        FB2 = alpha[1]*FMSU_d2 + beta[1]*FSKIP_d2
        FB2 = self.cbam_skip_d2(FB2)
        d2  = self.d2(d1, FB2)               # (256, H/2)

        # d3: context=d2 at H/2, anchor size = s2
        FSKIP_d3 = self.has.forward_level(2, [s1,s2,s3,s4], d2, s2)
        FB3 = alpha[2]*FMSU_d3 + beta[2]*FSKIP_d3
        FB3 = self.cbam_skip_d3(FB3)
        d3  = self.d3(d2, FB3)               # (128, H)

        # d4: context=d3 at H, anchor size = s1
        FSKIP_d4 = self.has.forward_level(3, [s1,s2,s3,s4], d3, s1)
        FB4 = alpha[3]*FMSU_d4 + beta[3]*FSKIP_d4
        FB4 = self.cbam_skip_d4(FB4)
        d4  = self.d4(d3, FB4)               # (64, H)

        return self.final(d4)                # logits
