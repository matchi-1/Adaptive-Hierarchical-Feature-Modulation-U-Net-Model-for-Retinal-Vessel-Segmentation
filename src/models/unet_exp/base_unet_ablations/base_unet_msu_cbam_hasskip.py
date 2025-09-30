# src/models/unet_with_msu_cbam.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.unet import ConvBlock, EncoderBlock, DecoderBlock
from src.models.blocks.msu import MSU
from src.models.blocks.cbam import CBAM


# ---------- Utilities ----------
def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] != ref.shape[-2:]:
        x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
    return x


# ---------- MSU alignment (unchanged) ----------
class AlignMSU(nn.Module):
    """
    Align two feature maps to a chosen 'anchor' resolution/channels, then apply MSU.
    - Projects A and B to out_ch via 1x1 convs.
    - Resizes B to A's spatial size (anchor = first input).
    - Runs MSU(out_ch -> out_ch).
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


# ---------- HAS-Skip (per paper Eq. 2.30–2.35) ----------
class HASSkip(nn.Module):
    """
    For each decoder level l, produce FSKIP_l:
      1) Upsample all encoders to level size; project to C_l.
      2) Aggregate with softmax scalars: Fagg = sum_i softmax(w_l)[i] * Ei_proj
      3) Gate current level encoder feature with decoder context:
            G_l = sigmoid( Wg_l(Fdecoder_l) + Wx_l(Fagg) )
         FSKIP_l = G_l * E_l_proj
    - Cin_list: channels of encoders [C1,C2,C3,C4] = [64,128,256,512]
    - Cout_list: target channels per level [512,256,128,64] for d1..d4
    - Cdec_list: decoder context channels per level [1024,512,256,128] for d1..d4
    """
    def __init__(self,
                 Cin_list=(64,128,256,512),
                 Cout_list=(512,256,128,64),
                 Cdec_list=(1024,512,256,128)):
        super().__init__()
        self.L = 4
        self.Cin_list  = Cin_list
        self.Cout_list = Cout_list
        self.Cdec_list = Cdec_list

        # Projections: for each level l, we need 4 convs (E1..E4 -> Cout_l)
        self.proj = nn.ModuleList()
        for l in range(self.L):
            Cout = Cout_list[l]
            convs_l = nn.ModuleList([
                nn.Conv2d(Cin_list[0], Cout, 1, bias=True),
                nn.Conv2d(Cin_list[1], Cout, 1, bias=True),
                nn.Conv2d(Cin_list[2], Cout, 1, bias=True),
                nn.Conv2d(Cin_list[3], Cout, 1, bias=True),
            ])
            self.proj.append(convs_l)

        # Softmax weights per level (over 4 encoders)
        self.w_logits = nn.ParameterList([nn.Parameter(torch.zeros(4)) for _ in range(self.L)])

        # Gating convs per level
        self.Wg = nn.ModuleList([nn.Conv2d(Cdec_list[l], Cout_list[l], 1, bias=True) for l in range(self.L)])
        self.Wx = nn.ModuleList([nn.Conv2d(Cout_list[l], Cout_list[l], 1, bias=True) for l in range(self.L)])

    def forward_level(self, level_idx: int,
                      encs: list[torch.Tensor],
                      dec_ctx: torch.Tensor,
                      target_ref: torch.Tensor) -> torch.Tensor:
        """
        level_idx: 0..3 -> d1..d4 (coarsest to finest)
        encs: [s1,s2,s3,s4]
        dec_ctx: decoder context feature for this level (B, Cdec[level], H_l, W_l)
        target_ref: tensor whose HxW defines the level size (e.g., s4 for d1, s3 for d2, ...)
        returns: FSKIP_l in (B, Cout_l, H_l, W_l)
        """
        l = level_idx
        Cout = self.Cout_list[l]

        # 1) Align encoders to level size and project to Cout
        Ei_proj = []
        for i in range(4):
            x = encs[i]
            x = _resize_like(x, target_ref)
            x = self.proj[l][i](x)  # (B, Cout, H_l, W_l)
            Ei_proj.append(x)

        # 2) Aggregate with softmax scalars
        w = torch.softmax(self.w_logits[l], dim=0)  # (4,)
        Fagg = w[0]*Ei_proj[0] + w[1]*Ei_proj[1] + w[2]*Ei_proj[2] + w[3]*Ei_proj[3]

        # 3) Context-conditioned gate
        dec_ctx = _resize_like(dec_ctx, target_ref)
        G = torch.sigmoid(self.Wg[l](dec_ctx) + self.Wx[l](Fagg))

        # 4) Gate current level's encoder stream (El is the level's own encoder)
        El_proj = Ei_proj[3-l]  # mapping: d1 uses E4, d2 uses E3, d3 uses E2, d4 uses E1
        FSKIP_l = G * El_proj
        return FSKIP_l


# ---------- Full Model: UNet + MSU + CBAM(bottleneck) + HAS-Skip ----------
class UNetWithMSUSkipsCBAM(nn.Module):
    """
    UNet with:
      • MSU graph (A12, A23, A34, P1223, P2334, Qlast) + top-level MSU(s4,b)
      • CBAM on bottleneck (option)
      • HAS-Skip per decoder level (context-conditioned)
      • Optional CBAM on fused skips (off by default to avoid over-gating)

    Flags:
      cbam_on_bottleneck: apply CBAM to B (Eq. 2.27)
      cbam_on_fused_skips: apply CBAM after (FMSU_l + FSKIP_l) [default False]
    """
    def __init__(self, in_channels: int = 1,
                 cbam_reduction: int = 16,
                 cbam_use_spatial: bool = True,
                 cbam_on_bottleneck: bool = True,
                 cbam_on_fused_skips: bool = False):
        super().__init__()

        # ---- base UNet (unchanged) ----
        self.e1 = EncoderBlock(in_channels, 64)
        self.e2 = EncoderBlock(64,   128)
        self.e3 = EncoderBlock(128,  256)
        self.e4 = EncoderBlock(256,  512)

        self.bottleneck = ConvBlock(512, 1024)

        self.d1 = DecoderBlock(1024, 512)  # expects skip 512   (H/8 -> H/4 inside)
        self.d2 = DecoderBlock(512,  256)  # expects skip 256   (H/4 -> H/2)
        self.d3 = DecoderBlock(256,  128)  # expects skip 128   (H/2 -> H)
        self.d4 = DecoderBlock(128,  64)   # expects skip 64    (H   -> H)
        self.final = nn.Conv2d(64, 1, kernel_size=1)

        # ---- MSU graph (as before) ----
        self.msu_A12   = AlignMSU(inA=64,  inB=128, out_ch=64)
        self.msu_A23   = AlignMSU(inA=128, inB=256, out_ch=128)
        self.msu_A34   = AlignMSU(inA=256, inB=512, out_ch=256)
        self.msu_P1223 = AlignMSU(inA=64,  inB=128, out_ch=64)
        self.msu_P2334 = AlignMSU(inA=128, inB=256, out_ch=128)
        self.msu_Qlast = AlignMSU(inA=64,  inB=128, out_ch=64)

        # NEW: MSU at E4 scale so d1 has an FMSU too (MSU(s4, b))
        self.msu_top   = AlignMSU(inA=512, inB=1024, out_ch=512)

        # ---- HAS-Skip ----
        self.has = HASSkip(
            Cin_list=(64,128,256,512),
            Cout_list=(512,256,128,64),      # skips expected by d1..d4
            Cdec_list=(1024,512,256,128),    # decoder context at d1..d4: b,d1,d2,d3
        )

        # ---- CBAM ----
        self.cbam_bott = CBAM(1024, reduction_ratio=cbam_reduction,
                              use_spatial=cbam_use_spatial) if cbam_on_bottleneck else nn.Identity()

        # optional CBAM after fusion (skip refinement)
        self.cbam_on_fused_skips = cbam_on_fused_skips
        if cbam_on_fused_skips:
            self.cbam_skip_d1 = CBAM(512,  reduction_ratio=cbam_reduction, use_spatial=cbam_use_spatial)
            self.cbam_skip_d2 = CBAM(256,  reduction_ratio=cbam_reduction, use_spatial=cbam_use_spatial)
            self.cbam_skip_d3 = CBAM(128,  reduction_ratio=cbam_reduction, use_spatial=cbam_use_spatial)
            self.cbam_skip_d4 = CBAM(64,   reduction_ratio=cbam_reduction, use_spatial=cbam_use_spatial)
        else:
            self.cbam_skip_d1 = self.cbam_skip_d2 = self.cbam_skip_d3 = self.cbam_skip_d4 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ---- encoders ----
        s1, p1 = self.e1(x)    # 64,   H
        s2, p2 = self.e2(p1)   # 128,  H/2
        s3, p3 = self.e3(p2)   # 256,  H/4
        s4, p4 = self.e4(p3)   # 512,  H/8

        # ---- bottleneck (+ optional CBAM) ----
        b = self.bottleneck(p4)            # 1024, H/8
        b = self.cbam_bott(b)

        # ---- MSU graph ----
        A12   = self.msu_A12(s1, s2)         # (64, H)
        A23   = self.msu_A23(s2, s3)         # (128, H/2)
        A34   = self.msu_A34(s3, s4)         # (256, H/4)
        P1223 = self.msu_P1223(A12, A23)     # (64, H)
        P2334 = self.msu_P2334(A23, A34)     # (128, H/2)
        Qlast = self.msu_Qlast(P1223, P2334) # (64, H)

        # Provide FMSU per decoder level:
        FMSU_d1 = self.msu_top(s4, b)        # (512, H/8)
        FMSU_d2 = A34                        # (256, H/4)
        FMSU_d3 = P2334                      # (128, H/2)
        FMSU_d4 = Qlast                      # ( 64, H)

        # ---- HAS-Skip per level ----
        # d1 level context is b at H/8; target ref is s4
        FSKIP_d1 = self.has.forward_level(level_idx=0, encs=[s1,s2,s3,s4], dec_ctx=b,  target_ref=s4)

        # FBl = FMSU + FSKIP (optionally refine with CBAM), then decode
        FB1 = self.cbam_skip_d1(FMSU_d1 + FSKIP_d1)
        d1  = self.d1(b, FB1)                # out: 512 at H/4

        # d2: context is d1 at H/4; target ref is s3
        FSKIP_d2 = self.has.forward_level(level_idx=1, encs=[s1,s2,s3,s4], dec_ctx=d1, target_ref=s3)
        FB2 = self.cbam_skip_d2(FMSU_d2 + FSKIP_d2)
        d2  = self.d2(d1, FB2)               # out: 256 at H/2

        # d3: context is d2 at H/2; target ref is s2
        FSKIP_d3 = self.has.forward_level(level_idx=2, encs=[s1,s2,s3,s4], dec_ctx=d2, target_ref=s2)
        FB3 = self.cbam_skip_d3(FMSU_d3 + FSKIP_d3)
        d3  = self.d3(d2, FB3)               # out: 128 at H

        # d4: context is d3 at H; target ref is s1
        FSKIP_d4 = self.has.forward_level(level_idx=3, encs=[s1,s2,s3,s4], dec_ctx=d3, target_ref=s1)
        FB4 = self.cbam_skip_d4(FMSU_d4 + FSKIP_d4)
        d4  = self.d4(d3, FB4)               # out: 64 at H

        return self.final(d4)                # logits
