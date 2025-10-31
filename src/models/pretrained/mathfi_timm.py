# mathfi_timm.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model

from src.models.dpcn.dpcn_v2 import DPCN
from src.models.blocks.msu import MSU
from src.models.blocks.cbam import CBAM
from src.models.unet_exp.base_unet_ablations.base_unet_msu_cbam_hasskip_improved3 import HASSkip, AdaptiveSelectiveFusionGate  
from src.models.unet import DecoderBlock, ConvBlock  
from src.models.pretrained.bridges.bridges import GrayToRGB, FuseCat1x1

class AlignMSU(nn.Module):
    """Align then MSU (compatible with your AlignMSU)."""
    def __init__(self, inA, inB, out_ch, use_bn=True, activation=True):
        super().__init__()
        self.projA = nn.Conv2d(inA, out_ch, 1, bias=True)
        self.projB = nn.Conv2d(inB, out_ch, 1, bias=True)
        self.msu   = MSU(in_channels=out_ch, out_channels=out_ch, use_bn=use_bn, activation=activation)
    def forward(self, A, B):
        A_ = self.projA(A); B_ = self.projB(B)
        if B_.shape[-2:] != A_.shape[-2:]:
            B_ = F.interpolate(B_, size=A_.shape[-2:], mode='bilinear', align_corners=False)
        return self.msu(A_, B_)

class MATHFI_TimmEncoder(nn.Module):
    """
    MATHFI with a pretrained encoder from timm.
    Keeps your MSU graph, HAS-Skip, ASFG, DecoderBlocks, final head.
    Optionally fuses DPCN-VAT into the shallowest encoder stage.
    """
    def __init__(self,
                 encoder_name: str = "res2net50_26w_4s",
                 use_dpcn: bool = True,
                 dpcn_ch: int = 32, dpcn_iters: int = 4,
                 cbam_reduction: int = 16):
        super().__init__()

        # === 0) Input adapter ===
        self.g2r = GrayToRGB()  # 1→3

        # === 1) Pretrained encoder that returns pyramid features ===
        # out_indices=(0,1,2,3) → [C1, C2, C3, C4] strides [2,4,8,16] typically
        self.encoder = create_model(encoder_name, pretrained=True, features_only=True,
                                    out_indices=(0,1,2,3))
        enc_chs = self.encoder.feature_info.channels()  # e.g., [64, 256, 512, 1024]
        C1, C2, C3, C4 = enc_chs

        # === 2) Optional DPCN-VAT on shallowest stage ===
        self.use_dpcn = use_dpcn
        if use_dpcn:
            # run DPCN on raw input and fuse into C1
            self.dpcn = DPCN(in_ch=1, channels=dpcn_ch, iters=dpcn_iters,
                             threshold_mode="scaled_vat", half_life=2.0, aggregate="mean")
            self.fuse_c1 = FuseCat1x1(inA=C1, inB=dpcn_ch, out_ch=C1)  # keep channel count stable

        # === 3) Bottleneck mock (we’ll use encoder C4 as "p4"); create a "bottleneck" conv like your UNet ===
        self.bottleneck = ConvBlock(C4, C4*2)  # like your 512→1024; adjust if needed
        B = C4*2  # bottleneck channels

        # === 4) Decoder path widths (mirror your UNet) ===
        # D1 expects 512-like width, etc. We'll map from encoder channels.
        d1_ch, d2_ch, d3_ch, d4_ch = C4, C3, C2, C1  # coarse→fine
        self.d1 = DecoderBlock(B,  d1_ch)
        self.d2 = DecoderBlock(d1_ch, d2_ch)
        self.d3 = DecoderBlock(d2_ch, d3_ch)
        self.d4 = DecoderBlock(d3_ch, d4_ch)

        # === 5) MSU graph (use encoder widths) ===
        self.msu_top   = AlignMSU(inA=C4, inB=B, out_ch=d1_ch)   # for d1
        self.msu_A23   = AlignMSU(inA=C2, inB=C3, out_ch=d2_ch)  # for d2
        self.msu_A34   = AlignMSU(inA=C3, inB=C4, out_ch=d1_ch)  # helper
        self.msu_P2334 = AlignMSU(inA=d2_ch, inB=d1_ch, out_ch=d3_ch)
        self.msu_Qlast = AlignMSU(inA=d3_ch, inB=d3_ch, out_ch=d4_ch)

        # For clarity: MSU features per level (coarse→fine)
        # FMSU_d1 uses (s4,b), FMSU_d2 uses (s3,s4) or (A23), FMSU_d3 uses P2334, FMSU_d4 uses Qlast.

        # === 6) HAS-Skip using dynamic Cin_list/Cout_list/Cdec_list from the encoder ===
        self.has = HASSkip(
            Cin_list=(C1, C2, C3, C4),
            Cout_list=(d1_ch, d2_ch, d3_ch, d4_ch),
            Cdec_list=(B, d1_ch, d2_ch, d3_ch)
        )

        # === 7) ASFG gates per level (reuse your policies) ===
        self.asfg_d1 = AdaptiveSelectiveFusionGate(channels=d1_ch, reduction=cbam_reduction,
                                                   use_spatial_cbam=False, tau=1.8,
                                                   prior_logits=(+0.7, +0.2, -0.3),
                                                   edge_boost_gain=0.5, agree_boost_gain=0.3)
        self.asfg_d2 = AdaptiveSelectiveFusionGate(channels=d2_ch, reduction=cbam_reduction,
                                                   use_spatial_cbam=False, tau=1.6,
                                                   prior_logits=(+0.4, +0.2,  0.0),
                                                   edge_boost_gain=0.4, agree_boost_gain=0.3)
        self.asfg_d3 = AdaptiveSelectiveFusionGate(channels=d3_ch, reduction=cbam_reduction,
                                                   use_spatial_cbam=False, tau=1.4,
                                                   prior_logits=(+0.1, +0.3, +0.1),
                                                   edge_boost_gain=0.3, agree_boost_gain=0.35)
        self.asfg_d4 = AdaptiveSelectiveFusionGate(channels=d4_ch, reduction=cbam_reduction,
                                                   use_spatial_cbam=True, tau=1.2,
                                                   prior_logits=(-0.2, +0.2, +0.7),
                                                   edge_boost_gain=0.0, agree_boost_gain=0.5)

        # === 8) Final head ===
        self.final = nn.Conv2d(d4_ch, 1, kernel_size=1)

    def forward(self, x1chw):
        # encoder
        x3chw = self.g2r(x1chw)              # 1→3
        s1, s2, s3, s4 = self.encoder(x3chw) # [C1,C2,C3,C4]

        # optional DPCN-VAT fused into s1
        if self.use_dpcn:
            dpcn_feat = self.dpcn(x1chw)     # [N, C_dpcn, H, W], aggregate="mean"
            s1 = self.fuse_c1(s1, dpcn_feat) # keep channels = C1

        # bottleneck
        b = self.bottleneck(s4)              # [N, B, H/16, W/16]

        # MSU graph (coarse helpers, similar semantics to your graph)
        A23   = self.msu_A23(s2, s3)         # -> d2_ch
        A34   = self.msu_A34(s3, s4)         # -> d1_ch
        P2334 = self.msu_P2334(A23, A34)     # -> d3_ch
        Qlast = self.msu_Qlast(P2334, P2334) # -> d4_ch

        FMSU_d1 = self.msu_top(s4, b)        # -> d1_ch
        FMSU_d2 = A23                        # -> d2_ch
        FMSU_d3 = P2334                      # -> d3_ch
        FMSU_d4 = Qlast                      # -> d4_ch

        # Decoder L1 (coarsest)
        FSKIP_d1 = self.has.forward_level(0, [s1, s2, s3, s4], b,  s4)
        FB1      = self.asfg_d1(FMSU_d1, FSKIP_d1)
        d1       = self.d1(b, FB1)

        # L2
        FSKIP_d2 = self.has.forward_level(1, [s1, s2, s3, s4], d1, s3)
        FB2      = self.asfg_d2(FMSU_d2, FSKIP_d2)
        d2       = self.d2(d1, FB2)

        # L3
        FSKIP_d3 = self.has.forward_level(2, [s1, s2, s3, s4], d2, s2)
        FB3      = self.asfg_d3(FMSU_d3, FSKIP_d3)
        d3       = self.d3(d2, FB3)

        # L4 (finest)
        FSKIP_d4 = self.has.forward_level(3, [s1, s2, s3, s4], d3, s1)
        FB4      = self.asfg_d4(FMSU_d4, FSKIP_d4)
        d4       = self.d4(d3, FB4)

        return self.final(d4)
