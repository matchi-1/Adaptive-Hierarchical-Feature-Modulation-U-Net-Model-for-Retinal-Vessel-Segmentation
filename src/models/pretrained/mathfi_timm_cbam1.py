# mathfi_timm.py
# added cbam in encoder levels
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model

from src.models.dpcn.dpcn_v2 import DPCN
from src.models.dpcn.dpcn_v3 import DPCN as DPCN_tamed
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
    
# --- 3) Make _resize_like debug-friendly (so you catch any future 5-D early) ---
def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if x.dim() != 4 or ref.dim() != 4:
        raise ValueError(f"_resize_like expects 4-D tensors, got {x.shape=} and {ref.shape=}")
    if x.shape[-2:] != ref.shape[-2:]:
        x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
    return x


class MATHFI_TimmEncoder(nn.Module):
    """
    MATHFI with a pretrained encoder from timm.
    Keeps your MSU graph, HAS-Skip, ASFG, DecoderBlocks, final head.
    Optionally fuses DPCN-VAT into the shallowest encoder stage.
    """
    def __init__(self,
                 encoder_name: str = "res2net50_26w_4s",
                 use_dpcn: bool = True,
                 dpcn_ch: int = 64, dpcn_iters: int = 6,
                 cbam_reduction: int = 16,
                 use_tamed_dpcn: bool = False):
        super().__init__()

        # === 0) Input adapter ===
        self.g2r = GrayToRGB()  # 1→3

        # === 1) Pretrained encoder that returns pyramid features ===
        # out_indices=(0,1,2,3) → [C1, C2, C3, C4] strides [2,4,8,16] typically
        self.encoder = create_model(encoder_name, pretrained=True, features_only=True,
                                    out_indices=(0,1,2,3))
        enc_chs = self.encoder.feature_info.channels()  # e.g., [64, 256, 512, 1024]
        C1, C2, C3, C4 = enc_chs

        # === 2) DPCN-VAT on shallowest stage ===
        self.use_dpcn = use_dpcn                      # <-- remember the flag
        self._dpcn_reduce_to_c1 = None                # <-- lazy reducer for (rare) 5-D case

        if self.use_dpcn:
            if use_tamed_dpcn:
                dpcn_out_ch = 64                      # MUST match channels below
                self.dpcn = DPCN_tamed(
                    in_ch=1, channels=dpcn_out_ch, iters=6,
                    threshold_mode="scaled_vat",
                    half_life=1.2,
                    vconf_from="x",
                    use_deformable=True,
                    vconf_gamma=1.6,
                    vconf_floor=0.30,
                    vconf_avgpool_erode=True,
                    gain_mode="tanh_exp",
                    smooth_E_twice=True,
                    aggregate="mean"             # 4-D output
                )
            else:
                dpcn_out_ch = dpcn_ch
                self.dpcn = DPCN(
                    in_ch=1, channels=dpcn_out_ch, iters=dpcn_iters,
                    threshold_mode="scaled_vat", half_life=2.0,
                    aggregate="mean"                  # 4-D output
                )

            # define fuse_c1 OUTSIDE the if/else so it exists for both variants
            self.fuse_c1 = FuseCat1x1(inA=C1, inB=dpcn_out_ch, out_ch=C1)


        # === 3) Bottleneck mock (we’ll use encoder C4 as "p4"); create a "bottleneck" conv like your UNet ===
        self.bottleneck = ConvBlock(C4, C4*2)  # like your 512→1024; adjust if needed
        B = C4*2  # bottleneck channels

        # after computing C1,C2,C3,C4 (and before HAS/MSU wiring)
        # Use plain CBAM (no residual mixing) on encoder feature maps
        self.enc_cbam = nn.ModuleList([
            CBAM(C1, reduction_ratio=cbam_reduction, use_spatial=True),   # s1: spatial+channel (boundary-heavy)
            CBAM(C2, reduction_ratio=cbam_reduction, use_spatial=False),  # s2: channel-only
            CBAM(C3, reduction_ratio=cbam_reduction, use_spatial=False),  # s3: channel-only
            CBAM(C4, reduction_ratio=cbam_reduction, use_spatial=False),  # s4: channel-only
        ])


        # === 4) Decoder path widths (mirror your UNet) ===
        # D1 expects 512-like width, etc. We'll map from encoder channels.
        d1_ch, d2_ch, d3_ch, d4_ch = C4, C3, C2, C1  # coarse→fine
        self.d1 = DecoderBlock(B,  d1_ch)
        self.d2 = DecoderBlock(d1_ch, d2_ch)
        self.d3 = DecoderBlock(d2_ch, d3_ch)
        self.d4 = DecoderBlock(d3_ch, d4_ch)

        # === 5) MSU graph (use encoder widths) ===
        # ==== MSU chain per paper (A, P, Q) ====
        # A* must output at the spatial size of the first arg (inA)
        self.msu_A12   = AlignMSU(inA=C1, inB=C2, out_ch=C1)  # -> size S1, C1 ch
        self.msu_A23   = AlignMSU(inA=C2, inB=C3, out_ch=C2)  # -> size S2, C2 ch
        self.msu_A34   = AlignMSU(inA=C3, inB=C4, out_ch=C3)  # -> size S3, C3 ch

        # P* keep paper’s intent: P12_23 at S1, P23_34 at S2
        self.msu_P12_23 = AlignMSU(inA=C1, inB=C2, out_ch=C1)  # MSU(A12, A23) -> size S1, C1 ch
        self.msu_P23_34 = AlignMSU(inA=C2, inB=C3, out_ch=C2)  # MSU(A23, A34) -> size S2, C2 ch

        # Qlast at S1 (used only at finest)
        self.msu_Qlast  = AlignMSU(inA=C1, inB=C2, out_ch=C1)  # MSU(P12_23, P23_34) -> size S1, C1 ch

        # ==== 1×1 projection heads for Step D ====
        # d1: only S4
        self.proj_d1 = nn.Conv2d(C4, d1_ch, kernel_size=1, bias=True) if C4 != d1_ch else nn.Identity()
        # d2: concat(A34[S3: C3], S3[C3]) -> 2*C3 → d2_ch
        self.proj_d2 = nn.Conv2d(2 * C3, d2_ch, kernel_size=1, bias=True)
        # d3: concat(A23[S2: C2], P23_34[S2: C2], S2[C2]) -> 3*C2 → d3_ch
        self.proj_d3 = nn.Conv2d(3 * C2, d3_ch, kernel_size=1, bias=True)
        # d4: concat(A12[S1: C1], P12_23[S1: C1], Qlast[S1: C1], S1[C1]) -> 4*C1 → d4_ch
        self.proj_d4 = nn.Conv2d(4 * C1, d4_ch, kernel_size=1, bias=True)


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

        # DPCN-VAT fused into s1
        # === 2) DPCN-VAT on shallowest stage ===
        if self.use_dpcn:
            dpcn_feat = self.dpcn(x1chw)  # expected [N, dpcn_out_ch, H, W] (aggregate='mean')
            if dpcn_feat.dim() == 5:       # safety for accidental [N,T,C,H,W]
                N, T, C, H, W = dpcn_feat.shape
                dpcn_feat = dpcn_feat.view(N, T*C, H, W)
                if self._dpcn_reduce_to_c1 is None:
                    self._dpcn_reduce_to_c1 = nn.Conv2d(T*C, C, kernel_size=1, bias=True).to(dpcn_feat.device)
                dpcn_feat = self._dpcn_reduce_to_c1(dpcn_feat)
            s1 = self.fuse_c1(s1, dpcn_feat)


        # ---- NEW: apply encoder CBAMs ----
        s1 = self.enc_cbam[0](s1)
        s2 = self.enc_cbam[1](s2)
        s3 = self.enc_cbam[2](s3)
        s4 = self.enc_cbam[3](s4)

        # bottleneck
        b = self.bottleneck(s4)              # [N, B, H/16, W/16]

        # MSU graph (coarse helpers, similar semantics to your graph)
        # === Step A: adjacent MSUs ===
        A12 = self.msu_A12(s1, s2)   # size S1, C1
        A23 = self.msu_A23(s2, s3)   # size S2, C2
        A34 = self.msu_A34(s3, s4)   # size S3, C3

        # === Step B: pair-of-pairs ===
        P12_23 = self.msu_P12_23(A12, A23)   # size S1, C1
        P23_34 = self.msu_P23_34(A23, A34)   # size S2, C2

        # === Step C: final pre-fine feature ===
        Qlast = self.msu_Qlast(P12_23, P23_34)  # size S1, C1

        # === Step D: per-decoder MSU features (resize+concat+1x1) ===
        # d1 (coarsest): Fd1MSU = S4
        FMSU_d1 = self.proj_d1(s4)  # size S4, d1_ch

        # d2: Fd2MSU = Conv1x1(Concat(A34@S3, S3@S3))
        A34_s3 = _resize_like(A34, s3)
        FMSU_d2 = self.proj_d2(torch.cat([A34_s3, s3], dim=1))  # size S3, d2_ch

        # d3: Fd3MSU = Conv1x1(Concat(A23@S2, P23_34@S2, S2@S2))
        A23_s2    = _resize_like(A23,    s2)
        P2334_s2  = _resize_like(P23_34, s2)
        FMSU_d3   = self.proj_d3(torch.cat([A23_s2, P2334_s2, s2], dim=1))  # size S2, d3_ch

        # d4 (finest): Fd4MSU = Conv1x1(Concat(A12@S1, P12_23@S1, Qlast@S1, S1@S1))
        A12_s1    = _resize_like(A12,    s1)
        P1223_s1  = _resize_like(P12_23, s1)
        Qlast_s1  = _resize_like(Qlast,  s1)
        FMSU_d4   = self.proj_d4(torch.cat([A12_s1, P1223_s1, Qlast_s1, s1], dim=1))  # size S1, d4_ch


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

        # --- final head ---
        logits = self.final(d4)

        # --- ensure logits match input HxW (and mask) ---
        if logits.shape[-2:] != x1chw.shape[-2:]:
            logits = F.interpolate(logits, size=x1chw.shape[-2:], mode="bilinear", align_corners=False)

        return logits