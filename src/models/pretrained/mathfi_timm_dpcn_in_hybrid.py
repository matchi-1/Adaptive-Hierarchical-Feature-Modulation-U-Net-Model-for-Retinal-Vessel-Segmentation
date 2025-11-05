import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from src.models.pretrained.bridges.bridges import GrayToRGB
from src.models.dpcn.dpcn_snake import DPCN           # <-- snake-aware DPCN
from src.models.blocks.msu import MSU
from src.models.blocks.cbam import CBAM
from src.models.unet_exp.base_unet_ablations.base_unet_msu_cbam_hasskip_improved3 import (
    HASSkip, AdaptiveSelectiveFusionGate
)
from src.models.unet import DecoderBlock, ConvBlock


class AlignMSU(nn.Module):
    """Align then MSU (compatible with AlignMSU)."""
    def __init__(self, inA, inB, out_ch, use_bn=True, activation=True):
        super().__init__()
        self.projA = nn.Conv2d(inA, out_ch, 1, bias=True)
        self.projB = nn.Conv2d(inB, out_ch, 1, bias=True)
        self.msu   = MSU(in_channels=out_ch, out_channels=out_ch, use_bn=use_bn, activation=activation)

    def forward(self, A, B):
        A_ = self.projA(A)
        B_ = self.projB(B)
        if B_.shape[-2:] != A_.shape[-2:]:
            B_ = F.interpolate(B_, size=A_.shape[-2:], mode='bilinear', align_corners=False)
        return self.msu(A_, B_)


def _resize_like(x, ref):
    if x.shape[-2:] != ref.shape[-2:]:
        x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
    return x


class DPCNStackedInput_MATHFI_Timm(nn.Module):
    """
    Variant where the *only* input to the timm encoder is the concatenated
    DPCN iterations.

        x [N,1,H,W]
          └─ DPCN (aggregate='stack') → ys [N,T,Cd,H,W]
               → reshape [N, T*Cd, H, W]
               → stem 1×1 conv → [N, 3, H, W]
               └─ timm encoder (pretrained, in_chans=3)
                    → MSU + HAS-Skip + ASFG + decoder (MATHFI)
                    → final logits [N,1,H,W]

    So the raw image is never fed directly to timm; all info goes through DPCN.
    """

    def __init__(
        self,
        encoder_name: str = "res2net50_26w_4s",
        in_ch: int = 1,                 # image channels
        dpcn_channels: int = 32,        # DPCN internal channels (Cd)
        dpcn_iters: int = 4,            # DPCN iterations (T)
        dpcn_threshold_mode: str = "scaled_vat",
        dpcn_half_life: float = 2.0,
        dpcn_conv_type: str = "deform", # "deform" | "snake" (depending on DPCN_snake)
        dpcn_use_deformable: bool | None = None,
        cbam_reduction: int = 16,
    ):
        super().__init__()

        self.T  = int(dpcn_iters)
        self.Cd = int(dpcn_channels)
        
        self.gray_to_rgb = GrayToRGB()  # 1 -> 3

        in_ch_dpcn = self.T * self.Cd

        # project DPCN stack to 3 channels
        self.dpcn_stem = nn.Conv2d(in_ch_dpcn, 3, kernel_size=1, bias=True)

        # tiny fusion conv after summing raw+enhanced
        self.fuse_rgb = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=True)


        # -------------------------
        # 1) DPCN front-end (stack)
        # -------------------------
        self.dpcn = DPCN(
            in_ch=in_ch,
            channels=self.Cd,
            iters=self.T,
            threshold_mode=dpcn_threshold_mode,
            half_life=dpcn_half_life,
            aggregate="stack",        # <--- KEEP ALL ITERATIONS
            conv_type=dpcn_conv_type, # uses deform/snake inside
            use_deformable=dpcn_use_deformable,
        )

        in_ch_dpcn = self.T * self.Cd  # after concatenation

        # Compress DPCN stack to 3 channels so we can feed a standard pretrained timm encoder.
        self.dpcn_stem = nn.Conv2d(in_ch_dpcn, 3, kernel_size=1, bias=True)

        # -------------------------
        # 2) timm encoder (pretrained)
        # -------------------------
        self.encoder = create_model(
            encoder_name,
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            in_chans=3,              # we feed 3-channel pseudo-RGB from DPCN stack
        )
        enc_chs = self.encoder.feature_info.channels()   # e.g. [64,256,512,1024]
        C1, C2, C3, C4 = enc_chs

        # -------------------------
        # 3) Bottleneck + decoder
        # -------------------------
        self.bottleneck = ConvBlock(C4, C4 * 2)  #  512→1024
        B = C4 * 2

        d1_ch, d2_ch, d3_ch, d4_ch = C4, C3, C2, C1  # coarse→fine
        self.d1 = DecoderBlock(B,    d1_ch)
        self.d2 = DecoderBlock(d1_ch, d2_ch)
        self.d3 = DecoderBlock(d2_ch, d3_ch)
        self.d4 = DecoderBlock(d3_ch, d4_ch)

        # -------------------------
        # 4) MSU graph (same pattern)
        # -------------------------
        self.msu_A12   = AlignMSU(inA=C1, inB=C2, out_ch=C1)
        self.msu_A23   = AlignMSU(inA=C2, inB=C3, out_ch=C2)
        self.msu_A34   = AlignMSU(inA=C3, inB=C4, out_ch=C3)

        self.msu_P12_23 = AlignMSU(inA=C1, inB=C2, out_ch=C1)
        self.msu_P23_34 = AlignMSU(inA=C2, inB=C3, out_ch=C2)
        self.msu_Qlast  = AlignMSU(inA=C1, inB=C2, out_ch=C1)

        self.proj_d1 = nn.Conv2d(C4, d1_ch, kernel_size=1, bias=True) if C4 != d1_ch else nn.Identity()
        self.proj_d2 = nn.Conv2d(2 * C3, d2_ch, kernel_size=1, bias=True)
        self.proj_d3 = nn.Conv2d(3 * C2, d3_ch, kernel_size=1, bias=True)
        self.proj_d4 = nn.Conv2d(4 * C1, d4_ch, kernel_size=1, bias=True)

        # -------------------------
        # 5) HAS-Skip + ASFG (unchanged)
        # -------------------------
        self.has = HASSkip(
            Cin_list=(C1, C2, C3, C4),
            Cout_list=(d1_ch, d2_ch, d3_ch, d4_ch),
            Cdec_list=(B, d1_ch, d2_ch, d3_ch),
        )

        self.asfg_d1 = AdaptiveSelectiveFusionGate(
            channels=d1_ch, reduction=cbam_reduction,
            use_spatial_cbam=False, tau=1.8,
            prior_logits=(+0.7, +0.2, -0.3),
            edge_boost_gain=0.5, agree_boost_gain=0.3,
        )
        self.asfg_d2 = AdaptiveSelectiveFusionGate(
            channels=d2_ch, reduction=cbam_reduction,
            use_spatial_cbam=False, tau=1.6,
            prior_logits=(+0.4, +0.2,  0.0),
            edge_boost_gain=0.4, agree_boost_gain=0.3,
        )
        self.asfg_d3 = AdaptiveSelectiveFusionGate(
            channels=d3_ch, reduction=cbam_reduction,
            use_spatial_cbam=False, tau=1.4,
            prior_logits=(+0.1, +0.3, +0.1),
            edge_boost_gain=0.3, agree_boost_gain=0.35,
        )
        self.asfg_d4 = AdaptiveSelectiveFusionGate(
            channels=d4_ch, reduction=cbam_reduction,
            use_spatial_cbam=True, tau=1.2,
            prior_logits=(-0.2, +0.2, +0.7),
            edge_boost_gain=0.0, agree_boost_gain=0.5,
        )

        # -------------------------
        # 6) Final head
        # -------------------------
        self.final = nn.Conv2d(d4_ch, 1, kernel_size=1)

    def forward(self, x1chw: torch.Tensor, fov: torch.Tensor | None = None) -> torch.Tensor:
        """
        x1chw: [N,1,H,W] raw image
        fov:   [N,1,H,W] optional mask 
        """

        # 1) DPCN front-end: [N, T, Cd, H, W]
        ys = self.dpcn(x1chw, fov=fov)

        if ys.dim() != 5:
            raise RuntimeError(f"DPCN with aggregate='stack' must return 5-D, got {ys.shape}")

        N, T, C, H, W = ys.shape
        assert T == self.T and C == self.Cd, f"Unexpected DPCN shape: got T={T},C={C}, expected T={self.T},C={self.Cd}"

        # concat iterations and channels → [N, T*Cd, H, W]
        feats = ys.reshape(N, T * C, H, W)

        # raw grayscale → 3ch
        raw_rgb = self.gray_to_rgb(x1chw)        # [N,3,H,W]

        # DPCN stack → 3ch
        dpcn_rgb = self.dpcn_stem(feats)         # [N,3,H,W]

        # hybrid: keep the raw as a backbone, let DPCN refine
        x3chw = raw_rgb + dpcn_rgb               # [N,3,H,W]
        x3chw = self.fuse_rgb(x3chw)             # light 3×3 refinement

        # 2) timm encoder pyramid
        s1, s2, s3, s4 = self.encoder(x3chw)


        if ys.dim() != 5:
            raise RuntimeError(f"DPCN with aggregate='stack' must return 5-D, got {ys.shape}")

        N, T, C, H, W = ys.shape
        assert T == self.T and C == self.Cd, f"Unexpected DPCN shape: got T={T},C={C}, expected T={self.T},C={self.Cd}"

        # concat iterations and channels → [N, T*Cd, H, W]
        feats = ys.reshape(N, T * C, H, W)

        # reduce to 3 channels (pseudo-RGB for timm)
        x3chw = self.dpcn_stem(feats)   # [N,3,H,W]

        # 2) timm encoder pyramid
        s1, s2, s3, s4 = self.encoder(x3chw)  # [C1,C2,C3,C4]

        # 3) bottleneck
        b = self.bottleneck(s4)              # [N,B,H/16,W/16]

        # 4) MSU graph
        A12 = self.msu_A12(s1, s2)
        A23 = self.msu_A23(s2, s3)
        A34 = self.msu_A34(s3, s4)

        P12_23 = self.msu_P12_23(A12, A23)
        P23_34 = self.msu_P23_34(A23, A34)
        Qlast  = self.msu_Qlast(P12_23, P23_34)

        FMSU_d1 = self.proj_d1(s4)

        A34_s3   = _resize_like(A34,    s3)
        FMSU_d2  = self.proj_d2(torch.cat([A34_s3, s3], dim=1))

        A23_s2   = _resize_like(A23,    s2)
        P2334_s2 = _resize_like(P23_34, s2)
        FMSU_d3  = self.proj_d3(torch.cat([A23_s2, P2334_s2, s2], dim=1))

        A12_s1   = _resize_like(A12,    s1)
        P1223_s1 = _resize_like(P12_23, s1)
        Qlast_s1 = _resize_like(Qlast,  s1)
        FMSU_d4  = self.proj_d4(torch.cat([A12_s1, P1223_s1, Qlast_s1, s1], dim=1))

        # 5) Decoder with HAS-Skip + ASFG
        FSKIP_d1 = self.has.forward_level(0, [s1, s2, s3, s4], b,  s4)
        FB1      = self.asfg_d1(FMSU_d1, FSKIP_d1)
        d1       = self.d1(b, FB1)

        FSKIP_d2 = self.has.forward_level(1, [s1, s2, s3, s4], d1, s3)
        FB2      = self.asfg_d2(FMSU_d2, FSKIP_d2)
        d2       = self.d2(d1, FB2)

        FSKIP_d3 = self.has.forward_level(2, [s1, s2, s3, s4], d2, s2)
        FB3      = self.asfg_d3(FMSU_d3, FSKIP_d3)
        d3       = self.d3(d2, FB3)

        FSKIP_d4 = self.has.forward_level(3, [s1, s2, s3, s4], d3, s1)
        FB4      = self.asfg_d4(FMSU_d4, FSKIP_d4)
        d4       = self.d4(d3, FB4)

        logits = self.final(d4)

        # make sure logits align with input H×W
        if logits.shape[-2:] != x1chw.shape[-2:]:
            logits = F.interpolate(logits, size=x1chw.shape[-2:], mode="bilinear", align_corners=False)

        return logits
