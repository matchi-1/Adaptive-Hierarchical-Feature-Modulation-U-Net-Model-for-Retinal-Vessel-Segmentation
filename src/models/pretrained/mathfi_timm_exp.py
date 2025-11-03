# mathfi_timm_spe.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model

from src.models.dpcn.dpcn_v2 import DPCN
from src.models.blocks.msu import MSU
from src.models.unet_exp.base_unet_ablations.base_unet_msu_cbam_hasskip_improved3 import (
    HASSkip, AdaptiveSelectiveFusionGate
)
from src.models.unet import DecoderBlock, ConvBlock
from src.models.pretrained.bridges.bridges import GrayToRGB, FuseCat1x1


# ---------- helpers ----------
class AlignMSU(nn.Module):
    def __init__(self, inA, inB, out_ch, use_bn=True, activation=True):
        super().__init__()
        self.projA = nn.Conv2d(inA, out_ch, 1, bias=True)
        self.projB = nn.Conv2d(inB, out_ch, 1, bias=True)
        self.msu   = MSU(in_channels=out_ch, out_channels=out_ch, use_bn=use_bn, activation=activation)

    def forward(self, A, B):
        A_ = self.projA(A); B_ = self.projB(B)
        if B_.shape[-2:] != A_.shape[-2:]:
            B_ = F.interpolate(B_, size=A_.shape[-2:], mode="bilinear", align_corners=False)
        return self.msu(A_, B_)


def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] != ref.shape[-2:]:
        x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
    return x


# ---------- Edge-aware final refiner (for SPE) ----------
class EdgeAwareRefiner(nn.Module):
    """
    Builds an attention map from input gradients + a decoder cue, then gates logits to reduce
    background haze / thickness. All ops are AMP-safe.
    """
    def __init__(self, dec_ch: int, k_dw: int = 5, tau: float = 6.0):
        super().__init__()
        pad = k_dw // 2
        # fixed depthwise smoothing (registered as a module so AMP handles casting)
        self.dw_smooth = nn.Conv2d(1, 1, kernel_size=k_dw, padding=pad, groups=1, bias=False)
        with torch.no_grad():
            self.dw_smooth.weight.fill_(1.0 / (k_dw * k_dw))
        for p in self.dw_smooth.parameters():
            p.requires_grad = False

        self.dec_proj = nn.Conv2d(dec_ch, 16, kernel_size=1, bias=True)
        self.att_head = nn.Sequential(
            nn.Conv2d(16 + 1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)
        )
        self.tau = tau

    @staticmethod
    def _grad_mag(x1chw: torch.Tensor) -> torch.Tensor:
        # Create Sobel kernels with the SAME dtype/device as input (AMP-safe).
        kx = torch.tensor([[-1., 0., 1.],
                           [-2., 0., 2.],
                           [-1., 0., 1.]], device=x1chw.device, dtype=x1chw.dtype).view(1,1,3,3)
        ky = torch.tensor([[-1., -2., -1.],
                           [ 0.,  0.,  0.],
                           [ 1.,  2.,  1.]], device=x1chw.device, dtype=x1chw.dtype).view(1,1,3,3)
        gx = F.conv2d(x1chw, kx, padding=1)
        gy = F.conv2d(x1chw, ky, padding=1)
        mag = torch.sqrt(torch.clamp_min(gx * gx + gy * gy, 1e-12))
        mag = mag / (mag.amax(dim=(2,3), keepdim=True) + 1e-6)
        return mag

    def forward(self, logits: torch.Tensor, x1chw: torch.Tensor, dec_feat: torch.Tensor) -> torch.Tensor:
        # match sizes
        if x1chw.shape[-2:] != logits.shape[-2:]:
            x_in = F.interpolate(x1chw, size=logits.shape[-2:], mode="bilinear", align_corners=False)
        else:
            x_in = x1chw
        if dec_feat.shape[-2:] != logits.shape[-2:]:
            dec_feat = F.interpolate(dec_feat, size=logits.shape[-2:], mode="bilinear", align_corners=False)

        g = self._grad_mag(x_in)     # [N,1,H,W], dtype matches logits
        g = self.dw_smooth(g)        # light de-noise
        q = self.dec_proj(dec_feat)  # [N,16,H,W]

        att = self.att_head(torch.cat([q, g], dim=1))
        att = torch.sigmoid(self.tau * att)
        return logits * att.clamp(0.6, 1.0)


# ---------- SPE-focused model ----------
class MATHFI_TimmEncoder_SPE(nn.Module):
    """
    MathFI with timm encoder + MSU/HAS/ASFG, plus:
      • Fine-scale ASFG prior tweaks (favor HAS to reduce thickening)
      • EdgeAwareRefiner to curb low-contrast false positives
    """
    def __init__(self,
                 encoder_name: str = "res2net50_26w_4s",
                 use_dpcn: bool = True,
                 dpcn_ch: int = 64, dpcn_iters: int = 6,
                 cbam_reduction: int = 16,
                 use_edge_refiner: bool = True):
        super().__init__()
        self.use_edge_refiner = use_edge_refiner

        # 0) input adapter (MODULE, used in forward)
        self.g2r = GrayToRGB()  # 1→3

        # 1) encoder (timm)
        self.encoder = create_model(encoder_name, pretrained=True, features_only=True, out_indices=(0, 1, 2, 3))
        C1, C2, C3, C4 = self.encoder.feature_info.channels()

        # 2) shallow DPCN fusion
        self.use_dpcn = use_dpcn
        if use_dpcn:
            self.dpcn   = DPCN(in_ch=1, channels=dpcn_ch, iters=dpcn_iters,
                               threshold_mode="scaled_vat", half_life=2.0, aggregate="mean")
            self.fuse_c1 = FuseCat1x1(inA=C1, inB=dpcn_ch, out_ch=C1)

        # 3) bottleneck
        self.bottleneck = ConvBlock(C4, C4 * 2)
        B = C4 * 2

        # 4) decoder widths
        d1_ch, d2_ch, d3_ch, d4_ch = C4, C3, C2, C1
        self.d1 = DecoderBlock(B,     d1_ch)
        self.d2 = DecoderBlock(d1_ch, d2_ch)
        self.d3 = DecoderBlock(d2_ch, d3_ch)
        self.d4 = DecoderBlock(d3_ch, d4_ch)

        # 5) MSU graph (A, P, Q)
        self.msu_A12    = AlignMSU(inA=C1, inB=C2, out_ch=C1)
        self.msu_A23    = AlignMSU(inA=C2, inB=C3, out_ch=C2)
        self.msu_A34    = AlignMSU(inA=C3, inB=C4, out_ch=C3)
        self.msu_P12_23 = AlignMSU(inA=C1, inB=C2, out_ch=C1)
        self.msu_P23_34 = AlignMSU(inA=C2, inB=C3, out_ch=C2)
        self.msu_Qlast  = AlignMSU(inA=C1, inB=C2, out_ch=C1)

        # Step D projections
        self.proj_d1 = nn.Conv2d(C4, d1_ch, 1, bias=True) if C4 != d1_ch else nn.Identity()
        self.proj_d2 = nn.Conv2d(2 * C3, d2_ch, 1, bias=True)
        self.proj_d3 = nn.Conv2d(3 * C2, d3_ch, 1, bias=True)
        self.proj_d4 = nn.Conv2d(4 * C1, d4_ch, 1, bias=True)

        # 6) HAS
        self.has = HASSkip(
            Cin_list=(C1, C2, C3, C4),
            Cout_list=(d1_ch, d2_ch, d3_ch, d4_ch),
            Cdec_list=(B, d1_ch, d2_ch, d3_ch)
        )

        # 7) ASFG gates — tweak priors to prefer HAS downstream
        self.asfg_d1 = AdaptiveSelectiveFusionGate(channels=d1_ch, reduction=cbam_reduction,
                                                   use_spatial_cbam=False, tau=1.8,
                                                   prior_logits=(+0.7, +0.2, -0.3),
                                                   edge_boost_gain=0.5, agree_boost_gain=0.3)
        self.asfg_d2 = AdaptiveSelectiveFusionGate(channels=d2_ch, reduction=cbam_reduction,
                                                   use_spatial_cbam=False, tau=1.6,
                                                   prior_logits=(+0.3, +0.3, -0.1),
                                                   edge_boost_gain=0.4, agree_boost_gain=0.3)
        self.asfg_d3 = AdaptiveSelectiveFusionGate(channels=d3_ch, reduction=cbam_reduction,
                                                   use_spatial_cbam=False, tau=1.4,
                                                   prior_logits=(0.0, +0.35, +0.05),
                                                   edge_boost_gain=0.3, agree_boost_gain=0.35)
        self.asfg_d4 = AdaptiveSelectiveFusionGate(channels=d4_ch, reduction=cbam_reduction,
                                                   use_spatial_cbam=True, tau=1.25,
                                                   prior_logits=(-0.5, +0.25, +0.95),
                                                   edge_boost_gain=0.0, agree_boost_gain=0.55)

        # 8) final head + optional edge-aware refiner
        self.final = nn.Conv2d(d4_ch, 1, kernel_size=1)
        if self.use_edge_refiner:
            self.edge_refiner = EdgeAwareRefiner(dec_ch=d3_ch, k_dw=5, tau=6.0)

    def forward(self, x1chw):
        # encoder
        x3 = self.g2r(x1chw)                       # use module (AMP-safe)
        s1, s2, s3, s4 = self.encoder(x3)

        # shallow DPCN fuse (use module created in __init__)
        if self.use_dpcn:
            dpcn_feat = self.dpcn(x1chw)           # [N, C_dpcn, H, W]
            s1 = self.fuse_c1(s1, dpcn_feat)       # channels remain C1

        # bottleneck
        b = self.bottleneck(s4)

        # MSU graph
        A12 = self.msu_A12(s1, s2)
        A23 = self.msu_A23(s2, s3)
        A34 = self.msu_A34(s3, s4)
        P12_23 = self.msu_P12_23(A12, A23)
        P23_34 = self.msu_P23_34(A23, A34)
        Qlast  = self.msu_Qlast(P12_23, P23_34)

        # Step D features
        FMSU_d1 = self.proj_d1(s4)
        FMSU_d2 = self.proj_d2(torch.cat([_resize_like(A34, s3), s3], dim=1))
        FMSU_d3 = self.proj_d3(torch.cat([_resize_like(A23, s2), _resize_like(P23_34, s2), s2], dim=1))
        FMSU_d4 = self.proj_d4(torch.cat([_resize_like(A12, s1), _resize_like(P12_23, s1),
                                          _resize_like(Qlast, s1), s1], dim=1))

        # decoder + ASFG gating
        FSKIP_d1 = self.has.forward_level(0, [s1, s2, s3, s4], b,  s4)
        FB1      = self.asfg_d1(FMSU_d1, FSKIP_d1);   d1 = self.d1(b,  FB1)

        FSKIP_d2 = self.has.forward_level(1, [s1, s2, s3, s4], d1, s3)
        FB2      = self.asfg_d2(FMSU_d2, FSKIP_d2);   d2 = self.d2(d1, FB2)

        FSKIP_d3 = self.has.forward_level(2, [s1, s2, s3, s4], d2, s2)
        FB3      = self.asfg_d3(FMSU_d3, FSKIP_d3);   d3 = self.d3(d2, FB3)

        FSKIP_d4 = self.has.forward_level(3, [s1, s2, s3, s4], d3, s1)
        FB4      = self.asfg_d4(FMSU_d4, FSKIP_d4);   d4 = self.d4(d3, FB4)

        # final logits
        logits = self.final(d4)

        # edge-aware refinement (SPE-focused)
        if self.use_edge_refiner:
            logits = self.edge_refiner(logits, x1chw, dec_feat=d3)

        # match input size
        if logits.shape[-2:] != x1chw.shape[-2:]:
            logits = F.interpolate(logits, size=x1chw.shape[-2:], mode="bilinear", align_corners=False)
        return logits
