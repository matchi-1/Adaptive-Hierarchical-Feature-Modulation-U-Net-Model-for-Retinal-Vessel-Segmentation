import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model

from src.models.dpcn.dpcn_v2 import DPCN
from src.models.blocks.cbam import CBAM
from src.models.unet_exp.base_unet_ablations.base_unet_msu_cbam_hasskip_improved3 import HASSkip, AdaptiveSelectiveFusionGate
from src.models.unet import DecoderBlock, ConvBlock
from src.models.pretrained.bridges.bridges import GrayToRGB, FuseCat1x1

class MATHFI_TimmEncoder_NoMSU(nn.Module):
    def __init__(self,
                 encoder_name: str = "res2net50_26w_4s",
                 use_dpcn: bool = True,
                 dpcn_ch: int = 64, dpcn_iters: int = 6,
                 cbam_reduction: int = 16):
        super().__init__()

        self.g2r = GrayToRGB()
        self.encoder = create_model(encoder_name, pretrained=True, features_only=True, out_indices=(0,1,2,3))
        C1, C2, C3, C4 = self.encoder.feature_info.channels()

        self.use_dpcn = use_dpcn
        if use_dpcn:
            self.dpcn = DPCN(in_ch=1, channels=dpcn_ch, iters=dpcn_iters,
                             threshold_mode="scaled_vat", half_life=2.0, aggregate="mean")
            self.fuse_c1 = FuseCat1x1(inA=C1, inB=dpcn_ch, out_ch=C1)

        self.bottleneck = ConvBlock(C4, C4*2)
        B = C4*2
        d1_ch, d2_ch, d3_ch, d4_ch = C4, C3, C2, C1

        self.d1 = DecoderBlock(B, d1_ch)
        self.d2 = DecoderBlock(d1_ch, d2_ch)
        self.d3 = DecoderBlock(d2_ch, d3_ch)
        self.d4 = DecoderBlock(d3_ch, d4_ch)

        self.has = HASSkip(
            Cin_list=(C1, C2, C3, C4),
            Cout_list=(d1_ch, d2_ch, d3_ch, d4_ch),
            Cdec_list=(B, d1_ch, d2_ch, d3_ch)
        )

        self.asfg_d1 = AdaptiveSelectiveFusionGate(channels=d1_ch, reduction=cbam_reduction)
        self.asfg_d2 = AdaptiveSelectiveFusionGate(channels=d2_ch, reduction=cbam_reduction)
        self.asfg_d3 = AdaptiveSelectiveFusionGate(channels=d3_ch, reduction=cbam_reduction)
        self.asfg_d4 = AdaptiveSelectiveFusionGate(channels=d4_ch, reduction=cbam_reduction)

        self.final = nn.Conv2d(d4_ch, 1, kernel_size=1)

    def forward(self, x1chw):
        x3chw = self.g2r(x1chw)
        s1, s2, s3, s4 = self.encoder(x3chw)

        if self.use_dpcn:
            dpcn_feat = self.dpcn(x1chw)
            s1 = self.fuse_c1(s1, dpcn_feat)

        b = self.bottleneck(s4)

        FSKIP_d1 = self.has.forward_level(0, [s1, s2, s3, s4], b, s4)
        d1 = self.d1(b, self.asfg_d1(s4, FSKIP_d1))

        FSKIP_d2 = self.has.forward_level(1, [s1, s2, s3, s4], d1, s3)
        d2 = self.d2(d1, self.asfg_d2(s3, FSKIP_d2))

        FSKIP_d3 = self.has.forward_level(2, [s1, s2, s3, s4], d2, s2)
        d3 = self.d3(d2, self.asfg_d3(s2, FSKIP_d3))

        FSKIP_d4 = self.has.forward_level(3, [s1, s2, s3, s4], d3, s1)
        d4 = self.d4(d3, self.asfg_d4(s1, FSKIP_d4))

        logits = self.final(d4)

        if logits.shape[-2:] != x1chw.shape[-2:]:
            logits = F.interpolate(logits, size=x1chw.shape[-2:], mode="bilinear", align_corners=False)

        return logits