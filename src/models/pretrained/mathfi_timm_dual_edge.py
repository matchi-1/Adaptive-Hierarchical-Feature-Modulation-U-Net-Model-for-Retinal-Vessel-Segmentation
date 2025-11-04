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

class SmallRefineNet(nn.Module):
    def __init__(self, in_ch=2, out_ch=1):
        super().__init__()
        self.conv1 = ConvBlock(in_ch, 32)
        self.conv2 = ConvBlock(32, 64)
        self.up   = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.out  = nn.Conv2d(32, out_ch, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.up(x)
        return self.out(x)

class MATHFI_TimmEncoder_Refined(nn.Module):
    def __init__(self,
                 encoder_name='res2net50_26w_4s',
                 use_dpcn=True,
                 dpcn_ch=64, 
                 dpcn_iters=6,
                 cbam_reduction=16):
        super().__init__()
        self.g2r = GrayToRGB()  
        self.encoder = create_model(encoder_name, pretrained=True, features_only=True, out_indices=(0,1,2,3))
        C1, C2, C3, C4 = self.encoder.feature_info.channels()

        self.use_dpcn = use_dpcn
        if use_dpcn:
            self.dpcn = DPCN(1, dpcn_ch, dpcn_iters, threshold_mode='scaled_vat', half_life=2.0, aggregate='mean')
            self.fuse_c1 = FuseCat1x1(C1, dpcn_ch, C1)

        self.bottleneck = ConvBlock(C4, C4*2)
        B = C4*2
        d1_ch, d2_ch, d3_ch, d4_ch = C4, C3, C2, C1

        self.d1 = DecoderBlock(B, d1_ch)
        self.d2 = DecoderBlock(d1_ch, d2_ch)
        self.d3 = DecoderBlock(d2_ch, d3_ch)
        self.d4 = DecoderBlock(d3_ch, d4_ch)

        self.msu_A12 = MSU(C1, C1)          # 64 → 64
        self.msu_A23 = MSU(C2, C2)          # 256 → 256
        self.msu_A34 = MSU(C3, C3)          # 512 → 512
        self.msu_P12_23 = MSU(C1, C1)       # 64 → 64
        self.msu_P23_34 = MSU(C2, C2)       # 256 → 256
        self.msu_Qlast = MSU(C1, C1)        # 64 → 64


        self.proj_d1 = nn.Conv2d(C4, d1_ch, 1) if C4 != d1_ch else nn.Identity()
        self.proj_d2 = nn.Conv2d(2 * C3, d2_ch, 1)
        self.proj_d3 = nn.Conv2d(3 * C2, d3_ch, 1)
        self.proj_d4 = nn.Conv2d(4 * C1, d4_ch, 1)

        self.has = HASSkip((C1,C2,C3,C4), (d1_ch,d2_ch,d3_ch,d4_ch), (B,d1_ch,d2_ch,d3_ch))

        self.asfg_d1 = AdaptiveSelectiveFusionGate(d1_ch, cbam_reduction)
        self.asfg_d2 = AdaptiveSelectiveFusionGate(d2_ch, cbam_reduction)
        self.asfg_d3 = AdaptiveSelectiveFusionGate(d3_ch, cbam_reduction)
        self.asfg_d4 = AdaptiveSelectiveFusionGate(d4_ch, cbam_reduction)

        self.seg_head = nn.Conv2d(d4_ch, 1, 1)
        self.skel_head = nn.Conv2d(d4_ch, 1, 1)
        self.edge_head = nn.Conv2d(d4_ch, 1, 1)

        self.refine_net = SmallRefineNet(2, 1)

    def forward(self, x):
        x_rgb = self.g2r(x)
        s1, s2, s3, s4 = self.encoder(x_rgb)
        if self.use_dpcn:
            dpcn_feat = self.dpcn(x)
            s1 = self.fuse_c1(s1, dpcn_feat)

        b = self.bottleneck(s4)

        A12 = self.msu_A12(s1, s2)
        A23 = self.msu_A23(s2, s3)
        A34 = self.msu_A34(s3, s4)
        P12_23 = self.msu_P12_23(A12, A23)
        P23_34 = self.msu_P23_34(A23, A34)
        Qlast = self.msu_Qlast(P12_23, P23_34)

        FMSU_d1 = self.proj_d1(s4)
        FMSU_d2 = self.proj_d2(torch.cat([A34, s3], dim=1))
        FMSU_d3 = self.proj_d3(torch.cat([A23, P23_34, s2], dim=1))
        FMSU_d4 = self.proj_d4(torch.cat([A12, P12_23, Qlast, s1], dim=1))

        FSKIP_d1 = self.has.forward_level(0, [s1,s2,s3,s4], b, s4)
        FB1 = self.asfg_d1(FMSU_d1, FSKIP_d1)
        d1 = self.d1(b, FB1)

        FSKIP_d2 = self.has.forward_level(1, [s1,s2,s3,s4], d1, s3)
        FB2 = self.asfg_d2(FMSU_d2, FSKIP_d2)
        d2 = self.d2(d1, FB2)

        FSKIP_d3 = self.has.forward_level(2, [s1,s2,s3,s4], d2, s2)
        FB3 = self.asfg_d3(FMSU_d3, FSKIP_d3)
        d3 = self.d3(d2, FB3)

        FSKIP_d4 = self.has.forward_level(3, [s1,s2,s3,s4], d3, s1)
        FB4 = self.asfg_d4(FMSU_d4, FSKIP_d4)
        d4 = self.d4(d3, FB4)

        edge = torch.sigmoid(self.edge_head(d4))
        seg = torch.sigmoid(self.seg_head(d4))
        skel = torch.sigmoid(self.skel_head(d4))

        seg_plus_edge = torch.cat([seg, edge], dim=1)
        refined = torch.sigmoid(self.refine_net(seg_plus_edge))

        return {'seg': seg, 'skel': skel, 'edge': edge, 'refined': refined}
