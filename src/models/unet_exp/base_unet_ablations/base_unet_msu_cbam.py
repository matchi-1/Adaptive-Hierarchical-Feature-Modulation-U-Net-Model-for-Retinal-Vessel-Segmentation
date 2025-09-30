# src/models/unet_with_msu_cbam.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.unet import ConvBlock, EncoderBlock, DecoderBlock 
from src.models.blocks.msu import MSU
from src.models.blocks.cbam import CBAM  

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


class UNetWithMSUSkipsCBAM(nn.Module):
    """
    MSU-enhanced UNet, plus CBAM applied only to the actual skip tensors used by the decoder,
    and on the bottleneck.

    Rationale:
      - keeps the MSU graph intact (A12, A23, A34, P1223, P2334, Qlast),
      - keeps DecoderBlock math identical,
      - clean ablation: "UNet + MSU" vs "UNet + MSU + CBAM-on-skips".
    """
    def __init__(self, in_channels: int = 1,
                 cbam_reduction: int = 16,
                 cbam_use_spatial: bool = True,
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
        self.d4 = DecoderBlock(128,  64)   # expects skip 64

        self.final = nn.Conv2d(64, 1, kernel_size=1)

        # ---- MSU graph (as before) ----
        self.msu_A12   = AlignMSU(inA=64,  inB=128, out_ch=64)
        self.msu_A23   = AlignMSU(inA=128, inB=256, out_ch=128)
        self.msu_A34   = AlignMSU(inA=256, inB=512, out_ch=256)

        self.msu_P1223 = AlignMSU(inA=64,  inB=128, out_ch=64)
        self.msu_P2334 = AlignMSU(inA=128, inB=256, out_ch=128)

        self.msu_Qlast = AlignMSU(inA=64,  inB=128, out_ch=64)

        # ---- 1x1 adapters to match expected decoder skip channels ----
        self.skip_d2_proj = nn.Conv2d(256 + 256, 256, kernel_size=1, bias=True)    # [E3(256), A34(256)] -> 256
        self.skip_d3_proj = nn.Conv2d(128 * 3,   128, kernel_size=1, bias=True)    # [E2, A23, P2334]   -> 128
        self.skip_d4_proj = nn.Conv2d(64  * 4,    64, kernel_size=1, bias=True)    # [E1, A12, P1223, Qlast] -> 64

        # ---- CBAM on *skip tensors* and (optionally) bottleneck ----
        self.cbam_skip_d1 = CBAM(512,  reduction_ratio=cbam_reduction, use_spatial=cbam_use_spatial)  # refines E4
        self.cbam_skip_d2 = CBAM(256,  reduction_ratio=cbam_reduction, use_spatial=cbam_use_spatial)  # refines fused skip for D2
        self.cbam_skip_d3 = CBAM(128,  reduction_ratio=cbam_reduction, use_spatial=cbam_use_spatial)  # refines fused skip for D3
        self.cbam_skip_d4 = CBAM(64,   reduction_ratio=cbam_reduction, use_spatial=cbam_use_spatial)  # refines fused skip for D4

        self.cbam_bott   = CBAM(1024, reduction_ratio=cbam_reduction, use_spatial=cbam_use_spatial) \
                           if cbam_on_bottleneck else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ---- encoders ----
        s1, p1 = self.e1(x)    # 64,   H
        s2, p2 = self.e2(p1)   # 128,  H/2
        s3, p3 = self.e3(p2)   # 256,  H/4
        s4, p4 = self.e4(p3)   # 512,  H/8

        # ---- bottleneck (+ optional CBAM) ----
        b = self.bottleneck(p4)          # 1024, H/8
        b = self.cbam_bott(b)            # (identity if disabled)

        # ---- MSU graph (unchanged) ----
        A12   = self.msu_A12(s1, s2)       # (N,64,H,W)
        A23   = self.msu_A23(s2, s3)       # (N,128,H/2,W/2)
        A34   = self.msu_A34(s3, s4)       # (N,256,H/4,W/4)

        P1223 = self.msu_P1223(A12, A23)   # (N,64,H,W)
        P2334 = self.msu_P2334(A23, A34)   # (N,128,H/2,W/2)

        Qlast = self.msu_Qlast(P1223, P2334)  # (N,64,H,W)

        # ---- refine the actual skip tensors that go into each decoder ----
        s4_ref   = self.cbam_skip_d1(s4)   # for D1 (keeps 512)
        skip_d2  = self.skip_d2_proj(torch.cat([s3, A34], dim=1))
        skip_d2  = self.cbam_skip_d2(skip_d2)  # keeps 256

        skip_d3  = self.skip_d3_proj(torch.cat([s2, A23, P2334], dim=1))
        skip_d3  = self.cbam_skip_d3(skip_d3)  # keeps 128

        skip_d4  = self.skip_d4_proj(torch.cat([s1, A12, P1223, Qlast], dim=1))
        skip_d4  = self.cbam_skip_d4(skip_d4)  # keeps 64

        # ---- decoders (unchanged) ----
        d1 = self.d1(b,  s4_ref)   # out: 512
        d2 = self.d2(d1, skip_d2)  # out: 256
        d3 = self.d3(d2, skip_d3)  # out: 128
        d4 = self.d4(d3, skip_d4)  # out: 64

        return self.final(d4)      # logits
