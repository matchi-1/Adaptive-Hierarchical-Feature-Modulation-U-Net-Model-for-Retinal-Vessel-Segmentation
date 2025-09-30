# src/models/unet_with_msu.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.unet import ConvBlock, EncoderBlock, DecoderBlock, UNet  
from src.models.blocks.msu import MSU

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

class UNetWithMSUSkips(nn.Module):
    """
    Wraps your original UNet and 'strictly adds' MSU-based skip enhancements
    without changing the base building blocks.

    Decoder expectations preserved:
      d1 expects skip=E4 (512ch)
      d2 expects skip=256ch   (we supply proj([E3, A34]) -> 256)
      d3 expects skip=128ch   (we supply proj([E2, A23, P2334]) -> 128)
      d4 expects skip=64ch    (we supply proj([E1, A12, P1223, Qlast]) -> 64)
    """
    def __init__(self, in_channels: int = 1):
        super().__init__()
        # --- copy of your base UNet modules (don’t modify their code) ---
        self.e1 = EncoderBlock(in_channels, 64)
        self.e2 = EncoderBlock(64,   128)
        self.e3 = EncoderBlock(128,  256)
        self.e4 = EncoderBlock(256,  512)

        self.bottleneck = ConvBlock(512, 1024)

        self.d1 = DecoderBlock(1024, 512)  # expects skip with 512 ch
        self.d2 = DecoderBlock(512,  256)  # expects skip with 256 ch
        self.d3 = DecoderBlock(256,  128)  # expects skip with 128 ch
        self.d4 = DecoderBlock(128,  64)   # expects skip with 64 ch

        self.final = nn.Conv2d(64, 1, kernel_size=1)

        # --- MSU graph per your spec ---
        # A12 at E1 scale/channels (64)
        self.msu_A12   = AlignMSU(inA=64,  inB=128, out_ch=64)
        # A23 at E2 scale/channels (128)
        self.msu_A23   = AlignMSU(inA=128, inB=256, out_ch=128)
        # A34 at E3 scale/channels (256)
        self.msu_A34   = AlignMSU(inA=256, inB=512, out_ch=256)

        # P1223 = MSU(A12, A23) at E1 scale (64)
        self.msu_P1223 = AlignMSU(inA=64,  inB=128, out_ch=64)
        # P2334 = MSU(A23, A34) at E2 scale (128)
        self.msu_P2334 = AlignMSU(inA=128, inB=256, out_ch=128)

        # Qlast = MSU(P1223, P2334) at E1 scale (64)
        self.msu_Qlast = AlignMSU(inA=64,  inB=128, out_ch=64)

        # --- 1x1 adapters to keep decoder skip channel counts unchanged ---
        # D2 wants 256ch skip: fuse [E3(256), A34(256)] -> 256
        self.skip_d2_proj = nn.Conv2d(256 + 256, 256, kernel_size=1, bias=True)
        # D3 wants 128ch skip: fuse [E2(128), A23(128), P2334(128)] -> 128
        self.skip_d3_proj = nn.Conv2d(128 * 3, 128, kernel_size=1, bias=True)
        # D4 wants 64ch skip: fuse [E1(64), A12(64), P1223(64), Qlast(64)] -> 64
        self.skip_d4_proj = nn.Conv2d(64 * 4, 64, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ---- encoders (unchanged) ----
        s1, p1 = self.e1(x)    # E1: 64,   H
        s2, p2 = self.e2(p1)   # E2: 128,  H/2
        s3, p3 = self.e3(p2)   # E3: 256,  H/4
        s4, p4 = self.e4(p3)   # E4: 512,  H/8

        # ---- bottleneck (unchanged) ----
        b = self.bottleneck(p4)  # 1024, H/8

        # ---- MSU graph (your spec) ----
        A12   = self.msu_A12(s1, s2)       # at E1 scale → (N,64,H,W)
        A23   = self.msu_A23(s2, s3)       # at E2 scale → (N,128,H/2,W/2)
        A34   = self.msu_A34(s3, s4)       # at E3 scale → (N,256,H/4,W/4)

        P1223 = self.msu_P1223(A12, A23)   # aligned to E1 scale → (N,64,H,W)
        P2334 = self.msu_P2334(A23, A34)   # aligned to E2 scale → (N,128,H/2,W/2)

        Qlast = self.msu_Qlast(P1223, P2334)  # aligned to E1 scale → (N,64,H,W)

        # ---- d1: unchanged skip (E4) ----
        d1 = self.d1(b, s4)  # out: 512, H/4    (DecoderBlock will upsample to H/4 internally)

        # ---- d2: skip = proj([E3, A34]) → 256ch ----
        # ensure A34 is already at E3 scale; concatenate channel-wise and project back to 256
        skip_d2 = self.skip_d2_proj(torch.cat([s3, A34], dim=1))
        d2 = self.d2(d1, skip_d2)  # out: 256, H/2

        # ---- d3: skip = proj([E2, A23, P2334]) → 128ch ----
        # P2334 is at E2 scale already
        skip_d3 = self.skip_d3_proj(torch.cat([s2, A23, P2334], dim=1))
        d3 = self.d3(d2, skip_d3)  # out: 128, H

        # ---- d4: skip = proj([E1, A12, P1223, Qlast]) → 64ch ----
        # All four are at E1 scale
        skip_d4 = self.skip_d4_proj(torch.cat([s1, A12, P1223, Qlast], dim=1))
        d4 = self.d4(d3, skip_d4)  # out: 64, H

        return self.final(d4)  # logits
