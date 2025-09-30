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
        return self.msu(A_, B_)


class UNetWithMSUSkipsCBAM_MNet(nn.Module):
    """
    UNet + MSU-on-skips + CBAM-on-skips (+ bottleneck) + M-Net LEFT LEG (multi-scale input).
    Optional RIGHT-LEG deep supervision (off by default to preserve SPE/ClDice).

    - Left leg: feed multi-scale raw input (1x, 1/2, 1/4, 1/8) into each encoder stage.
      We project the downsampled raw input and fuse (concat + 1x1) with the stage input BEFORE EncoderBlock.
    - MSU graph: unchanged (A12, A23, A34, P1223, P2334, Qlast).
    - CBAM: refines the actual skip tensors used by the decoder and the bottleneck (optional).
    - Deep supervision (optional): side logits at each decoder level, upsampled and fused with learnable non-negative weights.
      Disabled by default because DS can bias toward higher recall (SEN) at the cost of SPE on some datasets.

    Returns:
      - By default: a single [N,1,H,W] logit (same contract as your baseline).
      - If return_side_outputs=True: (logits, [side1..side4])
    """
    def __init__(self, in_channels: int = 1,
                 cbam_reduction: int = 16,
                 cbam_use_spatial: bool = True,
                 cbam_on_bottleneck: bool = True,
                 use_deep_supervision: bool = False,     # OFF by default to keep SPE/ClDice stable
                 return_side_outputs: bool = False):
        super().__init__()

        self.in_channels = in_channels
        self.use_deep_supervision = use_deep_supervision
        self.return_side_outputs  = return_side_outputs

        # ---- base UNet encoder/decoder ----
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

        # ---- M-Net LEFT LEG: multi-scale input → fuse with stage input BEFORE EncoderBlock ----
        # Pools to 1/2, 1/4, 1/8 for e2/e3/e4
        self.pool2 = nn.AvgPool2d(2, 2)
        self.pool4 = nn.AvgPool2d(4, 4)
        self.pool8 = nn.AvgPool2d(8, 8)

        # Project raw input at each scale to match stage input channels (before the EncoderBlock)
        # e1 stage input has channels = in_channels (usually 1)
        self.left_proj_e1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        # e2/e3/e4 stage inputs have channels 64/128/256 respectively (the output of previous pool)
        self.left_proj_e2 = nn.Conv2d(in_channels, 64,  kernel_size=3, padding=1, bias=False)
        self.left_proj_e3 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1, bias=False)
        self.left_proj_e4 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False)

        # 1×1 fusers to bring concat back to the stage input channels
        self.fuse_e1 = nn.Conv2d(in_channels + in_channels, in_channels, kernel_size=1)
        self.fuse_e2 = nn.Conv2d(64 + 64,   64,   kernel_size=1)
        self.fuse_e3 = nn.Conv2d(128 + 128, 128,  kernel_size=1)
        self.fuse_e4 = nn.Conv2d(256 + 256, 256,  kernel_size=1)

        # small norms to stabilize left-leg fusion
        self.bn_in  = nn.BatchNorm2d(in_channels)
        self.bn64   = nn.BatchNorm2d(64)
        self.bn128  = nn.BatchNorm2d(128)
        self.bn256  = nn.BatchNorm2d(256)
        self.relu   = nn.ReLU(inplace=True)

        # ---- MSU graph (unchanged) ----
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

        # ---- RIGHT-LEG deep supervision (optional) ----
        if self.use_deep_supervision:
            self.side1 = nn.Conv2d(512, 1, kernel_size=1)  # after d1
            self.side2 = nn.Conv2d(256, 1, kernel_size=1)  # after d2
            self.side3 = nn.Conv2d(128, 1, kernel_size=1)  # after d3
            self.side4 = nn.Conv2d(64,  1, kernel_size=1)  # after d4
            # learnable non-negative fusion weights (init skewed toward later decoders → often better SPE)
            self._w = nn.Parameter(torch.tensor([0.2, 0.4, 0.8, 1.6], dtype=torch.float32))

    # --- helper: left-leg fuse raw input with a stage tensor ---
    def _fuse_left_leg(self, x_raw, x_stage_in, proj, fuse, pool=None, bn: nn.BatchNorm2d | None = None):
        if pool is None:
            z = x_raw
        else:
            z = pool(x_raw)
            if z.shape[-2:] != x_stage_in.shape[-2:]:
                z = F.interpolate(z, size=x_stage_in.shape[-2:], mode="bilinear", align_corners=False)
        z = proj(z)
        if bn is not None:
            z = bn(z)
        z = self.relu(z)
        x_cat = torch.cat([x_stage_in, z], dim=1)
        return self.relu(fuse(x_cat))

    def forward(self, x: torch.Tensor):
        N, C, H, W = x.shape

        # ---- M-Net LEFT LEG + Encoder ----
        # e1: fuse raw(1x) with input before EncoderBlock
        x1_in = self._fuse_left_leg(self.bn_in(x), x, self.left_proj_e1, self.fuse_e1, pool=None, bn=None)
        s1, p1 = self.e1(x1_in)                     # 64,   H

        # e2: fuse raw(1/2) with p1
        p1n = self.bn64(p1)
        x2_in = self._fuse_left_leg(x, p1n, self.left_proj_e2, self.fuse_e2, pool=self.pool2, bn=self.bn64)
        s2, p2 = self.e2(x2_in)                     # 128,  H/2

        # e3: fuse raw(1/4) with p2
        p2n = self.bn128(p2)
        x3_in = self._fuse_left_leg(x, p2n, self.left_proj_e3, self.fuse_e3, pool=self.pool4, bn=self.bn128)
        s3, p3 = self.e3(x3_in)                     # 256,  H/4

        # e4: fuse raw(1/8) with p3
        p3n = self.bn256(p3)
        x4_in = self._fuse_left_leg(x, p3n, self.left_proj_e4, self.fuse_e4, pool=self.pool8, bn=self.bn256)
        s4, p4 = self.e4(x4_in)                     # 512,  H/8

        # ---- bottleneck (+ optional CBAM) ----
        b = self.bottleneck(p4)                     # 1024, H/8
        b = self.cbam_bott(b)

        # ---- MSU graph (unchanged) ----
        A12   = self.msu_A12(s1, s2)                # (N, 64,  H,   W)
        A23   = self.msu_A23(s2, s3)                # (N, 128, H/2, W/2)
        A34   = self.msu_A34(s3, s4)                # (N, 256, H/4, W/4)

        P1223 = self.msu_P1223(A12, A23)            # (N, 64,  H,   W)
        P2334 = self.msu_P2334(A23, A34)            # (N, 128, H/2, W/2)

        Qlast = self.msu_Qlast(P1223, P2334)        # (N, 64,  H,   W)

        # ---- CBAM-refined skips to the decoder ----
        s4_ref  = self.cbam_skip_d1(s4)             # 512 for d1
        skip_d2 = self.cbam_skip_d2(self.skip_d2_proj(torch.cat([s3, A34], dim=1)))              # 256
        skip_d3 = self.cbam_skip_d3(self.skip_d3_proj(torch.cat([s2, A23, P2334], dim=1)))       # 128
        skip_d4 = self.cbam_skip_d4(self.skip_d4_proj(torch.cat([s1, A12, P1223, Qlast], dim=1)))# 64

        # ---- decoders ----
        d1 = self.d1(b,  s4_ref)                    # 512
        d2 = self.d2(d1, skip_d2)                   # 256
        d3 = self.d3(d2, skip_d3)                   # 128
        d4 = self.d4(d3, skip_d4)                   # 64

        if not self.use_deep_supervision:
            logits = self.final(d4)
            return logits

        # ---- RIGHT-LEG deep supervision (optional) ----
        side1 = F.interpolate(self.side1(d1), size=(H, W), mode="bilinear", align_corners=False)
        side2 = F.interpolate(self.side2(d2), size=(H, W), mode="bilinear", align_corners=False)
        side3 = F.interpolate(self.side3(d3), size=(H, W), mode="bilinear", align_corners=False)
        side4 = F.interpolate(self.side4(d4), size=(H, W), mode="bilinear", align_corners=False)

        w = F.softplus(self._w)  # non-negative
        fused = (w[0]*side1 + w[1]*side2 + w[2]*side3 + w[3]*side4) / (w.sum() + 1e-8)

        if self.return_side_outputs:
            return fused, [side1, side2, side3, side4]
        return fused
