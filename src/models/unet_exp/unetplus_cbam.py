# --- imports ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log
from typing import List

from src.models.blocks.cbam import CBAM  
from src.models.unet_exp.unetplus import ResidualConvBlock, EncoderBlock, UpBlock

# --- helpers ---
def make_gn(num_channels: int, num_groups: int = 32):
    g = min(num_groups, num_channels)
    while num_channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, num_channels)


# --- U-Net with CBAM-after-encoder + CBAM at bottleneck ---
class CBAMUNetPlus(nn.Module):
    """
    Modern U-Net with:
      - GN, residual encoder/decoder blocks, bilinear upsampling
      - CBAM applied AFTER each encoder block output (s1..s4) before feeding to decoder
      - CBAM at the bottleneck
    """
    def __init__(self, in_channels=1, base_ch=64, bottleneck_dropout=0.1, init_pos_prior=0.1):
        super().__init__()
        C = base_ch

        # encoder
        self.e1 = EncoderBlock(in_channels, C)      # 64
        self.e2 = EncoderBlock(C, C*2)              # 128
        self.e3 = EncoderBlock(C*2, C*4)            # 256
        self.e4 = EncoderBlock(C*4, C*8)            # 512

        # CBAM after each encoder level (refine s1..s4)
        self.cbam_s1 = CBAM(gate_channels=C,     reduction_ratio=16, pool_types=['avg','max'], use_spatial=True)
        self.cbam_s2 = CBAM(gate_channels=C*2,   reduction_ratio=16, pool_types=['avg','max'], use_spatial=True)
        self.cbam_s3 = CBAM(gate_channels=C*4,   reduction_ratio=16, pool_types=['avg','max'], use_spatial=True)
        self.cbam_s4 = CBAM(gate_channels=C*8,   reduction_ratio=16, pool_types=['avg','max'], use_spatial=True)

        # bottleneck + CBAM
        self.bottleneck = ResidualConvBlock(C*8, C*16, dropout_p=bottleneck_dropout)  # 1024
        self.cbam_bottleneck = CBAM(gate_channels=C*16, reduction_ratio=16, pool_types=['avg','max'], use_spatial=True)

        # channel lists for HAS-Skip (decoder 1x1 reduces to these per level)
        enc_chs = [C, C*2, C*4, C*8]     # [64, 128, 256, 512]  (order: s1..s4)
        dec_chs = [C, C*2, C*4, C*8]     # [64, 128, 256, 512]  (order: levels 0..3)

        # decoder blocks
        self.u1 = UpBlock(C*16, C*8)                # -> 512
        self.u2 = UpBlock(C*8,  C*4)                # -> 256
        self.u3 = UpBlock(C*4,  C*2)                # -> 128
        self.u4 = UpBlock(C*2,  C)                  # -> 64

        self.head = nn.Conv2d(C, 1, 1)

        # init final bias to logit(prior)
        with torch.no_grad():
            p = float(init_pos_prior)
            p = min(max(p, 1e-4), 1-1e-4)
            self.head.bias.fill_(log(p/(1-p)))

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # encoder
        s1, p1 = self.e1(x)      # (B, C,   H,   W)
        s2, p2 = self.e2(p1)     # (B, 2C,  H/2, W/2)
        s3, p3 = self.e3(p2)     # (B, 4C,  H/4, W/4)
        s4, p4 = self.e4(p3)     # (B, 8C,  H/8, W/8)

        # CBAM refine each encoder skip BEFORE any MSU/HAS consumption
        r1 = self.cbam_s1(s1)    # (B, C,   H,   W)
        r2 = self.cbam_s2(s2)    # (B, 2C,  H/2, W/2)
        r3 = self.cbam_s3(s3)    # (B, 4C,  H/4, W/4)
        r4 = self.cbam_s4(s4)    # (B, 8C,  H/8, W/8)

        # bottleneck + CBAM
        b  = self.bottleneck(p4)         # (B, 16C, H/16, W/16)
        b  = self.cbam_bottleneck(b)     # (B, 16C, H/16, W/16)

        # decoder with HAS+MSU fusion (using CBAM-refined encoder features)
        d1 = self.u1(b,  r4)  # (B, 8C,  H/8,  W/8)
        d2 = self.u2(d1, r3)  # (B, 4C,  H/4,  W/4)
        d3 = self.u3(d2, r2)  # (B, 2C,  H/2,  W/2)
        d4 = self.u4(d3, r1)  # (B, C,   H,    W)

        logits = self.head(d4)           # raw logits; apply sigmoid only at eval
        return logits
