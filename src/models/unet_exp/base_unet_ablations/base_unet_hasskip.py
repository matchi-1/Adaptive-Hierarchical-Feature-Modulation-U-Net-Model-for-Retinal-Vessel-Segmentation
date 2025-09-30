# src/models/unet_with_hasskip.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.unet import ConvBlock, EncoderBlock, DecoderBlock
from src.models.blocks.has_skip import HASSkip


def _upsample_to(x, ref_like):
    """Bilinear upsample x to ref_like's HxW (channels unchanged)."""
    if x.shape[-2:] != ref_like.shape[-2:]:
        x = F.interpolate(x, size=ref_like.shape[-2:], mode="bilinear", align_corners=False)
    return x


class UNetWithHASSkip(nn.Module):
    """
    Base U-Net + HAS-Skip (only). The base EncoderBlock/DecoderBlock/ConvBlock code is left untouched.
    Mapping of decoder levels to encoder levels is the usual:
      d1 <- E4, d2 <- E3, d3 <- E2, d4 <- E1
    We form each skip by running HAS using:
      enc_feats = [E1,E2,E3,E4] (low -> high)
      dec_feat  = decoder context at that level, resized to the skip scale
      level_l   = {3,2,1,0} for d1..d4 respectively
    """
    def __init__(self, in_channels: int = 1):
        super().__init__()

        # --- base U-Net (identical to your original) ---
        self.e1 = EncoderBlock(in_channels, 64)
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)
        self.e4 = EncoderBlock(256, 512)

        self.bottleneck = ConvBlock(512, 1024)

        self.d1 = DecoderBlock(1024, 512)   # expects 512-ch skip @ E4 scale
        self.d2 = DecoderBlock(512,  256)   # expects 256-ch skip @ E3 scale
        self.d3 = DecoderBlock(256,  128)   # expects 128-ch skip @ E2 scale
        self.d4 = DecoderBlock(128,   64)   # expects  64-ch skip @ E1 scale

        self.final = nn.Conv2d(64, 1, kernel_size=1)

        # --- HAS-Skip blocks per decoder level ---
        # in_channels_per_level = [C1,C2,C3,C4] = [64,128,256,512]
        # dec_channels    : context channels at that level (b,d1,d2,d3) = [1024,512,256,128]
        # align_channels  : must match the expected skip channels of that decoder level
        self.has_d1 = HASSkip([64,128,256,512], dec_channels=1024, align_channels=512)  # for d1 (uses E4)
        self.has_d2 = HASSkip([64,128,256,512], dec_channels=512,  align_channels=256)  # for d2 (uses E3)
        self.has_d3 = HASSkip([64,128,256,512], dec_channels=256,  align_channels=128)  # for d3 (uses E2)
        self.has_d4 = HASSkip([64,128,256,512], dec_channels=128,  align_channels=64)   # for d4 (uses E1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- encoders (unchanged) ---
        s1, p1 = self.e1(x)     # E1: 64,  H
        s2, p2 = self.e2(p1)    # E2: 128, H/2
        s3, p3 = self.e3(p2)    # E3: 256, H/4
        s4, p4 = self.e4(p3)    # E4: 512, H/8

        # --- bottleneck (unchanged) ---
        b = self.bottleneck(p4) # 1024, H/16

        enc_feats = [s1, s2, s3, s4]

        # ---------------------------
        # d1 skip (target scale = E4 @ H/8, channels=512)
        # HAS expects dec_feat at the SAME spatial size as the skip scale -> upsample b to E4 size
        b_up   = _upsample_to(b, s4)                                  # 1024 @ H/8
        skip1  = self.has_d1(enc_feats, dec_feat=b_up,  level_l=3)    # gate E4 → [B,512,H/8,W/8]
        d1     = self.d1(b, skip1)                                    # result @ H/8

        # d2 skip (target scale = E3 @ H/4, channels=256)
        d1_up  = _upsample_to(d1, s3)                                 # 512 @ H/4
        skip2  = self.has_d2(enc_feats, dec_feat=d1_up, level_l=2)    # gate E3 → [B,256,H/4,W/4]
        d2     = self.d2(d1, skip2)                                   # result @ H/4

        # d3 skip (target scale = E2 @ H/2, channels=128)
        d2_up  = _upsample_to(d2, s2)                                 # 256 @ H/2
        skip3  = self.has_d3(enc_feats, dec_feat=d2_up, level_l=1)    # gate E2 → [B,128,H/2,W/2]
        d3     = self.d3(d2, skip3)                                   # result @ H/2

        # d4 skip (target scale = E1 @ H, channels=64)
        d3_up  = _upsample_to(d3, s1)                                 # 128 @ H
        skip4  = self.has_d4(enc_feats, dec_feat=d3_up, level_l=0)    # gate E1 → [B,64,H,W]
        d4     = self.d4(d3, skip4)                                   # result @ H

        return self.final(d4)                                         # logits
