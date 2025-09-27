import torch
import torch.nn as nn
import torch.nn.functional as F

# --- MSU block (shared-weight multi-scale absolute subtraction) ----------------
from src.models.blocks.msu import MSU


# --- Your original U-Net building blocks --------------------------------------

from src.models.unet import ConvBlock, EncoderBlock

# --- Decoder with MSU-enhanced skips ------------------------------------------
class DecoderBlockMSU(nn.Module):
    """
    Upsample, compute MSU between upsampled decoder feat and skip, overlay onto
    the SKIP branch (to inject differential cues), then concat → ConvBlock.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.msu  = MSU(out_channels, out_channels)  # up/skip both have out_channels here
        self.conv = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.up(x)                  # (N, C_out, H, W)
        msu_map = self.msu(skip, x)     # compare skip vs up at multiple scales
        skip = skip + msu_map           # overlay difference onto the skip features
        x = torch.cat([x, skip], dim=1) # concat as usual
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = EncoderBlock(1,   64)
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)
        self.e4 = EncoderBlock(256, 512)

        self.bottleneck = ConvBlock(512, 1024)

        # swap your DecoderBlock for our MSU variant
        self.d1 = DecoderBlockMSU(1024, 512)  # matches s4 channels
        self.d2 = DecoderBlockMSU(512,  256)  # matches s3 channels
        self.d3 = DecoderBlockMSU(256,  128)  # matches s2 channels
        self.d4 = DecoderBlockMSU(128,   64)  # matches s1 channels

        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        b  = self.bottleneck(p4)

        d1 = self.d1(b,  s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        return self.final(d4)
