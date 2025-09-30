# src/models/unet_cbam.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- reuse your base building blocks (unchanged) ----
# If these are defined in the same file, you can remove these imports.
# from src.models.unet_base import ConvBlock, EncoderBlock, DecoderBlock
# (You pasted them inline in your message; we redeclare them here for completeness.)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2, 2)
    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        return x, p

class DecoderBlock(nn.Module):
    """2× upsample (transpose conv) + concat with skip + ConvBlock."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.conv = ConvBlock(out_channels * 2, out_channels)
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

# ---- import your CBAM block exactly as given ----
# Adjust the import path if you saved CBAM elsewhere.
from src.models.blocks.cbam import CBAM


class UNetWithCBAM(nn.Module):
    """
    Drop-in UNet + CBAM:
      - Identical encoder/decoder to your base UNet.
      - Apply CBAM on *skip tensors* (s1..s4) before they go into the decoder.
      - Optionally apply CBAM on the bottleneck feature.

    This keeps the interface and tensor shapes unchanged for a clean ablation.
    """
    def __init__(self, in_channels: int = 1, use_spatial: bool = True, reduction_ratio: int = 16,
                 cbam_on_bottleneck: bool = True, cbam_on_decoder_outputs: bool = False):
        super().__init__()
        # base UNet modules (unchanged)
        self.e1 = EncoderBlock(in_channels, 64)
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)
        self.e4 = EncoderBlock(256, 512)

        self.bottleneck = ConvBlock(512, 1024)

        self.d1 = DecoderBlock(1024, 512)
        self.d2 = DecoderBlock(512,  256)
        self.d3 = DecoderBlock(256,  128)
        self.d4 = DecoderBlock(128,  64)

        self.final = nn.Conv2d(64, 1, kernel_size=1)

        # CBAM modules for each skip (channel sizes match the skips)
        self.cbam_s1 = CBAM(64,  reduction_ratio=reduction_ratio, use_spatial=use_spatial)
        self.cbam_s2 = CBAM(128, reduction_ratio=reduction_ratio, use_spatial=use_spatial)
        self.cbam_s3 = CBAM(256, reduction_ratio=reduction_ratio, use_spatial=use_spatial)
        self.cbam_s4 = CBAM(512, reduction_ratio=reduction_ratio, use_spatial=use_spatial)

        # optional: refine bottleneck
        self.cbam_bottleneck = CBAM(1024, reduction_ratio=reduction_ratio, use_spatial=use_spatial) \
                               if cbam_on_bottleneck else nn.Identity()

        # optional: refine decoder outputs too (kept off by default for a minimal ablation)
        self.cbam_d1 = CBAM(512,  reduction_ratio=reduction_ratio, use_spatial=use_spatial) if cbam_on_decoder_outputs else nn.Identity()
        self.cbam_d2 = CBAM(256,  reduction_ratio=reduction_ratio, use_spatial=use_spatial) if cbam_on_decoder_outputs else nn.Identity()
        self.cbam_d3 = CBAM(128,  reduction_ratio=reduction_ratio, use_spatial=use_spatial) if cbam_on_decoder_outputs else nn.Identity()
        self.cbam_d4 = CBAM(64,   reduction_ratio=reduction_ratio, use_spatial=use_spatial) if cbam_on_decoder_outputs else nn.Identity()

    def forward(self, x):
        # Encoder (unchanged)
        s1, p1 = self.e1(x)     # 64
        s2, p2 = self.e2(p1)    # 128
        s3, p3 = self.e3(p2)    # 256
        s4, p4 = self.e4(p3)    # 512

        # Apply CBAM *on the skips only* (standard and shape-safe)
        s1a = self.cbam_s1(s1)
        s2a = self.cbam_s2(s2)
        s3a = self.cbam_s3(s3)
        s4a = self.cbam_s4(s4)

        # Bottleneck + (optional) CBAM
        b = self.bottleneck(p4)     # 1024
        b = self.cbam_bottleneck(b) # 1024

        # Decoder: feed refined skips
        d1 = self.d1(b,  s4a); d1 = self.cbam_d1(d1)
        d2 = self.d2(d1, s3a); d2 = self.cbam_d2(d2)
        d3 = self.d3(d2, s2a); d3 = self.cbam_d3(d3)
        d4 = self.d4(d3, s1a); d4 = self.cbam_d4(d4)

        return self.final(d4)  # logits (no sigmoid)
