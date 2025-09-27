import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from src.models.blocks.has_skip import HASSkip

# -------------------------
# Base UNet building blocks
# -------------------------

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
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        return x, p


# ------------------------------------------
# Decoder block that uses HAS-Skip for skips
# ------------------------------------------

class DecoderBlockHAS(nn.Module):
    """
    A decoder stage that:
      (1) upsamples the previous decoder output,
      (2) builds a HAS-Skip skip feature for this level,
      (3) concatenates [upsampled decoder, gated skip], then
      (4) applies a ConvBlock.

    This mirrors original UNet DecoderBlock but generates the skip internally.
    """
    def __init__(self, in_channels, out_channels, has_skip: HASSkip, level_l: int):
        """
        Args:
            in_channels:  channels coming from the lower (deeper) decoder stage
            out_channels: channels produced by this stage (also the upsample output channels)
            has_skip:     the HAS-Skip module configured for this level
            level_l:      which encoder level this stage corresponds to (0..N-1)
        """
        super().__init__()
        self.level_l = level_l
        self.has_skip = has_skip

        # upsample deeper decoder feature to current resolution
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        # after concat([dec_up, FSKIP_l]), channels = out_channels + out_channels
        self.conv = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x_deeper: torch.Tensor, enc_feats: List[torch.Tensor]) -> torch.Tensor:
        # 1) upsample deeper decoder
        dec_up = self.up(x_deeper)  # shape [B, out_channels, H_d, W_d]

        # 2) build HAS-Skip for this level (uses all enc feats + current dec_up)
        fskip = self.has_skip(enc_feats, dec_up, level_l=self.level_l)  # [B, out_channels, H_d, W_d]

        # 3) fuse and convolve
        x = torch.cat([dec_up, fskip], dim=1)   # [B, 2*out_channels, H_d, W_d]
        x = self.conv(x)
        return x


# -------------
# Full HAS-UNet
# -------------

class UNet_HAS(nn.Module):
    """
    Your UNet, upgraded with HAS-Skip at each decoder stage.
    Channel plan mirrors your original model:
      Enc:  1->64->128->256->512
      Bott: 512->1024
      Dec:  1024->512->256->128->64
    """
    def __init__(self):
        super().__init__()
        # ---- Encoder ----
        self.e1 = EncoderBlock(1,   64)
        self.e2 = EncoderBlock(64,  128)
        self.e3 = EncoderBlock(128, 256)
        self.e4 = EncoderBlock(256, 512)

        # ---- Bottleneck ----
        self.bottleneck = ConvBlock(512, 1024)

        # Encoder output channels (shallow->deep)
        enc_channels = [64, 128, 256, 512]

        # ---- HAS-Skip modules, one per decoder level ----
        # At each decoder level l, we:
        #   - upsample to 'out_channels_l'
        #   - align skip to the same number of channels (align_channels = out_channels_l)
        #   - dec_channels for gate Wg is that same 'out_channels_l' (channels of dec_up)
        self.has4 = HASSkip(in_channels_per_level=enc_channels, dec_channels=512, align_channels=512)  # for d1 (level_l=3, E4)
        self.has3 = HASSkip(in_channels_per_level=enc_channels, dec_channels=256, align_channels=256)  # for d2 (level_l=2, E3)
        self.has2 = HASSkip(in_channels_per_level=enc_channels, dec_channels=128, align_channels=128)  # for d3 (level_l=1, E2)
        self.has1 = HASSkip(in_channels_per_level=enc_channels, dec_channels=64,  align_channels=64)   # for d4 (level_l=0, E1)

        # ---- Decoder (HAS-enabled) ----
        self.d1 = DecoderBlockHAS(in_channels=1024, out_channels=512, has_skip=self.has4, level_l=3)  # pairs with E4
        self.d2 = DecoderBlockHAS(in_channels=512,  out_channels=256, has_skip=self.has3, level_l=2)  # pairs with E3
        self.d3 = DecoderBlockHAS(in_channels=256,  out_channels=128, has_skip=self.has2, level_l=1)  # pairs with E2
        self.d4 = DecoderBlockHAS(in_channels=128,  out_channels=64,  has_skip=self.has1, level_l=0)  # pairs with E1

        # ---- Head ----
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        s1, p1 = self.e1(x)   # E1: [B,  64, H,   W]
        s2, p2 = self.e2(p1)  # E2: [B, 128, H/2, W/2]
        s3, p3 = self.e3(p2)  # E3: [B, 256, H/4, W/4]
        s4, p4 = self.e4(p3)  # E4: [B, 512, H/8, W/8]

        # Bottleneck
        b = self.bottleneck(p4)  # [B, 1024, H/16, W/16]

        # Prepare encoder list for HAS (shallow -> deep)
        enc_feats = [s1, s2, s3, s4]

        # Decoder with HAS-Skip
        d1 = self.d1(b,  enc_feats)  # uses E4 (level_l=3) + aggregated context
        d2 = self.d2(d1, enc_feats)  # uses E3 (level_l=2)
        d3 = self.d3(d2, enc_feats)  # uses E2 (level_l=1)
        d4 = self.d4(d3, enc_feats)  # uses E1 (level_l=0)

        return self.final(d4)