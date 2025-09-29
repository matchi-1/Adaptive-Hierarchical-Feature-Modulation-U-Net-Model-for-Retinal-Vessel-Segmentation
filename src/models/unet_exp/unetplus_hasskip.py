# --- imports ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log
from typing import List

from src.models.blocks.has_skip_exp1 import HASSkip

# --- helpers ---
def make_gn(num_channels: int, num_groups: int = 32):
    g = min(num_groups, num_channels)
    while num_channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, num_channels)

# --- core blocks (same as before) ---
class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.gn1   = make_gn(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.gn2   = make_gn(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.drop  = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()
        self.proj  = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        identity = self.proj(x)
        x = self.relu(self.gn1(self.conv1(x)))
        x = self.drop(x)
        x = self.gn2(self.conv2(x))
        x = self.relu(x + identity)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ResidualConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        s = self.conv(x)   # skip
        p = self.pool(s)   # pooled
        return s, p

# --- Up block that uses HAS-Skip ---
class UpBlockHAS(nn.Module):
    """
    Bilinear upsample + Conv1x1 (reduce channels) +
    HAS-Skip gating to produce a filtered skip (same channels as out_ch) +
    concat [upsampled, gated_skip] + ResidualConvBlock.
    """
    def __init__(self, in_ch: int, out_ch: int, has_module: HASSkip, level_idx: int):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.conv    = ResidualConvBlock(out_ch + out_ch, out_ch)  # after concat
        self.has     = has_module
        self.level   = int(level_idx)

    def forward(self, x: torch.Tensor, enc_feats: List[torch.Tensor]) -> torch.Tensor:
        # upsample decoder stream to target spatial size (HAS will match encoder maps internally)
        # we use E_level spatial size as the target for upsample
        target = enc_feats[self.level].shape[-2:]
        x = F.interpolate(x, size=target, mode="bilinear", align_corners=False)
        x = self.conv1x1(x)  # (B, out_ch, H, W)

        # HAS-Skip: gated skip for this level, channels == out_ch
        fskip = self.has(enc_feats=enc_feats, dec_feat=x, level_l=self.level)  # (B, out_ch, H, W)

        x = torch.cat([x, fskip], dim=1)  # (B, 2*out_ch, H, W)
        x = self.conv(x)
        return x

# --- U-Net with HAS-Skip (no MSU) ---
class UNetBaselinePlus(nn.Module):
    """
    Strong 'vanilla' U-Net + HAS-Skip for 1-channel inputs and 1-channel logits output.
    - GroupNorm, residual blocks, bilinear upsample
    - HAS-Skip provides gated skip per level (concatenated with decoder stream)
    """
    def __init__(self, in_channels=1, base_ch=64, bottleneck_dropout=0.1, init_pos_prior=0.1):
        super().__init__()
        C = base_ch

        # encoder
        self.e1 = EncoderBlock(in_channels, C)      # 64
        self.e2 = EncoderBlock(C, C*2)              # 128
        self.e3 = EncoderBlock(C*2, C*4)            # 256
        self.e4 = EncoderBlock(C*4, C*8)            # 512

        # bottleneck
        self.bottleneck = ResidualConvBlock(C*8, C*16, dropout_p=bottleneck_dropout)  # 1024

        # channel lists for HAS-Skip
        enc_chs = [C, C*2, C*4, C*8]       # [64, 128, 256, 512]
        dec_chs = [C, C*2, C*4, C*8]       # decoder widths per level for gating input

        # HAS-Skip per level, align_channels must match that level's out_ch
        self.has_l3 = HASSkip(enc_chs, dec_chs, align_channels=C*8)   # for level index 3 (s4), out_ch=512
        self.has_l2 = HASSkip(enc_chs, dec_chs, align_channels=C*4)   # for level index 2 (s3), out_ch=256
        self.has_l1 = HASSkip(enc_chs, dec_chs, align_channels=C*2)   # for level index 1 (s2), out_ch=128
        self.has_l0 = HASSkip(enc_chs, dec_chs, align_channels=C)     # for level index 0 (s1), out_ch=64

        # decoder blocks using HAS
        self.u1 = UpBlockHAS(in_ch=C*16, out_ch=C*8, has_module=self.has_l3, level_idx=3)  # uses s4
        self.u2 = UpBlockHAS(in_ch=C*8,  out_ch=C*4, has_module=self.has_l2, level_idx=2)  # uses s3
        self.u3 = UpBlockHAS(in_ch=C*4,  out_ch=C*2, has_module=self.has_l1, level_idx=1)  # uses s2
        self.u4 = UpBlockHAS(in_ch=C*2,  out_ch=C,   has_module=self.has_l0, level_idx=0)  # uses s1

        # head
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
        s1, p1 = self.e1(x)      # (B, C,   H,   W)
        s2, p2 = self.e2(p1)     # (B, 2C,  H/2, W/2)
        s3, p3 = self.e3(p2)     # (B, 4C,  H/4, W/4)
        s4, p4 = self.e4(p3)     # (B, 8C,  H/8, W/8)

        b  = self.bottleneck(p4) # (B, 16C, H/16, W/16)

        enc_feats = [s1, s2, s3, s4]

        d1 = self.u1(b,  enc_feats)  # uses HAS(level=3) -> (B, 8C,  H/8,  W/8)
        d2 = self.u2(d1, enc_feats)  # level=2        -> (B, 4C,  H/4,  W/4)
        d3 = self.u3(d2, enc_feats)  # level=1        -> (B, 2C,  H/2,  W/2)
        d4 = self.u4(d3, enc_feats)  # level=0        -> (B, C,   H,    W)

        logits = self.head(d4)       # raw logits; apply sigmoid only at eval
        return logits
