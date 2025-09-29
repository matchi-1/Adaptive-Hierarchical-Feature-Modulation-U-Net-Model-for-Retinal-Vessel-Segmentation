# --- imports ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log
from typing import List

from src.models.blocks.has_skip_exp1 import HASSkip
from src.models.blocks.msu import MSU

# --- helpers ---
def make_gn(num_channels: int, num_groups: int = 32):
    g = min(num_groups, num_channels)
    while num_channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, num_channels)

# --- core blocks (same as before) ---
class ResidualConvBlock(nn.Module):
    """
    2×(Conv3x3 + GN + ReLU) with residual connection.
    If in/out channels differ, use 1×1 projection for the skip.
    """
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

# --- Up block that fuses HAS-Skip + MSU ---
class UpBlockHASMSU(nn.Module):
    """
    Bilinear upsample + Conv1x1 (→ out_ch) +
    HAS-Skip(enc_feats, dec_feat=x, level) -> has (B,out_ch,H,W)
    MSU(x, raw_skip_level) -> msu (B,out_ch,H,W)
    Fuse: fused = alpha*msu + beta*has -> GN -> ReLU
    Concat [x, fused] -> ResidualConvBlock
    """
    def __init__(self, in_ch: int, out_ch: int,
                 has_module: HASSkip, msu_module: MSU, level_idx: int):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.conv    = ResidualConvBlock(out_ch + out_ch, out_ch)  # after concat

        self.has     = has_module
        self.msu     = msu_module
        self.level   = int(level_idx)

        # learnable fusion scalars (start at 1.0)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta  = nn.Parameter(torch.tensor(1.0))

        # stabilize fused stream
        self.fuse_norm = make_gn(out_ch)
        self.fuse_act  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, enc_feats: List[torch.Tensor]) -> torch.Tensor:
        # spatial target from the encoder level we pair with
        skip = enc_feats[self.level]
        target = skip.shape[-2:]

        # upsample decoder stream and set channels to out_ch
        x = F.interpolate(x, size=target, mode="bilinear", align_corners=False)
        x = self.conv1x1(x)  # (B, out_ch, H, W)

        # HAS-Skip (uses all encoder levels + current decoder x)
        has = self.has(enc_feats=enc_feats, dec_feat=x, level_l=self.level)  # (B, out_ch, H, W)

        # MSU between current decoder x and the raw skip at this level
        msu = self.msu(x, skip)  # (B, out_ch, H, W)

        # fuse with learnable scalars, then normalize/activate
        fused = self.alpha * msu + self.beta * has
        fused = self.fuse_act(self.fuse_norm(fused))

        # concatenate and decode
        x = torch.cat([x, fused], dim=1)  # (B, 2*out_ch, H, W)
        x = self.conv(x)
        return x

# --- U-Net with HAS-Skip + MSU fusion ---
class UNetBaselinePlus(nn.Module):
    """
    Modern U-Net with:
      - GN, residual encoder/decoder blocks, bilinear upsampling
      - HAS-Skip and MSU fused at every decoder level with learnable α/β
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
        enc_chs = [C, C*2, C*4, C*8]     # encoder widths: [64, 128, 256, 512]
        dec_chs = [C, C*2, C*4, C*8]     # decoder widths per level after 1x1 on upsampled stream

        # HAS-Skip per level (align_channels must == that level's out_ch)
        self.has_l3 = HASSkip(enc_chs, dec_chs, align_channels=C*8)   # level 3, out_ch=512
        self.has_l2 = HASSkip(enc_chs, dec_chs, align_channels=C*4)   # level 2, out_ch=256
        self.has_l1 = HASSkip(enc_chs, dec_chs, align_channels=C*2)   # level 1, out_ch=128
        self.has_l0 = HASSkip(enc_chs, dec_chs, align_channels=C)     # level 0, out_ch=64

        # MSU per level (in=out=out_ch at that level)
        self.msu_l3 = MSU(in_channels=C*8, out_channels=C*8, use_bn=True, activation=True)
        self.msu_l2 = MSU(in_channels=C*4, out_channels=C*4, use_bn=True, activation=True)
        self.msu_l1 = MSU(in_channels=C*2, out_channels=C*2, use_bn=True, activation=True)
        self.msu_l0 = MSU(in_channels=C,   out_channels=C,   use_bn=True, activation=True)

        # decoder blocks using HAS+MSU fusion
        self.u1 = UpBlockHASMSU(in_ch=C*16, out_ch=C*8, has_module=self.has_l3, msu_module=self.msu_l3, level_idx=3)
        self.u2 = UpBlockHASMSU(in_ch=C*8,  out_ch=C*4, has_module=self.has_l2, msu_module=self.msu_l2, level_idx=2)
        self.u3 = UpBlockHASMSU(in_ch=C*4,  out_ch=C*2, has_module=self.has_l1, msu_module=self.msu_l1, level_idx=1)
        self.u4 = UpBlockHASMSU(in_ch=C*2,  out_ch=C,   has_module=self.has_l0, msu_module=self.msu_l0, level_idx=0)

        # head
        self.head = nn.Conv2d(C, 1, 1)

        # initialize final bias to logit(prior)
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

        d1 = self.u1(b,  enc_feats)  # (B, 8C,  H/8,  W/8)
        d2 = self.u2(d1, enc_feats)  # (B, 4C,  H/4,  W/4)
        d3 = self.u3(d2, enc_feats)  # (B, 2C,  H/2,  W/2)
        d4 = self.u4(d3, enc_feats)  # (B, C,   H,    W)

        logits = self.head(d4)       # raw logits; apply sigmoid only at eval
        return logits
