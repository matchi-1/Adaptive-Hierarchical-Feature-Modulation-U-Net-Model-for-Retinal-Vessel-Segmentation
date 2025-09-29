# --- imports ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log
from typing import List

# your blocks
from src.models.blocks.has_skip_exp1 import HASSkip
from src.models.blocks.msu import MSU
from src.models.blocks.cbam import CBAM   # adjust import path if needed

# --- helpers ---
def make_gn(num_channels: int, num_groups: int = 32):
    g = min(num_groups, num_channels)
    while num_channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, num_channels)

# --- core blocks ---
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
        s = self.conv(x)   # skip (pre-CBAM)
        p = self.pool(s)
        return s, p

# --- Up block: HAS-Skip + MSU fused with learnable α/β ---
class UpBlockHASMSU(nn.Module):
    """
    Bilinear upsample + Conv1x1 (→ out_ch)
    HAS(enc_feats_cbam, dec_feat=x, level) -> has  (B,out_ch,H,W)
    MSU(x, skip_cbam[level])               -> msu  (B,out_ch,H,W)
    fused = α*msu + β*has -> GN -> ReLU
    Concat [x, fused] -> ResidualConvBlock
    """
    def __init__(self, in_ch: int, out_ch: int,
                 has_module: HASSkip, msu_module: MSU, level_idx: int):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.conv    = ResidualConvBlock(out_ch + out_ch, out_ch)  # after concat

        self.has   = has_module
        self.msu   = msu_module
        self.level = int(level_idx)

        # learnable fusion scalars
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta  = nn.Parameter(torch.tensor(1.0))

        self.fuse_norm = make_gn(out_ch)
        self.fuse_act  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, enc_feats_cbam: List[torch.Tensor]) -> torch.Tensor:
        skip = enc_feats_cbam[self.level]
        target = skip.shape[-2:]

        # upsample decoder stream and set channels to out_ch
        x = F.interpolate(x, size=target, mode="bilinear", align_corners=False)
        x = self.conv1x1(x)  # (B, out_ch, H, W)

        # HAS-skip: gate using all CBAM-refined encoder tiers + this decoder feature
        has = self.has(enc_feats=enc_feats_cbam, dec_feat=x, level_l=self.level)  # (B, out_ch, H, W)

        # MSU: multi-scale difference between decoder stream and CBAM-refined skip at this level
        msu = self.msu(x, skip)  # (B, out_ch, H, W)

        fused = self.alpha * msu + self.beta * has
        fused = self.fuse_act(self.fuse_norm(fused))

        x = torch.cat([x, fused], dim=1)  # (B, 2*out_ch, H, W)
        x = self.conv(x)
        return x

# --- U-Net with CBAM-after-encoder + CBAM at bottleneck + HAS-Skip + MSU ---
class UNetBaselinePlus(nn.Module):
    """
    Modern U-Net with:
      - GN, residual encoder/decoder blocks, bilinear upsampling
      - CBAM applied AFTER each encoder block output (s1..s4) before feeding MSU/HAS
      - CBAM at the bottleneck
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

        # HAS-Skip per level (align_channels == out_ch at that level)
        self.has_l3 = HASSkip(enc_chs, dec_chs, align_channels=C*8)   # level 3 → 512
        self.has_l2 = HASSkip(enc_chs, dec_chs, align_channels=C*4)   # level 2 → 256
        self.has_l1 = HASSkip(enc_chs, dec_chs, align_channels=C*2)   # level 1 → 128
        self.has_l0 = HASSkip(enc_chs, dec_chs, align_channels=C)     # level 0 →  64

        # MSU per level (in=out=out_ch at that level)
        self.msu_l3 = MSU(in_channels=C*8, out_channels=C*8, use_bn=True,  activation=True)
        self.msu_l2 = MSU(in_channels=C*4, out_channels=C*4, use_bn=True,  activation=True)
        self.msu_l1 = MSU(in_channels=C*2, out_channels=C*2, use_bn=True,  activation=True)
        self.msu_l0 = MSU(in_channels=C,   out_channels=C,   use_bn=True,  activation=True)

        # decoder blocks using HAS+MSU fusion
        self.u1 = UpBlockHASMSU(in_ch=C*16, out_ch=C*8, has_module=self.has_l3, msu_module=self.msu_l3, level_idx=3)
        self.u2 = UpBlockHASMSU(in_ch=C*8,  out_ch=C*4, has_module=self.has_l2, msu_module=self.msu_l2, level_idx=2)
        self.u3 = UpBlockHASMSU(in_ch=C*4,  out_ch=C*2, has_module=self.has_l1, msu_module=self.msu_l1, level_idx=1)
        self.u4 = UpBlockHASMSU(in_ch=C*2,  out_ch=C,   has_module=self.has_l0, msu_module=self.msu_l0, level_idx=0)

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
        enc_feats_cbam = [r1, r2, r3, r4]

        # bottleneck + CBAM
        b  = self.bottleneck(p4)         # (B, 16C, H/16, W/16)
        b  = self.cbam_bottleneck(b)     # (B, 16C, H/16, W/16)

        # decoder with HAS+MSU fusion (using CBAM-refined encoder features)
        d1 = self.u1(b,  enc_feats_cbam)  # (B, 8C,  H/8,  W/8)
        d2 = self.u2(d1, enc_feats_cbam)  # (B, 4C,  H/4,  W/4)
        d3 = self.u3(d2, enc_feats_cbam)  # (B, 2C,  H/2,  W/2)
        d4 = self.u4(d3, enc_feats_cbam)  # (B, C,   H,    W)

        logits = self.head(d4)           # raw logits; apply sigmoid only at eval
        return logits
