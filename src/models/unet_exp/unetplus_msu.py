# --- imports ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log

# --- helpers ---
def make_gn(num_channels: int, num_groups: int = 32):
    """GroupNorm is robust for small batches. Ensure groups divide channels."""
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
        s = self.conv(x)   # skip
        p = self.pool(s)   # pooled
        return s, p

# --- MSU (Eq. 2.28–2.29) ---
class MSU(nn.Module):
    """
    Multi-Scale Subtraction Unit:
      F_diff = |conv1(A)-conv1(B)| + |conv3(A)-conv3(B)| + |conv5(A)-conv5(B)|
      F_out  = Conv3x3(F_diff) -> GN -> ReLU
    """
    def __init__(self, in_channels, out_channels=None, use_norm=True, activation=True):
        super().__init__()
        out_channels = out_channels or in_channels

        # shared convs per scale (applied to both A and B)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.conv5 = nn.Conv2d(in_channels, out_channels, 5, padding=2, bias=False)

        # Eq. 2.29: post-diff compression/mixing
        self.post3 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)

        self.norm = make_gn(out_channels) if use_norm else nn.Identity()
        self.act  = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, A, B):
        d = ( (self.conv1(A) - self.conv1(B)).abs()
            + (self.conv3(A) - self.conv3(B)).abs()
            + (self.conv5(A) - self.conv5(B)).abs() )
        y = self.post3(d)
        y = self.norm(y)
        y = self.act(y)
        return y

# --- Up block with optional MSU fusion ---
class UpBlock(nn.Module):
    """
    Bilinear upsample + Conv1x1 (reduce channels) + concat with (skip OR MSU) + ResidualConvBlock.
    If use_msu=True: we fuse via MSU(upsampled, skip) and then concat [upsampled, msu].
    """
    def __init__(self, in_ch, out_ch, use_msu: bool = False):
        super().__init__()
        self.use_msu = use_msu
        self.conv1x1 = nn.Conv2d(in_ch, out_ch, 1, bias=False)

        # After concat we always have 2*out_ch channels going into the residual block
        self.conv    = ResidualConvBlock(out_ch + out_ch, out_ch)

        if self.use_msu:
            # MSU expects both inputs to have out_ch channels and same H×W
            self.msu = MSU(in_channels=out_ch, out_channels=out_ch, use_norm=True, activation=True)

    def forward(self, x, skip):
        # x: deeper decoder feature; skip: encoder feature at this level
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.conv1x1(x)  # (B, out_ch, H, W)

        if self.use_msu:
            fused = self.msu(x, skip)              # (B, out_ch, H, W)
            x = torch.cat([x, fused], dim=1)       # (B, 2*out_ch, H, W)
        else:
            x = torch.cat([x, skip], dim=1)        # (B, 2*out_ch, H, W)

        x = self.conv(x)
        return x

# --- U-Net + MSU in the decoder path ---
class UNetBaselinePlus(nn.Module):
    """
    Strong 'vanilla' U-Net for 1-channel inputs and 1-channel logits output.
    - GroupNorm (stable for small batches)
    - Residual blocks
    - Bilinear upsample
    - Optional MSU fusion inside each UpBlock
    - Light dropout only in bottleneck
    """
    def __init__(self, in_channels=1, base_ch=64, bottleneck_dropout=0.1, init_pos_prior=0.1):
        super().__init__()
        C = base_ch
        self.e1 = EncoderBlock(in_channels, C)      # 64
        self.e2 = EncoderBlock(C, C*2)              # 128
        self.e3 = EncoderBlock(C*2, C*4)            # 256
        self.e4 = EncoderBlock(C*4, C*8)            # 512

        self.bottleneck = ResidualConvBlock(C*8, C*16, dropout_p=bottleneck_dropout)  # 1024

        # MSU enabled at all decoder levels (you can toggle per-level if you wish)
        self.u1 = UpBlock(C*16, C*8, use_msu=True)   # with s4
        self.u2 = UpBlock(C*8,  C*4, use_msu=True)   # with s3
        self.u3 = UpBlock(C*4,  C*2, use_msu=True)   # with s2
        self.u4 = UpBlock(C*2,  C,   use_msu=True)   # with s1

        self.head = nn.Conv2d(C, 1, 1)

        # Initialize final bias to logit(prior) so early predictions aren't all-zero.
        with torch.no_grad():
            p = float(init_pos_prior)
            p = min(max(p, 1e-4), 1-1e-4)
            self.head.bias.fill_(log(p/(1-p)))

        # Kaiming init for all convs
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        b  = self.bottleneck(p4)

        # Each UpBlock computes MSU(upsampled, skip) internally and concatenates [upsampled, MSU]
        d1 = self.u1(b,  s4)
        d2 = self.u2(d1, s3)
        d3 = self.u3(d2, s2)
        d4 = self.u4(d3, s1)

        logits = self.head(d4)   # raw logits; apply sigmoid only at eval
        return logits
