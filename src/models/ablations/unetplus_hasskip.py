# has_unetplus_integration.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.blocks.has_skip import HASSkip
from src.models.unet_exp.unetplus import ResidualConvBlock


class UpBlockHAS(nn.Module):
    """
    Decoder upsampling block that *gates* the encoder skip via HAS-Skip.

    Pipeline:
      1) Bilinear upsample the decoder input to the target spatial size.
      2) 1×1 conv to reduce decoder channels to out_ch (matches aligned skip channels).
      3) Query the provided HASSkip module to obtain a gated skip for a fixed encoder level.
      4) Concatenate [decoder_reduced || gated_skip] and refine with a ResidualConvBlock.

    Args:
        in_ch:     channels of decoder input (before reduction).
        out_ch:    channels after 1×1 reduction; also the expected channels of the gated skip.
        has_mod:   HASSkip instance configured for this level (dec/align channels = out_ch).
        level_idx: which encoder level's skip to produce (0..N-1), fixed per block.

    Shapes:
        x_in:      [B, in_ch,  Hx, Wx]
        enc_feats: list of encoder maps; each shaped [B, C_i, H_i, W_i]
        out:       [B, out_ch, Hx, Wx] (refined)

    Notes:
        • We pass the *reduced* decoder feature into HASSkip as the gate's decoder context,
          so HASSkip.dec_channels == out_ch and HASSkip.align_channels == out_ch.
    """
    def __init__(self, in_ch: int, out_ch: int, has_mod: HASSkip, level_idx: int):
        super().__init__()
        self.level_idx = level_idx
        self.has_mod   = has_mod
        self.conv1x1   = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

        # after concat([out_ch, out_ch]) -> 2*out_ch
        self.refine    = ResidualConvBlock(out_ch * 2, out_ch)

    def forward(self, x_in: torch.Tensor, enc_feats: list[torch.Tensor], target_hw: tuple[int, int]):
        # 1) spatial upsample to match the target encoder level resolution
        x_up = F.interpolate(x_in, size=target_hw, mode="bilinear", align_corners=False)

        # 2) reduce decoder channels (this is the context for the HAS gate)
        x_red = self.conv1x1(x_up)  # [B, out_ch, H, W]

        # 3) request a gated skip for this level
        skip_gated = self.has_mod(enc_feats, x_red, level_l=self.level_idx)  # [B, out_ch, H, W]

        # 4) fuse + refine
        x = torch.cat([x_red, skip_gated], dim=1)  # [B, 2*out_ch, H, W]
        x = self.refine(x)                         # [B, out_ch, H, W]

        return x


class UNetPlusHASSkip(nn.Module):
    """
    UNetBaselinePlus (+) with Hierarchical Attention-Selective Skips (HAS-Skip) at all decoder levels.

    What changes vs UNetBaselinePlus:
      • Instead of concatenating raw encoder features, each decoder level queries a HAS-Skip module
        to produce a *gated, decoder-aware* skip map (per-level).
      • We keep the same depth and channel plan as UNetBaselinePlus for a drop-in replacement.

    Args:
        in_channels:         input channels (1 for fundus grayscale in your setup).
        base_ch:             base channel multiplier (default 64).
        bottleneck_dropout:  dropout prob in the bottleneck residual block.
        init_pos_prior:      initializes final 1×1 conv bias to logit(prior) for stable starts.

    Tips:
        • Use BCEWithLogitsLoss or similar; apply `torch.sigmoid` at eval/inference time.
        • This model is fully compatible with your existing training loops.

    Cite:
        HASSkip structure from your `has_skip.py`; UNet+ backbone and ResidualConvBlock from `unetplus.py`.
        :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}
    """
    def __init__(self, in_channels=1, base_ch=64, bottleneck_dropout=0.1, init_pos_prior=0.1):
        super().__init__()
        C = base_ch

        # --- Encoder (copied channel plan from UNetBaselinePlus) ---
        # We reuse the same blocks to keep behavior identical.                                           
        self.e1 = _Enc(in_channels, C)       # s1: C
        self.e2 = _Enc(C, C*2)               # s2: 2C
        self.e3 = _Enc(C*2, C*4)             # s3: 4C
        self.e4 = _Enc(C*4, C*8)             # s4: 8C

        self.bottleneck = ResidualConvBlock(C*8, C*16, dropout_p=bottleneck_dropout)  # 16C

        # --- HAS-Skip modules (one per decoder level so dec/align channels can differ) ---             
        in_per_level = [C, C*2, C*4, C*8]  # channels of [s1,s2,s3,s4]

        # Level mapping (decoder top -> bottom): d1 uses s4 (idx=3), d2 uses s3 (2), d3 uses s2 (1), d4 uses s1 (0)
        self.has_d1 = HASSkip(in_channels_per_level=in_per_level, dec_channels=C*8,  align_channels=C*8)
        self.has_d2 = HASSkip(in_channels_per_level=in_per_level, dec_channels=C*4,  align_channels=C*4)
        self.has_d3 = HASSkip(in_channels_per_level=in_per_level, dec_channels=C*2,  align_channels=C*2)
        self.has_d4 = HASSkip(in_channels_per_level=in_per_level, dec_channels=C,    align_channels=C)

        # --- Decoder blocks that consume gated skips ---
        self.u1 = UpBlockHAS(in_ch=C*16, out_ch=C*8,  has_mod=self.has_d1, level_idx=3)  # s4
        self.u2 = UpBlockHAS(in_ch=C*8,  out_ch=C*4,  has_mod=self.has_d2, level_idx=2)  # s3
        self.u3 = UpBlockHAS(in_ch=C*4,  out_ch=C*2,  has_mod=self.has_d3, level_idx=1)  # s2
        self.u4 = UpBlockHAS(in_ch=C*2,  out_ch=C,    has_mod=self.has_d4, level_idx=0)  # s1

        self.head = nn.Conv2d(C, 1, kernel_size=1)

        # initialize final bias like UNetBaselinePlus                                                    
        with torch.no_grad():
            from math import log
            p = float(init_pos_prior)
            p = min(max(p, 1e-4), 1-1e-4)
            self.head.bias.fill_(log(p/(1-p)))

        # Kaiming init for any new convs we introduced
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, in_channels, H, W]

        Returns:
            logits: [B, 1, H, W] (apply sigmoid outside if needed)
        """
        # --- Encode ---
        s1, p1 = self.e1(x)         # [B, C,   H/1,  W/1 ]
        s2, p2 = self.e2(p1)        # [B, 2C,  H/2,  W/2 ]
        s3, p3 = self.e3(p2)        # [B, 4C,  H/4,  W/4 ]
        s4, p4 = self.e4(p3)        # [B, 8C,  H/8,  W/8 ]

        # --- Bottleneck ---
        b = self.bottleneck(p4)     # [B, 16C, H/16, W/16]

        # Repeatedly provide *all* encoder features to the HASSkip modules (hierarchical fusion),
        # but each level requests a specific skip (idx 3->0).
        enc_feats = [s1, s2, s3, s4]

        # --- Decode with HAS-gated skips ---
        d1 = self.u1(b,  enc_feats, target_hw=s4.shape[-2:])  # -> [B, 8C, H/8,  W/8 ]
        d2 = self.u2(d1, enc_feats, target_hw=s3.shape[-2:])  # -> [B, 4C, H/4,  W/4 ]
        d3 = self.u3(d2, enc_feats, target_hw=s2.shape[-2:])  # -> [B, 2C, H/2,  W/2 ]
        d4 = self.u4(d3, enc_feats, target_hw=s1.shape[-2:])  # -> [B,  C,  H,    W  ]

        logits = self.head(d4)      # [B, 1, H, W]
        return logits


# --- a tiny internal encoder wrapper mirroring UNetBaselinePlus' EncoderBlock ---
class _Enc(nn.Module):
    """Mirror of unetplus.EncoderBlock: returns (skip, pooled).  Keeps code self-contained here."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ResidualConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        s = self.conv(x)
        p = self.pool(s)
        return s, p
