import torch
import torch.nn as nn
import torch.nn.functional as F

# Base blocks (unchanged)
from src.models.unet import ConvBlock, EncoderBlock, DecoderBlock
# Your HAS block
from src.models.blocks.has_skip import HASSkip
# Your CBAM block (Channel then optional Spatial)
from src.models.blocks.cbam import CBAM


def _upsample_to(x, ref_like):
    """Bilinear upsample x to ref_like's HxW."""
    if x.shape[-2:] != ref_like.shape[-2:]:
        x = F.interpolate(x, size=ref_like.shape[-2:], mode="bilinear", align_corners=False)
    return x


class ResidualCBAM(nn.Module):
    """
    y = x + alpha * (CBAM(x) - x), alpha∈[0,1] (learnable).
    Start with small alpha to protect recall; let training increase where helpful.
    """
    def __init__(self, channels, reduction=16, use_spatial=True, alpha_init=0.15):
        super().__init__()
        self.cbam = CBAM(channels, reduction_ratio=reduction, use_spatial=use_spatial)
        # Smoothly-bounded learnable gate in [0,1]
        self._raw_alpha = nn.Parameter(torch.tensor(float(alpha_init)).logit())  # inverse-sigmoid init
    def forward(self, x):
        y = self.cbam(x)
        alpha = torch.sigmoid(self._raw_alpha)  # in (0,1)
        return x + alpha * (y - x)


class UNetWithHASSkipCBAM(nn.Module):
    """
    Base U-Net + HAS-Skip + Residual CBAM on HAS-generated skips (and light bottleneck CBAM).
    Skips per decoder level:
      d1 <- HAS(E4), 512ch (CBAM channel-only)
      d2 <- HAS(E3), 256ch (CBAM channel-only)
      d3 <- HAS(E2), 128ch (CBAM channel-only)
      d4 <- HAS(E1),  64ch (CBAM channel+spatial)
    """
    def __init__(self, in_channels: int = 1,
                 cbam_reduction: int = 16,
                 use_bottleneck_cbam: bool = True,
                 bott_alpha_init: float = 0.10,
                 skip_alpha_inits = (0.20, 0.20, 0.20, 0.30)  # d1..d4
                 ):
        super().__init__()

        # -------- Base U-Net (unchanged math) --------
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

        # -------- HAS-Skip producers (one per decoder level) --------
        self.has_d1 = HASSkip([64,128,256,512], dec_channels=1024, align_channels=512)
        self.has_d2 = HASSkip([64,128,256,512], dec_channels=512,  align_channels=256)
        self.has_d3 = HASSkip([64,128,256,512], dec_channels=256,  align_channels=128)
        self.has_d4 = HASSkip([64,128,256,512], dec_channels=128,  align_channels=64)

        # -------- Residual CBAM on HAS-generated skips --------
        # Strategy: channel-only on deeper levels, spatial+channel only near the output.
        a1, a2, a3, a4 = skip_alpha_inits
        self.rcbam_skip1 = ResidualCBAM(512, reduction=cbam_reduction, use_spatial=False, alpha_init=a1)
        self.rcbam_skip2 = ResidualCBAM(256, reduction=cbam_reduction, use_spatial=False, alpha_init=a2)
        self.rcbam_skip3 = ResidualCBAM(128, reduction=cbam_reduction, use_spatial=False, alpha_init=a3)
        self.rcbam_skip4 = ResidualCBAM( 64, reduction=cbam_reduction, use_spatial=True,  alpha_init=a4)

        # -------- Optional light bottleneck CBAM (very gentle) --------
        if use_bottleneck_cbam:
            self.rcbam_bott = ResidualCBAM(1024, reduction=cbam_reduction, use_spatial=False, alpha_init=bott_alpha_init)
        else:
            self.rcbam_bott = nn.Identity()

        # with channel-only CBAM if you find it stabilizes precision further:
        # self.rcbam_d2 = ResidualCBAM(256, reduction=cbam_reduction, use_spatial=False, alpha_init=0.10)
        # self.rcbam_d3 = ResidualCBAM(128, reduction=cbam_reduction, use_spatial=False, alpha_init=0.10)
        # Keep them commented out for a clean ablation first.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ----- encoders -----
        s1, p1 = self.e1(x)     # 64,  H
        s2, p2 = self.e2(p1)    # 128, H/2
        s3, p3 = self.e3(p2)    # 256, H/4
        s4, p4 = self.e4(p3)    # 512, H/8

        # ----- bottleneck (+ gentle CBAM) -----
        b  = self.bottleneck(p4)           # 1024, H/16 (your base UNet uses 4 downsamples; keep as is)
        b  = self.rcbam_bott(b)

        enc_feats = [s1, s2, s3, s4]

        # ----- d1 (scale E4) -----
        b_up  = _upsample_to(b, s4)                            # 1024 @ E4 scale
        skip1 = self.has_d1(enc_feats, dec_feat=b_up, level_l=3)   # gate E4 -> 512
        skip1 = self.rcbam_skip1(skip1)                        # CBAM (channel-only)
        d1    = self.d1(b, skip1)                              # -> 512 @ E4 scale

        # ----- d2 (scale E3) -----
        d1_up = _upsample_to(d1, s3)                           # 512 @ E3 scale
        skip2 = self.has_d2(enc_feats, dec_feat=d1_up, level_l=2)  # gate E3 -> 256
        skip2 = self.rcbam_skip2(skip2)                        # CBAM (channel-only)
        d2    = self.d2(d1, skip2)                             # -> 256 @ E3 scale
        # d2 = self.rcbam_d2(d2)  # (optional)

        # ----- d3 (scale E2) -----
        d2_up = _upsample_to(d2, s2)                           # 256 @ E2 scale
        skip3 = self.has_d3(enc_feats, dec_feat=d2_up, level_l=1)  # gate E2 -> 128
        skip3 = self.rcbam_skip3(skip3)                        # CBAM (channel-only)
        d3    = self.d3(d2, skip3)                             # -> 128 @ E2 scale
        # d3 = self.rcbam_d3(d3)  # (optional)

        # ----- d4 (scale E1) -----
        d3_up = _upsample_to(d3, s1)                           # 128 @ E1 scale
        skip4 = self.has_d4(enc_feats, dec_feat=d3_up, level_l=0)  # gate E1 -> 64
        skip4 = self.rcbam_skip4(skip4)                        # CBAM (channel+spatial) to clean speckles
        d4    = self.d4(d3, skip4)                             # -> 64 @ E1 scale

        return self.final(d4)                                  # logits
