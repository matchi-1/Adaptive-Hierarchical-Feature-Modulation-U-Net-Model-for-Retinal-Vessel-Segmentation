import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- your blocks (unchanged) ----
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
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

class DecoderBlock(nn.Module):
    """2× upsample (transpose conv) + concat with skip + ConvBlock."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels * 2, out_channels)
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


# ---- M-Net style UNet (multi-scale left leg + deep-supervised right leg) ----
class MNetUNet(nn.Module):
    """
    M-Net style U-Net for 2D retinal vessel segmentation.
    - Left leg: feed multi-scale (1×, 1/2, 1/4, 1/8) versions of the raw input into each encoder stage.
      We project each scale with a small conv and fuse with the stage input via 1×1 conv before the EncoderBlock.
    - Right leg: deep supervision. Produce side outputs from each decoder stage, upsample to input size,
      and fuse them (learnable non-negative weights) into a single final logit.
    References: M-Net design ideas (multi-scale inputs + deep supervision). :contentReference[oaicite:1]{index=1}
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 base_ch: int = 64, return_side_outputs: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.return_side_outputs = return_side_outputs

        # Encoder channel plan
        c1, c2, c3, c4, cB = base_ch, base_ch*2, base_ch*4, base_ch*8, base_ch*16

        # ---- Left leg (multi-scale input projections) ----
        # Pools to 1/2, 1/4, 1/8 for feeding into e2, e3, e4
        self.pool2 = nn.AvgPool2d(2, 2)
        self.pool4 = nn.AvgPool2d(4, 4)
        self.pool8 = nn.AvgPool2d(8, 8)

        # Project raw input at each scale to the *stage input channels* we will fuse with
        self.left_proj_e1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.left_proj_e2 = nn.Conv2d(in_channels, c1,         kernel_size=3, padding=1, bias=False)
        self.left_proj_e3 = nn.Conv2d(in_channels, c2,         kernel_size=3, padding=1, bias=False)
        self.left_proj_e4 = nn.Conv2d(in_channels, c3,         kernel_size=3, padding=1, bias=False)

        # 1×1 fusers to bring concatenation back to the expected stage input channels
        self.fuse_e1 = nn.Conv2d(in_channels + in_channels, in_channels, kernel_size=1)
        self.fuse_e2 = nn.Conv2d(c1 + c1, c1, kernel_size=1)
        self.fuse_e3 = nn.Conv2d(c2 + c2, c2, kernel_size=1)
        self.fuse_e4 = nn.Conv2d(c3 + c3, c3, kernel_size=1)

        # ---- Encoder ----
        self.e1 = EncoderBlock(in_channels, c1)
        self.e2 = EncoderBlock(c1, c2)
        self.e3 = EncoderBlock(c2, c3)
        self.e4 = EncoderBlock(c3, c4)

        # ---- Bottleneck ----
        self.bottleneck = ConvBlock(c4, cB)

        # ---- Decoder ----
        self.d1 = DecoderBlock(cB, c4)
        self.d2 = DecoderBlock(c4, c3)
        self.d3 = DecoderBlock(c3, c2)
        self.d4 = DecoderBlock(c2, c1)

        # ---- Right leg: deep supervision heads (1×1 to logits) ----
        self.side1 = nn.Conv2d(c4, out_channels, kernel_size=1)  # after d1
        self.side2 = nn.Conv2d(c3, out_channels, kernel_size=1)  # after d2
        self.side3 = nn.Conv2d(c2, out_channels, kernel_size=1)  # after d3
        self.side4 = nn.Conv2d(c1, out_channels, kernel_size=1)  # after d4 (shallowest)

        # learnable non-negative fusion weights
        self._w = nn.Parameter(torch.ones(4))  # will pass through softplus to keep >=0

        # Final 1×1 if you want a single-head output (kept for parity with plain UNet)
        # Not strictly necessary because side4 already provides a head; we still keep it.
        self.final = nn.Conv2d(c1, out_channels, kernel_size=1)

        # lightweight norms for left-leg projections
        self.bn_in   = nn.BatchNorm2d(in_channels)
        self.bn_c1   = nn.BatchNorm2d(c1)
        self.bn_c2   = nn.BatchNorm2d(c2)
        self.bn_c3   = nn.BatchNorm2d(c3)

        self.relu = nn.ReLU(inplace=True)

    def _fuse_left_leg(self, x_raw, x_stage_in, proj, fuse, pool=None, bn=None):
        """
        x_raw: original input at full res
        x_stage_in: tensor that will enter the encoder stage (shape defines target H×W)
        proj: conv to project raw->stage_channels
        fuse: 1×1 conv to bring concat([x_stage_in, proj_scaled]) back to stage_channels
        pool: optional pooling op to downsample x_raw to x_stage_in's size
        """
        if pool is None:
            z = x_raw
        else:
            # ensure exact spatial alignment with the stage tensor
            z = pool(x_raw)
            if z.shape[-2:] != x_stage_in.shape[-2:]:
                z = F.interpolate(z, size=x_stage_in.shape[-2:], mode="bilinear", align_corners=False)

        z = proj(z)
        if bn is not None:
            z = bn(z)
        z = self.relu(z)
        x_fused = torch.cat([x_stage_in, z], dim=1)
        x_fused = self.relu(fuse(x_fused))
        return x_fused

    def forward(self, x):
        N, _, H, W = x.shape

        # ---- Left leg + Encoder ----
        # Stage e1: fuse raw(1×) with input
        x1_in = self._fuse_left_leg(x_raw=self.bn_in(x), x_stage_in=x, proj=self.left_proj_e1,
                                    fuse=self.fuse_e1, pool=None, bn=None)
        s1, p1 = self.e1(x1_in)

        # Stage e2: fuse raw(1/2) with p1
        p1n = self.bn_c1(p1)
        x2_in = self._fuse_left_leg(x_raw=x, x_stage_in=p1n, proj=self.left_proj_e2,
                                    fuse=self.fuse_e2, pool=self.pool2, bn=self.bn_c1)
        s2, p2 = self.e2(x2_in)

        # Stage e3: fuse raw(1/4) with p2
        p2n = self.bn_c2(p2)
        x3_in = self._fuse_left_leg(x_raw=x, x_stage_in=p2n, proj=self.left_proj_e3,
                                    fuse=self.fuse_e3, pool=self.pool4, bn=self.bn_c2)
        s3, p3 = self.e3(x3_in)

        # Stage e4: fuse raw(1/8) with p3
        p3n = self.bn_c3(p3)
        x4_in = self._fuse_left_leg(x_raw=x, x_stage_in=p3n, proj=self.left_proj_e4,
                                    fuse=self.fuse_e4, pool=self.pool8, bn=self.bn_c3)
        s4, p4 = self.e4(x4_in)

        # ---- Bottleneck ----
        b = self.bottleneck(p4)

        # ---- Decoder ----
        d1 = self.d1(b, s4)  # deepest decoder feat (c4)
        d2 = self.d2(d1, s3) # (c3)
        d3 = self.d3(d2, s2) # (c2)
        d4 = self.d4(d3, s1) # (c1)

        # ---- Right leg deep supervision: side logits upsampled to input size ----
        side1 = F.interpolate(self.side1(d1), size=(H, W), mode="bilinear", align_corners=False)
        side2 = F.interpolate(self.side2(d2), size=(H, W), mode="bilinear", align_corners=False)
        side3 = F.interpolate(self.side3(d3), size=(H, W), mode="bilinear", align_corners=False)
        side4 = F.interpolate(self.side4(d4), size=(H, W), mode="bilinear", align_corners=False)

        # Fuse with learned non-negative weights
        w = F.softplus(self._w)  # ensures >= 0
        fused = (w[0]*side1 + w[1]*side2 + w[2]*side3 + w[3]*side4) / (w.sum() + 1e-8)

        # You can also emit a plain single-head output (kept for parity with base UNet):
        # head = self.final(d4)  # not used in fused path

        if self.return_side_outputs:
            return fused, [side1, side2, side3, side4]
        return fused
