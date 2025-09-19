import torch
import torch.nn as nn
import torch.nn.functional as F

# --- MSU block (shared-weight multi-scale absolute subtraction) ----------------
class MSU(nn.Module):
    """
    Multi-scale Subtraction Unit (MSU)

    Purpose
    -------
    Compare two feature maps at three receptive fields (1×1, 3×3, 5×5) using
    shared convs; sum their absolute differences to produce a "difference map"
    that you can add (overlay) onto a base feature tensor.

    Parameters
    ----------
    in_channels : int
        Channels of each input map.
    out_channels : int, optional (default: in_channels)
        Channels of the fused difference map.
    use_bn : bool (default: True)
        Apply BatchNorm to the fused output.
    activation : bool (default: True)
        Apply ReLU to the fused output.

    Inputs
    ------
    F_A, F_B : (N, C_in, H, W) tensors, same dtype/device/shape.

    Outputs
    -------
    out : (N, C_out, H, W) tensor = |C1(F_A)-C1(F_B)| + |C3(F_A)-C3(F_B)| + |C5(F_A)-C5(F_B)|
    """
    def __init__(self, in_channels, out_channels=None, use_bn=True, activation=True):
        super().__init__()
        out_channels = out_channels or in_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=not use_bn)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_bn)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, bias=not use_bn)

        self.bn  = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = nn.ReLU(inplace=True) if activation else nn.Identity()

        for m in (self.conv1, self.conv3, self.conv5):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, F_A: torch.Tensor, F_B: torch.Tensor) -> torch.Tensor:
        a1, b1 = self.conv1(F_A), self.conv1(F_B)
        a3, b3 = self.conv3(F_A), self.conv3(F_B)
        a5, b5 = self.conv5(F_A), self.conv5(F_B)
        out = torch.abs(a1 - b1) + torch.abs(a3 - b3) + torch.abs(a5 - b5)
        return self.act(self.bn(out))


# --- Your original U-Net building blocks --------------------------------------
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

class UNetMSU(nn.Module):
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
