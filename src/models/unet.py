import torch
import torch.nn.functional as F
import torch.nn as nn
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

class DecoderBlock(nn.Module):
    """2× upsample (transpose conv) + concat with skip + ConvBlock."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # up: in_channels -> out_channels, spatial ×2
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        # after concat with skip (out_channels + out_channels)
        self.conv = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # x: decoder feature from the previous (deeper) level
        # skip: encoder feature at the corresponding scale
        
        x = self.up(x) # 2× spatial upsample + channel projection to out_channels
        
        # Due to pooling/upsampling integer rounding, shapes can drift by 1px
        # handle odd/even shapes due to pooling artifacts
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        
        # Channel-wise concatenate: [N, out_channels, H, W] + [N, out_channels, H, W]
        # → [N, 2*out_channels, H, W]
        x = torch.cat([x, skip], dim=1)
        
        # Fuse the concatenated features with two 3×3 convs (BN+ReLU inside ConvBlock),
        # bringing channels back down to `out_channels`.
        x = self.conv(x)
        
        return x

class UNet(nn.Module):
    def __init__(self, in_channels = 1):
        super().__init__()
        self.e1 = EncoderBlock(in_channels, 64)
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)
        self.e4 = EncoderBlock(256, 512)

        self.bottleneck = ConvBlock(512, 1024)

        self.d1 = DecoderBlock(1024, 512)
        self.d2 = DecoderBlock(512, 256)
        self.d3 = DecoderBlock(256, 128)
        self.d4 = DecoderBlock(128, 64)

        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        b = self.bottleneck(p4)

        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        return self.final(d4)