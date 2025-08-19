import torch
import torch.nn as nn
from blocks.cbam import CBAM

'''
Paper reference: Ronneberger, O., Fischer, P., & Brox, T. (2015). 
https://doi.org/10.48550/arXiv.1505.04597
'''

'''
class ConvBlock
    Purpose:
        Two 3x3 Conv2d -> BatchNorm2d -> ReLU layers; padding=1 to preserve H,W.

    Parameters:
        in_channels (int):  Input channel count.
        out_channels (int): Output channel count.

    Inputs:
        x: Tensor of shape (B, in_channels, H, W)

    Outputs:
        y: Tensor of shape (B, out_channels, H, W)
'''
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

'''
class EncoderBlock
    Purpose:
        Downsampling block: ConvBlock then 2x2 MaxPool.

    Parameters:
        in_channels (int):  Input channel count.
        out_channels (int): Output channel count.

    Inputs:
        x: Tensor (B, in_channels, H, W)

    Outputs:
        skip (Tensor): Features before pooling  (B, out_channels, H,   W)
        pooled (Tensor): Features after pooling (B, out_channels, H/2, W/2)
'''
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        return x, p



'''
class DecoderBlock
    Purpose:
        Upsampling block: 2x up-conv, concatenate skip, then ConvBlock to fuse.

    Parameters:
        in_channels (int):  Input channel count to the up-conv.
        out_channels (int): Output channel count after fusion.

    Inputs:
        x (Tensor):   Decoder input (B, in_channels, H/2, W/2)
        skip (Tensor):Encoder skip   (B, out_channels, H,   W)

    Outputs:
        y (Tensor):   Fused features (B, out_channels, H,   W)
'''
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


'''
    Purpose:
        encoder–decoder with skip connections for per-pixel segmentation,
        contracts to capture context, expands to recover resolution, outputs logits at input size.
        integrated with CBAM to improve the model’s ability to suppress noise and highlight 
        vascular features in low-contrast fundus images.

    Parameters:
        in_ch (int):       Input channels (e.g., 1 grayscale, 3 RGB). Default: 1.
        out_ch (int):      Output channels (e.g., 1 for binary logits, K for K-class). Default: 1.
        reduction (int):   CBAM reduction ratio r for the channel MLP (C -> C/r -> C). Default: 16.
        use_spatial (bool):Enable CBAM spatial attention (channel+spatial if True, channel-only if False). Default: True.

    Inputs:
        x: Tensor (B, in_ch, H, W)

    Outputs:
        logits: Tensor (B, out_ch, H, W)
'''
class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, reduction=16, use_spatial=True):
        super().__init__()
        self.e1 = EncoderBlock(1, 64)
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)
        self.e4 = EncoderBlock(256, 512)

        # self.cbam_e1 = CBAM(64,  reduction_ratio=reduction, 
        #                       use_spatial=use_spatial)
        # self.cbam_e2 = CBAM(128, reduction_ratio=reduction, 
        #                       use_spatial=use_spatial)
        self.cbam_e3 = CBAM(256, reduction_ratio=reduction,
                            use_spatial=use_spatial)
        self.cbam_e4 = CBAM(512, reduction_ratio=reduction,
                            use_spatial=use_spatial)


        self.bottleneck = ConvBlock(512, 1024)
        self.cbam_bott  = CBAM(1024, reduction_ratio=reduction,
                               use_spatial=use_spatial)

        self.d1 = DecoderBlock(1024, 512)
        self.d2 = DecoderBlock(512, 256)
        self.d3 = DecoderBlock(256, 128)
        self.d4 = DecoderBlock(128, 64)

        # CBAM on early decoder outputs
        self.cbam_d1 = CBAM(512, reduction_ratio=reduction,
                            use_spatial=use_spatial)
        self.cbam_d2 = CBAM(256, reduction_ratio=reduction,
                            use_spatial=use_spatial)

        self.final = nn.Conv2d(64, out_ch, kernel_size=1)

    def forward(self, x):
        s1, p1 = self.e1(x);    # s1 = self.cbam_e1(s1)       # 64
        s2, p2 = self.e2(p1)    # s2 = self.cbam_e1(s2)      # 128
        s3, p3 = self.e3(p2);   s3 = self.cbam_e3(s3)     # 256 + CBAM
        s4, p4 = self.e4(p3);   s4 = self.cbam_e4(s4)     # 512 + CBAM

        b = self.bottleneck(p4); b = self.cbam_bott(b)  # 1024 + CBAM

        d1 = self.d1(b,  s4); d1 = self.cbam_d1(d1)     # 512 + CBAM
        d2 = self.d2(d1, s3); d2 = self.cbam_d2(d2)     # 256 + CBAM
        d3 = self.d3(d2, s2)        # 128
        d4 = self.d4(d3, s1)        # 64

        return self.final(d4)