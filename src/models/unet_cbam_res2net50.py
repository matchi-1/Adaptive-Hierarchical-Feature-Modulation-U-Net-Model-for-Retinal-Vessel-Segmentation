import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.blocks.cbam import CBAM
from src.models.blocks.res2net50 import Res2Net, Bottle2neck

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
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
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
class EncoderBlock(nn.Module):  # Replaced by Res2Net50 block
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


# ---------- U-Net with Res2Net-50 encoder + optional CBAM ----------
class UNet_Res2Net50(nn.Module):
    """
    U-Net with a Res2Net-50 encoder.

    Encoder:
        - Uses Res2Net(bottle2neck, [3,4,6,3]) as the backbone.
        - We tap five feature maps:
            c1: output after stem conv+bn+relu      (≈ H/2,  64 ch; ResNet stem uses stride=2)
            c2: output of layer1 (stage 2)          (≈ H/4,  256 ch)
            c3: output of layer2 (stage 3)          (≈ H/8,  512 ch)
            c4: output of layer3 (stage 4)          (≈ H/16, 1024 ch)
            c5: output of layer4 (stage 5)          (≈ H/32, 2048 ch)

        Why Res2Net for encoder:
            Inside each bottleneck, Res2Net splits channels into `s` parts and processes them
            in a hierarchical chain before concatenation+1x1 fuse, which yields a rich mix
            of receptive-field sizes *within one block* while keeping the ResNet scaffold.
            That multi-scale richness is valuable for dense prediction (segmentation).

    Decoder:
        - Classic U-Net style:
            upsample by 2× (ConvTranspose2d) → concat with same-scale skip → 2×(3×3 conv + BN + ReLU).
        - Channel plan (to keep it light and match common practice):
            2048 (c5) → 1024 → 512 → 256 → 128 → 64, with skip channels coming from c4..c1 reduced via 1×1.

    CBAM (optional):
        - If you pass a CBAM class via `cbam_cls`, we insert it after each lateral 1×1 (on skips)
          and on the reduced bottleneck (c5→b). This mirrors the paper’s observation that attention
          integrates cleanly with the Res2Net block; CBAM is an analogous add-on.
          Set cbam_cls=None (default) to disable.

    Output:
        - Final 1×1 conv to `out_ch` logits.
        - We add a final 2× upsample to return to the input spatial size because the ResNet/Res2Net
          stem downsamples early (conv1 stride=2, then maxpool). So total upsampling steps:
          H/32 → H/16 → H/8 → H/4 → H/2 → H.

    Args:
        in_ch       (int):  input channels
        out_ch      (int):  output channels (1 for binary logits, K for K classes).
        baseWidth   (int):  Res2Net width (w)
        scale       (int):  Res2Net scale (s)
        pretrained  (bool): if True, leaves the default encoder weights loading to your code;
                            here we just build the module. (Hook up weight loading outside.)
        cbam_cls    (type): class implementing CBAM(ch, reduction_ratio=...).
        cbam_reduction (int): CBAM reduction ratio (channel MLP width C/r).


    """
    def __init__(self, in_ch=1, out_ch=1, baseWidth=26, scale=4,
                 pretrained=False, cbam_cls=None, cbam_reduction=16):
        super().__init__()

        # ---- Encoder (Res2Net-50 scaffold) ----
        self.encoder = Res2Net(block=Bottle2neck,
                               layers=[3, 4, 6, 3],
                               baseWidth=baseWidth,
                               scale=scale)

        # Adjust stem to desired input channels (default ResNet stem expects 3)
        if in_ch != 3:
            self.encoder.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)


        #############
        self.encoder.conv1.stride = (1, 1)     # was (2,2)
        #############

        # We won't use the classification head in segmentation
        self.encoder.fc = nn.Identity()
        self.encoder.avgpool = nn.Identity()

        # ---- Lateral channel adapters (1x1) for skips and bottleneck ----
        # Reduce encoder channels to a tidy decoder plan:
        #   c1: 64  ->  64
        #   c2: 256 -> 128
        #   c3: 512 -> 256
        #   c4: 1024-> 512
        #   c5: 2048-> 1024  (bottleneck)
        self.lat1 = nn.Conv2d(   64,   64, kernel_size=1, bias=False)
        self.lat2 = nn.Conv2d(  256,  128, kernel_size=1, bias=False)
        self.lat3 = nn.Conv2d(  512,  256, kernel_size=1, bias=False)
        self.lat4 = nn.Conv2d( 1024,  512, kernel_size=1, bias=False)
        self.latb = nn.Conv2d( 2048, 1024, kernel_size=1, bias=False)

        # ---- CBAM on skips and bottleneck (identity if not provided) ----
        # nn.Identity is a tiny PyTorch layer that does nothing but behaves like a normal nn.Module
        if cbam_cls is None:
            self.cbam1 = nn.Identity(); self.cbam2 = nn.Identity()
            self.cbam3 = nn.Identity(); self.cbam4 = nn.Identity()
            self.cbamB = nn.Identity()
        else:
            self.cbam1 = cbam_cls(  64, reduction_ratio=cbam_reduction)
            self.cbam2 = cbam_cls( 128, reduction_ratio=cbam_reduction)
            self.cbam3 = cbam_cls( 256, reduction_ratio=cbam_reduction)
            self.cbam4 = cbam_cls( 512, reduction_ratio=cbam_reduction)
            self.cbamB = cbam_cls(1024, reduction_ratio=cbam_reduction)

        # ---- Decoder blocks (UpConv + Concat + 2×(3×3+BN+ReLU)) ----
        # Stages: (b=1024)→512, 512→256, 256→128, 128→64; plus a final upsample to H.
        self.d1 = DecoderBlock(1024, 512)  # bottleneck 1024 → up → fuse with 512-skip
        self.d2 = DecoderBlock( 512, 256)
        self.d3 = DecoderBlock( 256, 128)
        self.d4 = DecoderBlock( 128,  64)

        # Final upsample (because encoder stem downsampled early): H/2 -> H
        self.up_final = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        #############
        self.up_final = nn.Identity()          # instead of ConvTranspose2d(64, 64, 2, 2)
        #############


        self.final = nn.Conv2d(64, out_ch, kernel_size=1)



    def _encode(self, x):
        x = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(x)))  # stem (≈ H/2)
        c1 = x
        x  = self.encoder.maxpool(x)             # ≈ H/4
        c2 = self.encoder.layer1(x)              # 256ch, H/4
        c3 = self.encoder.layer2(c2)             # 512ch, H/8
        c4 = self.encoder.layer3(c3)             # 1024ch, H/16
        c5 = self.encoder.layer4(c4)             # 2048ch, H/32
        return c1, c2, c3, c4, c5

    def forward(self, x):
        # ---- Encoder ----
        c1, c2, c3, c4, c5 = self._encode(x)

        # ---- Lateral reduce (+ CBAM) ----
        s1 = self.cbam1(self.lat1(c1))   #  64 @ H/2
        s2 = self.cbam2(self.lat2(c2))   # 128 @ H/4
        s3 = self.cbam3(self.lat3(c3))   # 256 @ H/8
        s4 = self.cbam4(self.lat4(c4))   # 512 @ H/16
        b  = self.cbamB(self.latb(c5))   # 1024@ H/32

        # ---- Decoder ----
        x = self.d1(b,  s4)              # H/16, 512ch
        x = self.d2(x,  s3)              # H/8,  256ch
        x = self.d3(x,  s2)              # H/4,  128ch
        x = self.d4(x,  s1)              # H/2,   64ch

        x = self.up_final(x)             # H/2 -> H
        return self.final(x)