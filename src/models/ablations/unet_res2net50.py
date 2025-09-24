import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Your existing building blocks
# ---------------------------

class ConvBlock(nn.Module):
    """Two 3×3 convs with BN+ReLU. Preserves H,W with padding=1."""
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

class DecoderBlock(nn.Module):
    """
    2× upsample with ConvTranspose2d, concatenate encoder skip, then fuse via ConvBlock.
    Expect skip to have same H,W as upsampled x.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels * 2, out_channels)
    def forward(self, x, skip):
        x = self.up(x)                      # H/.. -> 2×
        # If your input H,W aren’t multiples of 16, add a safe resize here (F.interpolate).
        x = torch.cat([x, skip], dim=1)     # concat along channels
        return self.conv(x)

# ---------------------------
# Res2Net encoder (Bottle2neck + 50-layer scaffold)
# ---------------------------

class Bottle2neck(nn.Module):
    """
    Res2Net block: 1×1 reduce → split into s groups (width each) → hierarchical 3×3 chain
    → concat → 1×1 fuse → residual add. 'stage' blocks handle downsampling safely.

    Args:
        inplanes (int): input channels
        planes   (int): inner base width of the block (output is planes*expansion)
        stride   (int): spatial stride for this block (downsample when >1)
        downsample (nn.Module|None): 1×1 projection for the residual when shape/stride changes
        baseWidth (int): w (channels per split at reference width 64)
        scale     (int): s (number of splits; s≥2 recommended)
        stype     (str): 'normal' or 'stage' (first block in a stage)
    """
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        super().__init__()
        assert scale >= 2, "scale (s) must be >= 2 for Res2Net"

        self.scale = scale
        self.stype = stype
        self.downsample = downsample

        # 1×1 reduce: produce s*width channels so we can split evenly into s parts
        width = int(math.floor(planes * (baseWidth / 64.0)))
        channel = width * scale
        self.conv1 = nn.Conv2d(inplanes, channel, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channel)
        self.relu  = nn.ReLU(inplace=True)

        # Build (s-1) 3×3 branch convs; use 'stride' on all branches in downsampling blocks
        self.nums = scale - 1
        self.convs = nn.ModuleList([
            nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False)
            for _ in range(self.nums)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm2d(width) for _ in range(self.nums)])

        # 1×1 fuse after concatenation
        self.conv3 = nn.Conv2d(channel, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * self.expansion)

        # Pool the last split only when actually downsampling (shape-safe concat)
        self.pool_last = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1) \
                         if (stype == 'stage' and stride > 1) else None

        self.width = width

    def forward(self, x):
        """
        Forward:
          - Reduce & split into s parts of 'width' channels.
          - For i in [0..s-2]: run branch 3×3; in 'normal' blocks do hierarchical add (x_i + prev).
          - Concatenate all branch outputs plus the last raw (or pooled) split.
          - Fuse (1×1), add residual, ReLU.
        """
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        spx = torch.split(out, self.width, dim=1)   # s tensors, each [B,width,H,W]

        # Branches over first s-1 splits
        for i in range(self.nums):
            if i == 0:
                sp = spx[0]
            else:
                # In downsampling ('stage') blocks we avoid adding tensors with diff H,W
                sp = sp + spx[i] if self.pool_last is None else spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            out = sp if i == 0 else torch.cat([out, sp], dim=1)

        # Append last split (raw for normal; pooled for stage/downsampling)
        last = spx[self.nums]
        if self.pool_last is not None:
            last = self.pool_last(last)
        out = torch.cat([out, last], dim=1)

        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(out + identity)
        return out

class Res2NetEncoder(nn.Module):
    """
    Res2Net-50 encoder that returns multi-scale feature maps (C1..C5) at OS=16.

    Design:
        - Gentler stem (conv1 stride=1) so C1 is full-res (H×W).
        - Then maxpool 2×, and stage strides [1,2,2,2] → C5 at H/16.
        - Each stage uses Bottle2neck blocks with (baseWidth=w, scale=s).

    Args:
        in_ch (int): input channels (1 for grayscale, 3 for RGB)
        baseWidth (int): Res2Net width w (e.g., 26 → “26w×4s”)
        scale (int): Res2Net scale s (e.g., 4)

    Returns (forward):
        C1: [B,  64,  H,   W]
        C2: [B, 256,  H/2, W/2]
        C3: [B, 512,  H/4, W/4]
        C4: [B,1024,  H/8, W/8]
        C5: [B,2048, H/16, W/16]
    """
    def __init__(self, in_ch=1, baseWidth=26, scale=4):
        super().__init__()
        self.inplanes = 64
        self.baseWidth = baseWidth
        self.scale = scale

        # Stem: keep stride=1 so C1 is at full resolution; simple 2× maxpool next
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)  # H -> H/2

        # Stages: first block 'stage' may downsample; later blocks 'normal'
        self.layer1 = self._make_layer(planes= 64, blocks=3, stride=1)  # H/2
        self.layer2 = self._make_layer(planes=128, blocks=4, stride=2)  # H/4
        self.layer3 = self._make_layer(planes=256, blocks=6, stride=2)  # H/8
        self.layer4 = self._make_layer(planes=512, blocks=3, stride=2)  # H/16

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottle2neck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottle2neck.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottle2neck.expansion)
            )
        layers = [Bottle2neck(self.inplanes, planes, stride=stride,
                              downsample=downsample, baseWidth=self.baseWidth,
                              scale=self.scale, stype='stage')]
        self.inplanes = planes * Bottle2neck.expansion
        for _ in range(1, blocks):
            layers.append(Bottle2neck(self.inplanes, planes,
                                      baseWidth=self.baseWidth, scale=self.scale, stype='normal'))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Returns five feature maps for skip connections (C1..C5).
        """
        x  = self.relu(self.bn1(self.conv1(x)))  # C1: H×W
        c1 = x
        x  = self.pool(x)                        # H/2
        c2 = self.layer1(x)                      # 256ch
        c3 = self.layer2(c2)                     # 512ch
        c4 = self.layer3(c3)                     # 1024ch
        c5 = self.layer4(c4)                     # 2048ch
        return c1, c2, c3, c4, c5

# ---------------------------
# U-Net with Res2Net encoder
# ---------------------------

class UNet_Res2Net50(nn.Module):
    """
    U-Net that uses Res2Net-50 as the encoder (OS=16) and your decoder.

    • Encoder: returns C1..C5 (64/256/512/1024/2048 ch at H, H/2, H/4, H/8, H/16).
    • Lateral 1×1 "adapters": shrink encoder channels to match your decoder plan
      (64, 128, 256, 512 for skips; 1024 for bottleneck).
    • Decoder: your four DecoderBlocks (×2 upsample) bring H/16 → H.

    Args:
        in_ch (int): input channels (1 or 3)
        out_ch (int): output channels (e.g., 1 for binary logits)
        baseWidth (int): Res2Net width w (e.g., 26)
        scale (int): Res2Net scale s (e.g., 4)
    """
    def __init__(self, in_ch=1, out_ch=1, baseWidth=26, scale=4):
        super().__init__()
        self.enc = Res2NetEncoder(in_ch=in_ch, baseWidth=baseWidth, scale=scale)

        # Lateral 1×1 to match decoder’s expected channel sizes
        self.lat1 = nn.Conv2d(   64,   64, kernel_size=1)   # C1 → 64
        self.lat2 = nn.Conv2d(  256,  128, kernel_size=1)   # C2 → 128
        self.lat3 = nn.Conv2d(  512,  256, kernel_size=1)   # C3 → 256
        self.lat4 = nn.Conv2d( 1024,  512, kernel_size=1)   # C4 → 512
        self.latb = nn.Conv2d( 2048, 1024, kernel_size=1)   # C5 → 1024

        # Your decoder (unchanged)
        self.d1 = DecoderBlock(1024, 512)  # H/16 -> H/8
        self.d2 = DecoderBlock( 512, 256)  # H/8  -> H/4
        self.d3 = DecoderBlock( 256, 128)  # H/4  -> H/2
        self.d4 = DecoderBlock( 128,  64)  # H/2  -> H

        self.final = nn.Conv2d(64, out_ch, kernel_size=1)

    def forward(self, x):
        """
        Forward:
          1) Encode: get C1..C5 from Res2Net-50 at scales H .. H/16.
          2) Adapt channels with 1×1 convs for clean skip sizes.
          3) Decode: 4×(up→concat→fuse) to full resolution.
        """
        c1, c2, c3, c4, c5 = self.enc(x)

        # shrink channels to match decoder
        s1 = self.lat1(c1)    #  64 @ H
        s2 = self.lat2(c2)    # 128 @ H/2
        s3 = self.lat3(c3)    # 256 @ H/4
        s4 = self.lat4(c4)    # 512 @ H/8
        b  = self.latb(c5)    # 1024@ H/16

        d1 = self.d1(b,  s4)  # H/8
        d2 = self.d2(d1, s3)  # H/4
        d3 = self.d3(d2, s2)  # H/2
        d4 = self.d4(d3, s1)  # H

        return self.final(d4)
