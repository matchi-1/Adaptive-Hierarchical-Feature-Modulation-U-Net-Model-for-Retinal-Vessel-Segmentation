import math
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Paper reference:
Res2Net: A New Multi-scale Backbone Architecture. Gao et al. (2019, PAMI 2021)
https://arxiv.org/abs/1904.01169

Code reference:
https://github.com/Res2Net/Res2Net-PretrainedModels/tree/master
'''

# ---------- Res2Net building block ----------
"""
Purpose:
    Res2Net “bottle2neck” block (drop-in replacement for the 3×3 group inside a
    standard bottleneck). It adds a *scale* dimension by splitting channels into
    s “lanes,” chaining them, then concatenating and fusing, all inside one block.

Parameters:
    inplanes (int):     Input channels to the block.
    planes (int):       Base channels of the stage (final output has planes*expansion).
    stride (int):       Conv stride. Default: 1.
    downsample:         Optional projection for the residual path when shape/stride changes.
    baseWidth (int):    Channels per split at reference width 64 (w in the paper).
                            The 1×1 “reduce” produces channel = s * floor(planes * (w/64)).
    scale (int):        Number of splits (s). Larger s → richer multi-scale mix.
    stype (str):        'normal' for intra-stage blocks, 'stage' for the first block of a stage
                            (uses pooled last split to keep shapes aligned when downsampling).

Shapes (typical bottleneck stage):
    Input:   [B, inplanes, H,   W]
    Output:  [B, planes*4, H/Δ, W/Δ]   where Δ = stride for the first block in a stage.

Notes:
    - conv1 produces width*scale channels so we can split evenly into s lanes.
    - We build (scale−1) 3×3 branches; x1 bypasses its 3×3 to reduce params. 
    - In a “stage” (downsampling) block, each branch uses the stride; we pool the
    last split before concatenation to keep spatial sizes consistent.
"""
class Bottle2neck(nn.Module):
   
    expansion = 4
    # keeps the standard ResNet bottleneck structure
    # the block’s output has 4× planes channels

    def __init__(self, inplanes, planes, stride=1, downsample=None, 
                 baseWidth=26, scale=4, stype='normal'):
        super(Bottle2neck, self).__init__()

        # Per-paper width calculation and 1×1 "reduce": channel = s * width. 
        width = int(math.floor(planes * (baseWidth/64.0)))
        self.conv1 = nn.Conv2d(inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width*scale)

        # Number of 3×3 conv branches (x1 has no 3×3 to save params). 
        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1

        # For downsampling blocks, define pooling for the last split to keep shapes aligned.
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)

        # Build the (scale−1) 3×3 convs (each branch works on "width" channels).
        convs, bns = [], []
        for _ in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        # 1×1 "expand" to planes*expansion after concatenating all splits.
        self.conv3 = nn.Conv2d(width*scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        """
        Forward pass (maps directly to Fig. 2 and Eq. (1) in the paper).

        Steps:
          1) Reduce + split:
             - Apply 1×1 reduce (conv1/bn1/relu), then split evenly into s parts (x1..x_s),
               each with "width" channels. 
          2) Hierarchical residual chain over (scale−1) branches:
             - For normal blocks: y2 = K2(x2); y_i = K_i(x_i + y_{i-1}) for i>2 (Eq. 1). 
             - For stage (downsampling) blocks: use stride in each branch and avoid adding
               tensors of mismatched spatial size; the chain is implemented shape-safely.
          3) Concatenate all branch outputs; append either the untouched last split (normal)
             or its pooled version (stage). Fuse with a 1×1 conv (conv3/bn3). 
          4) Residual add and ReLU:
             - Add the identity (downsampled if needed) and apply ReLU. The paper positions
               this module as a drop-in for the 3×3 group inside bottlenecks. 

        Ouput:
            Tensor of shape [B, planes*expansion, H/Δ, W/Δ]
        """
        residual = x

        # 1×1 reduce → s*width channels → split into s lanes
        out = self.relu(self.bn1(self.conv1(x)))
        spx = torch.split(out, self.width, 1)

        # Process (scale−1) branches; x1 has no 3×3 per paper (parameter saving).
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                # Hierarchical residual-like chain: x_i + y_{i-1} before K_i(·).
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        # Concatenate the last split (normal) or its pooled version (stage), then fuse.
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.bn3(self.conv3(out))

        # Residual projection if shape/stride changed
        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.relu(out + residual)
        return out

"""
Purpose:
    Res2Net backbone (ResNet-50 scaffold with [3,4,6,3] blocks), where each
    bottleneck’s 3×3 group is replaced by the Res2Net bottle2neck module.

Parameters:
    block (class):      Block class to use (Bottle2neck).
    layers (int list):  List of block counts per stage (e.g., [3,4,6,3] for 50 = (3+4+6+3)*3 + 2).
    baseWidth (int):    w in paper; channels per split at reference width 64.
    scale (int):        s in paper; number of splits (control parameter).
    num_classes (int):  Classifier head output dim (for ImageNet-style usage).

Notes:
    - Res2Net replaces the group of 3×3 filters in a bottleneck with smaller
    groups connected hierarchically inside the block—stronger multi-scale
    ability at similar compute. 
    - The scale dimension is orthogonal(separate) to width/cardinality, so it integrates
    with modules like SE and with other backbones (ResNeXt/DLA/etc.)

"""
class Res2Net(nn.Module):
    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale

        # Standard ResNet stem + maxpool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Four stages with Bottle2neck blocks; first block in a stage uses stype='stage'
        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Classification head (for segmentation, you’d typically drop this head)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Kaiming init as in standard ResNet
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        Purpose: 
            Build one ResNet stage of `blocks` Bottle2neck modules.
        Parameters:
            block (Type[nn.Module]):    The residual block class to instantiate (e.g., `Bottle2neck`).
                                        Must accept the signature:
                                        `block(inplanes, planes, stride=..., downsample=..., stype=..., baseWidth=..., scale=...)`
                                        and define a class attribute `expansion` (e.g., 4 for bottleneck).
            planes (int):               The base channel width *inside* the stage's blocks. 
                                        The block's output channel count will be `planes * block.expansion`.
            blocks (int):               Number of residual blocks to stack in this stage.
            stride (int, default=1):    Spatial stride applied by the *first* block of the stage. Use 2 to
                                        downsample H and W by 2× at the stage entrance; later blocks use stride=1.
                
   
        Inputs (when the returned module is called later during forward):
            x: Tensor of shape
                [B, C_in=self.inplanes, H, W]

        Outputs:
            y: Tensor of shape
                [B, C_out=planes * block.expansion, H_out, W_out]
            where:
                H_out = H // stride   (integer division; equal to H if stride==1)
                W_out = W // stride

        Notes:
            - The first block in a stage:
                - uses `stride`>1 to downsample spatially,
                - passes `stype='stage'` to keep shapes aligned inside the block,
                - applies a residual projection (1×1) when shape/stride changes.

            - Later blocks in the same stage use `stype='normal'` and stride=1.

            - Updates `self.inplanes` to `planes * block.expansion` so the next stage
                knows its expected input channel count.

        Returns:
            nn.Sequential:
                A sequential container of `blocks` residual modules forming this stage.

        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # First block of the stage: may downsample (based on stride); marked as 'stage' for shape-safe internals.
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion

        # Remaining blocks (no further spatial downsampling inside this stage)
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Purpose:
            Forward pass (classification mode).

        Pipeline:
            conv1 → BN → ReLU → maxpool → layer1 → layer2 → layer3 → layer4 → GAP → FC

        Parameters:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Class logits of shape (B, num_classes).

        Shape notes (default ResNet/Res2Net strides):
            After conv1:      ~ (B, 64,   H/2,  W/2)
            After maxpool:    ~ (B, 64,   H/4,  W/4)
            After layer1/C2:  ~ (B, 256,  H/4,  W/4)
            After layer2/C3:  ~ (B, 512,  H/8,  W/8)
            After layer3/C4:  ~ (B, 1024, H/16, W/16)
            After layer4/C5:  ~ (B, 2048, H/32, W/32)
            After GAP:        ~ (B, 2048)
            Logits:           ~ (B, num_classes)

        Notes:
            - For segmentation, tap C1..C5 before GAP/FC and feed them to a decoder.
            Typical taps:
                C1 = ReLU(BN(conv1(x)))                # ~ H/2,  64ch
                x  = maxpool(C1)                       # ~ H/4
                C2 = layer1(x); C3 = layer2(C2)        # ~ H/4, H/8
                C4 = layer3(C3); C5 = layer4(C4)       # ~ H/16, H/32
                
            - If you change stage strides/dilations (e.g., OS=16), the spatial scales shift accordingly.
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
