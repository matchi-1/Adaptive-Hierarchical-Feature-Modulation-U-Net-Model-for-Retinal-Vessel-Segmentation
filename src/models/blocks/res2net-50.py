import math
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Paper reference:
Res2Net: A New Multi-scale Backbone Architecture. Gao et al. (2019, PAMI 2021)
https://arxiv.org/abs/1904.01169

Code reference:
https://github.com/Res2Net/Res2Net-PretrainedModels/blob/master/res2net.py
'''

# ---------- Res2Net building block ----------
"""
  Res2Net “bottle2neck” block (drop-in replacement for the 3×3 group inside a
  standard bottleneck). It adds a *scale* dimension by splitting channels into
  s “lanes,” chaining them, then concatenating and fusing, all inside one block.

  Args:
      inplanes (int):   Input channels to the block.
      planes (int):     Base channels of the stage (final output has planes*expansion).
      stride (int):     Stride used by the 3×3 branches (downsampling when >1).
      downsample:       Optional projection for the residual path when shape/stride changes.
      baseWidth (int):  Channels per split at reference width 64 (w in the paper).
                        The 1×1 “reduce” produces channel = s * floor(planes * (w/64)).
      scale (int):      Number of splits (s). Larger s → richer multi-scale mix.
      stype (str):      'normal' for intra-stage blocks, 'stage' for the first block of a stage
                        (uses pooled last split to keep shapes aligned when downsampling).

  Shapes (typical bottleneck stage):
      Input:   [B, inplanes, H,   W]
      Output:  [B, planes*4, H/Δ, W/Δ]   where Δ = stride for the first block in a stage.

  Implementation notes:
    • conv1 produces width*scale channels so we can split evenly into s lanes.
    • We build (scale−1) 3×3 branches; x1 bypasses its 3×3 to reduce params. 
    • In a “stage” (downsampling) block, each branch uses the stride; we pool the
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

        Returns:
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
    Res2Net backbone (ResNet-50 scaffold with [3,4,6,3] blocks), where each
    bottleneck’s 3×3 group is replaced by the Res2Net bottle2neck module.

    Paper context:
      • Res2Net replaces the group of 3×3 filters in a bottleneck with smaller
        groups connected hierarchically inside the block—stronger multi-scale
        ability at similar compute. 
      • The scale dimension is orthogonal to width/cardinality, so it integrates
        with modules like SE and with other backbones (ResNeXt/DLA/etc.).
      • Multi-scale representations benefit detection/segmentation and other
        dense tasks; this backbone is commonly used as an encoder there. 

    Args:
        block:      Block class to use (Bottle2neck).
        layers:     List of block counts per stage (e.g., [3,4,6,3] for “50”).
        baseWidth:  w in paper; channels per split at reference width 64.
        scale:      s in paper; number of splits (control parameter).
        num_classes:Classifier head output dim (for ImageNet-style usage).
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
        Build one ResNet stage of `blocks` Bottle2neck modules.

        The first block in a stage:
          • uses `stride`>1 to downsample spatially,
          • passes `stype='stage'` to keep shapes aligned inside the block,
          • applies a residual projection (1×1) when shape/stride changes.

        Later blocks in the same stage use `stype='normal'` and stride=1.

        This preserves the classic ResNet scaffold while swapping in the
        within-block multi-scale module described by Res2Net.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Standard ResNet forward for classification:
            Stem → Stage1..4 → GAP → FC.

        For U-Net/segmentation:
            You would *tap* the outputs after stem/layer1..4 (C1..C5) and
            feed them to a decoder instead of (or in addition to) the FC head.
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


#def res2net50_26w_4s(pretrained=False, **kwargs):
#    """Constructs a Res2Net-50_26w_4s model.
#    Args:
#        pretrained (bool): If True, returns a model pre-trained on ImageNet
#    """
#    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth = 26, scale = 4, **kwargs)
#    if pretrained:
#        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_26w_4s']))
#    return model
# 'res2net50_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_4s-06e79181.pth',