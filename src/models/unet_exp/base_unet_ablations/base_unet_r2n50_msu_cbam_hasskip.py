# res2unet_msu_cbam_has.py
# ---------------------------------------------------------------
# Segmentation model:
#   Res2Net-50 encoder  → MSU graph → HAS-Skip → UNet decoder
#   + optional CBAM on bottleneck and on fused skips
# ---------------------------------------------------------------

from __future__ import annotations
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Utility
# ---------------------------
def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] != ref.shape[-2:]:
        x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
    return x


# ===============================================================
#               Res2Net-50 (minimal, segmentation-ready)
#   (bottleneck "Bottle2neck" and Res2Net backbone with
#    forward_features() that returns [C2,C3,C4,C5] = [256,512,1024,2048]
# ===============================================================

class Bottle2neck(nn.Module):
    """
    Res2Net bottleneck block.
    Args:
        inplanes: input channels
        planes:   base channels (before expansion)
        stride:   stride on 3x3 convs (applied on the first split path)
        baseWidth: Res2Net base width (default 26)
        scale:    number of splits (default 4)
        stype:    'normal' or 'stage' (stage adds avgpool on identity)
        expansion: output channels multiplier (ResNet bottleneck uses 4)
    """
    expansion = 4

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: nn.Module | None = None,
                 baseWidth: int = 26,
                 scale: int = 4,
                 stype: str = 'normal'):
        super().__init__()
        assert scale >= 1
        self.scale = scale
        self.stype = stype

        width = int(math.floor(planes * (baseWidth / 64.0)))
        channel = width * scale

        # 1x1 reduce to grouped width
        self.conv1 = nn.Conv2d(inplanes, channel, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channel)

        # Scale paths: (scale-1) 3x3 convs + the last split either pass-through (normal) or avgpool (stage)
        self.convs = nn.ModuleList([nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False)
                                    for _ in range(scale - 1)])
        self.bns   = nn.ModuleList([nn.BatchNorm2d(width) for _ in range(scale - 1)])
        self.relu  = nn.ReLU(inplace=True)

        # 1x1 expand back to planes*4
        self.conv3 = nn.Conv2d(channel, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * self.expansion)

        self.downsample = downsample
        self.stride     = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # split along channel (scale chunks)
        spx = torch.chunk(out, self.scale, dim=1)

        y = []
        for s in range(self.scale - 1):
            if s == 0:
                z = spx[s]
            else:
                z = spx[s]
            if self.stride > 1:
            # downsample spx[s] to match y[s-1]
                z = F.avg_pool2d(z, kernel_size=3, stride=self.stride, padding=1)
                z = z + y[s - 1]


            z = self.convs[s](z) # this conv already has stride=self.stride for s==0 in your build
            z = self.bns[s](z)
            z = self.relu(z)
            y.append(z)


        # last split remains as in 'stage' type
        if self.scale > 1:
            if self.stype == 'stage':
                y.append(F.avg_pool2d(spx[-1], kernel_size=3, stride=self.stride, padding=1))
            else:
                y.append(spx[-1])

        if self.scale > 1:
            # last split: either identity or avg-pooled (for stage transition)
            if self.stype == 'stage':
                y.append(F.avg_pool2d(spx[-1], kernel_size=3, stride=self.stride, padding=1))
            else:
                y.append(spx[-1])

        out = torch.cat(y, dim=1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out  = self.relu(out)
        return out


class Res2Net(nn.Module):
    """
    Res2Net like ResNet: conv1 → pool → layer1..layer4.
    We keep a classifier head for completeness but for segmentation we’ll use forward_features().
    """
    def __init__(self,
                 block: type[Bottle2neck],
                 layers: Tuple[int, int, int, int],
                 baseWidth: int = 26,
                 scale: int = 4,
                 num_classes: int = 1000):
        super().__init__()
        self.inplanes = 64
        self.baseWidth = baseWidth
        self.scale = scale

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # H/2
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)                # H/4

        self.layer1 = self._make_layer(block,  64, layers[0])                          # C2: 256 @ H/4
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, stype='stage') # C3: 512 @ H/8
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, stype='stage') # C4: 1024@ H/16
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, stype='stage') # C5: 2048@ H/32

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc      = nn.Linear(512 * block.expansion, num_classes)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def _make_layer(self, block, planes, blocks, stride=1, stype='normal'):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample,
                        baseWidth=self.baseWidth, scale=self.scale, stype=stype)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))
        return nn.Sequential(*layers)

    # classification forward (unused for segmentation)
    def forward(self, x):
        c2, c3, c4, c5 = self.forward_features(x)
        out = self.avgpool(c5).flatten(1)
        return self.fc(out)

    # segmentation-friendly forward
    def forward_features(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x)   # 64 @ H/2
        x = self.maxpool(x)                                     # 64 @ H/4
        c2 = self.layer1(x)                                     # 256 @ H/4
        c3 = self.layer2(c2)                                    # 512 @ H/8
        c4 = self.layer3(c3)                                    # 1024@ H/16
        c5 = self.layer4(c4)                                    # 2048@ H/32
        return c2, c3, c4, c5


def build_res2net50():
    return Res2Net(Bottle2neck, layers=(3, 4, 6, 3), baseWidth=26, scale=4)


# ===============================================================
#                        CBAM (Woo et al.)
#  - ChannelGate (avg+max → shared MLP) + SpatialGate (7x7 on pooled maps)
#  - We wrap in a tiny "ResidualCBAM" useful for precision bias
# ===============================================================

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg','max']):
        super().__init__()
        hidden = max(1, gate_channels // reduction_ratio)
        self.mlp = nn.Sequential(Flatten(), nn.Linear(gate_channels, hidden), nn.ReLU(), nn.Linear(hidden, gate_channels))
        self.pool_types = pool_types

    def forward(self, x):
        att = None
        for t in self.pool_types:
            if t == 'avg':
                y = F.adaptive_avg_pool2d(x, 1)
            else:
                y = F.adaptive_max_pool2d(x, 1)
            v = self.mlp(y)
            att = v if att is None else att + v
        scale = torch.sigmoid(att).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class BasicConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, s=1, p=0, relu=True, bn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=not bn)
        self.bn   = nn.BatchNorm2d(out_ch) if bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True) if relu else nn.Identity()
    def forward(self, x): return self.relu(self.bn(self.conv(x)))

class ChannelPool(nn.Module):
    def forward(self, x): return torch.cat([torch.max(x,1,keepdim=True)[0], torch.mean(x,1,keepdim=True)], dim=1)

class SpatialGate(nn.Module):
    def __init__(self): super().__init__(); self.compress=ChannelPool(); self.spatial=BasicConv(2, 1, 7, p=3, relu=False)
    def forward(self, x): return x * torch.sigmoid(self.spatial(self.compress(x)))

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, use_spatial=True):
        super().__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio)
        self.use_spatial = use_spatial
        self.SpatialGate = SpatialGate() if use_spatial else nn.Identity()
    def forward(self, x):
        x = self.ChannelGate(x)
        return self.SpatialGate(x)

class ResidualCBAM(nn.Module):
    """
    y = x + alpha * (CBAM(x) - x), with learnable alpha in (0,1)
    Helpful to bias for precision (SPE) without wrecking recall.
    """
    def __init__(self, channels, reduction=16, use_spatial=True, alpha_init=0.25):
        super().__init__()
        self.cbam = CBAM(channels, reduction_ratio=reduction, use_spatial=use_spatial)
        # logit parameterization to keep alpha in (0,1)
        self._alpha_logit = nn.Parameter(torch.log(torch.tensor(alpha_init/(1-alpha_init))))
    def forward(self, x):
        y = self.cbam(x)
        alpha = torch.sigmoid(self._alpha_logit)
        return x + alpha * (y - x)


# ===============================================================
#                             MSU
#  Multi-scale absolute difference + 3x3 compress + GN + ReLU
# ===============================================================

class MSU(nn.Module):
    def __init__(self, in_channels, out_channels=None, use_bn=True, activation=True):
        super().__init__()
        out_channels = out_channels or in_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=not use_bn)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_bn)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, bias=not use_bn)
        self.post3 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn  = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels) if use_bn else nn.Identity()
        self.act = nn.ReLU(inplace=True) if activation else nn.Identity()
        for m in [self.conv1, self.conv3, self.conv5, self.post3]:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, F_A, F_B):
        a1, b1 = self.conv1(F_A), self.conv1(F_B)
        a3, b3 = self.conv3(F_A), self.conv3(F_B)
        a5, b5 = self.conv5(F_A), self.conv5(F_B)
        out = torch.abs(a1-b1) + torch.abs(a3-b3) + torch.abs(a5-b5)
        out = self.post3(out)
        out = self.bn(out)
        return self.act(out)


class AlignMSU(nn.Module):
    """Project A,B to out_ch; resize B to A; MSU(A,B) at A's resolution."""
    def __init__(self, inA: int, inB: int, out_ch: int, use_bn=True, activation=True):
        super().__init__()
        self.projA = nn.Conv2d(inA, out_ch, 1, bias=True)
        self.projB = nn.Conv2d(inB, out_ch, 1, bias=True)
        self.msu   = MSU(out_ch, out_ch, use_bn=use_bn, activation=activation)

    def forward(self, A, B):
        A_ = self.projA(A)
        B_ = self.projB(B)
        if B_.shape[-2:] != A_.shape[-2:]:
            B_ = F.interpolate(B_, size=A_.shape[-2:], mode="bilinear", align_corners=False)
        return self.msu(A_, B_)


# ===============================================================
#                     HAS-Skip  (Eq. 2.30–2.35)
#   For level l: upsample E1..E4 → project to C_l →
#   softmax-weighted mix → decoder-aware gate → gate E_l
# ===============================================================

class HASSkip(nn.Module):
    """
    For each decoder level l, produce FSKIP_l:
      1) Upsample all encoder features to level size; project each to Cout_l.
      2) Aggregate with softmax scalars: Fagg = sum_i softmax(w_l)[i] * Ei_proj
      3) Gate: G_l = sigmoid( Wg_l(Fdec_l) + Wx_l(Fagg) )
      4) FSKIP_l = G_l * El_proj  (El is l's paired encoder: d1<-E4, d2<-E3, d3<-E2, d4<-E1)
    """
    def __init__(self,
                 Cin_list=(256, 512, 1024, 2048),
                 Cout_list=(1024, 512, 256, 128),
                 Cdec_list=(2048, 1024, 512, 256)):
        super().__init__()
        self.L = len(Cout_list)
        self.Cin_list  = Cin_list
        self.Cout_list = Cout_list
        self.Cdec_list = Cdec_list

        self.proj = nn.ModuleList()
        for l in range(self.L):
            Cout = Cout_list[l]
            self.proj.append(nn.ModuleList([
                nn.Conv2d(Cin_list[0], Cout, 1, bias=True),
                nn.Conv2d(Cin_list[1], Cout, 1, bias=True),
                nn.Conv2d(Cin_list[2], Cout, 1, bias=True),
                nn.Conv2d(Cin_list[3], Cout, 1, bias=True),
            ]))

        # level-wise mixture logits over 4 encoders
        self.w_logits = nn.ParameterList([nn.Parameter(torch.zeros(4)) for _ in range(self.L)])

        # gate projections per level (keep channels = Cout_l so G is per-channel spatial)
        self.Wg = nn.ModuleList([nn.Conv2d(Cdec_list[l], Cout_list[l], 1, bias=True) for l in range(self.L)])
        self.Wx = nn.ModuleList([nn.Conv2d(Cout_list[l], Cout_list[l], 1, bias=True) for l in range(self.L)])

    def forward_level(self, level_idx: int,
                      encs: List[torch.Tensor],
                      dec_ctx: torch.Tensor,
                      target_ref: torch.Tensor) -> torch.Tensor:
        l = level_idx
        Cout = self.Cout_list[l]

        # 1) align + project encoders to level size/Cout
        Ei_proj = []
        for i in range(4):
            x = _resize_like(encs[i], target_ref)
            Ei_proj.append(self.proj[l][i](x))   # (B, Cout, H_l, W_l)

        # 2) softmax mixing
        w = torch.softmax(self.w_logits[l], dim=0)
        Fagg = w[0]*Ei_proj[0] + w[1]*Ei_proj[1] + w[2]*Ei_proj[2] + w[3]*Ei_proj[3]

        # 3) decoder-aware gating
        dec_ctx = _resize_like(dec_ctx, target_ref)
        G = torch.sigmoid(self.Wg[l](dec_ctx) + self.Wx[l](Fagg))   # (B, Cout, H_l, W_l)

        # 4) pick this level's encoder stream: d1<-E4, d2<-E3, d3<-E2, d4<-E1
        El_proj = Ei_proj[3 - l]
        return G * El_proj


# ===============================================================
#                    Decoder blocks (flexible)
# ===============================================================

class DecoderBlockFlex(nn.Module):
    """ upconv(in_ch → out_ch) → cat(skip) → convs to out_ch """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x, skip):
        x = self.up(x)
        x = _resize_like(x, skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ===============================================================
#              Full model: Res2Net50 + MSU + CBAM + HAS
#   FMSU_l (vessel/edge bias)  +  FSKIP_l (semantic, recall)
#   Optional Residual-CBAM on fused skips to tilt precision
# ===============================================================

class Res2UNet_MSU_CBAM_HAS(nn.Module):
    """
    Encoder: Res2Net-50 → [C2,C3,C4,C5] = [256,512,1024,2048]
    MSU graph:
        A12 = MSU(C2,C3) @ C2 (256)
        A23 = MSU(C3,C4) @ C3 (512)
        A34 = MSU(C4,C5) @ C4 (1024)
        P1223 = MSU(A12,A23) @ C2 (256)
        P2334 = MSU(A23,A34) @ C3 (512)
        Qlast = MSU(P1223,P2334) @ C2 (256)
    Per-level FMSU:
        d1: MSU(C5, C5_bott) → 1024      (C5_bott = CBAM(C5) by default)
        d2: A34 (1024)
        d3: P2334 (512)
        d4: Qlast (256)
    HAS-Skip per level:
        FSKIP_d1..4 using decoder contexts [C5_bott, D1, D2, D3]
        (Cout for skips = [1024,512,256,128] to match decoder)
    Fusion:
        FB_l = FMSU_l + FSKIP_l
        (optionally Residual-CBAM on FB_l to raise precision/SPE safely)
    Decoder:
        D1: in=2048, skip=1024 → out=1024
        D2: in=1024, skip=512  → out=512
        D3: in=512,  skip=256  → out=256
        D4: in=256,  skip=128  → out=128
        Head: 1x1 → 1
    """
    def __init__(self,
                 in_channels: int = 3,
                 cbam_reduction: int = 16,
                 cbam_use_spatial_bottleneck: bool = True,
                 cbam_on_fused_skips: bool = True):
        super().__init__()

        # Encoder (Res2Net-50)
        self.backbone = build_res2net50()

        # If grayscale, prepend a 1→3 converter
        self.input_adapter = nn.Conv2d(in_channels, 3, 1, bias=False) if in_channels != 3 else nn.Identity()

        # Bottleneck CBAM (precision tilt)
        self.cbam_bott = ResidualCBAM(2048, reduction=cbam_reduction, use_spatial=cbam_use_spatial_bottleneck, alpha_init=0.25)

        # MSU graph
        self.msu_A12   = AlignMSU(256,  512,  256)
        self.msu_A23   = AlignMSU(512,  1024, 512)
        self.msu_A34   = AlignMSU(1024, 2048, 1024)
        self.msu_P1223 = AlignMSU(256,  512,  256)
        self.msu_P2334 = AlignMSU(512,  1024, 512)
        self.msu_Qlast = AlignMSU(256,  512,  256)
        # d1-specific MSU at top: MSU(C5, C5_bott) → 1024
        self.msu_top   = AlignMSU(2048, 2048, 1024)

        # HAS-Skip (match decoder skip widths)
        self.has = HASSkip(
            Cin_list=(256,512,1024,2048),
            Cout_list=(1024,512,256,128),
            Cdec_list=(2048,1024,512,256),
        )

        # Optional CBAM after fusion (Res-CBAM, conservative)
        if cbam_on_fused_skips:
            self.refine_d1 = ResidualCBAM(1024, reduction=cbam_reduction, use_spatial=False, alpha_init=0.25)
            self.refine_d2 = ResidualCBAM( 512, reduction=cbam_reduction, use_spatial=False, alpha_init=0.25)
            self.refine_d3 = ResidualCBAM( 256, reduction=cbam_reduction, use_spatial=False, alpha_init=0.25)
            self.refine_d4 = ResidualCBAM( 128, reduction=cbam_reduction, use_spatial=True,  alpha_init=0.25)
        else:
            self.refine_d1 = self.refine_d2 = self.refine_d3 = self.refine_d4 = nn.Identity()

        # Decoder
        self.d1 = DecoderBlockFlex(in_ch=2048, skip_ch=1024, out_ch=1024)  # H/32 → H/16
        self.d2 = DecoderBlockFlex(in_ch=1024, skip_ch= 512, out_ch= 512)  # H/16 → H/8
        self.d3 = DecoderBlockFlex(in_ch= 512, skip_ch= 256, out_ch= 256)  # H/8  → H/4
        self.d4 = DecoderBlockFlex(in_ch= 256, skip_ch= 128, out_ch= 128)  # H/4  → H/2

        # return to input size
        self.up_final = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)      # H/2 → H
        self.head     = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_adapter(x)

        # Encoder
        c2, c3, c4, c5 = self.backbone.forward_features(x)   # 256@H/4, 512@H/8, 1024@H/16, 2048@H/32

        # Bottleneck refinement (precision tilt)
        c5b = self.cbam_bott(c5)                              # 2048

        # MSU graph
        A12   = self.msu_A12(c2, c3)                          # 256 @ H/4
        A23   = self.msu_A23(c3, c4)                          # 512 @ H/8
        A34   = self.msu_A34(c4, c5)                          # 1024@ H/16
        P1223 = self.msu_P1223(A12, A23)                      # 256 @ H/4
        P2334 = self.msu_P2334(A23, A34)                      # 512 @ H/8
        Qlast = self.msu_Qlast(P1223, P2334)                  # 256 @ H/4

        # Per-level FMSU
        FMSU_d1 = self.msu_top(c5, c5b)                       # 1024 @ H/32
        FMSU_d2 = A34                                         # 1024 @ H/16
        FMSU_d3 = P2334                                       #  512 @ H/8
        FMSU_d4 = Qlast                                       #  256 @ H/4

        # HAS-Skip per level (decoder contexts are c5b, d1, d2, d3)
        FSKIP_d1 = self.has.forward_level(0, [c2,c3,c4,c5], c5b, c5)   # 1024 @ H/32

        # Fuse (sum) and optional CBAM refine → decode
        FB1 = self.refine_d1(FMSU_d1 + FSKIP_d1)
        d1  = self.d1(c5b, FB1)                                         # 1024 @ H/16

        FSKIP_d2 = self.has.forward_level(1, [c2,c3,c4,c5], d1, c4)     # 512 @ H/16
        FB2 = self.refine_d2(_resize_like(FMSU_d2, d1) + FSKIP_d2)
        d2  = self.d2(d1, FB2)                                          # 512 @ H/8

        FSKIP_d3 = self.has.forward_level(2, [c2,c3,c4,c5], d2, c3)     # 256 @ H/8
        FB3 = self.refine_d3(_resize_like(FMSU_d3, d2) + FSKIP_d3)
        d3  = self.d3(d2, FB3)                                          # 256 @ H/4

        FSKIP_d4 = self.has.forward_level(3, [c2,c3,c4,c5], d3, c2)     # 128 @ H/4
        FB4 = self.refine_d4(_resize_like(FMSU_d4, d3) + FSKIP_d4)
        d4  = self.d4(d3, FB4)                                          # 128 @ H/2

        out = self.up_final(d4)                                         # 64 @ H
        out = self.head(out)                                            # 1 @ H
        return out


# Handy factory
def res2unet_msu_cbam_has(in_channels: int = 1,
                          cbam_reduction: int = 16,
                          cbam_use_spatial_bottleneck: bool = True,
                          cbam_on_fused_skips: bool = True) -> nn.Module:
    return Res2UNet_MSU_CBAM_HAS(
        in_channels=in_channels,
        cbam_reduction=cbam_reduction,
        cbam_use_spatial_bottleneck=cbam_use_spatial_bottleneck,
        cbam_on_fused_skips=cbam_on_fused_skips
    )
