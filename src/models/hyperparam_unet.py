# unet.py (hyperparameterized)
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Utilities / Factories
# -----------------------------
def _act(name: str):
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(0.01, inplace=True)
    if name == "elu":
        return nn.ELU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "mish":
        return nn.Mish()
    raise ValueError(f"Unknown activation: {name}")

def _norm(name: str, num_features: int, gn_groups: int = 32):
    name = name.lower()
    if name == "bn":
        return nn.BatchNorm2d(num_features)
    if name == "in":
        return nn.InstanceNorm2d(num_features, affine=True, track_running_stats=True)
    if name == "gn":
        # Ensure groups divide channels
        g = max(1, min(gn_groups, num_features))
        while num_features % g != 0 and g > 1:
            g -= 1
        return nn.GroupNorm(g, num_features)
    if name in ("none", "identity", "id"):
        return nn.Identity()
    raise ValueError(f"Unknown norm: {name}")

def _init_weights(m: nn.Module, scheme: str = "kaiming"):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        if scheme == "kaiming":
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        elif scheme == "xavier":
            nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.ones_(m.weight)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.zeros_(m.bias)

# -----------------------------
# Config
# -----------------------------
@dataclass
class UNetConfig:
    in_channels: int = 1
    out_channels: int = 1
    depth: int = 4                          # encoder/decoder levels
    base_channels: int = 64                 # first level channels
    width_mult: int = 2                     # channel doubling each downsample
    channels: Optional[Sequence[int]] = None  # override channel schedule, e.g., [64,128,256,512]
    convs_per_block: int = 2               # convs inside ConvBlock
    kernel_size: int = 3
    norm: Literal["bn", "in", "gn", "none"] = "bn"
    gn_groups: int = 32
    activation: Literal["relu", "leaky_relu", "elu", "gelu", "mish"] = "relu"
    pool: Literal["max", "avg", "conv"] = "max"  # "conv" = stride-2 conv downsample
    up_mode: Literal["convtranspose", "bilinear", "nearest"] = "convtranspose"
    align_corners: bool = False             # for bilinear upsample
    dropout: float = 0.0                    # dropout inside ConvBlock
    final_activation: Optional[Literal["sigmoid", "softmax"]] = None
    # NOTE: keep final_activation=None when training with DiceBCEComplementLoss,
    # because that loss expects logits (not probabilities).
    init: Literal["kaiming", "xavier", "none"] = "kaiming"

# -----------------------------
# Blocks
# -----------------------------
class ConvBlock(nn.Module):
    """
    Flexible ConvBlock: (Conv2d -> Norm -> Act -> Dropout) × N
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        convs: int = 2,
        ks: int = 3,
        norm: str = "bn",
        act: str = "relu",
        dropout: float = 0.0,
        gn_groups: int = 32,
    ):
        super().__init__()
        pad = ks // 2
        layers: List[nn.Module] = []
        c_in = in_ch
        for i in range(convs):
            layers.append(nn.Conv2d(c_in, out_ch, kernel_size=ks, padding=pad, bias=True))
            layers.append(_norm(norm, out_ch, gn_groups))
            layers.append(_act(act))
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            c_in = out_ch
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class EncoderBlock(nn.Module):
    """
    Encoder step: ConvBlock -> Downsample
    downsample by: "max", "avg", or a stride-2 Conv2d when pool="conv".
    Returns (skip, downsampled)
    """
    def __init__(
        self, in_ch: int, out_ch: int, cfg: UNetConfig
    ):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch,
                              convs=cfg.convs_per_block,
                              ks=cfg.kernel_size,
                              norm=cfg.norm,
                              act=cfg.activation,
                              dropout=cfg.dropout,
                              gn_groups=cfg.gn_groups)
        if cfg.pool == "max":
            self.down = nn.MaxPool2d(2)
        elif cfg.pool == "avg":
            self.down = nn.AvgPool2d(2)
        elif cfg.pool == "conv":
            self.down = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)
        else:
            raise ValueError(f"Unknown pool: {cfg.pool}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        p = self.down(x)
        return x, p

class DecoderBlock(nn.Module):
    """
    2× upsample + concat skip + ConvBlock
    upsample by: ConvTranspose2d, bilinear, or nearest.
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, cfg: UNetConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.up_mode == "convtranspose":
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            reduce_in = out_ch + skip_ch
        else:
            self.up = None
            # reduce channels before concat when using interpolation
            self.reduce = nn.Conv2d(in_ch, out_ch, kernel_size=1)
            reduce_in = out_ch + skip_ch

        self.fuse = ConvBlock(reduce_in, out_ch,
                              convs=cfg.convs_per_block,
                              ks=cfg.kernel_size,
                              norm=cfg.norm,
                              act=cfg.activation,
                              dropout=cfg.dropout,
                              gn_groups=cfg.gn_groups)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if self.cfg.up_mode == "convtranspose":
            x = self.up(x)
        else:
            mode = "bilinear" if self.cfg.up_mode == "bilinear" else "nearest"
            x = F.interpolate(x, scale_factor=2, mode=mode,
                              align_corners=self.cfg.align_corners if mode == "bilinear" else None)
            x = self.reduce(x)

        # fix any off-by-one shape drift
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)

# -----------------------------
# UNet
# -----------------------------
class UNet(nn.Module):
    """
    Hyperparameterized UNet.

    Default = classic 4-level UNet:
      channels: 64,128,256,512; bottleneck=1024; ConvTranspose up; BN+ReLU
    """
    def __init__(self, cfg: UNetConfig):
        super().__init__()
        self.cfg = cfg

        # channel schedule
        if cfg.channels is not None:
            chs = list(cfg.channels)
            assert len(chs) == cfg.depth, "len(channels) must equal depth"
        else:
            chs = [cfg.base_channels * (cfg.width_mult ** i) for i in range(cfg.depth)]
        self.enc_channels = chs
        bottleneck_ch = chs[-1] * cfg.width_mult

        # encoder
        enc: List[nn.Module] = []
        in_ch = cfg.in_channels
        for out_ch in chs:
            enc.append(EncoderBlock(in_ch, out_ch, cfg))
            in_ch = out_ch
        self.encoder = nn.ModuleList(enc)

        # bottleneck
        self.bottleneck = ConvBlock(chs[-1], bottleneck_ch,
                                    convs=cfg.convs_per_block,
                                    ks=cfg.kernel_size,
                                    norm=cfg.norm,
                                    act=cfg.activation,
                                    dropout=cfg.dropout,
                                    gn_groups=cfg.gn_groups)

        # decoder
        dec: List[nn.Module] = []
        # iterate reversed over encoder levels for skip connections
        in_ch = bottleneck_ch
        for skip_ch, out_ch in zip(reversed(chs), reversed(chs)):
            dec.append(DecoderBlock(in_ch, skip_ch, out_ch, cfg))
            in_ch = out_ch
        self.decoder = nn.ModuleList(dec)

        # head
        self.head = nn.Conv2d(chs[0], cfg.out_channels, kernel_size=1)

        # weights
        if cfg.init != "none":
            self.apply(lambda m: _init_weights(m, cfg.init))

        # final activation layer (optional; keep None for losses on logits)
        if cfg.final_activation is None:
            self.final_act = None
        elif cfg.final_activation == "sigmoid":
            self.final_act = nn.Sigmoid()
        elif cfg.final_activation == "softmax":
            self.final_act = nn.Softmax(dim=1)
        else:
            raise ValueError(f"Unknown final_activation: {cfg.final_activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: List[torch.Tensor] = []
        for enc in self.encoder:
            s, x = enc(x)
            skips.append(s)

        x = self.bottleneck(x)

        for dec, skip in zip(self.decoder, reversed(skips)):
            x = dec(x, skip)

        x = self.head(x)
        if self.final_act is not None:
            x = self.final_act(x)
        return x

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_unet(**kwargs) -> UNet:
    """
    Convenience builder:
    build_unet(in_channels=1, out_channels=1, depth=4, base_channels=64, ...)
    """
    cfg = UNetConfig(**kwargs)
    return UNet(cfg)

"""
MINIMAL USAGE:

from unet import build_unet
model = build_unet(
    in_channels=1,
    out_channels=1,
    depth=4,
    base_channels=64,
    width_mult=2,
    norm="bn",
    activation="relu",
    pool="max",
    up_mode="convtranspose",
    final_activation=None  # keep None when using DiceBCEComplementLoss (expects logits)
).to(device)


"""

