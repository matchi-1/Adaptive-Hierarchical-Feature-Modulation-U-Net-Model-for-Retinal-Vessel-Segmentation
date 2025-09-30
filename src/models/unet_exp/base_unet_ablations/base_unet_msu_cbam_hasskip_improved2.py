# src/models/unet_with_msu_cbam_sfg.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.unet import ConvBlock, EncoderBlock, DecoderBlock
from src.models.blocks.msu import MSU
from src.models.blocks.cbam import CBAM
import math


# ---------- helpers ----------
def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] != ref.shape[-2:]:
        x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
    return x


# ---------- MSU alignment ----------
class AlignMSU(nn.Module):
    """
    Align two feature maps to anchor channels/size, then apply MSU.
    A -> 1x1->C, B -> 1x1->C, resize B to A, MSU(C->C).
    """
    def __init__(self, inA: int, inB: int, out_ch: int, use_bn: bool = True, activation: bool = True):
        super().__init__()
        self.projA = nn.Conv2d(inA, out_ch, kernel_size=1, bias=True)
        self.projB = nn.Conv2d(inB, out_ch, kernel_size=1, bias=True)
        self.msu   = MSU(in_channels=out_ch, out_channels=out_ch, use_bn=use_bn, activation=activation)

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A_ = self.projA(A)
        B_ = self.projB(B)
        if B_.shape[-2:] != A_.shape[-2:]:
            B_ = F.interpolate(B_, size=A_.shape[-2:], mode="bilinear", align_corners=False)
        return self.msu(A_, B_)  # (N, out_ch, H_anchor, W_anchor)


# ---------- HAS-Skip (your original effective version) ----------
class HASSkip(nn.Module):
    """
    For each decoder level l, produce FSKIP_l:
      1) Upsample all encoders to level size; project to C_l.
      2) Aggregate with softmax scalars: Fagg = sum_i softmax(w_l)[i] * Ei_proj
      3) Gate current level encoder feature with decoder context:
            G_l = sigmoid( Wg_l(Fdecoder_l) + Wx_l(Fagg) )
         FSKIP_l = G_l * E_l_proj
    """
    def __init__(self,
                 Cin_list=(64,128,256,512),
                 Cout_list=(512,256,128,64),
                 Cdec_list=(1024,512,256,128)):
        super().__init__()
        self.L = 4
        self.Cin_list  = Cin_list
        self.Cout_list = Cout_list
        self.Cdec_list = Cdec_list

        self.proj = nn.ModuleList()
        for l in range(self.L):
            Cout = Cout_list[l]
            convs_l = nn.ModuleList([
                nn.Conv2d(Cin_list[0], Cout, 1, bias=True),
                nn.Conv2d(Cin_list[1], Cout, 1, bias=True),
                nn.Conv2d(Cin_list[2], Cout, 1, bias=True),
                nn.Conv2d(Cin_list[3], Cout, 1, bias=True),
            ])
            self.proj.append(convs_l)

        self.w_logits = nn.ParameterList([nn.Parameter(torch.zeros(4)) for _ in range(self.L)])
        self.Wg = nn.ModuleList([nn.Conv2d(Cdec_list[l], Cout_list[l], 1, bias=True) for l in range(self.L)])
        self.Wx = nn.ModuleList([nn.Conv2d(Cout_list[l], Cout_list[l], 1, bias=True) for l in range(self.L)])

    def forward_level(self, level_idx: int,
                      encs: list[torch.Tensor],
                      dec_ctx: torch.Tensor,
                      target_ref: torch.Tensor) -> torch.Tensor:
        l = level_idx
        Cout = self.Cout_list[l]

        Ei_proj = []
        for i in range(4):
            x = _resize_like(encs[i], target_ref)
            Ei_proj.append(self.proj[l][i](x))  # -> (B, Cout, H_l, W_l)

        w = torch.softmax(self.w_logits[l], dim=0)
        Fagg = w[0]*Ei_proj[0] + w[1]*Ei_proj[1] + w[2]*Ei_proj[2] + w[3]*Ei_proj[3]

        dec_ctx = _resize_like(dec_ctx, target_ref)
        G = torch.sigmoid(self.Wg[l](dec_ctx) + self.Wx[l](Fagg))  # (B, Cout, H_l, W_l)

        # d1->E4, d2->E3, d3->E2, d4->E1
        El_proj = Ei_proj[3 - l]
        return G * El_proj


# ---------- Residual CBAM (safe precision filter) ----------
class ResidualCBAM(nn.Module):
    """
    y = x + alpha * (CBAM(x) - x), alpha in (0,1), learned.
    """
    def __init__(self, channels, reduction=16, use_spatial=True, alpha_init=0.2):
        super().__init__()
        self.cbam = CBAM(channels, reduction_ratio=reduction, use_spatial=use_spatial)
        self._raw_alpha = nn.Parameter(torch.log(torch.tensor(alpha_init/(1-alpha_init))))  # logit

    def forward(self, x):
        y = self.cbam(x)
        alpha = torch.sigmoid(self._raw_alpha)
        return x + alpha * (y - x)


# ---------- Selective Fusion Gate (SFG): dynamic MSU/HAS/CBAM mixing ----------
class SelectiveFusionGate(nn.Module):
    """
    Input:  f_msu (B,C,H,W), f_has (B,C,H,W)
    Steps:
      - f_cbam = ResidualCBAM(f_has)  (channel-only or ch+spatial)
      - GAP -> per-branch descriptors (B,C)
      - Three tiny heads -> logits for {msu, has, cbam}; add level-dependent priors; optional edge boost on msu
      - softmax( logits / tau ) -> weights w_msu, w_has, w_cbam
      - Fuse: FB = w_msu * f_msu + w_has * f_has + w_cbam * f_cbam
    """
    def __init__(self, channels: int,
                 reduction: int = 16,
                 use_spatial_cbam: bool = False,
                 tau: float = 1.5,
                 prior_logits=(0.0, 0.0, 0.0),
                 edge_boost_gain: float = 0.0):
        super().__init__()
        self.rcbam = ResidualCBAM(channels, reduction=reduction, use_spatial=use_spatial_cbam, alpha_init=0.25)
        self.tau = float(tau)
        self.edge_boost_gain = float(edge_boost_gain)

        self.head_msu = nn.Linear(channels, 1, bias=True)
        self.head_has = nn.Linear(channels, 1, bias=True)
        self.head_cbm = nn.Linear(channels, 1, bias=True)

        # initialize biases to priors (favor certain branches per level)
        with torch.no_grad():
            self.head_msu.bias.fill_(float(prior_logits[0]))
            self.head_has.bias.fill_(float(prior_logits[1]))
            self.head_cbm.bias.fill_(float(prior_logits[2]))

        # simple Kaiming init for weights
        for m in [self.head_msu, self.head_has, self.head_cbm]:
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))

    def _gap(self, x):  # (B,C,H,W) -> (B,C)
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

    def forward(self, f_msu, f_has):
        f_cbm = self.rcbam(f_has)

        # descriptors
        d_msu = self._gap(f_msu)
        d_has = self._gap(f_has)
        d_cbm = self._gap(f_cbm)

        # logits
        logit_m = self.head_msu(d_msu)  # (B,1)
        logit_h = self.head_has(d_has)
        logit_c = self.head_cbm(d_cbm)

        # optional edge boost for msu: add k * mean(|f_msu|)
        if self.edge_boost_gain > 0.0:
            edge = f_msu.abs().mean(dim=(1,2,3), keepdim=True)  # (B,1,1,1)
            logit_m = logit_m + self.edge_boost_gain * edge.flatten(1)  # (B,1)

        logits = torch.cat([logit_m, logit_h, logit_c], dim=1)  # (B,3)
        weights = F.softmax(logits / self.tau, dim=1)           # (B,3)

        # reshape to broadcast
        wm, wh, wc = [w.view(-1,1,1,1) for w in weights.split(1, dim=1)]
        return wm * f_msu + wh * f_has + wc * f_cbm


# ---------- Full model with SFG ----------
class UNetWithMSU_HASSkip_CBAM_SFG(nn.Module):
    """
    Start from your best-performing “original fusing” variant and replace
    FB_l = FMSU_l + FSKIP_l
    with
    FB_l = SFG_l( FMSU_l, FSKIP_l )       # dynamic MSU/HAS/CBAM mix per level
    """
    def __init__(self,
                 in_channels: int = 1,
                 cbam_reduction: int = 16):
        super().__init__()

        # base UNet
        self.e1 = EncoderBlock(in_channels, 64)
        self.e2 = EncoderBlock(64,   128)
        self.e3 = EncoderBlock(128,  256)
        self.e4 = EncoderBlock(256,  512)
        self.bottleneck = ConvBlock(512, 1024)

        self.d1 = DecoderBlock(1024, 512)  # expects 512
        self.d2 = DecoderBlock(512,  256)  # expects 256
        self.d3 = DecoderBlock(256,  128)  # expects 128
        self.d4 = DecoderBlock(128,   64)  # expects  64
        self.final = nn.Conv2d(64, 1, kernel_size=1)

        # MSU graph
        self.msu_A12   = AlignMSU(inA=64,  inB=128, out_ch=64)
        self.msu_A23   = AlignMSU(inA=128, inB=256, out_ch=128)
        self.msu_A34   = AlignMSU(inA=256, inB=512, out_ch=256)
        self.msu_P1223 = AlignMSU(inA=64,  inB=128, out_ch=64)
        self.msu_P2334 = AlignMSU(inA=128, inB=256, out_ch=128)
        self.msu_Qlast = AlignMSU(inA=64,  inB=128, out_ch=64)
        self.msu_top   = AlignMSU(inA=512, inB=1024, out_ch=512)  # MSU(s4, b) for d1

        # HAS-Skip
        self.has = HASSkip(
            Cin_list=(64,128,256,512),
            Cout_list=(512,256,128,64),
            Cdec_list=(1024,512,256,128),
        )

        # Selective Fusion Gates (per level)
        # priors tilt coarse levels toward MSU, finest toward CBAM
        self.sfg_d1 = SelectiveFusionGate(512, reduction=cbam_reduction,
                                          use_spatial_cbam=False, tau=1.8,
                                          prior_logits=(+0.8, +0.2, -0.2),
                                          edge_boost_gain=0.4)
        self.sfg_d2 = SelectiveFusionGate(256, reduction=cbam_reduction,
                                          use_spatial_cbam=False, tau=1.6,
                                          prior_logits=(+0.4, +0.2,  0.0),
                                          edge_boost_gain=0.3)
        self.sfg_d3 = SelectiveFusionGate(128, reduction=cbam_reduction,
                                          use_spatial_cbam=False, tau=1.4,
                                          prior_logits=(+0.1, +0.3, +0.1),
                                          edge_boost_gain=0.2)
        self.sfg_d4 = SelectiveFusionGate( 64, reduction=cbam_reduction,
                                          use_spatial_cbam=True,  tau=1.2,
                                          prior_logits=(-0.2, +0.2, +0.6),
                                          edge_boost_gain=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encoders
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        # bottleneck
        b = self.bottleneck(p4)

        # MSU graph
        A12   = self.msu_A12(s1, s2)
        A23   = self.msu_A23(s2, s3)
        A34   = self.msu_A34(s3, s4)
        P1223 = self.msu_P1223(A12, A23)
        P2334 = self.msu_P2334(A23, A34)
        Qlast = self.msu_Qlast(P1223, P2334)

        FMSU_d1 = self.msu_top(s4, b)   # (512)
        FMSU_d2 = A34                   # (256)
        FMSU_d3 = P2334                 # (128)
        FMSU_d4 = Qlast                 # ( 64)

        # HAS-Skip (decoder contexts: b,d1,d2,d3)
        FSKIP_d1 = self.has.forward_level(0, [s1,s2,s3,s4], b,  s4)

        # SFG fusion + decode level by level
        FB1 = self.sfg_d1(FMSU_d1, FSKIP_d1)
        d1  = self.d1(b, FB1)                # -> 512

        FSKIP_d2 = self.has.forward_level(1, [s1,s2,s3,s4], d1, s3)
        FB2 = self.sfg_d2(FMSU_d2, FSKIP_d2)
        d2  = self.d2(d1, FB2)               # -> 256

        FSKIP_d3 = self.has.forward_level(2, [s1,s2,s3,s4], d2, s2)
        FB3 = self.sfg_d3(FMSU_d3, FSKIP_d3)
        d3  = self.d3(d2, FB3)               # -> 128

        FSKIP_d4 = self.has.forward_level(3, [s1,s2,s3,s4], d3, s1)
        FB4 = self.sfg_d4(FMSU_d4, FSKIP_d4)
        d4  = self.d4(d3, FB4)               # -> 64

        return self.final(d4)
