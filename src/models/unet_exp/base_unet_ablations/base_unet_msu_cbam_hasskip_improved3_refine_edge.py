import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.unet import ConvBlock, EncoderBlock, DecoderBlock
from src.models.blocks.msu import MSU
from src.models.blocks.cbam import CBAM


import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.unet_exp.base_unet_ablations.base_unet_msu_cbam_hasskip_improved3 import (
    HASSkip, AdaptiveSelectiveFusionGate
)
from src.models.unet import DecoderBlock, ConvBlock


class FinalRefine(nn.Module):
    """
    Predicts skeleton & edge from decoder feat, then uses them to refine feat
    before the final logits. Returns (seg_logits, edge_logits, skel_logits).
    """
    def __init__(self, in_ch: int, *, use_spatial_cbam: bool = True):
        super().__init__()
        self.skel_head = nn.Conv2d(in_ch, 1, kernel_size=1)   # centerline prior
        self.edge_head = nn.Conv2d(in_ch, 1, kernel_size=1)   # boundary prior

        # optional tiny CBAM to precision-filter fused feat
        self.cbam = CBAM(in_ch, reduction_ratio=16, use_spatial=use_spatial_cbam)

        # learnable gates that control how strongly aux maps steer features
        self._alpha_skel = nn.Parameter(torch.tensor(0.75))   # boosts along centerlines
        self._alpha_edge = nn.Parameter(torch.tensor(0.60))   # suppresses off-edge bleed

        # fuse (feat + 2 aux channels) → refine
        self.fuse = nn.Conv2d(in_ch + 2, in_ch, kernel_size=3, padding=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=min(32, in_ch), num_channels=in_ch)
        self.act  = nn.ReLU(inplace=True)

        # final segmentation head
        self.out = nn.Conv2d(in_ch, 1, kernel_size=1)

        # optional temperature head (logit calibration -> sharper boundaries)
        self.temp = nn.Conv2d(in_ch, 1, kernel_size=1)

    def forward(self, feat: torch.Tensor):
        skel_logit = self.skel_head(feat)
        edge_logit = self.edge_head(feat)
        skel = torch.sigmoid(skel_logit)
        edge = torch.sigmoid(edge_logit)

        # centerline-attentive sharpening (thins vessels)
        alpha_s = torch.sigmoid(self._alpha_skel)   # (0,1)
        alpha_e = torch.sigmoid(self._alpha_edge)   # (0,1)
        guided  = feat * (1.0 + alpha_s * skel) * (1.0 - 0.5 * alpha_e * edge)

        # concatenate aux priors and re-filter
        fused = torch.cat([guided, skel, edge], dim=1)
        fused = self.fuse(fused)
        fused = self.norm(fused)
        fused = self.act(self.cbam(fused))

        seg_logit = self.out(fused)

        # per-pixel temperature (keeps confidence in check near edges)
        scale = 1.0 + F.softplus(self.temp(fused)) * (1.0 - edge)
        seg_logit = seg_logit / scale.clamp_min(1e-3)

        return seg_logit, edge_logit, skel_logit


# ----------------- utils -----------------
def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] != ref.shape[-2:]:
        x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
    return x


# ----------------- MSU alignment -----------------
class AlignMSU(nn.Module):
    """A->1x1->C, B->1x1->C, resize B to A, MSU(C->C)."""
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
        return self.msu(A_, B_)  # (B, C, H, W)


# ----------------- HAS-Skip (your effective version) -----------------
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


# ----------------- Residual CBAM (safe precision filter) -----------------
class ResidualCBAM(nn.Module):
    """y = x + alpha * (CBAM(x) - x),  alpha learned in (0,1)."""
    def __init__(self, channels, reduction=16, use_spatial=True, alpha_init=0.25):
        super().__init__()
        self.cbam = CBAM(channels, reduction_ratio=reduction, use_spatial=use_spatial)
        # logit so alpha = sigmoid(logit) in (0,1)
        self._alpha = nn.Parameter(torch.log(torch.tensor(alpha_init/(1.0 - alpha_init))))

    def forward(self, x):
        alpha = torch.sigmoid(self._alpha)
        y = self.cbam(x)
        return x + alpha * (y - x)


# ----------------- Adaptive Selective Fusion Gate (ASFG) -----------------
class AdaptiveSelectiveFusionGate(nn.Module):
    """
    Mix {MSU, HAS, CBAM(HAS)} with learned, content-adaptive weights:
      - f_cbam = ResidualCBAM(f_has)
      - descriptors = GAP(|f|) -> small heads -> logits
      - + edge boost for MSU: mean Sobel magnitude on f_msu
      - + agreement boost for CBAM: cosine agreement among branches
      - softmax(logits / tau) -> weights -> fused FB

    Notes:
      • edge favors MSU/HAS (thin vessels → recall/clDice)
      • agreement favors CBAM (precision/SPE)
      • CBAM spatial is only used where requested (typically finest level)
    """
    def __init__(self, channels: int,
                 reduction: int = 16,
                 use_spatial_cbam: bool = False,
                 tau: float = 1.4,
                 prior_logits=(0.0, 0.0, 0.0),
                 edge_boost_gain: float = 0.3,
                 agree_boost_gain: float = 0.4):
        super().__init__()
        self.rcbam = ResidualCBAM(channels, reduction=reduction,
                                  use_spatial=use_spatial_cbam, alpha_init=0.25)
        self.tau = float(tau)
        # learnable positive gains via softplus
        self._edge_g  = nn.Parameter(torch.tensor(edge_boost_gain))
        self._agree_g = nn.Parameter(torch.tensor(agree_boost_gain))

        self.head_m = nn.Linear(channels, 1, bias=True)
        self.head_h = nn.Linear(channels, 1, bias=True)
        self.head_c = nn.Linear(channels, 1, bias=True)

        with torch.no_grad():
            self.head_m.bias.fill_(float(prior_logits[0]))
            self.head_h.bias.fill_(float(prior_logits[1]))
            self.head_c.bias.fill_(float(prior_logits[2]))
        nn.init.kaiming_uniform_(self.head_m.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.head_h.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.head_c.weight, a=math.sqrt(5))

        # fixed Sobel kernels (depthwise)
        kx = torch.tensor([[1, 0, -1],
                           [2, 0, -2],
                           [1, 0, -1]], dtype=torch.float32).view(1,1,3,3)
        ky = torch.tensor([[ 1,  2,  1],
                           [ 0,  0,  0],
                           [-1, -2, -1]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer("sobel_kx", kx)
        self.register_buffer("sobel_ky", ky)

    @staticmethod
    def _gap_abs(x):  # (B,C,H,W) -> (B,C)
        return F.adaptive_avg_pool2d(x.abs(), 1).flatten(1)

    def _edge_score(self, x):  # (B,C,H,W) -> (B,1)
        B, C, H, W = x.shape
        kx = self.sobel_kx.expand(C, 1, 3, 3)
        ky = self.sobel_ky.expand(C, 1, 3, 3)
        gx = F.conv2d(x, kx, padding=1, groups=C)
        gy = F.conv2d(x, ky, padding=1, groups=C)
        mag = torch.sqrt(gx*gx + gy*gy + 1e-6)       # (B,C,H,W)
        return mag.mean(dim=(1,2,3), keepdim=True)    # (B,1,1,1) broadcastable

    @staticmethod
    def _cos_agree(u, v):  # (B,C) x (B,C) -> (B,1) in [0,1]
        u = F.normalize(u, dim=1, eps=1e-6)
        v = F.normalize(v, dim=1, eps=1e-6)
        sim = (u * v).sum(1, keepdim=True)           # [-1,1]
        return 0.5 * (sim + 1.0)

    def forward(self, f_msu, f_has):
        # third branch: residual CBAM on HAS
        f_cbm = self.rcbam(f_has)

        # descriptors (global, cheap)
        dm = self._gap_abs(f_msu)    # (B,C)
        dh = self._gap_abs(f_has)    # (B,C)
        dc = self._gap_abs(f_cbm)    # (B,C)

        # base logits
        lm = self.head_m(dm)         # (B,1)
        lh = self.head_h(dh)
        lc = self.head_c(dc)

        # edge/agree cues
        edge = self._edge_score(f_msu)                    # (B,1,1,1)
        amc  = self._cos_agree(dm, dc)                   # (B,1)
        amh  = self._cos_agree(dm, dh)
        ahc  = self._cos_agree(dh, dc)
        agree = (amc + amh + ahc) / 3.0                  # (B,1)

        edge_g  = F.softplus(self._edge_g)               # >= 0
        agree_g = F.softplus(self._agree_g)              # >= 0

        # steer logits with cues
        lm = lm + edge_g * edge.flatten(1) + agree_g * (1.0 - agree)  # edges & disagreement push MSU
        lh = lh + agree_g * (1.0 - agree)                              # disagreement pushes HAS
        lc = lc + agree_g * agree - edge_g * edge.flatten(1)           # agreement favors CBAM, edges suppress CBAM

        logits  = torch.cat([lm, lh, lc], dim=1)         # (B,3)
        weights = F.softmax(logits / self.tau, dim=1)    # (B,3)

        wm, wh, wc = [w.view(-1,1,1,1) for w in weights.split(1, dim=1)]
        return wm * f_msu + wh * f_has + wc * f_cbm      # fused FB


# ----------------- Full model -----------------
class UNetWithMSU_HASSkip_CBAM_ASFG(nn.Module):
    """
    Same MSU graph + HAS-Skip as your improved variant, but replaces
    FB_l = FMSU_l + FSKIP_l   with
    FB_l = ASFG_l( FMSU_l, FSKIP_l )
    where ASFG learns per-image, per-level mixing of {MSU, HAS, CBAM(HAS)}
    using edge- and agreement-aware gating.
    """
    def __init__(self, in_channels: int = 1, cbam_reduction: int = 16):
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
        self.msu_top   = AlignMSU(inA=512, inB=1024, out_ch=512)  # MSU(s4,b) for d1

        # HAS-Skip
        self.has = HASSkip(
            Cin_list=(64,128,256,512),
            Cout_list=(512,256,128,64),
            Cdec_list=(1024,512,256,128),
        )

        # Adaptive selective fusion gates per level
        # Coarse → favor MSU; fine → allow CBAM spatial to clean FPs
        self.asfg_d1 = AdaptiveSelectiveFusionGate(
            channels=512, reduction=cbam_reduction, use_spatial_cbam=False,
            tau=1.8, prior_logits=(+0.7, +0.2, -0.3), edge_boost_gain=0.5, agree_boost_gain=0.3
        )
        self.asfg_d2 = AdaptiveSelectiveFusionGate(
            channels=256, reduction=cbam_reduction, use_spatial_cbam=False,
            tau=1.6, prior_logits=(+0.4, +0.2,  0.0), edge_boost_gain=0.4, agree_boost_gain=0.3
        )
        self.asfg_d3 = AdaptiveSelectiveFusionGate(
            channels=128, reduction=cbam_reduction, use_spatial_cbam=False,
            tau=1.4, prior_logits=(+0.1, +0.3, +0.1), edge_boost_gain=0.3, agree_boost_gain=0.35
        )
        self.asfg_d4 = AdaptiveSelectiveFusionGate(
            channels=64,  reduction=cbam_reduction, use_spatial_cbam=True,   # only finest uses spatial CBAM
            tau=1.2, prior_logits=(-0.2, +0.2, +0.7), edge_boost_gain=0.0, agree_boost_gain=0.5
        )

        self.refine = FinalRefine(in_ch=64, use_spatial_cbam=True)

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

        FMSU_d1 = self.msu_top(s4, b)  # (512)
        FMSU_d2 = A34                  # (256)
        FMSU_d3 = P2334                # (128)
        FMSU_d4 = Qlast                # ( 64)

        # HAS-Skip (contexts: b, d1, d2, d3)
        FSKIP_d1 = self.has.forward_level(0, [s1,s2,s3,s4], b,  s4)
        FB1 = self.asfg_d1(FMSU_d1, FSKIP_d1)
        d1  = self.d1(b, FB1)

        FSKIP_d2 = self.has.forward_level(1, [s1,s2,s3,s4], d1, s3)
        FB2 = self.asfg_d2(FMSU_d2, FSKIP_d2)
        d2  = self.d2(d1, FB2)

        FSKIP_d3 = self.has.forward_level(2, [s1,s2,s3,s4], d2, s2)
        FB3 = self.asfg_d3(FMSU_d3, FSKIP_d3)
        d3  = self.d3(d2, FB3)

        FSKIP_d4 = self.has.forward_level(3, [s1,s2,s3,s4], d3, s1)
        FB4 = self.asfg_d4(FMSU_d4, FSKIP_d4)
        d4  = self.d4(d3, FB4)

        # --- refine edge + skeleton ---
        logits_refined, edge_logits, skel_logits = self.refine(d4)

        # safety: if input size isn’t exactly divisible by 16, upsample back
        if logits_refined.shape[-2:] != x.shape[-2:]:
            logits_refined = F.interpolate(logits_refined, size=x.shape[-2:], mode="bilinear", align_corners=False)
            edge_logits    = F.interpolate(edge_logits,    size=x.shape[-2:], mode="bilinear", align_corners=False)
            skel_logits    = F.interpolate(skel_logits,    size=x.shape[-2:], mode="bilinear", align_corners=False)

        return {
            "logits": logits_refined,   # main seg logits
            "edge_logits": edge_logits, # boundary prior
            "skel_logits": skel_logits  # centerline prior
        }
