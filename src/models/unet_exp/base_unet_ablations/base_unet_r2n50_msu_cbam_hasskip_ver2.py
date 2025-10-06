# file: (same module where your UNetWithMSU_HASSkip_CBAM_ASFG lives)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.models.blocks.msu import MSU
from src.models.blocks.cbam import CBAM
from src.models.unet import ConvBlock  # only for the final head's small convs if you prefer; safe to keep
from src.models.unet_exp.base_unet_ablations.base_unet_r2n50_msu_cbam_hasskip import build_res2net50  # <- Res2Net-50 factory



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


# ----------------- HAS-Skip -----------------
class HASSkip(nn.Module):
    """
    For each decoder level l, produce FSKIP_l:
      1) Upsample all encoders to level size; project to C_l.
      2) Aggregate with softmax scalars: Fagg = sum_i softmax(w_l)[i] * Ei_proj
      3) Gate current level encoder feature with decoder context:
            G_l = sigmoid( Wg_l(Fdecoder_l) + Wx_l(Fagg) )
         FSKIP_l = G_l * E_l_proj

    Here we target Res2Net feature widths:
      Cin_list = [256, 512, 1024, 2048]
      Cout_list (skip widths) = [1024, 512, 256, 128] to match the decoder
      Cdec_list (decoder ctx) = [2048, 1024, 512, 256]
    """
    def __init__(self,
                 Cin_list=(256,512,1024,2048),
                 Cout_list=(1024,512,256,128),
                 Cdec_list=(2048,1024,512,256)):
        super().__init__()
        self.L = 4
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

        self.w_logits = nn.ParameterList([nn.Parameter(torch.zeros(4)) for _ in range(self.L)])
        self.Wg = nn.ModuleList([nn.Conv2d(Cdec_list[l], Cout_list[l], 1, bias=True) for l in range(self.L)])
        self.Wx = nn.ModuleList([nn.Conv2d(Cout_list[l], Cout_list[l], 1, bias=True) for l in range(self.L)])

    def forward_level(self, level_idx: int,
                      encs: list[torch.Tensor],
                      dec_ctx: torch.Tensor,
                      target_ref: torch.Tensor) -> torch.Tensor:
        l = level_idx
        Ei_proj = []
        for i in range(4):
            x = _resize_like(encs[i], target_ref)
            Ei_proj.append(self.proj[l][i](x))  # -> (B, Cout_l, H_l, W_l)

        w = torch.softmax(self.w_logits[l], dim=0)
        Fagg = w[0]*Ei_proj[0] + w[1]*Ei_proj[1] + w[2]*Ei_proj[2] + w[3]*Ei_proj[3]

        dec_ctx = _resize_like(dec_ctx, target_ref)
        G = torch.sigmoid(self.Wg[l](dec_ctx) + self.Wx[l](Fagg))  # (B, Cout_l, H_l, W_l)

        # d1<-E4, d2<-E3, d3<-E2, d4<-E1
        El_proj = Ei_proj[3 - l]
        return G * El_proj


# ----------------- Residual CBAM -----------------
class ResidualCBAM(nn.Module):
    """y = x + alpha * (CBAM(x) - x),  alpha learned in (0,1)."""
    def __init__(self, channels, reduction=16, use_spatial=True, alpha_init=0.25):
        super().__init__()
        self.cbam = CBAM(channels, reduction_ratio=reduction, use_spatial=use_spatial)
        self._alpha = nn.Parameter(torch.log(torch.tensor(alpha_init/(1.0 - alpha_init))))

    def forward(self, x):
        alpha = torch.sigmoid(self._alpha)
        y = self.cbam(x)
        return x + alpha * (y - x)


# ----------------- ASFG -----------------
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
        if f_msu.shape[-2:] != f_has.shape[-2:]:
            f_msu = F.interpolate(f_msu, size=f_has.shape[-2:], mode="bilinear", align_corners=False)

        f_cbm = self.rcbam(f_has)
        if f_cbm.shape[-2:] != f_has.shape[-2:]:
            f_cbm = F.interpolate(f_cbm, size=f_has.shape[-2:], mode="bilinear", align_corners=False)

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
        agree = (amc + amh + ahc) / 3.0

        edge_g  = F.softplus(self._edge_g)
        agree_g = F.softplus(self._agree_g)

        lm = lm + edge_g * edge.flatten(1) + agree_g * (1.0 - agree)
        lh = lh + agree_g * (1.0 - agree)
        lc = lc + agree_g * agree - edge_g * edge.flatten(1)

        logits  = torch.cat([lm, lh, lc], dim=1)
        weights = F.softmax(logits / self.tau, dim=1)

        wm, wh, wc = [w.view(-1,1,1,1) for w in weights.split(1, dim=1)]
        return wm * f_msu + wh * f_has + wc * f_cbm


# ----------------- Decoder (flex) -----------------
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


# ----------------- Full model (Res2Net-50 encoder) -----------------
class UNetWithMSU_HASSkip_CBAM_ASFG_R2N50(nn.Module):
    """
    Res2Net-50 encoder + MSU graph + HAS-Skip + ASFG fusion at each level.
    Channels follow Res2Net-50:
        C2=256@H/4, C3=512@H/8, C4=1024@H/16, C5=2048@H/32
    Decoder widths: [1024, 512, 256, 128]
    """
    def __init__(self, in_channels: int = 1, cbam_reduction: int = 16):
        super().__init__()

        # Encoder (Res2Net-50)
        self.backbone = build_res2net50()
        self.input_adapter = nn.Conv2d(in_channels, 3, 1, bias=False) if in_channels != 3 else nn.Identity()

        # Bottleneck precision tilt (optional but helpful)
        self.cbam_bott = ResidualCBAM(2048, reduction=cbam_reduction, use_spatial=False, alpha_init=0.25)

        # MSU graph (pairwise absolute-diff fusions)
        self.msu_A12   = AlignMSU(256,  512,  256)
        self.msu_A23   = AlignMSU(512,  1024, 512)
        self.msu_A34   = AlignMSU(1024, 2048, 1024)
        self.msu_P1223 = AlignMSU(256,  512,  256)
        self.msu_P2334 = AlignMSU(512,  1024, 512)
        self.msu_Qlast = AlignMSU(256,  512,  256)
        self.msu_top   = AlignMSU(2048, 2048, 1024)   # MSU(C5, C5_bott) for d1

        # HAS-Skip (decoder-aware gating)
        self.has = HASSkip(
            Cin_list=(256,512,1024,2048),
            Cout_list=(1024,512,256,128),
            Cdec_list=(2048,1024,512,256),
        )

        # ASFG per level (channels match decoder skip widths)
        self.asfg_d1 = AdaptiveSelectiveFusionGate(
            channels=1024, reduction=cbam_reduction, use_spatial_cbam=False,
            tau=1.8, prior_logits=(+0.7, +0.2, -0.3), edge_boost_gain=0.5, agree_boost_gain=0.3
        )
        self.asfg_d2 = AdaptiveSelectiveFusionGate(
            channels=512, reduction=cbam_reduction, use_spatial_cbam=False,
            tau=1.6, prior_logits=(+0.4, +0.2,  0.0), edge_boost_gain=0.4, agree_boost_gain=0.3
        )
        self.asfg_d3 = AdaptiveSelectiveFusionGate(
            channels=256, reduction=cbam_reduction, use_spatial_cbam=False,
            tau=1.4, prior_logits=(+0.1, +0.3, +0.1), edge_boost_gain=0.3, agree_boost_gain=0.35
        )
        self.asfg_d4 = AdaptiveSelectiveFusionGate(
            channels=128, reduction=cbam_reduction, use_spatial_cbam=True,
            tau=1.2, prior_logits=(-0.2, +0.2, +0.7), edge_boost_gain=0.0, agree_boost_gain=0.5
        )
        
        #project FMSU_d2 (1024) -> 512 to match FSKIP_d2
        self.proj_msu_d2 = nn.Conv2d(1024, 512, kernel_size=1, bias=False)
        self.proj_msu_d3 = nn.Conv2d( 512, 256, kernel_size=1, bias=False)
        self.proj_msu_d4 = nn.Conv2d( 256, 128, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.proj_msu_d2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.proj_msu_d3.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.proj_msu_d4.weight, nonlinearity='relu')

        # Decoder
        self.d1 = DecoderBlockFlex(in_ch=2048, skip_ch=1024, out_ch=1024)  # H/32 → H/16
        self.d2 = DecoderBlockFlex(in_ch=1024, skip_ch= 512, out_ch= 512)  # H/16 → H/8
        self.d3 = DecoderBlockFlex(in_ch= 512, skip_ch= 256, out_ch= 256)  # H/8  → H/4
        self.d4 = DecoderBlockFlex(in_ch= 256, skip_ch= 128, out_ch= 128)  # H/4  → H/2

        self.up_final = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # H/2 → H
        self.head     = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]                     # <- cache input size
        x = self.input_adapter(x)
        c2, c3, c4, c5 = self.backbone.forward_features(x)   # 256, 512, 1024, 2048

        # Bottleneck refinement
        c5b = self.cbam_bott(c5)

        # MSU graph
        A12   = self.msu_A12(c2, c3)       # 256 @ H/4
        A23   = self.msu_A23(c3, c4)       # 512 @ H/8
        A34   = self.msu_A34(c4, c5)       # 1024@ H/16
        P1223 = self.msu_P1223(A12, A23)   # 256 @ H/4
        P2334 = self.msu_P2334(A23, A34)   # 512 @ H/8
        Qlast = self.msu_Qlast(P1223, P2334)  # 256 @ H/4

        # Per-level FMSU
        FMSU_d1 = self.msu_top(c5, c5b)    # 1024 @ H/32
        FMSU_d2 = A34                      # 1024 @ H/16
        FMSU_d3 = P2334                    #  512 @ H/8
        FMSU_d4 = Qlast                    #  256 @ H/4

        # HAS-Skip (contexts: c5b, d1, d2, d3)
        FSKIP_d1 = self.has.forward_level(0, [c2,c3,c4,c5], c5b, c5)     # 1024 @ H/32
        FB1 = self.asfg_d1(FMSU_d1, FSKIP_d1)
        d1  = self.d1(c5b, FB1)                                           # 1024 @ H/16

        # d2: 1024 → 512
        FSKIP_d2      = self.has.forward_level(1, [c2,c3,c4,c5], d1, c4)              # (B,512, H/16, W/16)
        FMSU_d2_align = _resize_like(FMSU_d2, d1)                                     # (B,1024,H/16,W/16)
        FMSU_d2_proj  = self.proj_msu_d2(FMSU_d2_align)                               # (B,512, H/16, W/16)
        FB2 = self.asfg_d2(FMSU_d2_proj, FSKIP_d2)
        d2  = self.d2(d1, FB2)

        # d3: 512 → 256
        FSKIP_d3      = self.has.forward_level(2, [c2,c3,c4,c5], d2, c3)              # (B,256,H/8,W/8)
        FMSU_d3_align = _resize_like(FMSU_d3, d2)                                     # (B,512,H/8,W/8)
        FMSU_d3_proj  = self.proj_msu_d3(FMSU_d3_align)                               # (B,256,H/8,W/8)
        FB3 = self.asfg_d3(FMSU_d3_proj, FSKIP_d3)
        d3  = self.d3(d2, FB3)

        # d4: 256 → 128
        FSKIP_d4      = self.has.forward_level(3, [c2,c3,c4,c5], d3, c2)              # (B,128,H/4,W/4)
        FMSU_d4_align = _resize_like(FMSU_d4, d3)                                     # (B,256,H/4,W/4)
        FMSU_d4_proj  = self.proj_msu_d4(FMSU_d4_align)                               # (B,128,H/4,W/4)
        FB4 = self.asfg_d4(FMSU_d4_proj, FSKIP_d4)
        d4  = self.d4(d3, FB4)

        out = self.up_final(d4)
        out = self.head(out)
        if out.shape[-2:] != (H, W):            # <- defensive align
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return out                                             # 1   @ H
