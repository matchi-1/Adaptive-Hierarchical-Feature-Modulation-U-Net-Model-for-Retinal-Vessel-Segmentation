
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any

from src.models.blocks.cbam import CBAM
from src.models.blocks.has_skip import HASSkip
from src.models.blocks.msu import MSU
from src.models.unet import ConvBlock, EncoderBlock

class Align1x1(nn.Module):
    """Channel alignment using 1x1 conv with optional resize.

    Args:
        in_ch: Input channels.
        out_ch: Desired output channels.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
        """Resizes spatially and aligns channels.

        Args:
            x: Feature map [B, C, H, W].
            size_hw: Target (H, W).
        Returns:
            Tensor [B, out_ch, size_hw[0], size_hw[1]].
        """
        if x.shape[-2:] != size_hw:
            x = F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)
        return self.proj(x)
    

class UNetEncoderWrapper(nn.Module):
    """
    4-level encoder built from the base UNet's EncoderBlock.
    Returns [E1, E2, E3, E4] and exposes .out_channels.
    """
    def __init__(self, in_ch=1, widths=(64, 128, 256, 512)):
        super().__init__()
        c1, c2, c3, c4 = widths
        self.stage1 = EncoderBlock(in_ch, c1)   # E1 at H×W, pool→ H/2
        self.stage2 = EncoderBlock(c1,  c2)     # E2 at H/2, pool→ H/4
        self.stage3 = EncoderBlock(c2,  c3)     # E3 at H/4, pool→ H/8
        self.stage4 = EncoderBlock(c3,  c4)     # E4 at H/8, pool→ H/16 (if you downsample here)

        self.out_channels = [c1, c2, c3, c4]

    def forward(self, x):
        e1, p1 = self.stage1(x)   # e1 used as skip, p1 goes deeper
        e2, p2 = self.stage2(p1)
        e3, p3 = self.stage3(p2)
        e4, _  = self.stage4(p3)
        return [e1, e2, e3, e4]


class DecoderStage(nn.Module):
    """One decoder stage that upsamples and fuses multiple inputs.

    Args:
        in_up_ch: Channels entering from the deeper decoder.
        fuse_in_chs: List of channel counts for extra inputs to concatenate.
        out_ch: Output feature channels after fusion.
    """
    def __init__(self, in_up_ch: int, fuse_in_chs: List[int], out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_up_ch, out_ch, kernel_size=2, stride=2)
        total_in = out_ch + sum(fuse_in_chs)
        self.fuse = ConvBlock(total_in, out_ch)

    def forward(self, x_up: torch.Tensor, *to_concat: torch.Tensor) -> torch.Tensor:
        """Fuses upsampled decoder feature with auxiliary inputs.

        Args:
            x_up: Feature from previous decoder stage [B, in_up_ch, H, W].
            *to_concat: Extra tensors aligned to the post-upsample size, each [B, C_i, H, W].
        Returns:
            Tensor [B, out_ch, H, W].
        """
        x = self.up(x_up)  # upsampled feature (sets the target HxW for this stage)

        if to_concat:
            # auto-resize every aux tensor to match x’s HxW
            H, W = x.shape[-2], x.shape[-1]
            resized = []
            for t in to_concat:
                if t.shape[-2:] != (H, W):
                    t = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
                resized.append(t)
            x = torch.cat([x, *resized], dim=1)

        x = self.fuse(x)
        return x


class UNet_CBAM_HAS_MSU(nn.Module):
    """U-Net decoder with vanilla encoder, CBAM, HAS-Skip, and 6-MSU pyramid.

    Wiring:
      Encoders: vanilla UNet encoders [E1..E4].
      CBAM at each encoder: E'i = CBAM(Ei).
      MSUs: M1=MSU(E1',E2'), M2=MSU(E2',E3'), M3=MSU(E3',E4'),
            M4=MSU(M1,M2), M5=MSU(M2,M3), M6=MSU(M4,M5).
      Decoder inputs:
        D4: up(D3), E4', HAS_4
        D3: up(D2), E3', M3, HAS_3
        D2: up(D1), E2', M2, M5, HAS_2
        D1: up(BN),  E1', M1, M4, M6, HAS_1
      CBAM outputs are always concatenated at their matching decoder level.

    Args:
        in_channels:        Input image channels.
        dec_channels:       Decoder stage channels from coarse→fine, length 4 (e.g., [512, 256, 128, 64]).
        out_channels:       Segmentation head output channels.
        msu_mid_channels:   Channel width used by MSU outputs before alignment.
        has_skip_cfg:       Kwargs for HASSkip per level (applied uniformly).
        cbam_shared:        If True, reuse CBAM per unique channel width; else, distinct per level.
    """
    def __init__(
        self,
        backbone: Optional[nn.Module] = None,
        in_channels: int = 1,
        dec_channels: List[int] = (512, 256, 128, 64),
        out_channels: int = 1,
        msu_mid_channels: int = 64,
        has_skip_cfg: Optional[Dict[str, Any]] = None,
        cbam_shared: bool = False,
    ):
        super().__init__()
        
        self.backbone = backbone or UNetEncoderWrapper(in_ch=in_channels, widths=(64, 128, 256, 512))
        self.enc_out_chs = list(backbone.out_channels)
        
        self.cbams = nn.ModuleList()
        if cbam_shared:
            shared = nn.ModuleDict()
            for ch in sorted(set(self.enc_out_chs)):
                shared[str(ch)] = CBAM(ch)
            for ch in self.enc_out_chs:
                self.cbams.append(shared[str(ch)])
        else:
            for ch in self.enc_out_chs:
                self.cbams.append(CBAM(ch))

        self.align_e1 = Align1x1(self.enc_out_chs[0], dec_channels[-1])
        self.align_e2 = Align1x1(self.enc_out_chs[1], dec_channels[-2])
        self.align_e3 = Align1x1(self.enc_out_chs[2], dec_channels[-3])
        self.align_e4 = Align1x1(self.enc_out_chs[3], dec_channels[-4])
        self.msu_align_e2_to_e1 = Align1x1(self.enc_out_chs[1], self.enc_out_chs[0])  # 512→256
        self.msu_align_e3_to_e2 = Align1x1(self.enc_out_chs[2], self.enc_out_chs[1])  # 1024→512
        self.msu_align_e4_to_e3 = Align1x1(self.enc_out_chs[3], self.enc_out_chs[2])  # 2048→1024


        self.msu_e1e2 = MSU(self.enc_out_chs[0], msu_mid_channels)
        self.msu_e2e3 = MSU(self.enc_out_chs[1], msu_mid_channels)
        self.msu_e3e4 = MSU(self.enc_out_chs[2], msu_mid_channels)
        self.msu_m1m2 = MSU(msu_mid_channels, msu_mid_channels)
        self.msu_m2m3 = MSU(msu_mid_channels, msu_mid_channels)
        self.msu_m4m5 = MSU(msu_mid_channels, msu_mid_channels)

        self.align_m1 = Align1x1(msu_mid_channels, dec_channels[-1])
        self.align_m4 = Align1x1(msu_mid_channels, dec_channels[-1])
        self.align_m6 = Align1x1(msu_mid_channels, dec_channels[-1])
        self.align_m2 = Align1x1(msu_mid_channels, dec_channels[-2])
        self.align_m5 = Align1x1(msu_mid_channels, dec_channels[-2])
        self.align_m3 = Align1x1(msu_mid_channels, dec_channels[-3])
        self.msu_align_m2_to_m1 = Align1x1(msu_mid_channels, msu_mid_channels)
        self.msu_align_m3_to_m2 = Align1x1(msu_mid_channels, msu_mid_channels)
        self.msu_align_m5_to_m4 = Align1x1(msu_mid_channels, msu_mid_channels)

        has_cfg = has_skip_cfg or {}
        # HAS for D4 (matches E4):
        # dec_feat you pass = interpolated bottleneck b  → channels = 512
        # we want the skip output aligned to D4’s width → 512
        self.has4 = HASSkip(self.enc_out_chs, 512, 512, **has_cfg)

        # HAS for D3 (matches E3):
        # dec_feat you pass = upsampled d4              → channels = 512
        # align to D3’s width                           → 256
        self.has3 = HASSkip(self.enc_out_chs, 512, 256, **has_cfg)

        # HAS for D2 (matches E2):
        # dec_feat you pass = upsampled d3              → channels = 256
        # align to D2’s width                           → 128
        self.has2 = HASSkip(self.enc_out_chs, 256, 128, **has_cfg)

        # HAS for D1 (matches E1):
        # dec_feat you pass = upsampled d2              → channels = 128
        # align to D1’s width                           → 64
        self.has1 = HASSkip(self.enc_out_chs, 128, 64, **has_cfg)

        self.dec4 = DecoderStage(
            in_up_ch=dec_channels[0],                     # b → 512
            fuse_in_chs=[dec_channels[0], dec_channels[0]],
            out_ch=dec_channels[0],                       # 512
        )
        self.dec3 = DecoderStage(
            in_up_ch=dec_channels[0],                     # d4 is 512
            fuse_in_chs=[dec_channels[1], dec_channels[1], dec_channels[1]],
            out_ch=dec_channels[1],                       # 256
        )
        self.dec2 = DecoderStage(
            in_up_ch=dec_channels[1],                     # d3 is 256
            fuse_in_chs=[dec_channels[2], dec_channels[2], dec_channels[2], dec_channels[2]],
            out_ch=dec_channels[2],                       # 128
        )
        self.dec1 = DecoderStage(
            in_up_ch=dec_channels[2],                     # d2 is 128
            fuse_in_chs=[dec_channels[3], dec_channels[3], dec_channels[3], dec_channels[3], dec_channels[3]],
            out_ch=dec_channels[3],                       # 64
        )

        self.bottleneck = ConvBlock(self.enc_out_chs[3], dec_channels[0])
        self.head = nn.Conv2d(dec_channels[-1], out_channels, kernel_size=1)

    def _run_cbam(self, e_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """Applies CBAM to each encoder output.

        Args:
            e_list: List of 4 tensors [E1..E4].
        Returns:
            List [E1', E2', E3', E4'].
        """
        return [m(e) for m, e in zip(self.cbams, e_list)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input image tensor [B, C, H, W].
        Returns:
            Segmentation logits [B, out_channels, H, W].
        """
        e1, e2, e3, e4 = self.backbone(x)
        e1p, e2p, e3p, e4p = self._run_cbam([e1, e2, e3, e4])

        b = self.bottleneck(e4)

        # sizes of the first inputs (E1,E2,E3)
        size_e1 = (e1.shape[-2], e1.shape[-1])
        size_e2 = (e2.shape[-2], e2.shape[-1])
        size_e3 = (e3.shape[-2], e3.shape[-1])

        # align channels + resize second arg to match the first
        e2_for_m1 = self.msu_align_e2_to_e1(e2p, size_e1)
        m1 = self.msu_e1e2(e1p, e2_for_m1)

        e3_for_m2 = self.msu_align_e3_to_e2(e3p, size_e2)
        m2 = self.msu_e2e3(e2p, e3_for_m2)

        e4_for_m3 = self.msu_align_e4_to_e3(e4p, size_e3)
        m3 = self.msu_e3e4(e3p, e4_for_m3)

        # align inputs for hierarchical MSUs
        m2_for_m4 = self.msu_align_m2_to_m1(m2, size_e1)  # -> size_e1
        m3_for_m5 = self.msu_align_m3_to_m2(m3, size_e2)  # -> size_e2

        m4 = self.msu_m1m2(m1, m2_for_m4)  # size_e1
        m5 = self.msu_m2m3(m2, m3_for_m5)  # size_e2

        m5_for_m6 = self.msu_align_m5_to_m4(m5, size_e1)  # -> size_e1
        m6 = self.msu_m4m5(m4, m5_for_m6)  # size_e1


        size_d1 = (e1.shape[-2], e1.shape[-1])
        size_d2 = (e2.shape[-2], e2.shape[-1])
        size_d3 = (e3.shape[-2], e3.shape[-1])
        size_d4 = (e4.shape[-2], e4.shape[-1])

        e1a = self.align_e1(e1p, size_d1)
        e2a = self.align_e2(e2p, size_d2)
        e3a = self.align_e3(e3p, size_d3)
        e4a = self.align_e4(e4p, size_d4)

        m1a = self.align_m1(m1, size_d1)
        m4a = self.align_m4(m4, size_d1)
        m6a = self.align_m6(m6, size_d1)
        m2a = self.align_m2(m2, size_d2)
        m5a = self.align_m5(m5, size_d2)
        m3a = self.align_m3(m3, size_d3)

        # sizes for convenience
        size_d1 = (e1.shape[-2], e1.shape[-1])
        size_d2 = (e2.shape[-2], e2.shape[-1])
        size_d3 = (e3.shape[-2], e3.shape[-1])
        size_d4 = (e4.shape[-2], e4.shape[-1])

        enc_list = [e1p, e2p, e3p, e4p]

        # --- D4 (matches E4). HAS_4 needs a decoder feature at E4 scale.
        dec4_query = F.interpolate(b, size=size_d4, mode="bilinear", align_corners=False)
        has4 = self.has4(enc_feats=enc_list, dec_feat=dec4_query, level_l=3)
        d4 = self.dec4(b, e4a, has4)

        # --- D3 (matches E3). Use upsampled d4 as dec_feat.
        dec3_query = F.interpolate(d4, size=size_d3, mode="bilinear", align_corners=False)
        has3 = self.has3(enc_feats=enc_list, dec_feat=dec3_query, level_l=2)
        d3 = self.dec3(d4, e3a, m3a, has3)

        # --- D2 (matches E2). Use upsampled d3 as dec_feat.
        dec2_query = F.interpolate(d3, size=size_d2, mode="bilinear", align_corners=False)
        has2 = self.has2(enc_feats=enc_list, dec_feat=dec2_query, level_l=1)
        d2 = self.dec2(d3, e2a, m2a, m5a, has2)

        # --- D1 (matches E1). Use upsampled d2 as dec_feat.
        dec1_query = F.interpolate(d2, size=size_d1, mode="bilinear", align_corners=False)
        has1 = self.has1(enc_feats=enc_list, dec_feat=dec1_query, level_l=0)
        d1 = self.dec1(d2, e1a, m1a, m4a, m6a, has1)

        out = self.head(d1)
        out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out
