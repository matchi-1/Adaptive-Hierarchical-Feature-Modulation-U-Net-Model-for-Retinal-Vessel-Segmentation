"""
    Integration: U-Net + HAS-Skip + MSU (MDFI-Net style residual overlays)
    
    Design summary :
    - Keep the original U-Net decoder CONCAT with the same-scale encoder skip.
    - Additionally add (residual, element-wise):
        (1) the channel-aligned raw encoder feature (same scale)
        (2) the HAS-Skip output for that level (hierarchical selective skip)
        (3) the MSU output for that level (multi-scale subtraction map)
    - HAS-Skip takes all encoder outputs + current decoder size feature as context.
    - MSU takes encoder-output pairs only (resized + channel-aligned to current level).
    - All overlay paths are summed to the decoder feature at each level.
    
    Assumptions / conventions:
    - Encoder feature order: [E1, E2, E3, E4] = shallow→deep (channels: 64, 128, 256, 512 typical)
    - Decoder levels (top→bottom): D1 uses E4, D2 uses E3, D3 uses E2, D4 uses E1.
    - Channel plan mirrors a classic U-Net: enc: 64-128-256-512, bottleneck: 1024, dec: 512-256-128-64.

"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================
# Utility / small components
# ==========================
from src.models.blocks.has_skip import HASSkip
from src.models.blocks.msu import MSU
from src.models.unet import ConvBlock, EncoderBlock, DecoderBlock

class Align1x1(nn.Module):
    """Per-level 1×1 aligner to map (C_in→C_out)."""
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.proj = nn.Conv2d(c_in, c_out, kernel_size=1)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity='relu')
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
        if x.shape[-2:] != size_hw:
            x = F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)
        return self.proj(x)

# ==================
# Model main module
# ==================
class UNet_HASSkip_MSU(nn.Module):
    """
    U-Net with:
      • Classic decoder concat skip from corresponding encoder level, plus
      • Three residual adds per decoder level:
           (a) aligned raw encoder feature (same scale)
           (b) HAS-Skip output
           (c) MSU output (difference of encoder features at/near the same scale)

    Channel plan (default):
        enc: [64, 128, 256, 512]
        bottleneck: 1024
        dec: [512, 256, 128, 64]

    Args:
        in_channels (int):  Number of channels in the input image.
                            Default: 1
        out_channels (int): Number of channels in the segmentation/logit output (e.g., 1 for binary).
                            Default: 1
        enc_channels(Tuple[int]):       Tuple of 4 ints giving encoder stage widths from shallow→deep (E1, E2, E3, E4). 
                                        Default: (64, 128, 256, 512).
        bottleneck_channels (int):      Channel width of the deepest bottleneck block before decoding.
                                        Default: 1024
        dec_channels(Tuple[int]):       Tuple of 4 ints giving decoder stage widths from top→bottom (D1 for the highest decoder level down to D4). 
                                        Default: (512, 256, 128, 64).
        use_two_msu_neighbors (bool):   If True, MSU at middle decoder levels (d2/d3) compares the
                                        same-scale encoder feature against two neighbors (one deeper 
                                        and one shallower) and sums both differential maps; 
                                        if False (default), uses a single neighbor per level for lower compute.
        
    Inputs:
        x: (N, in_channels, H, W)

    Output:
        y: (N, out_channels, H, W)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        enc_channels: List[int] = (64, 128, 256, 512),
        bottleneck_channels: int = 1024,
        dec_channels: List[int] = (512, 256, 128, 64),
        use_two_msu_neighbors: bool = False,
    ):
        super().__init__()

        assert len(enc_channels) == 4 and len(dec_channels) == 4, "This implementation assumes 4 encoder/decoder stages."

        C1, C2, C3, C4 = enc_channels
        D1, D2, D3, D4 = dec_channels

        # ---- Encoder ----
        self.e1 = EncoderBlock(in_channels, C1)
        self.e2 = EncoderBlock(C1, C2)
        self.e3 = EncoderBlock(C2, C3)
        self.e4 = EncoderBlock(C3, C4)

        # ---- Bottleneck ----
        self.bottleneck = ConvBlock(C4, bottleneck_channels)

        # ---- Decoder (classic concat path preserved) ----
        self.d1 = DecoderBlock(bottleneck_channels, D1)  # uses skip E4
        self.d2 = DecoderBlock(D1, D2)                   # uses skip E3
        self.d3 = DecoderBlock(D2, D3)                   # uses skip E2
        self.d4 = DecoderBlock(D3, D4)                   # uses skip E1

        # ---- Final head ----
        self.final = nn.Conv2d(D4, out_channels, kernel_size=1)

        # ---- Per-level 1×1 aligners ----
        # Same-scale raw encoder add (E_k -> D_k)
        self.align_e4_to_d1 = Align1x1(C4, D1)
        self.align_e3_to_d1 = Align1x1(C3, D1)
        self.align_e3_to_d2 = Align1x1(C3, D2)
        self.align_e2_to_d3 = Align1x1(C2, D3)
        self.align_e1_to_d4 = Align1x1(C1, D4)

        # Cross-scale aligners needed by MSU neighbors (E_j -> D_k)
        # Minimal set for single-neighbor policy:
        self.align_e3_to_d1 = Align1x1(C3, D1)  # for MSU at d1: E4 vs E3
        self.align_e4_to_d2 = Align1x1(C4, D2)  # for MSU at d2: E3 vs E4
        self.align_e3_to_d3 = Align1x1(C3, D3)  # d3: compare E2 vs E3   
        self.align_e2_to_d4 = Align1x1(C2, D4)  # for MSU at d4: E1 vs E2

        # If using two neighbors, add the remaining cross mapping aligners:
        self.use_two_msu_neighbors = use_two_msu_neighbors
        if self.use_two_msu_neighbors:
            self.align_e2_to_d2 = Align1x1(C2, D2)  # d2 extra neighbor (E2)
            self.align_e3_to_d3 = Align1x1(C3, D3)  # d3 extra neighbor (E3)

        # ---- HAS-Skip modules per decoder level ----
        self.has_d1 = HASSkip([C1, C2, C3, C4], dec_channels=D1, align_channels=D1)
        self.has_d2 = HASSkip([C1, C2, C3, C4], dec_channels=D2, align_channels=D2)
        self.has_d3 = HASSkip([C1, C2, C3, C4], dec_channels=D3, align_channels=D3)
        self.has_d4 = HASSkip([C1, C2, C3, C4], dec_channels=D4, align_channels=D4)

        # ---- MSU per decoder level ----
        self.msu_d1 = MSU(D1, D1)
        self.msu_d2 = MSU(D2, D2)
        self.msu_d3 = MSU(D3, D3)
        self.msu_d4 = MSU(D4, D4)

    # -------------
    # MSU helpers
    # -------------
    def _msu_level(self, level: int, encs: List[torch.Tensor], size_hw: Tuple[int, int]) -> torch.Tensor:
        """Compute the MSU map for a given decoder level using encoder-only pairs.
        Levels: 0→E1, 1→E2, 2→E3, 3→E4.
        Decoder mapping: d4←0, d3←1, d2←2, d1←3.
        """
        E1, E2, E3, E4 = encs  # shallow→deep

        if level == 3:  # d1 uses E4; compare E4 vs E3
            A = self.align_e4_to_d1(E4, size_hw)
            B = self.align_e3_to_d1(E3, size_hw)
            out = self.msu_d1(A, B)
            # no second neighbor above E4
            return out

        elif level == 2:  # d2 uses E3; compare E3 vs E4 (and optionally E2)
            A = self.align_e3_to_d2(E3, size_hw)
            B1 = self.align_e4_to_d2(E4, size_hw)
            out = self.msu_d2(A, B1)
            if self.use_two_msu_neighbors:
                B2 = self.align_e2_to_d2(E2, size_hw)
                out = out + self.msu_d2(A, B2)
            return out

        elif level == 1:  # d3 uses E2; compare E2 vs E3 (and optionally E1)
            A  = self.align_e2_to_d3(E2, size_hw)
            B1 = self.align_e3_to_d3(E3, size_hw)
            out = self.msu_d3(A, B1)
            if self.use_two_msu_neighbors:
                B2 = self.align_e1_to_d3(E1, size_hw)
                out = out + self.msu_d3(A, B2)
            return out

        else:  # level == 0: d4 uses E1; compare E1 vs E2
            A = self.align_e1_to_d4(E1, size_hw)
            B = self.align_e2_to_d4(E2, size_hw)
            out = self.msu_d4(A, B)
            # no second neighbor below E1 unless defined
            return out

    # ----------------
    # Forward pass
    # ----------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        s1, p1 = self.e1(x)   # E1
        s2, p2 = self.e2(p1)  # E2
        s3, p3 = self.e3(p2)  # E3
        s4, p4 = self.e4(p3)  # E4

        # Bottleneck
        b = self.bottleneck(p4)

        encs: List[torch.Tensor] = [s1, s2, s3, s4]

        # --------- Decoder level 1 (top): uses E4 ---------
        d1_base = self.d1(b, s4)  # concat path preserved
        size1 = d1_base.shape[-2:]
        r1 = self.align_e4_to_d1(s4, size1)
        hs1 = self.has_d1(encs, d1_base, level_l=3)
        m1 = self._msu_level(level=3, encs=encs, size_hw=size1)
        d1 = d1_base + r1 + hs1 + m1

        # --------- Decoder level 2: uses E3 ---------
        d2_base = self.d2(d1, s3)
        size2 = d2_base.shape[-2:]
        r2 = self.align_e3_to_d2(s3, size2)
        hs2 = self.has_d2(encs, d2_base, level_l=2)
        m2 = self._msu_level(level=2, encs=encs, size_hw=size2)
        d2 = d2_base + r2 + hs2 + m2

        # --------- Decoder level 3: uses E2 ---------
        d3_base = self.d3(d2, s2)
        size3 = d3_base.shape[-2:]
        r3 = self.align_e2_to_d3(s2, size3)
        hs3 = self.has_d3(encs, d3_base, level_l=1)
        m3 = self._msu_level(level=1, encs=encs, size_hw=size3)
        d3 = d3_base + r3 + hs3 + m3

        # --------- Decoder level 4 (bottom): uses E1 ---------
        d4_base = self.d4(d3, s1)
        size4 = d4_base.shape[-2:]
        r4 = self.align_e1_to_d4(s1, size4)
        hs4 = self.has_d4(encs, d4_base, level_l=0)
        m4 = self._msu_level(level=0, encs=encs, size_hw=size4)
        d4 = d4_base + r4 + hs4 + m4

        # Head
        out = self.final(d4)
        return out


if __name__ == "__main__":
    # quick smoke test
    model = UNet_HASSkip_MSU()
    x = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        y = model(x)
    print("Output:", y.shape)
