import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class HASSkip(nn.Module):
    """
    Hierarchical Attention-Selective Skip Connections (HAS-Skip)

    Inputs (forward):
        - enc_feats: list of encoder feature maps [E1, E2, ..., EN]
            * shapes typically: [B, C_i, H_i, W_i] (vary by level)
        - dec_feat: current decoder feature F_decoder at level l
            * shape: [B, C_dec, H_d, W_d]
        - level_l: int, the index (0..N-1) of the *current* encoder level for which we produce the skip

    Output:
        - FSKIP_l: filtered skip feature for decoder level l
            * shape: [B, C_align, H_d, W_d]

    Algorithm:
        1) (Eq. 2.31) For each encoder output E_i, spatially resize to (H_d, W_d) via bilinear upsampling.
        2) (Eq. 2.32) Channel alignment via 1×1 conv: E_i^aligned = Conv1x1(E_i^resized) -> [B, C_align, H_d, W_d].
        3) (Eq. 2.33) Hierarchical fusion with learnable scalars w_i:
               F_agg = sum_i ( w_i * E_i^aligned ).
        4) (Eq. 2.34) Decoder-aware attention gate:
               G = σ( W_g(F_decoder) + W_x(F_agg) + b )
           implemented with 1×1 convs (linear transforms).
           G is produced as a *spatial* mask with 1 channel and broadcast across channels.
        5) (Eq. 2.35 / 2.36) Apply the gate to the *current* level’s aligned feature:
               FSKIP_l = G ⊙ E_l^aligned

    Notes:
        • C_align is typically chosen to match the decoder level’s channel width.
        • The gate G is 1×H×W so it gates all channels of E_l^aligned uniformly at each pixel.
          (You can switch to a per-channel gate by making psi produce C_align channels.)
    """

    def __init__(
        self,
        in_channels_per_level: List[int],  # [C1, C2, ..., CN] for encoder levels (low->high)
        dec_channels: int,                  # C_dec for the current decoder level(s)
        align_channels: int,                # C_align after 1x1 alignment, usually == decoder channels at that level
        inter_channels: int = None,         # optional bottleneck for the gate
    ):
        super().__init__()

        self.num_levels = len(in_channels_per_level)
        self.align_channels = align_channels

        # (Eq. 2.32) 1×1 convs to align channels per encoder level
        self.align_convs = nn.ModuleList([
            nn.Conv2d(cin, align_channels, kernel_size=1, bias=True)
            for cin in in_channels_per_level
        ])

        # (Eq. 2.33) Learnable scalars w_i per level (initialized to 1.0)
        self.register_parameter(
            "level_weights",
            nn.Parameter(torch.ones(self.num_levels, dtype=torch.float32))
        )

        # (Eq. 2.34) Gate: G = σ(W_g·F_decoder + W_x·F_agg + b)
        # Use a lightweight attention block: reduce to inter_channels, sum, then map to 1
        if inter_channels is None:
            # a small bottleneck works well; fall back to half of align_channels (at least 8)
            inter_channels = max(align_channels // 2, 8)

        self.Wg = nn.Sequential(
            nn.Conv2d(dec_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        self.Wx = nn.Sequential(
            nn.Conv2d(align_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        # psi acts like the "readout" to 1 channel; bias here plays the role of +b
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

        # simple init: keep things stable at start
        self._init_weights()

    def _init_weights(self):
        # Kaiming init for convs; scalars start at 1
        for m in list(self.align_convs) + [self.Wg[0], self.Wx[0], self.psi]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _resize_to(self, x: torch.Tensor, size_hw: torch.Size) -> torch.Tensor:
        """Resize tensor x to spatial size (H, W) using bilinear upsampling. (Eq. 2.31)"""
        H, W = size_hw
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

    def forward(self, enc_feats: List[torch.Tensor], dec_feat: torch.Tensor, level_l: int) -> torch.Tensor:
        """
        Args:
            enc_feats: [E1, E2, ..., EN] encoder maps (low→high or high→low; index must match in_channels_per_level)
            dec_feat:  F_decoder at the current decoder level l
            level_l:   index of the encoder level whose skip we’re producing

        Returns:
            FSKIP_l: gated skip feature for level l, shape [B, C_align, H_d, W_d]
        """
        assert 0 <= level_l < self.num_levels, "level_l out of range"

        B, C_dec, H_d, W_d = dec_feat.shape

        # === Step 1 (Eq. 2.31): Resize all encoder features to (H_d, W_d) ===
        resized = [self._resize_to(Ei, (H_d, W_d)) for Ei in enc_feats]

        # === Step 2 (Eq. 2.32): Channel alignment via 1×1 conv per level ===
        aligned = [conv(ei) for conv, ei in zip(self.align_convs, resized)]
        # aligned[i]: [B, C_align, H_d, W_d]

        # === Step 3 (Eq. 2.33): Hierarchical fusion with learnable scalars w_i ===
        # F_agg = sum_i ( w_i * aligned[i] )
        # Broadcast w_i to [B, 1, 1, 1] and multiply
        F_agg = 0.0
        for i, Ei_al in enumerate(aligned):
            F_agg = F_agg + self.level_weights[i].view(1, 1, 1, 1) * Ei_al  # scalar weight per level

        # === Step 4 (Eq. 2.34): Decoder-aware gate ===
        # G = σ( W_g(F_decoder) + W_x(F_agg) + b ), with psi providing the bias/readout
        g_dec = self.Wg(dec_feat)            # [B, C_int, H_d, W_d]
        g_agg = self.Wx(F_agg)               # [B, C_int, H_d, W_d]
        g      = self.sigmoid(self.psi(torch.relu(g_dec + g_agg)))  # [B, 1, H_d, W_d] ∈ [0,1]

        # === Step 5 (Eq. 2.35 / 2.36): Apply gate to current level’s aligned feature ===
        E_l_aligned = aligned[level_l]       # [B, C_align, H_d, W_d]
        FSKIP_l     = g * E_l_aligned        # broadcast multiply (⊙)

        return FSKIP_l
