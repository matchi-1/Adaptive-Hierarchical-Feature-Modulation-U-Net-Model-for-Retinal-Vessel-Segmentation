# src\models\blocks\has_skip_exp1.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

def _make_gn(num_channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """GroupNorm with group count that divides channels."""
    g = min(max_groups, num_channels)
    while num_channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, num_channels)

class HASSkip(nn.Module):
    """
    Hierarchical Attention-Selective Skip (HAS-Skip)

    Forward(enc_feats, dec_feat, level_l) -> FSKIP_l  with shape (B, C_align, H_d, W_d)

    Improvements vs. baseline:
      - Per-level Wg (decoder projection): handles changing decoder widths across levels
      - GroupNorm (stable for small batch sizes) in gate projections
      - Safe tensor init for fused sum (no 0.0 scalar)
      - Optional softmax over level weights (convex fusion)
    """
    def __init__(
        self,
        in_channels_per_level: List[int],       # [C1, C2, ..., CN] from encoder (E1..EN)
        dec_channels_per_level: List[int],      # [D1, D2, ..., DN] decoder widths per level (U1..UN)
        align_channels: int,                    # C_align after 1x1 alignment (match MSU/HAS fusion channels)
        inter_channels: Optional[int] = None,   # hidden width for gate projections (Wg, Wx)
        use_softmax_weights: bool = False,      # if True: softmax-normalize level weights across i
    ):
        super().__init__()
        assert len(in_channels_per_level) == len(dec_channels_per_level), \
            "Encoder and decoder level lists must have same length"
        self.num_levels = len(in_channels_per_level)
        self.align_channels = int(align_channels)
        self.use_softmax_weights = bool(use_softmax_weights)

        # 1×1 per-encoder-level to align channels to C_align (Eq. 2.32)
        self.align_convs = nn.ModuleList([
            nn.Conv2d(cin, self.align_channels, kernel_size=1, bias=True)
            for cin in in_channels_per_level
        ])

        # Learnable scalar weight per encoder level (Eq. 2.33)
        self.level_weights = nn.Parameter(torch.ones(self.num_levels, dtype=torch.float32))

        # Gate bottleneck width
        if inter_channels is None:
            inter_channels = max(self.align_channels // 2, 8)
        self.inter_channels = int(inter_channels)

        # Per-level Wg: project decoder feature at level ℓ to inter_channels
        self.Wg = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dc, self.inter_channels, kernel_size=1, bias=True),
                _make_gn(self.inter_channels)
            )
            for dc in dec_channels_per_level
        ])

        # Single Wx shared across levels: project fused encoder agg to inter_channels
        self.Wx = nn.Sequential(
            nn.Conv2d(self.align_channels, self.inter_channels, kernel_size=1, bias=True),
            _make_gn(self.inter_channels)
        )

        # Readout to 1 channel gate (spatial mask), then sigmoid (Eq. 2.34)
        self.psi = nn.Conv2d(self.inter_channels, 1, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        # Kaiming init for convs; level_weights already init to 1
        for m in list(self.align_convs):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        for block in list(self.Wg) + [self.Wx]:
            conv = block[0]  # first layer is Conv2d
            nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
        nn.init.kaiming_normal_(self.psi.weight, nonlinearity="relu")
        if self.psi.bias is not None:
            nn.init.zeros_(self.psi.bias)

    @staticmethod
    def _resize_to(x: torch.Tensor, size_hw: torch.Size) -> torch.Tensor:
        H, W = size_hw
        return x if x.shape[-2:] == (H, W) else F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

    def forward(self, enc_feats: List[torch.Tensor], dec_feat: torch.Tensor, level_l: int) -> torch.Tensor:
        """
        enc_feats: [E1..EN], arbitrary native sizes/ch
        dec_feat:  decoder feature at current level ℓ, shape (B, Dℓ, H_d, W_d)
        level_l:   0..N-1 (which encoder level's skip we are producing)
        """
        assert 0 <= level_l < self.num_levels, "level_l out of range"

        B, _, H_d, W_d = dec_feat.shape

        # (1) Resize all encoder features to (H_d, W_d) (Eq. 2.31)
        resized = [self._resize_to(Ei, (H_d, W_d)) for Ei in enc_feats]

        # (2) Channel-align to C_align (Eq. 2.32)
        aligned = [conv(ei) for conv, ei in zip(self.align_convs, resized)]  # each: (B, C_align, H_d, W_d)

        # (3) Weighted fusion across encoder levels (Eq. 2.33)
        if self.use_softmax_weights:
            w = torch.softmax(self.level_weights, dim=0)
        else:
            w = self.level_weights
        # broadcast weights and sum
        F_agg = torch.zeros_like(aligned[0])
        for i, Ei in enumerate(aligned):
            F_agg = F_agg + w[i].view(1, 1, 1, 1) * Ei  # (B, C_align, H_d, W_d)

        # (4) Decoder-aware gate (Eq. 2.34): G = σ( psi( ReLU( Wgℓ(dec) + Wx(F_agg) ) ) )
        g_dec = self.Wg[level_l](dec_feat)   # (B, C_int, H_d, W_d)
        g_agg = self.Wx(F_agg)               # (B, C_int, H_d, W_d)
        gate  = self.sigmoid(self.psi(F.relu(g_dec + g_agg, inplace=True)))  # (B, 1, H_d, W_d)

        # (5) Apply gate to the *current* level aligned feature E_l (Eq. 2.35)
        E_l = aligned[level_l]               # (B, C_align, H_d, W_d)
        FSKIP_l = gate * E_l                 # broadcast multiply

        return FSKIP_l
