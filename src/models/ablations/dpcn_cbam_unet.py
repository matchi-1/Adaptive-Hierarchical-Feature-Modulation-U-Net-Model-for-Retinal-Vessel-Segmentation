from typing import Optional
import torch
import torch.nn as nn
from src.models.unet import UNet
from src.models.blocks.cbam import CBAM

class DPCN_CBAM_UNet(nn.Module):
    """
    Pipeline: DPCN -> concat T iteration maps along channels -> CBAM -> UNet.
    Returns logits and the raw per-iteration maps for visualization.
    """
    def __init__(self, dpcn: nn.Module, reduction_ratio: int = 8, use_spatial_cbam: bool = True):
        super().__init__()
        self.dpcn = dpcn                      # your DPCN instance
        T, C = int(dpcn.iters), int(dpcn.channels)
        self.T, self.C = T, C
        gate_channels = T * C                 # channels seen by CBAM

        # CBAM on the concatenated stack
        self.cbam = CBAM(gate_channels=gate_channels,
                         reduction_ratio=reduction_ratio,
                         pool_types=['avg', 'max'],
                         use_spatial=use_spatial_cbam)

        # UNet that accepts T*C channels
        self.unet = UNet(in_channels=gate_channels)

    def forward(self, x: torch.Tensor, fov: Optional[torch.Tensor] = None):
        ys = self.dpcn(x, fov=fov)                          # [B,T,C,H,W]
        B, T, C, H, W = ys.shape
        assert T == self.T and C == self.C, "DPCN config changed?"

        feats = ys.reshape(B, T*C, H, W).contiguous()       # [B,T*C,H,W]
        feats = self.cbam(feats)                            # CBAM refine
        logits = self.unet(feats)                           # [B,1,H,W]
        return logits, ys, feats
