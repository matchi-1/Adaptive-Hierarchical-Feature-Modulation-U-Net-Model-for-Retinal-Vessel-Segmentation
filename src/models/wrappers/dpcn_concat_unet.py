# src/models/wrappers/dpcn_concat_unet.py
import torch
import torch.nn as nn


from src.models.dpcn.dpcn_v2 import DPCN
from src.models.unet_exp.base_unet_ablations.base_unet_msu_cbam_hasskip_improved3 import UNetWithMSU_HASSkip_CBAM_ASFG

class DPCNConcatUNet(nn.Module):
    """
    Wraps DPCN (aggregate='stack') in front of your UNet/MSU/HAS/CBAM model.

    Pipeline:
        x [N,1,H,W]
          └─ DPCN (T iters, C enh channels, aggregate='stack')
               -> ys [N,T,C,H,W]
               -> concat over (T,C) -> x_cat [N, T*C, H, W]
               -> (optional) 1x1 Conv stem to reduce channels
               -> Base UNet expecting in_channels = T*C (or reduced)

    Args:
        in_ch:          input image channels (usually 1)
        enh_channels:   DPCN internal channels C
        iters:          DPCN iterations T
        threshold_mode: 'paper' | 'paper_mod' | 'vat' | 'scaled_vat'
        half_life:      DPCN threshold memory in iterations (aE = ln2/half_life)
        reduce_to:      if not None, apply 1x1 conv to map T*C -> reduce_to channels
        base_kwargs:    kwargs forwarded to your base UNet constructor
    """
    def __init__(self,
                 in_ch: int = 1,
                 enh_channels: int = 32,
                 iters: int = 4,
                 threshold_mode: str = "scaled_vat",
                 half_life: float = 2.0,
                 reduce_to: int | None = None,
                 base_kwargs: dict | None = None):
        super().__init__()
        self.iters = int(iters)
        self.enh_channels = int(enh_channels)

        # 1) DPCN enhancer returning the full stack
        self.enh = DPCN(
            in_ch=in_ch,
            channels=enh_channels,
            iters=iters,
            threshold_mode=threshold_mode,
            half_life=half_life,
            aggregate="stack",     # << important: keep per-iteration maps
        )

        in_ch_base = enh_channels * iters
        self.stem = nn.Identity()
        if reduce_to is not None and reduce_to != in_ch_base:
            # 2) optional compression if T*C is too large
            self.stem = nn.Conv2d(in_ch_base, reduce_to, kernel_size=1, bias=True)
            in_ch_base = reduce_to

        # 3) base model takes in_channels=in_ch_base; rest unchanged
        base_kwargs = base_kwargs or {}
        self.base = UNetWithMSU_HASSkip_CBAM_ASFG(in_channels=in_ch_base, **base_kwargs)

    def forward(self, x: torch.Tensor, fov: torch.Tensor | None = None) -> torch.Tensor:
        # DPCN stack: [N, T, C, H, W]
        ys = self.enh(x, fov=fov)

        # concat T and C -> [N, T*C, H, W]
        N, T, C, H, W = ys.shape
        x_cat = ys.reshape(N, T * C, H, W)

        # optional 1x1 compress
        x_cat = self.stem(x_cat)

        # pass to existing UNet/MSU/HAS/CBAM
        return self.base(x_cat)
