# src/models/wrappers/dpcn_concat_unet_exp1.py
import torch
import torch.nn as nn


from src.models.dpcn.dpcn_exp1 import DPCN as DPCN_Exp1

from src.models.unet_exp.base_unet_ablations.base_unet_msu_cbam_hasskip_improved3 import (
    UNetWithMSU_HASSkip_CBAM_ASFG
)

class DPCNConcatUNet_Exp1(nn.Module):
    """
    Pipeline:
        x [N,1,H,W]
          └─ DPCN_exp1 (returns [N,T,C,H,W])
               -> concat over T,C → [N, T*C, H, W]
               -> optional 1x1 stem to reduce channels
               -> Base UNet expecting in_channels = T*C (or reduced)
    """
    def __init__(
        self,
        in_ch: int = 1,
        enh_channels: int = 64,     
        iters: int = 5,             
        beta_init: float = 0.3,     
        aE: float = 0.8,            
        V_E: float = 0.3,          
        clamp_each_iter: bool = True,
        threshold_mode: str = "vat",   # "paper" | "paper_mod" | "vat" | "scaled_vat"
        reduce_to: int | None = None,  
        base_kwargs: dict | None = None
    ):
        super().__init__()
        self.iters = int(iters)
        self.enh_channels = int(enh_channels)

        # 1) Your exp1 enhancer (returns [N,T,C,H,W])
        self.enh = DPCN_Exp1(
            in_ch=in_ch,
            channels=enh_channels,
            iters=iters,
            beta_init=beta_init,
            aE=aE,
            V_E=V_E,
            clamp_each_iter=clamp_each_iter,
            threshold_mode=threshold_mode,
        )

        # 2) Prepare base input channels (T*C or reduced)
        in_ch_base = enh_channels * iters
        self.stem = nn.Identity()
        if reduce_to is not None and reduce_to != in_ch_base:
            self.stem = nn.Conv2d(in_ch_base, reduce_to, kernel_size=1, bias=True)
            in_ch_base = reduce_to

        # 3) MSU/HAS/CBAM UNet
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

        # pass to UNet/MSU/HAS/CBAM
        return self.base(x_cat)
