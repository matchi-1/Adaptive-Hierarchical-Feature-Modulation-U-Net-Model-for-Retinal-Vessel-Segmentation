# src/models/wrappers/dpcn_concat_unet_exp1.py
import torch
import torch.nn as nn


from src.models.dpcn.dpcn_exp1 import DPCN as DPCN_Exp1
from src.models.dpcn.dpcn_v2 import DPCN

from src.models.unet_exp.base_unet_ablations.base_unet_msu_cbam_hasskip_improved3 import (
    UNetWithMSU_HASSkip_CBAM_ASFG
)
from src.models.unet_exp.base_unet_ablations.base_unet_r2n50_msu_cbam_hasskip_ver2 import UNetWithMSU_HASSkip_CBAM_ASFG_R2N50

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




class DPCNConcatUNet(nn.Module):        # version 1
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


class DPCNConcatRes2UNet_ASFG(nn.Module):
    """
    DPCN → stack → concat(T*C) → (optional 1x1 stem) → UNetWithMSU_HASSkip_CBAM_ASFG_R2N50
    Keeps the exact logic of DPCNConcatUNet; only the base changes.
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

        # 1) DPCN enhancer returning [N,T,C,H,W]
        self.enh = DPCN(
            in_ch=in_ch,
            channels=enh_channels,
            iters=iters,
            threshold_mode=threshold_mode,
            half_life=half_life,
            aggregate="stack",
        )

        # 2) Prep base input channels (T*C or reduced)
        in_ch_base = enh_channels * iters
        self.stem = nn.Identity()
        if reduce_to is not None and reduce_to != in_ch_base:
            self.stem = nn.Conv2d(in_ch_base, reduce_to, kernel_size=1, bias=True)
            in_ch_base = reduce_to

        # 3) Base: Res2Net-50 encoder + MSU + HAS + ASFG
        base_kwargs = base_kwargs or {}
        self.base = UNetWithMSU_HASSkip_CBAM_ASFG_R2N50(in_channels=in_ch_base, **base_kwargs)

    def forward(self, x: torch.Tensor, fov: torch.Tensor | None = None) -> torch.Tensor:
        
        # Run DPCN in fp32 to avoid FP16–bias mismatch
        was_autocast = torch.is_autocast_enabled()
        with torch.amp.autocast(device_type="cuda", enabled=False):
            ys = self.enh(x.float(), fov=fov)           # [N,T,C,H,W] in fp32

        N, T, C, H, W = ys.shape
        x_cat = ys.reshape(N, T*C, H, W)
        x_cat = self.stem(x_cat)

        # Cast back to the active autocast dtype (fp16/bf16) if we were in AMP
        if was_autocast:
            x_cat = x_cat.to(dtype=torch.get_autocast_gpu_dtype())

        return self.base(x_cat)                         # rest runs under AMP as usual