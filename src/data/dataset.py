# src/data/dataset.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.data.preprocessing import (
    preprocess_image_retina,
    preprocess_image_intensity_hsi,
    preprocess_image_mdfi_weighted,
    preprocess_mask,
    preprocess_image_hsi,
    derive_fov_mask_path_from_image,
)

"""
Retinal vessel segmentation dataset.

Pipeline per sample:
    1) preprocess_image_retina(..., apply_fov=False) -> (1,H,W) float in [0,1]
    2) FOV-after: derive FOV path -> preprocess_mask -> multiply into image
    3) preprocess_mask(label_path) -> (1,H,W) {0,1}
    4) Optional Albumentations augs (expects HxW arrays), returns tensors
    5) Return dict with:
        image: torch.FloatTensor [1,H,W] in [0,1]
        mask:  torch.FloatTensor [1,H,W] in {0,1}
        image_path, label_path: str
"""

class FundusSegDataset(Dataset):

    def __init__(
        self,
        pairs: List[Tuple[str, str]],  # list of (image_path, label_path) strings (from prepare step)
        image_size: int = 512,         # square output size (preprocessing resizes/pads to this)
        augs=None,                     # albumentations Compose or None
        strict_fov: bool = True,       # if True, raise if the expected FOV file is missing; if False, skip FOV gating when absent
        
        # preprocessing values
        use_color_space: str = 'RGB', 
        use_gamma: bool = True,        
        gamma: float = 0.9,
        clahe_clip: float = 2.0,
        clahe_tiles: int = 8,
        patch_mode: bool = False,
        patch_size: int = 512,
        vessel_bias_p: float = 0.6,   # chance to center crop on vessel pixels
        dense_bias_p: float = None,
        min_percent_vessel: float = 0.01,  # min vessel pixels as percent of patch area; if not met, resample uniformly
        virtual_mult: int = 100,
        weights_rgb: tuple[float, float, float] = (0.2793, 0.7041, 0.0166),
        ):

        self.use_color_space = use_color_space
        
        # save config/inputs on the instance for later use in __getitem__
        self.pairs = pairs
        self.size = image_size
        self.augs = augs
        self.strict_fov = strict_fov
        self.patch_mode = patch_mode
        self.patch_size = patch_size
        self.vessel_bias_p = vessel_bias_p
        self.dense_bias_p = dense_bias_p
        self.min_percent_vessel = min_percent_vessel
        self.virtual_mult = virtual_mult
        self.weights_rgb = weights_rgb

        # configuration bundle for preprocess_image_retina
        self._pre_kw = dict(
            target_size=image_size,
            use_gamma=use_gamma,
            gamma=gamma,
            clahe_clip=clahe_clip,
            clahe_tiles=clahe_tiles,
        )

    # tells PyTorch how many samples the dataset has (for indexing, batching)
    def __len__(self) -> int:
        base = len(self.pairs)
        return base * self.virtual_mult if self.patch_mode else base

    # ----- internal helpers -----
    def _sample_center_uniform(self, fov_t, pad):
        H, W = fov_t.shape[-2:]
        # try up to N times to land inside FOV
        if H - pad <= pad or W - pad <= pad:
            return H // 2, W // 2
        
        for _ in range(64):
            y = torch.randint(pad, H - pad, (1,)).item()
            x = torch.randint(pad, W - pad, (1,)).item()
            if fov_t[0, y, x] > 0.5: 
                return y, x
        return H // 2, W // 2  # fallback

    def _sample_center_vessel(self, msk_t):
        ys, xs = (msk_t[0] > 0.5).nonzero(as_tuple=True)
        if len(ys) == 0:
            return None
        i = torch.randint(0, len(ys), (1,)).item()
        return ys[i].item(), xs[i].item()
    
    def _sample_center_dense(self, msk_t, fov_t, pad, ksize: int = None, topk_frac: float = 0.10):
        """
        Pick a center from high vessel-density pixels, constrained to FOV and valid crop margins.
        msk_t, fov_t: [1,H,W] float {0,1}
        ksize: window to measure density (defaults to ~patch_size/4, must be odd)
        topk_frac: sample from the top fraction of dense pixels to keep variety
        """
        _, H, W = msk_t.shape
        ps = self.patch_size
        if ksize is None:
            ksize = max(3, (ps // 4) | 1)  # odd kernel ~ ps/4

        # local mean vessel density with SAME padding
        dens = F.avg_pool2d(msk_t, kernel_size=ksize, stride=1, padding=ksize//2)  # [1,H,W]

        # mask out invalid centers: enforce FOV & crop margins
        valid = (fov_t > 0.5).float()
        margin = torch.zeros_like(valid)
        margin[:, pad:H-pad, pad:W-pad] = 1.0
        valid = valid * margin

        dens = dens * valid  # zero out invalid

        flat = dens.view(-1)
        nz = (flat > 0).nonzero(as_tuple=False).squeeze(-1)
        if nz.numel() == 0:
            return self._sample_center_uniform(fov_t, pad)  # fallback

        # choose from top-k% densest pixels
        k = max(1, int(topk_frac * nz.numel()))
        topk_vals, topk_idx = torch.topk(flat[nz], k, largest=True)
        # multinomial over the top-k for variety
        probs = (topk_vals / (topk_vals.sum() + 1e-8)).clamp_min(1e-8)
        pick = torch.multinomial(probs, 1).item()
        lin = nz[topk_idx[pick]].item()

        cy, cx = divmod(lin, W)
        return int(cy), int(cx)

    # load a single fundus image and preprocess it into a normalized, square array
    def _load_image_hw(self, img_path: str) -> np.ndarray:
        if   self.use_color_space == "RGB":
            x = preprocess_image_retina(img_path, **self._pre_kw)
        elif self.use_color_space == "HSI_I":
            x = preprocess_image_intensity_hsi(img_path, **self._pre_kw)
        elif self.use_color_space == "WRGB":
            x = preprocess_image_mdfi_weighted(img_path, weights_rgb=self.weights_rgb, **self._pre_kw)
        elif self.use_color_space == "HSI":
            x = preprocess_image_hsi(img_path, **self._pre_kw)
        else:
            raise ValueError(f"Unknown color_space: {self.use_color_space}")
        return x[0]  # removes the dummy channel dimension → final shape (H, W)


    # load the ground-truth vessel mask (binary label)
    def _load_label_hw(self, lab_path: str) -> np.ndarray:
        y = preprocess_mask(lab_path, target_size=self.size)   # (1,H,W)
        return y[0]  # (H,W)

    # apply the field-of-view (FOV) mask to blank out pixels outside the circular retina region
    def _apply_fov_after(self, img_hw: np.ndarray, img_path: str) -> np.ndarray:
        fov_path = derive_fov_mask_path_from_image(img_path) # figures out where the corresponding FOV mask should be
        fov_file = Path(fov_path)
        if not fov_file.exists():
            if self.strict_fov:
                raise FileNotFoundError(f"[FundusSegDataset] Missing FOV mask: {fov_file}")
            return img_hw
        
        fov = preprocess_mask(str(fov_file), target_size=self.size)[0]  # loads the FOV mask into (H, W), resize as target size
        return (img_hw * fov).astype(np.float32)  # zeroes out pixels outside FOV + change type 

    
    
    
# ----- main access -----
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.patch_mode:
            base_idx = torch.randint(0, len(self.pairs), (1,)).item()
        else:
            base_idx = idx
        img_path, lab_path = self.pairs[base_idx]    # picks the (image_path, label_path) pair at index idx
        
        img_hw = self._load_image_hw(img_path)          # load + preprocess image (no FOV yet)
        msk_hw = self._load_label_hw(lab_path)          # load vessel mask

        # Load FOV mask but do NOT apply yet (we’ll apply AFTER augs so background stays zero)
        fov_path = derive_fov_mask_path_from_image(img_path)
        fov_file = Path(fov_path)
        if not fov_file.exists():
            if self.strict_fov:
                raise FileNotFoundError(f"[FundusSegDataset] Missing FOV mask: {fov_file}")
            fov_hw = np.ones_like(img_hw, dtype=np.float32)  # fallback = no clamping
        else:
            fov_hw = preprocess_mask(str(fov_file), target_size=self.size)[0]  # (H,W) {0,1}

        if self.augs is not None:
            # run augmentation; pass fov so image/mask/fov share identical geometry
            out = self.augs(image=img_hw, mask=msk_hw, fov=fov_hw)
            img_t = out["image"]  # result of augmentation is torch [1,H,W]
            msk_t = out["mask"]
            fov_t = out["fov"]
            if msk_t.ndim == 2:  # if mask comes out as [H,W] then add channel dimension so it matches the image ([1,H,W])
                msk_t = msk_t.unsqueeze(0)
            if fov_t.ndim == 2:
                fov_t = fov_t.unsqueeze(0)
        else:  # directly converts numpy image to a PyTorch tensor
            img_t = torch.from_numpy(img_hw).unsqueeze(0).float()
            msk_t = torch.from_numpy(msk_hw).unsqueeze(0).float()
            fov_t = torch.from_numpy(fov_hw).unsqueeze(0).float()

        # Apply FOV AFTER augs so background is clamped to zero even after photometric ops
        img_t = img_t * fov_t


        # thresholds the mask/FOV to guarantee binary {0,1} values
        msk_t = (msk_t > 0.5).float()
        fov_t = (fov_t > 0.5).float()

        img_full, msk_full, fov_full = img_t, msk_t, fov_t
        
        if self.patch_mode:
            ps  = self.patch_size
            pad = ps // 2
            H, W = img_t.shape[-2:]

            # ---- mixture probs (dense is optional via None) ----
            p_dense  = self.dense_bias_p if self.dense_bias_p is not None else 0.0
            p_vessel = float(self.vessel_bias_p)
            # keep them sane
            p_dense  = max(0.0, min(1.0, p_dense))
            p_vessel = max(0.0, min(1.0 - p_dense, p_vessel))
            p_uniform = 1.0 - (p_dense + p_vessel)

            r = torch.rand(1).item()
            intended_vessel = False

            if r < p_dense:
                # requires you added _sample_center_dense; otherwise fallback to uniform
                if hasattr(self, "_sample_center_dense"):
                    cy, cx = self._sample_center_dense(msk_t, fov_t, pad)
                else:
                    cy, cx = self._sample_center_uniform(fov_t, pad)
                intended_vessel = True
            elif r < p_dense + p_vessel:
                c = self._sample_center_vessel(msk_t)
                if c is None:
                    cy, cx = self._sample_center_uniform(fov_t, pad)
                else:
                    cy, cx = c
                    intended_vessel = True
            else:
                cy, cx = self._sample_center_uniform(fov_t, pad)

            # ---- clamp & crop ----
            cy = max(pad, min(H - pad, cy))
            cx = max(pad, min(W - pad, cx))
            y0, y1 = cy - pad, cy + pad
            x0, x1 = cx - pad, cx + pad

            img_t = img_full[:, y0:y1, x0:x1]
            msk_t = msk_full[:, y0:y1, x0:x1]
            fov_t = fov_full[:, y0:y1, x0:x1]

            # ---- one-time fallback if intended vessel patch is too empty ----
            if intended_vessel and (msk_t > 0.5).sum().item() < (self.min_percent_vessel * (ps * ps)):
                cy, cx = self._sample_center_uniform(fov_t, pad)
                cy = max(pad, min(H - pad, cy)); cx = max(pad, min(W - pad, cx))
                y0, y1 = cy - pad, cy + pad; x0, x1 = cx - pad, cx + pad
                img_t = img_full[:, y0:y1, x0:x1]
                msk_t = msk_full[:, y0:y1, x0:x1]
                fov_t = fov_full[:, y0:y1, x0:x1]

        return {  # returns a structured dict for one training sample
            "image": img_t,   # preprocessed retina (float tensor [1,H,W])
            "mask": msk_t,    # preprocessed vessel segmentation ground truth (binary [1,H,W])
            "fov": fov_t,     # preprocessed FOV mask (binary [1,H,W])
            "image_path": img_path,
            "label_path": lab_path,
        }


