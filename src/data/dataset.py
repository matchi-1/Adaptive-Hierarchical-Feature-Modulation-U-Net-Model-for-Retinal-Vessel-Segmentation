# src/data/dataset.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import (
    preprocess_image_retina,
    preprocess_mask,
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
        use_gamma: bool = True,        
        gamma: float = 0.9,
        clahe_clip: float = 2.0,
        clahe_tiles: int = 8,
        patch_mode: bool = False,
        patch_size: int = 512,
        vessel_bias_p: float = 0.6,   # chance to center crop on vessel pixels
        min_vessel_px: int = 64,
        virtual_mult: int = 100
        ):
        
        # save config/inputs on the instance for later use in __getitem__
        self.pairs = pairs
        self.size = image_size
        self.augs = augs
        self.strict_fov = strict_fov
        self.patch_mode = patch_mode
        self.patch_size = patch_size
        self.vessel_bias_p = vessel_bias_p
        self.min_vessel_px = min_vessel_px
        self.virtual_mult = virtual_mult

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

    # load a single fundus image and preprocess it into a normalized, square array
    def _load_image_hw(self, img_path: str) -> np.ndarray:
        x = preprocess_image_retina(img_path, **self._pre_kw)  # read image from disk, preprocess, return (1,H,W)
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
        img_path, lab_path = self.pairs[idx]    # picks the (image_path, label_path) pair at index idx
        
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

        if self.patch_mode:
            ps = self.patch_size
            pad = ps // 2
            H, W = img_t.shape[-2:]

            # choose a center (vessel-biased with probability p)
            use_vessel = (torch.rand(1).item() < self.vessel_bias_p)
            c = self._sample_center_vessel(msk_t) if use_vessel else None
            if c is None:
                cy, cx = self._sample_center_uniform(fov_t, pad)
            else:
                cy, cx = c

            cy = max(pad, min(H - pad, cy))
            cx = max(pad, min(W - pad, cx))
            y0, y1 = cy - pad, cy + pad
            x0, x1 = cx - pad, cx + pad

            # slice patch (C,H,W)
            img_t = img_t[:, y0:y1, x0:x1]
            msk_t = msk_t[:, y0:y1, x0:x1]
            fov_t = fov_t[:, y0:y1, x0:x1]

            # optional: if we intended a vessel patch but it's empty, resample once uniformly
            if use_vessel and (msk_t > 0.5).sum().item() < self.min_vessel_px:
                cy, cx = self._sample_center_uniform(fov_t, pad)
                cy = max(pad, min(H - pad, cy)); cx = max(pad, min(W - pad, cx))
                y0, y1 = cy - pad, cy + pad; x0, x1 = cx - pad, cx + pad
                img_t = img_full[:, y0:y1, x0:x1] if 'img_full' in locals() else img_t

        return {  # returns a structured dict for one training sample
            "image": img_t,   # preprocessed retina (float tensor [1,H,W])
            "mask": msk_t,    # preprocessed vessel segmentation ground truth (binary [1,H,W])
            "fov": fov_t,     # preprocessed FOV mask (binary [1,H,W])
            "image_path": img_path,
            "label_path": lab_path,
        }


