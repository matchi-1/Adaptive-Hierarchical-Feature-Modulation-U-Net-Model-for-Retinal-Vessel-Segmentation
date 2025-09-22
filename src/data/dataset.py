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
    ):
        
        # save config/inputs on the instance for later use in __getitem__
        self.pairs = pairs
        self.size = image_size
        self.augs = augs
        self.strict_fov = strict_fov

        # configuration bundle for preprocess_image_retina
        self._pre_kw = dict(
            target_size=image_size,
            use_gamma=use_gamma,
            gamma=gamma,
            clahe_clip=clahe_clip,
            clahe_tiles=clahe_tiles,
            apply_fov=False,          # IMPORTANT: FOV is applied AFTER preprocessing
            mask_path=None,
            auto_discover_mask=False,
        )

    # tells PyTorch how many samples the dataset has (for indexing, batching)
    def __len__(self) -> int:
        return len(self.pairs)

    # ----- internal helpers -----
    
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
# def __getitem__(self, idx: int) -> Dict[str, Any]:
#     img_path, lab_path = self.pairs[idx]    # picks the (image_path, label_path) pair at index idx

#     img_hw = self._load_image_hw(img_path)          # load + preprocess image (no FOV yet)
#     msk_hw = self._load_label_hw(lab_path)          # load vessel mask

#     # Load FOV mask but do NOT apply yet (we’ll apply AFTER augs so background stays zero)
#     fov_path = derive_fov_mask_path_from_image(img_path)
#     fov_file = Path(fov_path)
#     if not fov_file.exists():
#         if self.strict_fov:
#             raise FileNotFoundError(f"[FundusSegDataset] Missing FOV mask: {fov_file}")
#         fov_hw = np.ones_like(img_hw, dtype=np.float32)  # fallback = no clamping
#     else:
#         fov_hw = preprocess_mask(str(fov_file), target_size=self.size)[0]  # (H,W) {0,1}

#     if self.augs is not None:
#         # run augmentation; pass fov so image/mask/fov share identical geometry
#         out = self.augs(image=img_hw, mask=msk_hw, fov=fov_hw)
#         img_t = out["image"]  # result of augmentation is torch [1,H,W]
#         msk_t = out["mask"]
#         fov_t = out["fov"]
#         if msk_t.ndim == 2:  # if mask comes out as [H,W] then add channel dimension so it matches the image ([1,H,W])
#             msk_t = msk_t.unsqueeze(0)
#         if fov_t.ndim == 2:
#             fov_t = fov_t.unsqueeze(0)
#     else:  # directly converts numpy image to a PyTorch tensor
#         img_t = torch.from_numpy(img_hw).unsqueeze(0).float()
#         msk_t = torch.from_numpy(msk_hw).unsqueeze(0).float()
#         fov_t = torch.from_numpy(fov_hw).unsqueeze(0).float()

#     # Apply FOV AFTER augs so background is clamped to zero even after photometric ops
#     img_t = img_t * fov_t

#     # thresholds the mask/FOV to guarantee binary {0,1} values
#     msk_t = (msk_t > 0.5).float()
#     fov_t = (fov_t > 0.5).float()

#     return {  # returns a structured dict for one training sample
#         "image": img_t,   # preprocessed retina (float tensor [1,H,W])
#         "mask": msk_t,    # preprocessed vessel segmentation ground truth (binary [1,H,W])
#         "fov": fov_t,     # preprocessed FOV mask (binary [1,H,W])
#         "image_path": img_path,
#         "label_path": lab_path,
#     }


    # ----- main access -----
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, lab_path = self.pairs[idx]    # (image_path, label_path)

        # 1) preprocess image/mask (no FOV yet)
        img_hw = self._load_image_hw(img_path)          # (H,W) float [0,1]
        msk_hw = self._load_label_hw(lab_path)          # (H,W) {0,1}

        # 2) load FOV mask but DO NOT apply yet (we'll clamp after augs)
        fov_path = derive_fov_mask_path_from_image(img_path)
        fov_file = Path(fov_path)
        if not fov_file.exists():
            if self.strict_fov:
                raise FileNotFoundError(f"[FundusSegDataset] Missing FOV mask: {fov_file}")
            fov_hw = np.ones_like(img_hw, dtype=np.float32)  # fallback = no clamping
        else:
            fov_hw = preprocess_mask(str(fov_file), target_size=self.size)[0]  # (H,W) {0,1}

        # 3) augs: transform image/mask/FOV with identical geometry
        if self.augs is not None:
            out = self.augs(image=img_hw, mask=msk_hw, fov=fov_hw)
            img_t = out["image"]               # torch [1,H,W]
            msk_t = out["mask"]                # torch [1,H,W] or [H,W]
            fov_t = out["fov"]                 # torch [1,H,W] or [H,W]
            if msk_t.ndim == 2: msk_t = msk_t.unsqueeze(0)
            if fov_t.ndim == 2: fov_t = fov_t.unsqueeze(0)
        else:
            img_t = torch.from_numpy(img_hw).unsqueeze(0).float()
            msk_t = torch.from_numpy(msk_hw).unsqueeze(0).float()
            fov_t = torch.from_numpy(fov_hw).unsqueeze(0).float()

        # 4) apply FOV AFTER augs to zero-out background
        img_t = img_t * fov_t

        # 5) hard-binarize masks
        msk_t = (msk_t > 0.5).float()
        fov_t = (fov_t > 0.5).float()

        return {
            "image": img_t,
            "mask":  msk_t,
            "fov":   fov_t,          # <-- REQUIRED so your notebook can access batch["fov"]
            "image_path": img_path,
            "label_path": lab_path,
        }

