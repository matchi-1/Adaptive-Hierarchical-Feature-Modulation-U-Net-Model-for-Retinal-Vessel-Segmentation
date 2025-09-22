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
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, lab_path = self.pairs[idx]    # picks the (image_path, label_path) pair at index idx

        img_hw = self._load_image_hw(img_path)    # load + preprocess image
        img_hw = self._apply_fov_after(img_hw, img_path)   # apply fov

        msk_hw = self._load_label_hw(lab_path)      # load vessel mask

        if self.augs is not None:
            out = self.augs(image=img_hw, mask=msk_hw)  # run augmentation, both image and mask are transformed the same way so they stay aligned
            img_t = out["image"]  # result of augmentation is torch [1,H,W]
            msk_t = out["mask"]
            if msk_t.ndim == 2:  # if mask comes out as [H,W] then add channel dimension so it matches the image ([1,H,W])
                msk_t = msk_t.unsqueeze(0)
        
        else:  # directly converts numpy image to a PyTorch tensor
            img_t = torch.from_numpy(img_hw).unsqueeze(0).float()
            msk_t = torch.from_numpy(msk_hw).unsqueeze(0).float()

        msk_t = (msk_t > 0.5).float()  # thresholds the mask to guarantee binary {0,1} values

        return {  # returns a structured dict for one training sample
            "image": img_t,   # preprocessed retina (float tensor [1,H,W])
            "mask": msk_t,    # preprocessed vessel segmentation ground truth (binary [1,H,W])
            "image_path": img_path,
            "label_path": lab_path,
        }
