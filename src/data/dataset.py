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

class FundusSegDataset(Dataset):
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

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        image_size: int = 512,
        augs=None,
        strict_fov: bool = True,
        use_gamma: bool = True,
        gamma: float = 0.9,
        clahe_clip: float = 2.0,
        clahe_tiles: int = 8,
    ):
        self.pairs = pairs
        self.size = image_size
        self.augs = augs
        self.strict_fov = strict_fov

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

    def __len__(self) -> int:
        return len(self.pairs)

    # ----- internal helpers -----
    def _load_image_hw(self, img_path: str) -> np.ndarray:
        x = preprocess_image_retina(img_path, **self._pre_kw)  # (1,H,W)
        return x[0]  # (H,W)

    def _load_label_hw(self, lab_path: str) -> np.ndarray:
        y = preprocess_mask(lab_path, target_size=self.size)   # (1,H,W)
        return y[0]  # (H,W)

    def _apply_fov_after(self, img_hw: np.ndarray, img_path: str) -> np.ndarray:
        fov_path = derive_fov_mask_path_from_image(img_path)
        fov_file = Path(fov_path)
        if not fov_file.exists():
            if self.strict_fov:
                raise FileNotFoundError(f"[FundusSegDataset] Missing FOV mask: {fov_file}")
            return img_hw
        fov = preprocess_mask(str(fov_file), target_size=self.size)[0]  # (H,W)
        return (img_hw * fov).astype(np.float32)

    # ----- main access -----
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, lab_path = self.pairs[idx]

        img_hw = self._load_image_hw(img_path)
        img_hw = self._apply_fov_after(img_hw, img_path)

        msk_hw = self._load_label_hw(lab_path)

        if self.augs is not None:
            out = self.augs(image=img_hw, mask=msk_hw)
            img_t = out["image"]  # torch [1,H,W]
            msk_t = out["mask"]
            if msk_t.ndim == 2:
                msk_t = msk_t.unsqueeze(0)
        else:
            img_t = torch.from_numpy(img_hw).unsqueeze(0).float()
            msk_t = torch.from_numpy(msk_hw).unsqueeze(0).float()

        msk_t = (msk_t > 0.5).float()

        return {
            "image": img_t,
            "mask": msk_t,
            "image_path": img_path,
            "label_path": lab_path,
        }
