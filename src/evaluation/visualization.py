"""
Visualization utilities for retinal vessel segmentation.

This module provides functions to:
- Load and normalize images.
- Run a single model prediction on an image.
- Visualize samples in a 4-column grid:
    [Original | Preprocessed | Ground Truth | Predicted].
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path
import os, re

def _read_original_rgb(path: str) -> np.ndarray:
    """
    Load an image from disk as RGB float32 in range [0,1].

    Args:
        path (str): Path to the image file.

    Returns:
        np.ndarray: Array of shape (H,W,3) with values in [0,1].
    """
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    if arr.max() > 1.0:  # Normalize if stored as 0–255
        arr = arr / 255.0
    return arr

def _read_mask_gray_01(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32)
    if arr.max() > 1.0:
        arr /= 255.0
    return arr

def _derive_drive_manual2_from_image_path(image_path: str,
                                          *,
                                          try_exts=(".png", ".jpg")) -> Optional[str]:
    """
    Given .../DRIVE/test/images/01_test.png → .../DRIVE/test/2nd_manual/01_manual2.<ext>
    Tries multiple extensions and returns the first path that actually exists, else None.
    """
    try:
        p = Path(image_path)
        # p.stem is '01_test' for '01_test.png'
        m = re.match(r"(\d+)_test$", p.stem)
        if not m:
            return None

        img_num = m.group(1)
        manual_dir = p.parent.parent / "2nd_manual"

        # 1) Try common lowercase & uppercase extensions explicitly
        cand_exts = list(try_exts) + [ext.upper() for ext in try_exts]
        for ext in cand_exts:
            cand = manual_dir / f"{img_num}_manual2{ext}"
            if cand.exists():
                return str(cand)

        # 2) Fallback: glob any extension matching the base pattern
        matches = sorted(manual_dir.glob(f"{img_num}_manual2.*"))
        if matches:
            return str(matches[0])

        return None
    except Exception:
        return None


def _to_hw_numpy_01(t: torch.Tensor) -> np.ndarray:
    """
    Convert a tensor into a numpy 2D array [H,W] in range [0,1].

    Handles tensors with extra batch/channel dimensions and rescales
    values if outside [0,1].
    """
    if t.ndim == 3 and t.shape[0] == 1:  # [1,H,W] → [H,W]
        t = t[0]

    # Move tensor to CPU, remove gradients, cast to float, then convert to NumPy array
    arr = t.detach().cpu().float().numpy()
    a_min, a_max = float(arr.min()), float(arr.max())
    if a_min < 0.0 or a_max > 1.0:
        # Rescale to [0,1]
        arr = (arr - a_min) / (a_max - a_min) if a_max > a_min else np.zeros_like(arr)
    return arr


@torch.no_grad() # Disable gradient tracking (saves memory and speeds up inference)
def _predict_single(model, img_1hw, device="cpu", threshold=0.5):
    """
    Run model prediction on a single 1xHxW image.

    Args:
        model: Segmentation model (PyTorch).
        img_1hw (torch.Tensor): Single-channel tensor [1,H,W].
        device (str): Device ('cpu' or 'cuda').
        threshold (float): Threshold for binary prediction.

    Returns:
        tuple:
            prob_np (np.ndarray): Probability map [H,W].
            bin_np (np.ndarray): Binary mask (thresholded).
    """
    model.eval()
    x = img_1hw.unsqueeze(0).to(device)  # Add batch → [1,1,H,W]
    logits = model(x)

    if isinstance(logits, (list, tuple)):
        logits = logits[-1]  # Use last output if multiple

    # Handle different output shapes
    if logits.ndim == 4:
        if logits.shape[1] == 2:  # Two-class output
            prob = torch.softmax(logits, dim=1)[:, 1, ...][0]
        elif logits.shape[1] == 1:  # Single channel logits
            prob = torch.sigmoid(logits[:, 0, ...])[0]
        else:  # Multi-class
            prob = torch.softmax(logits, dim=1).max(dim=1).values[0]
    elif logits.ndim == 3:  # [C,H,W]
        prob = torch.sigmoid(logits[0])
    else:
        raise ValueError(f"Unexpected model output shape: {tuple(logits.shape)}")

    prob_np = _to_hw_numpy_01(prob)
    bin_np = (prob_np >= threshold).astype(np.float32)
    return prob_np, bin_np


def visualize_samples(
    model,
    dataloader,
    n_rows=8,
    device=None,
    threshold=0.5,
    clamp_pred_with_fov=True,
    figsize_per_row=(12, 3),
):
    """
    Show a grid of model predictions compared to original and ground truth.

    Grid layout: [Original | Preprocessed | Ground Truth 1 | Ground Truth 1 | Predicted].

    Args:
        model: Trained segmentation model.
        dataloader: DataLoader yielding dicts with keys "image", "mask",
                    and optionally "fov", "image_path".
        n_rows (int): Number of rows to display.
        device (str): Torch device; defaults to GPU if available.
        threshold (float): Threshold for binary predictions.
        clamp_pred_with_fov (bool): Multiply prediction by FOV mask if available.
        figsize_per_row (tuple): Width, height of each row in figure.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    n_cols = 5  # Original | Preprocessed | Ground Truth 1 | Ground Truth 2 | Predicted
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(figsize_per_row[0], figsize_per_row[1] * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    rows_done = 0
    for batch in dataloader:
        if rows_done >= n_rows:
            break

        imgs = batch["image"]     # Preprocessed images
        msks = batch["mask"]      # Ground truth masks
        fovs = batch.get("fov", None)   # FOV
        paths = batch.get("image_path", None)

        for b in range(imgs.shape[0]):
            # Loop over all images in the current batch
            if rows_done >= n_rows:
                # Stop if we already filled the requested number of rows
                break

            img_1hw = imgs[b]
            msk_1hw = msks[b]
            fov_1hw = fovs[b] if fovs is not None else None
            path = paths[b] if isinstance(paths, (list, tuple)) and len(paths) > b else None

            # 1. Original (from path if available, else preprocessed)
            original = _read_original_rgb(path) if (path and os.path.exists(path)) else _to_hw_numpy_01(img_1hw)

            # 2. Preprocessed
            pre = _to_hw_numpy_01(img_1hw)

            # 3. Ground Truth 1 (from dataloader mask)
            gt1 = _to_hw_numpy_01(msk_1hw)

            # 4. Ground Truth 2 (derived from image_path)
            gt2_path = _derive_drive_manual2_from_image_path(path) if path else None
            gt2 = _read_mask_gray_01(gt2_path) if (gt2_path and os.path.exists(gt2_path)) else np.zeros_like(gt1)

            # 5. Predicted
            prob, pred = _predict_single(model, img_1hw, device=device, threshold=threshold)
            if clamp_pred_with_fov and fov_1hw is not None:
                fov_np = _to_hw_numpy_01(fov_1hw)
                pred = pred * fov_np

            row_imgs = [original, pre, gt1, gt2, pred]
            titles   = ["Original", "Preprocessed", "Ground Truth 1", "Ground Truth 2",
                        f"Predicted (≥{threshold:.2f})"]

            for c in range(n_cols):
                ax = axes[rows_done, c]
                ax.imshow(row_imgs[c], cmap="gray", vmin=0.0, vmax=1.0)
                ax.set_title(titles[c], fontsize=10)
                ax.axis("off")

            rows_done += 1

    plt.tight_layout()
    plt.show()



@dataclass
class ModelEntry:
    name: str
    model: torch.nn.Module
    threshold: float = 0.5             # for binary masks
    device: Optional[str] = None       # override per model if needed

def build_loaded(model_cls, pth, *, device, **kwargs):
    m = model_cls(**kwargs)
    sd = torch.load(pth, map_location="cpu")
    sd = sd.get("state_dict", sd)
    if any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    m.load_state_dict(sd, strict=True)
    return m.to(device).eval()

@torch.no_grad()
def visualize_models_from_loader(
    models: List[ModelEntry],
    dataloader,
    n_rows: int = 5,
    *,
    device: Optional[str] = None,
    image_key: str = "image",
    mask_key: str = "mask",
    clamp_pred_with_fov: bool = True,
    figsize_per_row: tuple[float, float] = (14, 3),
):
    """
    Visualize samples from a DataLoader with predictions from multiple models.

    Layout per row:
        [ Original | Preprocessed | Ground Truth 1 | Ground Truth 2 | <Model 1> | <Model 2> | ... ]


    Assumptions:
    - Binary segmentation (out_ch = 1). For multiclass, extend `_predict_single`.
    - If `image` has more than 1 channel, we convert to [1,H,W] for `_predict_single`
      by taking the first channel (or mean) for display and prediction.

    Args:
        models:       List of ModelEntry(name, model, threshold?, device?).
        dataloader:   PyTorch DataLoader yielding dicts with keys described above.
        n_rows:       Number of rows (samples) to display.
        device:       Default device to move models to; defaults to CUDA if available.
        image_key:    Key for the input image tensor in each batch dict.
        mask_key:     Key for the ground-truth mask in each batch dict (optional).
        clamp_pred_with_fov:
                      If True, multiply binary predictions by FOV (if provided)
                      to hide predictions outside the valid region.
        figsize_per_row:
                      Figure size per row as (width, height) in inches.

    Returns:
        None. Shows a matplotlib figure.
    """
    # ---------- Resolve device and prep models ----------
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Materialize models on their own device (if set) or the default device.
    # We do NOT (re)load state_dict here—load them outside for clarity/efficiency.
    ready_models: List[tuple[str, torch.nn.Module, float, str]] = []
    for m in models:
        d = m.device or device
        mdl = m.model.to(d).eval()
        ready_models.append((m.name, mdl, m.threshold, d))

    # ---------- Figure/grid setup ----------
    # 3 base columns: Original | Preprocessed | Ground Truth
    # + one column per model
    n_cols = 4 + len(ready_models)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.0 * n_cols, 3.0 * n_rows),
    )

    if n_rows == 1:
        axes = np.expand_dims(axes, 0)

    fig_w = figsize_per_row[0]
    fig_h = figsize_per_row[1] * n_rows
    
    col_titles = ["Original", "Preprocessed", "Ground Truth 1", "Ground Truth 2"] + [nm for (nm, _, _, _) in ready_models]

    rows_done = 0

    # ---------- Iterate like visualize_samples (per-batch, per-item) ----------
    for batch in dataloader:
        if rows_done >= n_rows:
            break

        imgs = batch[image_key]                          # [B,C,H,W] preprocessed tensors
        msks = batch.get(mask_key, None)                 # [B,1,H,W] or [B,H,W] (optional)
        fovs = batch.get("fov", None)                    # [B,1,H,W] or [B,H,W] (optional)
        paths = batch.get("image_path", None)            # list/tuple of file paths (optional)
        #print("[visualization.py] PATHS: ", paths)

        B = imgs.shape[0]

        for b in range(B):
            if rows_done >= n_rows:
                break

            img_chw = imgs[b]                            # [C,H,W]
            # Make a single-channel view for display & _predict_single:
            # - If C == 1: already [1,H,W]
            # - If C > 1: take the first channel (or use mean if you prefer)
            if img_chw.ndim != 3:
                raise ValueError(f"Expected [C,H,W], got shape {tuple(img_chw.shape)}")
            if img_chw.shape[0] == 1:
                img_1hw = img_chw
            else:
                img_1hw = img_chw[:1, ...]              # or img_chw.mean(0, keepdim=True)

            # Ground Truth 1
            if msks is not None:
                msk = msks[b]
                if msk.ndim == 3 and msk.shape[0] == 1:
                    msk = msk[0]                        # [H,W]
                gt1_np = _to_hw_numpy_01(msk)
            else:
                gt1_np = np.zeros_like(_to_hw_numpy_01(img_1hw))  # empty placeholder

            # Original (prefer reading from path if given)
            path = None
            if isinstance(paths, (list, tuple)) and len(paths) > b:
                path = paths[b]
            if path and os.path.exists(path):
                original_np = _read_original_rgb(path)           # H×W×3 in [0,1]
                # For consistency with grayscale display, show its luminance:
                #if original_np.ndim == 3 and original_np.shape[2] == 3:
                #    original_np = original_np.mean(axis=2)       # convert to H×W
            else:
                original_np = _to_hw_numpy_01(img_1hw)           # fallback to preprocessed view

            # Preprocessed view (what the model roughly sees)
            pre_np = _to_hw_numpy_01(img_1hw)

            # Ground Truth 2 (derive from image_path)
            gt2_path = _derive_drive_manual2_from_image_path(path) if path else None
            gt2_np = _read_mask_gray_01(gt2_path) if (gt2_path and os.path.exists(gt2_path)) else np.zeros_like(gt1_np)

            # For each model, run prediction on the same [1,H,W] tensor
            pred_cols: List[np.ndarray] = []
            for (name, mdl, thr, d) in ready_models:
                # If the model internally expects multi-channel, adapt _predict_single
                # (Here we assume binary with single-channel input; extend as needed.)
                prob_np, bin_np = _predict_single(mdl, img_1hw.to(d), device=d, threshold=thr)

                # Optionally restrict predictions to FOV
                if clamp_pred_with_fov and fovs is not None:
                    fov_ = fovs[b]
                    fov_np = _to_hw_numpy_01(fov_[0] if (fov_.ndim == 3 and fov_.shape[0] == 1) else fov_)
                    bin_np = bin_np * fov_np

                pred_cols.append(bin_np)

            # ---------- Draw current row ----------
            row_imgs = [original_np, pre_np, gt1_np, gt2_np] + pred_cols
            for c in range(n_cols):
                ax = axes[rows_done, c]
                im = row_imgs[c]
                if isinstance(im, np.ndarray) and im.ndim == 2:
                    ax.imshow(im, cmap="gray", vmin=0.0, vmax=1.0)
                else:
                    ax.imshow(im)
                ax.set_title(col_titles[c], fontsize=10, pad=3)
                ax.axis("off")

            rows_done += 1

    plt.tight_layout()
    plt.show()