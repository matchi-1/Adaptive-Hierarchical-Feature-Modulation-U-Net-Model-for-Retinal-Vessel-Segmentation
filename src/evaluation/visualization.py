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

    Grid layout: [Original | Preprocessed | Ground Truth | Predicted].

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

    n_cols = 4
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
            if path and os.path.exists(path):
                original = _read_original_rgb(path)
            else:
                original = _to_hw_numpy_01(img_1hw)

            # 2. Preprocessed
            pre = _to_hw_numpy_01(img_1hw)

            # 3. Ground Truth
            gt = _to_hw_numpy_01(msk_1hw)

            # 4. Model Prediction
            prob, pred = _predict_single(model, img_1hw, device=device, threshold=threshold)
            if clamp_pred_with_fov and fov_1hw is not None:
                fov_np = _to_hw_numpy_01(fov_1hw)
                pred = pred * fov_np  # Restrict prediction to field of view

            row_imgs = [original, pre, gt, pred]
            titles = ["Original", "Preprocessed", "Ground Truth", f"Predicted (≥{threshold:.2f})"]

            for c in range(n_cols):
                ax = axes[rows_done, c]
                ax.imshow(row_imgs[c], cmap="gray", vmin=0.0, vmax=1.0)
                ax.set_title(titles[c], fontsize=10)
                ax.axis("off")

            rows_done += 1

    plt.tight_layout()
    plt.show()
