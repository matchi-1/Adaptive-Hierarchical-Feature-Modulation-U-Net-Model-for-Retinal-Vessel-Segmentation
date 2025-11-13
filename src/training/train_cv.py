import math
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import sys, os
from src.data.augmentations import get_train_augs, get_val_augs
from src.data.dataloader import make_loaders
from src.data.splits import chase_mc_balanced_splits
from src.evaluation.evaluate import evaluate_and_print
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt



def train_one_epoch(model, loader, optimizer, scaler, loss_fn, device):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)  # [B,1,H,W]
        y = batch["mask"].to(device,  non_blocking=True)  # [B,1,H,W]

        # Optional: mask labels outside FOV if present (keeps loss fair)
        if "fov" in batch:
            fov = batch["fov"].to(device, non_blocking=True)
            y = y * (fov > 0.5).float()

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=(device=="cuda")):
            logits = model(x)          # ← PURE UNET: no fov passed
            loss   = loss_fn(logits, y)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()


        total += loss.item() * x.size(0)
        n     += x.size(0)


    return total / max(1, n)


@torch.no_grad()
def validate_loss_only(model, loader, loss_fn, window, overlap, device, use_fov=True):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)  # [B,1,H,W] (full images)
        y = batch["mask"].to(device,  non_blocking=True)
        fov = batch.get("fov", torch.ones_like(y)).to(device, non_blocking=True)

        for i in range(x.size(0)):
            # stitch logits over full image i
            logits_i = sliding_window_forward_logits(model, x[i:i+1], window=window, overlap=overlap, device=device)
            # restrict loss to FOV (keeps loss fair)
            yi = y[i:i+1]
            if use_fov:
                yi = yi * (fov[i:i+1] > 0.5).float()

            with torch.amp.autocast(device_type="cuda", enabled=(device=="cuda")):
                loss_i = loss_fn(logits_i, yi)

            total += loss_i.item()
            n     += 1
    return total / max(1, n)


def atomic_torch_save(state, path: Path):
    path = Path(path); tmp = path.with_suffix(path.suffix + ".partial")
    try: tmp.unlink()
    except FileNotFoundError: pass
    with open(tmp, "wb") as f:
        torch.save(state, f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def saveTrainVal(epochs_val, fold_id, train_losses, val_losses, save_path, Title):
  fig, ax = plt.subplots(figsize=(6, 3.5))
  ax.plot(epochs_val, train_losses, label="train")
  ax.plot(epochs_val, val_losses,   label="val")
  ax.set_xlabel("epoch")
  ax.set_ylabel("loss")
  ax.set_title(f"Train vs Val Loss (fold {fold_id}): {Title}")
  ax.grid(True)
  ax.legend()

  fig.tight_layout()
  fig.savefig(save_path, dpi=150)   # save first
  plt.show()                      # optional; skip inside loops to avoid blocking
  plt.close(fig)                    # always close to free memory
  print(f"\t TRAIN vs VAL SAVED TO: {save_path}")


# --- DATASET HYPERPARAM SCHEDULERS ---
def sched_vessel_bias(epoch):
    if epoch <= 10: return 0.70
    if epoch <= 35: return 0.60
    return 0.50

def sched_min_percent_vessel(epoch):
    if epoch <= 10: return 0.008
    if epoch <= 35: return 0.010
    return 0.008

def sched_patch_size(epoch):
    return 160 if epoch > 45 else 128

# ---- single updater ----
def update_dataset(ds, epoch):
    ds.vessel_bias_p      = sched_vessel_bias(epoch)
    ds.min_percent_vessel = sched_min_percent_vessel(epoch)
    ds.patch_size         = sched_patch_size(epoch)

@torch.no_grad()
def sliding_window_forward_logits(model, img_1chw, window, overlap, device):
    """
    img_1chw: torch.Tensor [1,1,H,W] on *any* device.
    Returns: averaged logits [1,1,H,W] on 'device'
    """
    model.eval()
    img = img_1chw.to(device, non_blocking=True)
    _, _, H, W = img.shape
    step = max(1, int(window * (1 - overlap)))

    # we'll average *logits* to keep your loss_fn (e.g., BCEWithLogits) unchanged
    acc  = torch.zeros_like(img, device=device)
    norm = torch.zeros_like(img, device=device)

    for y0 in range(0, max(1, H - window + 1), step):
        for x0 in range(0, max(1, W - window + 1), step):
            y1 = min(y0 + window, H); x1 = min(x0 + window, W)
            y0 = y1 - window;         x0 = x1 - window
            tile = img[:, :, y0:y1, x0:x1]                     # [1,1,win,win]
            with torch.amp.autocast(device_type="cuda", enabled=(device=="cuda")):
                logit = model(tile)                            # [1,1,win,win]
            acc[:, :, y0:y1, x0:x1] += logit
            norm[:, :, y0:y1, x0:x1] += 1.0

    logits = acc / torch.clamp_min(norm, 1.0)
    return logits