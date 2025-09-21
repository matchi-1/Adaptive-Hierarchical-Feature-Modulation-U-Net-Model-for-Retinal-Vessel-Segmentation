# src/data/dataloader.py
from __future__ import annotations
import re
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from src.data.dataset import FundusSegDataset

def _natural_key(p: Path):
    s = p.stem.lower()
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]

def pair_paths(images_dir: str, labels_dir: str) -> List[Tuple[str, str]]:
    """
    Pair files in `images_dir` and `labels_dir` by natural sort order.
    Assumes aligned indexing between images and 1st_manual.
    """
    img_dir = Path(images_dir)
    lab_dir = Path(labels_dir)
    imgs = sorted([p for p in img_dir.glob("*") if p.is_file()], key=_natural_key)
    labs = sorted([p for p in lab_dir.glob("*") if p.is_file()], key=_natural_key)
    n = min(len(imgs), len(labs))
    return [(str(imgs[i]), str(labs[i])) for i in range(n)]

def _worker_seed_init(base_seed: int):
    def _fn(worker_id: int):
        np.random.seed(base_seed + worker_id)
    return _fn

def make_loaders(
    train_pairs: List[Tuple[str, str]],
    val_pairs: Optional[List[Tuple[str, str]]] = None,
    image_size: int = 512,
    batch_size: int = 4,
    num_workers: int = 0,
    seed: int = 1337,
    strict_fov: bool = True,
    augs_train=None,
    augs_val=None,
):
    """
    Build PyTorch DataLoaders.
    - If val_pairs is None: make an 80/20 split from train_pairs (deterministic via seed).
    - Seeds numpy per worker for reproducible augmentations.
    """
    g = torch.Generator().manual_seed(seed)

    if val_pairs is None:
        n_total = len(train_pairs)
        n_val = max(1, int(round(0.2 * n_total)))
        n_trn = n_total - n_val
        # temporary dataset for deterministic split
        tmp = FundusSegDataset(train_pairs, image_size=image_size, augs=None, strict_fov=strict_fov)
        trn_ds, val_ds_idx = random_split(tmp, [n_trn, n_val], generator=g)
        trn_pairs = [train_pairs[i] for i in trn_ds.indices]
        val_pairs = [train_pairs[i] for i in val_ds_idx.indices]
    else:
        trn_pairs = train_pairs

    # Build final datasets with augmentations
    trn = FundusSegDataset(
        trn_pairs, image_size=image_size, augs=augs_train, strict_fov=strict_fov
    )
    val = FundusSegDataset(
        val_pairs, image_size=image_size, augs=augs_val, strict_fov=strict_fov
    )

    worker_init = _worker_seed_init(seed)

    train_loader = DataLoader(
        trn, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=False, worker_init_fn=worker_init
    )
    val_loader = DataLoader(
        val, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, drop_last=False, worker_init_fn=worker_init
    )
    return train_loader, val_loader
