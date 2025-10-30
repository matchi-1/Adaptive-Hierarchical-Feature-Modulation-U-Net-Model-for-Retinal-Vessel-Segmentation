# src/data/dataloader.py
from __future__ import annotations
import re
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from src.data.dataset import FundusSegDataset
from src.data.prepare_dataset import _natural_key


def seed_worker(worker_id: int):
    # PyTorch provides each worker a base seed via worker_info.seed
    worker_info = torch.utils.data.get_worker_info()
    # Make a NumPy/py random seed from it
    base_seed = worker_info.seed % 2**32
    np.random.seed(base_seed)
    try:
        import random
        random.seed(base_seed)
    except Exception:
        pass


def pair_paths(images_dir: str, labels_dir: str) -> List[Tuple[str, str]]:
    """
    Pair files in `images_dir` and `labels_dir` by natural sort order.
    Assumes aligned indexing between images and 1st_manual.
    """
    img_dir = Path(images_dir)
    lab_dir = Path(labels_dir)

    # sort based on numbering in files
    imgs = sorted([p for p in img_dir.glob("*") if p.is_file()], key=_natural_key)
    labs = sorted([p for p in lab_dir.glob("*") if p.is_file()], key=_natural_key)

    # handles mismatched counts by taking the smaller number
    n = min(len(imgs), len(labs))

    # returns list like [("../images/01.png", "../labels/01_manual.png"), ...]
    return [(str(imgs[i]), str(labs[i])) for i in range(n)]


# takes in a base random see + returns an inner function (_fn) that PyTorch’s DataLoader can call when each worker process starts
# in DataLoader, "workers" are multiple worker processes (subprocesses) that loads and preprocesses data in parallel and package them and send them back to the main processes

def _worker_seed_init(base_seed: int):
    def _fn(worker_id: int):  # every worker process (like worker 0, worker 1, …) gets its own worker_id
        np.random.seed(base_seed + worker_id) # seed is set as base_seed + worker_id
    return _fn


def make_loaders(
    train_pairs: List[Tuple[str, str]],            # list of (image_path, label_path) pairs for training
    val_pairs: Optional[List[Tuple[str, str]]] = None,  # optional validation pairs; if None, auto-split from train_pairs
    image_size: int = 512,                        # resize images/masks to this square size (H=W=image_size)
    batch_size: int = 4,                          # how many samples per batch to load
    num_workers: int = 0,                         # number of subprocesses to use for data loading (0 = main process)
    seed: int = 1337,                             # random seed for reproducibility (splits, augmentations)
    strict_fov: bool = True,                      # if True, raise error if FOV mask is missing; else skip applying
    augs_train=None,                              # Albumentations Compose object with training augmentations
    augs_val=None,                                # Albumentations Compose object with validation preprocessing

    # preprocessing 
    use_gamma: bool = True,        
    gamma: float = 0.9,
    clahe_clip: float = 2.0,
    clahe_tiles: int = 8,

    patch_train: bool = False,
    patch_size: int = 512,
    vessel_bias_p: float = 0.6,
    min_percent_vessel: float = 0.1,
    virtual_mult: int = 100,
):

    """
    Build PyTorch DataLoaders.
    - If val_pairs is None: make an 80/20 split from train_pairs (deterministic via seed).
    - Seeds numpy per worker for reproducible augmentations.
    """

    # create a torch random generator and seed it
    # this generator is passed to random_split so your 80/20 split is deterministic across runs
    g = torch.Generator().manual_seed(seed)

    if val_pairs is None: # if no explicit validation set is provided then make own validation set from training
        n_total = len(train_pairs)  # count how many (image, label) pairs we have in total for training
        n_val = max(1, int(round(0.2 * n_total)))  # compute how many samples go to validation = 20% of total | max makes sure there is atleast 1 in val set
        n_trn = n_total - n_val # rest goes to training

        # temporary dataset for deterministic split
        tmp = FundusSegDataset(train_pairs, image_size=image_size, augs=None, strict_fov=strict_fov)  # make dataset for splitting; no augs yet
        
        # trn_ds = training dataset subset
        # val_ds_idx = validation subset
        trn_ds, val_ds_idx = random_split(tmp, [n_trn, n_val], generator=g) 

        # map back the dataset indices into the original filepaths
        trn_pairs = [train_pairs[i] for i in trn_ds.indices]
        val_pairs = [train_pairs[i] for i in val_ds_idx.indices]

    else: # if val_pairs was provided, no splitting is needed
        trn_pairs = train_pairs

    # build final datasets with augmentations
    trn = FundusSegDataset(
        trn_pairs,
        image_size=image_size,
        augs=augs_train,
        strict_fov=strict_fov,
        use_gamma = use_gamma,        
        gamma = gamma,
        clahe_clip = clahe_clip,
        clahe_tiles = clahe_tiles,
        patch_mode=patch_train, 
        patch_size=patch_size,
        vessel_bias_p=vessel_bias_p, 
        min_percent_vessel=min_percent_vessel, 
        virtual_mult=virtual_mult
    )
    val = FundusSegDataset(
        val_pairs,
        image_size=image_size,
        augs=augs_val,
        strict_fov=strict_fov,
        use_gamma = use_gamma,        
        gamma = gamma,
        clahe_clip = clahe_clip,
        clahe_tiles = clahe_tiles,
        patch_mode=False  # keep full images for validation
    )

    # call helper func to create seed for each worker process
    worker_init = seed_worker  # _worker_seed_init

    # build dataloaders
    train_loader = DataLoader(
        trn,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init
    )
    val_loader = DataLoader(
        val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init
    )

    # returns training and validation dataloaders
    return train_loader, val_loader

    # can now be used like:
        # for batch in train_loader:
        #     imgs, masks = batch["image"], batch["mask"]
