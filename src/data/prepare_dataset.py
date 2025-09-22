# src/data/prepare_dataset.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Literal, Dict
import re

Split = Literal["training", "test"]

def _natural_key(p: Path):
    s = p.stem.lower() # .stem to get filename without extension ("21_training")
     # splits the stem into chunks, separating digit sequences ("21", "training") 
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)] # for each chunk, convert digit-only pieces
    # returns a list like [21, "_training"] -- will be used as a sort key


def _list_sorted_files(dirpath: Path):
    # Path.glob("*") lists all entries (files + directories) inside dirpath
    # p.is_file filters out everything that isn’t a file. removes directories like "subfolder" so we only keep .png, .jpg
    return sorted([p for p in dirpath.glob("*") if p.is_file()], key=_natural_key) # use natural key to sort by number


def build_pairs_for_split(
    dataset_root: str,           # e.g. "../data/raw/DRIVE"
    split: Split = "training",   # "training" or "test"
    label_folder: str = "1st_manual", # which label folder to pair with images (1st_manual by default)
) -> List[Tuple[str, str]]: # returns a list of (image_path, label_path) pairs
    
    """
    Build (image_path, label_path) pairs for a single dataset (DRIVE/CHASEDB1/STARE) and split.

    Expects structure:
      <root>/<split>/images/*.*
      <root>/<split>/<label_folder>/*.*  (1st_manual by default)
      <root>/<split>/mask/*.*
    Pairs are created by natural sort order (index-aligned filenames).
    """
    root = Path(dataset_root)                    # converts the dataset root into a Path object 
    img_dir = root / split / "images"            # builds the full path to the images folder  "../data/raw/DRIVE/training/images"
    lab_dir = root / split / label_folder        # builds the full path to the labels folder  "../data/raw/DRIVE/training/1st_manual"

    if not img_dir.exists():
        raise FileNotFoundError(f"[prepare_dataset] Missing images dir: {img_dir}")
    if not lab_dir.exists():
        raise FileNotFoundError(f"[prepare_dataset] Missing labels dir: {lab_dir}")

    imgs = _list_sorted_files(img_dir)  # get all files in dir, sorted naturally
    labs = _list_sorted_files(lab_dir)

    if len(imgs) == 0:
        raise RuntimeError(f"[prepare_dataset] No images found in {img_dir}")
    if len(labs) == 0:
        raise RuntimeError(f"[prepare_dataset] No labels found in {lab_dir}")

    n = min(len(imgs), len(labs)) # chooses the smaller count (in case one folder has extra files)
    pairs = [(str(imgs[i]), str(labs[i])) for i in range(n)] # builds a list of tuples where each image is paired with its corresponding label at the same index
    return pairs

    # sample return value:
    #  [
    #   ("../data/raw/DRIVE/training/images/01_training.jpg",
    #    "../data/raw/DRIVE/training/1st_manual/01_manual1.png"),
    #   ("../data/raw/DRIVE/training/images/02_training.jpg",
    #    "../data/raw/DRIVE/training/1st_manual/02_manual1.png"),
    #   ...
    # ]

def build_all_train_pairs(
    raw_root: str = "../data/raw",  # top-level folder where all datasets live
    datasets=("DRIVE", "CHASEDB1", "STARE"),
    label_folder: str = "1st_manual",
) -> List[Tuple[str, str]]:  # returns a flat list of (image_path, label_path) pairs across all datasets
    
    """
    Concatenate training pairs across datasets.
    """
    out: List[Tuple[str, str]] = []  # prep empty list to hold all pairs
    
    # iterate through each dataset (DRIVE, CHASEDB1, STARE)
    for ds in datasets:
        ds_root = str(Path(raw_root) / ds)  # build the path to that dataset’s folder;  if raw_root="../data/raw" and ds="DRIVE", → ds_root = "../data/raw/DRIVE".
        pairs = build_pairs_for_split(ds_root, split="training", label_folder=label_folder) # get the (image, label) pairs for that dataset’s training split
        out.extend(pairs) # add those pairs to the master list
    return out

# -------------------
# Sanity check helpers
# -------------------

# function that returns a dict of counts (images, labels, fov_masks) for one dataset/split
def sanity_check_counts(dataset_root: str, split: Split = "training", label_folder: str = "1st_manual") -> Dict[str, int]:
    root = Path(dataset_root)

    # build three folder paths like ../data/raw/DRIVE/training/images
    img_dir = root / split / "images"  
    lab_dir = root / split / label_folder
    msk_dir = root / split / "mask"

    # list all entries (files and folders) under each dir
    imgs = list((img_dir).glob("*"))
    labs = list((lab_dir).glob("*"))
    msks = list((msk_dir).glob("*")) if msk_dir.exists() else []

    # build a dict with file counts only (filters out subfolders)
    # returns something like {"images": 20, "labels": 20, "fov_masks": 20}
    return {
        "images": len([p for p in imgs if p.is_file()]),
        "labels": len([p for p in labs if p.is_file()]),
        "fov_masks": len([p for p in msks if p.is_file()]),
    }

# return the first k (image, label) basenames so we can quickly eyeball alignment in logs
def sanity_check_sample_alignment(pairs: List[Tuple[str, str]], k: int = 3) -> List[Tuple[str, str]]:
    """
    Return the first k (image, label) basenames so you can quickly eyeball alignment in logs.
    """
    view = []
    for (img, lab) in pairs[:k]: # iterate over the first k (image_path, label_path) tuples from the list
        view.append((Path(img).name, Path(lab).name)) # take just the filename part (no directories) and append the (image_name, label_name) pair
    return view # return a list like [("21_training.png", "21_manual1.png"), ...]


def assert_dataset_layout(dataset_root: str, split: Split = "training", label_folder: str = "1st_manual"):
    """
    Hard checks for expected dirs and non-empty contents. Raises on failure.
    """
    root = Path(dataset_root)
    for sub in ["images", label_folder, "mask"]:
        d = root / split / sub
        if not d.exists():
            raise FileNotFoundError(f"[prepare_dataset] Missing directory: {d}")
        if not any(d.iterdir()):
            raise RuntimeError(f"[prepare_dataset] Directory is empty: {d}")
