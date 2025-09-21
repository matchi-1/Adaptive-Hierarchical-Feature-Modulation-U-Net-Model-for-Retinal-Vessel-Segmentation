# src/data/prepare_dataset.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Literal, Dict
import re

Split = Literal["training", "test"]

def _natural_key(p: Path):
    s = p.stem.lower()
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]

def _list_sorted_files(dirpath: Path):
    return sorted([p for p in dirpath.glob("*") if p.is_file()], key=_natural_key)

def build_pairs_for_split(
    dataset_root: str,           # e.g. "../data/raw/DRIVE"
    split: Split = "training",   # "training" or "test"
    label_folder: str = "1st_manual",
) -> List[Tuple[str, str]]:
    """
    Build (image_path, label_path) pairs for a single dataset (DRIVE/CHASEDB1/STARE) and split.

    Expects structure:
      <root>/<split>/images/*.*
      <root>/<split>/<label_folder>/*.*  (1st_manual by default)
      <root>/<split>/mask/*.*
    Pairs are created by natural sort order (index-aligned filenames).
    """
    root = Path(dataset_root)
    img_dir = root / split / "images"
    lab_dir = root / split / label_folder

    if not img_dir.exists():
        raise FileNotFoundError(f"[prepare_dataset] Missing images dir: {img_dir}")
    if not lab_dir.exists():
        raise FileNotFoundError(f"[prepare_dataset] Missing labels dir: {lab_dir}")

    imgs = _list_sorted_files(img_dir)
    labs = _list_sorted_files(lab_dir)

    if len(imgs) == 0:
        raise RuntimeError(f"[prepare_dataset] No images found in {img_dir}")
    if len(labs) == 0:
        raise RuntimeError(f"[prepare_dataset] No labels found in {lab_dir}")

    n = min(len(imgs), len(labs))
    pairs = [(str(imgs[i]), str(labs[i])) for i in range(n)]
    return pairs

def build_all_train_pairs(
    raw_root: str = "../data/raw",
    datasets=("DRIVE", "CHASEDB1", "STARE"),
    label_folder: str = "1st_manual",
) -> List[Tuple[str, str]]:
    """
    Concatenate training pairs across datasets.
    """
    out: List[Tuple[str, str]] = []
    for ds in datasets:
        ds_root = str(Path(raw_root) / ds)
        pairs = build_pairs_for_split(ds_root, split="training", label_folder=label_folder)
        out.extend(pairs)
    return out

# -------------------
# Sanity check helpers
# -------------------
def sanity_check_counts(dataset_root: str, split: Split = "training", label_folder: str = "1st_manual") -> Dict[str, int]:
    root = Path(dataset_root)
    img_dir = root / split / "images"
    lab_dir = root / split / label_folder
    msk_dir = root / split / "mask"

    imgs = list((img_dir).glob("*"))
    labs = list((lab_dir).glob("*"))
    msks = list((msk_dir).glob("*")) if msk_dir.exists() else []

    return {
        "images": len([p for p in imgs if p.is_file()]),
        "labels": len([p for p in labs if p.is_file()]),
        "fov_masks": len([p for p in msks if p.is_file()]),
    }

def sanity_check_sample_alignment(pairs: List[Tuple[str, str]], k: int = 3) -> List[Tuple[str, str]]:
    """
    Return the first k (image, label) basenames so you can quickly eyeball alignment in logs.
    """
    view = []
    for (img, lab) in pairs[:k]:
        view.append((Path(img).name, Path(lab).name))
    return view

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
