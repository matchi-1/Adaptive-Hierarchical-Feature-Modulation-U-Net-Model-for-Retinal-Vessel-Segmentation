# src/data/prepare_dataset.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Literal, Dict
import re

Split = Literal["training", "test"]


'''
_natural_key
Purpose:
    - Produce a "natural" sort key so numbers inside filenames sort numerically (e.g., 2 < 10).
    - Avoids lexicographic misordering like "10_training" coming before "2_training".
Inputs:
    - p: pathlib.Path for a file (we only use the stem).
Outputs:
    - List of mixed ints/strings usable as a stable Python sort key, e.g., [21, "_training"].
Notes:
    - Splits the stem into digit and non-digit chunks; digit chunks are cast to int.
'''
def _natural_key(p: Path):
    s = p.stem.lower() # .stem to get filename without extension ("21_training")
     # splits the stem into chunks, separating digit sequences ("21", "training") 
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)] # for each chunk, convert digit-only pieces
    # returns a list like [21, "_training"] -- will be used as a sort key


'''
_list_sorted_files
Purpose:
    - List files directly under a directory and sort them in "natural" numeric order.
Inputs:
    - dirpath: pathlib.Path directory to scan (non-recursive).
Outputs:
    - List[pathlib.Path] of files sorted by _natural_key.
Notes:
    - Excludes subdirectories (filters with p.is_file()).
'''

def _list_sorted_files(dirpath: Path):
    # Path.glob("*") lists all entries (files + directories) inside dirpath
    # p.is_file filters out everything that isn’t a file. removes directories like "subfolder" so we only keep .png, .jpg
    return sorted([p for p in dirpath.glob("*") if p.is_file()], key=_natural_key) # use natural key to sort by number


'''
_build_pairs_for_split
Purpose:
    - Create (image_path, label_path) pairs for a single dataset and split.
    - Ensures image/label lists are aligned by natural sort order.
Inputs:
    - dataset_root: str, dataset root (e.g., "../data/raw/DRIVE").
    - split: {"training","test"} which subset to use.
    - label_folder: str name of the label directory (e.g., "1st_manual" or "2nd_manual").
Outputs:
    - List[Tuple[str, str]] of (image_path, label_path) pairs (as strings).
'''

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


'''
build_all_train_pairs
Purpose:
    - Aggregate training (image,label) pairs across multiple datasets into one list.
Inputs:
    - raw_root: str top-level path that contains dataset folders (e.g., "../data/raw").
    - datasets: Iterable of dataset folder names (e.g., ("DRIVE","CHASEDB1","STARE")).
    - label_folder: str label directory name to use (e.g., "1st_manual").
Outputs:
    - List[Tuple[str, str]] flattened list of (image_path, label_path) pairs across datasets.
Notes:
    - Calls build_pairs_for_split(..., split="training") for each dataset, then concatenates.
    - Order of datasets is preserved as provided in `datasets`.
'''

def build_all_train_pairs(
    raw_root: str = "../data/raw",  # top-level folder where all datasets live
    datasets=("DRIVE", "CHASEDB1", "STARE"),
    label_folder: str = "1st_manual",
    split = "training"
) -> List[Tuple[str, str]]:  # returns a flat list of (image_path, label_path) pairs across all datasets
    
    """
    Concatenate training pairs across datasets.
    """
    out: List[Tuple[str, str]] = []  # prep empty list to hold all pairs
    
    # iterate through each dataset (DRIVE, CHASEDB1, STARE)
    for ds in datasets:
        ds_root = str(Path(raw_root) / ds)  # build the path to that dataset’s folder;  if raw_root="../data/raw" and ds="DRIVE", → ds_root = "../data/raw/DRIVE".
        pairs = build_pairs_for_split(ds_root, split=split, label_folder=label_folder) # get the (image, label) pairs for that dataset’s training split
        out.extend(pairs) # add those pairs to the master list
    return out

# -------------------
# Sanity check helpers
# -------------------


'''
_sanity_check_counts
Purpose:
    - Report counts of images, labels, and FOV masks for a specific dataset split.
Inputs:
    - dataset_root: str dataset root (e.g., "../data/raw/DRIVE").
    - split: {"training","test"} which subset to inspect.
    - label_folder: str label directory to count (e.g., "1st_manual").
Outputs:
    - Dict[str,int] with keys {"images","labels","fov_masks"} and integer counts.
'''
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


'''
_sanity_check_sample_alignment
Purpose:
    - Print-friendly peek at the first k (image,label) pairs to verify index alignment.
Inputs:
    - pairs: List of (image_path, label_path) tuples (strings).
    - k: int number of pairs to preview.
Outputs:
    - List[Tuple[str,str]] of basenames only, e.g., [("21_training.png","21_manual1.png"), ...].
Notes:
    - Helps visually confirm that sorting and pairing rules are correct.
'''

# return the first k (image, label) basenames so we can quickly eyeball alignment in logs
def sanity_check_sample_alignment(pairs: List[Tuple[str, str]], k: int = 3) -> List[Tuple[str, str]]:
    """
    Return the first k (image, label) basenames so you can quickly eyeball alignment in logs.
    """
    view = []
    for (img, lab) in pairs[:k]: # iterate over the first k (image_path, label_path) tuples from the list
        view.append((Path(img).name, Path(lab).name)) # take just the filename part (no directories) and append the (image_name, label_name) pair
    return view # return a list like [("21_training.png", "21_manual1.png"), ...]


'''
_assert_dataset_layout
Purpose:
    - Hard validation that expected subdirectories exist and are non-empty for a given split.
Inputs:
    - dataset_root: str dataset root (e.g., "../data/raw/DRIVE").
    - split: {"training","test"} which subset to inspect.
    - label_folder: str label directory name to require (e.g., "1st_manual").
Outputs:
    - None (raises on failure).
'''

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
