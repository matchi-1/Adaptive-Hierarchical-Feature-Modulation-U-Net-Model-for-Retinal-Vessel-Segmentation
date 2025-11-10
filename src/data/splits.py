from __future__ import annotations
import random
import re
from collections import defaultdict
from pathlib import Path
from src.data.prepare_dataset import build_pairs_all, _natural_key
from src.evaluation.visualization import _read_original_rgb
from typing import List, Tuple


# -------------- DRIVE: 20/20 --------------
def split_drive_20_20(
    dataset_root: str,
    label_folder: str = "1st_manual",
    seed: int = 1337,
):
    """
    Use ALL images (training+test merged) then deterministically pick 20 for TEST and 20 for TRAIN.
    If a canonical 20/20 is already present in folders, the result will match file order deterministically.
    """
    all_pairs = build_pairs_all(dataset_root, label_folder=label_folder)
    if len(all_pairs) != 40:
        raise ValueError(f"[DRIVE] Expected 40 images total; got {len(all_pairs)}")

    rng = random.Random(seed)
    # Deterministic: keep order stable, then choose the last 20 as test or sample 20
    idxs = list(range(len(all_pairs)))
    # Option A (stable by order): first 20 train, last 20 test
    train_pairs = [all_pairs[i] for i in idxs[:20]]
    test_pairs  = [all_pairs[i] for i in idxs[20:]]
    return train_pairs, test_pairs


# -------------- CHASE_DB1: subject-wise 20/8 --------------
def _extract_first_number(stem: str) -> int | None:
    """
    From a filename stem like '01_test', '15_training', return the leading integer (e.g., 1, 15).
    Returns None if no digits are found.
    """
    m = re.search(r"(\d+)", stem)
    return int(m.group(1)) if m else None

def _chase_subject_id_from_number(n: int) -> str:
    """
    CHASE-DB1 convention in your folders:
      (01,02)->01, (03,04)->02, ..., (27,28)->14
    i.e., subject_id = (n+1)//2, zero-padded to 2 digits.
    """
    sid = (n + 1) // 2
    return f"{sid:02d}"

def _parse_chase_subject_id(img_path: str) -> str:
    """
    Map an image filename to a subject id using your numbering scheme.
    Works with stems like '01_test', '02_test', '15_training', '16_training', etc.
    """
    stem = Path(img_path).stem.lower()
    n = _extract_first_number(stem)
    if n is None:
        # Fallback: use full stem (unlikely with your dataset)
        return stem
    return _chase_subject_id_from_number(n)

def split_chase_subjectwise_20_8(
    dataset_root: str,
    label_folder: str = "1st_manual",
    seed: int = 1337,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Merge 'training' and 'test' into one pool, group by subject (consecutive pairs),
    then pick 4 subjects (8 images) for TEST, leaving 10 subjects (20 images) for TRAIN.
    Deterministic with `seed`.
    """
    # 1) Collect all (img, lab) pairs from both splits
    all_pairs = build_pairs_all(dataset_root, label_folder=label_folder)
    if len(all_pairs) != 28:
        raise ValueError(f"[CHASE_DB1] Expected 28 images total (found {len(all_pairs)}). "
                         f"Check dataset layout or label folder name '{label_folder}'.")

    # 2) Group by subject id
    by_subject: dict[str, list[Tuple[str, str]]] = defaultdict(list)
    for (img, lab) in all_pairs:
        sid = _parse_chase_subject_id(img)
        by_subject[sid].append((img, lab))

    subjects = sorted(by_subject.keys())
    if len(subjects) != 14:
        # Not fatal, but surface it clearly
        raise RuntimeError(f"[CHASE_DB1] Subject grouping mismatch: expected 14 subjects, found {len(subjects)}. "
                           f"Example groups: " +
                           ", ".join(f"{s}:{len(by_subject[s])}" for s in subjects[:6]))

    # Optional sanity: each subject should have exactly 2 images
    bad = [s for s in subjects if len(by_subject[s]) != 2]
    if bad:
        raise RuntimeError(f"[CHASE_DB1] Some subjects do not have exactly 2 images: "
                           f"{ {s: len(by_subject[s]) for s in bad} }")

    # 3) Deterministically choose 4 subjects for TEST
    rng = random.Random(seed)
    subs = subjects[:]      # copy
    subs.sort()             # stable base order
    rng.shuffle(subs)       # seeded shuffle
    test_subjects  = sorted(subs[:4])   # 4 subjects -> 8 images
    train_subjects = sorted(subs[4:])   # 10 subjects -> 20 images

    # 4) Flatten, keep natural order within each subject
    def _flatten(subject_list: List[str]) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        for sid in subject_list:
            # sort each subject's two images naturally (e.g., 15 before 16)
            out.extend(sorted(by_subject[sid], key=lambda t: _natural_key(Path(t[0]))))
        return out

    train_pairs = _flatten(train_subjects)
    test_pairs  = _flatten(test_subjects)

    # 5) Final sanity
    if len(train_pairs) != 20 or len(test_pairs) != 8:
        raise RuntimeError(f"[CHASE_DB1] Split mismatch: train={len(train_pairs)}, test={len(test_pairs)}. "
                           f"train_subjects={train_subjects}, test_subjects={test_subjects}")

    return train_pairs, test_pairs


# -------------- STARE: Leave-One-Out --------------
def split_stare_leave_one_out(
    dataset_root: str,
    label_folder: str = "1st_manual",
):
    """
    Generator that yields 20 folds. Each fold: 1 image for TEST, the remaining 19 for TRAIN.
    You can further carve a VAL set from TRAIN (e.g., fixed 3 images) deterministically later.
    """
    all_pairs = build_pairs_all(dataset_root, label_folder=label_folder)
    if len(all_pairs) != 20:
        raise ValueError(f"[STARE] Expected 20 images total; got {len(all_pairs)}")
    # deterministic order by natural sort of image path
    all_pairs = sorted(all_pairs, key=lambda p: _natural_key(Path(p[0])))

    for i in range(len(all_pairs)):
        test_pairs = [all_pairs[i]]
        train_pairs = all_pairs[:i] + all_pairs[i+1:]
        yield i, train_pairs, test_pairs

# STARE leave one out (LOO) fold function
def stare_loo_fold(dataset_root: str, label_folder: str = "1st_manual", fold_id: int = 0) -> Tuple[list, list]:
    
    import matplotlib.pyplot as plt
    
    pairs = build_pairs_all(dataset_root, label_folder=label_folder)
    pairs = sorted(pairs, key=lambda p: _natural_key(Path(p[0])))
    assert len(pairs) == 20, f"Expected 20 STARE images, got {len(pairs)}"
    assert 0 <= fold_id < 20
    test_pairs  = [pairs[fold_id]]
    train_pairs = pairs[:fold_id] + pairs[fold_id+1:]

    img_path = test_pairs[0][0]          # (image_path, label_path)
    img = _read_original_rgb(img_path)           

    plt.figure(figsize=(3,3))
    if img.ndim == 2:                     # grayscale
        plt.imshow(img, cmap='gray')
    else:                                 # RGB
        plt.imshow(img)
    plt.title(f"STARE • fold {fold_id} • {Path(img_path).name}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    return train_pairs, test_pairs





# ------- TO REVISIT -- Monte-Carlo Split cross validation -------

# ---------- CHASE-DB1: subject-wise random 20/8 ----------
def chase_random_subject_split(dataset_root: str, label_folder: str = "1st_manual",
                               seed: int = 1337, n_test_subjects: int = 4):
    """
    Random subject-wise split: pick n_test_subjects from the 14 subjects (→ 2*n_test_subjects images for TEST),
    keep both eyes of each subject together.
    """
    all_pairs = build_pairs_all(dataset_root, label_folder=label_folder)
    by_subject = defaultdict(list)
    for (img, lab, *rest) in all_pairs:
        sid = _parse_chase_subject_id(img)
        by_subject[sid].append((img, lab, *rest))

    subjects = sorted(by_subject.keys())
    rng = random.Random(seed)
    rng.shuffle(subjects)
    test_subjects  = subjects[:n_test_subjects]
    train_subjects = subjects[n_test_subjects:]

    train_pairs = [p for sid in train_subjects for p in sorted(by_subject[sid], key=lambda t: _natural_key(Path(t[0])))]
    test_pairs  = [p for sid in test_subjects  for p in sorted(by_subject[sid], key=lambda t: _natural_key(Path(t[0])))]
    return train_pairs, test_pairs

# ---------- DRIVE: random 20/20 (images, no subject pairing) ----------
def drive_random_20_20(dataset_root: str, label_folder: str = "1st_manual", seed: int = 1337):
    """
    Randomly choose 20 images for TEST, remaining 20 for TRAIN.
    """
    all_pairs = build_pairs_all(dataset_root, label_folder=label_folder)  # 40 images expected
    rng = random.Random(seed)
    idx = list(range(len(all_pairs)))
    rng.shuffle(idx)
    test_idx  = set(idx[:20])
    train_idx = [i for i in idx[20:]]
    train_pairs = [all_pairs[i] for i in train_idx]
    test_pairs  = [all_pairs[i] for i in test_idx]
    # stable sort for readability
    train_pairs.sort(key=lambda t: _natural_key(Path(t[0])))
    test_pairs.sort(key=lambda t: _natural_key(Path(t[0])))
    return train_pairs, test_pairs



import numpy as np
from collections import defaultdict

def weighted_sample_without_replacement(items, weights, k, rng: np.random.RandomState):
    """Simple weighted sampling w/o replacement using numpy."""
    items = np.array(items)
    weights = np.array(weights, dtype=float)
    chosen = []
    for _ in range(k):
        w = weights.copy()
        w_sum = w.sum()
        if w_sum <= 0:
            # fall back to uniform if all weights are zero
            w = np.ones_like(w) / len(w)
        else:
            w = w / w_sum
        idx = rng.choice(len(items), p=w)
        chosen.append(items[idx])
        # remove this item or zero its weight
        weights[idx] = 0.0
    return chosen

def chase_mc_balanced_splits(
    dataset_root: str,
    label_folder: str = "1st_manual",
    n_runs: int = 5,
    n_test_subjects: int = 4,
    seed: int = 1337,
):
    all_pairs = build_pairs_all(dataset_root, label_folder=label_folder)
    by_subject = defaultdict(list)
    for (img, lab) in all_pairs:
        sid = _parse_chase_subject_id(img)
        by_subject[sid].append((img, lab))

    subjects = sorted(by_subject.keys())
    rng = np.random.RandomState(seed)

    test_count = {sid: 0 for sid in subjects}
    splits = []

    for run_idx in range(n_runs):
        # weights inversely proportional to (1 + test_count)
        weights = [1.0 / (1 + test_count[sid]) for sid in subjects]
        chosen_test = weighted_sample_without_replacement(subjects, weights, n_test_subjects, rng)

        # update counts
        for sid in chosen_test:
            test_count[sid] += 1

        train_subjects = [sid for sid in subjects if sid not in chosen_test]

        def _flatten(sub_list):
            out = []
            for s in sub_list:
                out.extend(sorted(by_subject[s], key=lambda t: _natural_key(Path(t[0]))))
            return out

        train_pairs = _flatten(train_subjects)
        test_pairs  = _flatten(chosen_test)

        splits.append((train_pairs, test_pairs, chosen_test))

    return splits, test_count