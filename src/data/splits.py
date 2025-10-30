import random
import re
from collections import defaultdict
from pathlib import Path
from src.data.prepare_dataset import build_pairs_all, _natural_key

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
def _parse_chase_subject_id(img_path: str) -> str:
    """
    Parse CHASE_DB1 subject id from filename.
    Common patterns: 'Image_01L.jpg', 'Image_14R.png', etc.
    Returns e.g. '01'...'14' as subject key. If uncertain, fallback to natural stem chunks.
    """
    stem = Path(img_path).stem
    # Try common pattern: Image_XXL / Image_XXR
    m = re.search(r"(\d{2})[LR]?$", stem) or re.search(r"Image[_-]?(\d{2})[LR]?", stem, re.IGNORECASE)
    if m:
        return m.group(1)
    # Fallback: take the first numeric chunk found
    m2 = re.search(r"(\d+)", stem)
    return m2.group(1) if m2 else stem  # worst-case: use full stem

def split_chase_subjectwise_20_8(
    dataset_root: str,
    label_folder: str = "1st_manual",
    seed: int = 1337,
):
    """
    Merge both splits; group by subject (both eyes kept together), then sample subjects for 20/8 image split.
    CHASE_DB1 has 28 images = 14 subjects × 2 (L/R). We pick 4 subjects (8 imgs) for TEST, the rest TRAIN.
    """
    all_pairs = build_pairs_all(dataset_root, label_folder=label_folder)
    if len(all_pairs) != 28:
        raise ValueError(f"[CHASE_DB1] Expected 28 images total; got {len(all_pairs)}")

    # Group pairs by subject id
    by_subject = defaultdict(list)
    for (img, lab) in all_pairs:
        sid = _parse_chase_subject_id(img)
        by_subject[sid].append((img, lab))

    # We expect 14 subjects
    subjects = sorted(by_subject.keys())
    if len(subjects) != 14:
        # Not fatal, but warn
        print(f"[CHASE_DB1] Warning: expected 14 subjects, found {len(subjects)}")

    rng = random.Random(seed)
    # deterministic shuffle
    subs = subjects[:]  # copy
    subs.sort()  # stable anchor
    # choose 4 subjects for test (8 images)
    # We’ll just take the last 4 deterministically
    test_subjects = subs[-4:]
    train_subjects = subs[:-4]

    train_pairs = [p for sid in train_subjects for p in sorted(by_subject[sid], key=lambda t: _natural_key(Path(t[0])))]
    test_pairs  = [p for sid in test_subjects  for p in sorted(by_subject[sid], key=lambda t: _natural_key(Path(t[0])))]
    if len(train_pairs) != 20 or len(test_pairs) != 8:
        raise RuntimeError(f"[CHASE_DB1] Split mismatch: train={len(train_pairs)}, test={len(test_pairs)}")
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
