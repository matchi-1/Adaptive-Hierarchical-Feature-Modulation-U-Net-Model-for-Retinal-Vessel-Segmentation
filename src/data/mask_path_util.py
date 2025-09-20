import re
from pathlib import Path

_IMG_EXTS = [".png", ".jpg", ".jpeg", ".gif", ".tif", ".tiff", ".bmp"]

def _dataset_name_from_path(p: Path) -> str | None:
    up = str(p).upper()
    if "DRIVE" in up: return "DRIVE"
    if "CHASEDB1" in up or "CHASE" in up: return "CHASEDB1"
    if "STARE" in up: return "STARE"
    return None

def _mask_dir_candidates(p: Path) -> list[Path]:
    parts = list(p.parts)
    out = []
    # swap 'images' -> 'mask'
    if "images" in parts:
        q = parts.copy()
        q[parts.index("images")] = "mask"
        out.append(Path(*q).parent)
    # common GT dirs seen in datasets
    for gt in ("mask", "masks", "1st_manual", "manual", "manual1", "labels", "labels-ah", "label"):
        out.append(p.parent.parent / gt)
    # alongside
    out.append(p.parent)
    # de-dup, keep existing dirs first
    uniq = []
    seen = set()
    for d in out:
        if d in seen: continue
        seen.add(d)
        uniq.append(d)
    return uniq

def _try_paths(dirpath: Path, stems: list[str]) -> Path | None:
    for stem in stems:
        for ext in _IMG_EXTS:
            cand = dirpath / f"{stem}{ext}"
            if cand.exists(): return cand
    return None

def _digits_key(s: str) -> str:
    return "".join(re.findall(r"\d+", s))

def _infer_mask_path(image_path: str | Path) -> Path | None:
    """
    Resolve mask path across DRIVE / CHASEDB1 / STARE with heterogeneous names.
    Returns a Path or None if nothing matches.
    """
    p = Path(image_path)
    ds = _dataset_name_from_path(p)
    stem = p.stem

    # candidate directories to search
    dirs = _mask_dir_candidates(p)

    # dataset-specific stem candidates
    stems: list[str] = [stem]
    if ds == "CHASEDB1":
        # CHASEDB1: same stem, usually .png
        stems += [stem]  # keep as-is
    elif ds == "DRIVE":
        # DRIVE: patterns like 21_training -> 21_training_mask or _manual1
        s = stem
        stems += [
            f"{s}_mask",
            s.replace("_training", "_training_mask"),
            s.replace("_test", "_test_mask"),
            s.replace("_training", "_manual1"),
            s.replace("_test", "_manual1"),
            s.replace("_Training", "_manual1"),
            s.replace("_Test", "_manual1"),
        ]
        # also bare numeric prefix versions: 21 -> 21_training_mask / 21_test_mask
        num = _digits_key(s)
        if num:
            stems += [f"{num}_training_mask", f"{num}_test_mask",
                      f"{num}_manual1", f"{num}_mask"]
    elif ds == "STARE":
        # STARE: masks often jpg and may not share exact stem; match by numeric id
        # keep exact first
        stems += [stem.replace("image ", "im").replace(" ", "")]
        # fallbacks handled by numeric matching below

    # try exact stem candidates across dirs/exts
    for d in dirs:
        hit = _try_paths(d, stems)
        if hit is not None:
            return hit

    # numeric-ID fallback: pick file in candidate dirs whose digits match
    img_key = _digits_key(stem)
    if img_key:
        for d in dirs:
            if not d.exists(): continue
            best = None
            for f in d.iterdir():
                if not f.is_file(): continue
                if f.suffix.lower() not in _IMG_EXTS: continue
                if _digits_key(f.stem) == img_key:
                    best = f
                    break
            if best is not None:
                return best

    # nothing found
    return None
