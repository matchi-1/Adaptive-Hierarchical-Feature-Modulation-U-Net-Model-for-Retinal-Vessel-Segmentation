# src/data/generate_fov.py
# Generate FOV masks for CHASEDB1. Writes 0/255 PNGs into .../mask.

import sys
from pathlib import Path
import cv2
import numpy as np

# ensures project root (parent of 'src') is importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import _estimate_fov_mask  # expects RGB float32 in [0,1]

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

def _iter_images(img_dir: Path):
    for p in sorted(img_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p

def _ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def _to_rgb01(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

def _estimate_fov_compat(rgb01: np.ndarray) -> np.ndarray:
    """
    Call the project's _estimate_fov_mask whether or not it supports 'dataset'.
    For CHASE we’re fine with the default fast path.
    """
    try:
        return _estimate_fov_mask(rgb01, dataset="CHASE")
    except TypeError:
        # Older signature: _estimate_fov_mask(rgb01)
        return _estimate_fov_mask(rgb01)

def generate_masks_for_split(images_dir: Path, masks_dir: Path):
    _ensure_dir(masks_dir)
    made = skipped = 0

    for img_path in _iter_images(images_dir):
        out_path = masks_dir / (img_path.stem + ".png")
        if out_path.exists():
            skipped += 1
            continue

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[WARN] could not read: {img_path}")
            continue

        rgb01 = _to_rgb01(bgr)
        fov = _estimate_fov_compat(rgb01)        # float32 {0,1}, same HxW
        mask_u8 = (fov * 255.0).astype(np.uint8)

        if not cv2.imwrite(str(out_path), mask_u8):
            print(f"[WARN] write failed: {out_path}")
            continue

        made += 1

    return made, skipped

def main():
    root = Path("data/raw/CHASEDB1")  # run from project root
    pairs = [
        (root / "training" / "images", root / "training" / "mask"),
        (root / "test" / "images",     root / "test" / "mask"),
    ]

    total_made = total_skipped = 0
    for img_dir, msk_dir in pairs:
        print(f"[INFO] {img_dir} -> {msk_dir}")
        if not img_dir.exists():
            print(f"[INFO] missing images dir; skipping: {img_dir}")
            continue
        made, skipped = generate_masks_for_split(img_dir, msk_dir)
        print(f"[INFO] created={made} skipped_existing={skipped}")
        total_made += made
        total_skipped += skipped

    print(f"[SUMMARY] created={total_made} skipped_existing={total_skipped}")

if __name__ == "__main__":
    main()
