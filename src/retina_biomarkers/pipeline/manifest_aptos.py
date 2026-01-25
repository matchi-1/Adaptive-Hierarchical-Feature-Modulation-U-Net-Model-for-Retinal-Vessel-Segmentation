# This scans:

# APTOS_ROOT/
#   No_DR/
#   Mild/
#   Moderate/
#   Severe/
#   Proliferative/


# and builds a manifest where each base image has paths to its _fov and _fovea siblings.


# src/retina_biomarkers/pipeline/manifest_aptos.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Dict

APTOS_CLASSES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative"]


@dataclass(frozen=True)
class AptosItem:
    image_id: str            # e.g. "Moderate/img12"
    label: str               # folder name
    color_path: Path         # .../Moderate/img12.png
    fov_path: Optional[Path] # .../Moderate/img12_fov.png
    fovea_path: Optional[Path] # .../Moderate/img12_fovea.png


def _is_variant(stem: str) -> bool:
    return stem.endswith("_fov") or stem.endswith("_fovea")


def build_aptos_manifest(root_dir: str | Path,
                         classes: Iterable[str] = APTOS_CLASSES,
                         exts: Iterable[str] = (".png", ".jpg", ".jpeg")) -> List[AptosItem]:
    """
    Finds base images (not *_fov, not *_fovea) and pairs them with optional fov/fovea masks.
    Assumes naming:
      imgX.png, imgX_fov.png, imgX_fovea.png inside each label folder.
    """
    root_dir = Path(root_dir)
    items: List[AptosItem] = []

    for label in classes:
        d = root_dir / label
        if not d.exists():
            raise FileNotFoundError(f"Missing class folder: {d}")

        # map by stem for quick lookups
        files: Dict[str, Path] = {}
        for p in sorted(d.iterdir()):
            if not p.is_file():
                continue
            if p.suffix.lower() not in set(e.lower() for e in exts):
                continue
            files[p.stem] = p

        # base stems are those without variant suffix
        base_stems = [s for s in files.keys() if not _is_variant(s)]

        for base_stem in base_stems:
            color_path = files[base_stem]

            fov_path = files.get(base_stem + "_fov", None)
            fovea_path = files.get(base_stem + "_fovea", None)

            image_id = f"{label}/{base_stem}"  # stable ID
            items.append(AptosItem(
                image_id=image_id,
                label=label,
                color_path=color_path,
                fov_path=fov_path,
                fovea_path=fovea_path,
            ))

    return items
