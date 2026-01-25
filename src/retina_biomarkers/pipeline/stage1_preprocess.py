# This runs Stage 1 exactly using:
    #   _iso_resize_and_pad(...) → rgb_iso
    #   preprocess_image_retina_from_pil(...) → img_1hw
    #   builds aligned fov_1hw and img_fov_1hw
    #   (optional) aligns fovea_1hw too

# It saves per-image cache files:
# outputs/aptos_cache/<run_id>/stage1/<safe_image_id>/
#   rgb_iso.npy
#   img_1hw.npy
#   fov_1hw.npy
#   img_fov_1hw.npy
#   fovea_1hw.npy (if available)
#   pre_gray_u8.png (optional)
#   meta.json

# src/retina_biomarkers/pipeline/stage1_preprocess.py
from __future__ import annotations

import json, hashlib
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
from PIL import Image

from src.retina_biomarkers.notebook_utils.pipeline.config import PipelineConfig
from src.data.preprocessing import _iso_resize_and_pad
from apps.streamlit.lib.preprocess import preprocess_image_retina_from_pil


# ----------------------------
# helpers: config hashing
# ----------------------------
def _cfg_to_dict(cfg: Any) -> Dict[str, Any]:
    # robust for dataclass / pydantic / plain object
    if hasattr(cfg, "model_dump"):  # pydantic v2
        d = cfg.model_dump()
    elif is_dataclass(cfg):
        d = asdict(cfg)
    else:
        d = {k: v for k, v in vars(cfg).items() if not k.startswith("_") and not callable(v)}
    return d


def make_run_id(cfg: PipelineConfig, prefix: str = "aptos") -> str:
    d = _cfg_to_dict(cfg)
    payload = json.dumps(d, sort_keys=True, default=str).encode("utf-8")
    h = hashlib.sha1(payload).hexdigest()[:10]
    return f"{prefix}_stage1_{h}"


def safe_image_id(image_id: str) -> str:
    # "Moderate/img12" -> "Moderate__img12"
    return image_id.replace("/", "__").replace("\\", "__").replace(":", "_")


# ----------------------------
# stage1 paths
# ----------------------------
def stage1_dir(cache_root: str | Path, run_id: str, image_id: str) -> Path:
    return Path(cache_root) / run_id / "stage1" / safe_image_id(image_id)


def stage1_done_flag(out_dir: Path) -> Path:
    return out_dir / "meta.json"


# ----------------------------
# stage1 core
# ----------------------------
def stage1_preprocess_one(
    *,
    image_id: str,
    color_path: str | Path,
    fov_path: Optional[str | Path],
    fovea_path: Optional[str | Path],
    cfg: PipelineConfig,
    cache_root: str | Path,
    run_id: str,
    save_debug_png: bool = True,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Stage 1: fundus -> (rgb_iso, img_1hw, fov_1hw, img_fov_1hw, optional fovea_1hw)
    Writes cache files and returns metadata dict.
    """
    out_dir = stage1_dir(cache_root, run_id, image_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    done = stage1_done_flag(out_dir)
    if done.exists() and not overwrite:
        # Already processed
        return {"image_id": image_id, "status": "skipped", "out_dir": str(out_dir)}

    # 1) Load fundus and ISO canvas (for OD stage later)
    fundus_pil = Image.open(color_path).convert("RGB")
    rgb_iso = _iso_resize_and_pad(
        np.array(fundus_pil),
        target=cfg.image_size,
        pad_value=0
    ).astype(np.uint8)
    H, W = rgb_iso.shape[:2]

    # 2) Preprocess grayscale for vessel model
    img_1hw = preprocess_image_retina_from_pil(
        fundus_pil,
        target_size=cfg.image_size,
        use_gamma=cfg.use_gamma, gamma=cfg.gamma,
        clahe_clip=cfg.clahe_clip, clahe_tiles=cfg.clahe_tiles
    ).astype(np.float32)  # shape (1,H,W)

    # 3) FOV mask aligned to (H,W)
    if fov_path is not None:
        fov_pil = Image.open(fov_path).convert("L").resize((W, H), resample=Image.NEAREST)
        fov_2d = (np.array(fov_pil) > 0).astype(np.float32)  # (H,W)
        fov_1hw = fov_2d[None, ...]
    else:
        # fallback: assume background black so preprocessed grayscale > 0 approximates FOV
        fov_1hw = (img_1hw > 0).astype(np.float32)

    img_fov_1hw = (img_1hw * fov_1hw).astype(np.float32)

    # 4) Optional: fovea mask aligned (useful for later fovea center)
    fovea_1hw = None
    if fovea_path is not None:
        fovea_pil = Image.open(fovea_path).convert("L").resize((W, H), resample=Image.NEAREST)
        fovea_2d = (np.array(fovea_pil) > 0).astype(np.uint8)
        fovea_1hw = fovea_2d[None, ...].astype(np.uint8)

    # 5) Save artifacts
    np.save(out_dir / "rgb_iso.npy", rgb_iso)
    np.save(out_dir / "img_1hw.npy", img_1hw)
    np.save(out_dir / "fov_1hw.npy", fov_1hw)
    np.save(out_dir / "img_fov_1hw.npy", img_fov_1hw)

    if fovea_1hw is not None:
        np.save(out_dir / "fovea_1hw.npy", fovea_1hw)

    if save_debug_png:
        pre_gray_u8 = np.clip(img_1hw[0] * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(pre_gray_u8).save(out_dir / "pre_gray_u8.png")

        # optional: also dump fov preview
        fov_u8 = (fov_1hw[0] * 255).astype(np.uint8)
        Image.fromarray(fov_u8).save(out_dir / "fov_u8.png")

    meta = {
        "image_id": image_id,
        "color_path": str(color_path),
        "fov_path": str(fov_path) if fov_path is not None else None,
        "fovea_path": str(fovea_path) if fovea_path is not None else None,
        "H": int(H),
        "W": int(W),
        "cfg": _cfg_to_dict(cfg),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    return {"image_id": image_id, "status": "ok", "out_dir": str(out_dir), "H": H, "W": W}


def stage1_preprocess_batch(
    items,
    *,
    cfg: PipelineConfig,
    cache_root: str | Path,
    run_id: str,
    save_debug_png: bool = False,
    overwrite: bool = False,
):
    """
    Convenience batch runner for a list of items that have:
      item.image_id, item.color_path, item.fov_path, item.fovea_path
    """
    results = []
    for it in items:
        r = stage1_preprocess_one(
            image_id=it.image_id,
            color_path=it.color_path,
            fov_path=it.fov_path,
            fovea_path=it.fovea_path,
            cfg=cfg,
            cache_root=cache_root,
            run_id=run_id,
            save_debug_png=save_debug_png,
            overwrite=overwrite,
        )
        results.append(r)
    return results
