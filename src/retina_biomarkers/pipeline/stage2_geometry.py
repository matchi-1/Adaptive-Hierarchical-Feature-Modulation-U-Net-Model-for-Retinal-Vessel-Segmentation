# This loads OD model once, loops through Stage 1 outputs, computes OD mask + PD, computes fovea center if available, and saves everything.

# Stage 2 turns each image into stable geometry facts we’ll reuse everywhere:

    # disc_mask (binary mask of optic disc)
    # OD center (cy, cx)
    # PD_px (optic disc diameter in pixels; used for PD rings like 0.5–1.0 PD, etc.)
    # (recommended) fovea center from our fovea_1hw.npy so later we can orient ISNT properly

# This stage is “expensive-ish” because OD segmentation is a model inference, so we cache it once and never recompute unless we change OD model/config.


# src/retina_biomarkers/pipeline/stage2_geometry.py
from __future__ import annotations

import json, hashlib
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
from PIL import Image

from src.data.preprocessing import _iso_resize_and_pad
from src.retina_biomarkers.notebook_utils.pipeline.config import PipelineConfig

# OD segmentation
from src.retina_biomarkers.od_seg import (
    load_refuge_segformer, infer_label_map, extract_disc_mask_safe,
    center_and_pd_with_bounds,
)


# ----------------------------
# helpers
# ----------------------------
def _cfg_to_dict(cfg: Any) -> Dict[str, Any]:
    if hasattr(cfg, "model_dump"):
        return cfg.model_dump()
    if is_dataclass(cfg):
        return asdict(cfg)
    return {k: v for k, v in vars(cfg).items() if not k.startswith("_") and not callable(v)}

def make_run_id_stage2(cfg: PipelineConfig, prefix: str = "aptos") -> str:
    d = _cfg_to_dict(cfg)
    payload = json.dumps(d, sort_keys=True, default=str).encode("utf-8")
    h = hashlib.sha1(payload).hexdigest()[:10]
    return f"{prefix}_stage2_{h}"

def safe_image_id(image_id: str) -> str:
    # supports your earlier output style too
    # "No_DR/img1" -> "No_DR__img1"
    return image_id.replace("/", "__").replace("\\", "__").replace(":", "_")

def stage1_dir(cache_root: str | Path, run_id_stage1: str, image_id: str) -> Path:
    return Path(cache_root) / run_id_stage1 / "stage1" / safe_image_id(image_id)

def stage2_dir(cache_root: str | Path, run_id_stage2: str, image_id: str) -> Path:
    return Path(cache_root) / run_id_stage2 / "stage2" / safe_image_id(image_id)

def _center_from_binary_mask(mask_2d: np.ndarray) -> Optional[Tuple[float, float]]:
    """Return (cy,cx) center of mass of a binary mask; None if empty."""
    ys, xs = np.where(mask_2d > 0)
    if len(ys) == 0:
        return None
    return (float(ys.mean()), float(xs.mean()))


def find_fovea_yx_from_cyan_iso(fovea_img_path: str, image_size: int = 512, tol: int = 40):
    """
    Reads the RGB fovea image (fundus + cyan dot), applies the SAME iso resize+pad,
    then finds the cyan dot coordinates in the iso coordinate system.
    Returns (fy, fx) as floats.
    """
    f_rgb = np.array(Image.open(fovea_img_path).convert("RGB"), dtype=np.uint8)
    f_iso = _iso_resize_and_pad(f_rgb, target=image_size, pad_value=0).astype(np.uint8)

    R, G, B = f_iso[..., 0], f_iso[..., 1], f_iso[..., 2]

    # strict cyan: low R, high G, high B
    mask = (R <= tol) & (G >= 255 - tol) & (B >= 240 - tol)

    if not mask.any():
        # broader fallback
        mask = (R < 80) & (G > 170) & (B > 170)

    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None  # dot not found

    fy = float(ys.mean())
    fx = float(xs.mean())
    return (fy, fx)



# ----------------------------
# OD context (load once)
# ----------------------------
class ODContext:
    def __init__(self):
        self.processor, self.model, self.device = load_refuge_segformer()

    @torch.no_grad()
    def infer_disc(self, rgb_iso_u8: np.ndarray, cup_dilate_frac: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Returns: disc_mask (uint8 0/1), debug dict
        """
        pred_map = infer_label_map(rgb_iso_u8, self.processor, self.model, device=self.device)
        disc_mask = extract_disc_mask_safe(
            pred_map,
            self.model.config.id2label,
            img_shape=rgb_iso_u8.shape,
            cup_dilate_frac=cup_dilate_frac
        )
        # ensure uint8 0/1
        disc_mask = (disc_mask > 0).astype(np.uint8)
        dbg = {
            "pred_map_shape": tuple(pred_map.shape) if hasattr(pred_map, "shape") else None,
            "disc_area_px": int(disc_mask.sum())
        }
        return disc_mask, dbg


# ----------------------------
# stage2 per-image
# ----------------------------
def stage2_geometry_one(
    *,
    image_id: str,
    cfg: PipelineConfig,
    cache_root: str | Path,
    run_id_stage1: str,
    run_id_stage2: str,
    od_ctx: ODContext,
    overwrite: bool = False,
    save_preview_png: bool = True,
) -> Dict[str, Any]:
    """
    Reads Stage1 artifacts (rgb_iso.npy, optional fovea_1hw.npy),
    computes disc_mask + OD center + PD, and saves Stage2 artifacts.
    """
    s1 = stage1_dir(cache_root, run_id_stage1, image_id)
    if not s1.exists():
        raise FileNotFoundError(f"Stage1 folder missing for {image_id}: {s1}")

    out = stage2_dir(cache_root, run_id_stage2, image_id)
    out.mkdir(parents=True, exist_ok=True)

    done_flag = out / "meta.json"
    if done_flag.exists() and not overwrite:
        return {"image_id": image_id, "status": "skipped", "out_dir": str(out)}

    # ---- load inputs from Stage1
    rgb_iso = np.load(s1 / "rgb_iso.npy")  # (H,W,3) uint8
    H, W = rgb_iso.shape[:2]

    # read fovea_path directly from Stage1 meta.json (it already stores it)
    meta1 = json.loads((s1 / "meta.json").read_text())
    fovea_path = meta1.get("fovea_path", None)

    fovea_center_yx = None
    if fovea_path:
        fovea_center_yx = find_fovea_yx_from_cyan_iso(
            fovea_path, image_size=cfg.image_size, tol=40
        )


    # ---- OD segmentation + PD
    disc_mask, dbg = od_ctx.infer_disc(rgb_iso, cup_dilate_frac=cfg.cup_dilate_frac)

    # center + PD
    center_yx, PD_raw, PD_px = center_and_pd_with_bounds(
        disc_mask, rgb_iso.shape, r_frac=cfg.r_frac,
        allow_fallback=True,
        fallback_center_yx=(H / 2.0, W / 2.0),
        fallback_PD_px=0.20 * min(H, W)
    )

    # detect whether fallback likely happened (disc empty or tiny)
    disc_area = int(disc_mask.sum())
    fallback_used = (disc_area < 25)  # heuristic; tune if needed

    # ---- save outputs
    np.save(out / "disc_mask.npy", disc_mask.astype(np.uint8))

    od_json = {
        "center_yx": [float(center_yx[0]), float(center_yx[1])],
        "PD_px_raw": float(PD_raw),
        "PD_px": float(PD_px),
        "disc_area_px": disc_area,
        "fallback_used": bool(fallback_used),
        "H": int(H), "W": int(W),
    }
    (out / "od.json").write_text(json.dumps(od_json, indent=2))

    (out / "fovea.json").write_text(json.dumps({
        "fovea_center_yx": list(fovea_center_yx) if fovea_center_yx is not None else None
    }, indent=2))

    meta = {
        "image_id": image_id,
        "stage1_dir": str(s1),
        "cfg": _cfg_to_dict(cfg),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    # optional quick preview
    if save_preview_png:
        # draw OD mask contour on rgb_iso for fast sanity checks
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(5, 5))
        plt.imshow(rgb_iso)
        plt.contour(disc_mask.astype(float), levels=[0.5], linewidths=1.0)
        plt.scatter([center_yx[1]], [center_yx[0]], s=20)  # x=cx, y=cy
        plt.title(f"{image_id} | PD={float(PD_px):.1f}px | fallback={fallback_used}")
        plt.axis("off")
        fig.tight_layout()
        fig.savefig(out / "preview_od.png", dpi=140)
        plt.close(fig)

    return {"image_id": image_id, "status": "ok", "out_dir": str(out), **od_json}


# ----------------------------
# stage2 batch
# ----------------------------
def stage2_geometry_batch(
    items,
    *,
    cfg: PipelineConfig,
    cache_root: str | Path,
    run_id_stage1: str,
    run_id_stage2: str,
    overwrite: bool = False,
    save_preview_png: bool = False,
):
    """
    items: iterable with item.image_id (same as manifest)
    """
    od_ctx = ODContext()  # load ONCE
    results = []
    for it in items:
        r = stage2_geometry_one(
            image_id=it.image_id,
            cfg=cfg,
            cache_root=cache_root,
            run_id_stage1=run_id_stage1,
            run_id_stage2=run_id_stage2,
            od_ctx=od_ctx,
            overwrite=overwrite,
            save_preview_png=save_preview_png,
        )
        results.append(r)
    return results
