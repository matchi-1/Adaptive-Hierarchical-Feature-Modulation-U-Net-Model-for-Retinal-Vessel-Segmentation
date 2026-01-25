# src/retina_biomarkers/pipeline/stage4_biomarkers.py
from __future__ import annotations

import json, hashlib, math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from src.retina_biomarkers.notebook_utils.pipeline.config import PipelineConfig
from src.retina_biomarkers.notebook_utils.pipeline.retina import compute_biomarkers_from_mask_array
from src.retina_biomarkers.isnt_quadrants import isnt_quadrants_masks  # uses disc->fovea frame


# ----------------------------
# helpers
# ----------------------------
def _cfg_to_dict(cfg: Any) -> Dict[str, Any]:
    if hasattr(cfg, "model_dump"):
        return cfg.model_dump()
    if is_dataclass(cfg):
        return asdict(cfg)
    return {k: v for k, v in vars(cfg).items() if not k.startswith("_") and not callable(v)}

def safe_image_id(image_id: str) -> str:
    return image_id.replace("/", "__").replace("\\", "__").replace(":", "_")

def stage1_dir(cache_root: str | Path, run_id_stage1: str, image_id: str) -> Path:
    return Path(cache_root) / run_id_stage1 / "stage1" / safe_image_id(image_id)

def stage2_dir(cache_root: str | Path, run_id_stage2: str, image_id: str) -> Path:
    return Path(cache_root) / run_id_stage2 / "stage2" / safe_image_id(image_id)

def stage3_dir(cache_root: str | Path, run_id_stage3: str, image_id: str) -> Path:
    return Path(cache_root) / run_id_stage3 / "stage3" / safe_image_id(image_id)

def stage4_dir(cache_root: str | Path, run_id_stage4: str, image_id: str) -> Path:
    return Path(cache_root) / run_id_stage4 / "stage4" / safe_image_id(image_id)

def _jsonify(x):
    """Convert numpy types recursively so json.dumps works."""
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): _jsonify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonify(v) for v in x]
    return str(x)

def make_run_id_stage4(
    cfg: PipelineConfig,
    *,
    run_id_stage1: str,
    run_id_stage2: str,
    run_id_stage3: str,
    threshold_mask: float,
    sector_deg: float,
    prefix: str = "aptos2019"
) -> str:
    payload = {
        "cfg": _cfg_to_dict(cfg),
        "run1": run_id_stage1,
        "run2": run_id_stage2,
        "run3": run_id_stage3,
        "thr": float(threshold_mask),
        "sector_deg": float(sector_deg),
    }
    h = hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_stage4_{h}"


# ----------------------------
# core stage4 per-image
# ----------------------------
def stage4_biomarkers_one(
    *,
    image_id: str,
    cfg: PipelineConfig,
    cache_root: str | Path,
    run_id_stage1: str,
    run_id_stage2: str,
    run_id_stage3: str,
    run_id_stage4: str,
    threshold_mask: float = 0.5,
    sector_deg: float = 90.0,
    overwrite: bool = False,
    save_preview_png: bool = False,
) -> Dict[str, Any]:
    """
    Uses cached Stage1/2/3 to compute biomarkers:
      - global (FD, density, tortuosity, caliber) + rings
      - ISNT quadrants (same 4 metrics)
    Saves biomarkers.json + meta.json (+ optional preview png).
    """
    s1 = stage1_dir(cache_root, run_id_stage1, image_id)
    s2 = stage2_dir(cache_root, run_id_stage2, image_id)
    s3 = stage3_dir(cache_root, run_id_stage3, image_id)

    if not s1.exists(): raise FileNotFoundError(f"Missing Stage1: {s1}")
    if not s2.exists(): raise FileNotFoundError(f"Missing Stage2: {s2}")
    if not s3.exists(): raise FileNotFoundError(f"Missing Stage3: {s3}")

    out = stage4_dir(cache_root, run_id_stage4, image_id)
    out.mkdir(parents=True, exist_ok=True)

    done = out / "meta.json"
    if done.exists() and not overwrite:
        return {"image_id": image_id, "status": "skipped", "out_dir": str(out)}

    # ---- Load inputs
    # FOV mask (Stage1)
    fov = (np.load(s1 / "fov_1hw.npy")[0] > 0.5) if (s1 / "fov_1hw.npy").exists() else None

    # OD + PD (Stage2)
    od = json.loads((s2 / "od.json").read_text())
    disc_center_yx = tuple(od["center_yx"])  # (cy, cx)
    PD_px = float(od["PD_px"])

    # fovea center (Stage2)
    fovea_json = json.loads((s2 / "fovea.json").read_text())
    fovea_center_yx = fovea_json.get("fovea_center_yx", None)  # (fy, fx) or None

    # prob map (Stage3) -> pred mask
    prob = np.load(s3 / "prob_map.npy").astype(np.float32)  # (H,W)
    pred_mask = (prob >= float(threshold_mask)).astype(np.uint8)  # 0/1

    if fov is not None:
        pred_mask = (pred_mask.astype(bool) & fov.astype(bool)).astype(np.uint8)

    # ---- Global + rings
    biom_global = compute_biomarkers_from_mask_array(
        pred_mask,
        disc_center=disc_center_yx,
        PD_px=PD_px,
        max_gap_px=cfg.max_gap_px,
        angle_k_ahead=cfg.angle_k_ahead,
        ortho_step=cfg.ortho_step,
        ortho_max_radius=cfg.ortho_max_radius,
    )
    # biom_global is {"image_shape":.., "global":.., "topology":.., "rings":..}

    # ---- Quadrants (ISNT) for 4 metrics
    quadrants = None
    if fovea_center_yx is not None and fov is not None:
        # Build wedge masks in disc->fovea frame, clipped by FOV
        q_masks = isnt_quadrants_masks(
            shape_hw=pred_mask.shape,
            disc_yx=disc_center_yx,
            fovea_yx=tuple(fovea_center_yx),
            sector_deg=float(sector_deg),
            fov=fov.astype(bool),
        )

        quadrants = {}
        base = pred_mask.astype(bool)
        for qk, qm in q_masks.items():
            sub = (base & qm.astype(bool)).astype(np.uint8)

            # compute metrics for the submask (skip rings to save time)
            res = compute_biomarkers_from_mask_array(
                sub,
                disc_center=None,
                PD_px=None,
                max_gap_px=cfg.max_gap_px,
                angle_k_ahead=cfg.angle_k_ahead,
                ortho_step=cfg.ortho_step,
                ortho_max_radius=cfg.ortho_max_radius,
            )

            # normalize density by quadrant support area (FOV ∩ quadrant)
            denom = float(np.count_nonzero(qm.astype(bool) & fov.astype(bool)))
            if denom > 0:
                res["global"]["area_density"] = float(np.count_nonzero(sub)) / denom
            else:
                res["global"]["area_density"] = float("nan")

            quadrants[qk] = res

    # ---- Assemble final output structure (matches your notebook expectations)
    biom_out = {
        "image_id": image_id,
        "od": od,
        "fovea_center_yx": fovea_center_yx,
        "threshold_mask": float(threshold_mask),
        "biomarkers": {
            "global": biom_global["global"],
            "rings": biom_global.get("rings"),
            "quadrants": quadrants,
        }
    }

    (out / "biomarkers.json").write_text(json.dumps(_jsonify(biom_out), indent=2))
    meta = {
        "image_id": image_id,
        "run_id_stage1": run_id_stage1,
        "run_id_stage2": run_id_stage2,
        "run_id_stage3": run_id_stage3,
        "cfg": _cfg_to_dict(cfg),
        "threshold_mask": float(threshold_mask),
        "sector_deg": float(sector_deg),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    # ---- Optional preview
    if save_preview_png:
        rgb = np.load(s1 / "rgb_iso.npy")  # (H,W,3) uint8
        disc = np.load(s2 / "disc_mask.npy") > 0 if (s2 / "disc_mask.npy").exists() else None

        cy, cx = disc_center_yx
        fig = plt.figure(figsize=(6, 6))
        plt.imshow(rgb)
        if fov is not None:
            plt.contour(fov.astype(float), levels=[0.5], linewidths=1.0)
        if disc is not None:
            plt.contour(disc.astype(float), levels=[0.5], linewidths=1.0)
        # mask overlay
        plt.imshow(pred_mask.astype(float), alpha=0.25)

        plt.scatter([cx], [cy], s=35, c="yellow", marker="x")
        if fovea_center_yx is not None:
            fy, fx = fovea_center_yx
            plt.scatter([fx], [fy], s=35, c="cyan", marker="x")

        plt.title(f"{image_id} | thr={threshold_mask}")
        plt.axis("off")
        fig.tight_layout()
        fig.savefig(out / "preview_stage4.png", dpi=140)
        plt.close(fig)

    return {"image_id": image_id, "status": "ok", "out_dir": str(out)}


def stage4_biomarkers_batch(
    items,
    *,
    cfg: PipelineConfig,
    cache_root: str | Path,
    run_id_stage1: str,
    run_id_stage2: str,
    run_id_stage3: str,
    run_id_stage4: str,
    threshold_mask: float = 0.5,
    sector_deg: float = 90.0,
    overwrite: bool = False,
    save_preview_png: bool = False,
):
    results = []
    for it in items:
        r = stage4_biomarkers_one(
            image_id=it.image_id,
            cfg=cfg,
            cache_root=cache_root,
            run_id_stage1=run_id_stage1,
            run_id_stage2=run_id_stage2,
            run_id_stage3=run_id_stage3,
            run_id_stage4=run_id_stage4,
            threshold_mask=threshold_mask,
            sector_deg=sector_deg,
            overwrite=overwrite,
            save_preview_png=save_preview_png,
        )
        results.append(r)
    return results
