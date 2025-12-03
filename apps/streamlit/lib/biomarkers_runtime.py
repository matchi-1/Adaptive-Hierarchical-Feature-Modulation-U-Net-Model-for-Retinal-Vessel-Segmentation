# apps/streamlit/lib/biomarkers_runtime.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 👉 adjust these imports to match your actual repo layout
from src.retina_biomarkers.notebook_utils.pipeline.retina import (
    compute_biomarkers_from_mask_array,   # the function you pasted
    center_and_pd_with_bounds,
)
from src.retina_biomarkers.od_seg.postproc import extract_disc_mask_safe 
from src.retina_biomarkers.notebook_utils.report.compare import draw_overlay_ax
from src.data.preprocessing import _iso_resize_and_pad


# -------------------- Config for biometrics -------------------- #

@dataclass
class BiomarkerRuntimeConfig:
    # geometric / PD
    r_frac: Tuple[float, float] = (0.08, 0.16)
    fallback_PD_frac: float = 0.20  # PD ≈ 0.20 * min(H,W) when no disc

    # vessel metrics
    max_gap_px: int = 12
    angle_k_ahead: int = 3
    ortho_step: float = 0.5
    ortho_max_radius: float = 20.0


# ---------------- Core: compute biometrics from mask ---------------- #

def compute_biomarkers_from_segmentation(
    rgb_iso: np.ndarray,
    pred_mask: np.ndarray,
    od_pred_map: Optional[np.ndarray] = None,
    od_id2label: Optional[dict] = None,
    *,
    max_gap_px: int = 12,
    angle_k_ahead: int = 3,
    ortho_step: float = 0.5,
    ortho_max_radius: float = 20.0,
) -> Dict[str, Any]:
    """
    Take the current MATHFI segmentation (binary mask) + an RGB base image
    and compute:
      - OD center + PD (if OD map given; otherwise geometric fallback)
      - global + per-ring + per-quadrant biomarkers (via compute_biomarkers_from_mask_array)

    IMPORTANT: This function makes sure that rgb_iso is resized/padded to
    exactly match pred_mask.shape, so draw_overlay_ax can safely do overlays.
    """
    # ---- 0) Normalize mask to 0/1 uint8 ----
    mask = (np.asarray(pred_mask) > 0).astype(np.uint8)
    Hm, Wm = mask.shape[:2]

    # ---- 1) Force base RGB to same geometry as mask ----
    base_rgb = np.asarray(rgb_iso)
    if base_rgb.ndim == 2:  # grayscale → 3-channel
        base_rgb = np.repeat(base_rgb[..., None], 3, axis=2)

    Hr, Wr = base_rgb.shape[:2]
    if (Hr, Wr) != (Hm, Wm):
        # Use the same iso-resize + pad strategy as the main pipeline
        base_rgb = _iso_resize_and_pad(
            base_rgb,
            target=Hm,     # mask is square from the model (e.g. 512×512)
            pad_value=0,
        ).astype(np.uint8)

    # ---- 2) OD mask + PD estimation (if OD prediction exists) ----
    disc_mask = None
    center_yx = None
    PD_raw = None
    PD_px = None

    if od_pred_map is not None and od_id2label is not None:
        # OD SegFormer output case
        disc_mask = extract_disc_mask_safe(
            pred_map=np.asarray(od_pred_map),
            id2label=od_id2label,
            img_shape=mask.shape,
        )
        center_yx, PD_raw, PD_px = center_and_pd_with_bounds(
            disc_mask,
            img_shape=mask.shape,
            allow_fallback=True,
            fallback_center_yx=(Hm / 2.0, Wm / 2.0),
            fallback_PD_px=0.20 * min(Hm, Wm),
        )
    else:
        # No OD model wired in → geometric fallback (center = image center; PD ≈ 0.2 × min dim)
        center_yx, PD_raw, PD_px = center_and_pd_with_bounds(
            disc_bin=np.zeros_like(mask, dtype=np.uint8),
            img_shape=mask.shape,
            allow_fallback=True,
            fallback_center_yx=(Hm / 2.0, Wm / 2.0),
            fallback_PD_px=0.20 * min(Hm, Wm),
        )
        disc_mask = np.zeros_like(mask, dtype=np.uint8)

    # ---- 3) Core biomarker computation (skeleton, rings, topology, etc.) ----
    biom = compute_biomarkers_from_mask_array(
        mask,
        disc_center=center_yx,
        PD_px=float(PD_px) if PD_px is not None else None,
        max_gap_px=max_gap_px,
        angle_k_ahead=angle_k_ahead,
        ortho_step=ortho_step,
        ortho_max_radius=ortho_max_radius,
    )

    # ---- 4) Package result dict in the same structure as your notebook pipeline ----
    out = {
        "od": {
            "center_yx": (
                float(center_yx[0]),
                float(center_yx[1]),
            ) if center_yx is not None else None,
            "PD_px_raw": float(PD_raw) if PD_raw is not None else None,
            "PD_px": float(PD_px) if PD_px is not None else None,
        },
        "biomarkers": biom,
        "rgb_iso": base_rgb,  # <-- geometry now matches mask
        "disc_mask": disc_mask.astype(np.uint8),
        "pred_mask": mask.astype(np.uint8),
        "pred_mask_thr05": mask.astype(np.uint8),
    }

    return out



# ----------------- Overlay figure (skeleton + PD rings) ----------------- #

def make_skeleton_pd_overlay_figure(
    out: Dict[str, Any],
    title: str = "Skeleton graph + PD rings",
) -> plt.Figure:
    """
    Reuse your original `draw_overlay_ax` to render:
      - vessel skeleton + centerlines
      - orthogonal width samples (red segments)
      - disc mask + PD rings
      - background fundus
    """
    H, W = out["rgb_iso"].shape[:2]
    fig, ax = plt.subplots(figsize=(5, 5))
    draw_overlay_ax(ax, out, title)
    ax.set_xlim(0, W - 1)
    ax.set_ylim(H - 1, 0)
    ax.axis("off")
    fig.tight_layout()
    return fig


# ----------------- Flatten + tables (reusing your v5 logic) ----------------- #

def flatten_metrics_v5(out: Dict[str, Any], um_per_px: Optional[float] = None) -> Dict[str, float]:
    """
    Your v5 flattener, just moved here so app + notebook can share it.
    """
    bio = out["biomarkers"]
    g = bio["global"]
    t = bio.get("topology", {})
    rings = bio.get("rings") or {}
    quads = bio.get("quadrants") or out.get("quadrants") or {}

    row: Dict[str, float] = {}

    # GLOBAL
    row["fractal_dimension"] = float(g.get("fractal_dimension", np.nan))
    row["area_density_pct"] = 100.0 * float(g.get("area_density", np.nan))
    row["tortuosity_px2"] = float(g.get("tortuosity_mean", np.nan))

    vco = g.get("vc_orth") or {}
    med_w_px = float(vco.get("median_width", np.nan))
    iqr_w_px = float(vco.get("iqr_width", np.nan))

    if um_per_px is not None and np.isfinite(med_w_px):
        row["caliber_um"] = med_w_px * um_per_px
    else:
        row["caliber_px"] = med_w_px

    row["median_width_px"] = med_w_px
    row["iqr_width_px"] = iqr_w_px

    angle_mean = (t.get("angles_2PD") or {}).get("angle_mean", t.get("angle_mean", np.nan))
    row["angle_mean_deg"] = float(angle_mean)

    # RINGS (0.5–2.0 PD)
    wanted_starts = [0.5, 1.0, 1.5, 2.0]
    for k, r in rings.items():
        try:
            r0 = float(k.split("-")[0])
        except Exception:
            continue
        if r0 not in wanted_starts:
            continue

        row[f"{k}|fractal_dimension"] = float(r.get("fractal_dimension", np.nan))
        row[f"{k}|area_density_pct"] = 100.0 * float(r.get("area_density", np.nan))
        row[f"{k}|tortuosity_px2"] = float(r.get("tortuosity_mean", np.nan))

        r_med = float(r.get("median_width", np.nan))
        if um_per_px is not None and np.isfinite(r_med):
            row[f"{k}|caliber"] = r_med * um_per_px
            row[f"{k}|caliber_unit"] = "µm"
        else:
            row[f"{k}|caliber"] = r_med
            row[f"{k}|caliber_unit"] = "px"

    # QUADRANTS (ISNT)
    for Q in ["I", "S", "N", "T"]:
        qg = (quads.get(Q) or {}).get("global") or {}
        qvco = (qg.get("vc_orth") or {})

        row[f"{Q}|fractal_dimension"] = float(qg.get("fractal_dimension", np.nan))
        row[f"{Q}|area_density_pct"] = 100.0 * float(qg.get("area_density", np.nan))
        row[f"{Q}|tortuosity_px2"] = float(qg.get("tortuosity_mean", np.nan))

        q_med = float(qvco.get("median_width", np.nan))
        if um_per_px is not None and np.isfinite(q_med):
            row[f"{Q}|caliber"] = q_med * um_per_px
        else:
            row[f"{Q}|caliber"] = q_med

    return row


def extract_global_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Retinal vascular parameter", "Value"])

    row = df.iloc[0]
    cal_key = "caliber_um" if "caliber_um" in df.columns else "caliber_px"
    cal_label = "Vascular caliber (µm)" if cal_key == "caliber_um" else "Vascular caliber (px)"

    entries = [
        ("Vascular fractal dimension", float(row.get("fractal_dimension", np.nan))),
        ("Vascular density (%)", float(row.get("area_density_pct", np.nan))),
        ("Vascular tortuosity (px⁻²)", float(row.get("tortuosity_px2", np.nan))),
        (cal_label, float(row.get(cal_key, np.nan))),
        ("Median width (px, orth)", float(row.get("median_width_px", np.nan))),
        ("IQR width (px, orth)", float(row.get("iqr_width_px", np.nan))),
    ]
    return pd.DataFrame(entries, columns=["Retinal vascular parameter", "Value"])


def extract_ring_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-PD ring table for the selected image.

    Columns: Ring (PD), Fractal dimension, Density (%), Tortuosity (px⁻²), Caliber.
    Only ring-style columns like "0.5-1.0 PD|fractal_dimension" are used;
    quadrant-style columns like "I|fractal_dimension" are ignored.
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "Ring (PD)", "Fractal dimension", "Density (%)",
            "Tortuosity (px⁻²)", "Caliber"
        ])

    row = df.iloc[0]

    # Collect only ring IDs where the left side starts with a numeric token
    ring_ids = set()
    for c in df.columns:
        if "|" not in c:
            continue
        left = c.split("|", 1)[0]  # e.g. "0.5-1.0 PD" or "I"
        # Skip quadrant keys like "I", "S", "N", "T"
        try:
            float(left.split("-")[0])   # will fail on "I", "S", etc.
        except Exception:
            continue
        ring_ids.add(left)

    if not ring_ids:
        return pd.DataFrame(columns=[
            "Ring (PD)", "Fractal dimension", "Density (%)",
            "Tortuosity (px⁻²)", "Caliber"
        ])

    # Sort rings by their starting PD value (e.g., 0.5, 1.0, 1.5, 2.0, ...)
    ring_ids = sorted(ring_ids, key=lambda r: float(r.split("-")[0]))

    rows_out = []
    for r_id in ring_ids:
        rows_out.append({
            "Ring (PD)": r_id,
            "Fractal dimension": float(row.get(f"{r_id}|fractal_dimension", np.nan)),
            "Density (%)":       float(row.get(f"{r_id}|area_density_pct", np.nan)),
            "Tortuosity (px⁻²)": float(row.get(f"{r_id}|tortuosity_px2", np.nan)),
            "Caliber":           float(row.get(f"{r_id}|caliber", np.nan)),
        })

    return pd.DataFrame(rows_out)
