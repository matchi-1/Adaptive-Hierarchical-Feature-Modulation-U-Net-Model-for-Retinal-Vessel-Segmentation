# apps/streamlit/lib/biomarkers.py

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- project root (same pattern as app.py) ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# UPDATE THIS: set to the same checkpoint you used in your biomarker notebook
# e.g. "../../outputs/checkpoints/final/[DRIVE] dpcn_refineEdge.pth"
DEFAULT_BIOMARKER_CKPT = "../../outputs/checkpoints/final/[DRIVE] dpcn_refineEdge.pth"

# --- biomarker pipeline imports (same as in notebook) ---
from src.retina_biomarkers.notebook_utils.pipeline.config import PipelineConfig
from src.retina_biomarkers.notebook_utils.pipeline.retina import run_pipeline
from src.retina_biomarkers.notebook_utils.report.compare import (
    draw_overlay_ax,
)


# --------------------- Core helpers --------------------- #

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

DEFAULT_BIOMARKER_CKPT = Path("outputs/checkpoints/[DRIVE] MATHFI.pth")

def _resolve_ckpt_path(ckpt_path: Optional[str] = None) -> str:
    """
    Resolve which checkpoint to use for the biomarker pipeline.
    If an explicit path is provided, use that. Otherwise fall back to
    the default DRIVE checkpoint used in the original notebook.
    """
    if ckpt_path is not None:
        return str(ckpt_path)
    return str(DEFAULT_BIOMARKER_CKPT)


def save_uploaded_to_temp(img_file, subdir: str = "streamlit_biomarkers") -> Path:
    """
    Save a Streamlit UploadedFile / UploadedFileProxy to a temporary path
    so run_pipeline(<path>, ckpt, cfg) can reuse the same code as the notebook.
    """
    tmp_dir = PROJECT_ROOT / "tmp" / subdir
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Keep only the base name (avoid weird paths from browsers)
    name = Path(img_file.name).name
    out_path = tmp_dir / name

    # UploadedFileProxy forwards getvalue() to the underlying UploadedFile
    data = img_file.getvalue()
    with open(out_path, "wb") as f:
        f.write(data)

    return out_path


# ----------------- Flattening: v5 (from notebook) ----------------- #

def flatten_metrics_v5(out: Dict[str, Any], um_per_px: Optional[float] = None) -> Dict[str, float]:
    """
    Take a single pipeline 'out' dict and flatten global, ring, and (if present)
    ISNT quadrant biomarkers into a 1D row of scalars.

    This is adapted directly from your notebook version, just moved into a module.
    """
    bio = out["biomarkers"]
    g = bio["global"]
    t = bio.get("topology", {})
    rings = bio.get("rings") or {}
    # prefer biomarkers.quadrants; fallback to legacy root-level ‘quadrants’
    quads = bio.get("quadrants") or out.get("quadrants") or {}

    row: Dict[str, float] = {}

    # --- GLOBAL ---
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

    # Some topology goodies if present
    angle_mean = (t.get("angles_2PD") or {}).get("angle_mean", t.get("angle_mean", np.nan))
    row["angle_mean_deg"] = float(angle_mean)

    # --- RINGS ---
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

    # --- QUADRANTS (ISNT) ---
    # In the Streamlit app you might not have quadrant info if you
    # haven't enriched with fovea yet, so this is tolerant.
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


# ----------------- Single-image biomarker wrapper ----------------- #

# apps/streamlit/lib/biomarkers.py (for example)



def run_biomarker_pipeline_for_uploaded(
    img_file,
    um_per_px: Optional[float] = None,
    ckpt_path: Optional[str] = None,
):
    """
    Wrapper for the notebook biomarker pipeline that takes a Streamlit UploadedFile.

    ckpt_path:
        - If provided, we use that (dataset-specific MATHFI .pth).
        - If None, we fall back to DEFAULT_BIOMARKER_CKPT.
    """
    ckpt = _resolve_ckpt_path(ckpt_path)

    # this part should match your notebook logic
    cfg = PipelineConfig()
    # depending on your original run_pipeline signature:
    #   run_pipeline(image_path_or_bytes, ckpt_path, cfg=cfg, um_per_px=um_per_px, ...)
    out = run_pipeline(img_file, ckpt, cfg=cfg)

    # whatever you already do to flatten → DataFrame
    df = pd.DataFrame([flatten_metrics_v5(out, um_per_px=um_per_px)])
    return out, df



# ----------------- Figures & tables for Streamlit ----------------- #

def make_skeleton_pd_overlay_figure(
    out: Dict[str, Any],
    title: str = "Skeleton graph with per-PD rings",
) -> plt.Figure:
    """
    Build a single-image figure similar to Figure 3.3:
    fundus (iso), skeleton graph, PD rings overlay.
    """
    H, W = out["rgb_iso"].shape[:2]
    fig, ax = plt.subplots(figsize=(5, 5))
    draw_overlay_ax(ax, out, title)
    ax.set_xlim(0, W - 1)
    ax.set_ylim(H - 1, 0)
    ax.axis("off")
    fig.tight_layout()
    return fig


def extract_global_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn the flattened row into a small 2-column table for global metrics.
    """
    if df.empty:
        return pd.DataFrame(columns=["Retinal vascular parameter", "Value"])

    row = df.iloc[0]
    cal_key = "caliber_um" if "caliber_um" in df.columns else "caliber_px"
    cal_label = "Vascular caliber (µm)" if cal_key == "caliber_um" else "Vascular caliber (px)"

    entries: List[Tuple[str, float]] = [
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
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "Ring (PD)", "Fractal dimension", "Density (%)",
            "Tortuosity (px⁻²)", "Caliber"
        ])

    row = df.iloc[0]

    # Find all unique ring ids from "<ring>|suffix" columns
    ring_ids: List[str] = []
    for c in df.columns:
        if "|" not in c:
            continue
        ring, _suffix = c.split("|", 1)
        ring_ids.append(ring)
    if not ring_ids:
        return pd.DataFrame(columns=[
            "Ring (PD)", "Fractal dimension", "Density (%)",
            "Tortuosity (px⁻²)", "Caliber"
        ])

    ring_ids = sorted(set(ring_ids), key=lambda r: float(r.split("-")[0]))

    rows_out: List[Dict[str, Any]] = []
    for r_id in ring_ids:
        rec: Dict[str, Any] = {"Ring (PD)": r_id}
        rec["Fractal dimension"] = float(row.get(f"{r_id}|fractal_dimension", np.nan))
        rec["Density (%)"] = float(row.get(f"{r_id}|area_density_pct", np.nan))
        rec["Tortuosity (px⁻²)"] = float(row.get(f"{r_id}|tortuosity_px2", np.nan))
        rec["Caliber"] = float(row.get(f"{r_id}|caliber", np.nan))
        rows_out.append(rec)

    return pd.DataFrame(rows_out)
