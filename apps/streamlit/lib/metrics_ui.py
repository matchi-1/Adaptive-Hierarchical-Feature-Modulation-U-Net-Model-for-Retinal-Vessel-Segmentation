from typing import Optional, Dict
import numpy as np
import cv2
import math
import streamlit as st

# metrics module
from src.training.metrics import (
    confusion_counts, dice, iou,
    acc_from_counts, recall_from_counts, specificity_from_counts,
    precision_from_counts, fpr_from_counts, fdr_from_counts,
    cldice, thin_thick, roc_auc, pr_auc
)

def _linear_resize(img: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Resize float image to (H,W) with bilinear."""
    H, W = hw
    return cv2.resize(img.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)

def _nearest_resize(img: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Resize binary/label image to (H,W) with nearest."""
    H, W = hw
    return cv2.resize(img.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)

def compute_metrics_single(
    pred_probs: np.ndarray,          # (H,W) float in [0,1]
    gt_1hw: np.ndarray,              # (1,H,W) {0,1}
    fov_1hw: Optional[np.ndarray],   # (1,H,W) {0,1} or None
    *, threshold: float = 0.5,
    compute_auc: bool = True,
    use_fov: bool = False,           # <-- default: NO FOV masking for metrics
) -> Dict[str, float]:
    """
    Return a dict of metrics:
      Dice, IoU, Accuracy, Sensitivity, Specificity, Precision, FPR, FDR,
      clDice, Dice_thin, Dice_thick, ROC_AUC, PR_AUC

    If use_fov=True and fov_1hw is provided, metrics are computed on 2D FOV-masked
    versions (pred * fov, gt * fov) — preserves shape for clDice/thin_thick.
    """
    # --- normalize GT to 2D uint8 {0,1} and align sizes ---
    gt = (gt_1hw[0] > 0.5).astype(np.uint8)          # (Hgt,Wgt)
    H, W = gt.shape

    probs = pred_probs
    if probs.shape != (H, W):
        probs = _linear_resize(probs, (H, W)).astype(np.float32)

    pred = (probs >= threshold).astype(np.uint8)     # (H,W) {0,1}

    # --- optional FOV masking (keep 2D!) ---
    if use_fov and fov_1hw is not None:
        fov = (fov_1hw[0] > 0.5).astype(np.uint8)
        if fov.shape != (H, W):
            fov = _nearest_resize(fov, (H, W))
        pred_m = (pred & fov).astype(np.uint8)
        gt_m   = (gt   & fov).astype(np.uint8)
        prob_m = (probs * fov).astype(np.float32)
    else:
        pred_m = pred
        gt_m   = gt
        prob_m = probs

    # --- counts & overlap metrics (2D ok; helpers ravel internally) ---
    tp, fp, tn, fn = confusion_counts(pred_m, gt_m)

    out: Dict[str, float] = {}
    out["Dice"]        = float(dice(pred_m, gt_m))
    out["IoU"]         = float(iou(pred_m, gt_m))
    out["Accuracy"]    = float(acc_from_counts(tp, fp, tn, fn))
    out["Sensitivity"] = float(recall_from_counts(tp, fn))
    out["Specificity"] = float(specificity_from_counts(tn, fp))
    out["Precision"]   = float(precision_from_counts(tp, fp))
    out["FPR"]         = float(fpr_from_counts(tn, fp))
    out["FDR"]         = float(fdr_from_counts(tp, fp))

    # --- topology & thin/thick (require scikit-image) ---
    try:
        out["clDice"] = float(cldice(pred_m, gt_m))
    except Exception:
        out["clDice"] = float("nan")

    try:
        thin_p, thick_p = thin_thick(pred_m)
        thin_g, thick_g = thin_thick(gt_m)
        out["Dice_thin"]  = float(dice(thin_p, thin_g))
        out["Dice_thick"] = float(dice(thick_p, thick_g))
    except Exception:
        out["Dice_thin"] = out["Dice_thick"] = float("nan")

    # --- AUCs (probability-based; metrics helpers already return NaN if single-class) ---
    if compute_auc:
        try:
            out["ROC_AUC"] = float(roc_auc(prob_m, gt_m))
        except Exception:
            out["ROC_AUC"] = float("nan")
        try:
            out["PR_AUC"]  = float(pr_auc(prob_m, gt_m))
        except Exception:
            out["PR_AUC"]  = float("nan")
    else:
        out["ROC_AUC"] = out["PR_AUC"] = float("nan")

    return out

def _fmt(v):
        try:
            if v is None: return "—"
            v = float(v)
            if math.isnan(v) or math.isinf(v): return "—"
            return f"{v:.3f}"
        except Exception:
            return "—"

def render_metric_cards_main(metrics: dict[str, float], model_name):
    """
    Compact, readable metrics grid.
    Order:
      Row 1: Sensitivity, Specificity, clDice
      Row 2: Accuracy, IoU, ROC_AUC
    Extra metrics (if present) go under an expander.
    """

    st.markdown(f"#### Prediction Metric Scores ({model_name})")

    # Primary rows (3 × 2)
    row1 = ["Sensitivity", "Specificity", "clDice"]
    row2 = ["Accuracy", "IoU", "ROC_AUC"]

    cols = st.columns(3)
    for col, k in zip(cols, row1):
        col.metric(k, _fmt(metrics.get(k)))

    cols = st.columns(3)
    for col, k in zip(cols, row2):
        col.metric(k, _fmt(metrics.get(k)))

def render_metric_cards_others(metrics: dict[str, float], model_name):
    rest_keys = [
        "Precision", "Dice", 
        "FPR", "FDR", "Dice_thin", "Dice_thick", #"PR_AUC",
    ]
    any_rest = any(k in metrics for k in rest_keys)
    if any_rest:
        st.markdown(f"#### Extended metrics ({model_name})")
        cols = st.columns(3)
        for i, k in enumerate(rest_keys):
            if k in metrics:
                cols[i % 3].metric(k, _fmt(metrics.get(k)))


def render_delta_cards_grid(metrics_m: dict, metrics_u: dict, *, keys=None, title: str | None=None):
    """
    Δ (MATHFI − UNet) as a 2×3 grid with colored number + arrow and proper row spacing.
    Pass `keys` to control which metrics are shown (max 6 shown, 2 rows × 3 cols).
    """
    import math
    import streamlit as st

    # default = "main" set
    if keys is None:
        keys = ["Sensitivity", "Specificity", "clDice",
                "Accuracy", "IoU", "ROC_AUC"]

    # keep at most 6 to fit 2×3
    keys = [k for k in keys if k]  # drop blanks
    keys = keys[:6]

    def _delta(k: str):
        try:
            m = float(metrics_m.get(k, float("nan")))
            u = float(metrics_u.get(k, float("nan")))
            dv = m - u
            if math.isnan(dv) or math.isinf(dv):
                return None
            return dv
        except Exception:
            return None

    def _cell(col, name: str, dv: float | None):
        # label
        label_html = f"<div style='font-family: var(--font, inherit); font-size:0.90rem; color:white;'>{name}</div>"

        # value (+ arrow) — keep Streamlit default font
        if dv is None:
            val_html = "<div style='font-family: var(--font, inherit); font-size:2rem; font-weight:600; color:#6b7280'>—</div>"
        else:
            if dv > 0:
                color, arrow, val = "#16a34a", "↑ ", abs(dv)   # green
            elif dv < 0:
                color, arrow, val = "#ef4444", "↓ ", abs(dv)   # red
            else:
                color, arrow, val = "#6b7280", "– ", 0.0       # gray
            val_html = (
                f"<div style='font-family: var(--font, inherit); font-size:2rem; font-weight:600; color:{color}'>"
                f"{arrow}{val:.3f}"
                f"</div>"
            )

        col.markdown(label_html + val_html, unsafe_allow_html=True)

    # First row
    cols = st.columns(3)
    for col, k in zip(cols, keys[:3]):
        _cell(col, k, _delta(k))
    # gap between rows
    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
    # Second row
    cols = st.columns(3)
    for col, k in zip(cols, keys[3:6]):
        _cell(col, k, _delta(k))


def render_delta_cards_extended(metrics_m: dict, metrics_u: dict):
    # Preferred extended order (we’ll show the first 6 that are available)
    EXTENDED_ORDER = ["Precision", "Dice", "FPR", "FDR", "Dice_thin", "Dice_thick"]
    keys = [k for k in EXTENDED_ORDER if (k in metrics_m or k in metrics_u)][:6]
    render_delta_cards_grid(metrics_m, metrics_u, keys=keys)
