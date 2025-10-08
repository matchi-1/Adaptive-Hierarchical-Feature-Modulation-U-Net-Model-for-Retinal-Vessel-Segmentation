import cv2
import numpy as np
import sys
from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.training.metrics import (
    dice, iou, confusion_counts, recall_from_counts, specificity_from_counts,
    precision_from_counts, fpr_from_counts, fdr_from_counts, acc_from_counts,
    cldice, thin_thick,
)
try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    _HAS_SK = True
except Exception:
    _HAS_SK = False


def _nearest_resize(arr: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor resize for binary arrays (H,W) -> (H2,W2)."""
    return cv2.resize(arr.astype(np.uint8), (out_hw[1], out_hw[0]), interpolation=cv2.INTER_NEAREST)

def _linear_resize(arr: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Bilinear resize for float/prob arrays (H,W) -> (H2,W2)."""
    return cv2.resize(arr.astype(np.float32), (out_hw[1], out_hw[0]), interpolation=cv2.INTER_LINEAR)

def compute_metrics_single(
    pred_probs: np.ndarray,          # (H,W) float in [0,1]
    gt_1hw: np.ndarray,              # (1,H,W) {0,1}
    fov_1hw: np.ndarray | None,      # (1,H,W) {0,1} or None
    *, threshold: float = 0.5,
    compute_auc: bool = True,
) -> dict[str, float]:
    """Return a dict of metrics (Dice, IoU, Acc, Sen, Spe, Precision, FPR, FDR, clDice, Dice_thin, Dice_thick, ROC_AUC, PR_AUC)."""
    # 1) normalize shapes
    gt = (gt_1hw[0] > 0.5).astype(np.uint8)                       # (Hgt,Wgt) {0,1}
    H, W = gt.shape

    probs = pred_probs
    if probs.shape != (H, W):
        probs = _linear_resize(probs, (H, W))                     # (H,W) float

    pred = (probs >= threshold).astype(np.uint8)                  # (H,W) {0,1}

    if fov_1hw is not None:
        fov = (fov_1hw[0] > 0.5).astype(np.uint8)
        if fov.shape != (H, W):
            fov = _nearest_resize(fov, (H, W))
    else:
        fov = np.ones_like(gt, dtype=np.uint8)

    # 2) apply FOV and compute counts
    Pm = pred[fov == 1]
    Gm = gt[fov == 1]

    # Use shared metric helpers for consistency
    tp, fp, tn, fn = confusion_counts(Pm, Gm)

    out: dict[str, float] = {}
    out["Dice"]       = float(dice(Pm, Gm))
    out["IoU"]        = float(iou(Pm, Gm))
    out["Accuracy"]   = float(acc_from_counts(tp, fp, tn, fn))
    out["Sensitivity"]= float(recall_from_counts(tp, fn))
    out["Specificity"]= float(specificity_from_counts(tn, fp))
    out["Precision"]  = float(precision_from_counts(tp, fp))
    out["FPR"]        = float(fpr_from_counts(tn, fp))
    out["FDR"]        = float(fdr_from_counts(tp, fp))

    # Optional topology & thin/thick (skip if dependency missing)
    try:
        out["clDice"] = float(cldice(Pm, Gm))
    except Exception:
        out["clDice"] = float("nan")
    try:
        thin_p, thick_p = thin_thick(Pm)
        thin_g, thick_g = thin_thick(Gm)
        out["Dice_thin"]  = float(dice(thin_p, thin_g))
        out["Dice_thick"] = float(dice(thick_p, thick_g))
    except Exception:
        out["Dice_thin"] = out["Dice_thick"] = float("nan")

    # AUCs (if sklearn present and GT has positives/negatives inside FOV)
    if compute_auc and _HAS_SK and Gm.max() != Gm.min():
        # mask probs to FOV, align size already done
        prob_m = probs[fov == 1]
        try:
            out["ROC_AUC"] = float(roc_auc_score(Gm.astype(np.uint8), prob_m.astype(np.float32)))
        except Exception:
            out["ROC_AUC"] = float("nan")
        try:
            out["PR_AUC"] = float(average_precision_score(Gm.astype(np.uint8), prob_m.astype(np.float32)))
        except Exception:
            out["PR_AUC"] = float("nan")
    else:
        out["ROC_AUC"] = out["PR_AUC"] = float("nan")

    return out

def render_metric_cards(metrics: dict[str, float]):
    """Nice compact card grid using st.metric."""
    st.markdown("#### Prediction Scores (FOV-masked)")
    primary = ["Sensitivity", "Specificity", "clDice", "Precision", "Dice", "IoU", "Accuracy", "Dice_thin", "Dice_thick", "FPR", "FDR", "ROC_AUC", "PR_AUC"]


    # Row 1: primary (2 lines of 3)
    cols = st.columns(3)
    for i, k in enumerate(primary[:3]):
        cols[i].metric(k, f"{metrics.get(k, float('nan')):.3f}")
    cols = st.columns(3)
    for i, k in enumerate(primary[3:6]):
        cols[i].metric(k, f"{metrics.get(k, float('nan')):.3f}")
