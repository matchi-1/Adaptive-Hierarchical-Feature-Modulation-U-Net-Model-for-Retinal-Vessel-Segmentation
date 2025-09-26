# training/metrics.py
"""
Binary segmentation metrics for retinal vessels (numpy-based).
"""

import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-7  # numerical stability for ratios and division-by-zero safety


# ---------- basic helpers ----------

def _to_numpy_u8(x):
    """
    Convert torch/np input to a binary NumPy mask [H,W] with values {0,1} (uint8).

    Accepts:
        - torch.Tensor in shapes [B,1,H,W], [1,H,W], [H,W], or one-hot [2,H,W]
        - np.ndarray in equivalent shapes

    Rules:
        - For floats: threshold at 0.5
        - For ints/bools: nonzero → 1
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()        # no grad, CPU copy

        if x.ndim == 4:             # [B,1,H,W] -> take first batch
            x = x[0]

        if x.ndim == 3:             # [1,H,W] -> squeeze channel
            if x.shape[0] == 1:     # single-channel
                x = x[0]

            elif x.shape[0] == 2:   # assume one-hot [2,H,W], take channel 1
                x = x[1]

            else:
                x = x.argmax(0)     # multiclass → argmax
        
        x = x.numpy()

    else:
        x = np.asarray(x)

    # Binarize to {0,1}
    if x.dtype.kind in {"f", "c"}:  # float or complex
        x = (x >= 0.5).astype(np.uint8)
    
    else:                           # int, uint, bool
        x = (x != 0).astype(np.uint8)
    
    return x


"""Flatten a mask to 1D after normalizing to uint8 {0,1}."""
def _ravel_u8(x):   return _to_numpy_u8(x).ravel()
    

# ---------- pixel-wise counts & overlap ----------
"""
Compute TP/FP/TN/FN from two binary masks.

Returns:
    (tp, fp, tn, fn) as Python ints
"""
def confusion_counts(pred, target):

    p = _ravel_u8(pred)
    t = _ravel_u8(target)
    tp = int((p & t).sum())
    fp = int((p & (1 - t)).sum())
    tn = int(((1 - p) & (1 - t)).sum())
    fn = int(((1 - p) & t).sum())

    return tp, fp, tn, fn

"""Dice coefficient (a.k.a. F1 for sets)."""
def dice(pred, target):
    p = _ravel_u8(pred); t = _ravel_u8(target)
    inter = int((p & t).sum())

    return (2.0 * inter + EPS) / (p.sum() + t.sum() + EPS)

"""Intersection over Union (Jaccard index)."""
def iou(pred, target):
    p = _ravel_u8(pred); t = _ravel_u8(target)
    inter = int((p & t).sum())
    union = int(p.sum() + t.sum() - inter)

    return (inter + EPS) / (union + EPS)


# ---------- rates from counts (micro-safe) ----------

def precision_from_counts(tp, fp):   return (tp + EPS) / (tp + fp + EPS)
def recall_from_counts(tp, fn):      return (tp + EPS) / (tp + fn + EPS)  # sensitivity/TPR
def specificity_from_counts(tn, fp): return (tn + EPS) / (tn + fp + EPS)  # TNR
def fpr_from_counts(tn, fp):         return (fp + EPS) / (fp + tn + EPS)  # 1 - specificity
def fdr_from_counts(tp, fp):         return (fp + EPS) / (tp + fp + EPS)  # false discovery rate
def f1_from_counts(tp, fp, fn):      return (2*tp + EPS) / (2*tp + fp + fn + EPS)  # == Dice
def iou_from_counts(tp, fp, fn):     return (tp + EPS) / (tp + fp + fn + EPS)
def acc_from_counts(tp, fp, tn, fn): return (tp + tn) / (tp + fp + tn + fn + EPS)


# ---------- topology-aware & vessel splits ----------

def cldice(pred, target):
    """
    Centerline Dice (clDice): measures how well predicted centerlines overlap GT.
    Requires scikit-image for skeletonization.

    Returns:
        float in [0,1]; higher is better.
    """
    try:
        from skimage.morphology import skeletonize
    except Exception as e:
        raise ImportError("scikit-image is required for cldice()") from e

    p = _to_numpy_u8(pred).astype(bool)
    t = _to_numpy_u8(target).astype(bool)

    p_skel = skeletonize(p).astype(np.uint8)    # predicted centerlines
    t_skel = skeletonize(t).astype(np.uint8)    # GT centerlines

    # Topology precision/sensitivity (centerline coverage both ways)
    tprec = (p_skel & t).sum() / (p_skel.sum() + EPS)
    tsens = (t_skel & p).sum() / (t_skel.sum() + EPS)

    return (2 * tprec * tsens) / (tprec + tsens + EPS)

def thin_thick(mask):
    """
    Split a vessel mask into 'thin' (skeleton) and 'thick' (everything else).

    Returns:
        (thin_uint8, thick_uint8), both {0,1} masks.
    """
    try:
        from skimage.morphology import skeletonize
    except Exception as e:
        raise ImportError("scikit-image is required for thin_thick()") from e

    m = _to_numpy_u8(mask).astype(bool)
    skel = skeletonize(m)
    thin  = skel.astype(np.uint8)
    thick = (m & (~skel)).astype(np.uint8)

    return thin, thick


# ---------- threshold utilities ----------

def sweep_thresholds(prob, target, thresholds):
    """
    Evaluate Dice/IoU/counts over multiple thresholds for a single image.

    Args:
        prob (np.ndarray): probability map in [0,1]
        target (array-like): binary mask (will be normalized)
        thresholds (iterable of float): thresholds to test

    Returns:
        list[dict]: per-threshold metrics and counts
    """
    out = []
    t = _to_numpy_u8(target)

    for th in thresholds:
        pred = (np.asarray(prob) >= th).astype(np.uint8)
        tp, fp, tn, fn = confusion_counts(pred, t)
        out.append({
            "th": float(th),
            "dice": dice(pred, t),
            "iou":  iou(pred, t),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn
        })

    return out


# ---------- AUC metrics (probability-based) ----------

def roc_auc(prob, target):
    """ROC AUC (returns np.nan if GT is single-class)."""

    try:
        from sklearn.metrics import roc_auc_score
    except Exception as e:
        raise ImportError("scikit-learn is required for roc_auc()") from e
    
    p = np.asarray(prob).ravel()       # probs in [0,1]
    t = _ravel_u8(target)              # binary labels

    if t.max() == t.min():             # only one class present
        return np.nan
    
    return float(roc_auc_score(t, p))


"""PR AUC (Average Precision); robust under class imbalance."""
def pr_auc(prob, target):
    try:
        from sklearn.metrics import average_precision_score
    except Exception as e:
        raise ImportError("scikit-learn is required for pr_auc() (average_precision_score)") from e
    
    p = np.asarray(prob).ravel()
    t = _ravel_u8(target)
    
    if t.max() == t.min():
        return np.nan
    
    return float(average_precision_score(t, p))


# ---------- simple aggregators ----------

class ConfusionMeter:
    """
    Accumulate TP/FP/TN/FN across many images to compute micro-averaged metrics.

    Usage:
        cm = ConfusionMeter()
        cm.add(pred_mask, gt_mask)   # per image
        metrics = cm.micro()          # pooled metrics over all images
    """

    __slots__ = ("tp", "fp", "tn", "fn")

    def __init__(self):
        self.tp = self.fp = self.tn = self.fn = 0   # running sums


    """Add counts from a pair of masks."""
    def add(self, pred, target):
        tp, fp, tn, fn = confusion_counts(pred, target)
        self.tp += tp; self.fp += fp; self.tn += tn; self.fn += fn


    """Add raw counts directly (useful if you computed them elsewhere)."""
    def add_counts(self, tp, fp, tn, fn):
        self.tp += int(tp); self.fp += int(fp); self.tn += int(tn); self.fn += int(fn)


    def micro(self):
        """
        Compute micro-averaged metrics from pooled counts.
        Returns:
            dict with Precision, Sensitivity, 
            Specificity, F1/Dice, IoU, Accuracy, FPR, FDR
        """
        tp, fp, tn, fn = self.tp, self.fp, self.tn, self.fn
        return {
            "Precision":   precision_from_counts(tp, fp),
            "Sensitivity": recall_from_counts(tp, fn),
            "Specificity": specificity_from_counts(tn, fp),
            "F1/Dice":     f1_from_counts(tp, fp, fn),
            "IoU":         iou_from_counts(tp, fp, fn),
            "Accuracy":    acc_from_counts(tp, fp, tn, fn),
            "FPR":         fpr_from_counts(tn, fp),
            "FDR":         fdr_from_counts(tp, fp),
        }
