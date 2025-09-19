# training/metrics.py
"""
Binary segmentation metrics for retinal vessels (numpy-based).
All functions accept numpy arrays; tensors should be moved to CPU and converted.
"""

import numpy as np

EPS = 1e-7  # numerical stability for ratios


# ---------- basic helpers ----------

def _to_u8_bool(x):
    """Return uint8 {0,1} mask from arbitrary dtype/scale (incl. 0/255)."""
    return (np.asarray(x) > 0).astype(np.uint8)

def _ravel_u8(x):
    """Like ravel(), but ensures uint8 {0,1} first."""
    return _to_u8_bool(x).ravel()


# ---------- pixel-wise counts & overlap ----------

def confusion_counts(pred, target):
    """
    pred, target: binary-like arrays (any dtype) shape [H,W] (or broadcastable).
    Returns integer TP, FP, TN, FN.
    """
    p = _ravel_u8(pred)
    t = _ravel_u8(target)
    tp = int((p & t).sum())
    fp = int((p & (1 - t)).sum())
    tn = int(((1 - p) & (1 - t)).sum())
    fn = int(((1 - p) & t).sum())
    return tp, fp, tn, fn

def dice(pred, target):
    p = _ravel_u8(pred); t = _ravel_u8(target)
    inter = int((p & t).sum())
    return (2.0 * inter + EPS) / (p.sum() + t.sum() + EPS)

def iou(pred, target):
    p = _ravel_u8(pred); t = _ravel_u8(target)
    inter = int((p & t).sum())
    union = int(p.sum() + t.sum() - inter)
    return (inter + EPS) / (union + EPS)


# ---------- rates from counts (micro-safe) ----------

def precision_from_counts(tp, fp):   return (tp + EPS) / (tp + fp + EPS)
def recall_from_counts(tp, fn):      return (tp + EPS) / (tp + fn + EPS)  # sensitivity
def specificity_from_counts(tn, fp): return (tn + EPS) / (tn + fp + EPS)
def fpr_from_counts(tn, fp):         return (fp + EPS) / (fp + tn + EPS)
def fdr_from_counts(tp, fp):         return (fp + EPS) / (tp + fp + EPS)
def f1_from_counts(tp, fp, fn):      return (2*tp + EPS) / (2*tp + fp + fn + EPS)  # == Dice
def iou_from_counts(tp, fp, fn):     return (tp + EPS) / (tp + fp + fn + EPS)
def acc_from_counts(tp, fp, tn, fn): return (tp + tn) / (tp + fp + tn + fn + EPS)


# ---------- topology-aware & vessel splits ----------

def cldice(pred, target):
    """
    Centerline Dice (clDice): topology/centerline alignment.
    Requires scikit-image.
    """
    try:
        from skimage.morphology import skeletonize
    except Exception as e:
        raise ImportError("scikit-image is required for cldice()") from e

    p = _to_u8_bool(pred).astype(bool)
    t = _to_u8_bool(target).astype(bool)

    p_skel = skeletonize(p).astype(np.uint8)
    t_skel = skeletonize(t).astype(np.uint8)

    # topology precision & sensitivity
    tprec = (p_skel & t).sum() / (p_skel.sum() + EPS)
    tsens = (t_skel & p).sum() / (t_skel.sum() + EPS)
    return (2 * tprec * tsens) / (tprec + tsens + EPS)

def thin_thick(mask):
    """
    Split mask into thin (skeleton) and thick (everything else).
    Returns (thin_uint8, thick_uint8).
    """
    try:
        from skimage.morphology import skeletonize
    except Exception as e:
        raise ImportError("scikit-image is required for thin_thick()") from e

    m = _to_u8_bool(mask).astype(bool)
    skel = skeletonize(m)
    thin  = skel.astype(np.uint8)
    thick = (m & (~skel)).astype(np.uint8)
    return thin, thick


# ---------- threshold utilities ----------

def sweep_thresholds(prob, target, thresholds):
    """
    Evaluate Dice/IoU/counts over thresholds for a single image.
    prob: float map in [0,1]; target: binary mask.
    Returns list of dicts per threshold.
    """
    out = []
    t = _to_u8_bool(target)
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
    p = np.asarray(prob).ravel()
    t = _ravel_u8(target)
    if t.max() == t.min():
        return np.nan
    return float(roc_auc_score(t, p))

def pr_auc(prob, target):
    """PR AUC (Average Precision); robust under class imbalance."""
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
    """Accumulate TP/FP/TN/FN across many images (micro metrics)."""
    __slots__ = ("tp", "fp", "tn", "fn")
    def __init__(self):
        self.tp = self.fp = self.tn = self.fn = 0

    def add(self, pred, target):
        tp, fp, tn, fn = confusion_counts(pred, target)
        self.tp += tp; self.fp += fp; self.tn += tn; self.fn += fn

    def add_counts(self, tp, fp, tn, fn):
        self.tp += int(tp); self.fp += int(fp); self.tn += int(tn); self.fn += int(fn)

    def micro(self):
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
