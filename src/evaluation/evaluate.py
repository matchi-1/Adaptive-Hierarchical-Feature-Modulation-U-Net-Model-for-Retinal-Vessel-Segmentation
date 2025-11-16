import numpy as np
import torch
import torch.nn.functional as F
from src.training.metrics import *

# ------------------------------
# helpers: normalize to numpy
# ------------------------------

def _resize_mask_like(mask_np: np.ndarray, ref_hw: tuple[int,int]) -> np.ndarray:
    """
    Resize a binary mask to (H,W) using nearest; returns uint8 {0,1}.
    """
    if tuple(mask_np.shape[-2:]) == tuple(ref_hw):
        return (mask_np > 0.5).astype(np.uint8)
    t = torch.from_numpy(mask_np)[None, None].float()
    t = F.interpolate(t, size=ref_hw, mode="nearest")
    return (t[0,0].cpu().numpy() > 0.5).astype(np.uint8)


def _to_numpy_u8_2d(x) -> np.ndarray:
    """
    Compute micro-averaged metrics from pooled counts.
    Returns:
        dict with Precision, Sensitivity, Specificity, F1/Dice, IoU, Accuracy, FPR, FDR
    """

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()    # no grad; move to CPU
        if x.ndim == 4:         # [B,1,H,W] -> take first in batch
            x = x[0]
        
        if x.ndim == 3:         # [C,H,W]
            C = x.shape[0]
            
            if C == 1:          # single-channel
                x = x[0]
            
            elif C == 2:        # one-hot bg/fg -> take foreground
                x = x[1]
            
            else:               # multiclass → argmax over channels
                x = x.argmax(dim=0)
        
        x = x.numpy()
    
    else:
        x = np.asarray(x)

    # Binarize to {0,1}; treat floats as probabilities
    if x.dtype.kind in {"f", "c"}:
        x = (x >= 0.5).astype(np.uint8)
    
    else:
        x = (x != 0).astype(np.uint8)
    
    return x  # [H,W], uint8 {0,1}


# ------------------------------
# main evaluation
# ------------------------------

@torch.no_grad()
def evaluate_and_print(model, test_dataloader, device="cuda", threshold=0.5, compute_auc=True, apply_sigmoid=True):
    """
    Evaluate a segmentation model on a DataLoader and print macro/micro metrics.

    Inputs (each batch is a dict):
        'image' : torch.FloatTensor [B,1,H,W] — model inputs
        'mask'  : torch/np          [B,1,H,W] — binary ground-truth masks
        (optional) 'fov', 'image_path', etc. are ignored here

    Behavior:
        - Converts logits to probabilities, then thresholds at `threshold` to make masks.
        - Computes per-image metrics (macro = mean over images) and pooled metrics
          using summed TP/FP/TN/FN (micro).
        - Optionally computes ROC AUC and PR AUC (macro and micro) if sklearn is present.

    Printed sections:
        -- Macro (per-image mean) --
        -- Micro (pooled over all pixels) --
    """
    model.eval()
    model.to(device)

    # Macro accumulators (sum values to later divide by image count)
    macro_sums = {
        "Sensitivity": 0.0, "Specificity": 0.0, "Dice": 0.0, "Accuracy": 0.0,
        "IoU": 0.0, "Precision": 0.0, "FPR": 0.0, "FDR": 0.0, 
        "clDice": 0.0, "Dice_thin": 0.0, "Dice_thick": 0.0,
    }
    macro_sums_fov = {
        "Sensitivity": 0.0, "Specificity": 0.0, "Dice": 0.0, "Accuracy": 0.0,
        "IoU": 0.0, "Precision": 0.0, "FPR": 0.0, "FDR": 0.0, 
        "clDice": 0.0, "Dice_thin": 0.0, "Dice_thick": 0.0,
    }
    n_macro = 0

    # Micro aggregator (pooled counts over all images)
    cm = ConfusionMeter()

    # For AUC metrics (micro over all pixels; macro as per-image mean)
    probs_all, gts_all = [], []
    roc_sum = 0.0; pr_sum = 0.0; roc_n = 0; pr_n = 0

    for batch in test_dataloader:
        imgs = batch["image"].to(device)     # [B,1,H,W] inputs on target device
        gts  = batch["mask"]                 # [B,1,H,W] GT kept on CPU side for metrics

        # Forward pass: get logits (may be dict, tensor, or list/tuple)
        out = model(imgs)

        # If model returns a dict (e.g., {"logits", "edge_logits", "skel_logits"})
        if isinstance(out, dict):
            logits = out["logits"]
        else:
            logits = out

        # If model returns multiple outputs, use the last one
        if isinstance(logits, (list, tuple)):
            logits = logits[-1]

        # Convert logits to probabilities in [0,1]
        if logits.ndim == 4 and logits.shape[1] == 2:
            probs = torch.softmax(logits, dim=1)[:, 1]    # [B,H,W]
        
        elif logits.ndim == 4 and logits.shape[1] == 1:
                if apply_sigmoid: probs = torch.sigmoid(logits[:, 0])
                else: probs = logits[:,0]           # [B,H,W]
        
        elif logits.ndim == 3:               # [B,H,W]
            if apply_sigmoid: probs = torch.sigmoid(logits)                 # [B,H,W]
            else: probs = logits
        else:
            raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")

        # Threshold to get binary predictions
        preds = (probs >= threshold).to(torch.uint8)      # [B,H,W]

        B = preds.shape[0]
        for i in range(B):
            pred_np = _to_numpy_u8_2d(preds[i])           # [H,W] uint8
            gt_np   = _to_numpy_u8_2d(gts[i])             # [H,W] uint8

            # If spatial sizes mismatch, resize prediction to GT (nearest preserves binary)
            if pred_np.shape != gt_np.shape:
                # resize prediction to GT using nearest (preserves binary)
                ph, pw = pred_np.shape
                gh, gw = gt_np.shape
                pred_t = torch.from_numpy(pred_np)[None, None].float()
                pred_t = F.interpolate(pred_t, size=(gh, gw), mode="nearest")
                pred_np = (pred_t[0,0] > 0.5).byte().cpu().numpy()

            # ---- per-image (macro) ----
            macro_sums["Dice"]  += dice(pred_np, gt_np)
            macro_sums["IoU"]   += iou(pred_np, gt_np)

            # Confusion-derived rates (computed from counts for consistency)
            tp, fp, tn, fn            = confusion_counts(pred_np, gt_np)
            macro_sums["Sensitivity"] += recall_from_counts(tp, fn)
            macro_sums["Specificity"] += specificity_from_counts(tn, fp)
            macro_sums["Precision"]   += precision_from_counts(tp, fp)
            macro_sums["FPR"]         += fpr_from_counts(tn, fp)
            macro_sums["FDR"]         += fdr_from_counts(tp, fp)
            macro_sums["Accuracy"]    += acc_from_counts(tp, fp, tn, fn)

            # Topology/centerline and thin/thick splits (skip if skimage unavailable)
            try:
                macro_sums["clDice"]  += cldice(pred_np, gt_np)
            except Exception:
                pass  # scikit-image not installed

            try:
                thin_p, thick_p = thin_thick(pred_np)
                thin_t, thick_t = thin_thick(gt_np)
                macro_sums["Dice_thin"]  += dice(thin_p, thin_t)
                macro_sums["Dice_thick"] += dice(thick_p, thick_t)
            except Exception:
                pass  # scikit-image not installed

            n_macro += 1    # counting images processed

            # ---- micro (pooled) ----
            cm.add_counts(tp, fp, tn, fn)

            # ---- AUC (macro means) ----
            if compute_auc:
                # per-image prob (first item in batch already used above,
                # but we want the matching probs for this sample)
                prob_i = probs[i].detach().cpu().numpy()  # [H,W] float
                
                if prob_i.shape != gt_np.shape:
                    prob_t = torch.from_numpy(prob_i)[None, None].float()
                    prob_t = F.interpolate(prob_t, size=gt_np.shape, mode="bilinear", align_corners=False)
                    prob_i = prob_t[0,0].cpu().numpy()
                
                try:
                    auc_roc = roc_auc(prob_i, gt_np)
                    
                    if not np.isnan(auc_roc): roc_sum += auc_roc; roc_n += 1
                    auc_pr  = pr_auc(prob_i, gt_np)
                    
                    if not np.isnan(auc_pr):  pr_sum  += auc_pr;  pr_n  += 1
                
                except Exception:
                    pass  # sklearn not installed

                # for micro AUC
                probs_all.append(prob_i.ravel())
                gts_all.append(gt_np.ravel())

    # ---- finalize macro ----
    macro = {k: (v / max(n_macro, 1)) for k, v in macro_sums.items()}
    if compute_auc and roc_n:
        macro["ROC_AUC"] = roc_sum / roc_n
    
    if compute_auc and pr_n:
        macro["PR_AUC"]  = pr_sum  / pr_n

    # ---- finalize micro ----
    micro = cm.micro()
    if compute_auc and len(probs_all):
        probs_all = np.concatenate(probs_all, axis=0)
        gts_all   = np.concatenate(gts_all,   axis=0)
        
        if gts_all.max() != gts_all.min():
            
            try:
                from sklearn.metrics import roc_auc_score, average_precision_score
                micro["ROC_AUC"] = float(roc_auc_score(gts_all, probs_all))
                micro["PR_AUC"]  = float(average_precision_score(gts_all, probs_all))
            
            except Exception:
                pass

    # ---- print nicely ----
    print("=== Test Set Evaluation Metrics ===")
    print("-- Macro (per-image mean) --")
    
    for k in ["Sensitivity","Specificity","clDice","Accuracy","Dice", "IoU","Precision","FPR","FDR",
              "Dice_thin","Dice_thick","ROC_AUC","PR_AUC"]:
        
        if k in macro:
            print(f"{k:15s}: {macro[k]:.4f}")

    print("-- Micro (pooled over all pixels) --")
    
    for k in ["Sensitivity","Specificity","F1/Dice","Accuracy", "IoU",
              "Precision","FPR","FDR","ROC_AUC","PR_AUC"]:
        
        if k in micro:
            print(f"{k:15s}: {micro[k]:.4f}")

    return dict(ACC=macro["Accuracy"], 
                Dice=macro["Dice"], 
                IoU=macro["IoU"], 
                SEN=macro["Sensitivity"], 
                SPE=macro["Specificity"], 
                AUC=macro["ROC_AUC"], 
                AP=macro["PR_AUC"],
                CLDICE=macro["clDice"],
                tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))


from typing import List, Dict, Tuple


@torch.no_grad()
def evaluate_models_table(
    models: List,                 # list of visualization.ModelEntry (name, model, threshold, device)
    dataloader,
    *,
    average: str = "macro",       # "macro" (per-image mean) or "micro" (pooled)
    compute_auc: bool = True,
    image_key: str = "image",
    mask_key: str = "mask",
    fov_key: str  = "fov",
    decimals: int = 4,
    mark_best: bool = True,
    # Base comparison controls
    base_name: str = "BASE",             # compare everything to this model column (by name)
    rel_tol: float = 0.002,              # ~0.2% relative tolerance to call “same”
    abs_tol: float | None = None,        # optional absolute tolerance; if None we derive from `decimals`
    arrow_symbols: tuple[str, str, str] = ("↑", "↓", "–")
) -> Tuple["pd.DataFrame", "pd.DataFrame"]:
    """
    Evaluate multiple models and return a table of metrics.

    Columns = model names (ModelEntry.name)
    Rows    = metric names (Dice, IoU, Sensitivity, ...)

    Returns:
        (df_numeric, df_pretty)
            df_numeric : pandas.DataFrame of floats (NaN where not available)
            df_pretty  : same shape, values as strings rounded to `decimals`,
                         with a ★ marker on the best value in each row
                         (max by default; min for FPR/FDR).
    """
    # Import here to avoid hard dependency if user only wants printing
    import pandas as pd

    def _fmt(x: float) -> str:
        return "NaN" if pd.isna(x) else f"{x:.{decimals}f}"

    # Which metrics we try to report and in what order:
    macro_order = [
        "Sensitivity","Specificity", "Dice","clDice","Dice_thin","Dice_thick",
        "Accuracy","IoU","Precision","FPR","FDR","ROC_AUC","PR_AUC"
    ]

    micro_order = [
        "Sensitivity","Specificity","F1/Dice", "Precision","IoU","Accuracy",
        "FPR","FDR","ROC_AUC","PR_AUC"
    ]

    order = macro_order if average.lower() == "macro" else micro_order

    # Decide which direction is "better"
    # By default: larger is better; for FPR and FDR: smaller is better
    minimize = {"FPR", "FDR"}

    results: Dict[str, Dict[str, float]] = {}

    # Evaluate one model over the whole dataloader and return (macro, micro)
    def _eval_one(entry) -> Tuple[Dict[str, float], Dict[str, float]]:
        mdl = entry.model
        dev = entry.device or ("cuda" if torch.cuda.is_available() else "cpu")
        thr = float(getattr(entry, "threshold", 0.5))

        mdl.to(dev).eval()

        # Macro accumulators
        macro_sums = {
            "Dice": 0.0, "IoU": 0.0, "Sensitivity": 0.0, "Specificity": 0.0,
            "Precision": 0.0, "FPR": 0.0, "FDR": 0.0, "Accuracy": 0.0,
            "clDice": 0.0, "Dice_thin": 0.0, "Dice_thick": 0.0,
        }
        n_macro = 0

        # Micro confusion aggregator
        cm = ConfusionMeter()

        # For AUC
        probs_all, gts_all = [], []
        roc_sum = 0.0; pr_sum = 0.0; roc_n = 0; pr_n = 0

        for batch in dataloader:
            imgs = batch[image_key].to(dev)          # [B,1,H,W] or [B,C,H,W]
            gts  = batch[mask_key]                   # kept on CPU
            fovs = batch.get(fov_key, None)

            # Forward
            out = mdl(imgs)

            if isinstance(out, dict):
                logits = out["logits"]
            else:
                logits = out

            if isinstance(logits, (list, tuple)):
                logits = logits[-1]


            # Convert to probs in [0,1]
            if logits.ndim == 4 and logits.shape[1] == 2:
                probs = torch.softmax(logits, dim=1)[:, 1]
            
            elif logits.ndim == 4 and logits.shape[1] == 1:
                probs = torch.sigmoid(logits[:, 0])
            
            elif logits.ndim == 3:
                probs = torch.sigmoid(logits)
            
            else:
                raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")

            # Threshold → binary predictions (uint8)
            preds = (probs >= thr).to(torch.uint8)   # [B,H,W]

            B = preds.shape[0]
            for i in range(B):
                pred_np = _to_numpy_u8_2d(preds[i])  # from evaluate.py helper
                gt_np   = _to_numpy_u8_2d(gts[i])

                # Resize pred to GT if needed (nearest to preserve binary)
                if pred_np.shape != gt_np.shape:
                    ph, pw = pred_np.shape
                    gh, gw = gt_np.shape
                    pred_t = torch.from_numpy(pred_np)[None, None].float()
                    pred_t = F.interpolate(pred_t, size=(gh, gw), mode="nearest")
                    pred_np = (pred_t[0,0] > 0.5).byte().cpu().numpy()

                # Macro (per-image)
                macro_sums["Dice"]  += dice(pred_np, gt_np)
                macro_sums["IoU"]   += iou(pred_np, gt_np)
                tp, fp, tn, fn = confusion_counts(pred_np, gt_np)
                macro_sums["Sensitivity"] += recall_from_counts(tp, fn)
                macro_sums["Specificity"] += specificity_from_counts(tn, fp)
                macro_sums["Precision"]   += precision_from_counts(tp, fp)
                macro_sums["FPR"]         += fpr_from_counts(tn, fp)
                macro_sums["FDR"]         += fdr_from_counts(tp, fp)
                macro_sums["Accuracy"]    += acc_from_counts(tp, fp, tn, fn)

                # Optional topological & thin/thick
                try:
                    macro_sums["clDice"]  += cldice(pred_np, gt_np)
                
                except Exception:
                    pass
                
                try:
                    thin_p, thick_p = thin_thick(pred_np)
                    thin_t, thick_t = thin_thick(gt_np)
                    macro_sums["Dice_thin"]  += dice(thin_p, thin_t)
                    macro_sums["Dice_thick"] += dice(thick_p, thick_t)
                
                except Exception:
                    pass

                n_macro += 1

                # Micro pooled counts
                cm.add_counts(tp, fp, tn, fn)

                # AUC per-image & pooled
                if compute_auc:
                    prob_i = probs[i].detach().cpu().numpy()
                    
                    if prob_i.shape != gt_np.shape:
                        prob_t = torch.from_numpy(prob_i)[None, None].float()
                        prob_t = F.interpolate(prob_t, size=gt_np.shape, mode="bilinear", align_corners=False)
                        prob_i = prob_t[0,0].cpu().numpy()
                    try:
                        auc_roc = roc_auc(prob_i, gt_np)
                        
                        if not np.isnan(auc_roc): roc_sum += auc_roc; roc_n += 1
                        auc_pr  = pr_auc(prob_i, gt_np)
                        
                        if not np.isnan(auc_pr):  pr_sum  += auc_pr;  pr_n  += 1
                    
                    except Exception:
                        pass
                    
                    probs_all.append(prob_i.ravel()); gts_all.append(gt_np.ravel())

        # Finalize macro
        macro = {k: (v / max(n_macro, 1)) for k, v in macro_sums.items()}
        if compute_auc and roc_n:
            macro["ROC_AUC"] = roc_sum / roc_n
        
        if compute_auc and pr_n:
            macro["PR_AUC"]  = pr_sum  / pr_n

        # Finalize micro
        micro = cm.micro()
        if compute_auc and len(probs_all):
            probs_all = np.concatenate(probs_all, axis=0)
            gts_all   = np.concatenate(gts_all,   axis=0)
            
            if gts_all.max() != gts_all.min():
                
                try:
                    
                    from sklearn.metrics import roc_auc_score, average_precision_score
                    micro["ROC_AUC"] = float(roc_auc_score(gts_all, probs_all))
                    micro["PR_AUC"]  = float(average_precision_score(gts_all, probs_all))
                
                except Exception:
                    pass

        return macro, micro

    # Run evaluation per model
    for entry in models:
        macro, micro = _eval_one(entry)
        chosen = macro if average.lower() == "macro" else micro
        results[str(entry.name)] = chosen

    # Build DataFrame with unified metric index (ordered)
    all_keys = list(dict.fromkeys(order))  # preserve preferred order
    for col in results.values():
        for k in col.keys():
            if k not in all_keys:
                all_keys.append(k)
    df = pd.DataFrame({name: {k: results[name].get(k, np.nan) for k in all_keys}
                       for name in results.keys()})
    df = df.loc[all_keys]  # enforce row order

    # Rounded numeric copy
    df_numeric = df.astype(float)

    # Pretty copy with markers
    df_pretty = df_numeric.copy().astype(object)
    for r in df_pretty.index:
        for c in df_pretty.columns:
            df_pretty.loc[r, c] = _fmt(df_numeric.loc[r, c])

    # 1) STAR pass — mark best value(s) per metric row
#    (higher-is-better by default; lower-is-better for metrics in `minimize`)
    star_mask = pd.DataFrame(False, index=df_pretty.index, columns=df_pretty.columns)

    for metric in df_pretty.index:
        series = df_numeric.loc[metric]
        
        if series.isna().all():
            continue
        
        best_val = series.min(skipna=True) if metric in minimize else series.max(skipna=True)
        mask = (series == best_val)
        star_mask.loc[metric, mask.index[mask]] = True
        
        for col in series.index[mask]:
            
            # put ONLY the star (no arrows/dash will be added later for these cells)
            df_pretty.loc[metric, col] = f"{df_pretty.loc[metric, col]} ★"

    # 2) ARROW/DASH pass — compare to BASE, but skip any starred cells
    up, down, same = arrow_symbols

    # Choose base column (fallback to first if `base_name` not found)
    base_col_name = base_name if base_name in df_numeric.columns else df_numeric.columns[0]
    base_series = df_numeric[base_col_name]

    # Tolerances: – if nearly equal
    derived_abs_tol = 0.5 * (10 ** (-decimals))
    abs_tol = derived_abs_tol if abs_tol is None else abs_tol

    def _nearly_equal(a: float, b: float) -> bool:
        return abs(a - b) <= max(abs_tol, rel_tol * max(abs(b), 1.0))

    for metric in df_pretty.index:
        b = base_series.loc[metric]
        
        if pd.isna(b):
            continue
        
        for col in df_pretty.columns:
            if star_mask.loc[metric, col]:
                continue  # leave the star only
            v = df_numeric.loc[metric, col]
            
            if pd.isna(v):
                continue
            
            if col == base_col_name or _nearly_equal(v, b):
                sym = same
            
            else:
                sym = up if v > b else down
            
            df_pretty.loc[metric, col] = f"{df_pretty.loc[metric, col]} {sym}"
    
    return df_numeric, df_pretty
