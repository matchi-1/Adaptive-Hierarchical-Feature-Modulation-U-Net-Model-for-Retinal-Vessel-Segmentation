import numpy as np
import torch
import torch.nn.functional as F
from src.training.metrics import *

# ------------------------------
# helpers: normalize to numpy
# ------------------------------

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
def evaluate_and_print(model, test_dataloader, device="cuda", threshold=0.5, compute_auc=True):
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
        "Dice": 0.0, "IoU": 0.0, "Sensitivity": 0.0, "Specificity": 0.0,
        "Precision": 0.0, "FPR": 0.0, "FDR": 0.0, "Accuracy": 0.0,
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

        # Forward pass: get logits (may be tensor or list/tuple)
        logits = model(imgs)                # [B,1,H,W] or [B,2,H,W] (or list/tuple)

        # If model returns multiple outputs, use the last one (common practice)
        if isinstance(logits, (list, tuple)):
            logits = logits[-1]

        # Convert logits to probabilities in [0,1]
        if logits.ndim == 4 and logits.shape[1] == 2:
            probs = torch.softmax(logits, dim=1)[:, 1]    # [B,H,W]
        
        elif logits.ndim == 4 and logits.shape[1] == 1:
            probs = torch.sigmoid(logits[:, 0])           # [B,H,W]
        
        elif logits.ndim == 3:               # [B,H,W]
            probs = torch.sigmoid(logits)                 # [B,H,W]
        
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
    
    for k in ["Dice","IoU","Sensitivity","Specificity","Precision","FPR","FDR",
              "Accuracy","clDice","Dice_thin","Dice_thick","ROC_AUC","PR_AUC"]:
        
        if k in macro:
            print(f"{k:15s}: {macro[k]:.4f}")

    print("-- Micro (pooled over all pixels) --")
    
    for k in ["Precision","Sensitivity","Specificity","F1/Dice","IoU","Accuracy",
              "FPR","FDR","ROC_AUC","PR_AUC"]:
        
        if k in micro:
            print(f"{k:15s}: {micro[k]:.4f}")
