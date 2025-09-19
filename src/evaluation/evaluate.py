# evaluation/evaluate.py
import numpy as np
import torch
from src.training.metrics import *

def _as_numpy_mask(x):
    """Squeeze and to numpy; keeps 2D [H,W]."""
    if hasattr(x, "cpu"):
        x = x.squeeze().cpu().numpy()
    else:
        x = np.asarray(x).squeeze()
    return x

def evaluate_and_print(model, test_dataset, device="cuda", threshold=0.5, compute_auc=True):
    """
    model: torch nn.Module (binary head: [B,1,H,W] logits)
    test_dataset: yields (image, mask)
    """

    # --- macro accumulators (per-image means) ---
    macro_sums = {
        "Dice":0.0, "IoU":0.0, "Sensitivity":0.0, "Specificity":0.0,
        "Precision":0.0, "FPR":0.0, "FDR":0.0, "Accuracy":0.0,
        "clDice":0.0, "Dice_thin":0.0, "Dice_thick":0.0,
    }
    n_macro = 0

    # AUC macro
    roc_sum = 0.0; pr_sum = 0.0; roc_n = 0; pr_n = 0

    # --- micro accumulators (pooled) ---
    cm = ConfusionMeter()

    # AUC micro (pool probs & gts)
    probs_all, gts_all = [], []

    model.eval()
    with torch.no_grad():
        for i in range(len(test_dataset)):
            img, gt, _ = test_dataset[i]
            x = img.unsqueeze(0).to(device)          # [1,C,H,W]
            logits = model(x)
            if logits.ndim == 4 and logits.shape[1] == 2:
                # two-channel head: class 0 = background, class 1 = vessel
                prob = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()  # [H,W], vessel prob
            elif logits.ndim == 4 and logits.shape[1] == 1:
                # one-channel head: single vessel logit
                prob = torch.sigmoid(logits)[0, 0].cpu().numpy()         # [H,W]
            else:
                # fallback if model returns [2,H,W] or [H,W]
                arr = logits.squeeze(0)  # remove batch if present
                if arr.ndim == 3 and arr.shape[0] == 2:
                    prob = torch.softmax(arr, dim=0)[1].cpu().numpy()
                else:
                    prob = torch.sigmoid(arr).cpu().numpy()

            pred = (prob >= threshold).astype(np.uint8)  # [H,W] binary 0/1
            gt_np = _as_numpy_mask(gt)                    # ensure [H,W] 0/1
            prob = torch.sigmoid(logits).cpu().squeeze().numpy()  # [H,W] float

            # ---- macro per-image ----
            macro_sums["Dice"]        += dice(pred, gt_np)
            macro_sums["IoU"]         += iou(pred, gt_np)
            tp, fp, tn, fn             = confusion_counts(pred, gt_np)
            macro_sums["Sensitivity"]  += recall_from_counts(tp, fn)
            macro_sums["Specificity"]  += specificity_from_counts(tn, fp)
            macro_sums["Precision"]    += precision_from_counts(tp, fp)
            macro_sums["FPR"]          += fpr_from_counts(tn, fp)
            macro_sums["FDR"]          += fdr_from_counts(tp, fp)
            macro_sums["Accuracy"]     += acc_from_counts(tp, fp, tn, fn)
            try:
                macro_sums["clDice"]   += cldice(pred, gt_np)
            except Exception:
                pass  # skimage not installed

            thin_p, thick_p = thin_thick(pred)
            thin_t, thick_t = thin_thick(gt_np)
            macro_sums["Dice_thin"]  += dice(thin_p,  thin_t)
            macro_sums["Dice_thick"] += dice(thick_p, thick_t)
            n_macro += 1

            # ---- micro pooled ----
            cm.add_counts(tp, fp, tn, fn)

            # ---- AUC ----
            if compute_auc:
                probs_all.append(prob.ravel())
                gts_all.append((gt_np > 0).astype(np.uint8).ravel())
                try:
                    auc_roc = roc_auc(prob, gt_np)
                    if not np.isnan(auc_roc): roc_sum += auc_roc; roc_n += 1
                    auc_pr  = pr_auc(prob, gt_np)
                    if not np.isnan(auc_pr):  pr_sum  += auc_pr;  pr_n  += 1
                except Exception:
                    pass  # sklearn not installed

    # --- finalize macro ---
    macro = {k: (v / max(n_macro, 1)) for k, v in macro_sums.items()}
    if compute_auc and roc_n:
        macro["ROC_AUC"] = roc_sum / roc_n
    if compute_auc and pr_n:
        macro["PR_AUC"]  = pr_sum  / pr_n

    # --- finalize micro ---
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

    # --- print nicely ---
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

    #return {"macro": macro, "micro": micro}
