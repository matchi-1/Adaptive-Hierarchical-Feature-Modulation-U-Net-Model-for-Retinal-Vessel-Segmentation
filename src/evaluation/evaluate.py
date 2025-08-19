# evaluation/evaluate.py
import numpy as np
import torch
from training import metrics as M

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
    cm = M.ConfusionMeter()

    # AUC micro (pool probs & gts)
    probs_all, gts_all = [], []

    model.eval()
    with torch.no_grad():
        for i in range(len(test_dataset)):
            img, gt, _ = test_dataset[i]
            x = img.unsqueeze(0).to(device)          # [1,C,H,W]
            logits = model(x)
            prob = torch.sigmoid(logits).cpu().squeeze().numpy()  # [H,W] float
            pred = (prob >= threshold).astype(np.uint8)
            gt_np = _as_numpy_mask(gt)

            # ---- macro per-image ----
            macro_sums["Dice"]        += M.dice(pred, gt_np)
            macro_sums["IoU"]         += M.iou(pred, gt_np)
            tp, fp, tn, fn             = M.confusion_counts(pred, gt_np)
            macro_sums["Sensitivity"]  += M.recall_from_counts(tp, fn)
            macro_sums["Specificity"]  += M.specificity_from_counts(tn, fp)
            macro_sums["Precision"]    += M.precision_from_counts(tp, fp)
            macro_sums["FPR"]          += M.fpr_from_counts(tn, fp)
            macro_sums["FDR"]          += M.fdr_from_counts(tp, fp)
            macro_sums["Accuracy"]     += M.acc_from_counts(tp, fp, tn, fn)
            try:
                macro_sums["clDice"]   += M.cldice(pred, gt_np)
            except Exception:
                pass  # skimage not installed

            thin_p, thick_p = M.thin_thick(pred)
            thin_t, thick_t = M.thin_thick(gt_np)
            macro_sums["Dice_thin"]  += M.dice(thin_p,  thin_t)
            macro_sums["Dice_thick"] += M.dice(thick_p, thick_t)
            n_macro += 1

            # ---- micro pooled ----
            cm.add_counts(tp, fp, tn, fn)

            # ---- AUC ----
            if compute_auc:
                probs_all.append(prob.ravel())
                gts_all.append((gt_np > 0).astype(np.uint8).ravel())
                try:
                    auc_roc = M.roc_auc(prob, gt_np)
                    if not np.isnan(auc_roc): roc_sum += auc_roc; roc_n += 1
                    auc_pr  = M.pr_auc(prob, gt_np)
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


#evaluate_and_print(model, test_dataset)