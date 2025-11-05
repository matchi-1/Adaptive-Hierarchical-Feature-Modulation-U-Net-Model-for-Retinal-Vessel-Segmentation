import numpy as np
import torch
import torch.nn.functional as F
from src.training.metrics import ConfusionMeter, confusion_counts, cldice

@torch.no_grad()
def sweep_thresholds_on_val(model, loader, device="cuda", thresholds=None):
    """Sweeps thresholds over validation set and computes SEN, SPE, clDice."""
    model.eval()
    model.to(device)

    if thresholds is None:
        thresholds = np.linspace(0.10, 0.90, 17)
    thresholds = [float(t) for t in thresholds]

    cms     = {t: ConfusionMeter() for t in thresholds}
    cl_sums = {t: 0.0 for t in thresholds}
    cl_n    = {t: 0   for t in thresholds}

    for batch in loader:
        imgs = batch["image"].to(device)
        gts  = batch["mask"]
        fov  = batch.get("fov", None)

        with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
            logits = model(imgs)

        if isinstance(logits, (list, tuple)):
            logits = logits[-1]

        if logits.ndim == 4 and logits.shape[1] == 2:
            probs = torch.softmax(logits, dim=1)[:, 1]
        elif logits.ndim == 4 and logits.shape[1] == 1:
            probs = torch.sigmoid(logits[:, 0])
        elif logits.ndim == 3:
            probs = torch.sigmoid(logits)
        else:
            raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")

        B = probs.shape[0]
        for i in range(B):
            prob_i = probs[i].detach().cpu().numpy()
            gt_np  = gts[i].squeeze().numpy()
            if gt_np.ndim == 3:
                gt_np = gt_np[0]
            gt_np = (gt_np >= 0.5).astype(np.uint8)

            if fov is not None:
                fov_np = fov[i].squeeze().cpu().numpy()
                gt_np  = gt_np * (fov_np > 0.5).astype(np.uint8)

            if prob_i.shape != gt_np.shape:
                prob_t = torch.from_numpy(prob_i)[None, None].float()
                prob_t = F.interpolate(prob_t, size=gt_np.shape, mode="bilinear", align_corners=False)
                prob_i = prob_t[0, 0].cpu().numpy()

            for t in thresholds:
                pred_np = (prob_i >= t).astype(np.uint8)
                if fov is not None:
                    pred_np = pred_np * (fov_np > 0.5).astype(np.uint8)

                tp, fp, tn, fn = confusion_counts(pred_np, gt_np)
                cms[t].add_counts(tp, fp, tn, fn)
                try:
                    cl_sums[t] += cldice(pred_np, gt_np)
                    cl_n[t] += 1
                except Exception:
                    pass

    results = []
    for t in thresholds:
        micro = cms[t].micro()
        SEN = micro.get("Sensitivity", float("nan"))
        SPE = micro.get("Specificity", float("nan"))
        cld = cl_sums[t] / max(1, cl_n[t])
        results.append({
            "thr": float(t),
            "SEN": float(SEN),
            "SPE": float(SPE),
            "clDice": float(cld),
        })

    return results
