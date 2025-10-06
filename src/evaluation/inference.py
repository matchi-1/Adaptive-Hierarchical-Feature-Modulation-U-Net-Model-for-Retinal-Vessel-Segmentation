# ---- Metrics (Dice, IoU, AUC, + SEN, SPE, clDice) ----
def _flatten_masked(p, t, m=None):
    if m is None:
        return p.view(-1), t.view(-1)
    return (p*m).view(-1), (t*m).view(-1)

def dice_coefficient(p01, t01, m=None, eps=1e-6):
    p, t = _flatten_masked(p01, t01, m)
    tp = (p*t).sum()
    fp = (p*(1-t)).sum()
    fn = ((1-p)*t).sum()
    return (2*tp + eps) / (2*tp + fp + fn + eps)

def iou_score(p01, t01, m=None, eps=1e-6):
    p, t = _flatten_masked(p01, t01, m)
    tp = (p*t).sum()
    fp = (p*(1-t)).sum()
    fn = ((1-p)*t).sum()
    return (tp + eps) / (tp + fp + fn + eps)

def sensitivity_specificity(p01, t01, m=None, eps=1e-6):
    p, t = _flatten_masked(p01, t01, m)
    tp = (p*t).sum()
    tn = ((1-p)*(1-t)).sum()
    fp = (p*(1-t)).sum()
    fn = ((1-p)*t).sum()
    sen = (tp + eps) / (tp + fn + eps)  # recall
    spe = (tn + eps) / (tn + fp + eps)
    return sen, spe

# clDice requires skeletonization
import sys, subprocess
try:
    from skimage.morphology import skeletonize
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-image", "-q"])
    from skimage.morphology import skeletonize

def cldice_score(p01, t01, m=None, eps=1e-6):
    # convert to CPU numpy, apply mask, then skeletonize
    p = p01.detach().cpu().numpy().astype(np.uint8)[0,0]  # [H,W]
    g = t01.detach().cpu().numpy().astype(np.uint8)[0,0]
    if m is not None:
        mm = m.detach().cpu().numpy().astype(np.uint8)[0,0]
        p = (p * mm).astype(np.uint8)
        g = (g * mm).astype(np.uint8)
    sp = skeletonize(p > 0).astype(np.uint8)
    sg = skeletonize(g > 0).astype(np.uint8)
    # topology precision/recall
    tprec = (sp & g).sum() / (sp.sum() + eps)
    trec  = (sg & p).sum() / (sg.sum() + eps)
    return (2 * tprec * trec) / (tprec + trec + eps)


def roc_auc(y, probs, m=None):
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(
            (y*m).detach().cpu().numpy().reshape(-1),
            (probs*m).detach().cpu().numpy().reshape(-1)
        )
    except Exception:
        auc = None

    return auc

def visualize(x, y, probs, pred01):
    import matplotlib.pyplot as plt
    im  = x[0,0].detach().cpu().numpy()
    gt  = y[0,0].detach().cpu().numpy()
    pr  = probs[0,0].detach().cpu().numpy()
    pb  = pred01[0,0].detach().cpu().numpy()

    fig, axs = plt.subplots(1,4, figsize=(14,3.5))
    axs[0].imshow(im, cmap='gray'); axs[0].set_title('Image'); axs[0].axis('off')
    axs[1].imshow(gt, cmap='gray'); axs[1].set_title('GT'); axs[1].axis('off')
    axs[2].imshow(pr, cmap='gray', vmin=0, vmax=1); axs[2].set_title('Prob'); axs[2].axis('off')
    axs[3].imshow(pb, cmap='gray'); axs[3].set_title('Pred (τ=0.5)'); axs[3].axis('off')
    plt.tight_layout(); plt.show()