import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCEComplementLoss(nn.Module):
    """
    Implements loss:
      ℓ_W = sum_{k=0}^1 W_k [ 1/2 * (1 - Dice_k)  +  1/2 * (1/N) * Σ ( g^k log p^k + (1-g^k) log(1-p^k) ) ]
    with a single-channel model by using complements for k=0 (background) and k=1 (foreground).

    Args:
        w0, w1: class weights W_0 (background), W_1 (foreground)
        dice_weight: weight on Dice term inside the bracket (default 0.5)
        bce_weight:  weight on BCE   term inside the bracket (default 0.5)
        exact_equation: If True, use the BCE sign exactly as in formula (adds log-likelihood).
                        If False (default), negate the BCE part so the overall loss is minimized in practice.
        reduction: 'mean' (default), 'sum', or 'none' over the batch
        eps: small number for numerical stability
    Expected shapes:
        logits: [B, 1, H, W]  (raw scores)
        target: [B, 1, H, W]  (0/1 or soft labels in [0,1])
    """
    def __init__(self, w0=1.0, w1=1.0, dice_weight=0.5, bce_weight=0.5,
                 exact_equation=False, reduction="mean", eps=1e-7,
                 label_smoothing=0.0):
        super().__init__()
        self.w0, self.w1 = float(w0), float(w1)
        self.dice_w, self.bce_w = float(dice_weight), float(bce_weight)
        self.exact_equation = bool(exact_equation)
        self.reduction, self.eps = reduction, float(eps)
        self.label_smoothing = float(label_smoothing)

    @staticmethod
    def _ensure_channel_dim(x):
        # Accept [B, H, W] or [B, 1, H, W]; return [B, 1, H, W]
        if x.ndim == 3:
            return x.unsqueeze(1)
        return x

    def _dice_loss_per_sample_masked(self, y_true, y_prob, w):
        # y_true, y_prob, w: [B,1,H,W]; w is FOV mask in {0,1}
        dims = (2, 3)
        wp = w
        inter = (wp * y_true * y_prob).sum(dim=dims)
        denom = (wp * y_true).sum(dim=dims) + (wp * y_prob).sum(dim=dims)
        dice = 2.0 * inter / denom.clamp_min(self.eps)
        return 1.0 * (1.0 - dice)  # [B]

    def _bce_loglik_per_sample_with_logits_masked(self, y_true, z, w):
        """
        Returns per-sample average of g log σ(z) + (1-g) log(1-σ(z)).
        Uses a numerically stable identity:
            log σ(z)     = -softplus(-z)
            log (1-σ(z)) = -softplus(z)
        Output shape: [B]
        """
        dims = (2, 3)
        term_pos = -F.softplus(-z)   # log σ(z)
        term_neg = -F.softplus(z)    # log (1-σ(z))
        per_pix = y_true * term_pos + (1.0 - y_true) * term_neg  # [B,1,H,W]
        denom = w.sum(dim=dims).clamp_min(1.0)
        return (per_pix * w).sum(dim=dims) / denom  # [B]

    def forward(self, logits, target, fov=None):
        logits = self._ensure_channel_dim(logits)
        target = self._ensure_channel_dim(target).clamp(0.0, 1.0)

        if fov is None:
            # default: everything counts
            fov = torch.ones_like(target)
        else:
            fov = self._ensure_channel_dim((fov > 0.5).float())

        # (optional) very light label smoothing to reduce overconfidence
        if self.label_smoothing > 0:
            s = self.label_smoothing
            target = target * (1 - s) + 0.5 * s

        # Foreground (k=1): use logits as-is; Background (k=0): logits negated (σ(-z) = 1-σ(z))
        z1 = logits
        z0 = -logits
        # these are unbounded (−∞, +∞); BCE on probabilities can underflow/overflow

        p1 = torch.sigmoid(z1)          # foreground prob
        p0 = 1.0 - p1                   # background prob  (same as sigmoid(-z1))
        # DICE only makes sense when p is [0,1] or probability so it behaves like a soft mask

        g1 = target                     # foreground GT
        g0 = 1.0 - g1                   # background GT

        # Dice losses (per-sample)
        dice_k1 = self._dice_loss_per_sample_masked(g1, p1, fov)     # [B]  #Vessels
        dice_k0 = self._dice_loss_per_sample_masked(g0, p0, fov)     # [B]  #Background

        # BCE part as average log-likelihood per sample (no minus yet)
        bce_loglik_k1 = self._bce_loglik_per_sample_with_logits_masked(g1, z1, fov)  # [B]  #Vessels
        bce_loglik_k0 = self._bce_loglik_per_sample_with_logits_masked(g0, z0, fov)  # [B]  #Background

        # If we want a standard *loss to minimize*, negate the log-likelihood.
        if self.exact_equation:
            # Use the sign exactly as the printed formula (adds log-likelihood).
            bce_k1 = bce_loglik_k1
            bce_k0 = bce_loglik_k0
        else:
            # Typical training: minimize loss → use negative log-likelihood (i.e., BCE).
            bce_k1 = -bce_loglik_k1
            bce_k0 = -bce_loglik_k0

        # Per-class bracket: 1/2 * Dice + 1/2 * BCE  (weights configurable)
        term_k1 = self.dice_w * dice_k1 + self.bce_w * bce_k1    # [B]
        term_k0 = self.dice_w * dice_k0 + self.bce_w * bce_k0    # [B]

        # Class-weighted sum across k
        loss_per_sample = self.w0 * term_k0 + self.w1 * term_k1  # [B]

        if self.reduction == "mean":
            return loss_per_sample.mean()
        elif self.reduction == "sum":
            return loss_per_sample.sum()
        else:
            return loss_per_sample  # 'none'


class TverskyLoss(nn.Module):
    """
    Tversky Loss — balances FP and FN penalties.
    alpha → weight for false positives
    beta  → weight for false negatives
    """
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        y_true = y_true.float()

        TP = (y_true * y_pred).sum(dim=(1, 2, 3))
        FP = ((1 - y_true) * y_pred).sum(dim=(1, 2, 3))
        FN = (y_true * (1 - y_pred)).sum(dim=(1, 2, 3))

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        loss = 1 - tversky

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class FocalTverskyLoss(TverskyLoss):
    """
    Focal-Tversky Loss — focuses on hard examples by adding a gamma exponent.
    """
    def __init__(self, alpha=0.5, beta=0.5, gamma=1.0, smooth=1e-6, reduction="mean"):
        super().__init__(alpha=alpha, beta=beta, smooth=smooth, reduction=reduction)
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        base_loss = super().forward(y_pred, y_true)
        return base_loss.pow(self.gamma)


class HybridTverskyDiceBCELoss(nn.Module):
    def __init__(self, *, 
                 tversky_alpha=0.75, tversky_beta=0.25, tversky_weight=0.5,
                 dice_weight=0.5, bce_weight=0.5, dicebce_weight=0.5,
                 w0=1.0, w1=1.0, reduction="mean"):
        super().__init__()
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta, reduction=reduction)
        self.dicebce = DiceBCEComplementLoss(
            w0=w0, w1=w1,
            dice_weight=dice_weight,
            bce_weight=bce_weight,
            exact_equation=False,
            reduction=reduction,
        )
        self.tversky_weight  = float(tversky_weight)
        self.dicebce_weight  = float(dicebce_weight)

    def forward(self, logits, target):
        # NOTE: Both of component losses expect LOGITS (not probs).
        lt = self.tversky(logits, target)
        ld = self.dicebce(logits, target)
        return self.tversky_weight * lt + self.dicebce_weight * ld
    

class EdgeSobelLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3)
        ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer("kx", kx); self.register_buffer("ky", ky)
        self.reduction = reduction
    def forward(self, y_pred, y_true):
        p = torch.sigmoid(y_pred)
        gx = F.conv2d(p, self.kx, padding=1)
        gy = F.conv2d(p, self.ky, padding=1)
        # encourage sparse/clean edges (L1)
        loss = (gx.abs() + gy.abs()).mean(dim=(1,2,3))
        return loss.mean() if self.reduction=="mean" else loss.sum()
