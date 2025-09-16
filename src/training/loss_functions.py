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
    def __init__(
        self,
        w0: float = 1.0,
        w1: float = 1.0,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        exact_equation: bool = False,
        reduction: str = "mean",
        eps: float = 1e-7,
    ):
        super().__init__()
        assert reduction in ("mean", "sum", "none")
        self.w0 = float(w0)
        self.w1 = float(w1)
        self.dice_w = float(dice_weight)
        self.bce_w  = float(bce_weight)
        self.exact_equation = bool(exact_equation)
        self.reduction = reduction
        self.eps = float(eps)

    @staticmethod
    def _ensure_channel_dim(x):
        # Accept [B, H, W] or [B, 1, H, W]; return [B, 1, H, W]
        if x.ndim == 3:
            return x.unsqueeze(1)
        return x

    def _dice_loss_per_sample(self, y_true, y_prob):
        # y_true, y_prob: [B,1,H,W]
        # Dice_k loss = 1 - (2 * sum (y * yhat)) / (sum y + sum yhat)
        dims = (2, 3) # [H, W]
        inter = (y_true * y_prob).sum(dim=dims)
        denom = (y_true.sum(dim=dims) + y_prob.sum(dim=dims)).clamp_min(self.eps)
        dice = 2.0 * inter / denom
        return 1.0 - dice  # [B]

    def _bce_loglik_per_sample_with_logits(self, y_true, z):
        """
        Returns per-sample average of g log σ(z) + (1-g) log(1-σ(z)).
        Uses a numerically stable identity:
            log σ(z)     = -softplus(-z)
            log (1-σ(z)) = -softplus(z)
        Output shape: [B]
        """
        dims = (2, 3)
        #softplus() is smooth, always-positive function; behaves like a soft ReLU:
        #for large 𝑥, softplus(x)≈x; for very negative 𝑥, ≈0.
        term_pos = -F.softplus(-z)      # log σ(z)
        term_neg = -F.softplus(z)       # log (1-σ(z))
        loglik = (y_true * term_pos + (1.0 - y_true) * term_neg).mean(dim=dims)
        # g log p + (1 - g)log(1 - p)
        return loglik  # [B], typically <= 0

    def forward(self, logits, target):
        logits = self._ensure_channel_dim(logits)
        target = self._ensure_channel_dim(target).clamp(0.0, 1.0)

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
        dice_k1 = self._dice_loss_per_sample(g1, p1)     # [B]  #Vessels
        dice_k0 = self._dice_loss_per_sample(g0, p0)     # [B]  #Background

        # BCE part as average log-likelihood per sample (no minus yet)
        bce_loglik_k1 = self._bce_loglik_per_sample_with_logits(g1, z1)  # [B]  #Vessels
        bce_loglik_k0 = self._bce_loglik_per_sample_with_logits(g0, z0)  # [B]  #Background

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
