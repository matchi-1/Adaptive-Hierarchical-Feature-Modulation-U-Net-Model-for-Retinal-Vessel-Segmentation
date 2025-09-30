import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceBCEComplementLoss(nn.Module):
    """
    Same as your original loss, but now supports an optional FOV mask:
        forward(logits, target, mask=None)

    If mask is provided (shape [B,1,H,W] or [B,H,W]), the Dice and BCE terms
    are computed *only over masked pixels*.
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

    def _masked_mean_hw(self, x, m):
        """
        Compute mean over H,W with mask m; returns [B].
        x, m: [B,1,H,W]
        """
        # ensure float mask
        m = (m > 0.5).float()
        denom = m.sum(dim=(2,3)).clamp_min(1.0)  # [B]
        num   = (x * m).sum(dim=(2,3))           # [B]
        return num / denom

    def _dice_loss_per_sample(self, y_true, y_prob, mask=None):
        """
        y_true, y_prob: [B,1,H,W]; optional mask [B,1,H,W]
        returns [B]
        """
        if mask is None:
            dims = (2, 3)
            inter = (y_true * y_prob).sum(dim=dims)
            denom = (y_true.sum(dim=dims) + y_prob.sum(dim=dims)).clamp_min(self.eps)
            dice = 2.0 * inter / denom
            return 1.0 - dice
        else:
            m = (mask > 0.5).float()
            y_true = y_true * m
            y_prob = y_prob * m
            inter = (y_true * y_prob).sum(dim=(2,3))
            denom = (y_true.sum(dim=(2,3)) + y_prob.sum(dim=(2,3))).clamp_min(self.eps)
            dice  = 2.0 * inter / denom
            return 1.0 - dice  # [B]

    def _bce_loglik_per_sample_with_logits(self, y_true, z, mask=None):
        """
        Returns per-sample average of g log σ(z) + (1-g) log(1-σ(z)).
        If mask is provided, average only over masked pixels.
        Output shape: [B]
        """
        term_pos = -F.softplus(-z)   # log σ(z)
        term_neg = -F.softplus(z)    # log (1-σ(z))
        ll = y_true * term_pos + (1.0 - y_true) * term_neg  # [B,1,H,W]
        if mask is None:
            return ll.mean(dim=(2,3))
        else:
            return self._masked_mean_hw(ll, mask)

    def forward(self, logits, target, mask=None):
        logits = self._ensure_channel_dim(logits)
        target = self._ensure_channel_dim(target).clamp(0.0, 1.0)
        if mask is not None:
            mask = self._ensure_channel_dim(mask).float()

        # Foreground / background complements
        z1 = logits
        z0 = -logits
        p1 = torch.sigmoid(z1)
        p0 = 1.0 - p1
        g1 = target
        g0 = 1.0 - g1

        # Dice (masked if provided)
        dice_k1 = self._dice_loss_per_sample(g1, p1, mask=mask)  # [B]
        dice_k0 = self._dice_loss_per_sample(g0, p0, mask=mask)  # [B]

        # BCE (masked if provided): return average log-likelihood over H,W
        bce_loglik_k1 = self._bce_loglik_per_sample_with_logits(g1, z1, mask=mask)
        bce_loglik_k0 = self._bce_loglik_per_sample_with_logits(g0, z0, mask=mask)

        # Turn into a minimization objective if needed
        if self.exact_equation:
            bce_k1 = bce_loglik_k1
            bce_k0 = bce_loglik_k0
        else:
            bce_k1 = -bce_loglik_k1
            bce_k0 = -bce_loglik_k0

        # Per-class bracket
        term_k1 = self.dice_w * dice_k1 + self.bce_w * bce_k1  # [B]
        term_k0 = self.dice_w * dice_k0 + self.bce_w * bce_k0  # [B]

        # Class-weighted sum across k
        loss_per_sample = self.w0 * term_k0 + self.w1 * term_k1  # [B]

        if self.reduction == "mean":
            return loss_per_sample.mean()
        elif self.reduction == "sum":
            return loss_per_sample.sum()
        else:
            return loss_per_sample  # 'none'
