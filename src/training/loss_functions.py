import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    """
    Hybrid = 0.5 * DiceLoss + 0.5 * BCEWithLogitsLoss (with optional pos_weight).
    Use for binary segmentation with a single-channel logit output.
    """
    def __init__(self, dice_weight=0.5, bce_weight=0.5, eps=1e-7, pos_weight=None):
        super().__init__()
        self.dice_w = dice_weight
        self.bce_w  = bce_weight
        self.eps = eps
        self.register_buffer('pos_weight', torch.tensor(0.0) if pos_weight is None else torch.as_tensor(pos_weight, dtype=torch.float))

    def forward(self, logits, targets):
        """
        logits: [B,1,H,W] raw scores (no sigmoid)
        targets: [B,1,H,W] binary {0,1}
        """
        probs = torch.sigmoid(logits)
        # Dice (foreground)
        p = probs.view(probs.size(0), -1)
        t = targets.float().view(targets.size(0), -1)
        inter = (p * t).sum(dim=1)
        denom = p.sum(dim=1) + t.sum(dim=1)
        dice_loss = 1.0 - (2.0 * inter + self.eps) / (denom + self.eps)
        dice_loss = dice_loss.mean()

        # Weighted BCE on logits (numerically stable)
        if self.pos_weight.numel() == 1 and self.pos_weight.item() == 0.0:
            bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='mean')
        else:
            bce = F.binary_cross_entropy_with_logits(
                logits, targets.float(), reduction='mean', pos_weight=self.pos_weight
            )
        return self.dice_w * dice_loss + self.bce_w * bce


class SoftDiceCEHybrid(nn.Module):
    def __init__(self, class_weights=None, dice_weight=0.5, ce_weight=0.5, eps=1e-7):
        super().__init__()
        self.dice_w = dice_weight
        self.ce_w = ce_weight
        self.eps = eps
        self.ce = nn.CrossEntropyLoss(weight=None if class_weights is None
                                      else torch.as_tensor(class_weights, dtype=torch.float))

    def forward(self, logits, targets):
        """
        logits: [B,2,H,W] (raw)
        targets: [B,H,W] with values {0,1}
        """
        ce_loss = self.ce(logits, targets.long())

        probs = torch.softmax(logits, dim=1)                  # [B,2,H,W]
        onehot = F.one_hot(targets.long(), num_classes=2)     # [B,H,W,2]
        onehot = onehot.permute(0,3,1,2).float()              # [B,2,H,W]

        # class-wise soft dice
        p = probs.reshape(probs.size(0), 2, -1)
        t = onehot.reshape(onehot.size(0), 2, -1)
        inter = (p * t).sum(dim=2)
        denom = p.sum(dim=2) + t.sum(dim=2)
        dice_per_class = 1.0 - (2.0 * inter + self.eps) / (denom + self.eps)  # [B,2]
        dice_loss = dice_per_class.mean()  # average over batch and classes

        return self.dice_w * dice_loss + self.ce_w * ce_loss
