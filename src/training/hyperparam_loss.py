from dataclasses import dataclass
from typing import Tuple
from torch import optim
from src.training.loss_functions import DiceBCEComplementLoss
import torch.nn as nn

@dataclass
class LossConfig:
    w0: float = 0.9
    w1: float = 0.1
    dice_weight: float = 0.5
    bce_weight: float = 0.5
    exact_equation: bool = False
    reduction: str = "mean"
    eps: float = 1e-7

def make_loss(cfg: LossConfig):
    return DiceBCEComplementLoss(
        w0=cfg.w0, w1=cfg.w1,
        dice_weight=cfg.dice_weight, bce_weight=cfg.bce_weight,
        exact_equation=cfg.exact_equation,
        reduction=cfg.reduction,
        eps=cfg.eps,
    )

@dataclass
class OptimConfig:
    name: str = "adamw"      # "adamw" | "adam" | "sgd" | "rmsprop"
    lr: float = 1e-4
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    momentum: float = 0.9     # for SGD/RMSprop only
    nesterov: bool = False    # for SGD only

def make_optimizer(model: nn.Module, cfg: OptimConfig):
    if cfg.name.lower() == "adamw":
        return optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas, eps=cfg.eps)
    if cfg.name.lower() == "adam":
        return optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas, eps=cfg.eps)
    if cfg.name.lower() == "sgd":
        return optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, nesterov=cfg.nesterov, weight_decay=cfg.weight_decay)
    if cfg.name.lower() == "rmsprop":
        return optim.RMSprop(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum)
    raise ValueError(f"Unknown optimizer: {cfg.name}")


"""
MINIMAL USAGE:

# Loss & Optim hyperparams
loss_cfg = LossConfig(w0=0.9, w1=0.1, dice_weight=0.5, bce_weight=0.5, reduction="mean")
criterion = make_loss(loss_cfg).to(device)

opt_cfg = OptimConfig(name="adamw", lr=1e-4, weight_decay=1e-4)
optimizer = make_optimizer(model, opt_cfg)

"""