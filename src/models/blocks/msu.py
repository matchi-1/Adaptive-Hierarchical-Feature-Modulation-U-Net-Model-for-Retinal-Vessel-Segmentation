import torch
import torch.nn as nn
import torch.nn.functional as F

class MSU(nn.Module):
    """
    Purpose:
        Compute a multi-scale *difference map* between two feature maps by applying
        shared 1×1, 3×3, and 5×5 convolutions to each input, taking the absolute
        difference at each scale, then summing those differences. The result can be
        added onto a base feature map to enhance vessel-like structures
        and boundaries.

    Parameters:
        in_channels (int):  Number of channels in each input feature map.
        out_channels (int): Number of channels produced by each scale’s convolution and by the final
                            sum. If different from `in_channels`, each scale projects to this size.
        use_bn (bool):      If True, applies BatchNorm2d to the fused output.
        activation (bool):  If True, applies ReLU to the normalized output.

    Inputs:
        F_A : torch.Tensor
            First feature map, shape (N, C_in, H, W).
        F_B : torch.Tensor
            Second feature map, shape (N, C_in, H, W).

    Outputs:
        out : torch.Tensor
            Multi-scale absolute-difference map, shape (N, C_out, H, W).

    Notes:
        Operation: For scales k ∈ {1, 3, 5},
            out = Σ_k | (Conv_k(F_A) − Conv_k(F_B)) |
        where the Conv_k weights are *shared* between the two branches (F_A and F_B).
        The absolute function is non-differentiable at 0; PyTorch uses a subgradient
        there, which is standard and works well in practice.
        Convs use padding {0,1,2} for {1×1,3×3,5×5} to preserve H×W.
       
    """

    def __init__(self, in_channels, out_channels=None, use_bn=True, activation=True):
        super().__init__()
        out_channels = out_channels or in_channels

        # three parallel convs for each branch (A and B share weights per scale per paper’s intent)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=not use_bn)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_bn)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, bias=not use_bn)

        # optional post-fusion refinement
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = nn.ReLU(inplace=True) if activation else nn.Identity()

        # simple Kaiming init; weight + bias init for convs
        for m in [self.conv1, self.conv3, self.conv5]:

            # draws each conv weight from a normal distribution whose variance 
            # is chosen to keep activations well-scaled in ReLU
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu') 

            if m.bias is not None:
                # bias = 0 is a safe default; 
                # when a following BatchNorm is present, the conv bias is effectively redundant
                nn.init.zeros_(m.bias)

    def forward(self, F_A: torch.Tensor, F_B: torch.Tensor) -> torch.Tensor:
        """
        Purpose:
            Produce the fused multi-scale absolute-difference map between `F_A` and `F_B`.

        Inputs:
            F_A : torch.Tensor
                Feature map of shape (N, C_in, H, W).
            F_B : torch.Tensor
                Feature map of shape (N, C_in, H, W).

        Outputs
            out : torch.Tensor
                Tensor of shape (N, C_out, H, W):
                    out = |C1(F_A) − C1(F_B)| + |C3(F_A) − C3(F_B)| + |C5(F_A) − C5(F_B)|
                followed by optional BatchNorm and ReLU.

        Notes
            - `F_A` and `F_B` must have identical shape, dtype, and device.
            - Convolution weights are shared across the two inputs at each scale.
            - Use `base + out` externally if you want to overlay the difference map onto
            an existing feature tensor.
        """

        # multi-scale convs
        a1, b1 = self.conv1(F_A), self.conv1(F_B)
        a3, b3 = self.conv3(F_A), self.conv3(F_B)
        a5, b5 = self.conv5(F_A), self.conv5(F_B)

        # Eq. (8): sum of absolute differences across scales
        out = (torch.abs(a1 - b1) +
               torch.abs(a3 - b3) +
               torch.abs(a5 - b5))

        out = self.bn(out)
        out = self.act(out)
        return out
