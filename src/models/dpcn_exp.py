import torch
import torch.nn as nn

class DPCN(nn.Module):
    def __init__(self, in_ch, channels, iters=3):
        super().__init__()
        # layers and constants here
        self.in_ch = in_ch
        self.channels = channels
        self.iters = iters

    def forward(self, x, fov=None):
        """
        Args:
            x:   input image or shallow features, shape [N, C, H, W]
            fov: optional field-of-view mask, shape [N, 1, H, W]
        Returns:
            y:   refined features after DPCN iterations
        """
        # computations here
        return x
