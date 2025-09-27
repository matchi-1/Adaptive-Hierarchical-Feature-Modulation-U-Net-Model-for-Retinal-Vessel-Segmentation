import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.blocks.res2net50 import Res2Net, Bottle2neck

class Res2NetBackbone(nn.Module):
    """
    Wrapper that turns your Res2Net-50 into a 4-level feature pyramid:
    forward(x) -> (E1, E2, E3, E4), with .out_channels = [C1,C2,C3,C4].
    """
    def __init__(self, in_ch: int = 3, pretrained: bool = False):
        super().__init__()
        # Instantiate the base network. Adjust args to match your file.
        self.body = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth = 26, scale = 4)  # or Res2Net50(...)

        # --- Stem patching for non-RGB inputs ---
        if in_ch != 3:
            # Replace the first conv to accept in_ch
            old = self.body.conv1
            self.body.conv1 = nn.Conv2d(in_ch, old.out_channels,
                                        kernel_size=old.kernel_size,
                                        stride=old.stride,
                                        padding=old.padding,
                                        bias=False)

        # Typical Res(2)Net-50 stage channels (post-expansion): [256, 512, 1024, 2048].
        # If your file uses different widths/expansion, update these four numbers accordingly.
        self.out_channels = [256, 512, 1024, 2048]

    def forward(self, x: torch.Tensor):
        """
        Returns tuple of 4 stage features:
        E1 = output of layer1 (1/4 res), E2 = layer2 (1/8), E3 = layer3 (1/16), E4 = layer4 (1/32).
        """
        # Standard ResNet-like forward; adjust if your file differs.
        x = self.body.relu(self.body.bn1(self.body.conv1(x)))
        x = self.body.maxpool(x)

        e1 = self.body.layer1(x)  # C=256
        e2 = self.body.layer2(e1) # C=512
        e3 = self.body.layer3(e2) # C=1024
        e4 = self.body.layer4(e3) # C=2048

        return e1, e2, e3, e4
