import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d

class DPCN(nn.Module):
    def __init__(self, in_ch, channels=None, iters=3):
        super().__init__()
        self.in_ch = in_ch
        self.channels = channels or in_ch   # default: same channels as input
        self.iters = iters

        # ---- 1.) Coupled Linking Subsystem Setup ----

        # (insert eq here later)

        k = 3  # 3x3 deformable kernel 
        # in deformable conv, each location in the kernel (9 taps) needs 2 values: 
        #           Δ𝑚 (offset for y-direction), Δ𝑛 (offset for x-direction),
        #           since for each pixel/sample point we compute it as:
        #           (𝑚 + i + Δ𝑚, 𝑛 + j + Δ𝑛) where (i,j) are fixed grid offsets (e.g. (i = -1, j =-1 --> top left neighbor/tap)).
        
        # total offset channels = 2×3×3=18
        off_ch = 2 * k * k  # 18 channels for (dy,dx) per tap

        # Offset predictor: predicts offsets from the previous state Y(n-1)

        # Conv2d input:  feature map [N, C_in, H, W] ; output: offsets [N, 18, H, W]
        #   self.channels = C_in (number of channels in Y(n-1) )
        #   off_ch = 18 (number of output channels = 2*k*k of this convolutional layer)
        #   kernel filter with shape (out_ch, in_ch, k, k) = (18, C, 3, 3), padding is 1 to keep same H,W
        # Conceptually, at each pixel (h,w), this layer outputs 18 values: (dy,dx) offsets for each of the 9 taps in the 3x3 kernel
        self.offset_conv = nn.Conv2d(self.channels, off_ch, kernel_size=3, padding=1) 
        nn.init.zeros_(self.offset_conv.weight)  # init near zero so first == plain conv
        nn.init.zeros_(self.offset_conv.bias)    

        # Kernel weights for the deformable conv (W(i,j) in the paper)
        weight = torch.empty(self.channels, self.channels, k, k) # initialize weight tensor with shape (out_ch, in_ch, k, k) 
        nn.init.kaiming_normal_(weight, nonlinearity="relu") # give weights good starting values so they won't collapse or explode during training 
        self.weight = nn.Parameter(weight)  # make weight a learnable parameter thru backpropagation
        self.bias   = nn.Parameter(torch.zeros(self.channels)) # add bias term per output channel ; after summing all taps, add bias

        # normalization (helps stability)
        self.norm = nn.BatchNorm2d(self.channels) # convs can produce large values, batchnorm re-centers and rescales each channel to have mean=0, std=1 (per batch)

    def coupled_linking(self, y_prev):
        """
        Coupled Linking Subsystem:
        L(n) = DefConv(Y(n-1))

        Args:
            y_prev: previous iteration output Y(n-1), shape [N,C,H,W]
        Returns:
            L: locally enhanced feature map, shape [N,C,H,W]
        """
        # 1. predict offsets from Y(n-1) using normal Conv2d
        offsets = self.offset_conv(y_prev)  # [N,18,H,W]

        # 2. apply deformable conv
        L = deform_conv2d(
            input=y_prev,  # Y(n-1) previous output
            offset=offsets, # supplies all Δ𝑚, Δ𝑛 for the 9 taps of the kernel
            weight=self.weight, # all learnable weights W(i,j) 
            bias=self.bias, # learnable bias per output channel
            stride=1,   # always 1 keep same spatial size 
            padding=1,
            dilation=1,
            mask=None  # no per-tap amplitude mask
        )

        # 3. normalization
        L = self.norm(L) # stabilize activations
        return L

    def forward(self, x, fov=None):
        """
        x:   input shallow feature [N,C,H,W]
        fov: optional binary mask [N,1,H,W]
        """
        # for now: just demo coupled linking once
        y_prev = x
        L = self.coupled_linking(y_prev)

        return L
