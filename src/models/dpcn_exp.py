import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d

class DPCN(nn.Module):
    def __init__(self, in_ch, channels=None, iters=3, beta_init=0.5):
        super().__init__()
        self.in_ch = in_ch
        self.channels = channels or in_ch   # default: same channels as input (should be 32 / 64 / 128)
        self.iters = iters

        # ---- project input to internal channels once; F(n) will reuse this ----
        # if in_ch == channels, just use identity (no extra cost or no op)
        # else use 1x1 conv as a channel aligner -- mixes channels without changing H,W ; example if in_ch=1, channels=32 1×1 conv learns 32 filters over the single input channel, giving an 32-channel
        self.proj_in = nn.Identity() if self.in_ch == self.channels else nn.Conv2d(self.in_ch, self.channels, 1)

        # ---- learnable β for modulation (clamp at runtime to [0,1]) ----
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))  # learnable scalar parameter, will receive gradients and be updated by the optimizer during training

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


    # !! ---- SUBSYSTEM FUNCTIONS ---- !!

    # -------- Coupled Linking: L(n) = DefConv(Y(n-1)) ----------
    # expounded equation for deformable conv: 
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
    
    # -------- Feeding Input: F(n) = I_OR ----------
    # In practice we build F once from x (projection) and reuse it every iteration.
    def feeding_input(self, x):
        # x is the original shallow feature (or preprocessed image), shape [N, C_in, H, W]
        F = self.proj_in(x)  # [N, channels, H, W]
        return F

    # -------- Modulation: U(n) = F(n) * (1 + β * L(n)) ----------
    def modulation(self, F, L):
        beta = torch.clamp(self.beta, 0.0, 1.0)  # keep β in a sane range so modulation doesn’t blow up or flip signs
        U = F * (1.0 + beta * L)                 # formula for modulation. states of the feeding units and linking units combine in a second-order manner to produce the internal state 𝑈(𝑛) of the neuron, with the degree ofcombination controlled by the coefficient B
        return U


    def forward(self, x, fov=None):
        """
        Here we only wire the first three subsystems:
          - F(n) from x
          - L(n) from Y(n-1) (we initialize Y(0) from F)
          - U(n) = F * (1 + β*L)

        (Dynamic Threshold + Activation will come next.)
        """
        # Build feeding input once (used for all n)
        F = self.feeding_input(x)    # [N, C, H, W]

        # Simple initialization for Y(0): sigmoid(F) is a common choice
        y = torch.sigmoid(F)         # Y(0)

        # Run one iteration to demonstrate (you can loop self.iters)
        L = self.coupled_linking(y)  # L(1) from Y(0)
        U = self.modulation(F, L)    # U(1)

        # Return the ingredients so you can inspect them (and so we can add the next subsystems later)
        return {"F": F, "Y_prev": y, "L": L, "U": U}
