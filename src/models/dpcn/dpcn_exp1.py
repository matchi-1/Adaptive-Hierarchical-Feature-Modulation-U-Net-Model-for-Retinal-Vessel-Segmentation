# src/models/dpcn_exp1.py
# paper faithful implementation of DPCN but threshold update is E(n) = exp(-aE)*E(n-1) + V_E * Y(n-1)
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d

# if not hasattr(torch.ops.torchvision, "deform_conv2d"):
#     raise RuntimeError("This build of torchvision lacks deform_conv2d. Install matching torch/vision wheels.")

# add docstring here on dpcn including all formulas
print("has op:", hasattr(torch.ops.torchvision, "deform_conv2d"))

# Hard-require deformable conv
if not hasattr(torch.ops.torchvision, "deform_conv2d"):
    raise RuntimeError(
        "This build of torchvision lacks deform_conv2d. "
        "Install matching torch/torchvision wheels."
    )


class DPCNIter(nn.Module):
    """
    One iteration of DPCN (paper-faithful math):


      Coupled Linking:   L(n) = DefConv( Y(n-1) )
      Modulation:        U(n) = F(n) * ( 1 + β * L(n) )
      Threshold:         E(n) = exp(-aE)*E(n-1) + V_E * Y(n-1)
                         (you can switch to (1-exp(-aE))*V_E if desired)
      Activation:        Y(n) = sigmoid( U(n) - E(n) )


    Shapes are all [N, C, H, W].
    """

    def __init__(self, channels: int, beta: float, aE: float, V_E: float):
        super().__init__()
        self.channels = channels

        # β, aE, V_E as hyperparams (frozen to match your latest code)
        self.beta = nn.Parameter(torch.tensor(float(beta)), requires_grad=False)   # learnable scalar parameter, will receive gradients and be updated by the optimizer during training
        self.aE  = nn.Parameter(torch.tensor(float(aE)),   requires_grad=False)   # decay constant (hyperparam) - how fast the threshold decays
        self.V_E = nn.Parameter(torch.tensor(float(V_E)),  requires_grad=False)   # growth scale (hyperparam) - how much the last activation raises the threshold

        # ---- 1.) Coupled Linking Subsystem Setup ----
        # (insert eq here later)

        k = 3  # 3x3 deformable kernel
        # in deformable conv, each location in the kernel (9 taps) needs 2 values:
        #           Δ𝑚 (offset for y-direction), Δ𝑛 (offset for x-direction),
        #           since for each pixel/sample point we compute it as:
        #           (𝑚 + i + Δ𝑚, 𝑛 + j + Δ𝑛) where (i,j) are fixed grid offsets (e.g. (i = -1, j =-1 --> top left neighbor/tap)).
        #
        # total offset channels = 2×3×3=18
        off_ch = 2 * k * k  # 18 channels for (dy,dx) per tap

        # Offset predictor: predicts offsets from the previous state Y(n-1)
        #
        # Conv2d input:  feature map [N, C_in, H, W] ; output: offsets [N, 18, H, W]
        #   self.channels = C_in (number of channels in Y(n-1) )
        #   off_ch = 18 (number of output channels = 2*k*k of this convolutional layer)
        #   kernel filter with shape (out_ch, in_ch, k, k) = (18, C, 3, 3), padding is 1 to keep same H,W
        # Conceptually, at each pixel (h,w), this layer outputs 18 values: (dy,dx) offsets for each of the 9 taps in the 3x3 kernel
        self.offset_conv = nn.Conv2d(channels, off_ch, kernel_size=3, padding=1)
        nn.init.zeros_(self.offset_conv.weight)  # init near zero so first == plain conv
        nn.init.zeros_(self.offset_conv.bias)

        # Kernel weights for the deformable conv (W(i,j) in the paper)
        w = torch.empty(channels, channels, k, k) # initialize weight tensor with shape (out_ch, in_ch, k, k) since L(n) = Y(n-1) which means they should have the same shape
        nn.init.kaiming_normal_(w, nonlinearity="relu") # give weights good starting values so they won't collapse or explode during training
        self.weight = nn.Parameter(w)  # make weight a learnable parameter thru backpropagation
        self.bias   = nn.Parameter(torch.zeros(channels)) # add bias term per output channel ; after summing all taps, add bias

        # normalization (helps stability)
        self.norm_L = nn.BatchNorm2d(channels) # convs can produce large values, batchnorm re-centers and rescales each channel to have mean=0, std=1 (per batch)

        # optional norm on U(n) to stabilize multiplicative modulation
        self.norm_U = nn.BatchNorm2d(channels)

    # -------- Coupled Linking Subsystem ----------
    """
    Coupled Linking Subsystem:
    L(n) = DefConv(Y(n-1))
    TODO: add expounded formula here


    Args:
        y_prev: previous iteration output Y(n-1), shape [N,C,H,W]
    Returns:
        L: locally enhanced feature map, shape [N,C,H,W]
    """
    def forward(self, y_prev: torch.Tensor, F: torch.Tensor, E_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        L = self.norm_L(L) # stabilize activations

        # -------- Modulation Subsystem ----------
        """
        Modulation Subsystem
        U(n) = F(n) * (1 + β * L(n))


        Args:
            F: feeding input feature map
            L: linking feature map ; output of coupled linking subsystem
        Returns:
            U: modulated internal state controlled by β (combining feeding and linking)
        """
        beta = torch.clamp(self.beta, 0.0, 1.0)   # keep β in a sane range so modulation doesn’t blow up or flip signs
        U = F * (1.0 + beta * L)                  # formula for modulation. states of the feeding units and linking units combine in a second-order manner to produce the internal state 𝑈(𝑛) of the neuron, with the degree ofcombination controlled by the coefficient B
        U = self.norm_U(U)

        #  ---------- Dynamic Threshold Subsystem ----------
        """
        Dynamic Threshold Subsystem
        E(n) = e^(-aE) * E(n-1) + V_E * Y(n-1)


        Args:
            E_prev: previous threshold E(n-1)
            y_prev: previous output Y(n-1)
        Variables:
            aE: decay constant (hyperparam) - how fast the threshold decays
            V_E: growth scale (hyperparam) - how much the last activation raises the threshold
        Returns:
            U: modulated internal state controlled by β (combining feeding and linking)
        """
        # keep exp argument non-negative to avoid overflow on weird aE
        aE = torch.clamp(self.aE, min=1e-6)                      # ensure aE is non-negative and greater than 10^-6 (if its non-negative, it'll be more than one so it should remain a positive number)
        decay = torch.exp(-aE)                                   # computes decay rate -- how much of E(n-1) we carry forward. Results in a scalar in from (0,1)
        #E = decay * E_prev + self.V_E * y_prev
        grow  = (1.0 - decay) * self.V_E  
        E = decay * E_prev + grow * y_prev                       # updates the adaptive threshold using: decay term (ae) + growth term (V_e) proportional to previous output y (Y(n-1))

        # ---------- Activation Subsystem ----------
        """
        Activation Subsystem
        Y(n) = sigmoid( U(n) - E(n) )


        Args:
            U: modulated input from modulation subsystem
            E: current threshold from dynamic threshold subsystem
        Returns:
            Y: subtracted and squashed output
        """
        y = torch.sigmoid(U - E) # Y(n): subtract the threshold from the modulated input -- only inputs above the threshold will pass strongly; squashed to [0,1] range using sigmoid
        return y, E


class DPCN(nn.Module):
    """
    Wrapper that runs T iterations of DPCNIter on shallow features.
    Returns all iteration outputs stacked: [N, T, C, H, W].
    """

    def __init__(
        self,
        in_ch: int,
        channels: Optional[int] = None,
        iters: int = 3,
        beta_init: float = 0.5,
        aE: float = 0.5,
        V_E: float = 1.0,
        clamp_each_iter: bool = True,
        project_out: bool = False,
    ):
        super().__init__()
        channels = channels or in_ch
        self.iters = int(iters)
        self.channels = channels
        self.clamp_each_iter = clamp_each_iter

        # ---- project input to internal channels once; F(n) will reuse this ----
        # if in_ch == channels, just use identity (no extra cost or no op)
        # else use 1x1 conv as a channel aligner -- mixes channels without changing H,W ; example if in_ch=1, channels=32 1×1 conv learns 32 filters over the single input channel, giving an 32-channel
        self.proj_in  = nn.Identity() if in_ch == channels else nn.Conv2d(in_ch, channels, kernel_size=1)

        # optional projection back
        self.proj_out = nn.Identity() if not project_out else nn.Conv2d(channels, in_ch, kernel_size=1)

        # one iteration cell with paper-faithful math
        # ---- learnable β for modulation (clamp at runtime to [0,1]) ----
        self.cell = DPCNIter(
            channels=channels,
            beta=beta_init,
            aE=aE,
            V_E=V_E,
        )

    # -------- Feeding Input Subsystem: F(n) = I_OR ----------
    """
    Feeding Input Subsystem
    F(n) = I_OR
    Args:
        x: original shallow feature (or preprocessed image), shape [N, C_in, H, W]
    Returns:
        F: projected/computed input feature map, shape [N, C, H, W] where C = self.channels
    Notes:
        in practice, we build F once from x (projection) and reuse it every iteration.
    """
    def forward(self, x: torch.Tensor, fov: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Feeding input (Eq. 2.2): compute once and reuse
        F = self.proj_in(x)                          # [N, C, H, W]

        # Initialize states for iteration 0
        y = torch.sigmoid(F)                         # reasonable Y(0) -- values are from 0 to 1 so initial "activity" is meaningful
        E = torch.zeros_like(F)                      # E(0)=0 -- no initial threshold

        # --- make fov safe (device + dtype) ---
        if fov is not None:
            fov = fov.to(y.device).type_as(y)

        # collect outputs per iteration
        ys = []    # will hold each Y(n) for n in [1..T]

        # run dpcn for the specified number of iterations
        for _ in range(self.iters):
            print(f"DPCN ITER_exp1: {_+1}/{self.iters}")
            # 1) Coupled linking: L(n) + 2) Modulation + 3) Dynamic threshold + 4) Activation
            y, E = self.cell(y_prev=y, F=F, E_prev=E)  # produce contextual map using deformable conv; combines raw intensity input F with contextual link L + controlled by learnable β; updates the adaptive threshold; squashes to [0,1] range using sigmoid

            # 5) FOV clamp each step (keeps state zero outside retina)
            if fov is not None and self.clamp_each_iter:
                y = y * fov

            ys.append(y)  # store current output

        # if we didn’t clamp each iteration, at least clamp the final output:
        if fov is not None and not self.clamp_each_iter:
            ys[-1] = ys[-1] * fov
            #y = y * fov

        # stack outputs along new dim: [N, T, C, H, W]
        ys = torch.stack(ys, dim=1)
        return ys
