# src/models/dpcn_exp.py
# DPCN implemented from scratch but not formatted correctly yet for training
from typing import Optional
import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d

# if not hasattr(torch.ops.torchvision, "deform_conv2d"):
#     raise RuntimeError("This build of torchvision lacks deform_conv2d. Install matching torch/vision wheels.")

# add docstring here on dpcn including all formulas
print("has op:", hasattr(torch.ops.torchvision, "deform_conv2d"))

class DPCN(nn.Module):
    def __init__(self, 
                 in_ch,                 # number of input channels (should be 1 for grayscale input from preprocessing)
                 channels=None,         # internal number of channels (if None, same as in_ch, but should be 32/64/128 for best results)
                 iters=3,               # number of DPCN iterations (T)
                 beta_init=0.5,         # initial value for learnable β (will be clamped to [0,1] at runtime)
                 aE=0.5,                # decay constant for dynamic threshold subsystem
                 V_E=1.0,               # growth scale for dynamic threshold subsystem
                 clamp_each_iter=True   # whether to clamp output to FOV each iteration (True = clamp each iter, False = only final)
                 ):
        
        super().__init__()
        self.in_ch = in_ch
        self.channels = channels or in_ch   # default: same channels as input (should be 32 / 64 / 128)
        self.iters = iters
        self.clamp_each_iter = clamp_each_iter

        # ---- project input to internal channels once; F(n) will reuse this ----
        # if in_ch == channels, just use identity (no extra cost or no op)
        # else use 1x1 conv as a channel aligner -- mixes channels without changing H,W ; example if in_ch=1, channels=32 1×1 conv learns 32 filters over the single input channel, giving an 32-channel
        self.proj_in = nn.Identity() if self.in_ch == self.channels else nn.Conv2d(self.in_ch, self.channels, 1)

        # ---- learnable β for modulation (clamp at runtime to [0,1]) ----
        self.beta = nn.Parameter(torch.tensor(float(beta_init)), requires_grad=False)   # learnable scalar parameter, will receive gradients and be updated by the optimizer during training

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
        weight = torch.empty(self.channels, self.channels, k, k) # initialize weight tensor with shape (out_ch, in_ch, k, k) since L(n) = Y(n-1) which means they should have the same shape
        nn.init.kaiming_normal_(weight, nonlinearity="relu") # give weights good starting values so they won't collapse or explode during training 
        self.weight = nn.Parameter(weight)  # make weight a learnable parameter thru backpropagation
        self.bias   = nn.Parameter(torch.zeros(self.channels)) # add bias term per output channel ; after summing all taps, add bias

        # normalization (helps stability)
        self.norm = nn.BatchNorm2d(self.channels) # convs can produce large values, batchnorm re-centers and rescales each channel to have mean=0, std=1 (per batch)


        # --- Dynamic Threshold hyperparams from Eq.(5) ---
        # freeze these as hyperparams for now (no gradients)
        self.aE  = nn.Parameter(torch.tensor(float(aE)),  requires_grad=False)  # decay constant
        self.V_E = nn.Parameter(torch.tensor(float(V_E)), requires_grad=False)  # growth scale




    # !! ---- SUBSYSTEM FUNCTIONS ---- !!

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
    def coupled_linking(self, y_prev):
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
    def feeding_input(self, x):
        F = self.proj_in(x)  # [N, channels, H, W]
        return F

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
    def modulation(self, F, L):
        beta = torch.clamp(self.beta, 0.0, 0.1)  # keep β in a sane range so modulation doesn’t blow up or flip signs
        U = F * (1.0 + beta * L)                 # formula for modulation. states of the feeding units and linking units combine in a second-order manner to produce the internal state 𝑈(𝑛) of the neuron, with the degree ofcombination controlled by the coefficient B
        U = self.norm(U)
        return U
    
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
    def update_threshold(self, E_prev: torch.Tensor, y_prev: torch.Tensor) -> torch.Tensor:
        # keep exp argument non-negative to avoid overflow on weird aE
        aE = torch.clamp(self.aE, min=1e-6)         # ensure aE is non-negative and greater than 10^-6 (if its non-negative, it'll be more than one so it should remain a positive number)
        decay = torch.exp(-aE)                      # computes decay rate -- how much of E(n-1) we carry forward. Results in a scalar in from (0,1)
        E = (decay * E_prev) + (self.V_E * y_prev)  # updates the adaptive threshold using: decay term (ae) + growth term (V_e) proportional to previous output y (Y(n-1))
        
        # ! IMPORTANT, GO BACK TO THIS LATER:: stable numerically, experiment with this later
        # grow  = (1.0 - decay) * self.V_E   
        # E = decay * E_prev + grow * y_prev  
        return E           


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
    def activate(self, U: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(U - E) # Y(n): subtract the threshold from the modulated input -- only inputs above the threshold will pass strongly; squashed to [0,1] range using sigmoid
    

    # ---- FULL DPCN FORWARD PASS ----
    def forward(self,
                x: torch.Tensor,  # raw grayscale input preprocessed to [N,1,H,W] image
                fov: Optional[torch.Tensor] = None) -> torch.Tensor: # 
        """
        x:   [N, C_in, H, W]  (raw grayscale or shallow features)
        fov: [N, 1,   H, W]   (optional {0,1} mask)
        """
        # Feeding input (Eq. 2.2): compute once and reuse
        F = self.feeding_input(x)                          # [N, C, H, W]

        # Initialize states for iteration 0
        y = torch.sigmoid(F)                               # reasonable Y(0) -- values are from 0 to 1 so initial "activity" is meaningful
        E = torch.zeros_like(F)                            # E(0)=0 -- no initial threshold
        
        # collect outputs per iteration
        ys = []    # will hold each Y(n) for n in [1..T]

        # run dpcn for the specified number of iterations
        for _ in range(self.iters):
           #print(f"DPCN ITER: {_+1}/{self.iters}")
            # 1) Coupled linking: L(n)
            L = self.coupled_linking(y) # produce contextual map using deformable conv

            # 2) Modulation: U(n)  
            # U = self.modulation_add(F, L)
            U = self.modulation(F, L)  # combines raw intensity input F with contextual link L + controlled by learnable β

            # 3) Dynamic threshold: E(n) 
            E = self.update_threshold(E, y)  # updates the adaptive threshold using: decay term (ae) + growth term (V_e) proportional to previous output y (Y(n-1))

            # 4) Activation: Y(n)    
            y = self.activate(U, E) # squashes to [0,1] range using sigmoid

            # 5) FOV clamp each step (keeps state zero outside retina)
            if fov is not None and self.clamp_each_iter:
                y = y * fov

            ys.append(y)  # store current output

        # if we didn’t clamp each iteration, at least clamp the final output:
        if fov is not None and not self.clamp_each_iter:
            ys[-1] = y[-1] * fov
            #y = y * fov

        # stack outputs along new dim: [N, T, C, H, W]
        ys = torch.stack(ys, dim=1)  
        return ys
