# src/models/dpcn_exp1.py
# VAT implementation
# paper faithful implementation of DPCN but threshold update is E(n) = exp(-aE)*E(n-1) + V_E * Y(n-1) 
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d

# if not hasattr(torch.ops.torchvision, "deform_conv2d"):
#     raise RuntimeError("This build of torchvision lacks deform_conv2d. Install matching torch/vision wheels.")

# add docstring here on dpcn including all formulas
#print("has op:", hasattr(torch.ops.torchvision, "deform_conv2d"))

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

    def __init__(self, channels: int, beta: float, aE: float, V_E: float, threshold_mode: str = "vat"):
        super().__init__()
        self.channels = channels
        self.threshold_mode = threshold_mode

        # β, aE, V_E as hyperparams (frozen rn, revisit later if we want to make them learnable parameters) 
        self.beta = nn.Parameter(torch.tensor(float(beta)), requires_grad=False)   # scalar parameter (hyperparam) controls how much linking affects modulation
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
        self.norm_L = nn.BatchNorm2d(channels)
        #self.norm_L = nn.GroupNorm(8, channels) # convs can produce large values, batchnorm re-centers and rescales each channel to have mean=0, std=1 (per batch)

        # optional norm on U(n) to stabilize multiplicative modulation
        self.norm_U = nn.BatchNorm2d(channels)
        #self.norm_U = nn.GroupNorm(8, channels)

    
    def forward(self, y_prev: torch.Tensor, F: torch.Tensor, E_prev: torch.Tensor, Vconf: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. predict offsets from Y(n-1) using normal Conv2d
        offsets = self.offset_conv(y_prev)  # [N,18,H,W]


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
        Dynamic Threshold Subsystem (switchable)

        Computes the adaptive threshold E(n) from the previous threshold and activity.
        The exact update rule depends on `threshold_mode`:

            - "paper":
                E(n) = exp(-aE) * E(n-1) + V_E * Y(n-1)
            
            - "paper_mod":
                E(n) = exp(-aE) * E(n-1) + (1 - exp(-aE)) * V_E * Y(n-1)

            - "vat" (vessel-aware thresholding):
                E(n) = exp(-aE) * E(n-1) + V_E * ( Y(n-1) ⊙ Vconf )
                where Vconf ∈ [0,1] is a vessel-confidence map broadcast over channels.

            - "scaled_vat":
                E(n) = exp(-aE) * E(n-1) + (1 - exp(-aE)) * V_E * ( Y(n-1) ⊙ Vconf )

        Args:
            E_prev (Tensor): Previous threshold E(n-1), shape [N, C, H, W].
            y_prev (Tensor): Previous activation/output Y(n-1), shape [N, C, H, W].
            Vconf (Optional[Tensor]): Vessel-confidence map, shape [N, 1, H, W].
                Required for modes "vat" and "scaled_vat"; ignored in "paper" mode.

        Hyperparameters:
            aE (float): Non-negative decay constant controlling how fast E decays.
            V_E (float): Growth scale controlling how strongly the last activation raises E.

        Behavior:
            decay = exp(-aE) ∈ (0,1). Larger aE ⇒ faster decay.
            In "vat"/"scaled_vat", Y(n-1) is modulated by Vconf to increase thresholds
            primarily where vessels are likely.

        Returns:
            Tensor: Current threshold E(n), shape [N, C, H, W].
        """
        
        aE  = torch.clamp(self.aE, min=1e-6) # ensure aE is non-negative and greater than 10^-6 (if its non-negative, it'll be more than one so it should remain a positive number)
        V_E = self.V_E
        decay = torch.exp(-aE) # computes decay rate -- how much of E(n-1) we carry forward. Results in a scalar in from (0,1)

        mode = self.threshold_mode.lower()
        if mode == "paper":
            grow_term = V_E * y_prev
        elif mode == "paper_mod":
            grow_term = (1.0 - decay) * V_E * y_prev
        elif mode == "vat":
            if Vconf is None:
                raise ValueError("Vconf is required for 'vat' mode.")
            grow_term = V_E * (y_prev * Vconf)   # broadcast Vconf [N,1,H,W] → [N,C,H,W]
        elif mode == "scaled_vat":
            if Vconf is None:
                raise ValueError("Vconf is required for 'scaled_vat' mode.")
            grow_term = (1.0 - decay) * V_E * (y_prev * Vconf)
        else:
            raise ValueError(f"Unknown threshold_mode: {self.threshold_mode}")

        E = decay * E_prev + grow_term


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
        threshold_mode: str = "scaled_vat"  # "paper", "paper_mod", "vat", "scaled_vat"
    ):
        super().__init__()
        channels = channels or in_ch
        self.iters = int(iters)
        self.channels = channels
        self.clamp_each_iter = clamp_each_iter
        self.threshold_mode = threshold_mode

        # ---- project input to internal channels once; F(n) will reuse this ----
        # if in_ch == channels, just use identity (no extra cost or no op)
        # else use 1x1 conv as a channel aligner -- mixes channels without changing H,W ; example if in_ch=1, channels=32 1×1 conv learns 32 filters over the single input channel, giving an 32-channel
        self.proj_in  = nn.Identity() if in_ch == channels else nn.Conv2d(in_ch, channels, kernel_size=1)

        # optional projection back
        self.proj_out = nn.Identity() if not project_out else nn.Conv2d(channels, in_ch, kernel_size=1)

        
        # --- Vessel Confidence Estimation via Lightweight CNN (Eq. 2.6–2.7) ---
        # Simple attention-style module:
        # Fshallow = Conv3×3( ReLU( Conv3×3(I) ) )
        # Vconf    = σ( Conv1×1(Fshallow) ), Vconf ∈ [0,1], shape [N,1,H,W]
        self.vconf_branch = nn.Sequential(
            nn.Conv2d(in_ch, max(8, in_ch), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(8, in_ch), max(8, in_ch), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(8, in_ch), 1, kernel_size=1),
            nn.Sigmoid()
        )

        # one iteration cell with paper-faithful math
        # ---- learnable β for modulation (clamp at runtime to [0,1]) ----
        self.cell = DPCNIter(
            channels=channels,
            beta=beta_init,
            aE=aE,
            V_E=V_E,
            threshold_mode=threshold_mode
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

        # --- Vessel Confidence Map (computed once) ---
        # by default from raw input x; switch to F if you prefer feature-space confidence
        Vconf = self.vconf_branch(x).to(y.device).type_as(y)  # [N,1,H,W]
        Vconf = Vconf.to(y.device).type_as(y)
        
        #print("Vconf:", Vconf.shape, float(Vconf.min()), float(Vconf.max()))

        if fov is not None:
            # keep Vconf zero outside retina to avoid spurious growth
            Vconf = Vconf * fov

        # --- make fov safe (device + dtype) ---
        if fov is not None:
            fov = fov.to(y.device).type_as(y)

        # collect outputs per iteration
        ys = []    # will hold each Y(n) for n in [1..T]

        # run dpcn for the specified number of iterations
        for _ in range(self.iters):
            #print(f"DPCN ITER_exp1: {_+1}/{self.iters}")
            # 1) Coupled linking: L(n) + 2) Modulation + 3) Dynamic threshold + 4) Activation
            y, E = self.cell(y_prev=y, F=F, E_prev=E, Vconf=Vconf if self.threshold_mode != "paper" and self.threshold_mode != "paper_mod" else None) # produce contextual map using deformable conv; combines raw intensity input F with contextual link L + controlled by learnable β; updates the adaptive threshold; squashes to [0,1] range using sigmoid

            # 5) FOV clamp each step (keeps state zero outside retina)
            if fov is not None and self.clamp_each_iter:
                y = y * fov

            ys.append(y)  # store current output

            #print("E stats:", float(E.mean()), float(E.std()))


        # if we didn’t clamp each iteration, at least clamp the final output:
        if fov is not None and not self.clamp_each_iter:
            ys[-1] = ys[-1] * fov
            #y = y * fov

        # stack outputs along new dim: [N, T, C, H, W]
        ys = torch.stack(ys, dim=1)
        return ys
