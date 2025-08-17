import torch
import torch.nn as nn
import torch.nn.functional as F


'''
Paper Source: https://arxiv.org/pdf/1807.06521
Code Source: https://github.com/Jongchan/attention-module

'''

'''
class BasicConv
    Purpose:
        Implementation helper (not in paper). Convenience block for Conv2d with optional BatchNorm and ReLU.

    Parameters:
        in_planes (int):            Number of input channels C_in.
        out_planes (int):           Number of output channels C_out.
        kernel_size (int or tuple): Conv kernel size.
        stride (int or tuple):      Conv stride. Default: 1.
        padding (int or tuple):     Conv padding. Default: 0.
        dilation (int or tuple):    Conv dilation. Default: 1.
        groups (int):               Grouped convolution groups. Default: 1.
        relu (bool):                If True, append ReLU. Default: True.
        bn (bool):                  If True, append BatchNorm2d. Default: True.
        bias (bool):                Conv bias. Usually False when BN is used. Default: False.

    Inputs:
        x = feature map
            dims: B x C_in x H x W

    Outputs:
        returns (ReLU?) ∘ (BatchNorm?) ∘ Conv2d (x)
            dims: B x C_out x H_out x W_out
            (H_out/W_out depend on kernel_size/stride/padding/dilation)

    Notes:
        - Keeps shapes “same” spatially when padding = (kernel_size-1)//2 and stride=1.
        - BN and ReLU can be disabled via flags; this class is used by SpatialGate() to build its 7×7 conv.
'''

# BasicConv: convenience wrapper for Conv2d + optional BN + optional ReLU
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()

        # Expose this block's output channel count for external wiring/introspection.
        # Mirrors nn.Conv2d.out_channels and is NOT consumed inside forward().
        self.out_channels = out_planes

        # core convolution layer
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        
        # optional BatchNorm
        # eps and momentum value not in paper
        # eps=1e-5 is a common default in PyTorch/CUDA kernels; 
            # small enough not to bias stats, large enough to avoid divide-by-zero.
        # momentum=0.01 = slow EMA; Helpful to keep the spatial gate’s distribution stable at eval time,
        # affine=True learns γ/β
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        
        # optional ReLU activation
        # SpatialGate omits ReLU after its 7x7 conv and relies on the final sigmoid.
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


# Flatten: reshape (B, C, 1, 1) -> (B, C)
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


'''
class ChannelGate (Channel Attention Module; CAM)
    Purpose (from paper):
        Produce a channel attention map by exploiting inter-channel relationships.
        Use both global average-pooling and global max-pooling, each passed through a shared MLP, summed, then squashed.

    Parameters:
        gate_channels (int): Number of channels C of the input feature map.
        reduction_ratio (int): Bottleneck ratio r for the shared MLP (C -> C/r -> C). Default: 16.
        pool_types (list[str]): Which spatial poolings to use. Paper uses ['avg','max'].
                                This implementation also supports ['lp','lse'] as extras (not in paper, but in author code).

    Inputs:
        x = feature map F
            dims: B x C x H x W

    Outputs:
        returns F'
            formula:    F' = M_c(F) ⊙ F
                        M_c(F) = σ( MLP(AvgPool(F)) + MLP(MaxPool(F)) )
            M_c shape: B x C x 1 x 1  (broadcast across H x W)
            F' shape:  B x C x H x W

    Notes:
        - Avg/Max in CAM are GLOBAL spatial poolings (kernel = H×W) performed per channel.
        - The MLP weights are shared between avg and max descriptors.
        - Apply CAM before SAM (channel→spatial) as per the paper’s best-performing sequential order.
'''
# ChannelGate: channel attention (avg/max/lp/lse pooling -> shared MLP -> sigmoid)
class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels


        # shared MLP: (B,C,1,1) -> Flatten (B,C) -> Linear(C->C/r) -> ReLU -> Linear(C/r->C) -> (B,C)
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )

        # types of pooling to use
        # paper only uses ["avg", "max"]
        self.pool_types = pool_types

    def forward(self, x):  # x: (B, C, H, W)
        channel_att_sum = None

        # iterate selected pooling types and sum their outputs after MLP
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                # global avg pool over HxW -> (B, C, 1, 1)
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)),
                                        stride=(x.size(2), x.size(3)))
                
                # pass through shared MLP to output shape (B, C)
                channel_att_raw = self.mlp(avg_pool)

            elif pool_type == 'max':
                # global max pool over HxW -> (B, C, 1, 1)
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)),
                                        stride=(x.size(2), x.size(3)))
                
                # pass through shared MLP to output shape (B, C)
                channel_att_raw = self.mlp(max_pool)

            # NOT USED IN PAPER
            elif pool_type == 'lp':
                # Lp pooling (p=2 here)
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)),
                                      stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)

            # NOT USED IN PAPER
            elif pool_type == 'lse':
                # log-sum-exp pooling (numerically stable)
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            # merge the output feature vectors using element-wise summation
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw # input with first function in self.pool_type applied
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        # scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        
        # channel gates in (0,1): shape (B, C)
        gates = torch.sigmoid(channel_att_sum)

        # reshape to (B,C,1,1) so it can broadcast over H,W
        gates = gates.unsqueeze(2).unsqueeze(3)  # (B, C, 1, 1)

        # expand as a view to (B,C,H,W). 
        # No memory copy; values are virtually repeated.
        scale = gates.expand_as(x)               # (B, C, H, W)

        # per-channel gating: y[b,c,h,w] = x[b,c,h,w] * gates[b,c,1,1]
        return x * scale




'''
class ChannelPool
    Purpose (from paper, verbatim):
        "Applying pooling operations along the channel axis is shown to be effective in highlighting informative regions"
    
    Inputs:
        x = Channel-refined feature map (F') 
            dims: B x C x H x W
    Outputs:
        returns [AvgPool(F'); MaxPool(F')]
            dims: B x 2 x H x W 
            a.k.a concatenated (torch.cat) two 1 x H x W spatial feature descriptors 
                one from avg-pooling (torch.mean),
                one from max-pooling (torch.max)

    Notes:
        This is the first step in SAM (Spatial Attention Module).
        Output is to be convolved with 7x7 kernel
'''
class ChannelPool(nn.Module):
    def forward(self, x):
        # torch.max(x,1)[0] -> max values across channel dim, keep shape (B, H, W)
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1),          # MaxPool along channel axis (1, H, W)
                          torch.mean(x, 1).unsqueeze(1)), dim=1)    # AvgPool along channel axis (1, H, W)


'''
class SpatialGate (aka Spatial Attention Module)
    Purpose (from paper, verbatim):
        "utilizes inter-spatial relationship of features; focuses on 'where' is an informative part, complementary to the
        channel attention"
    
    Inputs:
        x = Channel-refined feature map (F') 
            dims: B x C x H x W
    Outputs:
        returns F'' 
            formula:    F'' =  Mₛ(F') ⊗ F' 
                        Mₛ(F) = σ(𝑓⁷ˣ⁷([AvgPool(F); MaxPool(F)]))

    Notes:
        Computes spatial attention map Mₛ(F) via a 7x7 conv on channel-pooled spatial feature descriptor, applies F.sigmoid(),
        , sigmoid yields a 1 × H × W mask that is broadcast over channels and multiplied with the input.

'''
class SpatialGate(nn.Module): 
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()  # produces 2-channel map

        # BasicConv(2 channels -> 1 channel) with 7x7 kernel, no ReLU (relies on sigmoid later)
        self.spatial = BasicConv(in_planes=2, 
                                 out_planes=1, 
                                 kernel_size=kernel_size,
                                 stride=1,
                                 padding=(kernel_size - 1) // 2,
                                 relu=False)

    def forward(self, x):
        # compress channels (B, C, H, W) -> (B, 2, H, W)
        x_compress = self.compress(x) # uses ChannelPool()

        # 𝑓⁷ˣ⁷(B, 2, H, W) -> (B, 1, H, W)
        x_out = self.spatial(x_compress) # uses BasicConv() without ReLU

        # applies σ
        scale = F.sigmoid(x_out)  # spatial weights in (0,1)

        # applies ⊗ F'
        return x * scale # multiply input by spatial map (broadcast across channels)



'''
class CBAM
    Purpose:
        Apply Channel Attention (CAM) then Spatial Attention (SAM) sequentially to refine features—
        learning "what" (channels) and then "where" (spatial) to emphasize.

    Parameters:
        gate_channels (int): Number of channels C of the input feature map handled by this CBAM block.
        reduction_ratio (int): Bottleneck ratio r for CAM’s shared MLP (C -> C/r -> C). Default: 16.
        pool_types (list[str]): CAM pooling types; paper uses ['avg','max']. Extras ['lp','lse'] available here but not used.
        use_spatial (bool):     If True, apply SAM after CAM. If False, channel-only attention.

    Inputs:
        x = feature map
            dims: B x C x H x W

    Outputs:
        x = refined feature map after CAM (and SAM if enabled)
            dims: B x C x H x W

    Notes:
        - Sequential order channel→spatial matches the paper’s best-performing variant.
        - Keep CBAM off the final 1-channel logit maps in segmentation heads.
'''

# CBAM: channel attention followed by optional spatial attention
class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16,
                 pool_types=['avg', 'max'], use_spatial=True):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.use_spatial = use_spatial
        if use_spatial:     # optional way to skip spatial attention module
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)    # apply channel attention first
        if self.use_spatial:
            x_out = self.SpatialGate(x_out)  # then spatial attention
        return x_out



# helper: numerically stable log-sum-exp over spatial dims -> shape (B, C, 1)
# NOT USED IN PAPER
def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)  # (B, C, H*W)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)             # (B, C, 1)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs