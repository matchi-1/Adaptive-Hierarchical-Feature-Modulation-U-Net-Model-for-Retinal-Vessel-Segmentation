import torch, torchvision
print("torch:", torch.__version__, "| torchvision:", torchvision.__version__)

# The op must be registered here:
print("has op:", hasattr(torch.ops.torchvision, "deform_conv2d"))

from torchvision.ops import deform_conv2d  # should import cleanly

# quick run
x   = torch.randn(1, 8, 32, 32)
w   = torch.randn(8, 8, 3, 3)
off = torch.zeros(1, 18, 32, 32)  # 2*k*k with k=3
y   = deform_conv2d(x, off, w, bias=None, padding=1)
print("ok shape:", y.shape)
