from pathlib import Path

# Model checkpoints
DATASET_CHECKPOINTS = {
    "DRIVE":     Path("outputs/checkpoints/[DRIVE] MATHFI.pth"),
    "CHASE-DB1": Path("outputs/checkpoints/[CHASEDB1] MATHFI.pth"),
    "STARE":     Path("outputs/checkpoints/[STARE] MATHFI.pth"),
    "ALL":       Path("outputs/checkpoints/[ALL] MATHFI.pth"),
}

# Unet checkpoints
UNET_CHECKPOINTS = {
    "DRIVE":     Path("outputs/checkpoints/unet/[DRIVE_UNET] base_unet.pth"), 
    "CHASE-DB1": Path("outputs/checkpoints/unet/[CHASEDB1_UNET] base_unet.pth"),
    "STARE":     Path("outputs/checkpoints/unet/[STARE_UNET] base_unet.pth"),
    "ALL":       Path("outputs/checkpoints/unet/[ALL_UNET] base_unet.pth"),
}

IMAGE_SIZE_BY_DATASET = {k: 512 for k in DATASET_CHECKPOINTS.keys()}

# Set True ONLY if checkpoint expects model(x, fov=...)
USE_FOV_IN_MODEL = False