from pathlib import Path

# Model checkpoints
DATASET_CHECKPOINTS = {
    "DRIVE":     Path("outputs/checkpoints/[DRIVE] baseunet_dpcn_6_iters_64ch_msu_cbam_hassskip_w_augs_newDataloader_drive_patching.pth"),
    "CHASE-DB1": Path("outputs/checkpoints/[CHASEDB1] baseunet_dpcn_6_iters_64ch_2hl_64rt_cbam16_msu_cbam_hassskip_50_epochs_w_augs_newDataloader_CHASEDB1_patching.pth"),
    "STARE":     Path("outputs/checkpoints/[STARE] baseunet_dpcn_6_iters_64ch_2hl_64rt_cbam16_msu_cbam_hassskip_50_epochs_w_augs_newDataloader_STARE_patching.pth"),
    "ALL":       Path("outputs/checkpoints/[ALL] baseunet_dpcn_6_iters_64ch_2hl_64rt_cbam16_msu_cbam_hassskip_50_epochs_w_augs_newDataloader_all_datasets_patching.pth"),
}

# Unet checkpoints
UNET_CHECKPOINTS = {
    "DRIVE":     Path("outputs/checkpoints/unet/[DRIVE]_unet.pth"),
    "CHASE-DB1": Path("outputs/checkpoints/unet/[CHASEDB1]_unet.pth"),
    "STARE":     Path("outputs/checkpoints/unet/[STARE]_unet.pth"),
    "ALL":       Path("outputs/checkpoints/unet/[ALL]_unet.pth"),
}

IMAGE_SIZE_BY_DATASET = {k: 512 for k in DATASET_CHECKPOINTS.keys()}

# Set True ONLY if checkpoint expects model(x, fov=...)
USE_FOV_IN_MODEL = False