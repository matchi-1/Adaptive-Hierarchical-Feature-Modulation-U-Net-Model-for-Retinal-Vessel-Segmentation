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
    "DRIVE":     Path("outputs/checkpoints/unet/[CHASEDB1_UNET] baseunet_w_halfaug_newDataloader_chasedb1.pth"), # change to drive
    "CHASE-DB1": Path("outputs/checkpoints/unet/[CHASEDB1_UNET] baseunet_w_halfaug_newDataloader_chasedb1.pth"),
    "STARE":     Path("outputs/checkpoints/unet/[STARE_UNET] baseunet_w_halfaug_newDataloader_stare.pth"),
    "ALL":       Path("outputs/checkpoints/unet/[CHASEDB1_UNET] baseunet_w_halfaug_newDataloader_chasedb1.pth"), # change to all
}

IMAGE_SIZE_BY_DATASET = {k: 512 for k in DATASET_CHECKPOINTS.keys()}

# Set True ONLY if checkpoint expects model(x, fov=...)
USE_FOV_IN_MODEL = False