# src/data/augmentations.py
from __future__ import annotations
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

'''
    """
    Train-time augmentations for retinal vessel segmentation (single-channel input).

    Design goals:
    - Preserve thin vessels: use light geometry only (small rotate/scale/shift).
    - Keep black borders outside FOV: border_mode=0 (constant 0) everywhere.
    - Add mild photometric noise/blur to improve robustness across cameras.
    - Do NOT do heavy elastic/perspective warps (they bend vessels).
    - Final output is torch tensors: image -> [1,H,W] float32 in [0,1]; mask -> [1,H,W] {0,1}.

    Expected inputs to Compose:
      image : HxW float32 in [0,1]
      mask  : HxW float32 in {0,1}
    Returns:
      A.Compose that produces:
        image : torch.FloatTensor [1,H,W]
        mask  : torch.FloatTensor [1,H,W]
    """
'''

def get_train_augs(size: int = 512):  # by default, expects images resized to 512 × 512

    return A.Compose([  # albumentations’ way to bundle a list of transforms into one pipeline
        
        # --- light, vessel-safe geometry (applied jointly to image & mask) ---
        A.Affine(
            scale=(0.95, 1.05),                                      # can zoom in/out by up to ~5%
            translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},  # can shift left/right/up/down by up to 3% of width/height
            rotate=(-15, 15),                                        # can rotate between –15° and +15°
            shear={"x": (0, 0), "y": (0, 0)},                        # no shear to avoid bending vessels
            interpolation=cv2.INTER_LINEAR,                          # bilinear for image
            mask_interpolation=cv2.INTER_NEAREST,                    # nearest for masks/FOV
            mode="constant",                                         # fills areas outside the FOV with cval
            cval=0,                                                  # fill value = 0 (black) to match FOV background
            p=0.75                                                   # applies this transform 75% of the time
        ),

        A.HorizontalFlip(p=0.50), # flip left<->right 50% of the time
        A.VerticalFlip(p=0.20),   # flip top<->bottom 20% of the time

        # --- photometric jitter on single-channel image (mild) Mimics different lighting or imaging conditions ---
        A.RandomBrightnessContrast(
            brightness_limit=0.15,   # darken or brighten by up to 15%
            contrast_limit=0.15,     # reduce/increase contrast by up to 15%
            p=0.40                   # applied 40% of the time
        ),

        # noise/Blur: use one at a time (small, realistic)
        A.OneOf([  # pick one of the listed transforms (or none, 75% of the time)
            A.MultiplicativeNoise(multiplier=(0.95, 1.05), per_channel=False),  # multiplies pixel values by ~0.95–1.05, simulating uneven illumination
        ], p=0.25),  # applies one of these noise types 25% of the time

        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),      # simulates slight out-of-focus or denoising (kernel size 3–5)
            A.MotionBlur(blur_limit=5),             # simulates camera shake or patient eye motion
        ], p=0.20),  # applies one of these blurs 20% of the time

        # defensive: ensure final size; masks use nearest under the hood
        A.Resize(size, size, interpolation=cv2.INTER_NEAREST),  # enforces the output to be exactly size x size (512 × 512)

        # grayscale HxW -> [1,H,W]
        ToTensorV2(transpose_mask=True),  # converts NumPy arrays to PyTorch tensors
    ],
    additional_targets={"fov": "mask"}  # <— IMPORTANT: carry FOV through the same geometry as mask
    )


def get_val_augs(size: int = 512):
    """
    Validation/Test 'augmentations' (really just formatting):
    - No randomness.
    - Enforce size and convert to tensors.
    """

    # no random augmentations here -> validation should reflect real input
    # only enforces the correct size and converts to PyTorch tensors
    return A.Compose([
        A.Resize(size, size, interpolation=0),
        ToTensorV2(transpose_mask=True),
    ],
        additional_targets={"fov": "mask"}  # <— IMPORTANT: carry FOV through the same geometry as mask
    )
