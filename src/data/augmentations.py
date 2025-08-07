import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_training_augmentation():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),  # ±15°
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),  # simulates gamma 0.8–1.2
        A.Normalize(),  # standard normalization to mean=0, std=1
        ToTensorV2(),   # converts to PyTorch tensor (C, H, W)
    ])
