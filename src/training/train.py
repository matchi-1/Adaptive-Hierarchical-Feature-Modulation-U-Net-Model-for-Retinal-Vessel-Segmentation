import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.preprocessing import preprocess_image_clahe, preprocess_image_rgb, preprocess_mask
from data.dataloader import get_dataloader
from data.augmentations import get_training_augmentation
from glob import glob

data_dir = "../../data/raw/"

chasedb1_train_img_dir = sorted(glob(os.path.join(data_dir, "CHASEDB1/training/images/*.jpg")))
chasedb1_train_mask_dir = sorted(glob(os.path.join(data_dir, "CHASEDB1/training/1st_manual/*.png")))

chasedb1_test_img_paths = sorted(glob(os.path.join(data_dir, "CHASEDB1/test/images/*.jpg")))
chasedb1_test_mask_paths = sorted(glob(os.path.join(data_dir, "CHASEDB1/test/1st_manual/*.png")))

drive_train_img_dir = sorted(glob(os.path.join(data_dir, "DRIVE/training/images/*.png")))
drive_train_mask_dir = sorted(glob(os.path.join(data_dir, "DRIVE/training/1st_manual/*.png")))

drive_test_img_paths = sorted(glob(os.path.join(data_dir, "DRIVE/test/images/*.png")))
drive_test_mask_paths = sorted(glob(os.path.join(data_dir, "DRIVE/test/1st_manual/*.png")))

stare_train_img_dir = sorted(glob(os.path.join(data_dir, "STARE/training/images/*.jpg")))
stare_train_mask_dir = sorted(glob(os.path.join(data_dir, "STARE/training/1st_manual/*.jpg")))

stare_test_img_paths = sorted(glob(os.path.join(data_dir, "STARE/test/images/*.jpg")))
stare_test_mask_paths = sorted(glob(os.path.join(data_dir, "STARE/test/1st_manual/*.jpg")))


chasedb1_test_loader = get_dataloader(
    image_paths=chasedb1_test_img_paths,
    mask_paths=chasedb1_test_mask_paths,
    batch_size=2,
    shuffle=False,  
    num_workers=2
)


chasedb1_train_loader = get_dataloader(
    image_paths=chasedb1_train_img_dir,
    mask_paths=chasedb1_train_mask_dir,
    batch_size=2,
    shuffle=True,
    num_workers=2
)

stare_test_loader = get_dataloader(
    image_paths=stare_test_img_paths,
    mask_paths=stare_test_mask_paths,
    batch_size=2,
    shuffle=False,
    num_workers=2
)

stare_train_loader = get_dataloader(
    image_paths=stare_train_img_dir,
    mask_paths=stare_train_mask_dir,
    batch_size=2,
    shuffle=True,
    num_workers=2
)

drive_test_loader = get_dataloader(
    image_paths=drive_test_img_paths,
    mask_paths=drive_test_mask_paths,
    batch_size=2,
    shuffle=False,
    num_workers=2
)

drive_train_loader = get_dataloader(
    image_paths=drive_train_img_dir,
    mask_paths=drive_train_mask_dir,
    batch_size=2,
    shuffle=True,
    num_workers=2
)


if __name__ == "__main__":
    # Example: show shape of a few batches from one loader
    for i, (images, masks) in enumerate(drive_train_loader):
        print(f"Batch {i}:")
        print(f"  Image shape: {images.shape}")  # (B, C, H, W)
        print(f"  Mask shape:  {masks.shape}")   # (B, 1, H, W)
        if i == 2:  # stop after 3 batches
            break