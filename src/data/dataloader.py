from torch.utils.data import Dataset, DataLoader
import torch
import os
from .preprocessing import preprocess_image_clahe, preprocess_mask, preprocess_image_rgb

class RetinalDataset(Dataset):
    def __init__(self, image_paths, mask_paths, use_clahe=True, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.use_clahe = use_clahe
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = preprocess_image_clahe(image_path) if self.use_clahe else preprocess_image_rgb(image_path)
        mask = preprocess_mask(mask_path)

        # Convert to HWC for albumentations
        image = image.squeeze() if image.shape[0] == 1 else image.transpose(1, 2, 0)
        mask = mask.squeeze()

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            image = torch.tensor(image).unsqueeze(0).float() if image.ndim == 2 else torch.tensor(image).permute(2, 0, 1).float()
            mask = torch.tensor(mask).unsqueeze(0).float()
        

        return image, mask

def get_dataloader(image_paths, mask_paths, use_clahe=True, transform=None, batch_size=8, shuffle=True, num_workers=2):
    dataset = RetinalDataset(
        image_paths=image_paths,
        mask_paths=mask_paths,
        use_clahe=use_clahe,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        # pin_memory=True
    )

    return loader