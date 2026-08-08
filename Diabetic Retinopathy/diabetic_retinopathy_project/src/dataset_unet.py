import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path

class REFUGEDataset(Dataset):
    """
    Dataset class for REFUGE2 Optic Disc and Cup Segmentation.
    Expects directory structure:
        data_dir/
            images/
            mask/
    """
    def __init__(self, data_dir: str, is_train: bool = True):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.masks_dir = self.data_dir / "mask"
        
        # Collect image paths
        self.image_paths = sorted(list(self.images_dir.glob("*.jpg")))
        
        self.is_train = is_train
        self.transform = self._get_transforms()
        
    def _get_transforms(self):
        if self.is_train:
            return A.Compose([
                A.Resize(256, 256),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(256, 256),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
            
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Try different possible extensions for the mask
        mask_path_bmp = self.masks_dir / (img_path.stem + ".bmp")
        mask_path_png = self.masks_dir / (img_path.stem + ".png")
        mask_path = mask_path_bmp if mask_path_bmp.exists() else mask_path_png
        
        # Read image (RGB)
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read mask (Grayscale)
        mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if mask_raw is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
            
        # Map pixels: 255 -> 0, 128 -> 1, 0 -> 2
        # Start with all zeros
        mask = np.zeros_like(mask_raw, dtype=np.int64)
        mask[mask_raw == 128] = 1 # Optic Disc Rim
        mask[mask_raw == 0] = 2   # Optic Cup
        
        # Apply albumentations
        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask'].long() # CrossEntropyLoss expects torch.int64
        
        return image, mask
