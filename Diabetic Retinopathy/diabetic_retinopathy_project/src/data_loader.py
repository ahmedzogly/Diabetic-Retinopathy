"""
Data Loading and Preprocessing Pipeline for APTOS 2019 Diabetic Retinopathy
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from typing import Tuple, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Class mapping: 0-4 as per APTOS
DR_CLASSES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

CLASS_NAMES_AR = {
    0: "لا اعتلال",
    1: "خفيف",
    2: "متوسط",
    3: "شديد",
    4: "تكاثري"
}

class DRDataset(Dataset):
    """PyTorch Dataset for Diabetic Retinopathy classification"""
    
    def __init__(self, 
                 df: pd.DataFrame, 
                 img_dir: str, 
                 transform: Optional[transforms.Compose] = None,
                 albumentations_transform: Optional[A.Compose] = None,
                 is_train: bool = True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.albumentations_transform = albumentations_transform
        self.is_train = is_train
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = str(row['id_code']) + '.png'
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load image
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"cv2.imread returned None for {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = circle_crop(image)
        except Exception as e:
            logger.warning(f"Failed to load {img_path}: {e}")
            # Return a black image as fallback
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        label = int(row['diagnosis'])
        
        # Apply Albumentations (recommended for medical images)
        if self.albumentations_transform:
            augmented = self.albumentations_transform(image=image)
            image = augmented['image']
        elif self.transform:
            image = self.transform(image)
        else:
            # Default transform
            image = transforms.ToTensor()(image)
        
        return image, torch.tensor(label, dtype=torch.long)
    
    def get_class_distribution(self):
        return self.df['diagnosis'].value_counts().sort_index().to_dict()


def get_train_transforms(image_size: int = 224):
    """Strong augmentation pipeline for training"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.05, 
            scale_limit=0.1, 
            rotate_limit=25, 
            p=0.7
        ),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
        ], p=0.6),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.4),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05, p=0.5),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_val_transforms(image_size: int = 224):
    """Validation transforms (minimal)"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def create_data_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_img_dir: str,
    val_img_dir: str = None,
    test_img_dir: str = None,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation and test data loaders"""
    
    val_img_dir = val_img_dir or train_img_dir
    test_img_dir = test_img_dir or train_img_dir
    
    train_transform = get_train_transforms(image_size)
    val_transform = get_val_transforms(image_size)
    
    train_dataset = DRDataset(
        train_df, train_img_dir, 
        albumentations_transform=train_transform, 
        is_train=True
    )
    
    val_dataset = DRDataset(
        val_df, val_img_dir, 
        albumentations_transform=val_transform, 
        is_train=False
    )
    
    test_dataset = DRDataset(
        test_df, test_img_dir, 
        albumentations_transform=val_transform, 
        is_train=False
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Created loaders - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def load_aptos_csv(csv_path: str, img_dir: str, sample_frac: float = 1.0) -> pd.DataFrame:
    """Load and prepare APTOS CSV dataframe"""
    df = pd.read_csv(csv_path)
    
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)
    
    # Ensure id_code and diagnosis columns exist
    if 'id_code' not in df.columns:
        df = df.rename(columns={df.columns[0]: 'id_code'})
    if 'diagnosis' not in df.columns:
        raise ValueError("CSV must contain 'diagnosis' column")
    
    # Clean image existence check
    existing_files = set(os.listdir(img_dir))
    df['full_path'] = df['id_code'].apply(lambda x: str(x) + '.png')
    df = df[df['full_path'].isin(existing_files)]
    
    logger.info(f"Loaded {len(df)} valid samples from {csv_path}")
    return df


def get_class_weights(df: pd.DataFrame) -> torch.Tensor:
    """Calculate class weights for imbalanced data with smoothing"""
    counts = df['diagnosis'].value_counts().sort_index()
    # Smoothed inverse frequency (square root)
    weights = 1.0 / np.sqrt(counts.values)
    # Normalize weights so they sum to num_classes
    weights = weights * (len(counts) / np.sum(weights))
    weights = torch.tensor(weights, dtype=torch.float32)
    logger.info(f"Smoothed class weights: {weights.tolist()}")
    return weights


def split_data(df: pd.DataFrame, 
               train_ratio: float = 0.7, 
               val_ratio: float = 0.15, 
               test_ratio: float = 0.15,
               stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split of dataset"""
    from sklearn.model_selection import train_test_split
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    if stratify:
        train_df, temp_df = train_test_split(
            df, test_size=(1 - train_ratio), 
            stratify=df['diagnosis'], random_state=42
        )
        
        val_size = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df, test_size=(1 - val_size),
            stratify=temp_df['diagnosis'], random_state=42
        )
    else:
        train_df, temp_df = train_test_split(df, test_size=1-train_ratio, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


# For demo purposes: generate synthetic dataset
def create_demo_dataset(num_samples: int = 200, output_dir: str = None) -> pd.DataFrame:
    """Create a small synthetic dataset for demo purposes"""
    np.random.seed(42)
    
    data = []
    for i in range(num_samples):
        label = np.random.choice([0,1,2,3,4], p=[0.35, 0.18, 0.25, 0.12, 0.10])
        img_id = f"demo_{i:05d}"
        data.append({"id_code": img_id, "diagnosis": label})
    
    df = pd.DataFrame(data)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        # Create fake images for demo
        for row in df.itertuples():
            img_path = os.path.join(output_dir, f"{row.id_code}.png")
            if not os.path.exists(img_path):
                # Create random colored fundus-like image
                img = np.random.randint(30, 200, (224, 224, 3), dtype=np.uint8)
                # Add some fake blood vessel patterns
                for _ in range(8):
                    x = np.random.randint(20, 200)
                    y = np.random.randint(20, 200)
                    cv2.line(img, (x, y), (x + np.random.randint(-50,50), y + np.random.randint(-50,50)), 
                            (100, 30, 30), thickness=np.random.randint(1,3))
                Image.fromarray(img).save(img_path)
    
    return df

def crop_image_from_gray(img, tol=7):
    """Crop black borders of fundus image"""
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0):
            return img
        else:
            img1 = img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2 = img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3 = img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img

def circle_crop(img, sigmaX=30):   
    """Create circular crop around image centre and apply Ben Graham lighting"""    
    img = crop_image_from_gray(img)    
    height, width, depth = img.shape    
    
    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0) , sigmaX) , -4 , 128)
    return img


if __name__ == "__main__":
    # Quick test
    print("DRDataset module loaded successfully.")
    print(f"Classes: {DR_CLASSES}")