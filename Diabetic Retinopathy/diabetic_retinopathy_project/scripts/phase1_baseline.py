#!/usr/bin/env python3
"""
=============================================================================
PHASE 1: Baseline Pipeline - Diabetic Retinopathy Detection
=============================================================================

Act as a Senior Machine Learning and Computer Vision Engineer.

This script implements **Phase 1: The Baseline Pipeline** for the medical AI project:
"Diabetic Retinopathy Detection from Fundus Images" using the APTOS 2019 dataset.

Ultimate Goal:
    Build a highly accurate, explainable, and production-ready model to grade 
    the severity of Diabetic Retinopathy on a scale of 0 to 4.

Tech Stack (as specified):
    - PyTorch
    - OpenCV (cv2)
    - TIMM (PyTorch Image Models)
    - Albumentations
    - Pandas
    - Scikit-learn

Key Features Implemented:
    1. Advanced Fundus Image Preprocessing:
       - Crop uninformative dark borders
       - Ben Graham preprocessing (Weighted Gaussian Blur)
    2. Custom PyTorch Dataset with albumentations
    3. EfficientNet-B0 via timm with custom classifier head
    4. Training loop with AdamW + CrossEntropyLoss
    5. Primary Metric: Quadratic Weighted Kappa (QWK)

Author: Senior ML/CV Engineer (built with Arena.ai Agent)
Date: 2026-07-31
"""

import os
import sys
import random
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict

import numpy as np
import pandas as pd
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix

# =============================================================================
# CONFIGURATION (Hyperparameters & Paths)
# =============================================================================

class Config:
    """Central configuration for Phase 1 Baseline"""
    
    # === Paths (Modify as needed) ===
    DATA_DIR: str = "data/raw"                    # Where train.csv + train_images/ live
    TRAIN_CSV: str = "train.csv"
    TRAIN_IMAGES_DIR: str = "train_images"
    OUTPUT_DIR: str = "models/phase1_baseline"
    
    # === Data ===
    IMAGE_SIZE: int = 224
    TRAIN_VAL_SPLIT: float = 0.15                 # 85% train / 15% validation
    NUM_CLASSES: int = 5                          # 0-4 severity grades
    
    # === Training ===
    BATCH_SIZE: int = 32
    NUM_EPOCHS: int = 10
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    NUM_WORKERS: int = 4
    
    # === Model ===
    MODEL_NAME: str = "efficientnet_b0"           # Using timm
    PRETRAINED: bool = True
    
    # === Reproducibility ===
    SEED: int = 42
    
    # === Device ===
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # === Advanced Preprocessing ===
    BEN_GRAHAM_SIGMA: int = 10                    # For Ben Graham method
    CROP_THRESHOLD: int = 7                       # For dark border cropping

    # === Class names (for logging) ===
    CLASS_NAMES = {
        0: "No DR",
        1: "Mild",
        2: "Moderate",
        3: "Severe",
        4: "Proliferative DR"
    }


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
def set_seed(seed: int = Config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed()

# Create output directory
Path(Config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# =============================================================================
# 1. IMAGE PREPROCESSING (Crucial for Fundus Images)
# =============================================================================

def crop_dark_borders(image: np.ndarray, threshold: int = Config.CROP_THRESHOLD) -> np.ndarray:
    """
    [EN] Crop uninformative dark borders from fundus images.
    [AR] قص الحواف الداكنة غير المفيدة من صور قاع العين.
    
    Why this is important:
    - Fundus images often have black borders from the camera.
    - These borders add noise and reduce model focus on the retina.
    - Cropping improves signal-to-noise ratio significantly.
    
    Args:
        image: RGB numpy array (H, W, 3)
        threshold: Pixel intensity below which we consider it "dark"
    
    Returns:
        Cropped RGB image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Create binary mask of non-dark pixels
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Find contours of the actual retina area
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return image  # fallback
    
    # Get largest contour (the retina)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Crop with small padding
    padding = 10
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2 * padding)
    h = min(image.shape[0] - y, h + 2 * padding)
    
    cropped = image[y:y+h, x:x+w]
    return cropped


def ben_graham_preprocess(image: np.ndarray, sigmaX: int = Config.BEN_GRAHAM_SIGMA) -> np.ndarray:
    """
    [EN] Apply Ben Graham's preprocessing method (used by many APTOS top solutions).
    [AR] تطبيق طريقة Ben Graham للمعالجة المسبقة (مستخدمة في أفضل الحلول على Kaggle).
    
    Why this preprocessing is critical:
    - Normalizes lighting variations across different cameras and clinics.
    - Enhances blood vessels and lesions (key indicators of DR).
    - The formula is: 4*image - 4*GaussianBlur(image, sigma) + 128
    - Widely adopted in medical imaging competitions for fundus photos.
    
    Reference: https://www.kaggle.com/competitions/aptos2019-blindness-detection/discussion
    
    Args:
        image: RGB numpy array
        sigmaX: Standard deviation for Gaussian kernel
    
    Returns:
        Preprocessed image with enhanced contrast and normalized illumination
    """
    # Convert to float32 for precision
    image = image.astype(np.float32)
    
    # Apply weighted Gaussian blur (Ben Graham technique)
    # 4*original - 4*blurred + 128
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX)
    processed = cv2.addWeighted(image, 4, blurred, -4, 128)
    
    # Clip to valid range and convert back to uint8
    processed = np.clip(processed, 0, 255).astype(np.uint8)
    
    return processed


def preprocess_fundus_image(image: np.ndarray) -> np.ndarray:
    """
    [EN] Full preprocessing pipeline for a single fundus image.
    [AR] خط أنابيب المعالجة الكاملة لصورة قاع العين.
    """
    # Step 1: Crop dark borders
    image = crop_dark_borders(image)
    
    # Step 2: Resize to target size (maintain aspect ratio)
    image = cv2.resize(image, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
    
    # Step 3: Apply Ben Graham preprocessing
    image = ben_graham_preprocess(image)
    
    # Step 4: Convert to RGB if needed (ensure 3 channels)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    return image


# =============================================================================
# 2. CUSTOM PYTORCH DATASET
# =============================================================================

class DiabeticRetinopathyDataset(Dataset):
    """
    [EN] Custom PyTorch Dataset for APTOS 2019 Diabetic Retinopathy.
    [AR] مجموعة بيانات PyTorch مخصصة لتصنيف اعتلال الشبكية السكري.
    
    Handles:
    - Loading images from disk
    - Applying advanced preprocessing (crop + Ben Graham)
    - Albumentations augmentations
    - Returning (image_tensor, label)
    """
    
    def __init__(self, 
                 df: pd.DataFrame, 
                 img_dir: str, 
                 transform: Optional[A.Compose] = None,
                 is_train: bool = True):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.is_train = is_train
        
        # Verify image paths exist (for debugging)
        self.valid_indices = []
        for idx, row in self.df.iterrows():
            img_path = self.img_dir / f"{row['id_code']}.png"
            if img_path.exists():
                self.valid_indices.append(idx)
        
        if len(self.valid_indices) != len(self.df):
            logger.warning(f"Found {len(self.valid_indices)} valid images out of {len(self.df)}")
            self.df = self.df.iloc[self.valid_indices].reset_index(drop=True)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_path = self.img_dir / f"{row['id_code']}.png"
        
        # Load image with OpenCV (faster than PIL for preprocessing)
        image = cv2.imread(str(img_path))
        if image is None:
            # Fallback: create black image
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
            logger.warning(f"Failed to load image: {img_path}")
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # === CRITICAL: Apply medical preprocessing ===
        image = preprocess_fundus_image(image)
        
        # Apply Albumentations transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback transform
            image = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])(image=image)["image"]
        
        label = int(row["diagnosis"])
        return image, torch.tensor(label, dtype=torch.long)


# =============================================================================
# 3. DATA AUGMENTATION (Albumentations)
# =============================================================================

def get_train_transforms() -> A.Compose:
    """
    [EN] Training augmentations using Albumentations.
    [AR] التحسينات المستخدمة أثناء التدريب باستخدام Albumentations.
    
    Why these augmentations:
    - Horizontal/Vertical Flip: Fundus images have no canonical orientation
    - RandomRotate90: Retina is rotationally symmetric
    - ColorJitter / Brightness: Different cameras produce different lighting
    """
    return A.Compose([
        A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=20, p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
        ], p=0.6),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def get_val_transforms() -> A.Compose:
    """
    [EN] Validation/Test transforms (no augmentation).
    [AR] تحويلات التحقق (بدون تحسينات).
    """
    return A.Compose([
        A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


# =============================================================================
# 4. MODEL ARCHITECTURE (using TIMM)
# =============================================================================

def create_efficientnet_model() -> nn.Module:
    """
    [EN] Create baseline model using TIMM's EfficientNet-B0.
    [AR] إنشاء النموذج الأساسي باستخدام EfficientNet-B0 من مكتبة timm.
    
    Why EfficientNet-B0?
    - Excellent accuracy / parameter trade-off (widely used in medical imaging)
    - Strong pretrained weights on ImageNet
    - Efficient for both training and inference
    - Proven on APTOS 2019 leaderboard (many top solutions used EfficientNet variants)
    
    We replace the final classifier with a 5-class head.
    """
    logger.info(f"Loading TIMM model: {Config.MODEL_NAME} (pretrained={Config.PRETRAINED})")
    
    model = timm.create_model(
        Config.MODEL_NAME,
        pretrained=Config.PRETRAINED,
        num_classes=Config.NUM_CLASSES
    )
    
    # Optional: Add dropout for regularization (good practice)
    if hasattr(model, 'classifier'):
        # For EfficientNet in timm, the classifier is usually a Linear layer
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, Config.NUM_CLASSES)
        )
    
    return model


# =============================================================================
# 5. QUADRATIC WEIGHTED KAPPA (PRIMARY METRIC)
# =============================================================================

def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    [EN] Compute Quadratic Weighted Kappa (QWK).
    [AR] حساب مقياس Quadratic Weighted Kappa (المقياس الرئيسي في مسابقة APTOS).
    
    Why QWK is the primary metric:
    - Official metric used in APTOS 2019 Kaggle competition
    - Penalizes larger disagreements more heavily than simple accuracy
    - Accounts for the ordinal nature of DR severity (0-4)
    - Range: -1 to +1 (1.0 = perfect agreement)
    """
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


# =============================================================================
# 6. TRAINING & EVALUATION LOOPS
# =============================================================================

def train_one_epoch(model: nn.Module, 
                    dataloader: DataLoader, 
                    criterion: nn.Module, 
                    optimizer: optim.Optimizer, 
                    device: str,
                    scaler: Optional[GradScaler] = None) -> Tuple[float, float]:
    """Train for one epoch and return average loss + accuracy."""
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        if (batch_idx + 1) % 20 == 0:
            logger.info(f"  Batch [{batch_idx+1}/{len(dataloader)}] Loss: {loss.item():.4f}")
    
    avg_loss = running_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def validate(model: nn.Module, 
             dataloader: DataLoader, 
             criterion: nn.Module, 
             device: str) -> Dict:
    """Run validation and compute all metrics including QWK."""
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = running_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    qwk = quadratic_weighted_kappa(np.array(all_labels), np.array(all_preds))
    
    return {
        "loss": avg_loss,
        "accuracy": acc,
        "qwk": qwk,
        "predictions": np.array(all_preds),
        "labels": np.array(all_labels)
    }


# =============================================================================
# 7. MAIN TRAINING PIPELINE
# =============================================================================

def main():
    logger.info("=" * 70)
    logger.info("PHASE 1: BASELINE PIPELINE - DIABETIC RETINOPATHY DETECTION")
    logger.info("=" * 70)
    logger.info(f"Device: {Config.DEVICE}")
    logger.info(f"Model: {Config.MODEL_NAME}")
    logger.info(f"Image Size: {Config.IMAGE_SIZE}")
    logger.info(f"Batch Size: {Config.BATCH_SIZE}")
    
    # === 1. Load Data ===
    csv_path = Path(Config.DATA_DIR) / Config.TRAIN_CSV
    img_dir = Path(Config.DATA_DIR) / Config.TRAIN_IMAGES_DIR
    
    # Fallback to demo data if real data not present
    if not csv_path.exists() or not img_dir.exists():
        logger.warning("Real APTOS data not found. Using generated demo data...")
        csv_path = Path("data/processed/train.csv")
        img_dir = Path("data/processed/demo")
        
        if not csv_path.exists():
            logger.error("No data found. Please run scripts/prepare_data.py first.")
            return
    
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} samples from {csv_path}")
    
    # === 2. Train / Validation Split (85/15) ===
    train_df, val_df = train_test_split(
        df, 
        test_size=Config.TRAIN_VAL_SPLIT, 
        stratify=df["diagnosis"], 
        random_state=Config.SEED
    )
    logger.info(f"Train samples: {len(train_df)} | Validation samples: {len(val_df)}")
    
    # === 3. Create Datasets & DataLoaders ===
    train_dataset = DiabeticRetinopathyDataset(
        train_df, str(img_dir), transform=get_train_transforms(), is_train=True
    )
    val_dataset = DiabeticRetinopathyDataset(
        val_df, str(img_dir), transform=get_val_transforms(), is_train=False
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    # === 4. Create Model ===
    model = create_efficientnet_model()
    model = model.to(Config.DEVICE)
    
    # === 5. Loss, Optimizer, Scheduler ===
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=Config.LEARNING_RATE, 
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # Optional: mixed precision
    scaler = GradScaler() if Config.DEVICE == "cuda" else None
    
    # === 6. Training Loop ===
    best_qwk = -1.0
    best_model_path = Path(Config.OUTPUT_DIR) / "best_efficientnet_b0_phase1.pth"
    
    logger.info("Starting training...")
    
    for epoch in range(Config.NUM_EPOCHS):
        logger.info(f"\n{'='*50}")
        logger.info(f"EPOCH {epoch + 1}/{Config.NUM_EPOCHS}")
        logger.info(f"{'='*50}")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE, scaler
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, Config.DEVICE)
        
        # Log
        logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        logger.info(f"Val   Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
        logger.info(f"Val   QWK : {val_metrics['qwk']:.4f}   ← PRIMARY METRIC")
        
        # Save best model based on QWK
        if val_metrics['qwk'] > best_qwk:
            best_qwk = val_metrics['qwk']
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "qwk": best_qwk,
                "config": Config.__dict__
            }, best_model_path)
            logger.info(f"✅ New best model saved! QWK = {best_qwk:.4f}")
        
        # Early stopping hint (simple)
        if val_metrics['qwk'] < 0.1 and epoch > 3:
            logger.warning("QWK still very low — consider increasing epochs or unfreezing layers.")
    
    # === 7. Final Evaluation ===
    logger.info("\n" + "="*70)
    logger.info("FINAL EVALUATION ON VALIDATION SET")
    logger.info("="*70)
    
    # Load best model
    checkpoint = torch.load(best_model_path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    final_metrics = validate(model, val_loader, criterion, Config.DEVICE)
    
    logger.info(f"Best Validation QWK: {final_metrics['qwk']:.4f}")
    logger.info(f"Best Validation Accuracy: {final_metrics['accuracy']:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(final_metrics["labels"], final_metrics["predictions"])
    logger.info(f"Confusion Matrix:\n{cm}")
    
    # Save final summary
    summary = {
        "best_qwk": float(final_metrics["qwk"]),
        "best_accuracy": float(final_metrics["accuracy"]),
        "model_path": str(best_model_path),
        "epochs_trained": Config.NUM_EPOCHS
    }
    
    import json
    with open(Path(Config.OUTPUT_DIR) / "phase1_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n✅ Phase 1 Baseline complete!")
    logger.info(f"Best model saved to: {best_model_path}")
    logger.info(f"Results saved to: {Config.OUTPUT_DIR}/phase1_results.json")


if __name__ == "__main__":
    main()