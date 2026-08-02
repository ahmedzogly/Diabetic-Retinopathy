"""
Prepare APTOS 2019 data: split + create processed versions
"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
from pathlib import Path
import shutil
from PIL import Image
import numpy as np

from src.data_loader import load_aptos_csv, split_data, create_demo_dataset

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def prepare_real_data():
    print("🔧 Preparing real APTOS dataset...")
    
    train_csv = RAW_DIR / "train.csv"
    train_images = RAW_DIR / "train_images"
    
    if not train_csv.exists() or not train_images.exists():
        print("❌ Real data not found. Run scripts/download_data.py first.")
        print("Using demo dataset instead.")
        return prepare_demo_data()
    
    # Load
    df = load_aptos_csv(str(train_csv), str(train_images), sample_frac=0.95)
    
    # Split
    train_df, val_df, test_df = split_data(df, 0.70, 0.15, 0.15)
    
    # Save CSVs
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)
    
    print(f"✅ Prepared splits:")
    print(f"   Train: {len(train_df)}")
    print(f"   Val:   {len(val_df)}")
    print(f"   Test:  {len(test_df)}")
    
    # Optionally copy images (or keep reference)
    # For simplicity we keep original image paths
    
    return train_df, val_df, test_df


def prepare_demo_data(num_samples: int = 300):
    """Create a demo dataset with synthetic images"""
    print("🧪 Creating synthetic demo dataset...")
    
    demo_img_dir = PROCESSED_DIR / "demo"
    demo_img_dir.mkdir(exist_ok=True)
    
    df = create_demo_dataset(num_samples, str(demo_img_dir))
    
    # Split demo
    train_df, val_df, test_df = split_data(df, 0.7, 0.15, 0.15)
    
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)
    
    print(f"✅ Demo dataset created with {num_samples} synthetic images.")
    print(f"   Saved to: {PROCESSED_DIR}")
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    if (RAW_DIR / "train.csv").exists():
        prepare_real_data()
    else:
        prepare_demo_data(250)
    print("\n✅ Data preparation complete.")