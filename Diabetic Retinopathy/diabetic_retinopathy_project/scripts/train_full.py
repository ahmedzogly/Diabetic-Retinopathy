"""
Full training script for production training on real APTOS data.
Usage:
    python scripts/train_full.py --epochs 15 --batch-size 32
"""

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.append(str(Path(__file__).parent.parent))

import torch
import pandas as pd

from src.data_loader import load_aptos_csv, create_data_loaders, get_class_weights, split_data
from src.classifier_model import create_model
from src.train import train_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--img-dir", type=str, default=r"C:\Users\Admin\Downloads\archive (8)\train_images\train_images")
    parser.add_argument("--csv", type=str, default=r"C:\Users\Admin\Downloads\archive (8)\train_1.csv")
    args = parser.parse_args()

    print("🚀 Starting FULL TRAINING for Diabetic Retinopathy")
    
    # Load data
    train_csv = Path(args.csv)
    img_dir = Path(args.img_dir)
    
    if not train_csv.exists():
        print("❌ Real data not found. Please download APTOS first.")
        return

    df = load_aptos_csv(str(train_csv), str(img_dir))
    train_df, val_df, test_df = split_data(df)

    train_loader, val_loader, _ = create_data_loaders(
        train_df, val_df, test_df,
        train_img_dir=str(img_dir),
        batch_size=args.batch_size,
        num_workers=4
    )

    model = create_model()
    class_weights = get_class_weights(train_df)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        class_weights=class_weights,
        save_dir="models"
    )

    print("✅ Full training completed.")
    print(f"Best QWK: {result['best_qwk']:.4f}")

if __name__ == "__main__":
    main()