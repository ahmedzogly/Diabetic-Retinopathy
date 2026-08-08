import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
import numpy as np

from src.dataset_unet import REFUGEDataset
from src.segmentation_model import UNet

def calculate_metrics(preds, targets, num_classes=3):
    """
    Calculate IoU and Dice for Disc and Cup.
    Class 0: Background
    Class 1: Disc Rim
    Class 2: Cup
    Disc = Class 1 + Class 2
    Cup = Class 2
    """
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()
    
    # Calculate for Disc (Mask >= 1)
    disc_pred = (preds >= 1).astype(np.uint8)
    disc_target = (targets >= 1).astype(np.uint8)
    
    intersection_disc = np.sum(disc_pred * disc_target)
    union_disc = np.sum(disc_pred) + np.sum(disc_target) - intersection_disc
    
    dice_disc = (2.0 * intersection_disc) / (np.sum(disc_pred) + np.sum(disc_target) + 1e-8)
    iou_disc = intersection_disc / (union_disc + 1e-8)
    
    # Calculate for Cup (Mask == 2)
    cup_pred = (preds == 2).astype(np.uint8)
    cup_target = (targets == 2).astype(np.uint8)
    
    intersection_cup = np.sum(cup_pred * cup_target)
    union_cup = np.sum(cup_pred) + np.sum(cup_target) - intersection_cup
    
    dice_cup = (2.0 * intersection_cup) / (np.sum(cup_pred) + np.sum(cup_target) + 1e-8)
    iou_cup = intersection_cup / (union_cup + 1e-8)
    
    return {
        "dice_disc": dice_disc, "iou_disc": iou_disc,
        "dice_cup": dice_cup, "iou_cup": iou_cup,
        "dice_mean": (dice_disc + dice_cup) / 2.0
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r"C:\Users\Admin\Desktop\Diabetic Retinopathy\Diabetic Retinopathy\data\REFUGE2")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test-run", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting U-Net Training on {device}")
    
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    
    if not os.path.exists(train_dir):
        print(f"❌ Training directory not found: {train_dir}")
        return

    train_dataset = REFUGEDataset(train_dir, is_train=True)
    val_dataset = REFUGEDataset(val_dir, is_train=False)
    
    if args.test_run:
        # subset for testing
        train_dataset.image_paths = train_dataset.image_paths[:10]
        val_dataset.image_paths = val_dataset.image_paths[:10]
        args.epochs = 1
        print("🔧 TEST RUN ENABLED: Reduced dataset and epochs.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = UNet(in_channels=3, out_channels=3).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_dice = 0.0
    patience_counter = 0
    patience_limit = 10
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        print(f"\n{'='*30}\nEpoch {epoch+1}/{args.epochs}\n{'='*30}")
        
        # Training Phase
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc="Training"):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        scheduler.step()
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_metrics = {"dice_disc": 0, "iou_disc": 0, "dice_cup": 0, "iou_cup": 0, "dice_mean": 0}
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validation"):
                images, masks = images.to(device), masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                
                batch_metrics = calculate_metrics(preds, masks)
                for k in val_metrics.keys():
                    val_metrics[k] += batch_metrics[k]
                    
        # Average metrics
        val_loss /= max(1, len(val_loader))
        train_loss /= max(1, len(train_loader))
        for k in val_metrics.keys():
            val_metrics[k] /= max(1, len(val_loader))
            
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Disc Dice: {val_metrics['dice_disc']:.4f} | Val Cup Dice: {val_metrics['dice_cup']:.4f}")
        print(f"Val Mean Dice: {val_metrics['dice_mean']:.4f}")
        
        # Checkpoint
        if val_metrics['dice_mean'] > best_dice:
            best_dice = val_metrics['dice_mean']
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "best_unet_glaucoma.pth"))
            print(f"🌟 New Best Mean Dice: {best_dice:.4f}! Model saved.")
        else:
            patience_counter += 1
            print(f"⚠️ No improvement. Patience: {patience_counter}/{patience_limit}")
            if patience_counter >= patience_limit:
                print("🛑 Early stopping triggered.")
                break

    print("\n✅ Training Complete!")
    print(f"🏆 Best Validation Mean Dice: {best_dice:.4f}")

if __name__ == "__main__":
    main()
