import argparse
import sys
import os
import json
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import pandas as pd
import timm

from src.data_loader import load_aptos_csv, create_data_loaders, get_class_weights, split_data
from src.train import train_model

MODELS_TO_TEST = [
    "efficientnet_b3",
    "resnet50",
    "densenet121",
    "convnext_tiny",
    "mobilenetv3_large_100",
    "inception_v3",
    "efficientnetv2_rw_s"
]

class TimmWrapper(nn.Module):
    def __init__(self, model_name, num_classes=5, pretrained=True):
        super().__init__()
        # Inception v3 in timm supports aux_logits=False by default for 224x224
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        
        # Freeze backbone
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Unfreeze classifier head only
        head = self.model.get_classifier()
        if head is not None:
            for param in head.parameters():
                param.requires_grad = True
                
    def forward(self, x):
        return self.model(x)
        
    def unfreeze_backbone(self, num_layers=None):
        for param in self.model.parameters():
            param.requires_grad = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--img-dir", type=str, default=r"C:\Users\Admin\Downloads\archive (8)\train_images\train_images")
    parser.add_argument("--csv", type=str, default=r"C:\Users\Admin\Downloads\archive (8)\train_1.csv")
    args = parser.parse_args()

    print("🚀 Starting FULL BENCHMARK for Diabetic Retinopathy")
    
    train_csv = Path(args.csv)
    img_dir = Path(args.img_dir)
    
    if not train_csv.exists():
        print("❌ Real data not found. Please verify the dataset path.")
        return

    # 1. Load Data (same robust logic as before)
    df = load_aptos_csv(str(train_csv), str(img_dir))
    train_df, val_df, test_df = split_data(df)

    train_loader, val_loader, _ = create_data_loaders(
        train_df, val_df, test_df,
        train_img_dir=str(img_dir),
        batch_size=args.batch_size,
        num_workers=4
    )

    class_weights = get_class_weights(train_df)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Store benchmark results
    results = {}

    for model_name in MODELS_TO_TEST:
        print(f"\n{'='*50}")
        print(f"🔬 Benchmarking Architecture: {model_name}")
        print(f"{'='*50}")
        
        try:
            model = TimmWrapper(model_name=model_name, num_classes=5, pretrained=True)
            
            save_dir = f"models/benchmark_{model_name}"
            
            result = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                class_weights=class_weights,
                save_dir=save_dir
            )
            
            # Extract final metrics from history
            best_epoch = next(epoch for epoch in result['history'] if epoch['val_qwk'] == result['best_qwk'])
            
            results[model_name] = {
                "best_qwk": result['best_qwk'],
                "accuracy": best_epoch['val_acc'],
                "sensitivity": best_epoch['val_sensitivity']
            }
            
            print(f"✅ {model_name} finished! Best QWK: {result['best_qwk']:.4f}")
            
        except Exception as e:
            print(f"❌ Failed to train {model_name}: {e}")
            results[model_name] = {
                "best_qwk": 0.0,
                "accuracy": 0.0,
                "sensitivity": 0.0,
                "error": str(e)
            }

    # Generate JSON
    with open("models/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    # Generate Markdown Report
    report = "# 🔬 Architecture Benchmarking Report\n\n"
    report += "| Architecture | Best QWK | Accuracy | Sensitivity |\n"
    report += "|--------------|----------|----------|-------------|\n"
    for model_name, metrics in results.items():
        if "error" in metrics:
            report += f"| {model_name} | ERROR | ERROR | ERROR |\n"
        else:
            report += f"| {model_name} | {metrics['best_qwk']:.4f} | {metrics['accuracy']:.4f} | {metrics['sensitivity']:.4f} |\n"
            
    with open("models/benchmark_report.md", "w") as f:
        f.write(report)
        
    print("\n🎉 Benchmarking complete! Report saved to models/benchmark_report.md")

if __name__ == "__main__":
    main()
