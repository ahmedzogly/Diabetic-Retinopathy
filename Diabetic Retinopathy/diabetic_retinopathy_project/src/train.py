"""
Full Training Pipeline for Diabetic Retinopathy Classifier
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report,
    cohen_kappa_score,
    roc_auc_score
)
import pandas as pd
import logging
from typing import Dict, Tuple, Optional
import json

from .classifier_model import create_model, save_model
from .data_loader import DR_CLASSES, get_class_weights

logger = logging.getLogger(__name__)


def quadratic_weighted_kappa(y_true, y_pred):
    """Compute Quadratic Weighted Kappa (QWK)"""
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict:
    """Compute comprehensive evaluation metrics"""
    acc = accuracy_score(y_true, y_pred)
    qwk = quadratic_weighted_kappa(y_true, y_pred)
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Per-class sensitivity / specificity
    sensitivities = []
    specificities = []
    
    for i in range(len(DR_CLASSES)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        sensitivities.append(sens)
        specificities.append(spec)
    
    metrics = {
        'accuracy': float(acc),
        'qwk': float(qwk),
        'sensitivity_mean': float(np.mean(sensitivities)),
        'specificity_mean': float(np.mean(specificities)),
        'sensitivities': [float(s) for s in sensitivities],
        'specificities': [float(s) for s in specificities],
        'confusion_matrix': cm.tolist()
    }
    
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
            metrics['auc_macro'] = float(auc)
        except:
            metrics['auc_macro'] = None
    
    return metrics


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none', weight=alpha)
    
    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """Train one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
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
        
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = running_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    
    return avg_loss, acc, np.array(all_preds), np.array(all_labels)


def validate_epoch(model, dataloader, criterion, device):
    """Validate one epoch"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probas = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            probas = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probas, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probas.extend(probas.cpu().numpy())
    
    avg_loss = running_loss / len(dataloader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probas = np.array(all_probas)
    
    metrics = compute_metrics(all_labels, all_preds, all_probas)
    metrics['loss'] = avg_loss
    
    return avg_loss, metrics, all_preds, all_labels, all_probas


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = "cpu",
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    class_weights: Optional[torch.Tensor] = None,
    save_dir: str = "models",
    use_focal_loss: bool = True,
    unfreeze_epoch: int = 3
) -> Dict:
    """Full training loop"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    model = model.to(device)
    
    # Loss function
    if use_focal_loss:
        criterion = FocalLoss(alpha=class_weights.to(device) if class_weights is not None else None, gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    
    # Optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr, 
        weight_decay=weight_decay
    )
    
    # Scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
    
    best_qwk = -1.0
    history = []
    
    patience = 10
    epochs_no_improve = 0
    
    logger.info(f"Starting training for {epochs} epochs on {device}")
    
    for epoch in range(epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*50}")
        
        # Unfreeze backbone after certain epochs
        if epoch == unfreeze_epoch:
            model.unfreeze_backbone(num_layers=None)
            # Update optimizer with new parameters
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()), 
                lr=lr * 0.3, 
                weight_decay=weight_decay
            )
            logger.info("Unfrozen backbone layers")
        
        # Train
        train_loss, train_acc, _, _ = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        
        # Validate
        val_loss, val_metrics, _, _, _ = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step()
        
        # Record history
        epoch_stats = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_metrics['accuracy'],
            'val_qwk': val_metrics['qwk'],
            'val_sensitivity': val_metrics['sensitivity_mean'],
            'val_auc': val_metrics.get('auc_macro')
        }
        history.append(epoch_stats)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"Val QWK: {val_metrics['qwk']:.4f} | Sensitivity: {val_metrics['sensitivity_mean']:.4f}")
        
        # Save best model
        if val_metrics['qwk'] > best_qwk:
            best_qwk = val_metrics['qwk']
            best_path = os.path.join(save_dir, "best_model.pth")
            save_model(model, best_path, epoch=epoch+1, metrics=val_metrics)
            print(f"✅ Saved best model (QWK={best_qwk:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"⚠️ No improvement in QWK for {epochs_no_improve} epoch(s).")
            
        # Save latest checkpoint
        latest_path = os.path.join(save_dir, "latest_checkpoint.pth")
        save_model(model, latest_path, epoch=epoch+1, metrics=val_metrics)
        
        # Early Stopping Check
        if epochs_no_improve >= patience:
            print(f"\n🛑 Early stopping triggered after {epoch+1} epochs due to no improvement.")
            break
    
    # Save training history
    with open(os.path.join(save_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    
    print("\n🎉 Training completed!")
    print(f"Best Validation QWK: {best_qwk:.4f}")
    
    return {
        'history': history,
        'best_qwk': best_qwk,
        'best_model_path': os.path.join(save_dir, "best_model.pth")
    }


if __name__ == "__main__":
    print("Training module ready. Use scripts/train_full.py to run full training.")