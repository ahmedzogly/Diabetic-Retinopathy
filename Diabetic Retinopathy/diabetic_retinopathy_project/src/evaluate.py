"""
Evaluation and Metrics Module for Diabetic Retinopathy
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report,
    roc_auc_score,
    precision_recall_fscore_support
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
import os
from tqdm import tqdm

from .data_loader import DR_CLASSES, CLASS_NAMES_AR


def evaluate_model(model, dataloader, device, return_preds: bool = True):
    """Full evaluation on a dataloader"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probas = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probas = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probas, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probas.extend(probas.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probas = np.array(all_probas)
    
    metrics = compute_detailed_metrics(all_labels, all_preds, all_probas)
    
    if return_preds:
        return metrics, all_preds, all_labels, all_probas
    return metrics


def compute_detailed_metrics(y_true: np.ndarray, 
                            y_pred: np.ndarray, 
                            y_proba: np.ndarray) -> Dict:
    """Compute comprehensive medical-grade metrics"""
    
    acc = accuracy_score(y_true, y_pred)
    
    # Quadratic Weighted Kappa
    from .train import quadratic_weighted_kappa
    qwk = quadratic_weighted_kappa(y_true, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Classification report
    report = classification_report(
        y_true, y_pred, 
        target_names=[DR_CLASSES[i] for i in range(len(DR_CLASSES))],
        output_dict=True,
        zero_division=0
    )
    
    # Per class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # AUC-ROC
    try:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        auc_per_class = roc_auc_score(y_true, y_proba, multi_class="ovr", average=None)
    except Exception:
        auc = None
        auc_per_class = [None] * len(DR_CLASSES)
    
    # Sensitivity and Specificity per class
    sensitivities = []
    specificities = []
    
    n_classes = len(DR_CLASSES)
    for i in range(n_classes):
        tp = cm[i, i]
        fn = sum(cm[i, :]) - tp
        fp = sum(cm[:, i]) - tp
        tn = cm.sum() - (tp + fn + fp)
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        sensitivities.append(sens)
        specificities.append(spec)
    
    metrics = {
        "accuracy": float(acc),
        "qwk": float(qwk),
        "auc_macro": float(auc) if auc else None,
        "auc_per_class": [float(x) if x else None for x in auc_per_class],
        "sensitivity_mean": float(np.mean(sensitivities)),
        "specificity_mean": float(np.mean(specificities)),
        "sensitivities": [float(s) for s in sensitivities],
        "specificities": [float(s) for s in specificities],
        "precision_per_class": [float(p) for p in precision],
        "recall_per_class": [float(r) for r in recall],
        "f1_per_class": [float(f) for f in f1],
        "support_per_class": [int(s) for s in support],
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }
    
    return metrics


def print_metrics_report(metrics: Dict):
    """Pretty print metrics"""
    print("\n" + "="*60)
    print("📊 DIABETIC RETINOPATHY DETECTION - EVALUATION REPORT")
    print("="*60)
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"QWK (Quadratic):   {metrics['qwk']:.4f}")
    if metrics.get('auc_macro'):
        print(f"AUC-ROC (macro):   {metrics['auc_macro']:.4f}")
    print(f"Sensitivity (avg): {metrics['sensitivity_mean']:.4f}")
    print(f"Specificity (avg): {metrics['specificity_mean']:.4f}")
    print("\nPer-class Performance:")
    print("-" * 60)
    
    for i in range(5):
        class_name = DR_CLASSES[i]
        print(f"{class_name:20} | Sens: {metrics['sensitivities'][i]:.3f} | Spec: {metrics['specificities'][i]:.3f} | F1: {metrics['f1_per_class'][i]:.3f}")
    
    print("\nConfusion Matrix:")
    print(np.array(metrics['confusion_matrix']))
    print("="*60 + "\n")


def plot_confusion_matrix(metrics: Dict, save_path: str = None, title: str = "Confusion Matrix"):
    """Plot and optionally save confusion matrix"""
    cm = np.array(metrics["confusion_matrix"])
    class_names = [DR_CLASSES[i] for i in range(5)]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved confusion matrix to {save_path}")
    
    plt.close()


def generate_detailed_report(metrics: Dict, output_path: str):
    """Generate a nice markdown report"""
    report = f"""# Diabetic Retinopathy Model Evaluation Report

## Overall Performance
- **Accuracy**: {metrics['accuracy']:.4f}
- **Quadratic Weighted Kappa (QWK)**: {metrics['qwk']:.4f}
- **AUC-ROC (Macro)**: {metrics.get('auc_macro', 'N/A')}
- **Mean Sensitivity**: {metrics['sensitivity_mean']:.4f}
- **Mean Specificity**: {metrics['specificity_mean']:.4f}

## Per-Class Performance

| Class               | Sensitivity | Specificity | Precision | Recall | F1-Score | Support |
|---------------------|-------------|-------------|-----------|--------|----------|---------|
"""
    
    for i in range(5):
        report += f"| {DR_CLASSES[i]:19} | {metrics['sensitivities'][i]:.3f}       | {metrics['specificities'][i]:.3f}       | {metrics['precision_per_class'][i]:.3f}    | {metrics['recall_per_class'][i]:.3f}  | {metrics['f1_per_class'][i]:.3f}    | {metrics['support_per_class'][i]}     |\n"
    
    report += f"""

## Confusion Matrix
```
{np.array(metrics['confusion_matrix'])}
```

## Clinical Recommendations
- **High Sensitivity is critical**: Model achieves {metrics['sensitivity_mean']*100:.1f}% average sensitivity.
- **Important classes** (Severe + Proliferative): Focus on recall for classes 3 and 4.

"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"✅ Report saved to {output_path}")
    return report


if __name__ == "__main__":
    print("Evaluation module loaded.")