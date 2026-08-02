"""
EfficientNetB0-based model for Diabetic Retinopathy Detection
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class DRClassifier(nn.Module):
    """Diabetic Retinopathy Classifier using EfficientNet-B0"""
    
    def __init__(self, 
                 num_classes: int = 5, 
                 pretrained: bool = True,
                 dropout: float = 0.3,
                 freeze_backbone: bool = True):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Load EfficientNet-B3 backbone
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b3(weights=weights)
        
        # Get the number of input features for the classifier
        in_features = self.backbone.classifier[1].in_features
        
        # Replace classifier with custom head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout * 0.7),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(256, num_classes)
        )
        
        # Freeze backbone initially for transfer learning
        if freeze_backbone and pretrained:
            for param in self.backbone.features.parameters():
                param.requires_grad = False
            logger.info("Backbone frozen for transfer learning")
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize new classifier layers"""
        for m in self.backbone.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def unfreeze_backbone(self, num_layers: Optional[int] = None):
        """Unfreeze backbone layers gradually"""
        if num_layers is None:
            # Unfreeze all
            for param in self.backbone.features.parameters():
                param.requires_grad = True
            logger.info("All backbone layers unfrozen")
        else:
            # Unfreeze last N layers
            total_layers = len(list(self.backbone.features.children()))
            unfreeze_from = max(0, total_layers - num_layers)
            
            for i, child in enumerate(self.backbone.features.children()):
                if i >= unfreeze_from:
                    for param in child.parameters():
                        param.requires_grad = True
            logger.info(f"Unfrozen last {num_layers} backbone layers")
    
    def get_feature_extractor(self):
        """Return feature extractor (backbone without classifier)"""
        return nn.Sequential(*list(self.backbone.children())[:-1])


def create_model(num_classes: int = 5, 
                 pretrained: bool = True,
                 dropout: float = 0.35) -> DRClassifier:
    """Factory function to create DR model"""
    model = DRClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        freeze_backbone=True
    )
    return model


def load_model(model_path: str, device: str = 'cpu', num_classes: int = 5) -> DRClassifier:
    """Load trained model from checkpoint"""
    model = create_model(num_classes=num_classes, pretrained=False)
    
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    logger.info(f"Model loaded from {model_path}")
    return model


def save_model(model: nn.Module, path: str, epoch: int = None, metrics: dict = None):
    """Save model with optional metadata"""
    save_dict = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'metrics': metrics
    }
    torch.save(save_dict, path)
    logger.info(f"Model saved to {path}")


# For inference on single image
def get_model_for_inference(model_path: Optional[str] = None, device: str = 'cpu'):
    """Quickly get ready-to-use inference model"""
    model = create_model(pretrained=True, freeze_backbone=False)
    
    if model_path and os.path.exists(model_path):
        try:
            model = load_model(model_path, device=device)
        except Exception as e:
            logger.warning(f"Could not load {model_path}. Using pretrained only: {e}")
    
    model.eval()
    model.to(device)
    return model


if __name__ == "__main__":
    import os
    print("Creating EfficientNet-B3 DR Classifier...")
    model = create_model()
    print(f"Model created. Total params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Output shape: {out.shape}")  # Should be [2, 5]