"""
Model Explainability: Grad-CAM and SHAP for Diabetic Retinopathy
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from typing import Tuple, Optional, List
import os
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import shap
import logging

from .classifier_model import DRClassifier

logger = logging.getLogger(__name__)

class ModelExplainer:
    """Unified Explainer for DR model using Grad-CAM and SHAP"""
    
    def __init__(self, model: DRClassifier, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
        # Grad-CAM setup
        # Target the last convolutional layer of EfficientNet
        target_layers = [self.model.backbone.features[-1]]
        self.grad_cam = GradCAM(
            model=self.model,
            target_layers=target_layers
        )
    
    def generate_gradcam(self, 
                         image_tensor: torch.Tensor, 
                         target_class: Optional[int] = None,
                         image_size: Tuple[int, int] = (224, 224)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Grad-CAM heatmap for a given image.
        
        Args:
            image_tensor: Normalized image tensor [1, 3, H, W]
            target_class: Specific class to explain (None = predicted)
            image_size: Original image size
            
        Returns:
            heatmap: Grad-CAM heatmap (numpy)
            overlay: Original image with heatmap overlay
        """
        image_tensor = image_tensor.to(self.device)
        
        # Get prediction if target_class not specified
        with torch.no_grad():
            outputs = self.model(image_tensor)
            if target_class is None:
                target_class = torch.argmax(outputs, dim=1).item()
        
        targets = [ClassifierOutputTarget(target_class)]
        
        # Generate CAM
        grayscale_cam = self.grad_cam(input_tensor=image_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]  # (H, W)
        
        # Convert tensor to numpy image
        img_np = image_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
        
        # Denormalize (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)
        
        # Resize cam to match image
        cam_resized = cv2.resize(grayscale_cam, image_size)
        
        # Create overlay
        visualization = show_cam_on_image(img_np, cam_resized, use_rgb=True)
        
        return cam_resized, visualization
    
    def generate_shap(self, 
                      image_tensor: torch.Tensor, 
                      background_samples: int = 10,
                      n_samples: int = 50) -> np.ndarray:
        """
        Generate SHAP values for the image.
        Note: SHAP for CNNs is expensive. This is a simplified version.
        """
        image_tensor = image_tensor.to(self.device)
        
        # Create a small background set from random noise (demo)
        background = torch.randn(background_samples, 3, 224, 224).to(self.device)
        
        # Use DeepExplainer
        e = shap.DeepExplainer(self.model, background)
        
        # Compute SHAP values
        shap_values = e.shap_values(image_tensor[:1])
        
        # shap_values shape: (1, num_classes, 3, H, W)
        # Return the absolute mean across channels for the predicted class
        return np.abs(shap_values[0]).mean(axis=1)  # (num_classes, H, W)
    
    def explain_image(self, 
                      image: Image.Image, 
                      target_class: Optional[int] = None,
                      save_path: Optional[str] = None) -> dict:
        """
        Full explanation pipeline.
        
        Returns:
            dict with keys: 'prediction', 'probabilities', 'gradcam', 'overlay'
        """
        # Preprocess
        from torchvision import transforms as T
        from .data_loader import circle_crop
        
        img_np = np.array(image)
        try:
            bg_np = circle_crop(img_np)
        except Exception:
            bg_np = img_np
            
        bg_img = Image.fromarray(bg_np)
        
        preprocess = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = preprocess(bg_img).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(image_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()
        
        # Generate GradCAM
        gradcam_map, overlay = self.generate_gradcam(
            image_tensor, 
            target_class=target_class or pred_class
        )
        
        # Create explanation dict
        explanation = {
            'predicted_class': int(pred_class),
            'confidence': float(confidence),
            'probabilities': probs[0].cpu().numpy().tolist(),
            'gradcam_map': gradcam_map,
            'gradcam_overlay': overlay,
            'target_class': target_class
        }
        
        # Save visualizations if requested
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save overlay
            overlay_pil = Image.fromarray(overlay)
            overlay_pil.save(save_path.replace('.png', '_gradcam.png'))
            
            # Save heatmap alone
            plt.figure(figsize=(6, 5))
            plt.imshow(gradcam_map, cmap='jet')
            plt.colorbar()
            plt.axis('off')
            plt.title(f"Grad-CAM: Class {pred_class}")
            plt.savefig(save_path.replace('.png', '_heatmap.png'), bbox_inches='tight', dpi=150)
            plt.close()
            
            explanation['saved_overlay_path'] = save_path.replace('.png', '_gradcam.png')
            explanation['saved_heatmap_path'] = save_path.replace('.png', '_heatmap.png')
        
        return explanation
    
    def create_explanation_report(self, 
                                  image: Image.Image, 
                                  explanation: dict,
                                  class_names: dict,
                                  save_path: str = None):
        """Create a combined explanation figure"""
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Grad-CAM overlay
        axes[1].imshow(explanation['gradcam_overlay'])
        pred_name = class_names.get(explanation['predicted_class'], "Unknown")
        axes[1].set_title(f"Grad-CAM Overlay\nPredicted: {pred_name} ({explanation['confidence']:.2%})")
        axes[1].axis('off')
        
        # Heatmap
        im = axes[2].imshow(explanation['gradcam_map'], cmap='jet')
        axes[2].set_title("Grad-CAM Heatmap")
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"✅ Full explanation saved to {save_path}")
        
        plt.close()
        return fig


def generate_interpretability_visuals(model_path: str, 
                                      sample_images: List[str], 
                                      output_dir: str = "explanations"):
    """Utility to batch generate explanations for sample images"""
    os.makedirs(output_dir, exist_ok=True)
    
    from .classifier_model import load_model
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_path, device=device)
    
    explainer = ModelExplainer(model, device)
    
    results = []
    for img_path in sample_images:
        try:
            image = Image.open(img_path).convert("RGB")
            explanation = explainer.explain_image(image, save_path=os.path.join(output_dir, os.path.basename(img_path)))
            results.append({
                'image': img_path,
                'prediction': explanation['predicted_class'],
                'confidence': explanation['confidence']
            })
        except Exception as e:
            logger.error(f"Failed to explain {img_path}: {e}")
    
    return results


if __name__ == "__main__":
    print("Explainability module ready (Grad-CAM + SHAP support).")