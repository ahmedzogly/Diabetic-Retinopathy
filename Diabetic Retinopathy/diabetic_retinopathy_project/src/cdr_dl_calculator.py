import os
import sys
import cv2
import numpy as np
import torch
import albumentations as A

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from albumentations.pytorch import ToTensorV2
from src.segmentation_model import UNet

class DeepCDRCalculator:
    """
    Robust Inference script for Vertical Cup-to-Disc Ratio (vCDR)
    using the trained U-Net.
    """
    def __init__(self, model_path: str = "models/best_unet_glaucoma.pth", device: str = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet(in_channels=3, out_channels=3).to(self.device)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"[+] Loaded U-Net model from {model_path}")
        else:
            print(f"[!] Warning: Model path {model_path} not found. Using untrained weights.")
            
        self.transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
    def _get_bounding_box_height(self, binary_mask: np.ndarray) -> float:
        """Calculates the vertical height of a bounding box for a given binary mask."""
        # Find contours
        contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
            
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return float(h)
        
    def calculate_vcdr(self, image_bgr: np.ndarray) -> tuple:
        """
        Calculates vCDR from a BGR image.
        Returns:
            vcdr (float): The calculated vertical Cup-to-Disc Ratio.
            pred_mask (np.ndarray): The 256x256 predicted mask (0, 1, 2) for visualization.
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Transform
        augmented = self.transform(image=image_rgb)
        input_tensor = augmented['image'].unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(input_tensor)
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            
        # Extract binary masks
        # Disc = Class 1 or Class 2
        disc_mask = (pred_mask >= 1)
        # Cup = Class 2
        cup_mask = (pred_mask == 2)
        
        disc_height = self._get_bounding_box_height(disc_mask)
        cup_height = self._get_bounding_box_height(cup_mask)
        
        # Hard Fallback Mechanism
        if disc_height == 0 or cup_height == 0:
            print("[WARNING] Model failed to detect Disc or Cup. Triggering Hard Fallback.")
            return 0.35, pred_mask
            
        vcdr = cup_height / disc_height
        
        # Sanity bounds check
        if vcdr < 0.1 or vcdr > 1.0:
            print(f"[WARNING] Invalid vCDR calculated: {vcdr:.2f}. Forcing fallback to 0.35")
            return 0.35, pred_mask
            
        return float(vcdr), pred_mask

if __name__ == "__main__":
    # Test Snippet
    print("Initializing DeepCDRCalculator...")
    calculator = DeepCDRCalculator(model_path="dummy.pth")
    dummy_img = np.zeros((512, 512, 3), dtype=np.uint8)
    vcdr, mask = calculator.calculate_vcdr(dummy_img)
    print(f"Calculated vCDR: {vcdr:.2f}")
