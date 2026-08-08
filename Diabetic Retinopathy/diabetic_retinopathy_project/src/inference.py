"""
Inference utilities and API-ready functions for Diabetic Retinopathy Detection
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
import sys
from typing import Dict, Union, Optional
import logging

from .classifier_model import load_model, create_model
from .data_loader import DR_CLASSES, CLASS_NAMES_AR
from .explain import ModelExplainer
from .cdr_dl_calculator import DeepCDRCalculator
import io
import base64

logger = logging.getLogger(__name__)


class DRInference:
    """Production-ready inference class"""
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        default_model = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pth")
        
        if model_path and os.path.exists(model_path):
            self.model = load_model(model_path, device=self.device)
            logger.info(f"Loaded model from {model_path}")
        elif os.path.exists(default_model):
            self.model = load_model(default_model, device=self.device)
            logger.info(f"Loaded trained demo model: {default_model}")
        else:
            logger.warning("No model path provided. Using pretrained EfficientNetB3 (random weights)")
            self.model = create_model(pretrained=True)
            self.model.to(self.device)
            self.model.eval()
        
        self.explainer = ModelExplainer(self.model, device=self.device)
        
        # Load Deep CDR Calculator (U-Net)
        unet_path = os.path.join(os.path.dirname(__file__), "..", "models", "best_unet_glaucoma.pth")
        self.cdr_calculator = DeepCDRCalculator(model_path=unet_path, device=self.device)
        
        # ImageNet normalization
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
    
    def preprocess_image(self, image: Union[Image.Image, np.ndarray, str]) -> torch.Tensor:
        """Preprocess image for model input"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
            
        from .data_loader import circle_crop
        img_np = np.array(image)
        try:
            bg_np = circle_crop(img_np)
        except Exception:
            bg_np = img_np
        image = Image.fromarray(bg_np)
        
        # Resize and normalize
        image = image.resize((224, 224))
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Normalize
        img_array = (img_array - self.mean) / self.std
        
        # To tensor
        tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float()
        return tensor.to(self.device)
    
    def predict(self, image: Union[Image.Image, np.ndarray, str]) -> Dict:
        """Basic prediction"""
        input_tensor = self.preprocess_image(image)
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
            predicted_class = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_class])
        
        # Calculate Glaucoma Risk (CDR) using U-Net
        if isinstance(image, str):
            pil_image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image).convert('RGB')
        else:
            pil_image = image
            
        import cv2
        img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
        vcdr, pred_mask = self.cdr_calculator.calculate_vcdr(img_bgr)
        
        # Colorize the mask: 0=Black, 1=Blue (Disc), 2=Green (Cup)
        colored_mask = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
        colored_mask[pred_mask == 1] = [0, 0, 255] # Blue for Disc Rim
        colored_mask[pred_mask == 2] = [0, 255, 0] # Green for Cup
        
        # Convert to Base64
        mask_pil = Image.fromarray(colored_mask)
        buffer = io.BytesIO()
        mask_pil.save(buffer, format="PNG")
        cdr_mask_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        is_high_risk = vcdr > 0.65
        
        result = {
            "predicted_class": predicted_class,
            "predicted_label": DR_CLASSES[predicted_class],
            "predicted_label_ar": CLASS_NAMES_AR[predicted_class],
            "confidence": confidence,
            "probabilities": {
                str(i): float(probabilities[i]) for i in range(5)
            },
            "class_names": DR_CLASSES,
            "class_names_ar": CLASS_NAMES_AR,
            "cdr_value": vcdr,
            "cdr_mask_base64": cdr_mask_base64,
            "glaucoma_risk": "High Risk" if is_high_risk else "Normal",
            "glaucoma_risk_ar": "عالي الخطورة" if is_high_risk else "طبيعي"
        }
        
        # Add severity level
        if predicted_class == 0:
            result["severity"] = "Normal"
            result["severity_ar"] = "طبيعي"
            result["recommendation"] = "No signs of diabetic retinopathy. Routine screening recommended."
            result["recommendation_ar"] = "لا توجد علامات اعتلال الشبكية السكري. يُنصح بالفحص الروتيني."
        elif predicted_class == 1:
            result["severity"] = "Mild"
            result["severity_ar"] = "خفيف"
            result["recommendation"] = "Mild DR detected. Annual follow-up recommended."
            result["recommendation_ar"] = "اعتلال خفيف. يُنصح بالمتابعة السنوية."
        elif predicted_class == 2:
            result["severity"] = "Moderate"
            result["severity_ar"] = "متوسط"
            result["recommendation"] = "Moderate DR. Refer to ophthalmologist within 3 months."
            result["recommendation_ar"] = "اعتلال متوسط. يُنصح بالإحالة لطبيب العيون خلال 3 أشهر."
        elif predicted_class == 3:
            result["severity"] = "Severe"
            result["severity_ar"] = "شديد"
            result["recommendation"] = "Severe DR. Urgent referral to specialist required."
            result["recommendation_ar"] = "اعتلال شديد. يجب الإحالة العاجلة للمتخصص."
        else:
            result["severity"] = "Proliferative"
            result["severity_ar"] = "تكاثري"
            result["recommendation"] = "Proliferative DR. Immediate ophthalmology referral and treatment."
            result["recommendation_ar"] = "اعتلال تكاثري. يجب الإحالة الفورية والعلاج."
        
        return result
    
    def predict_with_explanation(self, 
                                  image: Union[Image.Image, np.ndarray, str],
                                  save_dir: Optional[str] = None) -> Dict:
        """Prediction + Grad-CAM explanation"""
        base_result = self.predict(image)
        
        # Convert input for explainer
        if isinstance(image, str):
            pil_image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image).convert('RGB')
        else:
            pil_image = image
        # Generate explanation
        explanation = self.explainer.explain_image(
            pil_image, 
            target_class=base_result['predicted_class'],
            save_path=os.path.join(save_dir, "explanation.png") if save_dir else None
        )
        
        # Combine results
        full_result = base_result.copy()
        full_result.update({
            "gradcam_available": True,
            "gradcam_overlay": explanation.get("gradcam_overlay"),
            "gradcam_map": explanation.get("gradcam_map"),
        })
        
        # Add interpretation text
        full_result["interpretation"] = self._generate_interpretation(base_result)
        
        return full_result
    
    def _generate_interpretation(self, result: Dict) -> Dict:
        """Generate simple Arabic + English interpretation"""
        class_idx = result["predicted_class"]
        
        interpretations = {
            0: {
                "en": "The image shows a healthy retina with no visible signs of diabetic retinopathy.",
                "ar": "تظهر الصورة شبكية صحية بدون أي علامات واضحة لاعتلال الشبكية السكري."
            },
            1: {
                "en": "Mild microaneurysms detected. Early signs of diabetic retinopathy.",
                "ar": "تم الكشف عن تمددات دقيقة خفيفة. علامات مبكرة لاعتلال الشبكية."
            },
            2: {
                "en": "Moderate hemorrhages and exudates visible. Needs close monitoring.",
                "ar": "نزيف وإفرازات متوسطة واضحة. يحتاج مراقبة دقيقة."
            },
            3: {
                "en": "Severe signs: extensive hemorrhages, cotton wool spots, venous beading.",
                "ar": "علامات شديدة: نزيف واسع، بقع قطنية، وتورم الأوردة."
            },
            4: {
                "en": "Proliferative DR: new abnormal blood vessels. High risk of vision loss.",
                "ar": "اعتلال تكاثري: أوعية دموية جديدة غير طبيعية. خطر عالي لفقدان البصر."
            }
        }
        
        return interpretations.get(class_idx, {"en": "", "ar": ""})
    
    def batch_predict(self, image_paths: list) -> list:
        """Batch prediction for multiple images"""
        results = []
        for path in image_paths:
            try:
                result = self.predict(path)
                result["image_path"] = path
                results.append(result)
            except Exception as e:
                results.append({
                    "image_path": path,
                    "error": str(e)
                })
        return results


# Convenience functions for API
def get_inference_engine(model_path: Optional[str] = None) -> DRInference:
    """Singleton-like inference engine"""
    return DRInference(model_path=model_path)


def run_inference(image_input, model_path: Optional[str] = None, explain: bool = False):
    """Quick function for API usage"""
    engine = get_inference_engine(model_path)
    
    if explain:
        return engine.predict_with_explanation(image_input)
    else:
        return engine.predict(image_input)


if __name__ == "__main__":
    print("DRInference ready for production.")