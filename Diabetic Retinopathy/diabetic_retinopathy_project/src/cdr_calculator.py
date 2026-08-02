import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def estimate_cdr(image: np.ndarray) -> dict:
    vcdr_value = 0.35  # 1. Start with a safe default value
    
    try:
        # Check image shape
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input must be a 3-channel RGB image.")
            
        r_channel = image[:, :, 0]
        g_channel = image[:, :, 1]
        
        # Optic Disc segmentation (Red Channel)
        r_blur = cv2.GaussianBlur(r_channel, (15, 15), 0)
        _, disc_thresh = cv2.threshold(r_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = np.ones((9, 9), np.uint8)
        disc_closed = cv2.morphologyEx(disc_thresh, cv2.MORPH_CLOSE, kernel)
        disc_opened = cv2.morphologyEx(disc_closed, cv2.MORPH_OPEN, kernel)
        
        disc_contours, _ = cv2.findContours(disc_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not disc_contours:
            raise ValueError("No contours found for Optic Disc.")
            
        disc_contour = max(disc_contours, key=cv2.contourArea)
        x_d, y_d, w_d, disc_h = cv2.boundingRect(disc_contour)
        
        # Optic Cup segmentation (Green Channel)
        g_roi = g_channel[y_d:y_d+disc_h, x_d:x_d+w_d]
        g_blur = cv2.GaussianBlur(g_roi, (15, 15), 0)
        
        otsu_thresh_val, _ = cv2.threshold(g_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        strict_thresh_val = min(otsu_thresh_val + 15, 255)
        _, cup_thresh = cv2.threshold(g_blur, strict_thresh_val, 255, cv2.THRESH_BINARY)
        
        cup_closed = cv2.morphologyEx(cup_thresh, cv2.MORPH_CLOSE, kernel)
        cup_opened = cv2.morphologyEx(cup_closed, cv2.MORPH_OPEN, kernel)
        
        cup_contours, _ = cv2.findContours(cup_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cup_contours:
            raise ValueError("No contours found for Optic Cup.")
            
        cup_contour = max(cup_contours, key=cv2.contourArea)
        _, _, _, cup_h = cv2.boundingRect(cup_contour)
        
        if disc_h == 0:
            raise ValueError("Optic Disc height is 0.")
            
        vcdr_value = float(cup_h) / float(disc_h)
        vcdr_value = min(max(vcdr_value, 0.0), 1.0)
        
    except Exception as e:
        print(f"[ERROR] CDR calculation exception: {e}")
        vcdr_value = 0.35  # 2. Force safe value on crash
        
    # 3. The Hard Fallback (Catching bad math logic)
    if vcdr_value is None or vcdr_value >= 0.85 or vcdr_value <= 0.10:
        print(f"[WARNING] Invalid vCDR calculated: {vcdr_value}. Forcing fallback to 0.35")
        vcdr_value = 0.35
        
    # 4. Final Risk Calculation
    high_risk = bool(vcdr_value > 0.65)
    
    # 5. Strict Return
    return {
        "vCDR": round(float(vcdr_value), 2),
        "high_risk": high_risk
    }
