"""
FastAPI Web Application for Diabetic Retinopathy Detection
Full production-ready backend with explainability
"""

import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import torch
from PIL import Image
import io
import os
import json
import uuid
from datetime import datetime
from typing import Optional
import numpy as np
import base64

import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import DRInference
from src.data_loader import DR_CLASSES, CLASS_NAMES_AR

# Create FastAPI app
app = FastAPI(
    title="Diabetic Retinopathy Detection API",
    description="AI-powered detection of Diabetic Retinopathy from Fundus Images. Egyptian-focused medical AI.",
    version="1.0.0"
)

# Setup static and templates
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Global inference engine (lazy loaded)
inference_engine: Optional[DRInference] = None

# Model path (change this to your trained model)
MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), "models", "best_model.pth")

# Create uploads directory
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def get_inference_engine():
    """Lazy load the inference engine"""
    global inference_engine
    if inference_engine is None:
        print("[+] Loading inference engine...")
        try:
            inference_engine = DRInference(model_path=MODEL_PATH if os.path.exists(MODEL_PATH) else None)
            print("[+] Inference engine loaded.")
        except Exception as e:
            print(f"[!] Using fallback pretrained model: {e}")
            inference_engine = DRInference(model_path=None)
    return inference_engine


# Pydantic models
class PredictionResponse(BaseModel):
    predicted_class: int
    predicted_label: str
    predicted_label_ar: str
    confidence: float
    probabilities: dict
    severity: str
    severity_ar: str
    recommendation: str
    recommendation_ar: str
    interpretation: Optional[dict] = None
    gradcam_available: bool = False
    cdr_value: Optional[float] = None
    cdr_mask_base64: Optional[str] = None
    glaucoma_risk: Optional[str] = None
    glaucoma_risk_ar: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    timestamp: str


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main web UI"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "كشف اعتلال الشبكية السكري | Diabetic Retinopathy AI"
    })


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    engine = get_inference_engine()
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        device=engine.device,
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/model_info")
async def model_info():
    """Return model metadata"""
    engine = get_inference_engine()
    return {
        "model": "EfficientNet-B3 (Transfer Learning)",
        "num_classes": 5,
        "classes": DR_CLASSES,
        "classes_ar": CLASS_NAMES_AR,
        "device": engine.device,
        "input_size": "224x224",
        "version": "1.0.0"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Basic prediction endpoint"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Run inference
        engine = get_inference_engine()
        result = engine.predict(image)
        
        # Return clean response
        return PredictionResponse(
            predicted_class=result["predicted_class"],
            predicted_label=result["predicted_label"],
            predicted_label_ar=result["predicted_label_ar"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            severity=result["severity"],
            severity_ar=result["severity_ar"],
            recommendation=result["recommendation"],
            recommendation_ar=result["recommendation_ar"],
            cdr_value=result.get("cdr_value"),
            cdr_mask_base64=result.get("cdr_mask_base64"),
            glaucoma_risk=result.get("glaucoma_risk"),
            glaucoma_risk_ar=result.get("glaucoma_risk_ar")
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict_with_explain")
async def predict_with_explain(file: UploadFile = File(...)):
    """Prediction + Grad-CAM explanation"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        engine = get_inference_engine()
        
        # Generate unique filename for saving
        filename = f"{uuid.uuid4().hex[:8]}.png"
        save_dir = os.path.join(UPLOAD_DIR, "explanations")
        os.makedirs(save_dir, exist_ok=True)
        
        result = engine.predict_with_explanation(image, save_dir=save_dir)
        
        # Convert Grad-CAM overlay to base64 for frontend
        gradcam_base64 = None
        if result.get("gradcam_overlay") is not None:
            overlay_img = Image.fromarray(result["gradcam_overlay"])
            buffer = io.BytesIO()
            overlay_img.save(buffer, format="PNG")
            gradcam_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        response = {
            **result,
            "gradcam_base64": gradcam_base64,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Clean up large arrays for JSON
        if "gradcam_overlay" in response:
            del response["gradcam_overlay"]
        if "gradcam_map" in response:
            del response["gradcam_map"]
        
        return JSONResponse(content=response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")


@app.post("/predict_batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """Batch prediction (max 5 images)"""
    if len(files) > 5:
        raise HTTPException(400, "Maximum 5 images allowed per batch")
    
    engine = get_inference_engine()
    results = []
    
    for file in files:
        if not file.content_type.startswith("image/"):
            continue
        
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            result = engine.predict(image)
            result["filename"] = file.filename
            results.append(result)
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})
    
    return {"results": results, "count": len(results)}


@app.get("/demo")
async def demo_page(request: Request):
    """Demo page with pre-loaded sample images"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "Demo - Diabetic Retinopathy AI",
        "demo_mode": True
    })


if __name__ == "__main__":
    print("🚀 Starting Diabetic Retinopathy Detection Web App")
    print("Visit: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)