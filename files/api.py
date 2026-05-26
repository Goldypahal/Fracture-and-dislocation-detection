"""
FastAPI REST API for Bone Fracture & Dislocation Prediction (Ensemble Version)
=============================================================================
Endpoints:
  POST /predict          - full ensemble prediction with heatmap & uncertainty metrics
  POST /predict/batch    - batch prediction (no heatmap)
  GET  /health           - health check & ensemble status

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
"""

import io
import os
import base64
import numpy as np
import cv2
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict

from model import FractureDislocModel, TemperatureScaler, GradCAMPP, load_compatible_model
from dataset import load_xray, get_transforms, CLINICAL_COLS

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────
app = FastAPI(
    title="Bone Fracture & Dislocation Detection API (Ensemble)",
    description="Multi-modal AI: 3-Fold X-ray + patient clinical info Ensemble. Returns prediction, uncertainty, and Grad-CAM++ overlays.",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────
# 3-FOLD ENSEMBLE MODEL CLASS
# ─────────────────────────────────────────────
class EnsembleTemperatureScaler(nn.Module):
    """
    Ensemble Wrapper that feeds inputs into multiple TemperatureScaler models,
    averages their output probabilities (ensemble mean), and calculates 
    prediction variance/standard deviation as an estimator of clinical uncertainty.
    """
    def __init__(self, models: List[TemperatureScaler]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, image: torch.Tensor, clinical: torch.Tensor) -> Dict[str, torch.Tensor]:
        frac_probs, disl_probs = [], []
        frac_logits, disl_logits = [], []

        for m in self.models:
            out = m(image, clinical)
            frac_probs.append(out["fracture_prob"])
            disl_probs.append(out["dislocation_prob"])
            frac_logits.append(out["fracture_logit"])
            disl_logits.append(out["dislocation_logit"])

        # Stack across model dimension: (num_models, batch_size)
        frac_probs = torch.stack(frac_probs, dim=0)
        disl_probs = torch.stack(disl_probs, dim=0)
        frac_logits = torch.stack(frac_logits, dim=0)
        disl_logits = torch.stack(disl_logits, dim=0)

        # Compute mean and standard deviation
        mean_frac = frac_probs.mean(dim=0)
        mean_disl = disl_probs.mean(dim=0)
        
        # Standard deviation provides uncertainty estimation (agreement score)
        std_frac = frac_probs.std(dim=0) if len(self.models) > 1 else torch.zeros_like(mean_frac)
        std_disl = disl_probs.std(dim=0) if len(self.models) > 1 else torch.zeros_like(mean_disl)

        return {
            "fracture_prob": mean_frac,
            "dislocation_prob": mean_disl,
            "fracture_logit": frac_logits.mean(dim=0),
            "dislocation_logit": disl_logits.mean(dim=0),
            "fracture_std": std_frac,
            "dislocation_std": std_disl,
        }

# ─────────────────────────────────────────────
# MODEL LOADING  (on startup)
# ─────────────────────────────────────────────
model: Optional[EnsembleTemperatureScaler] = None
gradcam: Optional[GradCAMPP]               = None
loaded_ensemble_paths: List[str]            = []

@app.on_event("startup")
def load_ensemble():
    global model, gradcam, loaded_ensemble_paths
    loaded_models = []
    first_base_model = None
    loaded_ensemble_paths = []
    
    print(f"Initializing models on device: {DEVICE}...")
    
    # Load up to 3 folds (calibrated or raw)
    for i in range(1, 4):
        calibrated_path = f"checkpoints/best_fold{i}_calibrated.pt"
        raw_path = f"checkpoints/best_fold{i}.pt"
        
        target_path = None
        if os.path.exists(calibrated_path):
            target_path = calibrated_path
        elif os.path.exists(raw_path):
            target_path = raw_path
            
        if target_path:
            try:
                base_model, scaler = load_compatible_model(target_path, device=DEVICE)
                loaded_models.append(scaler)
                loaded_ensemble_paths.append(target_path)
                print(f"[SUCCESS] Loaded Fold {i} model from {target_path} (T = {scaler.temperature.item():.4f})")
                
                if first_base_model is None:
                    first_base_model = base_model
            except Exception as e:
                # Emojis stripped to prevent encoding crash on standard Windows shell
                print(f"[ERROR] Failed to load fold {i} from {target_path}: {str(e)}")
                
    if len(loaded_models) > 0:
        print(f"[SUCCESS] SUCCESSFULLY DEPLOYED {len(loaded_models)}-FOLD ENSEMBLE: {loaded_ensemble_paths}")
        model = EnsembleTemperatureScaler(loaded_models)
        # Use first model for Grad-CAM++ visualization to ensure high-performance inference
        gradcam = GradCAMPP(first_base_model)
    else:
        print("[WARNING] No checkpoints found in 'checkpoints/'! Initializing single random model.")
        base_model = FractureDislocModel(pretrained=False).to(DEVICE)
        scaler = TemperatureScaler(base_model).to(DEVICE)
        scaler.eval()
        model = EnsembleTemperatureScaler([scaler])
        gradcam = GradCAMPP(base_model)

# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────
class PatientInfo(BaseModel):
    age:           float = Field(..., ge=0,  le=120,  description="Patient age in years")
    sex:           int   = Field(..., ge=0,  le=1,    description="0=female, 1=male")
    bmi:           float = Field(..., ge=10, le=70,   description="Body Mass Index")
    mechanism:     int   = Field(..., ge=0,  le=2,    description="0=low-energy, 1=high-energy, 2=pathological")
    bone_density:  float = Field(0.0, ge=-5, le=3,   description="DEXA T-score")
    prior_fracture:int   = Field(0,   ge=0,  le=1,   description="Prior fracture history")
    pain_score:    float = Field(5.0, ge=0,  le=10,  description="Pain score 0–10")

    def to_tensor(self, device) -> torch.Tensor:
        vec = torch.tensor([
            self.age           / 100.0,
            float(self.sex),
            self.bmi           / 50.0,
            self.mechanism     / 2.0,
            (self.bone_density + 4.0) / 7.0,
            float(self.prior_fracture),
            self.pain_score    / 10.0,
        ], dtype=torch.float32).unsqueeze(0).to(device)
        return vec

class PredictionResponse(BaseModel):
    fracture_probability:    float
    dislocation_probability: float
    fracture_risk:           str   # "Low" / "Moderate" / "High"
    dislocation_risk:        str
    heatmap_base64:          Optional[str] = None   # JPEG base64 overlay image
    model_uncertainty:       Optional[dict] = None  # Stats across the ensemble folds

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def risk_label(prob: float) -> str:
    if prob < 0.35: return "Low"
    if prob < 0.65: return "Moderate"
    return "High"

def preprocess_image(file_bytes: bytes) -> tuple:
    """Decode uploaded image bytes → numpy array + torch tensor."""
    arr = np.frombuffer(file_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "Cannot decode image. Upload a JPEG/PNG X-ray.")
    bgr_display = cv2.resize(bgr, (224, 224))
    rgb = cv2.cvtColor(bgr_display, cv2.COLOR_BGR2RGB)

    transform = get_transforms("val")
    tensor    = transform(rgb).unsqueeze(0).to(DEVICE)
    return bgr_display, tensor

def heatmap_to_base64(overlay_bgr: np.ndarray) -> str:
    """Encode overlay image as base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", overlay_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("utf-8")

# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────
from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serves the premium dark-themed clinical diagnosis dashboard."""
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h3>Error: index.html not found in files directory.</h3>"

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "ensemble_size": len(model.models) if model else 0,
        "loaded_paths": loaded_ensemble_paths,
        "is_ensemble": len(loaded_ensemble_paths) > 1
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    xray:          UploadFile = File(...,  description="X-ray image (JPEG/PNG)"),
    age:           float      = Form(...),
    sex:           int        = Form(...),
    bmi:           float      = Form(...),
    mechanism:     int        = Form(...),
    bone_density:  float      = Form(0.0),
    prior_fracture:int        = Form(0),
    pain_score:    float      = Form(5.0),
    generate_heatmap: bool    = Form(True),
    target:        str        = Form("fracture"),   # 'fracture' or 'dislocation'
):
    patient = PatientInfo(
        age=age, sex=sex, bmi=bmi, mechanism=mechanism,
        bone_density=bone_density, prior_fracture=prior_fracture,
        pain_score=pain_score,
    )

    file_bytes         = await xray.read()
    bgr_display, img_t = preprocess_image(file_bytes)
    clinical_t         = patient.to_tensor(DEVICE)

    # ── Ensemble Prediction ──
    with torch.no_grad():
        out = model(img_t, clinical_t)

    # Extract averages (ensemble consensus values)
    frac_prob = round(float(out["fracture_prob"][0]),    4)
    disl_prob = round(float(out["dislocation_prob"][0]), 4)

    # ── Heatmap (using 1st fold model for speed and accuracy) ──
    heatmap_b64 = None
    if generate_heatmap:
        img_t_grad = img_t.clone().requires_grad_(True)
        hm         = gradcam.generate(img_t_grad, clinical_t, target=target)
        overlay    = GradCAMPP.overlay(bgr_display, hm, alpha=0.45)
        heatmap_b64 = heatmap_to_base64(overlay)

    # ── Ensemble Uncertainty Estimation ──
    uncertainty_dict = None
    if "fracture_std" in out:
        frac_std = float(out["fracture_std"][0])
        disl_std = float(out["dislocation_std"][0])
        
        # Build consensus index (Standard Deviation < 0.1 is very high consensus, > 0.2 is low)
        max_std = max(frac_std, disl_std)
        consensus_score = 100.0 * (1.0 - max_std * 2.0)
        consensus_score = max(0.0, min(100.0, consensus_score))
        
        if max_std < 0.08:
            agreement_label = "Strong Consensus"
        elif max_std < 0.18:
            agreement_label = "Moderate Consensus"
        else:
            agreement_label = "Divergent Folds (Manual Clinical Review Recommended)"

        uncertainty_dict = {
            "fracture_std": round(frac_std, 4),
            "dislocation_std": round(disl_std, 4),
            "consensus_agreement": agreement_label,
            "confidence_score_percent": round(consensus_score, 1),
            "num_folds_evaluated": len(model.models)
        }

    return PredictionResponse(
        fracture_probability    = frac_prob,
        dislocation_probability = disl_prob,
        fracture_risk           = risk_label(frac_prob),
        dislocation_risk        = risk_label(disl_prob),
        heatmap_base64          = heatmap_b64,
        model_uncertainty       = uncertainty_dict,
    )

@app.post("/predict/batch")
async def predict_batch(
    xrays:    List[UploadFile] = File(...),
    patients: str              = Form(...),  # JSON list of PatientInfo dicts
):
    """Batch prediction without heatmaps (highly optimized)."""
    import json
    patient_dicts = json.loads(patients)
    if len(xrays) != len(patient_dicts):
        raise HTTPException(400, "Number of images must match number of patient records.")

    results = []
    for xray, pdict in zip(xrays, patient_dicts):
        patient = PatientInfo(**pdict)
        fbytes  = await xray.read()
        _, img_t    = preprocess_image(fbytes)
        clinical_t  = patient.to_tensor(DEVICE)

        with torch.no_grad():
            out = model(img_t, clinical_t)

        frac_prob = float(out["fracture_prob"][0])
        disl_prob = float(out["dislocation_prob"][0])

        fold_stats = {}
        if "fracture_std" in out:
            fold_stats = {
                "fracture_std": round(float(out["fracture_std"][0]), 4),
                "dislocation_std": round(float(out["dislocation_std"][0]), 4)
            }

        results.append({
            "filename":              xray.filename,
            "fracture_probability":    round(frac_prob, 4),
            "dislocation_probability": round(disl_prob, 4),
            "fracture_risk":           risk_label(frac_prob),
            "dislocation_risk":        risk_label(disl_prob),
            "uncertainty":             fold_stats
        })

    return {"predictions": results}

# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
