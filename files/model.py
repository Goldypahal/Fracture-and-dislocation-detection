"""
Bone Fracture & Joint Dislocation Detection Model
==================================================
Multi-modal model: X-ray image + patient clinical info
Outputs: fracture probability, dislocation probability, Grad-CAM heatmap

Architecture: EfficientNetV2 (image) + MLP (clinical) → Fusion → Multi-head output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import cv2
from typing import Tuple, Dict, Optional


# ─────────────────────────────────────────────
# 1. CLINICAL BRANCH  (patient metadata)
# ─────────────────────────────────────────────
class ClinicalBranch(nn.Module):
    """
    Processes patient metadata:
      age, sex, bmi, mechanism_of_injury, bone_density_tscore,
      prior_fracture (0/1), pain_score (0-10)
    """
    def __init__(self, input_dim: int = 7, hidden_dims=(64, 128)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3)]
            prev = h
        self.net = nn.Sequential(*layers)
        self.output_dim = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────
# 2. IMAGE BRANCH  (X-ray CNN backbone)
# ─────────────────────────────────────────────
class ImageBranch(nn.Module):
    """
    EfficientNetV2-S backbone pretrained on ImageNet.
    Fine-tuned on medical X-rays.
    Returns feature vector + keeps last conv layer accessible for Grad-CAM.
    """
    def __init__(self, pretrained: bool = True, output_dim: int = 256):
        super().__init__()
        backbone = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        )
        # Keep all layers except the final classifier
        self.features = backbone.features          # Conv + BN + Swish blocks
        self.avgpool  = backbone.avgpool
        in_features   = backbone.classifier[1].in_features

        self.projection = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, output_dim),
            nn.ReLU(),
        )
        self.output_dim = output_dim

        # Store last feature map for Grad-CAM
        self.last_feature_map: Optional[torch.Tensor] = None
        self.last_grad:        Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self):
        """Register forward/backward hooks on the last conv block for Grad-CAM."""
        def forward_hook(module, inp, out):
            self.last_feature_map = out

        def backward_hook(module, grad_in, grad_out):
            self.last_grad = grad_out[0]

        # Last block of EfficientNetV2 features
        last_block = list(self.features.children())[-1]
        last_block.register_forward_hook(forward_hook)
        last_block.register_full_backward_hook(backward_hook)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        pooled = self.avgpool(feat).flatten(1)
        return self.projection(pooled)


# ─────────────────────────────────────────────
# 2.5 LEGACY MODELS (Image-only, checkpoint backward-compatible)
# ─────────────────────────────────────────────
class LegacyImageBranch(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.efficientnet_v2_s(weights=None)
        self.features = backbone.features
        self.avgpool  = backbone.avgpool
        self.last_feature_map = None
        self.last_grad = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.last_feature_map = out

        def backward_hook(module, grad_in, grad_out):
            self.last_grad = grad_out[0]

        last_block = list(self.features.children())[-1]
        last_block.register_forward_hook(forward_hook)
        last_block.register_full_backward_hook(backward_hook)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return self.avgpool(feat).flatten(1)

class LegacyFractureDislocModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_branch = LegacyImageBranch()
        self.head_frac    = nn.Linear(1280, 1)
        self.head_disl    = nn.Linear(1280, 1)

    def forward(self, image: torch.Tensor, clinical: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        pooled = self.image_branch(image)
        frac_logit = self.head_frac(pooled).squeeze(1)
        disl_logit = self.head_disl(pooled).squeeze(1)
        return {
            "fracture_logit":    frac_logit,
            "dislocation_logit": disl_logit,
            "fracture_prob":     torch.sigmoid(frac_logit),
            "dislocation_prob":  torch.sigmoid(disl_logit),
        }


# ─────────────────────────────────────────────
# 3. FUSION + MULTI-HEAD OUTPUT
# ─────────────────────────────────────────────
class FractureDislocModel(nn.Module):
    """
    Full multi-modal model.

    Inputs:
        image   : (B, 3, 224, 224) tensor — preprocessed X-ray
        clinical: (B, 7) tensor — patient features (see ClinicalBranch)

    Outputs (dict):
        fracture_prob    : (B,) probability of fracture
        dislocation_prob : (B,) probability of dislocation
    """
    def __init__(
        self,
        clinical_dim:  int  = 7,
        image_out_dim: int  = 256,
        fusion_dim:    int  = 256,
        pretrained:    bool = True,
    ):
        super().__init__()
        self.image_branch    = ImageBranch(pretrained=pretrained, output_dim=image_out_dim)
        self.clinical_branch = ClinicalBranch(input_dim=clinical_dim)

        combined = image_out_dim + self.clinical_branch.output_dim
        self.fusion = nn.Sequential(
            nn.Linear(combined, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
        )

        # Two separate heads
        self.head_fracture    = nn.Linear(128, 1)
        self.head_dislocation = nn.Linear(128, 1)

    def forward(
        self,
        image:    torch.Tensor,
        clinical: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        img_feat  = self.image_branch(image)
        clin_feat = self.clinical_branch(clinical)

        combined = torch.cat([img_feat, clin_feat], dim=1)
        fused    = self.fusion(combined)

        frac_logit = self.head_fracture(fused).squeeze(1)
        disl_logit = self.head_dislocation(fused).squeeze(1)

        return {
            "fracture_logit":    frac_logit,
            "dislocation_logit": disl_logit,
            "fracture_prob":     torch.sigmoid(frac_logit),
            "dislocation_prob":  torch.sigmoid(disl_logit),
        }


# ─────────────────────────────────────────────
# 4. GRAD-CAM++ HEATMAP
# ─────────────────────────────────────────────
class GradCAMPP:
    """
    Grad-CAM++ heatmap generator.
    Highlights regions contributing to a specific prediction.

    Usage:
        cam = GradCAMPP(model)
        heatmap = cam.generate(image_tensor, target='fracture')
        overlay = cam.overlay(original_img_bgr, heatmap)
    """
    def __init__(self, model: FractureDislocModel):
        self.model = model

    def generate(
        self,
        image:    torch.Tensor,   # (1, 3, 224, 224)
        clinical: torch.Tensor,   # (1, 7)
        target:   str = "fracture",  # 'fracture' or 'dislocation'
    ) -> np.ndarray:
        """Returns a (224, 224) float32 heatmap, values in [0, 1]."""
        self.model.eval()
        image    = image.requires_grad_(True)
        clinical = clinical.detach()

        out = self.model(image, clinical)
        logit = out[f"{target}_logit"]
        self.model.zero_grad()
        logit.backward()

        grads   = self.model.image_branch.last_grad    # (1, C, H, W)
        fmaps   = self.model.image_branch.last_feature_map.detach()  # (1, C, H, W)

        # Grad-CAM++ weights
        grad_sq   = grads ** 2
        grad_cu   = grads ** 3
        alpha_num = grad_sq
        alpha_den = 2 * grad_sq + (fmaps * grad_cu).sum(dim=(2, 3), keepdim=True) + 1e-7
        alpha     = alpha_num / alpha_den

        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam     = (weights * fmaps).sum(dim=1, keepdim=True)              # (1, 1, H, W)
        cam     = F.relu(cam).squeeze().cpu().numpy()

        # Normalise to [0, 1]
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam

    @staticmethod
    def overlay(
        original_bgr: np.ndarray,   # H×W×3 uint8
        heatmap:      np.ndarray,   # H×W float [0,1]
        alpha:        float = 0.45,
    ) -> np.ndarray:
        """Blends a coloured heatmap onto the original X-ray."""
        h, w = original_bgr.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_u8      = np.uint8(255 * heatmap_resized)
        colormap        = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
        blended         = cv2.addWeighted(original_bgr, 1 - alpha, colormap, alpha, 0)
        return blended


# ─────────────────────────────────────────────
# 5. LOSS FUNCTION  (weighted BCE for imbalance)
# ─────────────────────────────────────────────
class MultiTaskLoss(nn.Module):
    """
    Weighted binary cross-entropy for both tasks.
    pos_weight handles class imbalance (fractures are rarer than non-fractures).
    """
    def __init__(
        self,
        frac_pos_weight:    float = 3.0,   # ↑ if dataset has few positives
        disl_pos_weight:    float = 4.0,
        task_weight_frac:   float = 1.0,
        task_weight_disl:   float = 0.8,
    ):
        super().__init__()
        self.w_frac = task_weight_frac
        self.w_disl = task_weight_disl
        self.bce_frac = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(frac_pos_weight)
        )
        self.bce_disl = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(disl_pos_weight)
        )

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        frac_labels: torch.Tensor,
        disl_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        l_frac = self.bce_frac(outputs["fracture_logit"],    frac_labels.float())
        l_disl = self.bce_disl(outputs["dislocation_logit"], disl_labels.float())
        total  = self.w_frac * l_frac + self.w_disl * l_disl
        return {"total": total, "fracture": l_frac, "dislocation": l_disl}


# ─────────────────────────────────────────────
# 6. PROBABILITY CALIBRATION (Temperature Scaling)
# ─────────────────────────────────────────────
class TemperatureScaler(nn.Module):
    """
    Post-hoc calibration: learn a single temperature T.
    Fit on a held-out validation set AFTER main training.

    Usage:
        scaler = TemperatureScaler(model)
        scaler.calibrate(val_loader)
        # Now use scaler(image, clinical) for calibrated probs
    """
    def __init__(self, model: FractureDislocModel):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, image, clinical):
        out = self.model(image, clinical)
        return {
            "fracture_prob":     torch.sigmoid(out["fracture_logit"]    / self.temperature),
            "dislocation_prob":  torch.sigmoid(out["dislocation_logit"] / self.temperature),
            "fracture_logit":    out["fracture_logit"],
            "dislocation_logit": out["dislocation_logit"],
        }

    def calibrate(self, val_loader, device="cuda", epochs=50):
        """Optimise temperature on validation set using NLL loss."""
        self.model.eval()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        criterion = nn.BCEWithLogitsLoss()

        all_frac_logits, all_disl_logits = [], []
        all_frac_labels, all_disl_labels = [], []

        with torch.no_grad():
            for images, clinicals, frac_lbl, disl_lbl in val_loader:
                images    = images.to(device)
                clinicals = clinicals.to(device)
                out = self.model(images, clinicals)
                all_frac_logits.append(out["fracture_logit"].cpu())
                all_disl_logits.append(out["dislocation_logit"].cpu())
                all_frac_labels.append(frac_lbl)
                all_disl_labels.append(disl_lbl)

        frac_logits = torch.cat(all_frac_logits)
        disl_logits = torch.cat(all_disl_logits)
        frac_labels = torch.cat(all_frac_labels).float()
        disl_labels = torch.cat(all_disl_labels).float()

        def closure():
            optimizer.zero_grad()
            loss = (
                criterion(frac_logits / self.temperature, frac_labels) +
                criterion(disl_logits / self.temperature, disl_labels)
            )
            loss.backward()
            return loss

        for _ in range(epochs):
            optimizer.step(closure)

        print(f"Calibrated temperature: {self.temperature.item():.4f}")


def load_compatible_model(checkpoint_path: str, device: str = "cpu") -> Tuple[nn.Module, nn.Module]:
    """
    Auto-detects the model architecture inside the checkpoint,
    initializes the correct model (Legacy or Multi-modal),
    remaps keys if necessary, and returns (base_model, temperature_scaler).
    """
    state = torch.load(checkpoint_path, map_location=device)
    
    # 1. Determine if state is calibrated or raw state dict
    is_calibrated = "temperature" in state or ("state_dict" in state and "temperature" in state["state_dict"])
    
    # 2. Extract raw state keys
    raw_keys = []
    if "temperature" in state:
        raw_keys = list(state.keys())
    elif isinstance(state, dict) and "state_dict" in state:
        raw_keys = list(state["state_dict"].keys())
    else:
        raw_keys = list(state.keys())
        
    # Check if this checkpoint uses legacy image-only architecture
    is_legacy = any("head_frac" in k or "backbone" in k for k in raw_keys)
    
    if is_legacy:
        print(f"  [INFO] Auto-detected Legacy Image-Only Architecture for {checkpoint_path}")
        base_model = LegacyFractureDislocModel().to(device)
        scaler = TemperatureScaler(base_model).to(device)
        
        if "temperature" in state:
            # Calibrated checkpoint: key names in scaler state dict will be e.g. 'model.image_branch.features.0...'
            # We map legacy model.backbone.features keys to model.image_branch.features if needed
            new_state = {}
            for k, v in state.items():
                nk = k.replace("model.backbone.features.", "model.image_branch.features.")
                new_state[nk] = v
            scaler.load_state_dict(new_state)
        else:
            # Raw state dict
            raw_state = state["state_dict"] if ("state_dict" in state) else state
            new_state = {}
            for k, v in raw_state.items():
                nk = k.replace("backbone.features.", "image_branch.features.")
                new_state[nk] = v
            base_model.load_state_dict(new_state)
            # Use default temperature for raw models
            scaler.temperature.data = torch.ones(1, device=device) * 1.5
    else:
        print(f"  [INFO] Auto-detected Multi-Modal Architecture for {checkpoint_path}")
        base_model = FractureDislocModel(pretrained=False).to(device)
        scaler = TemperatureScaler(base_model).to(device)
        
        if "temperature" in state:
            scaler.load_state_dict(state)
        else:
            raw_state = state["state_dict"] if ("state_dict" in state) else state
            base_model.load_state_dict(raw_state)
            scaler.temperature.data = torch.ones(1, device=device) * 1.5
            
    scaler.eval()
    return base_model, scaler


if __name__ == "__main__":
    # Quick smoke test
    model = FractureDislocModel(pretrained=False)
    img   = torch.randn(2, 3, 224, 224)
    clin  = torch.randn(2, 7)
    out   = model(img, clin)
    print("fracture_prob   :", out["fracture_prob"])
    print("dislocation_prob:", out["dislocation_prob"])

    cam    = GradCAMPP(model)
    hm     = cam.generate(img[:1], clin[:1], target="fracture")
    print("heatmap shape:", hm.shape, "range:", hm.min(), "–", hm.max())
