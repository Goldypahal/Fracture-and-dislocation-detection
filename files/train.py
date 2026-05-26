"""
Training Pipeline for Bone Fracture & Dislocation Model (Self-Contained Kaggle Version)
=======================================================================================
Features:
  - CLAHE contrast enhancement for X-ray preprocessing
  - Unified multi-modal dataset (X-ray image + Patient clinical info)
  - EfficientNetV2-S backbone + MLP clinical branch + late fusion
  - Multi-task loss with task weighting and positive scaling (BCEWithLogitsLoss)
  - Temperature scaling for probability calibration
  - Mixed precision training (AMP) for speed and memory efficiency
  - Cosine learning rate scheduler with linear warmup
  - Stratified K-Fold splits
  - Early stopping based on combined validation AUC
  - Grad-CAM++ compatibility hooks built-in
"""

import os
import sys
import cv2
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from typing import Tuple, List, Dict, Optional

# Set environment variables
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# ─────────────────────────────────────────────
# GPU COMPATIBILITY CHECKER
# ─────────────────────────────────────────────
def check_gpu_compatibility():
    if not torch.cuda.is_available():
        print("❌ No GPU found — switch to GPU in Settings")
        sys.exit(1)
    
    gpu_name  = torch.cuda.get_device_name(0)
    cap_major = torch.cuda.get_device_capability(0)[0]
    cap_minor = torch.cuda.get_device_capability(0)[1]
    
    print(f"GPU detected: {gpu_name}")
    print(f"CUDA capability: sm_{cap_major}{cap_minor}")
    print(f"PyTorch version: {torch.__version__}")
    
    if cap_major < 7:
        print(f"""
❌ INCOMPATIBLE GPU: {gpu_name}
   CUDA capability sm_{cap_major}{cap_minor} is too old
   PyTorch needs sm_70 minimum
   
   FIX:
   1. Go to notebook Settings (right panel)
   2. Accelerator → Stop Session
   3. Change to T4 GPU
   4. Start Session again
   5. Run all cells
        """)
        sys.exit(1)
    else:
        print(f"✓ GPU compatible — ready to train")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CFG = dict(
    img_size      = 224,
    batch_size    = 32,
    num_workers   = 4,
    epochs        = 20,             # Optimized: 20 epochs is excellent for fine-tuning EfficientNetV2
    lr            = 3e-4,
    weight_decay  = 1e-4,
    warmup_epochs = 3,
    n_folds       = 5,
    seed          = 42,
    device        = "cuda" if torch.cuda.is_available() else "cpu",
    save_dir      = "checkpoints",
    frac_pos_w    = 3.0,
    disl_pos_w    = 4.0,
    early_stop    = 5,              # Early stopping patience
    amp           = True,
    freeze_epochs = 3,              # Freeze backbone for first N epochs
)

# ─────────────────────────────────────────────
# CLINICAL FEATURE COLUMNS (order matters!)
# ─────────────────────────────────────────────
CLINICAL_COLS = [
    "age",           # continuous, normalised 0-1  (age/100)
    "sex",           # 0=female, 1=male
    "bmi",           # continuous, normalised  (bmi/50)
    "mechanism",     # 0=low_energy, 1=high_energy, 2=pathological
    "bone_density",  # T-score, normalised  ((t+4)/7)
    "prior_fracture",# binary 0/1
    "pain_score",    # 0-10 normalised  (pain/10)
]

# ─────────────────────────────────────────────
# CLAHE X-RAY PREPROCESSOR
# ─────────────────────────────────────────────
def apply_clahe(image_bgr: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalisation.
    Dramatically improves bone/fracture visibility in X-rays.
    Works on grayscale or RGB (applies to L channel in LAB space).
    """
    if len(image_bgr.shape) == 2:
        # Already grayscale
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        return clahe.apply(image_bgr)

    lab   = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_eq  = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)

def load_xray(
    path: str,
    size: int = 224,
    apply_clahe_flag: bool = True,
    to_rgb: bool = True,
) -> np.ndarray:
    """Load X-ray, apply CLAHE, resize. Returns uint8 H×W×3 BGR."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    if apply_clahe_flag:
        img = apply_clahe(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# ─────────────────────────────────────────────
# TORCHVISION TRANSFORMS
# ─────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_transforms(split: str = "train", img_size: int = 224):
    if split == "train":
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

# ─────────────────────────────────────────────
# DATASET CLASS
# ─────────────────────────────────────────────
class FractureDataset(Dataset):
    """
    Unified dataset for bone fracture + dislocation prediction.
    Expected CSV columns:
        image_path       : absolute or relative path to X-ray
        fracture_label   : 0 or 1
        dislocation_label: 0 or 1
        age, sex, bmi, mechanism, bone_density, prior_fracture, pain_score
    """
    def __init__(
        self,
        df:         pd.DataFrame,
        split:      str  = "train",
        img_size:   int  = 224,
        clahe:      bool = True,
    ):
        self.df        = df.reset_index(drop=True)
        self.transform = get_transforms(split, img_size)
        self.img_size  = img_size
        self.clahe     = clahe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ── Image ──
        img_np = load_xray(row["image_path"], self.img_size, self.clahe)
        image  = self.transform(img_np)

        # ── Clinical ──
        clinical = torch.tensor([
            row["age"]           / 100.0,
            float(row["sex"]),
            row["bmi"]           / 50.0,
            row["mechanism"]     / 2.0,
            (row["bone_density"] + 4.0) / 7.0,
            float(row["prior_fracture"]),
            row["pain_score"]    / 10.0,
        ], dtype=torch.float32)

        frac_label = torch.tensor(row["fracture_label"],    dtype=torch.float32)
        disl_label = torch.tensor(row["dislocation_label"], dtype=torch.float32)

        return image, clinical, frac_label, disl_label

# ─────────────────────────────────────────────
# DATA LOADING HELPERS (per dataset format)
# ─────────────────────────────────────────────
def load_fractatlas(data_root: str) -> pd.DataFrame:
    """
    FracAtlas: JSON annotations.
    """
    # Look for COCO JSON in the standard or nested path
    ann_path = os.path.join(data_root, "Annotations", "COCO JSON", "COCO_fracture_masks.json")
    if not os.path.exists(ann_path):
        ann_path = os.path.join(data_root, "annotations.json")
        
    with open(ann_path) as f:
        ann = json.load(f)

    # Walk directory to find exact path for each image (since they are split into fractured/non-fractured subfolders)
    img_files = {}
    images_dir = os.path.join(data_root, "images")
    if os.path.exists(images_dir):
        for root, _, files in os.walk(images_dir):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    img_files[f] = os.path.join(root, f)

    id2path = {}
    for img in ann["images"]:
        fname = img["file_name"]
        if fname in img_files:
            id2path[img["id"]] = img_files[fname]
        else:
            id2path[img["id"]] = os.path.join(data_root, "images", fname)

    frac_ids = {a["image_id"] for a in ann["annotations"]}

    rows = []
    for img in ann["images"]:
        rows.append({
            "image_path":        id2path[img["id"]],
            "fracture_label":    int(img["id"] in frac_ids),
            "dislocation_label": 0,   # FracAtlas doesn't label dislocations
            "age": 45, "sex": 0, "bmi": 25.0,
            "mechanism": 1, "bone_density": 0.0,
            "prior_fracture": 0, "pain_score": 5,
        })
    return pd.DataFrame(rows)

def load_grazpedwri(data_root: str, label_csv: str) -> pd.DataFrame:
    """
    GRAZPEDWRI-DX: pediatric wrist X-rays.
    """
    df = pd.read_csv(label_csv)
    
    # ── Image Path ──
    if "filestem" in df.columns:
        df["image_path"] = df["filestem"].apply(
            lambda f: os.path.join(data_root, str(f) + ".png")
        )
    elif "filename" in df.columns:
        df["image_path"] = df["filename"].apply(
            lambda f: os.path.join(data_root, str(f))
        )
    else:
        raise ValueError("Neither 'filestem' nor 'filename' column found in label CSV.")

    # ── Fracture Label ──
    if "fracture_visible" in df.columns:
        df["fracture_label"] = df["fracture_visible"].fillna(0).astype(int)
    elif "fracture" in df.columns:
        df["fracture_label"] = df["fracture"].astype(int)
    else:
        df["fracture_label"] = 0

    df["dislocation_label"] = 0

    # ── Sex / Gender ──
    if "gender" in df.columns:
        df["sex"] = df["gender"].apply(lambda g: 1 if str(g).strip().upper() == "M" else 0)
    elif "sex" in df.columns:
        df["sex"] = df["sex"].astype(int)
    else:
        df["sex"] = 0

    # ── Age ──
    if "age" in df.columns:
        df["age"] = df["age"].astype(float)
    elif "age_months" in df.columns:
        df["age"] = df["age_months"].astype(float) / 12.0
    else:
        df["age"] = 10.0  # pediatric default

    # ── Clinical Defaults ──
    df["bmi"]            = 18.0   # pediatric default
    df["mechanism"]      = 1
    df["bone_density"]   = 0.0
    df["prior_fracture"] = 0
    df["pain_score"]     = 5

    return df[["image_path", "fracture_label", "dislocation_label",
               "age", "sex", "bmi", "mechanism", "bone_density",
               "prior_fracture", "pain_score"]]

def build_stratified_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, val_idx in skf.split(df, df["fracture_label"]):
        yield df.iloc[train_idx], df.iloc[val_idx]

def get_dataloaders(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = FractureDataset(train_df, split="train")
    val_ds   = FractureDataset(val_df,   split="val")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader

# ─────────────────────────────────────────────
# 1. CLINICAL BRANCH  (patient metadata)
# ─────────────────────────────────────────────
class ClinicalBranch(nn.Module):
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
    def __init__(self, pretrained: bool = True, output_dim: int = 256):
        super().__init__()
        backbone = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        )
        self.features = backbone.features
        self.avgpool  = backbone.avgpool
        in_features   = backbone.classifier[1].in_features

        self.projection = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, output_dim),
            nn.ReLU(),
        )
        self.output_dim = output_dim

        self.last_feature_map: Optional[torch.Tensor] = None
        self.last_grad:        Optional[torch.Tensor] = None
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
# 4. LOSS FUNCTION
# ─────────────────────────────────────────────
class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        frac_pos_weight:    float = 3.0,
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
# 5. TEMPERATURE SCALER
# ─────────────────────────────────────────────
class TemperatureScaler(nn.Module):
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

        frac_logits = torch.cat(all_frac_logits).to(self.temperature.device)
        disl_logits = torch.cat(all_disl_logits).to(self.temperature.device)
        frac_labels = torch.cat(all_frac_labels).float().to(self.temperature.device)
        disl_labels = torch.cat(all_disl_labels).float().to(self.temperature.device)

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


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def compute_metrics(
    probs:  np.ndarray,
    labels: np.ndarray,
    thresh: float = 0.5,
    task_name: str = "",
) -> Dict[str, float]:
    preds = (probs >= thresh).astype(int)
    auc   = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
    f1    = f1_score(labels, preds, zero_division=0)
    tp    = ((preds == 1) & (labels == 1)).sum()
    tn    = ((preds == 0) & (labels == 0)).sum()
    fp    = ((preds == 1) & (labels == 0)).sum()
    fn    = ((preds == 0) & (labels == 1)).sum()
    sens  = tp / (tp + fn + 1e-8)
    spec  = tn / (tn + fp + 1e-8)
    print(f"  [{task_name}] AUC={auc:.4f}  F1={f1:.4f}  Sensitivity={sens:.4f}  Specificity={spec:.4f}")
    return {"auc": auc, "f1": f1, "sensitivity": sens, "specificity": spec}

# ─────────────────────────────────────────────
# WARMUP LR SCHEDULER
# ─────────────────────────────────────────────
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr):
        self.optimizer    = optimizer
        self.warmup       = warmup_epochs
        self.total        = total_epochs
        self.base_lr      = base_lr

    def step(self, epoch):
        if epoch < self.warmup:
            lr = self.base_lr * (epoch + 1) / self.warmup
        else:
            progress = (epoch - self.warmup) / (self.total - self.warmup)
            lr = 1e-6 + 0.5 * (self.base_lr - 1e-6) * (1.0 + np.cos(np.pi * progress))
        
        # Apply lr dynamically. Maintain 0.1x scaling for backbone (param group 1)
        for i, pg in enumerate(self.optimizer.param_groups):
            if i == 0:
                pg["lr"] = lr
            elif i == 1:
                pg["lr"] = lr * 0.1

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

# ─────────────────────────────────────────────
# ONE EPOCH
# ─────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, scaler, device, training=True):
    model.train() if training else model.eval()
    total_loss = 0.0
    all_frac_probs, all_disl_probs = [], []
    all_frac_lbls,  all_disl_lbls  = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, clinicals, frac_lbl, disl_lbl in loader:
            images    = images.to(device, non_blocking=True)
            clinicals = clinicals.to(device, non_blocking=True)
            frac_lbl  = frac_lbl.to(device)
            disl_lbl  = disl_lbl.to(device)

            with autocast("cuda", enabled=CFG["amp"]):
                out    = model(images, clinicals)
                losses = criterion(out, frac_lbl, disl_lbl)
                loss   = losses["total"]

            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()
            all_frac_probs.extend(out["fracture_prob"].detach().cpu().numpy())
            all_disl_probs.extend(out["dislocation_prob"].detach().cpu().numpy())
            all_frac_lbls.extend(frac_lbl.cpu().numpy())
            all_disl_lbls.extend(disl_lbl.cpu().numpy())

    avg_loss = total_loss / len(loader)
    frac_m   = compute_metrics(np.array(all_frac_probs), np.array(all_frac_lbls), task_name="Fracture")
    disl_m   = compute_metrics(np.array(all_disl_probs), np.array(all_disl_lbls), task_name="Dislocation")
    return avg_loss, frac_m, disl_m

# ─────────────────────────────────────────────
# OPTIMIZER GENERATOR (RESUME COMPATIBLE)
# ─────────────────────────────────────────────
def get_optimizer(model, epoch, lr, weight_decay, freeze_epochs):
    if epoch < freeze_epochs:
        # Backbone is frozen
        for param in model.image_branch.features.parameters():
            param.requires_grad = False
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=weight_decay
        )
    else:
        # Backbone is unfrozen
        for param in model.image_branch.features.parameters():
            param.requires_grad = True
        
        # Group parameters
        backbone_params = list(model.image_branch.features.parameters())
        backbone_param_ids = set(id(p) for p in backbone_params)
        
        other_params = [p for p in model.parameters() if id(p) not in backbone_param_ids]
        
        return torch.optim.AdamW([
            {"params": other_params, "lr": lr, "weight_decay": weight_decay},
            {"params": backbone_params, "lr": lr * 0.1, "weight_decay": weight_decay}
        ])

# ─────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────
def train(df: pd.DataFrame):
    os.makedirs(CFG["save_dir"], exist_ok=True)
    device = CFG["device"]
    print(f"Training on: {device}")

    # Check for latest epoch-level crash recovery checkpoint
    latest_path = os.path.join(CFG["save_dir"], "latest_checkpoint.pt")
    root_latest_path = "latest_checkpoint.pt"
    
    existing_latest = None
    if os.path.exists(latest_path):
        existing_latest = latest_path
    elif os.path.exists(root_latest_path):
        existing_latest = root_latest_path
        
    resume_fold = 0
    resume_epoch = 0
    resume_checkpoint = None
    all_fold_aucs = []
    
    if existing_latest:
        try:
            resume_checkpoint = torch.load(existing_latest, map_location=device)
            resume_fold = resume_checkpoint["fold"]
            resume_epoch = resume_checkpoint["epoch"] + 1  # Resume at the next epoch
            all_fold_aucs = resume_checkpoint.get("all_aucs", [])
            print(f"✓ Found latest checkpoint. Resuming from Fold {resume_fold+1}, Epoch {resume_epoch+1}")
        except Exception as e:
            print(f"⚠️ Error loading latest checkpoint: {e}. Starting fresh.")
            resume_checkpoint = None

    for fold_idx, (train_df, val_df) in enumerate(
        build_stratified_folds(df, n_splits=CFG["n_folds"], seed=CFG["seed"])
    ):
        # 1. Skip fully completed folds in the resumed session
        if resume_checkpoint is not None and fold_idx < resume_fold:
            print(f"  >> Skipping Fold {fold_idx+1} (handled by active resumed session)")
            continue

        best_path = os.path.join(CFG["save_dir"], f"best_fold{fold_idx+1}.pt")
        calibrated_path = best_path.replace(".pt", "_calibrated.pt")
        
        # Also check root directory in case checkpoints were downloaded/copied there
        root_best_path = f"best_fold{fold_idx+1}.pt"
        root_calibrated_path = f"best_fold{fold_idx+1}_calibrated.pt"
        
        existing_best = None
        if os.path.exists(best_path):
            existing_best = best_path
        elif os.path.exists(root_best_path):
            existing_best = root_best_path
            
        existing_calibrated = None
        if os.path.exists(calibrated_path):
            existing_calibrated = calibrated_path
        elif os.path.exists(root_calibrated_path):
            existing_calibrated = root_calibrated_path

        # 2. Skip fully completed folds (prior finished folds from a previous run)
        if existing_best and (resume_checkpoint is None or fold_idx != resume_fold):
            try:
                checkpoint = torch.load(existing_best, map_location=device)
                best_val_auc = checkpoint.get("auc", 0.0)
                print(f"\n✓ Found existing completed checkpoint for Fold {fold_idx+1}: {existing_best} (AUC={best_val_auc:.4f})")
                
                # Copy files to standard save_dir if they were only in the root directory
                if existing_best != best_path:
                    import shutil
                    shutil.copy(existing_best, best_path)
                
                if existing_calibrated:
                    print(f"✓ Fold {fold_idx+1} is already trained and calibrated. Skipping fold.")
                    if existing_calibrated != calibrated_path:
                        import shutil
                        shutil.copy(existing_calibrated, calibrated_path)
                    all_fold_aucs.append(best_val_auc)
                    continue
                else:
                    print(f"✓ Fold {fold_idx+1} model exists but is not calibrated. Running calibration now...")
                    train_loader, val_loader = get_dataloaders(
                        train_df, val_df,
                        batch_size=CFG["batch_size"],
                        num_workers=CFG["num_workers"],
                    )
                    print("  Loading checkpoint safely with compatibility mapping...")
                    model, scaler_ts = load_compatible_model(existing_best, device=device)
                    
                    print("  Calibrating temperature...")
                    scaler_ts.calibrate(val_loader, device=device)
                    torch.save(scaler_ts.state_dict(), calibrated_path)
                    
                    all_fold_aucs.append(best_val_auc)
                    continue
            except Exception as e:
                print(f"⚠️ Error loading checkpoint {existing_best}: {e}. Retraining fold.")

        print(f"\n{'='*55}")
        print(f"  FOLD {fold_idx + 1} / {CFG['n_folds']}")
        print(f"  Train: {len(train_df)}  |  Val: {len(val_df)}")
        print(f"{'='*55}")

        train_loader, val_loader = get_dataloaders(
            train_df, val_df,
            batch_size=CFG["batch_size"],
            num_workers=CFG["num_workers"],
        )

        model = FractureDislocModel(pretrained=True).to(device)
        criterion = MultiTaskLoss(
            frac_pos_weight=CFG["frac_pos_w"],
            disl_pos_weight=CFG["disl_pos_w"],
        ).to(device)

        start_epoch = 0
        best_val_auc = 0.0
        patience_cnt = 0

        # Resume state if we crashed mid-fold
        if resume_checkpoint is not None and fold_idx == resume_fold:
            print(f"  >> Resuming Fold {fold_idx+1} from epoch {resume_epoch+1}")
            model.load_state_dict(resume_checkpoint["state_dict"])
            start_epoch = resume_epoch
            best_val_auc = resume_checkpoint["best_auc"]
            patience_cnt = resume_checkpoint["patience"]
            
            optimizer = get_optimizer(model, start_epoch, CFG["lr"], CFG["weight_decay"], CFG["freeze_epochs"])
            optimizer.load_state_dict(resume_checkpoint["optimizer"])
            
            scaler = GradScaler("cuda", enabled=CFG["amp"])
            scaler.load_state_dict(resume_checkpoint["scaler"])
            
            # Reset resume checkpoint after restoring to avoid affecting future folds
            resume_checkpoint = None
        else:
            optimizer = get_optimizer(model, 0, CFG["lr"], CFG["weight_decay"], CFG["freeze_epochs"])
            scaler    = GradScaler("cuda", enabled=CFG["amp"])

        scheduler = WarmupCosineScheduler(optimizer, CFG["warmup_epochs"], CFG["epochs"], CFG["lr"])

        for epoch in range(start_epoch, CFG["epochs"]):
            t0 = time.time()

            # Unfreeze backbone after warmup (if running normally and hitting freeze_epochs threshold)
            if epoch == CFG["freeze_epochs"] and start_epoch < CFG["freeze_epochs"]:
                print("  >> Unfreezing backbone")
                for param in model.image_branch.features.parameters():
                    param.requires_grad = True
                optimizer.add_param_group({
                    "params": model.image_branch.features.parameters(),
                    "lr": CFG["lr"] * 0.1,
                    "weight_decay": CFG["weight_decay"],
                })

            # Train
            train_loss, _, _ = run_epoch(
                model, train_loader, criterion, optimizer, scaler, device, training=True
            )
            scheduler.step(epoch)

            # Validate
            val_loss, frac_m, disl_m = run_epoch(
                model, val_loader, criterion, optimizer, scaler, device, training=False
            )

            combined_auc = (frac_m["auc"] + disl_m["auc"]) / 2

            print(
                f"  Epoch {epoch+1:3d}/{CFG['epochs']}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"comb_AUC={combined_auc:.4f}  LR={scheduler.get_lr():.2e}  "
                f"({time.time()-t0:.1f}s)"
            )

            if combined_auc > best_val_auc:
                best_val_auc = combined_auc
                patience_cnt = 0
                torch.save({
                    "epoch":      epoch,
                    "state_dict": model.state_dict(),
                    "auc":        combined_auc,
                    "cfg":        CFG,
                }, best_path)
                print(f"  ✓ Saved best model  (AUC={combined_auc:.4f})")
            else:
                patience_cnt += 1
                
            # Save latest checkpoint for crash recovery at the end of each epoch
            torch.save({
                "fold":       fold_idx,
                "epoch":      epoch,
                "state_dict": model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "scaler":     scaler.state_dict(),
                "best_auc":   best_val_auc,
                "patience":   patience_cnt,
                "all_aucs":   all_fold_aucs,
            }, latest_path)

            if patience_cnt >= CFG["early_stop"]:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        # Temperature calibration on val set
        print("\n  Calibrating temperature...")
        model.load_state_dict(torch.load(best_path)["state_dict"])
        scaler_ts = TemperatureScaler(model).to(device)
        scaler_ts.calibrate(val_loader, device=device)
        torch.save(scaler_ts.state_dict(), calibrated_path)

        all_fold_aucs.append(best_val_auc)
        print(f"\n  Fold {fold_idx+1} best AUC: {best_val_auc:.4f}")

    # Remove the crash recovery checkpoint upon successful completion of all training
    if os.path.exists(latest_path):
        os.remove(latest_path)
    if os.path.exists(root_latest_path):
        os.remove(root_latest_path)

    print(f"\n{'='*55}")
    print(f"  Cross-Val AUC: {np.mean(all_fold_aucs):.4f} ± {np.std(all_fold_aucs):.4f}")
    print(f"{'='*55}")

if __name__ == "__main__":
    # ── GPU compatibility check ──
    check_gpu_compatibility()

    # ── Load real datasets from Kaggle's mount paths ──
    print("Loading datasets...")
    
    # 1. FracAtlas
    df_fracatlas = load_fractatlas("/kaggle/input/fracatlas/FracAtlas")
    
    # 2. GRAZPEDWRI-DX
    df_graz = load_grazpedwri(
        "/kaggle/input/complete-grazpedwri-dx/Total wrist/images", 
        "/kaggle/input/grazpedwri-dx/dataset.csv"
    )
    
    # Combine datasets
    df = pd.concat([df_fracatlas, df_graz], ignore_index=True)
    print(f"Total dataset shape: {df.shape}")
    
    # Start training!
    train(df)
