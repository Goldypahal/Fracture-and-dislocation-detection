"""
Dataset & Preprocessing for Bone Fracture Detection
====================================================
Handles: FracAtlas, GRAZPEDWRI-DX, RSNA datasets
Preprocessing: CLAHE enhancement, normalisation, augmentation
"""

import os
import cv2
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from typing import Tuple, List, Dict, Optional


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
# ImageNet stats — fine for X-rays when using pretrained backbone
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
# TEST-TIME AUGMENTATION (TTA)
# ─────────────────────────────────────────────
def tta_predict(
    model,
    image_path:   str,
    clinical_vec: torch.Tensor,
    device:       str = "cuda",
    n_augments:   int = 5,
) -> Dict[str, float]:
    """
    Average predictions over n_augments random augmentations.
    Significantly boosts robustness on uncertain cases.
    """
    model.eval()
    tta_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    img_np   = load_xray(image_path)
    clinical = clinical_vec.unsqueeze(0).to(device)

    frac_probs, disl_probs = [], []
    with torch.no_grad():
        for _ in range(n_augments):
            img_t = tta_transform(img_np).unsqueeze(0).to(device)
            out   = model(img_t, clinical)
            frac_probs.append(out["fracture_prob"].item())
            disl_probs.append(out["dislocation_prob"].item())

    return {
        "fracture_prob":    float(np.mean(frac_probs)),
        "dislocation_prob": float(np.mean(disl_probs)),
        "fracture_std":     float(np.std(frac_probs)),   # uncertainty estimate
        "dislocation_std":  float(np.std(disl_probs)),
    }


# ─────────────────────────────────────────────
# DATA LOADING HELPERS  (per dataset format)
# ─────────────────────────────────────────────

def load_fractatlas(data_root: str) -> pd.DataFrame:
    """
    FracAtlas: JSON annotations.
    Expected structure:
        data_root/
          images/
          annotations.json  (COCO-like: images[], annotations[])
    """
    ann_path = os.path.join(data_root, "annotations.json")
    with open(ann_path) as f:
        ann = json.load(f)

    id2path  = {img["id"]: os.path.join(data_root, "images", img["file_name"])
                for img in ann["images"]}
    frac_ids = {a["image_id"] for a in ann["annotations"]}

    rows = []
    for img in ann["images"]:
        rows.append({
            "image_path":        id2path[img["id"]],
            "fracture_label":    int(img["id"] in frac_ids),
            "dislocation_label": 0,   # FracAtlas doesn't label dislocations
            # Clinical fields default — fill from real patient CSV if available
            "age": 45, "sex": 0, "bmi": 25.0,
            "mechanism": 1, "bone_density": 0.0,
            "prior_fracture": 0, "pain_score": 5,
        })
    return pd.DataFrame(rows)


def load_grazpedwri(data_root: str, label_csv: str) -> pd.DataFrame:
    """
    GRAZPEDWRI-DX: pediatric wrist X-rays.
    label_csv has columns: filename, fracture (0/1), age_months, sex
    """
    labels = pd.read_csv(label_csv)
    labels["image_path"] = labels["filename"].apply(
        lambda f: os.path.join(data_root, f)
    )
    labels["fracture_label"]    = labels["fracture"].astype(int)
    labels["dislocation_label"] = 0
    labels["age"]           = labels["age_months"] / 12.0   # convert to years
    labels["bmi"]           = 18.0   # pediatric default
    labels["mechanism"]     = 1
    labels["bone_density"]  = 0.0
    labels["prior_fracture"]= 0
    labels["pain_score"]    = 5
    return labels[["image_path","fracture_label","dislocation_label",
                   "age","sex","bmi","mechanism","bone_density",
                   "prior_fracture","pain_score"]]


def build_stratified_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 42):
    """
    Stratified K-Fold splits on fracture label.
    Yields (train_df, val_df) tuples.
    """
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
