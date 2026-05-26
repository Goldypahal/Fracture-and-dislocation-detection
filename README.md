# Multi-Modal 3-Fold Ensemble Bone Fracture & Dislocation Detection

A production-grade, offline clinical decision-support system that integrates a **3-Fold PyTorch Deep Learning Ensemble** (EfficientNetV2) and **Patient Clinical Feature Fusion** to identify bone fractures and joint dislocations. The system is designed to provide calibrated medical predictions, explainability overlays, and quantitative clinical uncertainty boundaries.

---

## 🚀 Key Features

* **3-Fold PyTorch Deep Learning Ensemble**: Integrates predictions from three independent models to reduce variance and eliminate diagnostic blind spots.
* **Clinical Feature Fusion**: Combines spatial X-ray features with tabular clinical indicators (Age, Sex, BMI, Injury Mechanism, Bone Density, prior fractures, and Pain Score) to replicate holistic clinical decision-making.
* **Grad-CAM++ Explainability**: Projects visual attention heatmaps directly over the source X-ray to highlight localized fractures.
* **Empirical Temperature Calibration**: Employs post-hoc probability scaling ($T \approx 1.50$) to ensure predicted probability percentages match true empirical diagnostic rates.
* **Consensus Uncertainty Quantification**: Calculates prediction variance (Standard Deviation) across the 3 independent folds, generating transparent warning flags for low-consensus cases.
* **Premium Offline UI**: A gorgeous dark glassmorphism dashboard served directly from the FastAPI root `/` endpoint, complete with a drag-and-drop uploader and a dynamic Grad-CAM++ blend opacity slider.

---

## 📂 Codebase Directory Structure

```text
├── checkpoints/
│   ├── best_fold1.pt              # Raw PyTorch state-dict (Fold 1)
│   ├── best_fold2.pt              # Raw PyTorch state-dict (Fold 2)
│   ├── best_fold3.pt              # Raw PyTorch state-dict (Fold 3)
│   ├── best_fold1_calibrated.pt   # Calibrated Temperature Scale weights (Fold 1)
│   ├── best_fold2_calibrated.pt   # Calibrated Temperature Scale weights (Fold 2)
│   └── best_fold3_calibrated.pt   # Calibrated Temperature Scale weights (Fold 3)
├── files/
│   ├── api.py                     # FastAPI backend & image decoders
│   ├── model.py                   # Production FusionModel & Legacy Remappers
│   ├── dataset.py                 # Dataset transforms & CLAHE X-ray preprocessing
│   ├── train.py                   # Self-contained training & calibration loops
│   ├── index.html                 # Premium dark-mode HTML dashboard
│   └── predict_client.py          # Interactive REST API client & test utility
└── .gitignore                     # Excludes binary weights from Git tracking
```

---

## 📥 1. Raw Dataset Download Instructions

To train this multi-modal ensemble model natively, you need to acquire the two core open-source clinical datasets:

### Dataset A: FracAtlas Dataset
* **Description**: Contains 4,083 high-resolution X-ray scans annotated for fracture presence and localized bounding boxes.
* **Acquisition**: Download the dataset from [Kaggle FracAtlas](https://www.kaggle.com/datasets/vuppalaadithyasriharsha/fracatlas).
* **Expected Training Folder Path**: `/kaggle/input/fracatlas/FracAtlas`

### Dataset B: GRAZPEDWRI-DX Dataset
* **Description**: A highly detailed pediatric wrist trauma dataset featuring 20,327 images with comprehensive diagnostic classifications.
* **Acquisition**: Download the CSV and images from [Kaggle GRAZPEDWRI-DX](https://www.kaggle.com/datasets/adam1656/grazpedwri-dx) and [Complete GRAZPEDWRI-DX Wrist Images](https://www.kaggle.com/datasets/adam1656/complete-grazpedwri-dx).
* **Expected Training Image Path**: `/kaggle/input/complete-grazpedwri-dx/Total wrist/images`
* **Expected Training CSV Path**: `/kaggle/input/grazpedwri-dx/dataset.csv`

---

## 🏋️ 2. Step-by-Step Model Training & Cross-Validation Guide

The network trains using standard **Stratified 5-Fold Cross Validation** to partition images without leakage.

### System Prerequisites
Ensure your hardware has **CUDA Compute Capability sm_70+** (e.g., NVIDIA T4, L4, RTX 30/40 series, A100). The training execution script automatically verifies your GPU hardware limits before starting.

### Step 1: Install Python Requirements
```bash
pip install torch torchvision opencv-python pydantic requests numpy pandas scikit-learn
```

### Step 2: Run the Cross-Validation Engine
From the repository root folder, start the training pipeline:
```bash
python files/train.py
```

### 🛡️ Built-in Crash-Recovery & Auto-Resume Logic
If your training crashes midway due to Kaggle quota expirations or host environment disconnects, **do not panic!**
* The training script automatically outputs an epoch-level state tracking checkpoint named **`checkpoints/latest_checkpoint.pt`**.
* When you rerun the script, it detects this file, recovers the exact optimizer state, learning rate schedule, loss scaling, and skips already finished folds to **resume training seamlessly from the last successful epoch**.

---

## 🌡️ 3. Temperature Calibration & Reliability Optimization

Raw outputs of neural networks tend to be **overconfident** and clinically unreliable. To solve this, our training pipeline applies post-hoc **Temperature Scaling** using a dedicated validation set split at the end of each fold:

### The Mathematics
A single scalar temperature parameter ($T > 0$) is optimized to scale the logits before passing them to the final sigmoid activation function:
$$\hat{p}_i = \sigma\left(\frac{z_i}{T}\right)$$
This aligns the model's confidence with true statistical accuracy without shifting accuracy boundaries.

### Automatic Calibration Loop
* Once the training of Fold $N$ finishes, `train.py` automatically freezes model parameters and launches an **L-BFGS optimizer** on the validation set.
* It minimizes the Cross-Entropy loss by searching for the perfect scaling coefficient ($T \approx 1.50$).
* The optimized scaling state is saved natively alongside the model in a corresponding calibrated weights file:
  **`checkpoints/best_foldN_calibrated.pt`**

### Manual Retroactive Calibration (For Legacy Checkpoints)
If you have legacy raw weight checkpoints and wish to calibrate them manually, you can execute our helper script:
```bash
python scratch/calibrate_checkpoints.py
```
This processes any raw weights in `checkpoints/` and outputs optimized calibrated parameters ready for medical deployment.

---

## ⚙️ Running the Production API & Console UI

### 1. Launch the Server
To boot the FastAPI server and load all calibrated ensemble folds:
```powershell
$env:PYTHONPATH="files"; uvicorn files.api:app --host 0.0.0.0 --port 8000
```

### 2. Access the Console UI
Simply open your web browser and navigate to:
👉 **[http://localhost:8000/](http://localhost:8000/)**

You can now drag and drop any test image, configure clinical patient inputs, and interactively adjust visual Grad-CAM++ opacities.
