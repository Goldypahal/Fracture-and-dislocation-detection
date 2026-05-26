# Multi-Modal 3-Fold Ensemble Bone Fracture & Dislocation Detection

A production-grade, offline clinical decision-support system that integrates a **3-Fold PyTorch Deep Learning Ensemble** (EfficientNetV2) and **Patient Clinical Feature Fusion** to identify bone fractures and joint dislocations. The system is designed to provide calibrated medical predictions, explainability overlays, and quantitative clinical uncertainty boundaries.

---

## 🚀 Key Features

* **3-Fold PyTorch Deep Learning Ensemble**: Integrates predictions from three independent models to reduce variance and eliminate diagnostic blind spots.
* **Clinical Feature Fusion**: Combines spatial X-ray features with tabular clinical indicators (Age, Sex, BMI, Injury Mechanism, Bone Density, prior fractures, and Pain Score) to replicate holistic clinical decision-making.
* **Grad-CAM++ Explainability**: Backpropagates target gradients to the final convolutional block of the EfficientNetV2 backbone to project visual attention heatmaps directly over the source X-ray.
* **Empirical Temperature Calibration**: Employs post-hoc probability scaling ($T \approx 1.50$) to ensure predicted probability percentages match true empirical diagnostic rates.
* **Consensus Uncertainty Quantification**: Calculates prediction variance (Standard Deviation) across the 3 independent folds, generating transparent warning flags for low-consensus cases.
* **Premium Offline UI**: A gorgeous dark glassmorphism dashboard served directly from the FastAPI root `/` endpoint, complete with a drag-and-drop uploader and a dynamic Grad-CAM++ blend opacity slider.

---

## 🛠️ Technology Stack

* **Core AI**: PyTorch, Torchvision, OpenCV
* **Backend REST API**: FastAPI, Uvicorn, Pydantic
* **Frontend Web Dashboard**: HTML5, Vanilla CSS3 (Glassmorphism design system), Pure ES6 JavaScript

---

## ⚙️ Quick Start Installation

### 1. Clone the Repository & Install Dependencies
Ensure you have Python 3.10+ installed. Open your terminal in the project directory and install the necessary libraries:
```bash
pip install torch torchvision opencv-python fastapi uvicorn pydantic requests numpy pandas
```

### 2. Prepare Checkpoints
Place your trained PyTorch fold checkpoints inside the `checkpoints/` directory:
* `checkpoints/best_fold1_calibrated.pt`
* `checkpoints/best_fold2_calibrated.pt`
* `checkpoints/best_fold3_calibrated.pt`

*(If only raw `.pt` weights are available, the API automatically falls back to an empirical temperature scale $T = 1.50$ during model loading).*

### 3. Launch the API Server
Run the following command from the root directory to boot the FastAPI server and host the interactive dashboard:

**On Windows (PowerShell):**
```powershell
$env:PYTHONPATH="files"; uvicorn files.api:app --host 0.0.0.0 --port 8000
```

**On Linux/macOS (Bash):**
```bash
PYTHONPATH=files uvicorn files.api:app --host 0.0.0.0 --port 8000
```

### 4. Access the Clinical Console
Navigate to your web browser and open:
👉 **[http://localhost:8000/](http://localhost:8000/)**

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
└── .gitignore                     # Excludes binary artifacts from version control
```

---

## 🩺 Clinical Explainability & Uncertainty Ratings

Predictions are accompanied by consensus indicators calculated directly from standard deviation ($\sigma$) thresholds:

* **Strong Consensus** ($\sigma \le 0.08$): High model agreement; extreme prediction confidence.
* **Moderate Consensus** ($0.08 < \sigma \le 0.18$): Standard clinical reliability.
* **Divergent Folds** ($\sigma > 0.18$): High model uncertainty; triggers a visual alert urging manual physician overview.
