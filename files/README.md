# Bone Fracture & Joint Dislocation Detection
## ML Model + FastAPI Deployment

---

## Project Structure

```
fracture_model/
├── model.py        ← EfficientNetV2 + clinical fusion + Grad-CAM++
├── dataset.py      ← CLAHE preprocessing, loaders for each dataset
├── train.py        ← Training loop, metrics, calibration
├── api.py          ← FastAPI REST server
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Training

1. **Prepare your datasets** on Kaggle (FracAtlas, GRAZPEDWRI-DX, RSNA Bone Age)
2. **Edit `train.py`** — uncomment the `load_fractatlas` / `load_grazpedwri` lines and point to your paths
3. Run:

```bash
python train.py
```

Checkpoints saved to `checkpoints/best_fold{N}_calibrated.pt`

---

## Running the API

```bash
MODEL_PATH=checkpoints/best_fold1_calibrated.pt uvicorn api:app --host 0.0.0.0 --port 8000
```

Interactive docs: http://localhost:8000/docs

---

## API Usage (Python client example)

```python
import requests, base64, json

# Single prediction
with open("patient_xray.jpg", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/predict",
        files={"xray": ("xray.jpg", f, "image/jpeg")},
        data={
            "age": 45,
            "sex": 1,
            "bmi": 27.5,
            "mechanism": 1,      # high-energy
            "bone_density": -0.5,
            "prior_fracture": 0,
            "pain_score": 7,
            "generate_heatmap": True,
            "target": "fracture",
        }
    )

result = resp.json()
print(f"Fracture probability:    {result['fracture_probability']:.1%}")
print(f"Dislocation probability: {result['dislocation_probability']:.1%}")
print(f"Fracture risk:           {result['fracture_risk']}")

# Save heatmap image
if result["heatmap_base64"]:
    img_data = base64.b64decode(result["heatmap_base64"])
    with open("heatmap_overlay.jpg", "wb") as f:
        f.write(img_data)
    print("Heatmap saved to heatmap_overlay.jpg")
```

---

## Clinical Input Fields

| Field | Type | Range | Description |
|---|---|---|---|
| `age` | float | 0–120 | Patient age (years) |
| `sex` | int | 0/1 | 0=female, 1=male |
| `bmi` | float | 10–70 | Body Mass Index |
| `mechanism` | int | 0/1/2 | 0=low-energy, 1=high-energy, 2=pathological |
| `bone_density` | float | −5 to 3 | DEXA T-score |
| `prior_fracture` | int | 0/1 | Prior fracture history |
| `pain_score` | float | 0–10 | VAS pain score |

---

## Improvements Checklist

- [ ] Add DICOM (.dcm) support via `pydicom`
- [ ] Train on all 5 datasets jointly with domain adaptation
- [ ] Add joint type classification (wrist / knee / shoulder / spine)
- [ ] Export to ONNX for faster inference
- [ ] Add Explainability report (SHAP for clinical features)
- [ ] Dockerize the API
- [ ] Add Redis caching for repeated predictions
- [ ] Integrate with PACS / hospital EMR system

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| EfficientNetV2 over ResNet | 15% better accuracy, 40% fewer params |
| CLAHE preprocessing | Enhances fracture lines in X-ray contrast |
| Grad-CAM++ over Grad-CAM | More precise localization of small fracture lines |
| Temperature scaling | Calibrated probabilities → clinically meaningful risk scores |
| Stratified K-Fold | Handles class imbalance in fracture datasets |
| Multi-task loss with pos_weight | Prevents model from predicting all-negative on imbalanced data |
| Backbone freeze for 3 epochs | Prevents destroying ImageNet features before clinical head learns |
