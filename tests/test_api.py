import os
import sys
import io
import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient

# Ensure the "files" directory is in python search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "files")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api import app

@pytest.fixture(scope="module")
def client():
    """Create a FastAPI TestClient and ensure startup event handlers are executed."""
    with TestClient(app) as c:
        yield c

def test_health_endpoint(client):
    """Verify that /health responds successfully with the system state."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "device" in data
    assert "ensemble_size" in data

def test_dashboard_endpoint(client):
    """Verify that the dashboard serves the HTML page successfully on root path."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "clinical" in response.text.lower() or "diagnosis" in response.text.lower()

def test_predict_endpoint_with_mock_image(client):
    """Verify that multi-modal predictions evaluate successfully with a mock generated X-ray."""
    # Create a 224x224 mock grayscale image using numpy and encode as JPEG bytes
    img = (np.random.rand(224, 224) * 255).astype(np.uint8)
    _, img_encoded = cv2.imencode(".jpg", img)
    img_bytes = io.BytesIO(img_encoded.tobytes())

    # Build multi-part clinical form data
    payload = {
        "age": "45.0",
        "sex": "1",
        "bmi": "25.5",
        "mechanism": "1",
        "bone_density": "-1.5",
        "prior_fracture": "0",
        "pain_score": "7.0",
        "generate_heatmap": "true",
        "target": "fracture"
    }

    files = {
        "xray": ("mock_xray.jpg", img_bytes, "image/jpeg")
    }

    # Execute predictions
    response = client.post("/predict", data=payload, files=files)
    assert response.status_code == 200
    
    data = response.json()
    assert "fracture_probability" in data
    assert "dislocation_probability" in data
    assert "fracture_risk" in data
    assert "dislocation_risk" in data
    assert "model_uncertainty" in data
    uncertainty = data["model_uncertainty"]
    assert "fracture_std" in uncertainty
    assert "dislocation_std" in uncertainty
    assert "consensus_agreement" in uncertainty
    assert "confidence_score_percent" in uncertainty
    
    # Verify values are bound within logical clinical boundaries
    assert 0.0 <= data["fracture_probability"] <= 1.0
    assert 0.0 <= data["dislocation_probability"] <= 1.0
    assert data["fracture_risk"] in ["Low", "Moderate", "High"]
