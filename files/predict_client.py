#!/usr/bin/env python
"""
Ensemble REST API Client & Visualization Tool
=============================================
This script sends a patient clinical profile and X-ray image to the active 
3-Fold Ensemble API, parses the medical prediction, consensus ratings, 
uncertainty indexes, and automatically decodes and saves the Grad-CAM++ heatmap overlay.

Usage:
    python files/predict_client.py --image path/to/xray.jpg --age 45 --sex 1 --pain 6.5
"""

import os
import argparse
import base64
import requests
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Clinical Ensemble REST Client")
    parser.add_argument("--image", type=str, required=True, help="Path to patient X-ray image (JPG/PNG)")
    parser.add_argument("--url", type=str, default="http://localhost:8000/predict", help="FastAPI prediction URL")
    
    # Clinical patient features
    parser.add_argument("--age", type=float, default=45.0, help="Patient age in years")
    parser.add_argument("--sex", type=int, default=1, choices=[0, 1], help="Patient sex (0=Female, 1=Male)")
    parser.add_argument("--bmi", type=float, default=24.5, help="Patient BMI")
    parser.add_argument("--mechanism", type=int, default=1, choices=[0, 1, 2], help="Injury mechanism (0=Low, 1=Medium, 2=High)")
    parser.add_argument("--bone-density", type=float, default=-1.0, help="T-score bone density (e.g. -1.0 to -2.5 is osteopenic)")
    parser.add_argument("--prior-fracture", type=int, default=0, choices=[0, 1], help="Prior history of fracture (0=No, 1=Yes)")
    parser.add_argument("--pain-score", type=float, default=7.0, help="Clinical pain score on 1-10 scale")
    
    # Visualization options
    parser.add_argument("--target", type=str, default="fracture", choices=["fracture", "dislocation"], help="Grad-CAM target class")
    parser.add_argument("--output", type=str, default="heatmap_result.jpg", help="Filename to save the Grad-CAM overlay")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.image):
        print(f"\n[INFO] Input image '{args.image}' not found.")
        print("       Generating a high-quality mock X-ray scan on the fly for demonstration...")
        try:
            import cv2
            import numpy as np
            # Generate a 256x256 simulated bone-scan (gray gradient with center bone structure)
            img = np.zeros((256, 256, 3), dtype=np.uint8)
            # Gray background
            img[:, :] = [30, 30, 30]
            # Draw a simulated bone shaft in the center
            cv2.ellipse(img, (128, 128), (40, 110), 0, 0, 360, (200, 200, 200), -1)
            cv2.ellipse(img, (128, 128), (35, 100), 0, 0, 360, (230, 230, 230), -1)
            # Simulated fracture line
            cv2.line(img, (98, 120), (158, 130), (40, 40, 40), 3)
            # Add Gaussian blur to simulate X-ray soft tissue glow
            img = cv2.GaussianBlur(img, (5, 5), 0)
            
            # Save it temporarily to allow loading
            cv2.imwrite(args.image, img)
            print(f"       Created simulated X-ray scan: {args.image}")
        except Exception as e:
            print(f"[ERROR] Failed to auto-generate mock image: {e}")
            return
        
    print("\n" + "="*60)
    print("      CLINICAL 3-FOLD ENSEMBLE REST CLIENT & INTERFACE")
    print("="*60)
    
    # Prepared payload
    patient_data = {
        "age": args.age,
        "sex": args.sex,
        "bmi": args.bmi,
        "mechanism": args.mechanism,
        "bone_density": args.bone_density,
        "prior_fracture": args.prior_fracture,
        "pain_score": args.pain_score,
        "generate_heatmap": "true",
        "target": args.target
    }
    
    print("\n[1/3] Preparing Patient Profile...")
    for k, v in patient_data.items():
        if k != "generate_heatmap" and k != "target":
            print(f"  - {k.replace('_', ' ').title()}: {v}")
            
    print(f"\n[2/3] Sending Request to Local REST Server ({args.url})...")
    
    try:
        with open(args.image, 'rb') as f:
            files = {'xray': (os.path.basename(args.image), f, 'image/jpeg')}
            response = requests.post(args.url, files=files, data=patient_data)
            
        if response.status_code != 200:
            print(f"\n[ERROR] Server returned status code {response.status_code}")
            print(f"Details: {response.text}")
            return
            
        res = response.json()
        print("[SUCCESS] API Response received!")
        
        # ── Medical Prediction Report ──
        print("\n" + "="*60)
        print("                   CLINICAL EVALUATION REPORT")
        print("="*60)
        
        # Plain terminal formatting for robust output compatibility
        print(f"  - Fracture Probability : {res['fracture_probability']:.2%} ({res['fracture_risk']} Risk)")
        print(f"  - Joint Dislocation    : {res['dislocation_probability']:.2%} ({res['dislocation_risk']} Risk)")
            
        # ── Ensemble Consensus & Uncertainty ──
        if "model_uncertainty" in res and res["model_uncertainty"]:
            unc = res["model_uncertainty"]
            print("\n  [ENSEMBLE CLINICAL QUANTIFICATION]")
            print(f"    - Consensus Agreement : {unc['consensus_agreement']}")
            print(f"    - Confidence Index    : {unc['confidence_score_percent']:.1f}%")
            print(f"    - Fracture SD         : {unc['fracture_std']:.4f}")
            print(f"    - Dislocation SD      : {unc['dislocation_std']:.4f}")
            print(f"    - Evaluated Folds     : {unc['num_folds_evaluated']} independent networks")
            
        print("="*60)
        
        # ── Decode & Save Grad-CAM Heatmap ──
        if "heatmap_base64" in res and res["heatmap_base64"]:
            print(f"\n[3/3] Decoding Grad-CAM++ Visual Explanation Overlay (Target: {args.target})...")
            
            # Base64 string might have standard data URL header
            b64_data = res["heatmap_base64"]
            if "," in b64_data:
                b64_data = b64_data.split(",")[1]
                
            img_data = base64.b64decode(b64_data)
            
            with open(args.output, "wb") as out_file:
                out_file.write(img_data)
                
            print(f"[OK] Visual explanation saved successfully to: {args.output}")
            print("  -> Open this file to see where the ensemble focused its attention!")
            
        else:
            print("\n[WARNING] No heatmap base64 data returned from the server.")
            
    except requests.exceptions.ConnectionError:
        print("\n[ERROR] Connection refused. Is your FastAPI server running?")
        print("  -> Start it with: uvicorn api:app --host 0.0.0.0 --port 8000")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
