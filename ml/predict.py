"""
ml/predict.py
-------------
Handles Model loading and inference for FASTAPI Layer

Key Design: The model is loaded once at startup, not per request
loading a joblib file takes ~200ms. If its done per request assuming
100 requests per second, thats 20 seconds wasted on loading the model everytime
    once its loaded at startup it takes no time for each api

"""

import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from app.config import settings

#__________Feature List must match pipeline.py exactly_______
FEATURE_COLS = [
    "population_log",
    "urban_score",
    "elderly_ratio",
    "gp_count",
    "specialist_density",
    "pharmacy_score",
    "wealth_index",
    "population_growth_rate",
    "avg_gp_age",
    "teleconsult_score",
]

RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}
RISK_COLORS = {0: "#22c55e", 1: "#f59e0b", 2: "#ef4444"}

# Actionable recommendations per risk level
# These come from the French government's official guidelines
RECOMMENDATIONS = {
    0: [
        "Maintain current healthcare infrastructure.",
        "Monitor GP age distribution for upcoming retirements.",
        "Continue telemedicine investment to stay resilient.",
    ],
    1: [
        "Prioritize recruitment of 2+ new GPs within 18 months.",
        "Establish a Maison de Santé Pluriprofessionnelle (MSP).",
        "Expand teleconsultation infrastructure.",
        "Apply for Zone d'Intervention Prioritaire (ZIP) status.",
    ],
    2: [
        "⚠️ URGENT: Apply for Zone Sous-Dense (ZSD) designation.",
        "Request deployment from pacte contre les déserts médicaux.",
        "Deploy mobile medical units for immediate coverage.",
        "Fast-track MSP creation with government subsidies.",
        "Partner with medical schools for rural internship placements.",
    ],
}

# Module level singeltons
# NONE until load_model() is called at API startup
_model = None
_metadata = None

def load_model() -> tuple:
    """
    Loads model pipeline and metadata from disk.
    Called ONCE at FastAPI startup via the lifespan function
    """
    global _model, _metadata

    model_path = Path(settings.model_path)
    meta_path  = Path(settings.metadata_path)

    if not model_path.exists():
            raise FileNotFoundError(f"No model found at {model_path}" "\nRun : python ml/pipeline.py ")

    _model = joblib.load(model_path)

    if meta_path.exists():
        with open(meta_path) as f:
            _metadata = json.load(f)

    model_name = _metadata.get("model_name", "unknown") if _metadata else "unknown"
    accuracy   = _metadata["metrics"]["accuracy"] if _metadata else "?"
    print(f"Model Loaded : {model_name} (accuracy = {accuracy})")
    
    return _model, _metadata


def get_model():
    """
        Returns loaded model + metadata. Loads if not loaded yet
    """
    if _model is None:
          load_model()
    return _model, _metadata 

def predict_one(features: dict) -> dict:
    """
        Runs inference for a single commune.

    Args:
        features: dict with keys matching FEATURE_COLS

    Returns:
        dict with risk_level, risk_label, confidence,
        probabilities, recommendations 
    """

    model, metadata = get_model()

    # Build DataFrame with exact column order used in training
    # Column order matters — wrong order = wrong predictions
    X = pd.DataFrame([{col: features.get(col, 0) for col in FEATURE_COLS }]) 
     
    risk_level = int(model.predict(X)[0])
    proba      = model.predict_proba(X)[0].tolist()
    confidence = round(float(max(proba)), 4)

    return {
        "risk_level":      risk_level,
        "risk_label":      RISK_LABELS[risk_level],
        "risk_color":      RISK_COLORS[risk_level],
        "confidence":      confidence,
        "probabilities": {
            "low":    round(proba[0], 4),
            "medium": round(proba[1], 4),
            "high":   round(proba[2], 4),
        },
        "recommendations": RECOMMENDATIONS[risk_level],
        "model_version":   metadata.get("version", "unknown") if metadata else "unknown",
    }


def predict_batch(records : list[dict]) -> list[dict]:
    """
    Runs inference for multiple communes efficiently.
    Batch prediction is faster than calling predict_one() in a loop
    because the model processes all rows in one matrix operation.
    """
    
    model, metadata = get_model()

    # Build Matrix --- all rows at once 
    X = pd.DataFrame([{col: r.get(col, 0) for col in FEATURE_COLS} for r in records])

    risk_levels  = model.predict(X).tolist()
    probabilities = model.predict_proba(X).tolist()

    return [
        {
            "risk_level":  int(risk),
            "risk_label":  RISK_LABELS[int(risk)],
            "risk_color":  RISK_COLORS[int(risk)],
            "confidence":  round(float(max(proba)), 4),
            "probabilities": {
                "low":    round(proba[0], 4),
                "medium": round(proba[1], 4),
                "high":   round(proba[2], 4),
            },
        }
        for risk, proba in zip(risk_levels, probabilities)
    ]


def get_metadata() -> dict:
    """Returns model metadata dict for the /model/info endpoint."""
    _, metadata = get_model()
    return metadata

if __name__ == "__main__":
    load_model()

    # Test both ends of the spectrum
    rural_creuse = {
        "population_log":         7.1,
        "urban_score":            0,
        "elderly_ratio":          0.38,
        "gp_count":               1,
        "specialist_density":     25.0,
        "pharmacy_score":         1.5,
        "wealth_index":           0.32,
        "population_growth_rate": -0.02,
        "avg_gp_age":             61.0,
        "teleconsult_score":      1.5,
    }

    paris_8e = {
        "population_log":         11.5,
        "urban_score":            3,
        "elderly_ratio":          0.16,
        "gp_count":               85,
        "specialist_density":     180.0,
        "pharmacy_score":         4.8,
        "wealth_index":           0.92,
        "population_growth_rate": 0.02,
        "avg_gp_age":             49.0,
        "teleconsult_score":      4.5,
    }

    print("\n Inference tests:")
    for name, features in [("Rural Creuse", rural_creuse), ("Paris 8e", paris_8e)]:
        result = predict_one(features)
        print(f"\n  {name}:")
        print(f"    Risk    : {result['risk_label']}")
        print(f"    Confidence : {result['confidence']:.1%}")
        print(f"    Probs   : {result['probabilities']}")