"""
app/models/schemas.py
---------------------
Pydantic schemas - define the shape of data coming in and out of the api

Rule of thumb :
    - One schema per direction per feature
    - input schemas validate and check what the client sends in
    - Response schema control what the api outputs
    - they dont have to match - and often shouldnt
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

#_________________________________________
# PREDICTION SCHEMAS
#_________________________________________

class CommuneFeatures(BaseModel):
    """
    Input schemas for a single commune prediction

    Every Field has:
    - A type (float, int, str)
    - Validtion constraints enforced using fields (ge, le, gt, lt)
    - A description that shows up in the docs automatically

    Field() lets you add constraints and Documentation
    if a client sends age = -5, elderly ratio = 2.5
    FAST API rejects it BEFORE it even reaches your route function
    """

    # Optional identifiers - not used in ML , kept for display
    commune_code:   Optional[str] = None
    commune_name:   Optional[str] = None
    department_code:Optional[str] = None
    
    #________ ML FEATURES ______________
    population_log: float = Field(
        ..., # ... means required (no default)
        ge = 0,
        description= "log1p of commune population. log1p(500)=6.2, log1p(100000)≈11.5"
    )
    urban_score: float = Field(
        ...,
        ge=0, le=3,
        description="0=rural, 1=peri-urban, 2=small city, 3=urban"
    )
    elderly_ratio: float = Field(
        ...,
        ge = 0.0, le=1.0,
        description="Proportion of population aged 65+, typical range: 0.13 - 0.42"
    )
    gp_count: int = Field(
        ...,
        ge=0,
        decsription = "Number Of general practitioners in the commune"
    )
    specialist_density : float = Field(
        ...,
        ge = 0.0,
        description="Medical specialists per 100,000 inhabitants"
    )
    pharmacy_score : float = Field(
        ...,
        ge=0.0, le=5.0,
        description="Pharmacy access score. 1=no pharmacy within 10km, 5=multiple pharmacies"
    )
    wealth_index: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="Socioeconomic wealth index. 0=deprived, 1=affluent"
    )
    population_growth_rate: float = Field(
        ...,
        description="Annual population growth rate. Negative means declining population"
    )
    avg_gp_age: float = Field(
        ...,
        ge=25.0, le=80.0,
        description="Average age of GPs in the commune. High age = retirement risk"
    )
    teleconsult_score: float = Field(
        ...,
        ge=1.0, le=5.0,
        description="Telemedicine infrastructure score. 1=no coverage, 5=excellent"
    )

    model_config = {
        "json_schema_extra": {
            "example" : {
                "commune_name":           "Ahun",
                "commune_code":           "23001",
                "department_code":        "23",
                "population_log":         7.1,
                "urban_score":            0,
                "elderly_ratio":          0.38,
                "gp_count":              1,
                "specialist_density":     25.0,
                "pharmacy_score":         1.5,
                "wealth_index":           0.32,
                "population_growth_rate": -0.02,
                "avg_gp_age":             61.0,
                "teleconsult_score":      1.5,
            }
        }
    }

class PredictionResponse(BaseModel):
    """
    Output schema for a single prediction.
    Returned to the client after inference.
    """
    risk_level : int
    risk_label : str
    risk_color : str                # hex color for UI
    confidence : float
    probabilities : dict[str, float]    # low, medium, high
    recommendations : list[str]
    model_version : str
    commune_name : Optional[str] = None 
    department_code : Optional[str] = None

    model_config = {"from_attributes": True}

class BatchPredictionRequest(BaseModel):
    """
        input for batch predictions - list of communes    
    """
    communes: list[CommuneFeatures]

    model_config = {
        "json_schema_extra": {
            "example": {
                "communes" : [
                    {
                        "commune_name": "Paris 8e",
                        "department_code": "75",
                        "population_log": 11.5,
                        "urban_score": 3,
                        "elderly_ratio": 0.16,
                        "gp_count": 85,
                        "specialist_density": 180.0,
                        "pharmacy_score": 4.8,
                        "wealth_index": 0.92,
                        "population_growth_rate": 0.02,
                        "avg_gp_age": 49.0,
                        "teleconsult_score": 4.5,
                    }
                ]
            }
        }
    }

class BatchPredictionResponse(BaseModel):
    """Output for Batch Predictions"""
    count : int
    results : list[dict]

#________________________________________________________
# MODEL MANAGEMENT SCHEMAS
#________________________________________________________

class ModelInfoResponse(BaseModel):
    """
    Returned by GET /model/info.
    Everything a stakeholder needs to know about
    the currently deployed model.
    """

    model_name: str
    version: str
    trained_at: str
    features: list[str]
    metrics: dict
    feature_importance: dict[str, float]

class RetrainResponse(BaseModel): 
    """Returned by POST /model/retrain."""
    status: str
    message: str

# ═══════════════════════════════════════════════════════════════
# STATISTICS SCHEMAS
# ═══════════════════════════════════════════════════════════════
class StatsResponse(BaseModel):
    """Aggregated Prediciton statistics"""
    total_predictions: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    avg_confidence: float
    top_at_risk_departments: list[dict]

#_________________________________________________________
# PREDICTION LOG SCHEMA
#_________________________________________________________ 

class PredictionLogResponse(BaseModel):
    """ Single prediction log entry """
    id : int
    commune_name: Optional[str]
    department_code: Optional[str]
    risk_level : int
    risk_label : str
    confidence : float
    created_at : datetime

    model_config = {"from_attributes": True}
