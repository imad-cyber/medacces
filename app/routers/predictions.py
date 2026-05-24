"""
app/routers/predictions.py
--------------------------
Handles all prediction endpoints.

Two endpoints:
- POST /predict/commune → single commune prediction
- POST /predict/batch   → multiple communes at once

Enterprise patterns:
- BackgroundTasks for DB logging (doesn't slow down response)
- HTTPException for proper error codes
- Response logged to DB for audit trail + drift monitoring
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.db_models import PredictionLog
from app.models.schemas import (
    CommuneFeatures,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
)
from ml.predict import predict_one, predict_batch

router = APIRouter(
    prefix="/predict",
    tags=["Predictions"],
)


# ── Helper: log prediction to DB ────────────────────────────────────
def _save_to_db(
    db: Session,
    commune: CommuneFeatures,
    result: dict,
):
    """
    Saves prediction to database for audit and monitoring.

    This runs as a BackgroundTask — AFTER the response is sent.
    The client doesn't wait for DB write to finish.
    Response time stays fast even if DB is slow.

    In production you'd also:
    - Emit a metric to Datadog/Prometheus
    - Push to a data warehouse for retraining
    - Trigger an alert if high-risk confidence > 95%
    """
    log = PredictionLog(
        commune_code    = commune.commune_code,
        commune_name    = commune.commune_name,
        department_code = commune.department_code,

        # Store full feature dict — allows replaying predictions
        input_features  = commune.model_dump(
            exclude={"commune_code", "commune_name", "department_code"}
        ),

        risk_level    = result["risk_level"],
        risk_label    = result["risk_label"],
        confidence    = result["confidence"],
        probabilities = result["probabilities"],
        model_version = result["model_version"],
    )
    db.add(log)
    db.commit()


# ── Endpoint 1: Single Prediction ───────────────────────────────────
@router.post(
    "/commune",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict medical desert risk for one commune",
    description="""
Submit commune features and receive a medical desert risk assessment.

**Risk Levels:**
- **0 — Low**: Adequate GP coverage (≥70% of national average)
- **1 — Medium**: Emerging risk (55-70% of national average)
- **2 — High**: Medical desert — urgent intervention needed (<55%)

Returns confidence scores, class probabilities,
and government-aligned recommendations.
    """,
)
def predict_commune(
    commune: CommuneFeatures,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    The route function receives:
    - commune: validated by Pydantic (CommuneFeatures schema)
    - background_tasks: FastAPI injects this automatically
    - db: provided by Depends(get_db) — see database.py

    Flow:
    1. Extract ML features from commune object
    2. Run model inference
    3. Schedule DB logging as background task
    4. Return result immediately
    """

    # Extract only the ML features (drop identifiers)
    features = commune.model_dump(
        exclude={"commune_code", "commune_name", "department_code"}
    )

    # Run inference
    try:
        result = predict_one(features)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model not loaded. Run python ml/pipeline.py first.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )

    # Log to DB in background — client doesn't wait for this
    background_tasks.add_task(_save_to_db, db, commune, result)

    # Return full response
    return PredictionResponse(
        **result,
        commune_name    = commune.commune_name,
        department_code = commune.department_code,
    )


# ── Endpoint 2: Batch Prediction ────────────────────────────────────
@router.post(
    "/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict risk for multiple communes at once",
    description="""
Batch endpoint for scoring multiple communes in one request.
More efficient than calling /predict/commune in a loop.
Maximum 500 communes per request.
    """,
)
def predict_communes_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    # Guard against huge batches
    if len(request.communes) > 500:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch size limit is 500 communes per request.",
        )

    if len(request.communes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch cannot be empty.",
        )

    # Extract features for all communes
    records = [
        c.model_dump(
            exclude={"commune_code", "commune_name", "department_code"}
        )
        for c in request.communes
    ]

    # Run batch inference — one matrix operation, much faster than loop
    try:
        results = predict_batch(records)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

    # Enrich results with commune identifiers
    for i, result in enumerate(results):
        result["commune_name"]    = request.communes[i].commune_name
        result["commune_code"]    = request.communes[i].commune_code
        result["department_code"] = request.communes[i].department_code

    # Log each prediction in background
    for commune, result in zip(request.communes, results):
        background_tasks.add_task(_save_to_db, db, commune, result)

    return BatchPredictionResponse(
        count=len(results),
        results=results,
    )