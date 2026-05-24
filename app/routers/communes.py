"""
app/routers/communes.py
-----------------------
Data browsing and statistics endpoints.

GET /communes/stats        → aggregate prediction statistics
GET /communes/dataset      → browse the training dataset
GET /communes/predictions  → browse prediction audit log
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Optional
import pandas as pd
from pathlib import Path

from app.database import get_db
from app.models.db_models import PredictionLog
from app.models.schemas import StatsResponse, PredictionLogResponse
from app.config import settings

router = APIRouter(prefix="/communes", tags=["Communes & Statistics"])


#____ API 1 : stats ____________________
@router.get("/stats", response_model=StatsResponse, summary="Aggregate Stats across all predictions")
def get_stats(db : Session = Depends(get_db)):
    """
    Returns a summary of all predictions made so far.
    Useful for dashboards and monitoring

    SQLAlchemy aggregations used here:
    - func.count() -> COUNT(*)
    - func.avg() -> AVG(column)

    These Run in the database - faster than fetching all rows to Python
    """
    total = db.query(PredictionLog).count()

    if total == 0:
        return StatsResponse(
            total_predictions=0,
            high_risk_count=0,
            medium_risk_count=0,
            low_risk_count=0,
            avg_confidence=0.0,
            top_at_risk_departments=[],
        )
    
    high = db.query(PredictionLog).filter(PredictionLog.risk_level==2).count()
    medium = db.query(PredictionLog).filter(PredictionLog.risk_level==1).count()
    low = db.query(PredictionLog).filter(PredictionLog.risk_level==0).count()
    
    avg_conf = db.query(
        func.avg(PredictionLog.confidence)
    ).scalar() or 0.0

    # Departments with highest average risk
    # This is a GROUP BY query - returns one row per department

    dept_risks = (
        db.query(
            PredictionLog.department_code,
            func.count(PredictionLog.id).label("total"),
            func.count(PredictionLog.risk_level).label("avg_risk"),
        )
        .filter(PredictionLog.department_code.is_not(None))
        .group_by(PredictionLog.department_code)
        .order_by(func.avg(PredictionLog.risk_level).desc())
        .limit(10)
        .all()
    )

    return StatsResponse(
        total_predictions=total,
        high_risk_count=high,
        medium_risk_count=medium,
        low_risk_count=low,
        avg_confidence=round(float(avg_conf), 4),
        top_at_risk_departments=[
            {
                "department_code": r.department_code,
                "total_communes":  r.total,
                "avg_risk_score":  round(float(r.avg_risk), 3),
            }
            for r in dept_risks
        ],
    )

@router.get(
    "/dataset",
    summary="Browse The commune training dataset"
)
def get_dataset(
    department: Optional[str] = Query(None, description="Filter by department code e.g. '23'"),
    risk_level: Optional[str] = Query(None, ge=0, le=2, description="0=Low, 1=Medium, 2=High"),
    limit: int = Query(50, ge=0, le=500),
    skip: int = Query(0, ge=0),
):
    """
    Serves the raw training dataset for exploration.
    Supports filtering by department and risk level.
    Paginated with limit + skip.
    
    """
    data_path = Path(settings.data_path)

    if not data_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found. Run python ml/data_ingestion.py",
        )
    
    df = pd.read_csv(data_path)

    # apply filters as request demands
    if department:
        df = df[df["department_code"] == department]
    
    if risk_level is not None:
        df = df[df["medical_desert_risk"] == risk_level]
    
    total = len(df)

    # Apply pagination
    page = df.iloc[skip: skip + limit]

    return {
        "total" : total,
        "returned" : len(page),
        "skip" : skip,
        "limit" : limit,
        "data" : page.to_dict(orient="records"),
    }

@router.get("/predictions", summary="Browse the prediction audit log")
def get_predictions(
    department: Optional[str] = Query(None),
    risk_level: Optional[str] = Query(None, ge=0, le=2),
    limit: int = Query(50,  ge=1, le=200),
    skip:  int = Query(0,   ge=0),
    db: Session = Depends(get_db),
):
    """
    Returns all logged predictions with optional filters.
    Ordered newest first.

    Use this to:
    - See which communes were scored recently
    - Find all High-risk predictions
    - Monitor prediction patterns over time
    """
    query = db.query(PredictionLog)

    if department:
        query = query.filter(PredictionLog.department_code == department)

    if risk_level is not None:
        query = query.filter(PredictionLog.risk_level == risk_level)
    
    total = query.count()

    logs = (
        query.order_by(PredictionLog.created_at.desc()).offset(skip).limit(limit).all()
    )

    return {
        "total":    total,
        "returned": len(logs),
        "predictions": [
            {
                "id":             log.id,
                "commune_name":   log.commune_name,
                "department_code": log.department_code,
                "risk_level":     log.risk_level,
                "risk_label":     log.risk_label,
                "confidence":     log.confidence,
                "created_at":     str(log.created_at),
            }
            for log in logs
        ],
    }

    