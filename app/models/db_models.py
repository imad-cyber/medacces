from sqlalchemy import (
    Column, String, Integer, Float, Boolean, JSON, TIMESTAMP
)
from sqlalchemy.sql import func
from app.database import Base

class PredictionLog(Base):
    """
    Every prediction the API makes gets logged here.

    Why log predictions?
    - Audit trail (who predicted what, when)
    - Monitor model drift (is accuracy dropping over time?)
    - Business analytics (which departments request most)
    - Debugging (reproduce any prediction from its inputs)

    This is standard practice in production ML systems.
    """

    __tablename__ = "prediciton_logs"
    
    #--Identity-------
    id = Column(Integer, primary_key=True, nullable=False)

    #---commune identifiers-----
    commune_code    = Column(String,   nullable=True)
    commune_name    = Column(String,   nullable=True)
    department_code = Column(String,   nullable=True)

    # --- what went into the model ----
    # JSON column stores the full feature dict
    # lets us rerun any prediction exactly as it was
    input_features = Column(JSON,   nullable=False)

    # --- what came out of the model ----
    risk_level = Column(Integer, nullable=False)
    risk_label = Column(String, nullable=False)
    confidence= Column(Float, nullable=False)
    probabilities= Column(JSON, nullable=False)

    # --- Metadata ----
    model_version = Column(String, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)

class Commune(Base):
    """
    Stores commune data + their latest risk assessment.

    This table is populated when we bulk-score all communes.
    The /communes endpoint reads from here — fast, no ML needed.
    """
    __tablename__ = "communes"
    
    id = Column(Integer, primary_key=True, nullable=False)
    commune_code = Column(String, unique = True, nullable=False)
    commune_name = Column(String, nullable=False)
    department_code = Column(String, nullable = False)
    region_code = Column(String, nullable=True)
    population = Column(Integer, nullable=True)

    # latest ML prediciton
    risk_level = Column(Integer, nullable=True)
    risk_label = Column(String, nullable=True)
    confidence = Column(Float, nullable=True)

    # key features stored for display
    gp_density_per_100k = Column(Float, nullable=True)
    elderly_ratio = Column(Float, nullable=True)
    urban_score = Column(Integer, nullable=True)

    last_scored_at = Column(TIMESTAMP(timezone=True), nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

