"""
\medacces\app\routers\model.py
------------------------------
Model Management endpoints

GET /model/info -> current model metadata + metrics
GET /model/health -> quick healthcheck
GET /model/retrain -> trigger background retraining

"""

from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from app.models.schemas import ModelInfoResponse, RetrainResponse

router = APIRouter(prefix="/model", tags=["Model Management"])

# track retraining state - prevents duplicate runs

_retraining = False

# _____ API 1 : /info ____________


@router.get("/info", response_model=ModelInfoResponse, summary="Get current Model Metadata and performance metrics")
def get_model_info():
    """
    Returns everything about the currently deployed model
    - Which Algorithm won
    - When it was trained
    - Performance metrics
    - Feaature Importance

    This is used by dashboard and ML engineers monitoring the system
    """

    try:
        from ml.predict import get_metadata
        metadata = get_metadata()

        return ModelInfoResponse(
            model_name=metadata["model_name"],
            version=metadata["version"],
            trained_at=metadata["trained_at"],
            features=metadata["features"],
            metrics=metadata["metrics"],
            feature_importance=metadata.get("feature_importance",  {})
        )

    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="No trained model found, Run python ml/pipeline.py",
        )
    
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=str(e)
        )
    

# ___ API 2 : Health _________________________

@router.get("/health", summary="Quick Model Health Check")
def model_health():
    """
    Lightweight check — is the model loaded and ready?
    Used by load balancers and monitoring systems to verify
    the service is operational.

    Returns 200 if healthy, 503 if model not loaded.
    """
    try: 
        from ml.predict import get_metadata
        meta = get_metadata()
        return {
           "status":   "healthy",
            "model":    meta["model_name"],
            "version":  meta["version"],    
            "accuracy": meta["metrics"]["accuracy"],     
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model Not Ready: {str(e)}",
        )

@router.post("/retrain", 
             response_model=RetrainResponse,
             status_code=status.HTTP_202_ACCEPTED,
             summary="trigger model retraining with latest data",
             )
def retrain_model(background_tasks: BackgroundTasks):
    """
    Triggers a full retrain pipeline in the background.

    Returns 202 ACCEPTED immediately — does NOT wait for training.
    Retraining takes 2-5 minutes. Client polls /model/info
    to see when the new version is ready.

    Why 202 and not 200?
    200 = "I did the thing"
    202 = "I accepted your request, I'm working on it"
    Training is async so 202 is semantically correct.
    """
    global _retraining

    if _retraining:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                             detail="Retraining already in progress. Check /model/info for updates.",
        )
    
    def _run_retrain():
        """The Actual Work - runs in the background"""
        global _retraining
        _retraining = True

        try:
            print("background Retraining started .....")
            from ml.data_ingestion import run_ingestion
            from ml.pipeline import run_pipeline
            from ml.predict import load_model

            run_ingestion() # refresh data
            run_pipeline()  # retrain models
            load_model()    # hot-reload new models into memory
            print("Retraining complete - new model is live")
        except Exception as e:
            print(f"Failure -> {str(e)}")
        finally:
            _retraining = False
    
    background_tasks.add_task(_run_retrain)

    return RetrainResponse(
        status="Accepted",
        message=(
            "Retraining started in the background"
            "Poll GET /model/info to see the new version when ready"
        )
    )



