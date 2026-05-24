"""
app/main.py
-----------
FastAPI application entry point.

Responsibilities:
1. Create the FastAPI app instance
2. Load ML model at startup
3. Create database tables
4. Apply middleware
5. Register all routers
6. Define root endpoints

This file should stay THIN — it wires things together,
it doesn't implement business logic.

"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.config import settings
from app.database import engine
from app.models import db_models
from app.routers import predictions, model, communes

#_____ create db tables at startup _____
# Safe to call repeatedly - skips tables that already exists
db_models.Base.metadata.create_all(bind = engine)


#____ Lifespan ___________
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs startup logic before the API accepts requests,
    and cleanup logic when the server shuts down.

    Startup: load the ML model into memory
    Shutdown: could close connections, flush caches etc.

    Using lifespan instead of @app.on_event("startup")
    because on_event is deprecated in modern FastAPI.
    """

    #_____startup_____
    print("medacces API starting......")

    try:
        from ml.predict import load_model
        load_model()
    except FileNotFoundError:
        print("No trained Model found - run python ml/pipeline.py")
    except Exception as e:
        print(f"Model Load Error : {e}")

    print(f"Environment : {settings.environment}")
    print(f"Database    : connected")
    print(f"API ready   : http://localhost:8000")
    print(f"Docs        : http://localhost:8000/docs")

    yield       # -> API runs here, handling requests

    #______Shutdown__________
    print("Medacces shutting down... :(")


#_____ APP Instance ______
app = FastAPI(
    title="🇫🇷 MedAccès API",
    description=
    """
    ## Medical Desert Risk Predictor for France

Uses machine learning to predict healthcare desert risk
across French communes — helping policymakers prioritize
resource allocation where it's needed most.

### Problem
8+ million French people lack adequate GP access.
87% of France is affected by medical deserts (DREES 2025).

### Solution
XGBoost classifier trained on INSEE demographic data
and DREES healthcare supply features.

### Quick Start
1. `POST /predict/commune` — score a single commune
2. `POST /predict/batch`   — score up to 500 communes
3. `GET  /model/info`      — see model performance
4. `GET  /communes/stats`  — see prediction statistics
    """,
    version = settings.version,
    lifespan=lifespan
)

#___ CORS Middleware ________________
# Without this, browser blocks frontend from calling the api
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials=True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

# ___ Request Timing Middleware ________
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """
    Adds X-Process-Time-Ms header to every response.
    Lets you see how long each request takes.
    Useful for performance monitoring.
    """
    start = time.time()
    response = await call_next(request)
    duration = round((time.time() - start) * 1000, 2 )
    response.headers["X-Process-Time-Ms"] = str(duration)

    return response


# _______ Global Validation Error Handler _________
@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request,
    exc: RequestValidationError,
):
    """
    Makes Pydantic validation errors readable.

    Default FastAPI validation errors are verbose and technical.
    This reformats them into clean, actionable messages.

    Example — instead of:
    {"detail": [{"loc": ["body", "elderly_ratio"], "msg": "..."}]}

    Returns:
    {"detail": "Validation failed", "errors": [{"field": "elderly_ratio", "message": "..."}]}
    """
    errors = [
        {
            "field": str(error["loc"][-1]),
            "message": error["msg"]
        }
        for error   in exc.errors()
    ]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Validation failed",
            "errors": errors,
        }
    )

# ── Global Catch-All Error Handler ──────────────────────────────────
@app.exception_handler(Exception)
async def global_error_handler(request: Request, exc: Exception):
    """
    Catches any unhandled exception.
    Logs it server-side but returns a generic message to the client.

    Why generic? Never expose internal error details to clients.
    Stack traces reveal your architecture to attackers.
    """
    print(f"Unhandled error on {request.method} {request.url}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )

# ___ Register Routes ________
app.include_router(predictions.router)
app.include_router(model.router)
app.include_router(communes.router)

#____ Root Endpoints _________
@app.get("/", tags=["Root"])
def root():
    """API info and navigation."""
    return {
        "name":        "MedAccès API",
        "version":     settings.version,
        "description": "AI-powered medical desert risk predictor for France",
        "endpoints": {
            "docs":     "/docs",
            "predict":  "/predict/commune",
            "batch":    "/predict/batch",
            "model":    "/model/info",
            "health":   "/model/health",
            "stats":    "/communes/stats",
        },
    }

@app.get("/health", tags=["Root"])
def health_check():
    """
    Simple liveness check.
    Load balancers call this to know if the server is alive.
    Returns 200 as long as the process is running.
    """
    return {
        "status":      "ok",
        "environment": settings.environment,
        "version":     settings.version,
    }


