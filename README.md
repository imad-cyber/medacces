MedAccès — AI platform for healthcare access risk across France
MedAccès is an AI-powered, API-first platform that analyzes healthcare access indicators across France and visualizes risk at department and commune level. It combines a production-style FastAPI backend (inference + logging) with a polished Streamlit dashboard for exploration, drill-down, and auditing.

What it does
Predicts medical desert risk for French communes (Low / Medium / High).
Explains risk drivers through analytics and feature-oriented views.
Visualizes national distribution with an interactive, design-forward France map view and department drill-down.
Logs predictions for auditability (monitoring + transparency).
Supports retraining via API trigger (asynchronous background task).
Why it matters
Healthcare access is uneven across territories. MedAccès helps turn public indicators into a clear operational view to support prioritization, planning, and intervention strategy—without burying users in raw data.

Architecture (high level)
Dashboard (Streamlit): product UI, analytics, and operational views
API (FastAPI): inference endpoints, dataset/stats endpoints, model metadata & retraining
ML pipeline: ingestion + feature preparation + training (XGBoost) + artifacts
Storage: dataset + prediction audit log (database-backed)
Product pages (UI)
Overview: value proposition, capabilities, data sources, and system preview
Map: stylized France distribution view + department drill-down panel
APIs: dedicated API documentation pages with structured outputs + charts
Predict: live inference demo (single + batch)
Analytics: dataset exploration and audit log
Model: metrics, feature importance, health checks, retrain trigger
API surfaces (examples)
POST /predict/commune — score a single commune
POST /predict/batch — score multiple communes
GET /communes/stats — aggregated risk distribution
GET /communes/dataset — dataset browser
GET /communes/predictions — prediction audit log
GET /model/info — model metadata + metrics
GET /model/health — service/model health check
POST /model/retrain — trigger background retraining (returns 202)
Tech stack
Backend: FastAPI, Uvicorn/Gunicorn, SQLAlchemy, PostgreSQL (or compatible)
ML: scikit-learn, XGBoost, pandas/numpy, joblib, MLflow
Frontend: Streamlit + Plotly (data visualization)
Local setup
1) Install dependencies
pip install -r requirements.txt
2) Run the API
uvicorn app.main:app --reload
3) Run the dashboard
streamlit run dashboard/app.py
4) (Optional) Train / retrain model locally
python -m ml.data_ingestion
python -m ml.pipeline
Environment variables
Dashboard:

API_URL — URL of the backend API (default: http://localhost:8000)
Backend:

Configure database URL / secrets according to your deployment environment (Railway variables recommended).
Deployment (Railway)
Deploy backend (FastAPI) as one service.
Deploy frontend (Streamlit) as a separate service.
Set API_URL on the frontend service to the backend public URL.
Notes
This repo is built as a portfolio-grade, production-style data product: clear UI hierarchy, API-first design, and audit-ready workflows.
