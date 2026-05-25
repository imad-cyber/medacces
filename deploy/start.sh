#!/usr/bin/env sh
set -eu

# Single entrypoint for Railway multi-service deploys.
# Choose which server to run via SERVICE_TYPE env var:
# - SERVICE_TYPE=frontend -> Streamlit dashboard
# - (default)            -> FastAPI backend

SERVICE_TYPE="${SERVICE_TYPE:-backend}"
PORT="${PORT:-8000}"

if [ "$SERVICE_TYPE" = "frontend" ]; then
  exec streamlit run dashboard/app.py \
    --server.address 0.0.0.0 \
    --server.port "$PORT" \
    --server.headless true \
    --browser.gatherUsageStats false
fi

exec gunicorn app.main:app \
  --workers 2 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind "0.0.0.0:$PORT" \
  --timeout 120 \
  --access-logfile - \
  --error-logfile -

