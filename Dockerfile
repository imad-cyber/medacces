# ─────────────────────────────────────────────────────────────
# Stage 1: Builder
# Installs dependencies and trains the ML model
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# SET WORKING DIRECTORY INSIDE THE CONTAINER
WORKDIR /app

# Install system dependencies needed by XGBoost and psycopg2
# we do this before copying code - docker layer caching means
# this layer is reused unless requirements.txt change

RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy Requirements first ( layer caching optimisation )
# if requirements.txt hasnt changed, docker skips pip install
# and reuses the cached layer - much faster builds
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the application code
COPY . .

# Train the ML Model at runtime
# This means the container starts with a ready model
# No cold-start training delay when the container launches
RUN python -m ml.data_ingestion && python -m ml.pipeline


#________________________________________
# Stage 2: Runtime
# Lean Final image - only whats needed to run the api 
#________________________________________

FROM python:3.11-slim

WORKDIR /app

# Runtime system dependency (PostgreSQL client library)
RUN apt-get update && apt-get install -y \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages

# Copy executables (uvicorn, gunicorn, etc.)
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application + trained model artifacts
COPY --from=builder /app .

# The actual port is provided by the platform via $PORT.
EXPOSE 8000

# Production server command
# Gunicorn manages worker processes
# UvicornWorker handles async FastAPI requests
# Railway injects $PORT — we fall back to 8000 locally
CMD ["sh", "-c", "./deploy/start.sh"]
