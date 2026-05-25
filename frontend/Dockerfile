FROM python:3.11-slim

WORKDIR /app

# Install only what the dashboard needs (keeps image small).
RUN pip install --no-cache-dir \
    streamlit \
    plotly \
    pandas \
    numpy \
    requests

# Copy the Streamlit dashboard into the container as the app entrypoint.
COPY dashboard/app.py /app/app.py

# Streamlit config: bind to 0.0.0.0 and disable usage stats.
RUN mkdir -p /root/.streamlit && printf '%s\n' \
  '[server]' \
  'headless = true' \
  'address = "0.0.0.0"' \
  'port = 8501' \
  'enableCORS = false' \
  'enableXsrfProtection = false' \
  '' \
  '[browser]' \
  'gatherUsageStats = false' \
  '' \
  '[theme]' \
  'base = "dark"' \
  > /root/.streamlit/config.toml

EXPOSE 8501

CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false"]

