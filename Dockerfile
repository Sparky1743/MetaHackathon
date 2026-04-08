FROM python:3.11-slim

WORKDIR /app

# Install dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY models.py .
COPY scenarios.py .
COPY environment.py .
COPY graders.py .
COPY app.py .
COPY inference.py .
COPY openenv.yaml .
COPY README.md .
COPY pyproject.toml .
COPY server/ server/

# Expose HF Spaces default port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s \
  CMD python -c "import requests; r=requests.get('http://localhost:7860/'); assert r.status_code==200"

# Run the server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
