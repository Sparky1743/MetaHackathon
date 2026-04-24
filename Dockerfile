FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY src/ src/
COPY scenarios_seed/ scenarios_seed/
COPY openenv.yaml .
COPY README.md .
COPY pyproject.toml .
COPY .env.example .

ENV PYTHONPATH=/app/src

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s \
  CMD python -c "import requests; r=requests.get('http://localhost:7860/health'); assert r.status_code==200"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
