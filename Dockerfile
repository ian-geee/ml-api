FROM python:3.11-slim

# Evite les .pyc et flush des logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# On copie tout le repo (y compris app/models avec les artefacts déjà entraînés)
COPY . .

EXPOSE 8000
# Render fournit $PORT ; en local ça retombe à 8000 par défaut
CMD ["sh","-c","uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

