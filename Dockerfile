FROM python:3.12-slim

WORKDIR /app

# System deps: libpq for psycopg, build tools for any source builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY core/ core/
COPY ingestion/ ingestion/
COPY evaluation/ evaluation/
COPY monitoring/ monitoring/
COPY api/ api/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
