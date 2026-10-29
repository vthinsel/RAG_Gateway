FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[loader]" \
    && useradd --create-home --uid 10001 appuser \
    && mkdir -p /var/lib/rag-gateway \
    && chown appuser:appuser /var/lib/rag-gateway

USER appuser
EXPOSE 8000
CMD ["uvicorn", "rag_gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
