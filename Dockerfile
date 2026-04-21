# ---- Build stage ----
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build deps
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY pyproject.toml ./
COPY crispr_rl/ ./crispr_rl/

# Install package + runtime deps only
RUN pip install --no-cache-dir ".[viz]"

# ---- Runtime stage ----
FROM python:3.10-slim

LABEL maintainer="Rutuja <rutuja@crispr-rl.bio>"
LABEL version="4.0.0"
LABEL description="CRISPR RL v4 — Ayurgenomic CRISPR Intelligence Engine"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source
COPY crispr_rl/ ./crispr_rl/
COPY scripts/ ./scripts/

# Create log directory
RUN mkdir -p /app/logs /app/plots

# Non-root user for security
RUN useradd -m -u 1001 crisprl
RUN chown -R crisprl:crisprl /app
USER crisprl

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "crispr_rl.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
