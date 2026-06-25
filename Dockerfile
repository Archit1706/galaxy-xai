# syntax=docker/dockerfile:1
# Multi-stage build: a builder installs deps into a venv (CPU-only torch to keep
# the image lean and CUDA-free), the runtime stage copies just that venv + src.

# ---------- builder ----------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install the CPU build of torch/torchvision first, from PyTorch's CPU index.
# Doing this before the project install means the project's torch requirement is
# already satisfied and pip won't pull the (huge) default CUDA wheels.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[serve]"

# ---------- runtime ----------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    GALAXYSERVE_WEIGHTS_PATH=/app/models/resnet18_galaxy_best.pth \
    GALAXYSERVE_DEVICE=cpu

# curl is used by the container HEALTHCHECK.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY src ./src
COPY pyproject.toml README.md ./

# Weights and data are mounted at runtime (see docker-compose.yml), not baked in.
USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Single worker keeps the in-process Prometheus registry coherent.
CMD ["uvicorn", "src.service:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
