# syntax=docker/dockerfile:1.7
# Multi-stage image: build wheels once, ship a lean Jupyter Lab runtime.

ARG PYTHON_VERSION=3.12

# ---------------------------------------------------------------------------
# Stage 1: resolve and build dependencies with uv
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.4.30 /uv /uvx /bin/

WORKDIR /build
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

COPY pyproject.toml README.md LICENSE ./
COPY scripts ./scripts
RUN uv venv /opt/venv \
 && uv pip install --python /opt/venv/bin/python ".[app,notebooks]"

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL org.opencontainers.image.title="NumPyMasterPro" \
      org.opencontainers.image.description="Hands-on NumPy mastery: notebooks, utilities, algorithms from scratch" \
      org.opencontainers.image.source="https://github.com/SatvikPraveen/NumPyMasterPro" \
      org.opencontainers.image.licenses="GPL-3.0-or-later"

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

# Run as an unprivileged user; keep the classic Jupyter-stack home for familiarity.
RUN useradd --create-home --uid 1000 jovyan
WORKDIR /home/jovyan/work

COPY --from=builder /opt/venv /opt/venv
COPY --chown=jovyan:jovyan . .

USER jovyan
EXPOSE 8888 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8888/api', timeout=3)" || exit 1

# Token/password are intentionally empty for a local, login-free lab.
# Override JUPYTER_TOKEN in docker-compose or `docker run -e` to secure it.
CMD ["sh", "-c", "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser \
     --ServerApp.token=\"${JUPYTER_TOKEN:-}\" --ServerApp.password=\"${JUPYTER_PASSWORD:-}\" \
     --ServerApp.root_dir=/home/jovyan/work"]
