# =============================================================================
# Cortex — Dockerfile
# =============================================================================
# Multi-stage build:
#   builder  — installs dependencies + builds wheel
#   runtime  — minimal image with only what's needed to run
#
# Build:
#   docker build -t cortex:latest .
#   docker build --build-arg EXTRAS="grpc,redis" -t cortex:grpc-redis .
#
# Run gRPC server:
#   docker run -p 50051:50051 \
#     -e CORTEX_SERVER_PLATFORM_ID=arm_east_01 \
#     -e CORTEX_API_KEYS=cx_live_your_key_here \
#     cortex:latest cortex-serve
#
# Run HTTP server:
#   docker run -p 8765:8765 \
#     -e CORTEX_SERVER_PLATFORM_ID=arm_east_01 \
#     -e CORTEX_API_KEYS=cx_live_your_key_here \
#     cortex:latest cortex-serve-http
# =============================================================================

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Which optional extras to install (override at build time)
ARG EXTRAS="grpc,redis,http"

WORKDIR /build

# Install build tools
RUN pip install --upgrade pip build wheel

# Copy only what's needed for dependency installation first (cache layer)
COPY pyproject.toml setup.py ./
COPY cortex/_version.py cortex/_version.py
COPY cortex/__init__.py cortex/__init__.py

# Install all dependencies into a prefix (not editable — clean wheel)
RUN pip install --prefix=/install ".[${EXTRAS}]" --no-cache-dir

# Now copy the full source and build the wheel
COPY . .
RUN python -m build --wheel --outdir /dist

# Install the wheel on top
RUN pip install --prefix=/install /dist/*.whl --no-cache-dir --no-deps


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Security: run as non-root
RUN groupadd -r cortex && useradd -r -g cortex -s /sbin/nologin cortex

# Copy only the installed packages from builder
COPY --from=builder /install /usr/local

# Runtime directory for config and certs
RUN mkdir -p /etc/cortex /var/cortex/certs && \
    chown -R cortex:cortex /etc/cortex /var/cortex

WORKDIR /app
USER cortex

# ── Ports ─────────────────────────────────────────────────────────────────────
# gRPC server
EXPOSE 50051
# HTTP server
EXPOSE 8765
# Prometheus metrics
EXPOSE 9090

# ── Health check ──────────────────────────────────────────────────────────────
# Uses the cortex-health CLI which exits 0 if healthy, 1 if degraded
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD cortex-health --server http://localhost:8765 --json || exit 1

# ── Environment defaults ──────────────────────────────────────────────────────
# These are safe defaults. Override in docker run / docker-compose / K8s.
ENV CORTEX_SERVER_LOG_FORMAT=json \
    CORTEX_SERVER_LOG_LEVEL=INFO \
    CORTEX_GRPC_HOST=0.0.0.0 \
    CORTEX_GRPC_PORT=50051 \
    CORTEX_HTTP_HOST=0.0.0.0 \
    CORTEX_HTTP_PORT=8765 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ── Default command: gRPC server ──────────────────────────────────────────────
CMD ["cortex-serve"]
