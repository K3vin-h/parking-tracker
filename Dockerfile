# ============================================================
# Dockerfile — parking tracker web application (multi-stage)
# ============================================================
# Two-stage build:
#   builder  — installs all Python deps (needs gcc + libpq-dev for compilation)
#   runtime  — lean production image; only runtime system libs, no compiler
#
# WHY multi-stage?
#   gcc and libpq-dev are needed at pip install time (to compile C extensions)
#   but NOT at runtime. Keeping them in the final image adds ~200 MB and
#   unnecessary attack surface. Multi-stage discards the build tools entirely.
# ============================================================


# ── Stage 1: builder ──────────────────────────────────────────────────────────
# Install Python packages with all build tooling present.
# Nothing from this stage reaches production except the installed packages.
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Build-time system dependencies:
#   libpq-dev — PostgreSQL headers needed if any package compiles against libpq
#   gcc       — C compiler for Python packages that include C extensions
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libpq-dev \
        gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create a virtual environment so we can copy a single directory to the runtime stage.
# Using a venv is cleaner than --prefix= because it sets its own PYTHONPATH correctly.
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Copy only requirements first for layer-caching.
# If requirements.txt hasn't changed, Docker reuses this cached pip layer
# even when source code changes — much faster rebuilds.
COPY requirements.txt requirements-dev.txt ./

# Production builds install only runtime dependencies. docker-compose opts into
# requirements-dev.txt so DEBUG=True can load django_extensions locally.
ARG INSTALL_DEV_REQUIREMENTS=false
RUN if [ "$INSTALL_DEV_REQUIREMENTS" = "true" ]; then \
        pip install --no-cache-dir -r requirements-dev.txt; \
    else \
        pip install --no-cache-dir -r requirements.txt; \
    fi


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
# Lean production image — no compiler, no development headers.
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Activate the venv copied from the builder stage.
ENV PATH="/venv/bin:$PATH"

# Runtime-only system libraries:
#   libgl1        — OpenCV links against libGL.so.1 even in headless mode
#   libglib2.0-0  — OpenCV uses GLib threading primitives (GThread) at runtime
#
# Note: gcc and libpq-dev are intentionally absent — not needed at runtime.
# psycopg2-binary bundles its own libpq, so libpq5 is not required either.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from the builder stage.
# This brings in Django, PyTorch, OpenCV, and all other deps
# without copying gcc or development headers.
COPY --from=builder /venv /venv

WORKDIR /app

# Create a non-root user.
# WHY non-root: if an attacker exploits the application, they get a limited
# user account rather than root on the host. This limits the blast radius
# of any container escape or RCE vulnerability.
RUN groupadd -r appgroup \
    && useradd -r -g appgroup -u 1000 appuser \
    && chown -R appuser:appgroup /app

# Copy project source and transfer ownership to appuser in one layer.
COPY --chown=appuser:appgroup . .

# Make the entrypoint script executable.
RUN chmod +x entrypoint.sh

# Drop privileges before running any application code.
USER appuser

# Health check: verify Django is running and the DB connection is alive.
# --start-period=30s: don't count health failures during initial startup.
# The /health/ endpoint returns 200 if DB is reachable, 503 if not.  When
# HEALTH_CHECK_TOKEN is configured, even loopback probes must send it because
# same-host reverse proxies can make public traffic appear to originate from
# 127.0.0.1.
# /health/ is exempted from SECURE_SSL_REDIRECT in settings.py
# (SECURE_REDIRECT_EXEMPT), so this plain-HTTP probe needs no spoofed
# X-Forwarded-Proto header — trusting that header from localhost would
# let any in-container process bypass the SSL redirect.
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c \
        "import os, urllib.request; req = urllib.request.Request('http://localhost:8000/health/'); token = os.environ.get('HEALTH_CHECK_TOKEN', '').strip(); token and req.add_header('X-Health-Check-Token', token); urllib.request.urlopen(req)" \
        || exit 1

# entrypoint.sh waits for Postgres and runs migrations before starting the server.
ENTRYPOINT ["./entrypoint.sh"]

# Default command for local development.
# docker-compose.yml overrides this with the same command explicitly,
# but having a default here makes the image usable standalone.
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
