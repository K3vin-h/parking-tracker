# ============================================================
# Dockerfile — parking tracker web application
# ============================================================
# Builds the Django application container.
# Used by docker-compose.yml as the 'web' service.
#
# Build stages (in order):
#   1. Start from a slim Python 3.12 base image
#   2. Install system dependencies needed by OpenCV and psycopg2
#   3. Install Python packages from requirements.txt
#   4. Copy the project source code
# ============================================================

# python:3.12-slim is the official Python image based on Debian bookworm.
# "slim" strips most optional system packages for a smaller image size.
# We re-add only the packages we specifically need below.
FROM python:3.12-slim

# PYTHONDONTWRITEBYTECODE=1:
#   Prevents Python from writing .pyc bytecode files to disk.
#   Inside a Docker container, .pyc files are never reused across runs
#   (the container is ephemeral), so they just waste space.
ENV PYTHONDONTWRITEBYTECODE=1

# PYTHONUNBUFFERED=1:
#   Forces Python's stdout and stderr streams to be unbuffered.
#   Without this, print() and logging output gets buffered — you won't see
#   log lines in 'docker-compose logs' until the buffer flushes (which can
#   be minutes later). Unbuffered = logs appear immediately.
ENV PYTHONUNBUFFERED=1

# Install system dependencies.
# --no-install-recommends: skip optional packages that apt would normally pull in
#   (documentation, development headers, etc.) — keeps the image smaller.
# After installing, remove the apt cache to reduce image layer size.
#
# Packages needed:
#   libpq-dev      — PostgreSQL client library headers and shared lib.
#                    psycopg2-binary bundles its own copy but still needs libpq at runtime.
#   gcc            — C compiler. Some Python packages compile C extensions during pip install.
#   libgl1         — Provides libGL.so.1. opencv-python-headless needs this even in headless mode
#                    because OpenCV links against it for some internal drawing operations.
#   libglib2.0-0   — GLib runtime library. OpenCV uses GLib for threading primitives (GThread).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libpq-dev \
        gcc \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container.
# All subsequent commands run relative to this path.
# The source code is copied here; Django is also run from here.
WORKDIR /app

# Copy requirements.txt BEFORE copying the rest of the source code.
# WHY copy requirements separately first?
#   Docker builds images in layers. Each instruction creates a new layer.
#   Docker caches layers and only rebuilds from the first changed layer onward.
#   If we copy requirements.txt first and install, that pip layer is cached.
#   When you change your application code (but not requirements.txt), Docker
#   reuses the cached pip layer → much faster rebuilds.
COPY requirements.txt .

# Install Python dependencies.
# --no-cache-dir: pip stores a local cache of downloaded packages by default.
#   In a Docker image, this cache is never reused (each build starts fresh),
#   so it just wastes image space. This flag disables it.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container.
# In development, docker-compose.yml mounts .:/app as a volume, which overlays
# this copy with your live source code. Changes to code are reflected immediately
# without rebuilding the image.
COPY . .
