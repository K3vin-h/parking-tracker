#!/bin/sh
# entrypoint.sh — wait for Postgres, run migrations, then exec the server command.
#
# WHY this script instead of relying solely on depends_on?
#   depends_on: condition: service_healthy waits for Docker's healthcheck on the db
#   service.  That covers the local docker-compose case.  This script is a safety
#   net for any environment where the depends_on healthcheck is not in effect
#   (Kubernetes, CI runners, plain docker run) so the container never tries to
#   migrate against a Postgres that isn't ready yet.

set -e

# ── Production safety guard: detect a leaked dev bind mount ────────────────────
# docker-compose.prod.yml uses the Compose `!override` tag to drop the dev
# `.:/app` bind mount. On Docker Compose < 2.24 that tag is silently ignored and
# the mount is kept — which would re-expose the host source tree and the
# plaintext .env (SECRET_KEY, DB_PASSWORD) inside the production container.
# .env is excluded from the image by .dockerignore, so /app/.env can only exist
# if that bind mount was applied. In a non-debug (production) run that means the
# override did not take effect: abort loudly instead of serving with leaked
# secrets on an unexpectedly public port.
if [ "${DEBUG:-false}" != "true" ] && [ -f /app/.env ]; then
    echo "ERROR: /app/.env present in a non-debug run — the dev bind mount leaked in." >&2
    echo "       docker-compose.prod.yml needs Docker Compose >= 2.24 for its" >&2
    echo "       !override tags to take effect. Check: docker compose version" >&2
    exit 1
fi

MAX_RETRIES=10
RETRY=0
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"

echo "Waiting for PostgreSQL at ${DB_HOST}:${DB_PORT}..."

# Use psycopg2 (already installed) to test connectivity — avoids needing
# the postgresql-client system package (pg_isready) in the runtime image.
#
# WHY the probe prints its error: with a silent `except: exit(1)` a missing
# psycopg2 package or a wrong DB_PASSWORD was indistinguishable from
# "Postgres not ready yet" — the loop would burn all retries and report a
# generic timeout while the real cause never appeared in the container log.
until python - <<'PYEOF'
import os, sys
try:
    import psycopg2
    psycopg2.connect(
        host=os.environ.get('DB_HOST', 'localhost'),
        port=os.environ.get('DB_PORT', '5432'),
        user=os.environ.get('DB_USER', 'parking_user'),
        password=os.environ.get('DB_PASSWORD', ''),
        dbname=os.environ.get('DB_NAME', 'parking_tracker'),
    ).close()
except Exception as exc:
    print(f"  DB probe failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    sys.exit(1)
PYEOF
do
    RETRY=$((RETRY + 1))
    if [ "$RETRY" -ge "$MAX_RETRIES" ]; then
        echo "ERROR: PostgreSQL not ready after ${MAX_RETRIES} attempts. Giving up."
        exit 1
    fi
    echo "  Not ready yet (attempt ${RETRY}/${MAX_RETRIES}). Retrying in 3s..."
    sleep 3
done

echo "PostgreSQL is ready."

# Apply any pending migrations before starting the server.
# --no-input: never prompt for user input — required for non-interactive startup.
echo "Running migrations..."
python manage.py migrate --no-input

# Collect static files into the shared 'staticfiles' volume so a reverse proxy
# can serve /static/ in production. (gunicorn itself does not serve static, and
# Django only serves it under runserver/DEBUG.) Skipped in development, where
# 'runserver' serves static directly from STATICFILES_DIRS.
#
# WHY at startup and not in the Dockerfile build: 'staticfiles' is a
# runtime-mounted named volume that would shadow anything baked into the image
# at build time, so the collect must happen here, after the volume is mounted.
# set -e is active, so a real collectstatic failure aborts startup loudly
# rather than serving a site with missing CSS/JS.
if [ "${DEBUG:-false}" != "true" ]; then
    echo "Collecting static files..."
    python manage.py collectstatic --no-input
fi

# Replace this shell process with the server process so that Docker's SIGTERM
# goes directly to Django / gunicorn instead of being intercepted by the shell.
exec "$@"
