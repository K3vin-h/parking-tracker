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

MAX_RETRIES=10
RETRY=0
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"

echo "Waiting for PostgreSQL at ${DB_HOST}:${DB_PORT}..."

# Use psycopg2 (already installed) to test connectivity — avoids needing
# the postgresql-client system package (pg_isready) in the runtime image.
until python - <<'PYEOF' 2>/dev/null
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
except Exception:
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

# Replace this shell process with the server process so that Docker's SIGTERM
# goes directly to Django / gunicorn instead of being intercepted by the shell.
exec "$@"
