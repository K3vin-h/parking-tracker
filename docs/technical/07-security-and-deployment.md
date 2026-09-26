# Security and Deployment

| Area | Protection |
| :--- | :--- |
| **Access control** | Every operator page and `/staff/api/` endpoint requires an authenticated staff account (`is_staff = True`). Public kiosk, registration, plate, and wallet routes are the only unauthenticated surface. Login and Django auth routes remain public. |
| **Image uploads** | Declared MIME type, Pillow structure, and format checks run before any CV decode (`scan_core.run_plate_scan`, shared by the kiosk and formerly by staff upload). Uploads are capped at 10 MB (compressed) and 12 MP (pre-decode). Files are saved under randomized names in private storage. |
| **CV image decode** | `load_image()` is a second, CV-side security boundary on the same public upload — path containment under `MEDIA_ROOT`, content-based format allowlist, a 12 MP decompression-bomb cap, a single bounded read, and path-stripped errors. Full control list in [CV Pipeline → Image Preprocessing](01-cv-pipeline.md#image-preprocessing); rationale in [Preprocessing as a Security Boundary](03-design-rationale.md#preprocessing-as-a-security-boundary). |
| **Plate images** | Never served via public `MEDIA_URL`, and never returned by the public kiosk response at all. Only accessible through the authenticated `GET /staff/api/events/<id>/image/` endpoint, which validates the stored path (must start with `plates/`, no `..`, extension allowlist) and sets `Cache-Control: private, no-store` on every response. The reverse proxy or object-storage bucket must also keep the backing media directory private. |
| **Kiosk activation** | `POST /kiosk/activate/` exchanges the server-held `KIOSK_ACTIVATION_TOKEN` and a lane scope for a short-lived, revocable browser capability; the scan endpoint requires that capability plus a single-use nonce rather than trusting the token directly on every request. |
| **Public rate limiting** | Cache-based per-IP limiter (`apps/public/ratelimit.py`) applied to kiosk activation, kiosk scanning, wallet top-up, login, and password reset — bounding both credential-guessing and plate-scan abuse. |
| **State-changing endpoints** | CSRF protection on all forms and PATCH endpoints. `correct_event` additionally uses `select_for_update()` inside `transaction.atomic()` to prevent concurrent double-correction. Wallet debits/credits are similarly atomic and row-locked (`apps/parking/wallet.py`), with `Wallet.balance == SUM(WalletTransaction.amount)` as an invariant. |
| **Injection** | Parameterized Django ORM throughout — no raw SQL. Revenue date inputs parsed with `date.fromisoformat()` (raises `ValueError` on bad input → HTTP 400). Lot IDs cast to `int()` before ORM lookup. |
| **Secrets** | All secrets via environment variables or a host `.env` file. `.dockerignore` excludes `.env` from the image build so secrets are never baked in. |
| **Content Security Policy** | A production CSP header is enforced via `django-csp` with `script-src 'self'` and no `unsafe-eval` or `unsafe-inline`. HTMX's `allowEval` and `allowScriptTags` options are disabled to align with this policy. |
| **HTTPS / transport** | HSTS, secure cookies, and SSL redirect are enabled in the production settings. |
| **Production deployment** | Gunicorn runs behind a reverse proxy. Port 8000 is bound to host loopback only (`127.0.0.1`). The dev source bind mount is dropped in the production Compose override. A startup guard in `entrypoint.sh` aborts the container if `/app/.env` is present in a non-debug run, detecting a silently failed bind-mount drop. |

**Known product tradeoff:** the kiosk accepts an uploaded photo of any plate
rather than reading a live camera feed, so anyone at the kiosk can upload a
photo of any plate and open/close a session and bill that plate's registered
wallet. This mirrors a real ANPR gate, which reads whatever plate is
physically present in the lane; the activation-capability requirement and
per-IP rate limiting blunt casual abuse but do not eliminate it. Tighten with
device- or gate-level authentication if the kiosk is exposed beyond a
controlled, physically-gated lane.
## Docker

The application runs as two containers orchestrated by Docker Compose:

| Container | Description |
| :--- | :--- |
| `db` | PostgreSQL 16 with a persistent named volume |
| `web` | Django application server |

### Development

```bash
# Start all services (Django runserver with live code mount)
docker-compose up --build

# Run migrations
docker-compose exec web python manage.py migrate

# Seed initial data — creates the default ParkingLot and LotSettings (safe to run repeatedly)
docker-compose exec web python manage.py setup_defaults

# Create an admin user
docker-compose exec web python manage.py createsuperuser

# Run the test suite with coverage gate
docker-compose exec web pytest --cov=apps/accounts --cov=apps/parking --cov-fail-under=80
```

### Production

The base `docker-compose.yml` targets local development (runserver, live code mount). For production, layer `docker-compose.prod.yml` on top — it swaps in Gunicorn, drops the dev source bind mount, and publishes port 8000 on host loopback only (`127.0.0.1`) so you must front it with a reverse proxy:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up --build -d
```

The production override also runs `collectstatic` at startup via `entrypoint.sh`.

> **Requires Docker Compose ≥ 2.24.** The override uses the `!override` YAML tag to drop the development bind mount. On older Compose versions this tag is ignored and the mount is silently kept — re-exposing host source and `.env` inside the container. Verify with `docker compose version` before deploying.

**Startup guard:** `entrypoint.sh` aborts the container if `/app/.env` is present in a non-debug run, detecting a silently failed bind-mount drop before the server accepts traffic.

## See also

- [00-architecture.md](00-architecture.md): system overview and module boundaries
- [08-limitations.md](08-limitations.md): known constraints and roadmap
