# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Parking lot management system. Computer vision pipeline (PyTorch + OpenCV, built from scratch) reads US/Canada license plates from uploaded images, opens/closes parking sessions, and calculates charges. No external CV APIs — all models are custom-trained on synthetic data.

See `PLAN.md` for the complete architecture, 12-day work plan, and verification checklist.

## Implementation Status

| Area | Status | Location |
|------|--------|----------|
| Django project + Docker + PostgreSQL | Done | `config/`, `docker-compose.yml` |
| Data models (User, plates, sessions, events, lot settings) | Done | `apps/parking/models.py`, `apps/accounts/models.py` |
| Admin + model tests + auth tests | Done | `apps/*/admin.py`, `apps/*/tests/` |
| Seed data command | Done | `apps/parking/management/commands/setup_defaults.py` |
| CV device auto-detection | Done | `apps/cv/utils/device.py` |
| CV image preprocessing | Done | `apps/cv/preprocessing.py` (58 tests) |
| Plate detector / recognizer models | Planned | `PLAN.md` — not in repo yet |
| Training scripts + synthetic data | Planned | `apps/cv/training/` — not in repo yet |
| Session/billing services | Planned | `apps/parking/services.py` — not in repo yet |
| Dashboard views + HTMX UI | Planned | `apps/dashboard/views.py` is placeholder |
| REST API (`/api/upload/`, etc.) | Planned | `apps/dashboard/api.py` — not in repo yet |

Current branch focus: **none** — `feat/cv-image-preprocessing` is complete and pending merge into `feat/django-project-foundation-docker-postgresql-models`.

## Commands

```bash
# Start all services
docker-compose up --build

# Run migrations
docker-compose exec web python manage.py migrate

# Seed default lot + LotSettings (required before using the app)
docker-compose exec web python manage.py setup_defaults

# Run all tests
docker-compose exec web pytest

# Run tests with coverage (accounts + parking gate only)
docker-compose exec web pytest --cov=apps/accounts --cov=apps/parking --cov-fail-under=80

# CV tests only (excluded from coverage gate)
docker-compose exec web pytest apps/cv/tests/ -v

# Run a single test file
docker-compose exec web pytest apps/parking/tests/test_models.py

# Create superuser
docker-compose exec web python manage.py createsuperuser

# Run image cleanup (dry run) — command planned Day 11
docker-compose exec web python manage.py cleanup_old_images --dry-run

# Refresh knowledge graph after code changes (no API cost)
graphify update .

# Train CV models (run outside Docker, uses MPS on Apple Silicon) — not implemented yet
python apps/cv/training/train_detector.py --epochs 50 --data-dir data/detector --output apps/cv/weights/detector.pth
python apps/cv/training/train_recognizer.py --epochs 100 --data-dir data/recognizer --output apps/cv/weights/recognizer.pth
```

## Architecture

### App Boundaries

| App | Owns |
|-----|------|
| `apps.accounts` | Custom `User(AbstractUser)` — no extra fields |
| `apps.parking` | Models, admin, `setup_defaults`; **services/billing** planned in `services.py` |
| `apps.cv` | Preprocessing + device utils (done); models/training/inference planned |
| `apps.dashboard` | URL config stub; views/API/templates planned Days 8–10 |

### CV Pipeline Flow

**Implemented today** (`apps/cv/preprocessing.py`):

```
path → load_image() → bgr_to_rgb() → resize_for_detector(640×480)
     → normalize_pixels() → to_tensor()     # detector input (planned)
bbox → crop_plate_region() → prepare_for_recognizer(128×32 gray)  # recognizer input (planned)
```

Public functions: `load_image`, `bgr_to_rgb`, `resize_for_detector`, `normalize_pixels`, `to_tensor`, `crop_plate_region`, `prepare_for_recognizer`.

**Security constraints on `load_image()`:**
- Resolved path must stay under `MEDIA_ROOT` (`_assert_safe_path`)
- Pillow header inspect before decode; formats JPEG/PNG/WEBP only
- Max 12 MP (`4000×3000`); rejects decompression bombs and uninspectable headers
- OpenCV decode only after validation; generic `FileNotFoundError` to callers (no path leaks)

**Planned** (see `PLAN.md`):

```
… → PlateDetectorCNN → crop → PlateRecognizerCRNN → plate_text + confidence
```

- Detector: CNN → `[x, y, w, h]` bounding box (Smooth L1 loss)
- Recognizer: CNN backbone + Bidirectional LSTM + CTC loss → plate text
- Weights live in `apps/cv/weights/` (gitignored)
- Device auto-detect: MPS → CUDA → CPU (`apps/cv/utils/device.py`)
- CTC loss may need CPU fallback on MPS — training scripts will handle this

### Session Logic (planned — `apps/parking/services.py`)

Specified in `PLAN.md`; not implemented yet. When built:

- `handle_entry` / `handle_exit` are the two entry points for all CV-driven events
- Plate normalization: strip whitespace + uppercase only — `"ABC 123"` → `"ABC123"`, hyphens preserved
- Exact match only — `"ABC123"` ≠ `"ABCI23"`
- Orphan handling: if a plate enters while already active → old session voided (`was_orphaned=True`), new session flagged (`has_duplicate_warning=True`)
- Low-confidence events: session IS created with best-guess text, event flagged `is_low_confidence=True`, appears in error queue for manual correction
- All monetary values use `Decimal` — never `float`

### API Endpoints (planned — `apps/dashboard/api.py`)

| Method | URL | Purpose |
|--------|-----|---------|
| POST | `/api/upload/` | Accept image, run CV, create session/event |
| GET | `/api/sessions/` | List sessions (HTMX log page) |
| GET | `/api/dashboard-stats/` | Active count + revenue (polled every 10s) |
| PATCH | `/api/events/<id>/correct/` | Manual plate correction |
| GET | `/api/revenue-data/` | Chart.js data |

### Frontend (planned)

Django templates + HTMX + Chart.js. No Node.js, no React.

- HTMX polling every 10s on dashboard for active sessions and running costs
- Upload result swapped in via HTMX partial response
- All pages behind `@login_required` / `LoginRequiredMixin` — no public routes

### Design Tokens

```css
--bg-primary: #0f1117;  --bg-secondary: #1a1d27;  --bg-tertiary: #252832;
--text-primary: #e4e4e7;  --text-secondary: #a1a1aa;
--accent-blue: #3b82f6;
--status-active: #22c55e;  --status-warning: #eab308;
--status-error: #ef4444;  --status-void: #6b7280;
--font-mono: 'JetBrains Mono', monospace;
```

## Rules

1. **Check Context7** before using any Django, PyTorch, HTMX, or Chart.js API — never assume training-data knowledge is current.
2. **Every function and class needs a comment** explaining the *why*. CV code gets extra-verbose educational comments on every architectural decision.
3. **After every file:** run `code-reviewer` agent + `security-reviewer` agent + check for unused imports. Fix all CRITICAL and HIGH findings before continuing.
4. **No silent failures** — every error path must log, raise, or return an explicit error.
5. **Frontend:** generate a Claude Design prompt first, paste into Claude Design, then copy the output exactly. Do not invent UI.
6. **Coverage gate:** `apps/accounts` and `apps/parking` must stay at ≥80%. `apps/cv` is excluded from the gate.

## graphify

This project has a knowledge graph at `graphify-out/` (596 nodes, 717 edges as of 2026-05-26). Open `graphify-out/graph.html` in a browser for the interactive tree.

**God nodes (highest connectivity):** `load_image()`, `make_bgr_image()`, `make_rgb_image()`, preprocessing helpers, core Django models.

**Key hyperedge:** CV Inference Chain — Preprocessing → Detector → Recognizer → Plate Text (detector/recognizer nodes are plan-only until models land).

Rules:
- For codebase questions, first run `graphify query "<question>"` when `graphify-out/graph.json` exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts.
- If `graphify-out/wiki/index.md` exists, use it for broad navigation instead of raw source browsing.
- Read `graphify-out/GRAPH_REPORT.md` only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
