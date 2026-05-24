# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Parking lot management system. Computer vision pipeline (PyTorch + OpenCV, built from scratch) reads US/Canada license plates from uploaded images, opens/closes parking sessions, and calculates charges. No external CV APIs — all models are custom-trained on synthetic data.

See `PLAN.md` for the complete architecture, 12-day work plan, and verification checklist.

## Commands

```bash
# Start all services
docker-compose up --build

# Run migrations
docker-compose exec web python manage.py migrate

# Run all tests
docker-compose exec web pytest

# Run tests with coverage
docker-compose exec web pytest --cov=apps/accounts --cov=apps/parking --cov-fail-under=80

# Run a single test file
docker-compose exec web pytest apps/parking/tests/test_services.py

# Run a single test
docker-compose exec web pytest apps/parking/tests/test_services.py::test_calculate_charge_grace_period

# Create superuser
docker-compose exec web python manage.py createsuperuser

# Run image cleanup (dry run)
docker-compose exec web python manage.py cleanup_old_images --dry-run

# Train CV models (run outside Docker, uses MPS on Apple Silicon)
python apps/cv/training/train_detector.py --epochs 50 --data-dir data/detector --output apps/cv/weights/detector.pth
python apps/cv/training/train_recognizer.py --epochs 100 --data-dir data/recognizer --output apps/cv/weights/recognizer.pth
```

## Architecture

### App Boundaries

| App | Owns |
|-----|------|
| `apps.accounts` | Custom `User(AbstractUser)` — no extra fields |
| `apps.parking` | All business logic: sessions, billing, plate normalization, orphan handling, image cleanup |
| `apps.cv` | Neural networks, preprocessing, training scripts, inference pipeline |
| `apps.dashboard` | Views, HTMX partials, API endpoints, template rendering |

### CV Pipeline Flow

```
Image → preprocessing.py → PlateDetectorCNN → crop → PlateRecognizerCRNN → plate_text + confidence
```

- Detector: CNN → [x, y, w, h] bounding box (Smooth L1 loss)
- Recognizer: CNN backbone + Bidirectional LSTM + CTC loss → plate text
- Weights live in `apps/cv/weights/` (gitignored)
- Device auto-detect: MPS → CUDA → CPU (`apps/cv/utils/device.py`)
- CTC loss may need CPU fallback on MPS — training scripts handle this edge case

### Session Logic (`apps/parking/services.py`)

- `handle_entry` / `handle_exit` are the two entry points for all CV-driven events
- Plate normalization: strip whitespace + uppercase only — `"ABC 123"` → `"ABC123"`, hyphens preserved
- Exact match only — `"ABC123"` ≠ `"ABCI23"`
- Orphan handling: if a plate enters while already active → old session voided (`was_orphaned=True`), new session flagged (`has_duplicate_warning=True`)
- Low-confidence events: session IS created with best-guess text, event flagged `is_low_confidence=True`, appears in error queue for manual correction
- All monetary values use `Decimal` — never `float`

### API Endpoints (`apps/dashboard/api.py`)

| Method | URL | Purpose |
|--------|-----|---------|
| POST | `/api/upload/` | Accept image, run CV, create session/event |
| GET | `/api/sessions/` | List sessions (HTMX log page) |
| GET | `/api/dashboard-stats/` | Active count + revenue (polled every 10s) |
| PATCH | `/api/events/<id>/correct/` | Manual plate correction |
| GET | `/api/revenue-data/` | Chart.js data |

### Frontend

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
