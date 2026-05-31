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
| CV image preprocessing | Done | `apps/cv/preprocessing.py` (~57 tests) |
| Plate detector model (`PlateDetectorCNN`) | Done | `apps/cv/models/plate_detector.py` (17 tests) |
| Plate recognizer model (`PlateRecognizerCRNN`) | Planned | `apps/cv/models/recognizer.py` — not in repo yet |
| Synthetic training data + augmentations + Datasets | Done | `apps/cv/training/synthetic_data.py`, `augment.py`, `dataset.py` |
| Dataset hardening (bbox + background dir validation) | Done | `dataset.py`, `synthetic_data.py` |
| Detector training script (`train_detector.py`) | Done | `apps/cv/training/train_detector.py` |
| Recognizer training script (`train_recognizer.py`) | Planned | `apps/cv/training/` — not in repo yet |
| Session/billing services | Planned | `apps/parking/services.py` — not in repo yet |
| Dashboard views + HTMX UI | Planned | `apps/dashboard/views.py` is placeholder |
| REST API (`/api/upload/`, etc.) | Planned | `apps/dashboard/api.py` — not in repo yet |

Current open PRs:
- **PR #4** — `feat/synthetic-training-data-pipeline` — synthetic data + augmentation + Dataset classes + review hardening
- **PR #5** — `feat/plate-detection-cnn` — `PlateDetectorCNN` model + `train_detector.py` + 17 tests (base: PR #4 branch)

Next: plate recognizer CRNN model + `train_recognizer.py` (Day 5).

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

# CV tests only (~188 tests, excluded from coverage gate)
docker-compose exec web pytest apps/cv/tests/ -v

# Run a single test file
docker-compose exec web pytest apps/parking/tests/test_models.py

# Create superuser
docker-compose exec web python manage.py createsuperuser

# Run image cleanup (dry run) — command planned Day 11
docker-compose exec web python manage.py cleanup_old_images --dry-run

# Refresh knowledge graph after code changes (no API cost)
graphify update .

# Generate synthetic training data locally (backgrounds required — see Training Data below)
python -c "from apps.cv.training.synthetic_data import generate_detector_dataset; generate_detector_dataset('data/backgrounds', 'data/detector', n=1000)"
python -c "from apps.cv.training.synthetic_data import generate_recognizer_dataset; generate_recognizer_dataset('data/recognizer', n=5000)"

# Train CV models (run outside Docker, uses MPS on Apple Silicon)
python apps/cv/training/train_detector.py --epochs 50 --data-dir data/detector --output apps/cv/weights/detector.pth
python apps/cv/training/train_recognizer.py --epochs 100 --data-dir data/recognizer --output apps/cv/weights/recognizer.pth  # not implemented yet
```

## Architecture

### App Boundaries

| App | Owns |
|-----|------|
| `apps.accounts` | Custom `User(AbstractUser)` — no extra fields |
| `apps.parking` | Models, admin, `setup_defaults`; **services/billing** planned in `services.py` |
| `apps.cv` | Preprocessing, device utils, synthetic data, augment, datasets, `PlateDetectorCNN`, `train_detector.py` (done); recognizer model/training/inference pipeline planned |
| `apps.dashboard` | URL config stub; views/API/templates planned Days 8–10 |

### CV Pipeline Flow

**Implemented** (`apps/cv/preprocessing.py`):

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

**Implemented** (`apps/cv/models/plate_detector.py`):

```
… → PlateDetectorCNN → [cx, cy, w, h] bbox (normalised 0–1)
```

- 3-block CNN (conv+BN+ReLU+MaxPool) → `AdaptiveAvgPool2d(4×4)` → FC 2048→256→4
- `forward()` applies sigmoid internally — training and inference share the same output space
- `predict(x)` wraps `forward()` under `@no_grad`; caller sets `model.eval()` for determinism
- Trained with `SmoothL1Loss` + Adam + `ReduceLROnPlateau` via `train_detector.py`
- Target: >0.7 IoU on synthetic validation data after 50 epochs

**Planned** (see `PLAN.md`):

```
bbox → crop → PlateRecognizerCRNN → plate_text + confidence
```

- Recognizer: CNN backbone + Bidirectional LSTM + CTC loss → plate text
- Weights live in `apps/cv/weights/` (gitignored)
- Device auto-detect: MPS → CUDA → CPU (`apps/cv/utils/device.py`)
- CTC loss may need CPU fallback on MPS — recognizer training script will handle this

### Training Data (local, gitignored)

Datasets are generated at runtime, not committed. See `.gitignore` for paths.

| Path | Purpose |
|------|---------|
| `data/backgrounds/` | Curated parking-lot photos for detector compositing (must exist before `generate_detector_dataset`) |
| `data/detector/` | YOLO-format detector set (`images/`, `labels/`) |
| `data/recognizer/` | Recognizer crops + `labels.csv` |
| `data/detector_smoke/` | Small local smoke output (10 samples) |

`generate_detector_dataset()` raises if the background directory is missing or has no decodable images — it will not silently produce an empty dataset.

**Dataset classes** (`apps/cv/training/dataset.py`):

- `PlateDetectorDataset` — image + normalised YOLO bbox `[cx, cy, w, h]`; rejects malformed/out-of-range labels
- `PlateRecognizerDataset` — grayscale crop + encoded label; use `ctc_collate_fn` in DataLoader
- `CHAR_TO_IDX` / `VOCAB_SIZE=37` — shared CTC encoding (blank at index 0)

**Augmentation** (`apps/cv/training/augment.py`): `DetectorAugment`, `RecognizerAugment` — compose with dataset transforms via `torchvision.transforms.v2.Compose([augment, dataset_transform])`.

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

This project has a knowledge graph at `graphify-out/` (**885 nodes, 1178 edges** as of 2026-05-29, built from `c4ca70e`). Open `graphify-out/graph.html` in a browser for the interactive tree.

**God nodes (highest connectivity):** `PlateDetectorDataset`, `render_plate_image()`, `PlateRecognizerDataset`, `load_image()`, `RecognizerAugment`, `generate_detector_dataset()`, `composite_on_background()`.

**Key hyperedges:**

- **CV Training Pipeline** — Synthetic Data → Augmentation → Dataset → Model Training
- **CV Inference Chain** — Preprocessing → Detector → Recognizer → Plate Text (detector/recognizer model nodes are plan-only until implemented)
- **Session & Event Flow** — Upload API → CV Pipeline → Parking Services → DB Models (mostly plan-only)

**Named communities:** CV Model Architecture, Synthetic Data & Training, Core Data Models, Session & Billing Logic, Dashboard & Frontend.

Rules:
- For codebase questions, first run `graphify query "<question>"` when `graphify-out/graph.json` exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts.
- If `graphify-out/wiki/index.md` exists, use it for broad navigation instead of raw source browsing.
- Read `graphify-out/GRAPH_REPORT.md` only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
