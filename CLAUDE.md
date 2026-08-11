# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Parking lot management system modeled on **Chinese self-service LPR parking**. A public, unmanned **gate kiosk** lets a driver upload a plate image (replacing the camera); the CV pipeline (PyTorch + OpenCV, built from scratch, no external APIs) reads the plate, opens/closes a session, and calculates the charge. **Registered plates are billed automatically to the owner's prepaid wallet** (auto-pay); guests see the amount due at the kiosk. Any user can self-register an account, link plates, and top up a wallet (through a placeholder payment seam). **Staff are reduced to configuration + oversight**, nested under `/staff/`.

See `PLAN.md` for the complete architecture, 12-day work plan, and verification checklist.

## Status

Do not keep a feature checklist in this file. Current work and remaining items live in `PLAN.md` and `docs/superpowers/plans/`. This file is architecture, commands, and rules only.

## Commands

```bash
# Start all services (development — runserver, live code mount)
docker-compose up --build

# Start in production mode (gunicorn, no dev bind mount, port bound to 127.0.0.1)
# Requires Docker Compose >= 2.24 for the !override tags. entrypoint.sh aborts if
# a leaked dev bind mount is detected in a non-debug run.
docker compose -f docker-compose.yml -f docker-compose.prod.yml up --build -d

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

# Run private image cleanup without deleting files
docker-compose exec web python manage.py cleanup_old_images --dry-run

# Refresh knowledge graph after code changes (no API cost)
graphify update .

# Generate synthetic training data locally (backgrounds required — see Training Data below)
python -c "from apps.cv.training.synthetic_data import generate_detector_dataset; generate_detector_dataset(n=1000, output_dir='data/detector', bg_dir='data/backgrounds')"
python -c "from apps.cv.training.synthetic_data import generate_recognizer_dataset; generate_recognizer_dataset(n=5000, output_dir='data/recognizer')"

# Train CV models (run outside Docker, uses MPS on Apple Silicon)
python apps/cv/training/train_detector.py --epochs 50 --data-dir data/detector --output apps/cv/weights/detector.pth
python apps/cv/training/train_recognizer.py --epochs 100 --data-dir data/recognizer --output apps/cv/weights/recognizer.pth
```

## Architecture

### App Boundaries

| App | Owns |
|-----|------|
| `apps.accounts` | Custom `User(AbstractUser)` — no extra fields |
| `apps.parking` | Models (incl. `Wallet`/`WalletTransaction`), admin, `setup_defaults`, session/billing services, wallet money ops (`wallet.py`), payment seam (`payments.py`) |
| `apps.cv` | Preprocessing, custom models, synthetic data, training scripts, and inference pipeline |
| `apps.dashboard` | **Staff-only** (nested at `/staff/`) oversight pages + APIs (dashboard, log, error queue, revenue, settings, image); shared `staff_required`; `scan_core.run_plate_scan` (the reusable image→CV→session core) |
| `apps.public` | **Public** (site root): unmanned gate kiosk (`KioskView`, `kiosk_scan`), resident signup, plate management, wallet + top-up/history, per-IP rate limiter |

### CV Pipeline Flow

**Implemented** (`apps/cv/preprocessing.py`):

```
path → load_image() → bgr_to_rgb() → resize_for_detector(640×480)
     → normalize_pixels() → to_tensor()     # detector input
bbox → crop_plate_region() → prepare_for_recognizer(128×32 gray)  # recognizer input
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
- `predict(x)` wraps `forward()` under `@no_grad`; auto-switches to eval mode and restores prior training state via try/finally — safe to call mid-training
- Trained with `SmoothL1Loss` + Adam + `ReduceLROnPlateau` via `train_detector.py`
- Target: >0.7 IoU on synthetic validation data after 50 epochs

**Implemented** (`apps/cv/models/recognizer.py`):

```
bbox → crop → PlateRecognizerCRNN → plate_text (greedy CTC decode)
```

**Implemented** (`apps/cv/pipeline.py`):

```
image_path → PlateRecognitionPipeline.process() → {plate_text, confidence, bounding_box, is_low_confidence}
```

- `PlateRecognitionPipeline.__init__(detector_path, recognizer_path)` — loads both models via `_load_weights()` helper; raises `FileNotFoundError` (path-stripped) if weights missing, `RuntimeError` (path-stripped) if corrupt
- `process(image_path)` — full pipeline: load → detect → crop → recognise → confidence
- YOLO center bbox `[cx, cy, w, h]` converted to top-left `[x, y, w, h]` in result (matches `PlateDetectionEvent.bounding_box` field)
- `get_pipeline(detector_path, recognizer_path)` — lazy module-level singleton; double-checked locking for thread safety
- `LOW_CONFIDENCE_THRESHOLD = 0.6`; bboxes smaller than 5 % of image → empty plate, confidence 0.0
- `UnsafeImagePathError(ValueError)` raised by `load_image()` on path-traversal attempts (importable from `preprocessing`)

- 3-block CNN (1→64→128→256, final MaxPool(1×2) preserves width) → reshape (B,2048,16) → BiLSTM(hidden=256,layers=2) → FC(512→37) → log_softmax
- `forward()` returns `(T=16, N, C=37)` log-probs — CTC-ready; do NOT re-apply log_softmax
- `predict(x)` wraps `forward()` under `@no_grad`; auto-switches eval mode + restores state via try/finally
- `decode_predictions(output)` — greedy CTC: argmax → collapse repeats → remove blank → `list[str]`
- CTCLoss must run on CPU even on MPS (PyTorch limitation) — `log_probs.cpu()` before loss in training loop
- Weights live in `apps/cv/weights/` (gitignored); load with `torch.load(..., weights_only=True)`
- Targets: >90% char accuracy, >80% full-plate accuracy on synthetic val data after 100 epochs

### Training Data (local, gitignored)

Datasets are generated at runtime, not committed. See `.gitignore` for paths.

| Path | Purpose |
|------|---------|
| `data/backgrounds/` | Curated parking-lot photos for detector compositing (must exist before `generate_detector_dataset`) |
| `data/detector/` | YOLO-format detector set (`images/`, `labels/`) |
| `data/recognizer/` | Recognizer crops + `labels.csv` |

`generate_detector_dataset()` raises if the background directory is missing or has no decodable images, or if the yield falls below 90 % of requested count — it will not silently produce an under-sized dataset.

**Dataset classes** (`apps/cv/training/dataset.py`):

- `PlateDetectorDataset` — image + normalised YOLO bbox `[cx, cy, w, h]`; rejects malformed/out-of-range labels
- `PlateRecognizerDataset` — grayscale crop + encoded label; use `ctc_collate_fn` in DataLoader
- `CHAR_TO_IDX` / `VOCAB_SIZE=37` — shared CTC encoding (blank at index 0)

**Augmentation** (`apps/cv/training/augment.py`): `DetectorAugment`, `RecognizerAugment` — compose with dataset transforms via `torchvision.transforms.v2.Compose([augment, dataset_transform])`.

### DB Integrity (done — migration 0003)

- `LicensePlate`: `UniqueConstraint(user, plate_text)` replacing deprecated `unique_together`
- `ParkingLot.name`: `unique=True`
- `ParkingSession.lot`: `on_delete=PROTECT` (billing records must not cascade-delete)
- `ParkingSession.charge_amount`: `MinValueValidator(Decimal('0.00'))`
- `ParkingSession`: 4 `CheckConstraint`s using `condition=` — charge non-negative, exit after entry, duration non-negative, voided sessions have no charge
- `ParkingSession`: partial indexes `session_active_plate_idx` and `session_active_lot_idx` (`status='active'`)
- `PlateDetectionEvent.confidence_score`: `MinValueValidator(0.0)` + `MaxValueValidator(1.0)`
- `PlateDetectionEvent`: partial index `detection_unreviewed_idx` + confidence range `CheckConstraint`

### Session Logic (done — `apps/parking/services.py`)

Pure business logic; does NOT load CV weights — callers pass extracted detection data. Public functions: `normalize_plate`, `calculate_charge`, `handle_entry`, `handle_exit`, `correct_plate`.

- `handle_entry` / `handle_exit` are the two entry points for all CV-driven events; both wrap their writes in `transaction.atomic()`
- Plate normalization: strip all whitespace + uppercase — `"ABC 123"` → `"ABC123"`, hyphens preserved. Exact match only — `"ABC123"` ≠ `"ABCI23"`
- `calculate_charge`: grace period → $0; `ceil(units) × rate` per minute/hour; optional daily cap. Builds Decimals from integer seconds — never `Decimal(float)`. Unknown `billing_unit` logs + falls back to per-hour
- Orphan handling: re-entry of an active plate voids the old session via one atomic `UPDATE` (`status='void'`, `charge=0`, `was_orphaned=True`); the new session is flagged `has_duplicate_warning=True`
- Confidence is judged against the per-lot `LotSettings.confidence_threshold` (NOT the CV pipeline's fixed constant); below → event `is_low_confidence=True`, session still created
- Exit with no active session → flagged review event (`session=None`, `is_low_confidence=True`), returns `None` (no auto-create, no raise)
- `correct_plate(event_id, corrected_text)`: marks event corrected, updates session `plate_text`, and re-links/clears the registered plate+user. **No authz in the service — the caller (correct API) must enforce staff/operator access**
- Boundary validation: empty/over-length (`> 20`) plate raises `ValueError`; `bounding_box` is sanitized to a 4-float `[0,1]` list (else `[]`); confidence clamped to `[0,1]`
- All monetary values use `Decimal` — never `float`

### Routes & Endpoints

**Public** (`apps/public`, site root — no auth):

| Method | URL | Purpose |
|--------|-----|---------|
| GET | `/` | Gate kiosk shell (entry/exit + plate-image upload) |
| POST | `/kiosk/scan/` | Run CV on the uploaded plate; open/close a session; **privacy-reduced** response (plate + status + charge only — no image URL, event id, owner, or balance). Rate-limited. |
| GET/POST | `/register/` | Resident signup (forces non-staff); provisions a wallet |
| GET | `/post-login/` | Post-login dispatch: staff → `/staff/`, resident → `/wallet/` |
| GET/POST | `/plates/`, `/plates/<id>/delete/` | Manage the logged-in user's own plates (ownership-scoped) |
| GET | `/wallet/` | Balance + transaction history (self only) |
| GET/POST | `/wallet/topup/` | Add funds via the placeholder gateway (rate-limited) |

**Staff** (`apps/dashboard`, nested under `/staff/`, `is_staff` required; `dashboard:` namespace unchanged):

| Method | URL | Purpose |
|--------|-----|---------|
| GET | `/staff/`, `/staff/log/`, `/staff/errors/`, `/staff/revenue/`, `/staff/settings/` | Operator oversight pages |
| GET | `/staff/api/sessions/` , `/staff/api/dashboard-stats/` , `/staff/api/revenue-data/` | HTMX/JSON support |
| PATCH | `/staff/api/events/<id>/correct/` | Correct a queued plate and reconcile its session |
| GET | `/staff/api/events/<id>/image/` | Stream a detection image privately to authenticated staff |

The staff manual-upload endpoint was removed; uploading a plate is now the public
kiosk action, but its hardened core (`scan_core.run_plate_scan`) is unchanged:
uploads are checked by declared MIME type, Pillow structure, and dimensions before
CV decode; capped at 10 MB / 12 MP; saved under randomized names in private
storage. State-changing endpoints use CSRF protection; public endpoints are
rate-limited per IP. Staff use one global `is_staff` role (no per-lot permissions).
Wallet money is `Decimal` only, mutated atomically under `select_for_update`, with
`Wallet.balance == SUM(WalletTransaction.amount)` as an invariant.

**Known product tradeoff of the upload-instead-of-camera model:** anyone can
upload a photo of any plate at the public kiosk, which can open/close a session and
bill that plate's registered wallet. This mirrors a real ANPR gate reading whatever
plate is present; rate limiting blunts abuse. Tighten with device/gate
authentication if the kiosk is exposed beyond a controlled lane.

### Frontend (done — Days 9–10)

Django templates + HTMX + Chart.js. No Node.js, no React.

- `templates/base.html` provides the responsive sidebar, top bar, active navigation,
  queue badge, flash messages, and shared self-hosted HTMX asset.
- `/` shows active sessions, running charges, today's revenue and traffic, and the
  ten most recent events. Its live region polls every 10 seconds.
- `/upload/` supports drag/drop JPEG and PNG uploads, HTMX result swaps, confidence
  bands, and a canvas overlay for the normalized plate bounding box.
- `/log/` provides plate, status, lot, registration, and UTC entry-date filters
  with 25-row pagination and All/Registered/Guest tabs.
- `/errors/` provides the unresolved low-confidence/unmatched queue with private
  thumbnails and inline PATCH correction.
- `/revenue/` renders 7/30/90-day or custom analytics using self-hosted Chart.js,
  including total revenue, sessions, average duration, daily revenue, and lot/hour
  breakdowns.
- `/settings/` validates and saves per-lot billing, grace period, cap, retention,
  and confidence threshold values.
- Confidence display bands are fixed at green `>= 0.8`, yellow `>= 0.6`, red `< 0.6`.
- Every operator page and dashboard API route is restricted to authenticated
  staff accounts. Login and Django authentication routes remain public.

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

1. **Check current official documentation** before using version-sensitive Django,
   PyTorch, HTMX, or Chart.js APIs.
2. **Every function and class needs a comment** explaining the *why*. CV code gets extra-verbose educational comments on every architectural decision.
3. **After a coherent feature/branch diff (not every file):** run `code-reviewer` agent + `security-reviewer` agent + check for unused imports. Fix all CRITICAL and HIGH findings before continuing.
4. **No silent failures** — every error path must log, raise, or return an explicit error.
5. **Frontend:** preserve the imported Claude Design direction and tokens. Extend it
   with Django templates, HTMX, and Chart.js rather than introducing a frontend
   framework.
6. **Coverage gate:** `apps/accounts` and `apps/parking` must stay at ≥80%. `apps/cv` is excluded from the gate.

## graphify

This project has a knowledge graph at `graphify-out/`. Open
`graphify-out/graph.html` in a browser for the interactive tree. Run
`graphify update .` after changes instead of relying on a manually maintained
node/edge count here.

**God nodes (highest connectivity):** `PlateDetectorDataset`, `render_plate_image()`, `PlateRecognizerDataset`, `load_image()`, `RecognizerAugment`, `generate_detector_dataset()`, `composite_on_background()`, `PlateRecognizerCRNN`.

**Key hyperedges:**

- **CV Training Pipeline** — Synthetic Data → Augmentation → Dataset → Model Training
- **CV Inference Chain** — Preprocessing → Detector → Recognizer → Plate Text
- **Session & Event Flow** — Upload API → CV Pipeline → Parking Services → DB Models
- **Operator Dashboard Flow** — Staff page → HTMX/Chart.js endpoint → query builder → parking records

**Named communities:** CV Model Architecture, Synthetic Data & Training, Core Data Models, Session & Billing Logic, Dashboard & Frontend.

Rules:
- For codebase questions, first run `graphify query "<question>"` when `graphify-out/graph.json` exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts.
- If `graphify-out/wiki/index.md` exists, use it for broad navigation instead of raw source browsing.
- Read `graphify-out/GRAPH_REPORT.md` only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
