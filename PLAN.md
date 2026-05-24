# Parking Tracker — License Plate Detection System

## Context

A web application for parking lot operators that uses computer vision to read license plates from uploaded images. The system opens a parking session when a car enters, closes it when the car exits, and calculates the charge based on admin-configurable billing rules. The CV pipeline is built from scratch in PyTorch as a learning exercise — every architectural decision is explained in plain language with detailed comments.

The initial workflow is manual screenshot upload via the dashboard. The upload endpoint is designed so a real camera feed can call it later without code changes.

---

## Authentication

| Rule | Detail |
|------|--------|
| Library | `django.contrib.auth` (built-in) |
| User model | Custom `User` extending `AbstractUser` (no extra fields) |
| Access control | Every dashboard URL requires `@login_required` or `LoginRequiredMixin` |
| Role model | Single role — `is_staff=True` grants full access |
| Login page | Styled to match the dark dashboard theme (designed Day 9) |
| Logout | Redirects to `/login/` |
| Public pages | None — all routes redirect unauthenticated users to `/login/` |

---

## Global Rules

These apply to **every file written**, every day:

1. **Check Context7** for current library API before writing code — never trust training-data assumptions about Django, PyTorch, HTMX, or Chart.js APIs.
2. **Detailed simple-language comments** on every function, class, model, and non-obvious line. CV code gets extra-verbose educational comments explaining *why* each architectural decision was made.
3. **After every file:** run `code-reviewer` agent + `security-reviewer` agent + check for unused imports / dead code. Fix all CRITICAL and HIGH findings before moving on.
4. **No silent failures** — every error path must log, raise, or return an explicit error.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend framework | Django 5.x (Python 3.11+) |
| Database | PostgreSQL 16 |
| Computer vision | PyTorch + OpenCV |
| Frontend | Django templates + HTMX + Chart.js |
| Containers | Docker Compose |
| Testing | pytest + pytest-django |
| CV training hardware | Apple Silicon MPS (auto-detect: MPS → CUDA → CPU) |

No Node.js. No Flask. No external CV APIs. Everything runs in Python.

---

## Architecture

### Directory Structure

```
parking tracker/
├── PLAN.md
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
├── manage.py
├── pytest.ini
├── config/
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── apps/
│   ├── accounts/
│   │   ├── __init__.py
│   │   ├── apps.py
│   │   ├── models.py
│   │   ├── admin.py
│   │   ├── tests/
│   │   │   ├── __init__.py
│   │   │   ├── test_models.py
│   │   │   └── test_auth.py
│   │   └── migrations/
│   │       └── __init__.py
│   ├── parking/
│   │   ├── __init__.py
│   │   ├── apps.py
│   │   ├── models.py
│   │   ├── services.py        # entry/exit/billing logic
│   │   ├── utils.py           # plate normalization
│   │   ├── admin.py
│   │   ├── management/
│   │   │   └── commands/
│   │   │       └── cleanup_old_images.py
│   │   ├── tests/
│   │   │   ├── __init__.py
│   │   │   ├── test_models.py
│   │   │   ├── test_services.py
│   │   │   ├── test_utils.py
│   │   │   └── test_cleanup.py
│   │   └── migrations/
│   │       └── __init__.py
│   ├── cv/
│   │   ├── __init__.py
│   │   ├── apps.py
│   │   ├── preprocessing.py
│   │   ├── pipeline.py
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   └── device.py      # MPS/CUDA/CPU auto-detect
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── plate_detector.py
│   │   │   └── recognizer.py
│   │   ├── training/
│   │   │   ├── __init__.py
│   │   │   ├── synthetic_data.py
│   │   │   ├── dataset.py
│   │   │   ├── augment.py
│   │   │   ├── train_detector.py
│   │   │   └── train_recognizer.py
│   │   └── weights/
│   │       ├── .gitkeep
│   │       ├── detector.pth   # gitignored
│   │       └── recognizer.pth # gitignored
│   └── dashboard/
│       ├── __init__.py
│       ├── apps.py
│       ├── views.py
│       ├── api.py
│       ├── urls.py
│       └── tests/
│           ├── __init__.py
│           ├── test_views.py
│           └── test_api.py
├── templates/
│   ├── base.html
│   ├── registration/
│   │   └── login.html
│   ├── dashboard.html
│   ├── upload.html
│   ├── log.html
│   ├── errors.html
│   ├── revenue.html
│   └── settings.html
├── static/
│   ├── css/
│   │   └── main.css
│   └── js/
│       └── main.js
└── media/                     # uploaded plate images (gitignored)
    └── plates/
```

### App Responsibilities

| App | Responsibility |
|-----|---------------|
| `apps.accounts` | Custom User model, authentication |
| `apps.parking` | ParkingLot, LotSettings, ParkingSession, LicensePlate, PlateDetectionEvent, billing logic, plate normalization, image cleanup |
| `apps.cv` | Neural network models, preprocessing, training scripts, inference pipeline, synthetic data generation |
| `apps.dashboard` | Views, API endpoints, template rendering, HTMX partials |

---

## Database Models

All models live in their respective app's `models.py`. Relationships and constraints are defined exactly as specified below.

### `apps/accounts/models.py`

```python
class User(AbstractUser):
    """
    Custom user model extending Django's AbstractUser.
    No extra fields needed — guests are represented by session.user=null,
    not by a special user flag.
    Defined here so AUTH_USER_MODEL can point to it, allowing future
    extension without migrations headaches.
    """
    pass
```

### `apps/parking/models.py`

```python
class LicensePlate:
    user            = ForeignKey(User, on_delete=CASCADE, related_name='plates')
    plate_text      = CharField(max_length=20)   # stored normalized: stripped, uppercase
    is_primary      = BooleanField(default=False)
    label           = CharField(max_length=100, blank=True)  # e.g. "Work Truck"

class ParkingLot:
    name            = CharField(max_length=200)

class LotSettings:
    lot                   = OneToOneField(ParkingLot, on_delete=CASCADE, related_name='settings')
    rate                  = DecimalField(max_digits=8, decimal_places=2)
    billing_unit          = CharField(choices=[('hour','Hour'),('minute','Minute')], default='hour')
    grace_period_minutes  = IntegerField(default=1)
    daily_cap_enabled     = BooleanField(default=False)
    daily_cap_amount      = DecimalField(max_digits=8, decimal_places=2, null=True, blank=True)
    image_retention_days  = IntegerField(null=True, blank=True)  # null = forever; UI: 7/30/90/null
    confidence_threshold  = FloatField(default=0.6)

class ParkingSession:
    plate_text            = CharField(max_length=20)  # normalized
    license_plate         = ForeignKey(LicensePlate, on_delete=SET_NULL, null=True, blank=True)
    user                  = ForeignKey(User, on_delete=SET_NULL, null=True, blank=True)
    lot                   = ForeignKey(ParkingLot, on_delete=CASCADE, related_name='sessions')
    entry_time            = DateTimeField()
    exit_time             = DateTimeField(null=True, blank=True)
    duration_seconds      = IntegerField(default=0)
    charge_amount         = DecimalField(max_digits=10, decimal_places=2, default=0)
    status                = CharField(choices=[('active','Active'),('completed','Completed'),('void','Void')], default='active')
    has_duplicate_warning = BooleanField(default=False)
    was_orphaned          = BooleanField(default=False)

class PlateDetectionEvent:
    session              = ForeignKey(ParkingSession, on_delete=SET_NULL, null=True, blank=True, related_name='detection_events')
    image                = ImageField(upload_to='plates/')
    raw_plate_text       = CharField(max_length=20)
    confidence_score     = FloatField()
    event_type           = CharField(choices=[('entry','Entry'),('exit','Exit')])
    is_low_confidence    = BooleanField(default=False)
    manually_corrected   = BooleanField(default=False)
    corrected_plate      = CharField(max_length=20, null=True, blank=True)
    bounding_box         = JSONField(default=list)  # [x, y, w, h]
    timestamp            = DateTimeField(auto_now_add=True)
```

### Key Constraints

- `LicensePlate.plate_text` is always stored normalized (stripped whitespace, uppercased).
- `ParkingSession.user` is nullable — null means guest/unregistered plate.
- `ParkingSession.license_plate` is nullable — null means no registered plate matched.
- `LotSettings.daily_cap_amount` is nullable — only meaningful when `daily_cap_enabled=True`.
- `LotSettings.image_retention_days` nullable — null means keep forever.
- `PlateDetectionEvent.session` is nullable — event may exist before session is linked.

---

## CV Pipeline

### Overview

Two neural networks run in sequence:

```
Input Image (640×480 RGB)
    │
    ▼
┌────────────────────┐
│  Plate Detection   │  → Outputs bounding box [x, y, w, h]
│  CNN               │
└────────────────────┘
    │
    ▼
  Crop plate region from original image
    │
    ▼
┌────────────────────┐
│  Plate Recognizer  │  → Outputs plate text string + confidence
│  CRNN + CTC        │
└────────────────────┘
    │
    ▼
  Normalized plate text + confidence score
```

### Plate Detection CNN (`apps/cv/models/plate_detector.py`)

```
Input: 640×480 RGB image (resized, normalized)

→ Conv2d(3→32, 3×3, padding=1) + BatchNorm + ReLU + MaxPool(2×2)
  # WHY: First conv layer learns low-level features — edges, corners, gradients.
  # 32 filters is enough to capture basic edge orientations.
  # MaxPool halves spatial dims → 320×240

→ Conv2d(32→64, 3×3, padding=1) + BatchNorm + ReLU + MaxPool(2×2)
  # WHY: Second layer combines edges into shapes — rectangles, lines.
  # Plates are rectangular, so shape detection matters here.
  # → 160×120

→ Conv2d(64→128, 3×3, padding=1) + BatchNorm + ReLU + MaxPool(2×2)
  # WHY: Third layer learns complex patterns — plate-like regions with
  # text-like textures inside rectangular boundaries.
  # → 80×60

→ AdaptiveAvgPool2d(4×4) → Flatten
  # WHY: Adaptive pooling makes the network accept any input size.
  # Produces fixed 128×4×4 = 2048 features.

→ FC(2048→256) + ReLU + Dropout(0.3)
  # WHY: Dense layer compresses features. Dropout prevents overfitting
  # on synthetic data which has limited variety.

→ FC(256→4)
  # WHY: Four outputs = [x, y, w, h] bounding box coordinates.
  # These are normalized to [0,1] range relative to image dimensions.
```

- **Loss:** Smooth L1 (Huber loss) — robust to outlier bounding boxes, smoother gradients than MSE near zero.
- **Optimizer:** Adam (lr=1e-3) — adaptive learning rate handles sparse gradients well.
- **Training data:** Synthetic plates composited onto background images with known bounding boxes.

### Plate Recognizer CRNN (`apps/cv/models/recognizer.py`)

```
Input: 128×32 grayscale cropped plate image

CNN Backbone:
→ Conv2d(1→64, 3×3, padding=1) + BatchNorm + ReLU + MaxPool(2×2)
  # WHY: Extract basic character stroke features from grayscale input.
  # → 64×16

→ Conv2d(64→128, 3×3, padding=1) + BatchNorm + ReLU + MaxPool(2×2)
  # WHY: Combine strokes into partial characters — verticals, curves, junctions.
  # → 32×8

→ Conv2d(128→256, 3×3, padding=1) + BatchNorm + ReLU + MaxPool(2×1)
  # WHY: MaxPool(2×1) preserves width while reducing height — critical because
  # characters are arranged horizontally and we need width resolution for
  # the sequence model. → 16×8

Reshape to Sequence:
→ Permute + reshape so width becomes the sequence dimension
  # WHY: Each vertical slice of the feature map becomes one timestep.
  # The LSTM will read the plate left-to-right, one slice at a time.
  # Sequence length = 16 (one per horizontal position)

Sequence Model:
→ Bidirectional LSTM(256 hidden, 2 layers)
  # WHY bidirectional: Reading both left-to-right AND right-to-left helps
  # disambiguate characters. "D" vs "O" is easier when you know what comes
  # after it too.
  # WHY LSTM over GRU: LSTM's forget gate handles the longer sequences
  # of 7-8 character plates better than GRU.

→ FC(512→num_classes)
  # WHY 512: bidirectional doubles the hidden size (256×2).
  # num_classes = 37: 26 letters + 10 digits + 1 CTC blank token.
```

- **Loss:** CTC (Connectionist Temporal Classification) — handles variable-length output without needing character-level alignment labels. The network just needs the plate text, not where each character is.
- **Optimizer:** Adam (lr=1e-3).
- **Decoding:** Greedy CTC decode (collapse repeated chars, remove blanks). Beam search is an optional future improvement.

### Synthetic Data Generation (`apps/cv/training/synthetic_data.py`)

Generates fake US-style license plates programmatically:

1. **Plate template:** White rectangle with blue header strip, rendered via PIL.
2. **Random text:** Format patterns like `ABC 1234`, `123 ABC`, `AB 1234` (common US formats).
3. **Font rendering:** Load a plate-like monospace font, render characters onto the template.
4. **Background compositing:** Paste the plate onto random background images (solid colors, gradients, simple scenes) at random positions, scales, and slight rotations.
5. **Ground truth:** Bounding box coordinates and plate text are known exactly because we placed them.
6. **Volume:** Generate 10,000+ detector samples, 50,000+ recognizer samples.

### Augmentations (`apps/cv/training/augment.py`)

Applied during training to simulate real-world conditions:

| Augmentation | Why |
|---|---|
| Random brightness/contrast | Different lighting conditions |
| Gaussian blur (slight) | Camera out of focus |
| Salt-and-pepper noise | Sensor noise |
| Random rotation (±5°) | Car not perfectly aligned |
| Random perspective warp | Camera angle variation |
| Motion blur (horizontal) | Car moving during capture |
| Random crop/zoom | Plate at different distances |

### Preprocessing (`apps/cv/preprocessing.py`)

Steps before inference:

1. Read image with OpenCV (`cv2.imread`)
2. Convert BGR → RGB (OpenCV loads as BGR)
3. Resize to 640×480 for detector input
4. Normalize pixel values to [0, 1]
5. Convert to PyTorch tensor
6. After detection: crop plate region, resize to 128×32, convert to grayscale for recognizer

### Device Auto-Detection (`apps/cv/utils/device.py`)

```python
def get_device():
    """
    Auto-detect the best available compute device.
    Priority: MPS (Apple Silicon) → CUDA (NVIDIA GPU) → CPU.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
```

### Full Pipeline (`apps/cv/pipeline.py`)

```python
def process_plate_image(image_path: str) -> dict:
    """
    End-to-end: image file → {plate_text, confidence, bounding_box}

    Steps:
    1. Load and preprocess image
    2. Run plate detector → bounding box
    3. Crop plate region from original image
    4. Run recognizer on cropped plate → text + confidence
    5. Return results dict
    """
```

---

## Session Logic

### `apps/parking/services.py`

#### Entry Flow (`handle_entry`)

```
receive detection event (plate_text, confidence, bounding_box, image)
    │
    ├── normalize plate text (strip whitespace, uppercase)
    │
    ├── check confidence vs threshold
    │   └── below threshold → set is_low_confidence=True
    │
    ├── look up existing active session for this plate
    │   └── found → ORPHAN HANDLING:
    │       ├── void old session (status='void', charge=0, was_orphaned=True)
    │       └── new session gets has_duplicate_warning=True
    │
    ├── try to match plate to registered LicensePlate
    │   ├── found → link session to LicensePlate + User
    │   └── not found → session.user=null (guest)
    │
    ├── create ParkingSession(status='active', entry_time=now)
    │
    └── create PlateDetectionEvent(event_type='entry', linked to session)
```

#### Exit Flow (`handle_exit`)

```
receive detection event (plate_text, confidence, bounding_box, image)
    │
    ├── normalize plate text
    │
    ├── find active session with matching plate_text
    │   └── not found → create entry instead (or flag error)
    │
    ├── calculate charge:
    │   ├── duration = now - entry_time
    │   ├── if duration ≤ grace_period → charge = $0
    │   ├── else:
    │   │   ├── billing_unit='hour' → charge = ceil(hours) × rate
    │   │   └── billing_unit='minute' → charge = ceil(minutes) × rate
    │   ├── if daily_cap_enabled and charge > daily_cap_amount:
    │   │   └── charge = daily_cap_amount
    │   └── set duration_seconds, charge_amount
    │
    ├── update session: status='completed', exit_time=now
    │
    └── create PlateDetectionEvent(event_type='exit', linked to session)
```

### `apps/parking/utils.py`

```python
def normalize_plate(raw_text: str) -> str:
    """
    Strip all whitespace, convert to uppercase.
    'abc 123' → 'ABC123'
    '  Ab C-123 ' → 'ABC-123'

    Note: Only whitespace is stripped. Hyphens and other characters are preserved.
    This means 'ABC-123' does NOT match 'ABC123' — this is intentional for
    exact matching.
    """
```

### Billing Calculation (`apps/parking/services.py :: calculate_charge`)

```python
def calculate_charge(entry_time, exit_time, lot_settings) -> Decimal:
    """
    Calculate parking charge based on lot settings.

    Rules:
    1. Duration ≤ grace_period_minutes → $0.00
    2. billing_unit='minute' → ceil(total_minutes) × rate
    3. billing_unit='hour' → ceil(total_hours) × rate
    4. If daily_cap_enabled and charge > daily_cap_amount → cap it
    """
```

### Plate Matching

- **Method:** EXACT match after normalization.
- `normalize_plate("ABC 123")` → `"ABC123"`
- `normalize_plate("abc123")` → `"ABC123"`
- These match: `"ABC 123"` and `"abc123"` (both normalize to `"ABC123"`)
- These do NOT match: `"ABCI23"` and `"ABC123"` (different characters)

### Low-Confidence Flow

1. Detection event created with `is_low_confidence=True`.
2. Session IS created with the best-guess plate text.
3. Event appears in error queue on dashboard.
4. Admin clicks to manually correct → sets `manually_corrected=True`, `corrected_plate` value.
5. On correction: session's `plate_text` is updated to the corrected value, and the system re-checks for a matching registered `LicensePlate` to link.

### Orphan/Duplicate Handling

When plate `X` enters but already has an active session:

1. Old session: `status='void'`, `charge_amount=0`, `was_orphaned=True`.
2. New session: `has_duplicate_warning=True`, `status='active'`.
3. Both events are visible in the dashboard log.

---

## Dashboard Pages

All pages are behind `@login_required`. The frontend uses Django templates + HTMX for dynamic updates + Chart.js for charts. Style: dark mode, data-dense ops console.

### Page Inventory

| URL | Template | Description |
|-----|----------|-------------|
| `/login/` | `registration/login.html` | Login form, dark-themed |
| `/` | `dashboard.html` | Main overview: active sessions count, today's revenue, recent events, live running costs |
| `/upload/` | `upload.html` | Image upload form (entry or exit), shows detection result after processing |
| `/log/` | `log.html` | Full session log with filters (status, date range, plate search), paginated |
| `/errors/` | `errors.html` | Low-confidence detections queue, manual correction form per event |
| `/revenue/` | `revenue.html` | Revenue analytics: daily/weekly/monthly charts (Chart.js), total revenue, average session duration |
| `/settings/` | `settings.html` | Lot settings: rate, billing unit, grace period, daily cap, image retention, confidence threshold |

### Design Tokens (CSS Custom Properties)

```css
:root {
    --bg-primary: #0f1117;
    --bg-secondary: #1a1d27;
    --bg-tertiary: #252832;
    --text-primary: #e4e4e7;
    --text-secondary: #a1a1aa;
    --accent-blue: #3b82f6;
    --status-active: #22c55e;
    --status-warning: #eab308;
    --status-error: #ef4444;
    --status-void: #6b7280;
    --border-color: #2e3039;
    --radius: 8px;
    --font-mono: 'JetBrains Mono', monospace;
}
```

### HTMX Patterns

- **Upload result:** Form submits via HTMX, response swaps in detection result partial.
- **Error correction:** Inline edit on errors page, HTMX PATCH to update plate text.
- **Dashboard refresh:** Polling every 10s for active session count and running costs.
- **Log filters:** HTMX GET with query params, swaps table body.

### API Endpoints (`apps/dashboard/api.py`)

| Method | URL | Purpose |
|--------|-----|---------|
| `POST` | `/api/upload/` | Accept image upload, run CV pipeline, create session/event. Designed for future camera feed integration. |
| `GET` | `/api/sessions/` | List sessions (filterable), used by HTMX on log page |
| `GET` | `/api/dashboard-stats/` | Active count, today's revenue, recent events — polled by dashboard |
| `PATCH` | `/api/events/<id>/correct/` | Manual plate correction on low-confidence event |
| `GET` | `/api/revenue-data/` | Chart data for revenue page (JSON for Chart.js) |

---

## Daily Work Plan

### Day 1 — Project Foundation

**Goal:** Working Django project in Docker with all models migrated, admin configured, and auth wired up.

**Tasks:**
- [ ] Copy `PLAN.md` (this file) to project root
- [ ] Create `docker-compose.yml` with Django + PostgreSQL 16 services
- [ ] Create `Dockerfile` (Python 3.11, install system deps for OpenCV and Pillow)
- [ ] Create `requirements.txt` (Django, psycopg2-binary, Pillow, opencv-python-headless, torch, torchvision, pytest, pytest-django, pytest-cov, django-extensions)
- [ ] Create `.env.example` with all required env vars (SECRET_KEY, DB creds, DEBUG, ALLOWED_HOSTS)
- [ ] Scaffold Django project: `config/settings.py`, `config/urls.py`, `config/wsgi.py`
- [ ] Configure `settings.py`: database (PostgreSQL from env), AUTH_USER_MODEL, INSTALLED_APPS, MEDIA_ROOT, STATIC_ROOT, LOGIN_URL, LOGIN_REDIRECT_URL
- [ ] Create `apps/accounts/` app with custom `User(AbstractUser)` model
- [ ] Create `apps/parking/` app with all models: `LicensePlate`, `ParkingLot`, `LotSettings`, `ParkingSession`, `PlateDetectionEvent`
- [ ] Create `apps/cv/` app (empty models, just app config)
- [ ] Create `apps/dashboard/` app (empty for now)
- [ ] Run `makemigrations` and `migrate`
- [ ] Register all models in `admin.py` with useful list displays and filters
- [ ] Wire auth URLs: login, logout, redirect config
- [ ] Create a basic `base.html` template (minimal, will be redesigned Day 9)
- [ ] Create a basic `registration/login.html` template
- [ ] Create management command or fixture to create default superuser and default ParkingLot + LotSettings
- [ ] Verify `docker-compose up --build` runs clean and admin is accessible

**Quality Gate:** code-reviewer + security-reviewer on every file. No hardcoded secrets in settings (all from env). Migrations run without errors.

**Deliverable:** Running Django app in Docker. Admin accessible at `/admin/`. Login required for all pages. All models visible in admin.

---

### Day 2 — CV Preprocessing

**Goal:** OpenCV preprocessing pipeline that takes a raw image and prepares it for the neural networks.

**Tasks:**
- [ ] Create `apps/cv/utils/device.py` — auto-detect MPS → CUDA → CPU with educational comments
- [ ] Create `apps/cv/preprocessing.py` with functions:
  - `load_image(path) → np.ndarray` — read with OpenCV, handle errors
  - `bgr_to_rgb(image) → np.ndarray` — with comment explaining why OpenCV uses BGR
  - `resize_for_detector(image, target=(640, 480)) → np.ndarray` — with aspect ratio handling
  - `normalize_pixels(image) → np.ndarray` — scale to [0, 1], explain why
  - `to_tensor(image) → torch.Tensor` — HWC to CHW conversion, explain PyTorch format
  - `crop_plate_region(image, bbox) → np.ndarray` — extract bounding box region
  - `prepare_for_recognizer(plate_crop) → torch.Tensor` — resize to 128×32, grayscale, normalize
- [ ] Add comprehensive educational comments to every function explaining the *why*
- [ ] Write unit tests for each preprocessing function (test with synthetic test images)
- [ ] Verify all preprocessing functions work on MPS tensors

**Quality Gate:** All tests pass. code-reviewer + security-reviewer. Every function has docstring + inline educational comments.

**Deliverable:** Complete preprocessing module that transforms raw images into tensors ready for both neural networks.

---

### Day 3 — Synthetic Training Data Pipeline

**Goal:** Generate thousands of synthetic license plate images with known ground truth for training both networks.

**Tasks:**
- [ ] Create `apps/cv/training/synthetic_data.py`:
  - `generate_plate_text() → str` — random US-format plate strings (e.g., "ABC 1234", "123 ABC")
  - `render_plate_image(text) → PIL.Image` — draw plate template with text using PIL
  - `composite_on_background(plate_img, bg_size) → (image, bbox)` — paste plate at random position/scale/rotation onto background, return image + ground truth bbox
  - `generate_detector_dataset(n=10000, output_dir)` — batch generate for detector training
  - `generate_recognizer_dataset(n=50000, output_dir)` — batch generate cropped plates for recognizer
- [ ] Create `apps/cv/training/dataset.py`:
  - `PlateDetectorDataset(Dataset)` — loads images + bbox labels, applies transforms
  - `PlateRecognizerDataset(Dataset)` — loads cropped plates + text labels, applies transforms
  - Implement `__len__`, `__getitem__` with proper tensor conversion
  - Label encoding for recognizer: character → index mapping (A-Z, 0-9, blank)
- [ ] Create `apps/cv/training/augment.py`:
  - `DetectorAugment` — brightness, contrast, blur, noise, rotation for full images
  - `RecognizerAugment` — brightness, contrast, blur, noise, perspective warp for plate crops
  - Educational comments explaining why each augmentation helps
- [ ] Generate sample datasets and verify DataLoader iteration works
- [ ] Write tests for data generation (output shapes, label formats, character encoding)

**Quality Gate:** DataLoaders produce correct batch shapes. Labels are valid. Augmentations don't corrupt ground truth. code-reviewer + security-reviewer.

**Deliverable:** Working synthetic data pipeline that generates training data for both networks. Sample data generated and verified.

---

### Day 4 — Plate Detection CNN

**Goal:** Build, train, and evaluate the plate detection CNN on synthetic data.

**Tasks:**
- [ ] Create `apps/cv/models/plate_detector.py`:
  - `PlateDetectorCNN(nn.Module)` — architecture exactly as specified above
  - Educational comments on every layer explaining what it learns and why
  - `forward(x) → bbox_predictions` with shape comments
  - Helper: `predict(image_tensor) → [x, y, w, h]` with sigmoid activation for [0,1] range
- [ ] Create `apps/cv/training/train_detector.py`:
  - CLI script (argparse) — epochs, batch_size, learning_rate, data_dir, output_path
  - Training loop with: Smooth L1 loss, Adam optimizer, learning rate scheduler
  - Validation split (80/20)
  - Save best model weights based on validation loss
  - Print training/validation loss per epoch
  - IoU (Intersection over Union) metric for evaluation
  - Educational comments on loss function choice, optimizer, and training loop mechanics
- [ ] Run training on MPS backend (~50 epochs or until convergence)
- [ ] Evaluate: measure IoU on held-out validation set
- [ ] Save trained weights to `apps/cv/weights/detector.pth`
- [ ] Add `detector.pth` and `recognizer.pth` to `.gitignore`

**Quality Gate:** Model achieves reasonable IoU on synthetic validation data (target: >0.7 IoU). Training completes without errors on MPS. code-reviewer on model and training code.

**Deliverable:** Trained plate detection model with saved weights. Training script is reproducible.

---

### Day 5 — CRNN Recognizer

**Goal:** Build, train, and evaluate the CRNN plate text recognizer.

**Tasks:**
- [ ] Create `apps/cv/models/recognizer.py`:
  - `PlateRecognizerCRNN(nn.Module)` — architecture exactly as specified above
  - CNN backbone (3 conv blocks) with educational comments on each
  - Reshape layer with comment explaining feature-map-to-sequence conversion
  - Bidirectional LSTM with comment explaining why bidirectional helps
  - FC output layer sized for character set (26 + 10 + 1 blank = 37)
  - `decode_predictions(output) → list[str]` — greedy CTC decode
  - Educational comments throughout explaining CTC concepts
- [ ] Create `apps/cv/training/train_recognizer.py`:
  - CLI script — epochs, batch_size, learning_rate, data_dir, output_path
  - Training loop with: CTC loss, Adam optimizer, scheduler
  - Validation split (80/20)
  - Character-level accuracy metric + full-plate accuracy metric
  - Save best model weights
  - Educational comments on CTC loss mechanics (alignment-free training)
- [ ] Run training on MPS backend (~100 epochs or until convergence)
- [ ] Evaluate: measure character accuracy and full-plate accuracy on validation set
- [ ] Save trained weights to `apps/cv/weights/recognizer.pth`

**Quality Gate:** Model achieves reasonable accuracy on synthetic validation data (target: >80% full-plate accuracy). Training completes on MPS. code-reviewer on model and training code.

**Deliverable:** Trained CRNN recognizer with saved weights. Training script is reproducible.

---

### Day 6 — Full CV Pipeline

**Goal:** Wire detector + recognizer into a single end-to-end pipeline that takes an image and returns plate text + confidence.

**Tasks:**
- [ ] Create `apps/cv/pipeline.py`:
  - `PlateRecognitionPipeline` class:
    - `__init__` — load both models, move to device, set to eval mode
    - `process(image_path) → dict` — full pipeline: preprocess → detect → crop → recognize → return {plate_text, confidence, bounding_box}
    - Confidence score: softmax probability from recognizer output, averaged across characters
    - Handle edge cases: no plate detected, very small bounding box, low confidence
  - Singleton/lazy-loading pattern so models are loaded once per process
- [ ] Integration test: process a synthetic plate image end-to-end
- [ ] Test with varied inputs: clean plates, rotated, blurry, partial
- [ ] Measure inference time per image
- [ ] Add error handling: corrupt images, unsupported formats, missing model weights

**Quality Gate:** Pipeline processes synthetic images and returns correct plate text. Handles errors gracefully. Inference time < 2 seconds per image on MPS. code-reviewer + security-reviewer.

**Deliverable:** Working end-to-end CV pipeline callable from Django views.

---

### Day 7 — Parking Session Logic

**Goal:** Implement all business logic for parking sessions: entry, exit, billing, orphan handling, plate normalization.

**Tasks:**
- [ ] Implement `apps/parking/utils.py`:
  - `normalize_plate(raw_text) → str` — strip whitespace, uppercase
  - Unit tests: various inputs including edge cases (empty, special chars, unicode)
- [ ] Implement `apps/parking/services.py`:
  - `calculate_charge(entry_time, exit_time, lot_settings) → Decimal`:
    - Grace period check
    - Per-hour vs per-minute billing
    - Daily cap enforcement
    - Return Decimal with 2 decimal places
  - `handle_entry(plate_text, confidence, bounding_box, image, lot) → ParkingSession`:
    - Normalize plate
    - Check for existing active session (orphan handling)
    - Match to registered plate/user
    - Create session + detection event
    - Flag low confidence
  - `handle_exit(plate_text, confidence, bounding_box, image, lot) → ParkingSession`:
    - Normalize plate
    - Find active session
    - Calculate charge
    - Update session (completed)
    - Create detection event
  - `correct_plate(event_id, corrected_text) → PlateDetectionEvent`:
    - Update event: manually_corrected, corrected_plate
    - Update linked session's plate_text
    - Re-check for registered plate match
- [ ] Write comprehensive unit tests for `calculate_charge`:
  - [ ] Duration under grace period → $0
  - [ ] Per-hour billing: 1.5 hours → ceil to 2 hours × rate
  - [ ] Per-minute billing: 90 minutes → 90 × rate
  - [ ] Daily cap: charge exceeds cap → capped
  - [ ] Daily cap disabled: charge exceeds cap → not capped
  - [ ] Zero duration → $0
  - [ ] Exact grace period boundary → $0
  - [ ] One second over grace → charged
- [ ] Write unit tests for `handle_entry`:
  - [ ] New plate → new active session
  - [ ] Registered plate → session linked to user
  - [ ] Guest plate → session.user=null
  - [ ] Duplicate entry → old session voided, new session has warning
  - [ ] Low confidence → flagged
- [ ] Write unit tests for `handle_exit`:
  - [ ] Matching active session → completed with charge
  - [ ] No matching session → appropriate error handling
- [ ] Write unit tests for `correct_plate`:
  - [ ] Updates event and session
  - [ ] Re-links to registered plate if match found

**Quality Gate:** All billing edge cases tested and passing. Orphan handling works. Coverage ≥80% on `apps/parking/`. code-reviewer + security-reviewer.

**Deliverable:** Complete, tested business logic layer with no untested edge cases.

---

### Day 8 — Upload View & API

**Goal:** Build the upload endpoint and integration tests for the full upload → CV → session flow.

**Tasks:**
- [ ] Implement `apps/dashboard/api.py`:
  - `POST /api/upload/` — accept image file, run CV pipeline, call `handle_entry` or `handle_exit` (based on auto-detection or explicit `event_type` parameter), return JSON result
  - Request format: multipart form with `image` file and optional `event_type` field (default: auto-detect based on whether an active session exists for the detected plate)
  - Response format: `{session_id, plate_text, confidence, event_type, charge_amount, status}`
  - Input validation: file type check (JPEG, PNG only), file size limit (10MB)
  - Error responses: 400 for bad input, 500 for CV pipeline failure (with logged details)
- [ ] Implement `apps/dashboard/views.py`:
  - `UploadView` — renders upload form, handles HTMX response with detection result
- [ ] Implement `apps/dashboard/urls.py` — wire all URL patterns
- [ ] Wire dashboard URLs into `config/urls.py`
- [ ] Write integration tests:
  - [ ] Upload valid image → session created, event recorded
  - [ ] Upload entry then exit for same plate → session completed with charge
  - [ ] Upload invalid file type → 400 error
  - [ ] Upload with no image → 400 error
  - [ ] Upload when CV pipeline fails → 500 with meaningful error
  - [ ] All endpoints require authentication (test unauthenticated → redirect)
- [ ] Create basic upload template (functional, will be redesigned Day 10)

**Quality Gate:** Upload flow works end-to-end. Integration tests pass. All endpoints require auth. code-reviewer + security-reviewer (especially on file upload handling).

**Deliverable:** Working upload endpoint ready for manual testing and future camera integration.

---

### Day 9 — Frontend Design Handoff

**Goal:** Generate a comprehensive Claude Design prompt, receive the design, and set up the base template infrastructure to implement it.

**Tasks:**
- [ ] Write a detailed Claude Design prompt covering all pages:
  - Login page (dark theme, centered form)
  - Dashboard (active sessions, revenue today, recent events, live running costs)
  - Upload page (drag-and-drop area, result display with plate image + detected text + confidence badge)
  - Session log (dense table, status badges, filters, pagination)
  - Error queue (low-confidence events, inline correction, confidence scores)
  - Revenue analytics (Chart.js charts, date range selector, summary cards)
  - Settings page (lot configuration form)
  - Include: color palette, typography, spacing, badge styles, table density
- [ ] Paste prompt into Claude Design, receive full visual design
- [ ] Set up `templates/base.html`:
  - Dark theme CSS variables (from design)
  - Navigation sidebar/header
  - HTMX script inclusion
  - Chart.js script inclusion (CDN with SRI)
  - Responsive layout structure
- [ ] Set up `static/css/main.css`:
  - CSS custom properties from design tokens
  - Base typography
  - Component classes: cards, badges, tables, forms, buttons
  - Status colors: green (active), yellow (warning), red (error), gray (void)
- [ ] Set up `templates/registration/login.html` matching the design exactly
- [ ] Verify login page renders correctly and authentication works

**Quality Gate:** Login page matches Claude Design output exactly. Base template has all design tokens. CSS variables defined. code-reviewer on template structure.

**Deliverable:** Complete base template infrastructure. Login page fully styled. All design tokens in CSS.

---

### Day 10 — All Dashboard Pages

**Goal:** Implement every dashboard page to match the Claude Design output exactly.

**Tasks:**
- [ ] `templates/dashboard.html`:
  - Summary cards: active sessions count, today's revenue, today's entries/exits
  - Live running costs table for active sessions (HTMX polling every 10s)
  - Recent events list (last 10 detection events)
  - Guest vs registered session counts
- [ ] `templates/upload.html`:
  - Drag-and-drop upload zone
  - HTMX form submission
  - Result display: plate image, bounding box overlay (canvas), detected text, confidence badge (green ≥0.8, yellow ≥0.6, red <0.6), session details
- [ ] `templates/log.html`:
  - Dense session table: plate, entry time, exit time, duration, charge, status badge
  - Filters: status dropdown, date range, plate search (HTMX GET)
  - Pagination
  - Separate section/tab for guest sessions
- [ ] `templates/errors.html`:
  - List of low-confidence detection events
  - Each row: thumbnail, detected text, confidence score, timestamp
  - Inline correction form (HTMX PATCH)
  - Mark as reviewed/corrected
- [ ] `templates/revenue.html`:
  - Chart.js line chart: daily revenue (last 30 days)
  - Chart.js bar chart: hourly distribution
  - Summary cards: total revenue, avg session duration, total sessions
  - Date range filter
- [ ] `templates/settings.html`:
  - Form for LotSettings: rate, billing unit, grace period, daily cap toggle + amount, image retention dropdown, confidence threshold slider
  - Save button (standard form POST)
  - Current values pre-filled
- [ ] Implement all supporting views in `apps/dashboard/views.py`
- [ ] Implement all API endpoints in `apps/dashboard/api.py`
- [ ] Wire all URLs in `apps/dashboard/urls.py`
- [ ] Add navigation to `base.html` (sidebar or top nav with active state)
- [ ] Verify every page requires login

**Quality Gate:** Every page matches Claude Design output. All HTMX interactions work. No unstyled elements. Every route requires auth. code-reviewer.

**Deliverable:** Complete, styled dashboard. All pages functional with real data.

---

### Day 11 — Settings, Image Cleanup, Docker Polish

**Goal:** Settings page saves correctly, image cleanup runs, Docker setup is production-ready.

**Tasks:**
- [ ] Settings page save logic:
  - Form validation (rate > 0, grace period ≥ 0, etc.)
  - Success message after save
  - Handle daily cap toggle (clear amount when disabled)
- [ ] Create `apps/parking/management/commands/cleanup_old_images.py`:
  - Read `image_retention_days` from LotSettings
  - If null → skip (keep forever)
  - Delete `PlateDetectionEvent.image` files older than retention period
  - Clear the `image` field on the model (set to empty, keep the record)
  - Log how many images deleted
  - Dry-run flag (`--dry-run`) that reports what would be deleted
- [ ] Write tests for cleanup command:
  - [ ] Images older than retention → deleted
  - [ ] Images newer than retention → kept
  - [ ] Retention set to null → nothing deleted
  - [ ] Dry run → reports but doesn't delete
  - [ ] Session records remain after image deletion
- [ ] Docker polish:
  - `docker-compose.yml`: volume mounts for media, static, postgres data
  - Healthcheck on Django and PostgreSQL services
  - Environment variable validation on startup
  - `collectstatic` runs in Dockerfile
  - Gunicorn as production server (with `--reload` in dev)
  - `.dockerignore` file (exclude .git, __pycache__, .env, weights/)
- [ ] Create `pytest.ini` with coverage configuration:
  - Cover `apps/accounts` and `apps/parking`
  - Exclude `apps/cv` from coverage requirement
  - Fail under 80%
- [ ] Run full test suite, verify ≥80% coverage on target apps
- [ ] Add cron instruction to README for cleanup command (or Docker healthcheck-based trigger)

**Quality Gate:** Settings save and load correctly. Cleanup command works. Docker builds clean. Tests pass with ≥80% coverage. code-reviewer + security-reviewer.

**Deliverable:** Production-ready Docker setup. Image retention working. Test coverage target met.

---

### Day 12 — Buffer / Security Review / Edge Cases

**Goal:** Final polish, security audit, edge case testing, and verification that everything works end-to-end.

**Tasks:**
- [ ] Full security review with `security-reviewer` agent:
  - [ ] CSRF protection on all forms
  - [ ] File upload validation (type, size, no path traversal)
  - [ ] No hardcoded secrets
  - [ ] SQL injection prevention (ORM usage verified)
  - [ ] XSS prevention (template auto-escaping verified)
  - [ ] Auth on every endpoint (test unauthenticated access)
  - [ ] Error messages don't leak sensitive info
  - [ ] Media files not served with directory listing
- [ ] Edge case testing:
  - [ ] Upload extremely large image (>10MB) → rejected
  - [ ] Upload non-image file → rejected
  - [ ] Plate with special characters → normalized correctly
  - [ ] Empty plate text from CV → handled gracefully
  - [ ] Simultaneous entries for same plate (race condition) → handled by orphan logic
  - [ ] Session with null exit_time → running cost displayed correctly
  - [ ] Revenue calculations with no sessions → zero, no errors
  - [ ] Settings with edge values (rate=0, grace=0, etc.)
- [ ] Run `code-reviewer` agent on entire codebase
- [ ] Check for unused imports and dead code across all files
- [ ] Verify `docker-compose up --build` from clean state
- [ ] Run full verification checklist (below)
- [ ] Write/update README.md with:
  - Setup instructions
  - Environment variables
  - Running locally
  - Running tests
  - Training CV models
  - Cleanup command usage

**Quality Gate:** All verification checklist items pass. Security review clean. No CRITICAL or HIGH findings. Docker runs from clean state.

**Deliverable:** Production-ready application. All features working. Documentation complete.

---

## Testing Strategy

### Unit Tests (`apps/parking/tests/`, `apps/accounts/tests/`)

| Test Area | File | Key Cases |
|-----------|------|-----------|
| Plate normalization | `test_utils.py` | Whitespace stripping, uppercase, special chars, empty input |
| Charge calculation | `test_services.py` | Grace period, per-hour, per-minute, daily cap, zero duration, boundary cases |
| Entry handling | `test_services.py` | New entry, duplicate/orphan, guest plate, registered plate, low confidence |
| Exit handling | `test_services.py` | Normal exit, no matching session, charge correctness |
| Plate correction | `test_services.py` | Update event, update session, re-link plate |
| User model | `test_models.py` | Custom user creation, plate association |
| Image cleanup | `test_cleanup.py` | Retention logic, dry run, null retention |

### Integration Tests (`apps/dashboard/tests/`)

| Test Area | File | Key Cases |
|-----------|------|-----------|
| Upload flow | `test_api.py` | Valid upload → session, entry+exit → charge, invalid file → 400 |
| Auth enforcement | `test_views.py` | Every URL redirects to login when unauthenticated |
| Settings save | `test_views.py` | Form save updates LotSettings |
| Error correction | `test_api.py` | PATCH correction updates event and session |

### Coverage Requirements

| App | Target | Enforced |
|-----|--------|----------|
| `apps/accounts` | ≥ 80% | Yes (pytest-cov) |
| `apps/parking` | ≥ 80% | Yes (pytest-cov) |
| `apps/cv` | Best effort | No (excluded from coverage gate) |
| `apps/dashboard` | Best effort | No (view/template testing is supplemental) |

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Synthetic data doesn't represent real plates well enough | Medium | Add diverse augmentations (Day 3), can retrain with real data later without architecture changes |
| CTC training is unstable / doesn't converge | Medium | Use gradient clipping, reduce learning rate, increase training data volume. Fallback: start with shorter plates (3-4 chars) and increase |
| MPS backend has PyTorch compatibility issues | Medium | Device auto-detection falls back to CPU. Check PyTorch MPS support status for specific ops (CTC loss may need CPU fallback) |
| File upload security (path traversal, malicious files) | High | Validate file type by magic bytes (not just extension), use Django's `ImageField` with Pillow validation, serve media through Django (not raw nginx in dev) |
| Race condition on duplicate plate handling | Medium | Use Django `select_for_update()` or database-level locking when checking for existing active sessions |
| Large media directory slows down system | Low | Image retention cleanup command runs daily. Index `PlateDetectionEvent.timestamp` for efficient cleanup queries |
| CV inference too slow for real-time | Low | Acceptable for manual upload. For future camera feed: add async task queue (Celery) — architecture supports this without changes to pipeline code |

---

## Verification Checklist

Run through every item before considering the project complete:

- [ ] All pages redirect to `/login/` when unauthenticated
- [ ] Upload screenshot → plate text + confidence shown on screen
- [ ] Same plate twice → entry then exit → charge calculated correctly
- [ ] Stay under grace period → $0 charge
- [ ] Daily cap enabled → charge never exceeds cap amount
- [ ] Low confidence detection → flagged in error queue with `is_low_confidence=True`
- [ ] Manual correction on error queue → session plate text updates, re-links if registered plate found
- [ ] Duplicate entry (plate enters while previous session active) → old session auto-voided with `was_orphaned=True`, new session opens with `has_duplicate_warning=True`
- [ ] Guest plate (unregistered) → session shows with `user=null`, visible in guest section
- [ ] Plate matching: `"ABC 123"` matches `"abc123"` (both normalize to `"ABC123"`)
- [ ] Plate matching: `"ABCI23"` does NOT match `"ABC123"` (different characters)
- [ ] Image retention set to 30 days → cleanup command deletes images older than 30 days, session records remain
- [ ] Image retention set to null (forever) → cleanup command skips deletion
- [ ] `docker-compose up --build` runs clean from a fresh clone (after providing `.env`)
- [ ] `pytest` passes with ≥80% coverage on `apps/accounts` and `apps/parking`
- [ ] No hardcoded secrets in any source file
- [ ] All forms have CSRF protection
- [ ] File uploads reject non-image files and files >10MB
- [ ] Error messages do not expose stack traces or internal paths to the user
- [ ] Navigation shows active page state
- [ ] Revenue page shows correct totals matching session data
- [ ] Settings page saves and loads all fields correctly

---

## Notes

- **Multi-lot readiness:** `ParkingLot` and `LotSettings` are separate models linked by `OneToOneField`. Adding a second lot requires only creating new `ParkingLot` + `LotSettings` rows — zero schema changes. Sessions are FK-linked to a lot. The UI currently assumes a single lot but can be extended with a lot selector.

- **Camera integration path:** The `POST /api/upload/` endpoint accepts a multipart image and returns JSON. A camera system only needs to POST images to this endpoint with proper auth headers. No code changes required — just point the camera's HTTP client at the endpoint.

- **CV model retraining:** Training scripts are CLI-only and independent of the Django app. To retrain: generate new synthetic data, run training script, replace `.pth` weights file, restart the Django server. No migrations or code changes needed.

- **CTC loss on MPS:** As of PyTorch 2.x, CTC loss may not be fully supported on MPS. The training script should catch this and fall back to CPU for the loss computation while keeping the model on MPS. The `device.py` utility handles general device selection, but the training scripts need specific handling for this edge case.

- **Confidence scoring:** The recognizer's confidence is computed as the average of per-character softmax probabilities. This is a simple heuristic — more sophisticated methods (e.g., beam search with score normalization) can be added later without changing the pipeline interface.

- **Billing precision:** All monetary values use `DecimalField` with 2 decimal places. Python `Decimal` is used in calculations to avoid floating-point rounding issues. Never use `float` for money.

- **Comment philosophy:** CV code comments are written for someone learning deep learning for the first time. They explain not just *what* the code does, but *why* this specific choice was made over alternatives. Business logic comments are standard professional-grade docstrings and inline comments.
