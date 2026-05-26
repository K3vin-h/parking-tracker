# Parking Lot Tracker

## Database Models

The database schema is built around five models that capture the full lifecycle of a parking session — from a registered plate to a completed billing record.

### LicensePlate

Stores plates registered by users. A user can register multiple plates; each plate belongs to exactly one user.

- `plate_text` — normalized plate string (uppercase, no whitespace)
- `is_primary` — marks the user's default plate
- `label` — optional human-readable label (e.g., "Daily Driver")
- **Constraint:** a user cannot register the same plate text twice

### ParkingLot

Represents a physical lot. The schema is multi-lot ready — all session and billing records are scoped to a lot.

### LotSettings

One-to-one with `ParkingLot`. Stores all operator-configurable parameters for a lot.

- **Billing:** hourly or per-minute rate, grace period, optional daily cap
- **CV:** confidence threshold — detections below this score are flagged for manual review
- **Retention:** optional image retention window in days

<details>
<summary><strong>Grace Period & Daily Cap</strong></summary>

Sessions shorter than `grace_period_minutes` are charged nothing — the car came and left within the free window.

For sessions that exceed the grace period:

$$\text{charge} = \left\lceil \frac{\text{duration}}{\text{billing\_unit}} \right\rceil \times \text{rate}$$

Where `billing_unit` is either 3600 seconds (hourly) or 60 seconds (per-minute).

If `daily_cap_enabled` is true, the charge is clamped:

$$\text{final\_charge} = \min(\text{charge},\ \text{daily\_cap\_amount})$$

All monetary values use `Decimal` — never `float` — to prevent floating-point rounding errors on financial data.

</details>

### ParkingSession

The core transactional record. One row per car visit.

- `plate_text` — source of truth for the plate (always stored normalized)
- `license_plate` / `user` — nullable foreign keys; null for unregistered (guest) plates
- `entry_time` / `exit_time` — null `exit_time` means the session is still active
- `status` — `active`, `completed`, or `void`
- `charge_amount` — final calculated charge in dollars
- `has_duplicate_warning` — set when a new session is created while the plate was already active
- `was_orphaned` — set on the old session that was voided by a re-entry

<details>
<summary><strong>Orphan Handling</strong></summary>

If a plate triggers an entry event while it already has an active session, the system assumes the exit was missed (e.g., camera outage). The old session is voided (`was_orphaned=True`, `status="void"`) and a new session is opened (`has_duplicate_warning=True`). No charge is issued on the voided session.

</details>

### PlateDetectionEvent

Audit trail for every image the CV pipeline processes.

- `image` — the uploaded plate photo stored on disk
- `raw_plate_text` — exactly what the model read, before normalization
- `confidence_score` — model confidence (0.0 – 1.0)
- `event_type` — `entry` or `exit`
- `is_low_confidence` — true if score is below the lot's threshold
- `bounding_box` — normalized `[x, y, w, h]` coordinates as JSON
- `manually_corrected` / `corrected_plate` — operator correction fields

## Web Application

Django 5.1 backend with HTMX for reactive partials and Chart.js for revenue visualization. No Node.js, no React — server-rendered templates with targeted DOM swaps.

All pages require authentication. No public routes.

## Docker

The application runs as two containers orchestrated by Docker Compose:

1. **db** — PostgreSQL 16 with a persistent named volume
2. **web** — Django served by Gunicorn on port 8000

```bash
# Start all services
docker-compose up --build

# Run migrations
docker-compose exec web python manage.py migrate

# Seed initial data (creates default lot and settings)
docker-compose exec web python manage.py setup_defaults

# Create an admin user
docker-compose exec web python manage.py createsuperuser

# Run the test suite
docker-compose exec web pytest --cov=apps/accounts --cov=apps/parking --cov-fail-under=80
```

## Project Structure

```
parking-tracker/
├── docker-compose.yml
├── Dockerfile
├── manage.py
├── requirements.txt
├── .env.example
├── config/
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
└── apps/
    ├── accounts/
    │   ├── models.py          (custom User model)
    │   ├── admin.py
    │   └── tests/
    ├── parking/
    │   ├── models.py          (LicensePlate, ParkingLot, LotSettings, ParkingSession, PlateDetectionEvent)
    │   ├── admin.py
    │   └── tests/
    ├── cv/                    (computer vision — in progress)
    │   └── weights/           (trained model weights, gitignored)
    └── dashboard/             (views and API endpoints — in progress)
```
