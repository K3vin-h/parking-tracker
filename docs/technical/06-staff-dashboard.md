# Web Application and Staff Dashboard

Django 5.2 LTS backend with server-rendered templates, HTMX for targeted live updates, and Chart.js for revenue visualization. HTMX and Chart.js are self-hosted under `static/js/vendor/`; the application does not require Node.js or React.

`templates/base.html` provides the responsive sidebar, top bar, active navigation, queue badge, flash messages, and the shared self-hosted HTMX asset.

## Pages

| Audience | Page | URL | Main features |
|----------|------|-----|---------------|
| Public | Gate kiosk | `/` | Token activation, entry/exit lane selection, and plate-image scanning |
| Public | Resident registration | `/register/` | Creates an unprivileged resident account |
| Public | Login | `/login/` | Shared sign-in with role-based post-login routing |
| Resident | Plates | `/plates/` | Add and remove registered licence plates |
| Resident | Wallet | `/wallet/` | View balance and wallet ledger activity |
| Resident | Wallet top-up | `/wallet/topup/` | Add funds through the configured payment connector |
| Staff | Dashboard | `/staff/` | Live summary cards, active sessions, running charges, revenue, and traffic |
| Staff | Session log | `/staff/log/` | Plate/status/lot/date filters, session tabs, charges, and pagination |
| Staff | Error queue | `/staff/errors/` | Review private thumbnails and correct low-confidence or unmatched events |
| Staff | Revenue | `/staff/revenue/` | Date ranges, summary cards, daily charts, and lot/hour breakdowns |
| Staff | Settings | `/staff/settings/` | Configure rates, billing units, grace periods, caps, retention, and confidence |
| Administrator | Django Admin | `/admin/` | Manage users and registered database models subject to Django permissions |

The kiosk owns `/`; there is no standalone `/upload/` page. Every operator
page and supporting `/staff/api/` endpoint requires an authenticated account
with `is_staff=True`. The login and Django authentication routes remain public.

Staff and superusers see the same operator pages under `/staff/`. A superuser
also bypasses Django's model permission checks in `/admin/`; an ordinary staff
account can enter Django Admin only for models covered by permissions explicitly
granted to that account.

**Confidence indicator bands** are fixed across all pages:

- Green: ≥ 80%
- Yellow: 60–79%
- Red: < 60%

Authorization uses one global `is_staff` operator role. There is no per-lot tenant isolation — a staff user can access every configured lot.

## API Endpoints

Kiosk endpoints are capability-protected and rate-limited. Dashboard endpoints
are staff-only. The scan endpoint runs the full CV pipeline and creates the
session/event records. The image endpoint streams plate images privately; they
are never served via a public media URL.

| Method | URL | Purpose |
|--------|-----|---------|
| POST | `/kiosk/activate/` | Exchange the environment token and lane scope for a revocable browser capability |
| POST | `/kiosk/scan/` | Validate a plate image, consume a kiosk nonce, run CV, and create an entry/exit event |
| GET | `/staff/api/sessions/` | Return the filtered, paginated HTMX session table |
| GET | `/staff/api/dashboard-stats/` | Return the live dashboard region polled every 10 seconds |
| PATCH | `/staff/api/events/<id>/correct/` | Correct a queued plate and reconcile its session |
| GET | `/staff/api/revenue-data/` | Return exact-money summary, daily, lot, and hourly chart data |
| GET | `/staff/api/events/<id>/image/` | Stream a detection image privately to authenticated staff |

The dashboard API module is split across four files for clarity: `api.py`
(shared `staff_required` decorator), `partials_api.py`
(sessions/stats/correct), `revenue_api.py`, and `image_api.py`. Public kiosk
activation and scanning live in `apps/public/scan.py`.

## Scheduled Maintenance

`cleanup_old_images` deletes uploaded plate images older than each lot's `image_retention_days` setting. It clears the `image` field on the `PlateDetectionEvent` row but **keeps** the session and event records intact for billing and audit purposes.

A lot with `image_retention_days = NULL` is treated as "keep forever" and is skipped entirely.

**Preview without deleting (always safe to run):**

```bash
docker-compose exec web python manage.py cleanup_old_images --dry-run
```

**Host crontab — run nightly at 02:00:**

```
0 2 * * * cd /path/to/parking-tracker && \
    docker compose -f docker-compose.yml -f docker-compose.prod.yml exec -T web \
    python manage.py cleanup_old_images
```

Orchestrators can use a native scheduler instead — for example, a Kubernetes `CronJob` running `python manage.py cleanup_old_images` directly in the web pod.

## See also

- [00-architecture.md](00-architecture.md): system overview and module boundaries
- [08-limitations.md](08-limitations.md): known constraints and roadmap
