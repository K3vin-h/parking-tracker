# Day 9 and Day 10 Explained: Operator Dashboard

Branch: `feat/frontend-base`

Pull request: https://github.com/K3vin-h/parking-tracker/pull/11

Main commits:

```text
e0ee836 fix: harden frontend base integrations
f306525 feat: complete operator dashboard
65383b7 fix: complete Day 9 and Day 10 checklist
```

## The Short Version

Day 9 creates the visual foundation for the parking tracker.

Day 10 turns that foundation into the complete operator dashboard.

Together they add:

```text
staff login
-> responsive application shell
-> live parking dashboard
-> image upload and CV result
-> searchable session history
-> manual error correction
-> revenue analytics
-> per-lot settings
```

The frontend is built with:

```text
Django templates
+ HTMX
+ Chart.js
+ small framework-free JavaScript helpers
```

There is no React application, Node.js build, or public CDN dependency.

## What Problem Do Day 9 and Day 10 Solve?

Before this work, the project already had the important backend pieces:

```text
CV pipeline
-> reads a plate

parking services
-> opens or closes a session

database models
-> store events, sessions, charges, and settings
```

But an operator still needed a safe way to use and inspect those systems.

Day 9 and Day 10 add that operating surface:

```text
operator action
-> staff-only Django page
-> HTMX or JSON endpoint
-> existing CV/service/query logic
-> focused UI update
```

The frontend does not duplicate billing or recognition rules. It presents and
controls the backend that already exists.

## Day 9: Design Handoff and Frontend Foundation

Day 9 imports the visual direction from the supplied Claude Design project and
turns it into reusable Django infrastructure.

### Base Application Shell

`templates/base.html` owns the structure shared by every operator page:

- responsive sidebar navigation
- active-page highlighting
- queue-count badge
- configured lot count
- operator identity
- POST-only logout form
- flash messages
- main content region

On narrow screens, the sidebar becomes a drawer. `static/js/app.js` keeps the
visual open state and `aria-expanded` accessibility state synchronized. The
drawer closes from the scrim, a navigation link, or the Escape key.

### Design System

`static/css/main.css` contains the visual system instead of scattering styles
through templates:

- dark background layers
- primary and secondary text colors
- blue action color
- active, warning, error, and void state colors
- monospace data styling
- cards, tables, badges, buttons, form controls, empty states, and responsive
  layout rules

This keeps all Day 10 pages visually consistent with the imported design.

### Self-Hosted Browser Assets

HTMX and Chart.js are stored in the repository:

```text
static/js/vendor/htmx-2.0.10.min.js
static/js/vendor/chart-4.5.1.umd.min.js
```

Self-hosting avoids a runtime CDN dependency and gives the application a fixed,
reviewable frontend version. Chart.js is loaded only by the revenue page.

### Login Page

`templates/registration/login.html` uses the same design language as the
dashboard while remaining outside the authenticated application shell.

It includes:

- username and password fields
- Django authentication error feedback
- safe handling of the `next` redirect
- CSRF protection

## Day 10: Complete Operator Pages

Day 10 connects real database data and APIs to every page in the design.

## Dashboard

URL:

```text
/
```

Main files:

```text
templates/dashboard.html
templates/partials/dashboard_stats.html
apps/dashboard/views.py
GET /api/dashboard-stats/
```

The dashboard shows:

- active session count
- today's completed-session revenue
- today's entry and exit counts
- average stay
- registered and guest active-session counts
- running duration and estimated charge for active sessions
- the ten most recent detection events

The live dashboard region uses HTMX polling every 10 seconds. The server
recalculates active durations and running charges from one UTC timestamp, using
the existing `calculate_charge()` service so the screen cannot drift from the
real billing rules.

## Upload

URL:

```text
/upload/
```

Main files:

```text
templates/upload.html
templates/partials/upload_result*.html
static/js/upload.js
POST /api/upload/
```

The operator chooses:

- entry or exit
- parking lot
- JPEG or PNG image

The form submits through HTMX and replaces only the result panel.

The endpoint keeps two response contracts:

- normal clients receive JSON
- HTMX requests receive an HTML result partial

If an entry image cannot produce plate text, no parking session is opened. The
server stores a private, sessionless, low-confidence event for review. JSON
clients receive HTTP 422, while HTMX receives an HTML 200 response so the
operator can see the unreadable result in the page.

The result can show:

- detected plate text
- confidence score and color band
- event and session status
- charge information for a completed exit
- the uploaded image
- the detected plate bounding box

`static/js/upload.js` adds drag-and-drop and a local preview. The server remains
the authority for all validation.

### Bounding-Box Canvas

The CV pipeline stores a normalized box:

```text
[x, y, width, height]
```

`static/js/app.js` draws that box on a canvas over the protected image. The
calculation accounts for `object-fit: contain`, including empty letterbox space,
so the box stays aligned when the image and result panel have different aspect
ratios.

The canvas redraws after:

- initial page load
- HTMX result swaps
- browser resize

## Session Log

URL:

```text
/log/
```

Main files:

```text
templates/log.html
templates/partials/session_table.html
GET /api/sessions/
```

The log provides:

- plate search
- status filter
- lot filter
- UTC entry-date range
- All, Registered, and Guest tabs
- 25-row pagination
- live duration and estimated charge for active sessions

The page and HTMX endpoint share `build_session_context()`. This prevents the
initial server render and later filtered results from implementing different
query rules. Date filters apply to `entry_time`; they do not select by exit time
or by any session that merely overlaps the selected period.

## Error Queue

URL:

```text
/errors/
```

Main files:

```text
templates/errors.html
templates/partials/queue_row.html
PATCH /api/events/<id>/correct/
GET /api/events/<id>/image/
```

The queue contains unresolved events that are:

- below the lot confidence threshold, or
- unmatched to a parking session

Each row shows the protected thumbnail, detected text, confidence, event type,
lot, and time. The operator can correct the plate inline.

The queue is ordered oldest first and paginated at 25 events per page.

The correction endpoint:

1. accepts only form-encoded PATCH requests
2. locks and rechecks the event inside a transaction
3. rejects events that are already resolved
4. calls the existing `correct_plate()` service
5. returns an HTMX confirmation row
6. broadcasts the authoritative remaining queue count

The navigation badge updates from the server count rather than subtracting
locally, which remains correct when multiple operators work simultaneously.

## Revenue Analytics

URL:

```text
/revenue/
```

Main files:

```text
templates/revenue.html
static/js/revenue.js
GET /api/revenue-data/
```

The operator can choose:

- 7 days
- 30 days
- 90 days
- custom range up to 366 days

The API returns:

- total revenue
- completed session count
- average charge
- average session duration
- zero-filled daily revenue
- revenue grouped by lot
- zero-filled 24-hour distribution

The current page shows all lots and provides a Lot/Hour breakdown switch. The
API supports an optional lot query for other authorized consumers, but the
current revenue template does not expose a lot selector.

Money is serialized as an exact two-decimal string. JavaScript converts it only
for chart rendering and display; billing calculations remain `Decimal` on the
server.

Chart.js renders:

- a daily revenue line chart
- a bar chart switchable between lot and hour

## Settings

URL:

```text
/settings/
```

Main files:

```text
templates/settings.html
apps/dashboard/forms.py
apps/dashboard/views.py
```

The settings page edits one parking lot at a time:

- rate
- hourly or minute billing unit
- grace period
- daily cap toggle and amount
- image retention period
- confidence threshold slider

The operator sees confidence as a percentage, while the model stores a value
between `0.0` and `1.0`. `LotSettingsForm` performs the conversion and validates
cross-field rules. A disabled daily cap clears the stored cap amount so old
values cannot be mistaken for an active policy.

## Confidence Bands

Display colors use fixed thresholds:

| Score | UI band | Meaning |
|-------|---------|---------|
| `>= 0.80` | Green | Good recognition confidence |
| `>= 0.60` and `< 0.80` | Yellow | Warning |
| `< 0.60` | Red | Poor recognition confidence |

These bands control presentation. Whether an event enters the review queue is
still determined by the configurable per-lot confidence threshold.

## Authentication and Security

Every Day 9–10 operator page and dashboard API is staff-only. The login page is
public because it is the entry point used to establish the staff session.

The authorization model is one global `is_staff` operator role. There is no
per-lot tenant isolation, so a staff user can view and configure every lot.

Anonymous users and authenticated non-staff users are redirected through the
login flow. Important protections include:

- Django CSRF validation on forms and state-changing APIs
- POST-only logout
- PATCH-only manual correction
- upload MIME, Pillow structure, and format verification before CV decode
- 10 MB compressed upload limit
- pre-decode 12 MP dimension limit
- randomized upload names
- path-containment checks before CV processing
- private event-image endpoint
- `Cache-Control: private, no-store` for plate images
- no public media URL for detection images
- template auto-escaping for operator-visible values
- ORM queries rather than interpolated SQL

The deployment must also keep `MEDIA_ROOT` and any object-storage bucket private.
A reverse proxy or bucket policy that serves those files directly would bypass
the protected Django image endpoint.

## Main Files

```text
apps/dashboard/api.py
apps/dashboard/forms.py
apps/dashboard/urls.py
apps/dashboard/views.py

templates/base.html
templates/dashboard.html
templates/upload.html
templates/log.html
templates/errors.html
templates/revenue.html
templates/settings.html
templates/registration/login.html
templates/partials/

static/css/main.css
static/js/app.js
static/js/upload.js
static/js/revenue.js
static/js/vendor/
```

Tests live in:

```text
apps/dashboard/tests/test_dashboard_api.py
apps/dashboard/tests/test_forms.py
apps/dashboard/tests/test_upload_api.py
apps/dashboard/tests/test_views.py
```

## How To Check Day 9 and Day 10

Start the stack:

```bash
docker-compose up --build
docker-compose exec web python manage.py migrate
docker-compose exec web python manage.py setup_defaults
```

`setup_defaults` creates the initial superuser from the configured
`DEFAULT_SUPERUSER_EMAIL` and `DEFAULT_SUPERUSER_PASSWORD`. To create a different
operator interactively instead, run:

```bash
docker-compose exec web python manage.py createsuperuser
```

Open:

```text
http://localhost:8000/login/
```

Then check:

1. `/` — cards, recent events, active sessions, and 10-second updates.
2. `/upload/` — drag/drop, entry/exit, HTMX result, confidence, and canvas box.
3. `/log/` — every filter, registration tab, pagination, and running values.
4. `/errors/` — private image, inline correction, row removal, and badge update.
5. `/revenue/` — presets, custom dates, summary cards, and both breakdowns.
6. `/settings/` — lot selection, validation, slider output, save, and reload.
7. Mobile widths — navigation drawer at 768 and 320 pixels.
8. Authorization — verify a non-staff user cannot open any operator page or API.

Real upload inference requires:

```text
apps/cv/weights/detector.pth
apps/cv/weights/recognizer.pth
```

Without configured model weights, HTTP 503 from the upload endpoint is expected.

Run the automated verification:

```bash
docker-compose exec web pytest
docker-compose exec web pytest --cov=apps/accounts --cov=apps/parking --cov-fail-under=80
docker-compose exec web python manage.py check
docker-compose exec web python manage.py collectstatic --noinput --dry-run
```

The latest completed verification passed the full suite and measured
accounts/parking coverage at 92.91%, above the required 80% gate. Rerun the
commands above after rebasing because the exact test count can change as later
days add coverage.
