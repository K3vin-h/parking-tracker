# Parking Lot Tracker

A self-service parking lot system modeled on Chinese license-plate-recognition (LPR) gates. A from-scratch PyTorch plate reader opens and closes sessions at an unmanned kiosk, and registered plates are billed automatically to the owner's prepaid wallet.

<a href="docs/images/demo.gif"><img src="docs/images/demo.gif" alt="Walkthrough: login, plate registration, kiosk activation, photo upload, entry and exit scans, and wallet billing" width="100%"></a>

[View the demo at full size](docs/images/demo.gif).

*An 87-second walkthrough of login, plate registration, kiosk activation, photo selection, entry and exit scans, and the final wallet charge. The demo starts with a preloaded $25 wallet; one $0.50 charge leaves $24.50. Screens are held longer for readability. The plate photos are selected synthetic inputs that the current model reads correctly.*

**Recognition limitation:** Clearer or more realistic-looking photos are not necessarily easier for this model. During demo preparation, it misread all three cleaner AI-generated test images, with confidence scores above the low-confidence threshold. This small check is not a benchmark, but it shows that confidence does not guarantee a correct plate. The recorded held-out synthetic evaluation achieved only 22% exact end-to-end reads; accuracy on real camera photos is unmeasured. The GIF demonstrates the application flow with selected successful inputs, not reliable real-world recognition. See [known CV limitations](docs/technical/08-limitations.md#cv-models).

## Why

Camera-based parking systems in Chinese cities inspired me to try building my own version. I wanted to understand the whole process: taking an image, locating the plate, extracting its text, and connecting that result to a backend API that manages parking sessions and billing.

Instead of using a ready-made OCR system or a cloud recognition API, I built and trained my own detector and recognizer using PyTorch, with OpenCV and Pillow for image processing. The goal was to learn each stage from scratch and understand how the pieces work together, including what happens when a plate is misread.

## Features

- **Unmanned gate kiosk.** Upload a plate photo at an entry or exit lane to open or close a parking session.
- **Custom CV pipeline.** A CNN plate detector plus a CRNN + CTC plate recognizer, built and trained from scratch with no external CV APIs.
- **Auto-pay.** A registered plate's charge is deducted from the owner's wallet on exit. Guests see the amount due at the kiosk.
- **Configurable billing.** Per-minute or per-hour rates, a grace period and a daily cap, set per lot.
- **Resident self-service.** Sign up, link plates, and view the wallet balance and ledger history.
- **Staff oversight.** A live dashboard, session log, and a correction queue for low-confidence or unmatched reads.
- **Revenue analytics.** Daily, per-lot and per-hour charts.
- **Privacy defaults.** Plate images are kept in private storage, the kiosk returns reduced responses, and a cleanup command enforces image retention when scheduled.

## How it works

```mermaid
flowchart LR
    Kiosk["Gate kiosk"] -->|"plate photo"| Django["Django backend"]
    Django --> CV["PyTorch detector + recognizer"]
    Django <--> DB[("PostgreSQL")]
    Django -->|"auto-pay on exit"| Wallet["Resident wallet"]
    Staff["Staff dashboard<br/>HTMX + Chart.js"] <--> Django
```

A driver uploads a plate photo at the kiosk, which stands in for the lane camera. The upload is validated and stored privately. The CV pipeline finds the plate, reads its text, and scores its confidence. The session layer opens a session on entry. On exit, it closes the session, calculates the charge, and debits the registered owner's wallet in the same transaction. Low-confidence reads are flagged for staff review, and the kiosk asks the driver to retake the photo or contact an attendant. This project simulates the gate workflow; it does not control a physical barrier.

A detailed technical walkthrough is in [`docs/technical/`](docs/technical/00-architecture.md).

## Tech stack

| Layer | Technology |
|---|---|
| CV models | PyTorch, OpenCV, Pillow (custom CNN detector + CRNN/CTC recognizer) |
| Backend | Django 5.2 LTS |
| Database | PostgreSQL 16 |
| Frontend | Django templates, HTMX, Chart.js (self-hosted, no Node build) |
| Deployment | Docker Compose, Gunicorn |
| CI | GitHub Actions |

## Setup

Requires Docker with Compose ≥ 2.24. Python 3.11+ is needed only to train the CV models outside Docker.

Create a `.env` file from `.env.example`, then replace its placeholder credentials before starting the app. It stores the Django secret key, database credentials, `DEBUG`, and the kiosk activation token. Generate a separate random value for each secret; `openssl rand -hex 32` works for the secret key and kiosk token.

```bash
cp .env.example .env
docker compose up --build -d
docker compose exec web python manage.py setup_defaults     # admin account, default lot, billing settings
```

Migrations run automatically when the container starts. Open `http://localhost:8000/`:

- the kiosk is at `/` (activate it with the token from `.env`);
- residents sign up at `/register/`;
- staff land on `/staff/` after logging in.

Run the tests:

```bash
docker compose exec web pytest
docker compose exec web pytest --cov=apps/accounts --cov=apps/parking --cov-fail-under=80
```

### CV model weights

Scans need both trained weight files in `apps/cv/weights/` (gitignored): `detector.pth` and `recognizer.pth`. Generate synthetic data and train them outside Docker. Training uses MPS on Apple Silicon or CUDA when available. The detector dataset needs parking-lot photos in `data/backgrounds/`, and plates render best with the font described in `apps/cv/training/assets/README.md`.

Use weights trained with the matching model architecture and preprocessing version. This checkout includes the five-block detector used in the demo. The quick-start commands below split generated samples into training and validation sets; use the [held-out-background workflow](docs/technical/02-cv-training.md#training-the-models) for a stronger evaluation and mixed recognizer crops.

In a Python virtual environment, install the training dependencies with `python -m pip install -r requirements-dev.txt` before running these commands.

```bash
python -c "from apps.cv.training.synthetic_data import generate_detector_dataset; generate_detector_dataset(n=2500, output_dir='data/detector', bg_dir='data/backgrounds')"
python -c "from apps.cv.training.synthetic_data import generate_recognizer_dataset; generate_recognizer_dataset(n=8000, output_dir='data/recognizer')"
python apps/cv/training/train_detector.py --epochs 40 --data-dir data/detector --output apps/cv/weights/detector.pth
python apps/cv/training/train_recognizer.py --epochs 20 --data-dir data/recognizer --output apps/cv/weights/recognizer.pth
```

Each script saves a training-curve plot next to its weights. These plots show the experimental retraining runs:

![Plate detector training curves](docs/images/detector_training.png)
![Plate recognizer training curves](docs/images/recognizer_training.png)

Experimental retraining results (synthetic data, with separate held-out backgrounds for detector and end-to-end evaluation):

- **Detector:** 0.60 IoU, below its 0.70 target (up from 0.43).
- **Recognizer:** 90.5% character accuracy, 59.1% full-plate accuracy.
- **End to end:** 22% of plates read exactly right through the whole pipeline (up from 0%).

See [CV Model Status](docs/technical/01-cv-pipeline.md#cv-model-status) for what the results mean and what to try next, and [CV Training Data](docs/technical/02-cv-training.md#training-the-models) for the experimental training workflow.

### Production

Set `DEBUG=False`, your deployment hostname in `ALLOWED_HOSTS`, and a separate `HEALTH_CHECK_TOKEN` before starting production mode. Keep the kiosk activation token and all other credentials unique to the deployment.

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up --build -d
```

This runs Gunicorn, drops the dev bind mount and binds only to `127.0.0.1`, so put it behind a reverse proxy. Schedule `python manage.py cleanup_old_images` nightly to enforce image retention. See [`docs/technical/07-security-and-deployment.md`](docs/technical/07-security-and-deployment.md).

## API reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Gate kiosk |
| POST | `/kiosk/activate/` | Exchange the kiosk token and lane scope for a browser capability |
| POST | `/kiosk/scan/` | Run CV on an uploaded plate and open/close a session (privacy-reduced response) |
| GET/POST | `/register/` | Resident signup (always non-staff), provisions a wallet |
| GET/POST | `/plates/` | List or add the signed-in user's plates |
| POST | `/plates/<id>/delete/` | Delete one of the signed-in user's plates |
| GET | `/wallet/` | Balance and ledger history |
| GET/POST | `/wallet/topup/` | Top up through the payment connector (placeholder, fails closed) |
| GET | `/staff/`, `/staff/log/`, `/staff/errors/`, `/staff/revenue/`, `/staff/settings/` | Staff pages |
| GET | `/staff/api/sessions/`, `/staff/api/dashboard-stats/`, `/staff/api/revenue-data/` | HTMX/JSON data for staff pages |
| PATCH | `/staff/api/events/<id>/correct/` | Correct a queued plate read and reconcile its session |
| GET | `/staff/api/events/<id>/image/` | Stream a detection image privately |

Resident routes require login, and `/staff/` routes require `is_staff`. Kiosk, signup, top-up, login (including the admin login), and password-reset routes are rate-limited per IP.

## Limitations and roadmap

This is a portfolio-scale system, and several choices reflect that:

- the kiosk takes an uploaded photo instead of a camera feed;
- inference runs in the request, with no task queue;
- there is one global staff role;
- the payment connector is a placeholder that refuses to create credit.

The CV models were trained only on synthetic data, and the detector is the current accuracy bottleneck. The full list, and what changing scope would require, is in [`docs/technical/08-limitations.md`](docs/technical/08-limitations.md).

## Security notes

Two layers check every public upload before any decode:

- **Web layer:** declared MIME type, Pillow header, 10 MB and 12 MP caps, randomized names in private storage.
- **CV layer:** path containment, a content-based format allowlist and a bounded single read.

Model weights load with `torch.load(..., weights_only=True)` and a checked preprocessing version. Plate images are served only to authenticated staff, with `Cache-Control: private, no-store`. Kiosk scans need an activated capability plus a single-use nonce. Wallet money is `Decimal`-only and row-locked, with `balance == SUM(ledger)` as an invariant. Production enforces CSP, HSTS and secure cookies. Details are in [`docs/technical/07-security-and-deployment.md`](docs/technical/07-security-and-deployment.md).

## License

MIT. See [LICENSE](LICENSE).
