# Full-Codebase Review — 2026-06-09

Branch: `feat/cv-inference-pipeline` (PR #8 open). Agents: security-reviewer, python-reviewer, silent-failure-hunter, database-reviewer (all on claude-fable-5).

**Verdict: 0 CRITICAL. 10 HIGH, 13 MEDIUM, 8 LOW.** No exploitable vulnerability today — the two security HIGHs become real when `/api/upload/` ships (Day 8). One python-reviewer CRITICAL (CRNN reshape) was self-retracted after tracing the memory layout — the reshape is correct.

---

## HIGH

| # | Finding | Location | Source |
|---|---------|----------|--------|
| H1 | `FileNotFoundError` in pipeline `__init__` embeds weight file paths in message → CWE-209 path disclosure when a future view serializes it | `apps/cv/pipeline.py:130-138` | security |
| H2 | `apps` logger pinned at DEBUG with no production guard → future `logger.debug` of plate text / charges leaks PII to aggregated logs (CWE-532) | `config/settings.py:327-344` | security |
| H3 | `torch.load` + `load_state_dict` unguarded — corrupt/truncated/incompatible `.pth` crashes first request with raw traceback, no actionable context | `apps/cv/pipeline.py:144-156` | security + silent-failure |
| H4 | `subprocess.Popen(["open", ...])` inside swallow-all `except Exception` — fails silently every run on Linux/CI; masks savefig failures | `train_detector.py:577-585`, `train_recognizer.py:545-551` | silent-failure |
| H5 | Detector dataset gen only raises if **zero** samples generated — 90% skip rate completes "successfully" → silently undertrained model | `apps/cv/training/synthetic_data.py:419-450` | silent-failure |
| H6 | Recognizer dataset gen has no `generated_count` at all — can return with only a CSV header, failure surfaces later as confusing "Dataset is empty" in training | `apps/cv/training/synthetic_data.py:493-524` | silent-failure |
| H7 | Char-accuracy metric is positional `zip()`, not edit distance — one shifted char scores ~0/6; metric materially under-reports model quality | `train_recognizer.py:196-198` | python |
| H8 | Double-`try` TypeError probe in `__getitem__`: a transform that raises TypeError internally silently drops the bbox transform → augmented image with misaligned label, no log | `apps/cv/training/dataset.py:216-228` | silent-failure |
| H9 | No CHECK constraint on `confidence_score` (0–1) or `charge_amount >= 0` — bulk ops/raw SQL bypass Django validators; negative charge corrupts revenue | `apps/parking/models.py:399,533` | database |
| H10 | `ParkingSession.lot` is `CASCADE` — deleting a lot silently wipes billing history (sessions + events). Should be `PROTECT` | `apps/parking/models.py:359` | database |

## MEDIUM

| # | Finding | Location |
|---|---------|----------|
| M1 | `train_detector.py` missing the `sys.path` insert `train_recognizer.py` has — fails with ModuleNotFoundError when run as documented | `train_detector.py` |
| M2 | `/health/` is an unauthenticated DB-connectivity oracle, no rate limit | `config/urls.py:28-46` |
| M3 | Healthcheck trusts spoofable `X-Forwarded-Proto: https`; use `SECURE_REDIRECT_EXEMPT = [r'^health/$']` instead | `Dockerfile:107` |
| M4 | Degraded pipeline results (tiny-bbox, crop-failure → empty plate) logged at DEBUG — invisible in production | `apps/cv/pipeline.py:210-241` |
| M5 | Path-traversal rejection raises plain `ValueError` — docstring pattern invites views to swallow it; wrap in distinct exception → HTTP 400 | `apps/cv/preprocessing.py:112-113` |
| M6 | `crop_plate_region` uses `int()` truncation not `round()` — sub-pixel bboxes degenerate to w_px=0 | `apps/cv/preprocessing.py:474-477` |
| M7 | Missing partial indexes on active sessions (`plate_text WHERE status='active'`, `lot WHERE status='active'`) — the entry/exit + dashboard hot path | `models.py:446-459` |
| M8 | Missing FK indexes: `ParkingSession.user`, `ParkingSession.license_plate`, `PlateDetectionEvent.session` | `models.py:334,346,494` |
| M9 | Admin N+1: `LicensePlateAdmin`, `LotSettingsAdmin`, `ParkingSessionAdmin` lack `list_select_related` | `apps/parking/admin.py:34,53,78` |
| M10 | Missing CheckConstraints: `exit_time > entry_time`, `duration_seconds >= 0`, void ⇒ charge 0 | `models.py:376-418` |
| M11 | `setup_defaults.handle()` not wrapped in `transaction.atomic()` — partial-init state possible and masked on re-run | `setup_defaults.py:61` |
| M12 | `entrypoint.sh` DB probe: bare `except: sys.exit(1)` with stderr suppressed — misconfig indistinguishable from "not ready yet" | `entrypoint.sh:33` |
| M13 | `_load_font` lru_cache: degraded-font warning logged once, entire 50k-sample run uses fallback font with no summary notice | `synthetic_data.py:123-144` |

## LOW

- L1 Superuser email printed to stdout (`setup_defaults.py:108`)
- L2 `.env` readable in container via `.:/app` bind mount (dev-only tradeoff, don't replicate in prod)
- L3 `unique_together` deprecated → `UniqueConstraint` (`models.py:136`)
- L4 Redundant standalone `db_index=True` on `status` (`models.py:418`)
- L5 No partial index for error queue (`is_low_confidence=True, manually_corrected=False`) (`models.py:551`)
- L6 `ParkingLot.name` not unique — `setup_defaults` get_or_create relies on it (`models.py:161`)
- L7 `DetectorAugment` grayscale comment factually wrong (normalize-then-grayscale ≠ equal channels) (`augment.py:46-56`)
- L8 Confidence calc `squeeze(1)` fragile if ever batched >1 (`pipeline.py:259`)

## Clean / done well

- `load_image()` hardening fully verified: path guard (realpath + os.sep), no TOCTOU (single read into bytes), magic-byte format allowlist, 3-layer bomb defense, no path leaks
- `torch.load(weights_only=True)` ✓; singleton double-checked locking correct ✓; no PII in logs ✓
- Settings: fail-loud env secrets, DEBUG-gated HTTPS settings ✓. Docker: non-root, multi-stage, DB not exposed ✓
- Decimal everywhere for money ✓; SET_NULL on user/plate FKs preserves billing history ✓; composite indexes correctly ordered ✓
- `preprocessing.py`, both model files, `_image_io.py`, `augment.py`, `cleanup_old_images.py` — clean on error paths
