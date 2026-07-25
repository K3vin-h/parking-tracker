# Parking Tracker — Repository-Wide Code and Security Audit

Date: 2026-07-24
Scope: Current working tree, including modified, deleted, and untracked files
Method: Read-only source review, knowledge-graph queries, three independent specialist reviews, tests, coverage, Django checks, dependency audit, Ruff, Bandit, and AgentShield

## Remediation Update — 2026-07-24

The approved non-wallet remediation has now been implemented. This section
supersedes the original read-only status below.

| Finding | Resolution |
|---|---|
| H1 kiosk mutation/replay | Kiosk browsers now require a strong environment-only activation secret, are bound server-side to one lot and entry/exit direction, expire with the session, revoke on token rotation, and use database-row-locked one-time scan nonces. A bounded digest history rejects exact-image replay, including alternating-image sequences. Production rejects the published example token and secrets shorter than 32 characters. |
| H2 placeholder top-ups | **Deferred by explicit user instruction.** Payment/top-up behavior was left unchanged. |
| H3 correction charge reconciliation | **Deferred by explicit user instruction.** Wallet/ledger behavior was left unchanged. |
| H4 ambiguous plate ownership | Canonical plate text is globally unique at the database layer. Historical collisions abort migration with record IDs; model/form writes canonicalize before validation and save. Matching no longer chooses an arbitrary lowest-ID owner. |
| H5 ledger deletion | **Deferred by explicit user instruction.** Wallet/ledger schema and deletion behavior were left unchanged. |
| H6 admin lifecycle edits | Parking sessions and detection events are audit-only in admin: add, change, and delete are denied and service-owned fields are read-only. Wallet-admin behavior was intentionally untouched. |
| H7 invalid lot settings | Positive rates, coherent daily caps, confidence range, grace range, billing-unit choices, and retention bounds now have model validation and database checks. Migration preflight reports invalid record IDs before constraints are installed. |
| H8 unresolved upload growth | Kiosk authentication/rate limiting reduces anonymous creation, and unresolved evidence now has a configurable finite privacy ceiling even for lots whose resolved-image policy is “Forever.” |
| H9 plate normalization gap | One canonicalization function is enforced in model validation, model save, forms/services, and the collision-aware migration. |
| M1 login brute force | Login and password-reset POSTs use the shared database-backed limiter. |
| M2 proxy/per-worker limiter | Counters are shared in PostgreSQL, forwarded addresses are accepted only from configured proxy peers, errors fail closed, and expired rows are globally reclaimed in bounded batches. |
| M3 synchronous CV exhaustion | Access is now limited to activated, scoped kiosks and request volume is shared-worker throttled. A dedicated inference queue remains an optional deployment-scale improvement. |
| M4 signup/email race | Account plus companion provisioning is atomic; populated emails are normalized and case-insensitively unique in the database; user-identity races return a generic form error while unrelated provisioning failures remain explicit. Email ownership verification remains a product-level enhancement. |
| M5 upload commit leak | File ownership transfers only after `transaction.atomic()` exits successfully; commit failures delete the upload. |
| M6 wallet idempotency/concurrency | **Deferred by explicit user instruction.** Wallet/payment/ledger code and schema were left unchanged. |
| M7 bootstrap admin conflict | Bootstrap rejects a resident or username collision, validates privileged passwords against the prospective admin identity, handles concurrent conflicts explicitly, and stays idempotent for the matching superuser. |
| M8 unused augmentation | Detector and recognizer CLIs now use reproducible shared indices with stochastic train-only augmentation and deterministic validation normalization. |
| L1 public registration disclosure | Scan results are available only after scoped kiosk activation; owner identity and balance remain absent. |
| L2 shared temp directory | Remote-storage scratch roots reject symlinks/non-directories/wrong owners, repair mode to `0700`, create files privately, and remove copies after inference. |
| L3 static quality debt | The confirmed unused import was removed. Changed Python files pass Ruff (with the two direct-execution training scripts’ intentional `E402` bootstrap pattern excluded). |

Additional edge coverage was added for kiosk activation/scope/nonce/CSRF,
proxy trust, shared limiter persistence and expiry, login/reset throttling,
canonical collisions, invalid settings at model and database boundaries,
audit-only admin permissions, commit-time file cleanup, scratch-directory
safety, unresolved retention, signup rollback/races, bootstrap conflicts,
training split/augmentation behavior, and deterministic timestamp fixtures.

Final verification after remediation:

- 483 tests passed; one intentional Pillow decompression-bomb warning.
- Accounts + parking coverage: 96.23% (required: 80%).
- `makemigrations --check --dry-run`: no changes detected.
- Django `check` and production `check --deploy`: no issues.
- Production startup rejects the example kiosk token as intended.
- Ruff changed-scope check and `git diff --check`: passed.
- Knowledge graph refreshed: 2,934 nodes, 5,743 edges, 192 communities.
- `pip check` still reports the pre-existing platform mismatch for
  `nvidia-cusparselt-cu13`; application dependency auditing previously found no
  known vulnerable resolved production requirements.

## Executive Summary

The repository has strong foundations: all 439 tests pass, the database migration state is synchronized, production Django security checks pass, monetary calculations use `Decimal`, uploads are substantially hardened, and the core parking service has unusually good boundary coverage.

It is not ready for a real-money or internet-exposed deployment in its current working-tree state. The highest risks are:

1. The anonymous kiosk endpoint can mutate parking sessions and debit a resident's wallet from a replayed plate image.
2. The placeholder payment gateway lets any authenticated user mint unlimited wallet credit.
3. Correcting a completed session can change its owner without moving or reversing the wallet charge.
4. Multiple users may register the same plate, after which the lowest database ID is billed.
5. Admin and cascade behavior can bypass lifecycle rules, duplicate charges, or erase the financial ledger.

No committed secrets, SQL injection, template XSS, unsafe model deserialization, or public media route were found.

## High-Severity Findings

### H1. Anonymous plate replay can close sessions and debit another user's wallet

- Evidence:
  - `apps/public/scan.py:91-103`
  - `apps/dashboard/scan_core.py:413-454`
  - `apps/parking/services.py:513-520`
- Impact: Anyone with a photograph or generated image of a registered plate can submit an exit, close the active session, and debit the linked resident. Replayed entries can void legitimate sessions and open replacements. CSRF verifies browser origin; it does not authenticate a kiosk device.
- Recommendation: Authenticate each lane device with a scoped credential or mTLS. Bind lot and direction to the device server-side. Require signed requests with a short-lived timestamp and nonce, store processed nonces, and reject replayed scans.

### H2. Authenticated users can create unlimited wallet credit without payment

- Evidence:
  - `apps/parking/payments.py:35-55`
  - `apps/public/wallet_views.py:40-60`
- Impact: The placeholder gateway always succeeds. A user can repeatedly POST top-ups of up to $1,000 each and use the fabricated balance for parking.
- Recommendation: Disable top-ups unless a real payment backend is explicitly configured. Credit only after signature-verified server-side confirmation. Add unique provider event/reference constraints and idempotent callback processing.

### H3. Completed-session corrections leave charges on the wrong wallet

- Evidence:
  - Debit occurs at `apps/parking/services.py:513-520`.
  - Ownership can later change at `apps/parking/services.py:609-631`.
- Impact: Correcting a completed session from user A to user B leaves A charged. Correcting a registered session to a guest leaves the former owner charged. Correcting a completed guest session to a registered owner leaves the new owner uncharged.
- Recommendation: Either make financial ownership immutable after billing or atomically append reversal and replacement ledger entries. Never rewrite historical ledger rows.

### H4. Ambiguous plate registrations can bill the wrong resident

- Evidence:
  - Plate text is unique only per user at `apps/parking/models.py:126-145`.
  - Ambiguity is resolved by choosing the lowest primary key at `apps/parking/services.py:255-276`.
- Impact: Two residents may register the same normalized plate. Entry links and automatic billing then depend on registration order rather than verified ownership.
- Recommendation: Model one verified current owner for a normalized plate and jurisdiction. Enforce it in the database. Reject existing or future collisions into manual review rather than auto-billing.

### H5. The financial ledger can be erased through parent deletion

- Evidence:
  - User-to-wallet cascade at `apps/parking/models.py:770-775`.
  - Wallet-to-transaction cascade at `apps/parking/models.py:823-826`.
  - Wallet deletion remains enabled at `apps/parking/admin.py:171-186`.
- Impact: Deleting a user or wallet deletes its entire audit history. A later portal visit recreates a zero-balance wallet, erasing either funds or debt.
- Recommendation: Protect ledger parents from deletion, disable wallet deletion in admin, and implement account deactivation plus explicit archival/anonymization. Enforce append-only ledger permissions at the database layer where practical.

### H6. Admin lifecycle edits can bypass services and cause duplicate billing

- Evidence:
  - Session lifecycle fields remain editable at `apps/parking/admin.py:76-107`.
  - Event correction/link fields remain editable at `apps/parking/admin.py:110-137`.
- Impact: Changing a completed session back to active permits another exit and wallet debit. Directly editing event correction fields can remove an item from review without relinking its session or reconciling money.
- Recommendation: Make service-owned state, ownership, correction, and billing fields read-only. Expose explicit admin actions that call the transactional service layer. Add a database guarantee of at most one charge ledger entry per session.

### H7. Admin can save invalid billing settings

- Evidence:
  - Default admin form at `apps/parking/admin.py:57-73`.
  - Missing model invariants at `apps/parking/models.py:228-275`.
  - A missing cap amount is skipped at `apps/parking/services.py:225-235`.
- Impact: A negative rate causes exit transactions to fail against the non-negative charge constraint. A cap marked enabled with no amount silently produces uncapped billing.
- Recommendation: Add positive-rate and enabled-cap consistency validation to the model and database constraints. Reuse the validated dashboard settings form in admin.

### H8. Anonymous unresolved uploads can exhaust private storage

- Evidence:
  - Unreadable/unmatched uploads are retained at `apps/dashboard/scan_core.py:423-435` and `:455-473`.
  - Cleanup excludes unresolved/sessionless review events at `apps/parking/management/commands/cleanup_old_images.py:86-95`.
- Impact: Anonymous clients can continuously create retained files. At up to 10 MB per request, unresolved events can fill the media volume indefinitely.
- Recommendation: Authenticate kiosks, enforce per-device and global storage quotas, give unresolved scans a finite quarantine retention period, monitor capacity, and fail closed before the volume fills.

### H9. Plate normalization is documented but not enforced at the model boundary

- Evidence:
  - Canonical-value assumption at `apps/parking/models.py:98-104`.
  - Admin writes directly at `apps/parking/admin.py:33-46`.
- Impact: An admin-created value such as `abc 123` is stored verbatim while scans search for `ABC123`. A registered driver is treated as a guest and bypasses account billing.
- Recommendation: Normalize in model validation/save or a dedicated canonical field and enforce canonical uniqueness in the database.

## Medium-Severity Findings

### M1. Login has no brute-force protection

- Evidence: Standard login is mounted directly at `config/urls.py:95-108`; the custom limiter covers signup, kiosk scans, and top-ups only.
- Impact: Attackers can make unlimited resident and staff password guesses.
- Recommendation: Add shared-cache throttling by account and trusted client IP, progressive backoff or temporary lockout, security audit logging, and MFA for staff.

### M2. Rate limiting becomes global behind a proxy and remains per worker

- Evidence: `apps/public/ratelimit.py:10-19`, `:36-38`, and `:56-73`.
- Impact: When every request arrives from the reverse proxy address, one attacker can consume the shared limit for all users. With `LocMemCache`, each Gunicorn worker also enforces a separate counter. Cache errors fail open.
- Recommendation: Trust a client-IP header only when overwritten by a known proxy, use Redis or another shared atomic backend, add account/device keys, and enforce concurrency/rate limits at the reverse proxy.

### M3. Synchronous anonymous CV work can exhaust all web workers

- Evidence:
  - Inference at `apps/dashboard/scan_core.py:376-390`.
  - Public access at `apps/public/scan.py:91-103`.
  - Three-worker, 120-second Gunicorn configuration in `docker-compose.prod.yml:54-55`.
- Impact: Three concurrent scans can occupy every web worker; the request-rate limit permits more work than the server can process.
- Recommendation: Authenticate devices, queue inference behind a bounded worker pool, allow one active scan per device, and set reverse-proxy concurrency and request-body limits.

### M4. Signup is non-atomic and email identity is raceable/unverified

- Evidence:
  - Application-only email check at `apps/public/forms.py:46-51`.
  - Separate user, wallet, and login operations at `apps/public/registration.py:35-48`.
  - Email has no unique database constraint in `apps/accounts/migrations/0001_initial.py:28`.
- Impact: A wallet creation failure leaves a partial account. Concurrent requests can register the same email. Anyone can claim an unverified address and block its owner.
- Recommendation: Wrap account and wallet creation in one transaction, verify email before activation, add case-insensitive database uniqueness after deduplication, and handle `IntegrityError`.

### M5. Commit-time scan failures can orphan uploaded files

- Evidence: `apps/dashboard/scan_core.py:413-491`.
- Impact: `keep_file` is set before returning from inside `transaction.atomic()`. If commit fails, the response is HTTP 500 but cleanup is skipped even though the event rolled back.
- Recommendation: Build the outcome inside the transaction, leave the atomic block successfully, and only then set `keep_file` and return.

### M6. Wallet concurrency, retry idempotency, and rollback are not guaranteed by schema

- Evidence:
  - Row-lock logic at `apps/parking/wallet.py:69-84` and `:116-130`.
  - Transaction schema at `apps/parking/models.py:799-881`.
- Impact: The schema has no unique charge-per-session constraint or unique provider reference/event. Retries or lifecycle bypasses can create multiple debits for one session. Concurrency behavior is not tested with real PostgreSQL threads.
- Recommendation: Add conditional uniqueness for charge transactions by session, provider idempotency keys, sign/kind check constraints, and `TransactionTestCase` concurrency tests.

### M7. The bootstrap command can silently leave the installation without an admin

- Evidence: `apps/parking/management/commands/setup_defaults.py:128-130`.
- Impact: Any ordinary account with the configured email is reported as an existing superuser. Public signup can claim that address before bootstrap.
- Recommendation: Verify username and privilege flags. Raise `CommandError` for a conflicting non-superuser instead of skipping.

### M8. The implemented augmentation pipelines are not used by training commands

- Evidence:
  - Detector dataset construction at `apps/cv/training/train_detector.py:482`.
  - Recognizer dataset construction at `apps/cv/training/train_recognizer.py:498`.
- Impact: Documented training silently omits blur, lighting, perspective, and other synthetic-to-real augmentations, which can materially reduce real-world recognition accuracy.
- Recommendation: Use separate training and validation dataset instances, attach augmentation only to training, and keep deterministic preprocessing aligned with inference.

## Low-Severity Findings

### L1. Anonymous scan responses disclose registration status

- Evidence: `apps/public/scan.py:77-85` and `templates/public/kiosk_result.html:15-38`.
- Impact: A caller with a plate image learns whether it is registered and whether billing used an account.
- Recommendation: Return a generic gate result publicly and show account status only to the owner or trusted device UI.

### L2. Fixed shared temporary directory has avoidable local denial-of-service exposure

- Evidence: `config/settings.py:321-324` and `apps/dashboard/scan_core.py:242-252`.
- Impact: A hostile local process could pre-create or manipulate the shared scratch path. Temporary files themselves are mode `0600`, so no direct disclosure was established.
- Recommendation: Create and validate the directory at startup, verify ownership/mode and non-symlink status, or use a service-private runtime directory.

### L3. Static quality debt

- Ruff reported 112 diagnostics, mostly formatting, import ordering, and framework class-variable false positives. One definite production issue is the unused `LicensePlate` import at `apps/public/forms.py:16`.
- Recommendation: Add a project Ruff configuration, exclude generated migrations where appropriate, fix real diagnostics, and enforce it in CI.

## Edge-Case and Test-Coverage Gaps

These are coverage risks unless a confirmed defect is noted above.

### Upload hardening regression coverage was substantially reduced

The current working tree deletes 37 tests from `apps/dashboard/tests/test_upload_api.py`. The replacement public kiosk file has 11 tests and globally mocks storage and the CV pipeline at `apps/public/tests/test_kiosk_scan.py:91`.

Restore or port tests for:

- MIME spoofing, corrupt images, and decompression bombs;
- exact and oversized byte/dimension boundaries;
- rejection during streaming before storage;
- stored-size mismatch and remote-storage temporary copies;
- missing/corrupt weights and malformed model outputs;
- unsafe paths and decode failures;
- rollback plus file deletion on every failure;
- unreadable entries and unmatched exits;
- CSRF with `enforce_csrf_checks=True`.

### Private image serving is under-tested

`apps/dashboard/image_api.py` has 38% statement coverage.

Add tests for anonymous and non-staff denial, traversal/absolute names, unsupported extensions, missing objects, JPEG/PNG content types, download filename, `Cache-Control`, and absence of public `/media/` routing.

### Payment and wallet failure paths need integration coverage

Add tests for provider decline and exception, wallet failure after provider success, duplicate callback/retry, maximum and excessive-precision amounts, cross-user isolation, simultaneous first-wallet creation, simultaneous top-ups, duplicate exits, correction-versus-exit, and rollback after ledger creation.

### `setup_defaults` has no coverage

Test missing/weak credentials, password hashing, idempotency, preservation of existing settings, conflicting resident email, and rollback on partial failure.

### Real OCR quality and training are effectively unverified

- `train_recognizer.py`: 0% coverage.
- `train_detector.py`: 25% coverage.
- The real-weight pipeline test is conditional on gitignored weights and does not assert exact recognized text at `apps/cv/tests/test_pipeline.py:394`.

Add tiny deterministic training smoke tests, invalid hyperparameter tests, empty/single-sample validation, NaN loss handling, MPS CTC fallback, checkpoint selection, and known-image exact-text/accuracy gates using a reduced tracked fixture or CI artifact.

### No browser or JavaScript tests exist

Add browser coverage for kiosk upload/reset/drag-drop, HTMX swaps, CSRF, signup/login role routing, plate ownership, top-up history, correction queue updates, and staff/resident navigation. `static/js/kiosk.js` currently has no direct tests.

### Rate-limit behavior is incompletely tested

Test distinct client IPs, ignored spoofed forwarding headers, trusted proxy behavior, safe methods not counted, window rollover, expiry races, cache failure-open behavior, and separate signup/top-up/kiosk scopes.

## Verification Results

| Check | Result |
|---|---|
| Full test suite | 439 passed, 1 Pillow warning |
| Repository-wide statement coverage | 88% |
| Accounts + parking coverage gate | 94.47%; 126 passed; required 80% |
| Migration drift | No changes detected |
| Django production deployment check | No issues with `DEBUG=False` and required production token |
| `pip-audit -r requirements.txt` | 18 resolved dependencies; 0 known vulnerabilities |
| `pip check` in running container | Unsupported `nvidia-cusparselt-cu13` package reported on this platform |
| Bandit | 0 high; 1 medium heuristic for the configured `/tmp` scratch path; 7 low |
| AgentShield | Grade A, 100/100, 0 findings; only one agent-configuration file was in scope |
| Ruff | 112 diagnostics, overwhelmingly style/configuration noise; one confirmed unused import |

The development container's ordinary `manage.py check --deploy` produced the expected warnings because it runs with `DEBUG=True`. Re-running the check under the repository's production branch (`DEBUG=False`) produced no warnings.

## Notable Strengths

- Money and duration calculations consistently use `Decimal`, explicit rounding, and timezone-aware timestamps.
- Session mutations generally use transactions, row locks, and an active-session uniqueness constraint.
- Upload processing has streaming byte limits, format/MIME verification, decompression-bomb and dimension limits, randomized names, bounded decoding, path containment, and private authenticated image serving.
- Model weights are loaded with `weights_only=True`.
- Staff views and APIs consistently apply staff authorization; mutations apply CSRF and HTTP method restrictions.
- Templates retain Django auto-escaping; no application `mark_safe`, unsafe dynamic execution, or raw SQL was found.
- Error paths generally log or return explicit failures rather than silently swallowing exceptions.
- Billing boundaries, grace periods, caps, clock skew, orphan handling, CV preprocessing, dataset validation, and cleanup behavior have strong unit coverage.

## Recommended Remediation Order

1. Disable placeholder top-ups and restrict/authenticate the kiosk mutation endpoint.
2. Define verified, globally unambiguous plate ownership.
3. Reconcile wallet transactions during corrections and add idempotency/schema constraints.
4. Lock down admin lifecycle and ledger deletion paths.
5. Enforce billing and plate invariants at model/database boundaries.
6. Add shared proxy-aware rate limiting, login protection, bounded inference concurrency, and unresolved-file quotas.
7. Restore upload/image/payment/bootstrap/concurrency regression tests.
8. Connect training augmentations and establish a real OCR accuracy gate.

No production code was changed as part of this audit.
