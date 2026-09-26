# Sessions, Billing, and Data Model

The CV pipeline answers *"what plate is in this photo?"*. The session and billing layer (`apps/parking/services.py`) answers the next question: *"what should happen now?"* — open a session, close one and charge for it, void a duplicate, or flag a bad read for an operator. It is the bridge between CV output and the database models.

Every detection is routed to one of two entry points based on whether the car is arriving or leaving:

- **Entry** → `handle_entry()` — voids any prior active session for the same plate (missed exit), opens a new active `ParkingSession`, and records a `PlateDetectionEvent`.
- **Exit** → `handle_exit()` — if it matches an active session, bills it and completes the session; if no active session matches, records a flagged event (`session=None`) for the operator review queue.

This layer is **pure business logic** — it never loads CV model weights or calls the pipeline. The caller runs the pipeline first and passes the already-extracted detection data (`plate_text`, `confidence`, `bounding_box`, `image`, `lot`) into these functions. That keeps it fast and trivially unit-testable with no `.pth` files required.

Two rules hold throughout: **all money is `Decimal`, never `float`** (float rounding errors accumulate into wrong revenue totals), and **no silent failures** — every branch logs, returns an explicit value, or raises.

<details>
<summary><code>normalize_plate(raw_text)</code></summary>

Collapses a raw plate reading into one canonical matching key. CV output and human input vary in spacing and case — `"abc 123"`, `"ABC 123"`, and `" abc123 "` all mean the same car — so all whitespace is stripped and the result is uppercased (`"ABC123"`). Hyphens and other characters are kept: the project uses an **exact-match policy**, so `"ABC-123"` stays distinct from `"ABC123"` and the system never guesses that two similar plates are the same vehicle. Empty, `None`, or whitespace-only input returns `""` (and logs a warning) rather than crashing.

</details>

<details>
<summary><code>calculate_charge(entry_time, exit_time, lot_settings)</code></summary>

Turns parking duration into a charge in dollars, as a `Decimal`. It is pure (no database writes) and isolated so the one place a bug costs real money can be tested against every boundary. The duration is built from **integer seconds**, never `Decimal(float)`, so binary-float noise can never pollute the cents. Four rules apply, in order:

1. **Grace period** — duration at or under `grace_period_minutes` is free (`$0.00`).
2. **Per-minute billing** — `ceil(total_minutes) × rate`.
3. **Per-hour billing** — `ceil(total_hours) × rate`. The billed quantity always rounds **up** because a car that parks 61 minutes occupied the spot into a second hour.
4. **Daily cap** — if `daily_cap_enabled` and the charge exceeds `daily_cap_amount`, the cap wins. If the cap is enabled but no amount is set, the charge is **not** silently zeroed — it logs a warning and bills the uncapped amount. An unknown `billing_unit` falls back to per-hour with a loud log.

The final result is rounded to the cent (`ROUND_HALF_UP`) before returning.

</details>

<details>
<summary><code>handle_entry(plate_text, confidence, bounding_box, image, lot)</code></summary>

Opens an active session when a car arrives, and records the entry event. Wrapped in `transaction.atomic()` because it may void a prior session **and** create a new session **and** create a detection event — those must all commit together or not at all.

- **Low confidence** is judged against the lot's own `confidence_threshold` (configurable per lot), not the CV pipeline's fixed constant, so operators can tune sensitivity per lot.
- **Orphan handling** — if the plate already has an active session in this lot, a single atomic `UPDATE` voids it (`status="void"`, `charge_amount=0`, `was_orphaned=True`) and the new session is flagged `has_duplicate_warning=True`. One `UPDATE` statement leaves no race window for two concurrent entries.
- **Guest vs registered** — if the normalized plate matches a registered `LicensePlate`, the session links to that user; otherwise it's a guest (`user=None`).
- An **empty plate** after normalization raises `ValueError` — an empty key would "match" every other blank read and corrupt the orphan/billing logic.

</details>

<details>
<summary><code>handle_exit(plate_text, confidence, bounding_box, image, lot)</code></summary>

Closes the matching active session when a car leaves and bills it. Also `transaction.atomic()`. It locks the oldest active session for the plate with `select_for_update()` so a concurrent exit can't double-bill, ordered by `entry_time` for a deterministic choice.

- **Exit without entry** — if no active session matches, it does **not** auto-create one and does **not** raise. It records a flagged event with `session=None` and `is_low_confidence=True` (forced, so it always lands in the review queue) and returns `None`.
- **Clock-skew guard** — to satisfy the exit-after-entry and non-negative-duration database constraints even with clock skew or sub-second turnaround, the exit time is bumped to at least one second after entry, and duration is `max(1, ...)`.
- On success, sets `status="completed"`, `exit_time`, `duration_seconds`, and `charge_amount` (via `calculate_charge`), saving only those changed fields.
- **Auto-pay** — for a registered plate, `debit_wallet_for_session` deducts the charge from the owner's wallet inside the same transaction, so the exit and its payment commit together (see [05-kiosk-wallet-payments.md](05-kiosk-wallet-payments.md#wallet-ledger-appsparkingwalletpy)).

</details>

<details>
<summary><code>correct_plate(event_id, corrected_text)</code></summary>

Applies an operator's manual correction to a detection event that landed in the review queue. Also `transaction.atomic()`. It marks the event `manually_corrected`, updates the linked session's `plate_text`, and **re-evaluates the registration link** — the corrected plate might now match a registered user, or no longer match (reverting the session to a guest). Both the event and session rows are locked with `select_for_update()` so the relink can't race a concurrent exit.

> **Authorization:** this service performs **no** access control. The `PATCH /staff/api/events/<id>/correct/` view restricts access to authenticated staff before calling it; any direct callers must enforce equivalent access.

</details>

<details>
<summary><strong>Boundary Validation</strong></summary>

`services.py` is a system boundary — data arrives from CV output and web requests, both of which can be wrong or hostile — so inputs are cleaned before they reach the database. Plate text over 20 characters raises `ValueError` instead of being truncated (a truncated plate is a silently wrong matching key that would mis-bill the wrong car). An untrusted `bounding_box` is coerced to a 4-float list clamped to `[0, 1]`, or `[]` if malformed. Confidence is clamped to `[0.0, 1.0]` so an out-of-range value can't trip the `confidence_score` check constraint mid-insert.

</details>

## Database Models

PostgreSQL is used for the database, because it can store decimal values exactly, support for native JSON columns, and allows for multiple simultaneous writers.

### User

Built on Django's `AbstractUser`. Controls who can access the dashboard or admin panel.

| Field | Description |
| :--- | :--- |
| `username` | Login identifier |
| `email` | Contact email address |
| `password` | Stored as a hashed password, never plain text |
| `first_name` | Optional display name |
| `last_name` | Optional display name |
| `is_staff` | `True` grants operator-dashboard access and permits Django Admin login; model actions still require explicit permissions |
| `is_active` | `False` disables the account without deleting it |
| `is_superuser` | `True` bypasses all permission checks in the admin |
| `date_joined` | Auto-set timestamp when the account was created |
| `last_login` | Auto-updated timestamp on each authentication |

> Guest parking sessions are not linked to a user account.

### LicensePlate

License plates registered to a user account. A user can register multiple plates; each plate belongs to exactly one user.

| Field | Description |
| :--- | :--- |
| `user` | The user account that owns this plate |
| `plate_text` | The text of the license plate |
| `is_primary` | Whether this is the user's primary plate |
| `label` | Optional user-side label to identify the plate |

### ParkingLot

Each record represents one parking lot.

| Field | Description |
| :--- | :--- |
| `name` | The name of the parking lot (unique) |

### LotSettings

Per-lot billing and operational configuration.

| Field | Description |
| :--- | :--- |
| `lot` | The parking lot these settings apply to |
| `rate` | Rate per billing unit (hour or minute) in dollars |
| `billing_unit` | Unit of time for the rate (`hour` or `minute`) |
| `grace_period_minutes` | Minutes before a charge is issued |
| `daily_cap_enabled` | Whether to enable the daily charge cap |
| `daily_cap_amount` | Maximum charge per session |
| `image_retention_days` | How many days to keep uploaded plate images on disk before cleanup |
| `confidence_threshold` | Minimum CV confidence score to trust automatically |

### ParkingSession

The core transactional record — one row per car visit.

| Field | Description |
| :--- | :--- |
| `plate_text` | The text of the license plate |
| `license_plate` | The registered plate record (if any) |
| `user` | The user account the car is registered to |
| `lot` | The parking lot the car is parked in |
| `entry_time` | Time the car entered |
| `exit_time` | Time the car exited |
| `duration_seconds` | Duration of the parking session in seconds |
| `charge_amount` | Charge for the session in dollars |
| `status` | `active`, `completed`, or `void` |
| `has_duplicate_warning` | Whether this session replaced a missed exit |
| `was_orphaned` | Whether this session was voided due to a missed exit |

<details>
<summary><strong>Orphan Handling</strong></summary>

If a plate triggers an entry event while it already has an active session, the system assumes the exit was missed (e.g., camera outage). The old session is voided (`was_orphaned=True`, `status="void"`) and a new session is opened (`has_duplicate_warning=True`). No charge is issued on the voided session.

</details>

### PlateDetectionEvent

The CV audit log — records every entry and exit event from the CV pipeline.

| Field | Description |
| :--- | :--- |
| `session` | The parking session this event belongs to |
| `lot` | The parking lot this event belongs to |
| `image` | Uploaded plate image file path |
| `raw_plate_text` | Plate text as read by the CV pipeline |
| `confidence_score` | Confidence score from the CV pipeline |
| `event_type` | `entry` or `exit` |
| `is_low_confidence` | Whether score is below the confidence threshold |
| `manually_corrected` | Whether an operator corrected the plate text |
| `corrected_plate` | The manually corrected plate text |
| `bounding_box` | Plate bounding box as a JSON array `[x, y, w, h]` |
| `timestamp` | Time the event was created |

### Wallet

A prepaid balance attached to a user account — the China auto-pay model. When a registered plate exits, the parking charge is deducted automatically; no cashier, no per-visit payment.

| Field | Description |
| :--- | :--- |
| `user` | The account this wallet belongs to (one wallet per user) |
| `balance` | Cached running total in dollars; must always equal `SUM(WalletTransaction.amount)` |
| `created_at` / `updated_at` | Auto-managed timestamps |

`balance` is deliberately allowed to go **negative** — an exit is never blocked for insufficient funds, because the barrier must not strand a car. A short account simply owes money, visible to staff. There is no `MinValueValidator` on `balance` by product decision.

### WalletTransaction

One immutable, signed ledger entry — the money audit trail. Rows are insert-only: created once, never updated or deleted.

| Field | Description |
| :--- | :--- |
| `wallet` | The wallet this entry belongs to (`PROTECT` — deleting a wallet can never erase its ledger) |
| `amount` | Signed dollars: positive = credit (top-up), negative = debit (parking charge) |
| `kind` | `topup`, `charge`, or `adjustment` |
| `session` | The `ParkingSession` this charge settled, if any (`SET_NULL` — the ledger outlives the session record) |
| `description` | Human-readable note (never raw plate text, to limit PII exposure) |
| `reference` | External payment-provider confirmation id, for top-ups |
| `created_at` | Insert timestamp (no `updated_at` — rows never change) |

`SUM(amount) == balance` is the money invariant; every credit/debit is written atomically with the balance update under `select_for_update()` (`apps/parking/wallet.py`), and a test reconciles the two. A unique constraint on non-blank `reference` values makes a provider's retried top-up confirmation idempotent — the same confirmation can authorize exactly one credit.

**Payment gateway seam** (`apps/parking/payments.py`) — `PaymentConnector` is a placeholder `Protocol` for a future real provider (Stripe/WeChat Pay/Alipay). It holds no secrets and **fails closed**: the top-up page stays available, but no spendable credit is created until a real connector verifies a provider response.

### Database Integrity Rules

The database itself enforces billing-critical rules so bad data can't sneak in. Django validators only run when a model is saved through a form or `full_clean()` — `bulk_create`, `update()`, and raw SQL skip them entirely. Anything that protects billing math is therefore duplicated as a database-level constraint.

| Rule | Description |
| :--- | :--- |
| No duplicate plates per user | A user cannot register the same `plate_text` twice |
| Unique lot names | `setup_defaults` uses `get_or_create` — duplicate names would return an arbitrary row |
| Sessions survive lot deletion | Sessions are billing records, so deleting a lot with sessions is blocked (`PROTECT`) instead of cascading and wiping revenue history |
| Charges can't be negative | Enforced by both a validator and a database check constraint |
| Exit after entry | A car cannot exit before it entered — clock skew would otherwise produce negative durations |
| No negative durations | `duration_seconds` must be zero or greater |
| Voided sessions carry no charge | A voided session with a charge would corrupt revenue totals |
| Confidence stays in range | `confidence_score` must be between 0.0 and 1.0 |

<details>
<summary><strong>Partial Indexes</strong></summary>

Active sessions are a tiny fraction of the table once months of completed sessions accumulate. Two partial indexes (`plate_text` and `lot`, each filtered to `status='active'`) cover only the rows the entry/exit matcher and the 10-second dashboard poll actually touch, so they stay small enough to live in cache. A third partial index covers unreviewed low-confidence detection events for the manual review queue.

</details>

## See also

- [00-architecture.md](00-architecture.md): system overview and module boundaries
- [08-limitations.md](08-limitations.md): known constraints and roadmap
