# Kiosk Upload State Design

**Date:** 2026-07-24
**Status:** Approved
**Scope:** Public gate-kiosk desktop experience before, during, and after a plate-image upload

## 1. Purpose

The gate kiosk must make the result of a plate scan immediately understandable without exposing private resident information. A successful scan should feel decisive: the upload controls disappear and a single result tells the driver whether to proceed. An uncertain or failed scan should preserve the upload controls because the next useful action is to retry, choose another image, or contact the attendant.

This design uses a hybrid presentation:

- Successful entry and exit scans use **focused state replacement**. The uploader is replaced by the result.
- Low-confidence readings, invalid uploads, model/system failures, and unmatched exits use **stacked recovery**. The uploader remains visible and a specific warning appears directly below it.

The kiosk remains a public, privacy-reduced interface. It must not display an owner name, account balance, event identifier, stored image URL, or exact model-confidence percentage.

## 2. Existing Architecture

The kiosk page is rendered by `apps.public` and submits the activated device's fixed lot and lane to the public scan endpoint through HTMX. The scan endpoint delegates untrusted-image validation and CV processing to the shared scan core, which delegates parking-session and billing behavior to `apps.parking.services`.

The new presentation does not move business logic into templates or JavaScript:

```text
Activated kiosk
    -> local image selection and preview
    -> HTMX multipart scan request
    -> upload validation
    -> CV detector and recognizer
    -> entry/exit parking service
    -> privacy-reduced result fragment
    -> success replacement OR stacked recovery
```

The activated kiosk's lot and event type remain server-controlled device context. Drivers do not select the parking lot or entry/exit direction during each scan.

## 3. Desktop State Sequence

### 3.1 Ready for Image

The initial card displays:

- `Parking Gate`
- The activated context, for example `Downtown Lot — Entry lane`
- A large drag-and-drop target
- `Drop a plate photo here or click to choose`
- `JPEG or PNG · up to 10 MB`
- Primary action: `Choose plate photo`

There is one dominant action. The lot and lane are informative labels, not editable controls.

### 3.2 Photo Selected

After a valid local file is selected:

- The drop-zone prompt becomes a local image preview.
- The heading becomes `Review plate photo`.
- Supporting copy says `Make sure the plate is visible and readable`.
- Primary action becomes `Scan plate`.
- A secondary text action says `Choose a different photo`.

The preview is created from a browser object URL. It is not a server-hosted image and does not expose a private-media route.

Client-side format checks are only usability hints. Server-side validation remains authoritative.

### 3.3 Processing

After `Scan plate` is submitted, the uploader is temporarily replaced by a focused processing card:

- Animated spinner
- `Reading the plate…`
- `Please wait. Do not close this screen.`
- Three semantic stages: `Validate`, `Detect`, and `Record`

The interface must not show a fabricated percentage. The stages communicate real work without claiming timing precision the backend does not provide.

The file input, drop zone, and submit action are disabled for the duration of the request to prevent accidental duplicate submissions. The current single-use kiosk nonce behavior remains in force; the response supplies the next nonce for any retry or subsequent scan.

## 4. Successful Results

Successful results replace the uploader completely. The driver sees only the outcome and the next permitted action.

### 4.1 Entry Accepted

Display:

- Green success treatment and check icon
- Status: `Welcome`
- Recognized plate in the largest text, for example `ABC 123`
- Optional `Registered vehicle` badge when the plate is registered
- `Entry recorded. Please proceed through the gate.`
- Secondary action: `Scan another vehicle`

Do not display financial information on entry.

### 4.2 Exit Accepted

Display:

- Green success treatment and check icon
- Status: `Goodbye`
- Recognized plate
- Charge amount in prominent monospace text
- Payment disposition:
  - `Billed to your account`, or
  - `Amount due — please pay at the gate`
- `Exit recorded. Please proceed.`
- Secondary action: `Scan another vehicle`

The result may say whether the amount was billed, but it must not display the account holder's identity or remaining wallet balance.

### 4.3 Reset Behavior

`Scan another vehicle` restores the ready-for-image state, clears the result, resets the form, revokes any remaining browser object URL, and moves focus back to the file-selection control.

This design preserves the current explicit reset behavior. Automatic timeout reset is not part of this change.

## 5. Recovery Results

Recovery states retain or restore the uploader and render a warning or error panel directly below it. The panel must explain what happened and offer an action that resolves that specific problem.

### 5.1 Low Confidence

Display:

- Warning heading: `Check this plate`
- Best model reading in large monospace text, visibly marked as uncertain
- `The reading was uncertain. Retake a closer photo or ask the attendant.`
- Primary action: `Retake photo`
- Secondary action: `Call attendant`

The best guess must not be styled like a successful reading. The exact numeric confidence is not shown.

The existing backend behavior remains authoritative: a low-confidence event may still be recorded and routed to the operator review queue. The public kiosk copy does not imply that the uncertain reading was silently accepted as final.

### 5.2 No Readable Plate

Display:

- `We couldn't read the plate clearly`
- Short corrective guidance covering distance, lighting, and camera angle
- `Retake photo`
- `Call attendant`

The uploader is available immediately so the driver can select a replacement image.

### 5.3 Invalid Image or Form Input

Examples include a missing image, unsupported real file format, mismatched MIME type, oversized file, or excessive decoded dimensions.

Display:

- Error heading: `This image cannot be used`
- A specific, safe message such as `The file must be a valid JPEG or PNG smaller than 10 MB.`
- The empty uploader with `Choose plate photo`

The invalid file is cleared so the same bad value cannot be accidentally resubmitted. Internal paths, decoder details, stack traces, and server configuration are never exposed.

### 5.4 Model or System Failure

Display:

- Heading: `We could not process the scan`
- `Try once more. If this continues, contact the gate attendant.`
- Primary action: `Try again`
- Secondary action: `Call attendant`

The selected local preview may remain for one retry. The retry uses the next server-issued kiosk nonce and never reuses a consumed request capability.

Unexpected failures are logged server-side with enough context for operators, while the public response remains generic and non-sensitive.

### 5.5 Unmatched Exit

Display:

- Warning heading: `No matching entry found`
- Recognized plate
- `Retake the photo or contact the attendant before leaving.`
- Primary action: `Retake photo`
- Secondary action: `Call attendant`

The kiosk must not invent a charge or tell the driver to proceed when no active session matches the plate. The underlying unmatched detection remains available to the operator review workflow.

### 5.6 Rate Limit

Display:

- `Too many scan attempts`
- A short instruction to wait before trying again
- The retry action remains disabled until the server-provided retry interval has elapsed, when such an interval is available
- `Call attendant`

The interface must not encourage repeated clicking while throttled.

## 6. Visual and Interaction Rules

The kiosk continues using the project's existing dark design tokens:

- Primary background: `#0f1117`
- Secondary surface: `#1a1d27`
- Primary text: `#e4e4e7`
- Secondary text: `#a1a1aa`
- Accent: `#3b82f6`
- Success: `#22c55e`
- Warning: `#eab308`
- Error: `#ef4444`
- Plate and monetary values: JetBrains Mono

Desktop priorities:

1. Outcome status
2. Recognized plate
3. Charge or corrective instruction
4. One primary next action

The main card remains centered and comfortably readable at kiosk distance. Success results use green status treatment. Low-confidence and unmatched-session states use yellow. Invalid-input and system-failure states use red.

The result container remains an `aria-live="polite"` region. Explicit failures use `role="alert"`. Processing status is announced without repeatedly announcing decorative stage changes. Focus moves to the result heading after a response and returns to the file input after reset.

## 7. Component Boundaries

The UI should be divided into small, purpose-specific template fragments:

- **Kiosk shell:** page heading, activated lot/lane context, and result live region
- **Uploader:** empty, selected-preview, and disabled/processing presentation
- **Success result:** entry and exit variants
- **Recovery result:** low confidence, invalid image, model/system failure, unmatched exit, and rate limit variants
- **Browser controller:** preview lifecycle, form state, focus management, reset, nonce update, and retry timing

The server decides the semantic result state. JavaScript only presents the state and manages local interaction; it must not infer whether an entry, exit, billing operation, or confidence threshold succeeded.

## 8. Response Contract

The rendered result context needs a stable presentation state, rather than relying on broad template fall-through:

- `entry_success`
- `exit_success`
- `low_confidence`
- `unreadable`
- `invalid_image`
- `model_error`
- `unmatched_exit`
- `rate_limited`

Each response supplies only fields appropriate to that state. Public responses may include the recognized plate, charge, registered flag, billed-to-account flag, safe message, and retry timing. They must not include owner identity, wallet balance, event ID, image URL, server path, or raw exception text.

All error paths must log, raise, or return an explicit error. There are no silent client or server failures.

## 9. Testing Strategy

### Server and Template Tests

Verify that:

- Entry success renders the welcome state and replaces the uploader.
- Exit success renders charge and payment disposition.
- Low confidence renders the stacked warning and preserves retry controls.
- Unreadable scans, invalid images, model errors, unmatched exits, and rate limits each render their own specific recovery state.
- Public fragments never contain owner identity, account balance, event ID, image URL, local path, raw exception text, or numeric confidence.
- Kiosk activation still fixes lot and lane server-side.
- Single-use nonce consumption and refresh work on success and every error response.

### Browser Behavior Tests

Verify that:

- Selecting or dropping a valid image shows the local preview.
- Choosing a different image revokes the old object URL.
- Submission disables repeat interaction and shows processing.
- Successful responses remove the uploader.
- Recovery responses restore or retain the uploader and place the warning below it.
- Invalid files are cleared.
- Reset clears the preview/result and returns focus to image selection.
- HTMX failure paths do not leave the kiosk permanently disabled.

### Accessibility and Visual Checks

Verify keyboard-only operation, focus order, screen-reader announcements, status contrast, desktop readability, and the responsive single-column fallback.

## 10. Acceptance Criteria

The design is complete when:

1. A driver can distinguish ready, selected, processing, success, warning, and error states without operator explanation.
2. Successful scans replace the upload controls with one decisive result.
3. Any state requiring corrective action keeps or restores the uploader and places the explanation below it.
4. Entry success clearly authorizes proceeding.
5. Exit success clearly shows the charge and whether it was billed or is due.
6. Low-confidence output is visibly uncertain and offers retake and attendant paths.
7. Invalid-image, model/system, unmatched-exit, and rate-limit outcomes use distinct and actionable messages.
8. No public kiosk response exposes resident identity, account balance, internal identifiers, private image locations, exact confidence, or raw errors.
9. Existing kiosk activation, rate limiting, nonce protection, scan validation, CV processing, parking-session behavior, and billing behavior remain authoritative.
10. Payment, wallet, and ledger behavior and schema remain unchanged by this UI work.

## 11. Out of Scope

- Changes to CV architecture, model weights, or confidence thresholds
- Changes to parking-session matching or billing rules
- Changes to payment, wallet, or ledger behavior or schema
- Operator-side manual correction workflow
- Camera hardware capture or live video
- Automatic result reset timers
- Resident portal changes
