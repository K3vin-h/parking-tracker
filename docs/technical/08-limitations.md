# Known Limitations

## CV models

- **The detector misses its target.** Best IoU is about 0.43 against a >0.70 target. A loose crop directly degrades the recognizer's input. See [01-cv-pipeline.md → CV Model Status](01-cv-pipeline.md#cv-model-status) for the diagnosis.
- **The recognizer is undertrained.** It met its targets (98.59% character / 91.50% full-plate accuracy) but stopped at epoch 36 of 100.
- **Synthetic data only.** Neither model has been evaluated on real photographs, so every reported number describes in-distribution synthetic performance.
- **A narrow synthetic distribution:**
  - one plate font;
  - five US and Canadian format templates;
  - 11 reused background photos;
  - no perspective warp or directional motion blur in `DetectorAugment`.
- **The detector can't say "no plate here".** It regresses exactly one box per image, with no objectness score, and cannot handle multiple plates in a frame.
- **Greedy CTC decode only.** Beam search was deliberately skipped (see [03-design-rationale.md](03-design-rationale.md#recognizer-design-choices)).
- **No feedback loop.** Staff corrections fix the session record but are never fed back into training data.

## Kiosk trust model

- **An uploaded photo stands in for a camera.** Anyone with an activated kiosk can upload a photo of any plate, which opens or closes that plate's session and bills its owner's wallet. Activation capabilities, one-time nonces, the image replay window and per-IP rate limits blunt abuse but do not eliminate it. Device or gate-level authentication would be needed before exposing a kiosk beyond a controlled lane.
- **Inference runs in the request.** There is no task queue. That is fine for manual uploads, but a continuous camera feed would need async workers such as Celery. The pipeline code itself would not need to change.

## Money

- **No real payment provider.** `PlaceholderPaymentConnector` fails closed, so wallets can only be funded by staff from the shell or admin until a real connector is wired in.
- **Balances can go negative by design.** There is no collections or notification flow for accounts in debt, only staff visibility.
- **Single currency, no tax, receipts or refunds.**

## Access control and data

- **One global `is_staff` role.** There are no per-lot permissions or tenant isolation, so any staff user sees every lot.
- **Image retention needs a scheduler.** `cleanup_old_images` must be scheduled externally (cron or a Kubernetes `CronJob`). Nothing runs it in-process.
- **Plate images live on local disk.** They sit under the private `MEDIA_ROOT`. Multi-host deployment needs private object storage.

## Frontend

- **Polling, not push.** Live dashboard regions poll every 10 seconds over HTMX. There is no WebSocket.
- **No public demo mode.** Evaluating the app requires running it locally with trained weights.

## See also

- [00-architecture.md](00-architecture.md)
- [07-security-and-deployment.md](07-security-and-deployment.md)
