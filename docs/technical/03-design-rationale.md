# CV Design Rationale

This section explains the *why* behind the CV stack in `apps/cv/` — not
that a detector and recognizer exist, but why each shape was chosen and
what was traded away to get it. For the exact layers, shapes, and constants
behind each claim, see [CV Pipeline](01-cv-pipeline.md); for measured results, see
[CV Model Status](01-cv-pipeline.md#cv-model-status).

## Why Build the CV Stack From Scratch

**Decision:** PyTorch + OpenCV, two custom-trained models, zero external
ANPR/OCR APIs (no Google Vision, no AWS Rekognition, no commercial LPR SDK).

**Why:** This constraint was self-imposed from the outset — the point of
the project was to build and understand a CV stack end to end, not to wire
up a vendor SDK. That goal happens to line up with several properties the
deployment genuinely needs. A hosted ANPR API would solve plate reading in an afternoon at
near-100% accuracy — but it would also mean per-scan cost that scales with
traffic, a hard dependency on a third party's uptime for a physical gate
that must open cars in and out, and no way to reason about *why* a read
failed (commercial APIs are black boxes; you get a string and a
confidence float, not a bounding box you can debug or a training set you
can extend). It would also make "confidence" mean whatever the vendor
defines it to mean, when the billing logic in `services.py` needs a
per-lot tunable threshold (`LotSettings.confidence_threshold`) that gates
real money movement.

**Tradeoff accepted:** the from-scratch stack is exactly as good as its
training data, which here is 100% synthetic (see below). The detector
misses its accuracy target as a direct, measured consequence of that
choice — an honest cost, not swept under the rug (see
[CV Model Status](01-cv-pipeline.md#cv-model-status)). In exchange, the system never has a
network dependency in its billing-critical path, never sends a photo of
someone's car to a third party, and every failure mode is inspectable down
to the tensor.

## Two-Stage Detector → Recognizer, Not One End-to-End Model

**Decision:** `PlateDetectorCNN` finds *where* the plate is; a separate crop
step hands that region to `PlateRecognizerCRNN`, which reads *what* it says
(see [Plate Recognition Pipeline](01-cv-pipeline.md#plate-recognition-pipeline) for the exact
call chain).

**Why split instead of one network doing both:** the two sub-problems have
different input scales (a 640×480 scene vs. a 128×32 crop), different loss
functions (bbox regression vs. CTC sequence loss), and different failure
modes that need to be diagnosable independently. A combined model (e.g. a
single YOLO-style head that outputs both box and per-character classes)
conflates them: if the plate text comes out wrong, you cannot tell whether
the box was off, the crop was bad, or the reader itself misclassified a
character, without additional instrumentation. Two small models trained and
validated separately let [CV Model Status](01-cv-pipeline.md#cv-model-status) report the
detector's IoU and the recognizer's character/plate accuracy as two
independent numbers — which is exactly how the current failure was found
(recognizer met its targets in isolation; detector did not, and that gap is
now visible instead of hidden inside one combined loss curve).

**Alternative considered and rejected:** an end-to-end single-box detector
with a joint OCR head (closer to real-world ANPR systems). Rejected because
it needs a much larger, harder-to-synthesize training set to converge
(joint losses on small custom nets are notoriously unstable), and it removes
the ability to swap or retrain just the underperforming piece — here, the
detector — without touching the piece that already meets its target.

**Cost of the split:** a bad detector crop (loose box, clipped characters,
included car body) directly degrades the recognizer's input regardless of
how good the recognizer is on its own — documented explicitly in
[CV Model Status](01-cv-pipeline.md#cv-model-status). The two-stage design makes this
failure legible instead of eliminating it.

## Detector Design Choices

| Decision | Why | Alternative rejected |
|---|---|---|
| Direct regression of one normalized box, not multi-box detection | One gate camera frame has exactly one plate to find. A single-box regressor is the simplest model that fits that constraint. | Anchor-based / YOLO-style multi-box detection with objectness scores and NMS. Rejected as over-engineered for a single-plate-per-frame problem; it also would have let the model express "no plate here," which the current architecture explicitly cannot (a real gap, tracked in [CV Model Status](01-cv-pipeline.md#cv-model-status)) — deferred rather than solved with disproportionate complexity for v1. |
| `AdaptiveAvgPool2d((4, 4))` before the FC head (see [Plate Detector CNN](01-cv-pipeline.md#plate-detector-cnn) for the exact shapes) | Lets the model tolerate minor resizing from augmentation without being hard-wired to one exact input resolution, while keeping enough spatial structure (vs. a 1×1 global average pool) that the FC head still knows roughly *where* in the frame the plate-like texture concentrated. | A fixed `Flatten()` after a fixed-size conv stack — simpler, but locks the architecture to one exact input size and discards even more spatial position information. |
| `SmoothL1Loss` (Huber) instead of `MSELoss` | Bounding-box regression targets can have occasional bad synthetic labels or extreme early-training predictions; `MSELoss` penalizes those quadratically, producing gradient spikes that destabilize a small network with no batch-norm-heavy backbone to absorb them. `SmoothL1` behaves like L2 near zero (smooth convergence) and like L1 on large errors (bounded gradient). | Plain L1 — more outlier-robust but has a constant-magnitude gradient even very close to the optimum, making fine-grained convergence noisier; rejected because it would make the last few percent of IoU improvement harder to reach. |
| `ReduceLROnPlateau(factor=0.5, patience=5)` | Lets the optimizer take large steps early and only slow down once validation loss actually plateaus, without hand-scheduling epoch cutoffs the way a fixed step-decay would require guessing in advance. | A fixed step-decay schedule — rejected because the 50-epoch budget is small enough that guessing the right decay epoch ahead of time is more likely to hurt than a reactive scheduler. |
| Dropout `p=0.3` before the final FC layer | Synthetic data has far less visual variance than real camera footage (one font, ~11 backgrounds — see [Synthetic Data](#synthetic-data-why-and-how-its-kept-honest) below), so without regularization the model can memorize background-specific cues instead of plate features. | No dropout — rejected because the training run shows a pattern (bottomed-out val loss, low IoU; diagnosed in [CV Model Status](01-cv-pipeline.md#cv-model-status)) consistent with fitting the limited synthetic distribution rather than plate geometry generally — more regularization pressure, not less, is the likely next lever. |

## Recognizer Design Choices

**CRNN + CTC, not per-character segmentation.** The detector hands over a
crop of unknown character count (US plates run 6–7 characters, Canadian
formats add 2 more format variants) with no per-character bounding boxes.
A segmentation-then-classify approach would need the synthetic generator to
*also* fabricate character-level boxes, and it breaks the moment two
characters touch or the crop is slightly skewed — exactly the conditions a
loose detector crop produces. A CRNN with a CTC output sidesteps both
problems: it emits a fixed-length sequence regardless of plate length, and
CTC's alignment is learned, not hand-labeled (see
[Plate Recognizer CRNN](01-cv-pipeline.md#plate-recognizer-crnn) for the exact architecture).

| Decision | Why | Alternative rejected |
|---|---|---|
| Final block halves width but preserves height (see [Plate Recognizer CRNN](01-cv-pipeline.md#plate-recognizer-crnn) for the exact pooling shapes) | Horizontal resolution *is* the character sequence here — each resulting column is what the LSTM reads one time-step at a time. Halving height as aggressively as width (the detector's pattern) would throw away the sequence signal the whole architecture depends on; preserving height also keeps the vertical stroke detail that disambiguates look-alikes like `I`/`1` or `O`/`0`. | Symmetric pooling on all three blocks (as in the detector) — rejected because it would leave too few time-steps, too coarse to place 6–8 characters distinctly for CTC alignment. |
| Bidirectional LSTM over the character sequence | Reading both directions gives every time-step context from the *entire* plate, not just what came before it — resolves ambiguous single-frame reads (e.g. a smudged `D` is easier to call once you also know a digit run follows). | A unidirectional LSTM or plain 1D-CNN sequence head — rejected as strictly less contextual for a sequence this short; the added BiLSTM cost is negligible at this scale. |
| CTC loss instead of a fixed-length cross-entropy per position | Plate length varies (`ABC123` is 6 chars, `LLL DDDD` runs 7) and there is no reliable per-character ground-truth alignment to a fixed-length target grid. CTC learns the alignment between the emitted time-steps and the variable-length label itself, via the blank token. | Fixed-length classification per output slot — would require padding/truncating every label to one length and inventing an alignment (which character occupies which slot) the model has no principled way to learn correctly. |
| Greedy CTC decode (argmax → collapse repeats → drop blank) instead of beam search | Sufficient accuracy on a short sequence over a small vocabulary at negligible compute; this is a synchronous per-scan kiosk path where added decode latency has no accuracy return the current model (undertrained, see [CV Model Status](01-cv-pipeline.md#cv-model-status)) can actually cash in on. | CTC beam search — meaningfully helps only when the language model / prefix scoring has signal to exploit; on short, low-vocabulary plate strings with a still-improving base model, the accuracy gain would not justify the added latency and complexity. |
| `CTCLoss` forced onto CPU even when the model runs on MPS | PyTorch's MPS backend has no native CTC-loss kernel as of PyTorch 2.x (mechanism and exact call site in [Plate Recognizer CRNN → Training](01-cv-pipeline.md#plate-recognizer-crnn)). | Training entirely on CPU to avoid the split — rejected as far slower for the conv/LSTM forward-backward passes, which *are* MPS-accelerated; only the loss call needs the workaround. |
| 37-class vocab (26 letters + 10 digits + CTC blank) | Matches exactly the character set the synthetic generator emits — no lowercase, no punctuation, so the model never has to represent a class it will never see. | A larger vocab including punctuation/lowercase — rejected as pure unused capacity; the format templates in `synthetic_data.py` never produce those characters. |

## Synthetic Data: Why, and How It's Kept Honest

**Why synthetic instead of a real labeled plate dataset:** there is no
project-owned corpus of labeled parking-lot plate photos, real plates are
personally identifying (a committed dataset of real plates would itself be a
privacy liability this project explicitly tries to minimize elsewhere — see
the kiosk's privacy-reduced responses), and a generator gives exact control
over the label distribution (format mix, plate count per frame, occlusion
level) that a scraped dataset would not. The cost of this choice is measured,
not hidden — see [CV Model Status](01-cv-pipeline.md#cv-model-status).

**Why composite onto real backgrounds, not synthetic ones** — plates are
pasted onto curated real parking-lot photos rather than solid colors or
procedurally generated scenes. Training the detector against flat
backgrounds would let it learn "plate = the only textured rectangle in the
frame," a shortcut that collapses instantly against real clutter (parked
cars, curbs, shadows, signage). Real backgrounds force the detector to learn
actual plate appearance rather than a scene-composition trick. See
[Synthetic Training Data](02-cv-training.md) for exactly how images
are generated, augmented, and turned into datasets.

**Why the 90%-yield floor fails loudly instead of writing whatever it got:**
a high skip rate during generation means something is systemically broken —
a corrupt background directory, a missing font, a full disk — not a handful
of unlucky samples. Silently training on an undersized, skip-biased dataset
would produce a model with unknown, untraceable blind spots; the loud
failure (mechanics in [Data Generation](02-cv-training.md#data-generation)) forces the root
cause to be fixed before a single epoch runs.

**Known, documented limits of the synthetic distribution** — one plate font,
only 5 fixed format templates, and a small, reused set of background photos.
These are load-bearing facts behind the domain-gap discussion, not a
separate concern from the model architecture; see
[CV Model Status](01-cv-pipeline.md#cv-model-status) for the full list and its measured
impact.

## Preprocessing as a Security Boundary

`load_image()` is not "resize the image" — it is the first parser that
touches attacker-controlled bytes on a **public, unmanned kiosk**. Anyone can
upload a file there with no account and no staff oversight, which makes
image decode itself part of the attack surface, not an implementation
detail. Every concrete control (path containment, content-based format
allowlist, decompression-bomb cap, single bounded read, path-stripped
errors) is documented once, as part of the function's load contract, in
[CV Pipeline → Image Preprocessing](01-cv-pipeline.md#image-preprocessing); the summary of
the overall security posture lives in [Security](07-security-and-deployment.md). None of it is
generic "input validation" boilerplate — each check maps to a specific
attack a public, credential-free upload endpoint invites.

## Inference Engineering

- **Lazy singleton with double-checked locking** — loading two `.pth` files
  costs hundreds of milliseconds, too slow to redo per kiosk request. Lazy
  (rather than at Django startup) because startup code also runs during
  `migrate`/`collectstatic`, where weight files may not exist yet — eager
  loading would break those commands in CI just for lacking a trained model
  no one asked for at that point. See
  [Plate Recognition Pipeline → Singleton](01-cv-pipeline.md#plate-recognition-pipeline) for
  the exact locking mechanism.
- **`predict()` under `@torch.no_grad()`, restoring train/eval state** (both
  models) — makes inference safe to call *during* training (e.g. a
  mid-epoch validation callback) without corrupting the training loop's own
  mode state; exact contract in [CV Pipeline](01-cv-pipeline.md).
- **Device auto-detection, MPS → CUDA → CPU** — one function
  (`get_device()`) used by both training scripts and the inference
  pipeline, so there is exactly one place that decides hardware, not
  independently-drifting copies.
- **The CTC-on-CPU workaround is training-only** — `predict()` and
  `decode_predictions()` never touch `CTCLoss`, so kiosk inference runs
  fully on MPS/CUDA with no forced CPU hop; only the training loop pays that
  cost (see [Recognizer Design Choices](#recognizer-design-choices)).
- **Weight files are versioned, not bare state dicts** — a checkpoint that
  doesn't declare a matching preprocessing version is rejected outright
  rather than loaded and silently fed mismatched input statistics (exact
  mechanism in [Plate Recognition Pipeline → Model Loading](01-cv-pipeline.md#plate-recognition-pipeline)).
  Failing closed here trades convenience (you must retrain through the
  current scripts) for never confidently misreading a plate because of an
  invisible normalization mismatch.

## Confidence as a Product Decision, Not Just a Metric

Two different thresholds exist on purpose, at two different layers:

| Threshold | Lives in | Who can change it | What it does |
|---|---|---|---|
| `LOW_CONFIDENCE_THRESHOLD` | `apps/cv/pipeline.py` (fixed constant) | No one at runtime | Flags `PipelineResult.is_low_confidence` as a CV-layer signal (exact value in [Confidence Score](01-cv-pipeline.md#plate-recognition-pipeline)) |
| `LotSettings.confidence_threshold` | `apps/parking` model, per-lot, DB-backed | Staff, via `/staff/settings/` | The value `services.handle_entry`/`handle_exit` actually check before trusting a read for billing |

The pipeline's constant is a reasonable default, not the value that gates
money movement — `services.py` deliberately reads the *operator-tunable*
threshold instead, so a lot with worse camera placement or more glare can be
made stricter (or looser) without touching CV code or redeploying weights.

**Tiny-box rejection** — a detected box below a minimum fraction of the
frame (exact constant in
[Plate Recognition Pipeline → Processing Steps](01-cv-pipeline.md#plate-recognition-pipeline))
is treated as "no plate found" rather than handed to the recognizer, because
a crop that small cannot contain readable character strokes regardless of
recognizer quality; forcing a read anyway would just manufacture a
confident-looking wrong answer.

**Low confidence degrades to human review, it never blocks the gate.** This
is the core product decision: `handle_exit`/`handle_entry` still open or
close the session even when `is_low_confidence=True` — they just also
create a flagged `PlateDetectionEvent` for the `/staff/errors/` queue (see
[Session & Billing](04-sessions-and-billing.md)). A barrier-arm system that refuses
to act on a low-confidence read would strand a car at the gate every time
the model is unsure — worse than occasionally billing off a low-confidence
guess and letting a human correct it after the fact via `correct_plate()`.
The system is designed to degrade gracefully to staff correction, not to
refuse service.

## See also

- [00-architecture.md](00-architecture.md): system overview and module boundaries
- [08-limitations.md](08-limitations.md): known constraints and roadmap
