# CV Training Data

The CV models are trained entirely on synthetic data generated at runtime. No real plate images are committed to the repository.

| Path | Purpose |
|------|---------|
| `data/backgrounds/` | Curated parking-lot photos for detector compositing (must exist before generating the detector dataset) |
| `data/detector/` | YOLO-format detector dataset (`images/`, `labels/`) |
| `data/recognizer/` | Recognizer crops + `labels.csv` |

## Data Generation

**`generate_detector_dataset()`** and **`generate_recognizer_dataset()`** live in `apps/cv/training/synthetic_data.py`. Plates are rendered for both US and Canadian formats at 400×120 pixels.

**How a synthetic plate image is built:**

1. **Generate plate text** randomly following country-specific format conventions:
   - US plates: `ABC 1234` (most common), `123 ABC`, or `ABC123`
   - Canadian plates: `ABC 123` or `A1B 2C3` (Ontario-style alphanumeric)
2. **Build the plate background** — a white rectangle with a dark border. Canadian plates add a solid blue strip across the top quarter to visually differentiate them from US plates.
3. **Render plate text** onto the background using a TrueType plate font (`composite_on_background`). `textbbox` determines the plate center and the text is drawn in black ink. If the font file (`apps/cv/training/assets/plate_font.ttf`, Liberation Mono Bold) is missing, Pillow's default bitmap font is used as a fallback and an error is logged — install it before training, since every run before the 2026-09-26 retrain silently used the fallback.
4. **Composite onto a background** (`composite_on_background`) — the background photo is first cropped to a random region covering 50–100% of its area at a 4:3 aspect ratio, then resized to 640×480, so a small background set still yields many distinct framings. The plate is scaled to 15–60% of the image width, given a mild perspective warp (each corner jittered by up to 12% of the plate size), rotated −15° to +15°, and pasted at a random position fully inside the frame. The bounding box is taken from the warped plate's visible pixels, so it stays tight after every transform.

**Detector dataset output** — saves full-scene `images/*.jpg` with paired `labels/*.txt` in YOLO format: `class_index cx cy w h` (all values normalized to `[0, 1]`). Existing files in the output directory are deleted before each run so re-runs don't mix generations.

**Recognizer dataset output** — saves only the cropped plate `images/*.png` (grayscale) with a `labels.csv` (`filename`, `text`, `country`). Existing files are deleted before each run. `crop_source` chooses how each crop is made:

- `"flat"` (default) — the rendered plate on its own, flattened to 128×32. Clean and sharp.
- `"scene"` — the plate is composited into a background exactly as for the detector, then cut back out with the ground-truth box jittered by ±10%, so it carries the blur, perspective and loose framing of a real detector crop.
- `"mixed"` — a `scene_fraction` share of samples (default 0.5) use `"scene"`, the rest `"flat"`. This is the recommended setting: a recognizer trained on flat crops alone read only 3% of plates correctly even when handed a perfect scene crop (see [CV Model Status](01-cv-pipeline.md#cv-model-status)).

**The 90%-yield hard failure** (`generate_detector_dataset` / `generate_recognizer_dataset`, `synthetic_data.py:461-476, 552-564`) — both builders count how many images they actually produced. If fewer than 90% of the requested samples were generated successfully, the run raises `RuntimeError` instead of silently writing an undersized dataset. Both functions accept an optional `seed` parameter to make the generated dataset reproducible across runs.

## Dataset Classes

**`PlateDetectorDataset`** (`apps/cv/training/dataset.py`)

1. At startup, scans `images/*.jpg` (skips symlinks) and pairs each file with `labels/<same-stem>.txt`.
2. Each label file contains one YOLO line: `0 cx cy w h`. The leading class index `0` is dropped; the four floats are the box normalized to `[0, 1]`.
3. `__getitem__` loads the JPG, converts to an RGB tensor `(3, H, W)`, and returns `(image_tensor, bbox_tensor)` where `bbox_tensor` has shape `(4,)`.
4. Use the **default collate** function with a standard `DataLoader` — not `ctc_collate_fn`.

**`PlateRecognizerDataset`** (`apps/cv/training/dataset.py`)

1. At startup, reads `labels.csv` (`filename`, `text`; `country` is stored but not returned per sample).
2. `__getitem__` loads the matching PNG, converts to a grayscale tensor `(1, 32, 128)`, encodes the text to a list of character indices (spaces skipped), and returns `(image_tensor, label_list)`.
3. A `DataLoader` **must** set `collate_fn=ctc_collate_fn` because label lengths vary. The collate function stacks images to `(N, 1, 32, 128)`, concatenates all label lists into one 1D `targets` tensor, and builds `target_lengths` (how many indices belong to each sample).

**Character encoding** (recognizer only, shared with [Plate Recognizer CRNN](01-cv-pipeline.md#plate-recognizer-crnn)'s output layer):

- `A→1` … `Z→26`, `0→27` … `9→36`
- Index `0` is reserved for the CTC blank token
- Spaces are skipped (not encoded)
- `CHAR_TO_IDX` / `VOCAB_SIZE=37` are the shared CTC encoding constants

## Augmentations

`apps/cv/training/augment.py` provides two transform classes that slightly modify training images so the models generalize to real parking cameras, applied **in memory** after the dataset loads the tensor — this module does not read files from disk.

**Two modes:**

- `train=True` — random changes each pass (used during training).
- `train=False` — normalization only, no random changes (used during evaluation).

**`DetectorAugment`** (full parking-lot photo, color):

| Augmentation | Real-world failure it targets |
|---|---|
| `ColorJitter` | Camera white-balance/exposure drift across time of day and lot lighting |
| `GaussianBlur` | Lens defocus, motion blur, JPEG compression artifacts |
| `RandomGrayscale` (10%) | Monochrome/IR security camera feeds |
| Horizontal flip (50%, bbox-aware) | Vehicles entering from either direction |
| ImageNet mean/std normalization | — |

**`RecognizerAugment`** (small grayscale plate crop):

| Augmentation | Real-world failure it targets |
|---|---|
| Brightness/contrast tweaks | Faded or dirty plates |
| `GaussianBlur` | Lens defocus, motion blur |
| `RandomPerspective` (50%, mild) | Off-axis gate camera angle — plate not shot square-on |
| **No flip** (deliberately absent) | Mirrored plate text is not a valid alternate reading, it's wrong data — flipping would poison the labels, not augment them |
| Grayscale normalization (mean 0.5, std 0.5) | — |

The recognizer **never** flips the image horizontally — `"ABC 123"` backwards would not match the ground-truth label. The detector **can** flip because it only predicts where the plate is, not what it says.

## Training the Models

Run the training scripts outside Docker to use MPS on Apple Silicon (or CUDA on NVIDIA). Keep a few background photos out of training entirely so validation measures generalization rather than memorized scenes:

```bash
# Detector data: train on one background set, validate on a held-out set
python -c "from apps.cv.training.synthetic_data import generate_detector_dataset as g; g(n=2500, output_dir='data/detector', bg_dir='data/backgrounds_train', seed=1)"
python -c "from apps.cv.training.synthetic_data import generate_detector_dataset as g; g(n=400, output_dir='data/detector_val', bg_dir='data/backgrounds_holdout', seed=2)"

# Recognizer data: half clean crops, half crops cut out of composited scenes
python -c "from apps.cv.training.synthetic_data import generate_recognizer_dataset as g; g(n=8000, output_dir='data/recognizer', crop_source='mixed', bg_dir='data/backgrounds_train', seed=1)"

# Train (--output must resolve inside the project; the script refuses paths outside it)
python apps/cv/training/train_detector.py \
    --data-dir data/detector \
    --val-data-dir data/detector_val \
    --epochs 40 --patience 8 \
    --output apps/cv/weights/detector.pth
python apps/cv/training/train_recognizer.py \
    --data-dir data/recognizer \
    --epochs 20 \
    --output apps/cv/weights/recognizer.pth

# Measure end to end on the held-out backgrounds (detector IoU, full-plate
# reads through the whole pipeline, and reads given a perfect "oracle" box)
SECRET_KEY=eval DEBUG=True DB_PASSWORD=unused PYTHONPATH=. \
    python apps/cv/training/evaluate_detector.py \
    --bg-dir data/backgrounds_holdout --n 100 --json eval.json
```

Both training scripts save a training-curve plot next to their `.pth` file. On an Apple M3, a detector epoch over 2,500 images took roughly 5–20 minutes (slowing as the machine heated up) and a recognizer epoch over 8,000 crops under a minute. Most of the detector time is the MPS forward and backward pass (about 2 s per batch of 32 on that machine); CPU-side augmentation adds only about 0.3 s per batch.

## See also

- [00-architecture.md](00-architecture.md): system overview and module boundaries
- [08-limitations.md](08-limitations.md): known constraints and roadmap
