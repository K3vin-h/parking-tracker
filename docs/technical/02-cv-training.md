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
3. **Render plate text** onto the background using a TrueType plate font (`composite_on_background`). `textbbox` determines the plate center and the text is drawn in black ink. If the font file is missing, Pillow's default font is used as a fallback.
4. **Composite onto a background** — for the detector dataset, the plate is pasted onto a random 640×480 parking-lot background image at a random position, random scale, and random rotation (−15° to +15°). The plate is constrained to fit fully within the background.

**Detector dataset output** — saves full-scene `images/*.jpg` with paired `labels/*.txt` in YOLO format: `class_index cx cy w h` (all values normalized to `[0, 1]`). Existing files in the output directory are deleted before each run so re-runs don't mix generations.

**Recognizer dataset output** — saves only the cropped plate `images/*.png` (grayscale) with a `labels.csv` (`filename`, `text`, `country`). Existing files are deleted before each run.

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

Run the training scripts outside Docker to use MPS on Apple Silicon (or CUDA on NVIDIA):

```bash
# Train the plate detector (target: >0.7 IoU after 50 epochs;
# actual last run: completed all 50 epochs, best IoU ~0.43 — target not met)
python apps/cv/training/train_detector.py \
    --epochs 50 \
    --data-dir data/detector \
    --output apps/cv/weights/detector.pth

# Train the plate recognizer (target: >90% char accuracy, >80% full-plate after 100 epochs;
# actual last run: concluded at epoch 36/100 (best epoch on all metrics, kept as final) —
# 98.59% char / 91.50% full-plate — both targets met)
python apps/cv/training/train_recognizer.py \
    --epochs 100 \
    --data-dir data/recognizer \
    --output apps/cv/weights/recognizer.pth
```

Both scripts save training-curve plots alongside the `.pth` files.

## See also

- [00-architecture.md](00-architecture.md): system overview and module boundaries
- [08-limitations.md](08-limitations.md): known constraints and roadmap
