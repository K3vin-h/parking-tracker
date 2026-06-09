# Parking Lot Tracker

A parking lot management system that uses computer vision to read license plates from cars entering and exiting. It calculates parking duration and issues charges automatically.

## Table of Contents

- [Database Models](#database-models)
- [CV Pipeline](#cv-pipeline)
- [Branch: Synthetic Training Data Pipeline + Plate Detector CNN](#branch-synthetic-training-data-pipeline--plate-detector-cnn)
- [Branch: Plate Recognizer CRNN](#branch-plate-recognizer-crnn)
- [Creating the Dataset for Training](#creating-the-dataset-for-training)
- [Web Application](#web-application)
- [Docker](#docker)

---

## Database Models

### User

Built on top of Django's built-in user model. Controls who can access the dashboard or admin panel.

| Field&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Description |
| :--- | :--- |
| `username` | Login identifier |
| `email` | Contact email address |
| `password` | Stored as a hashed password, never plain text |
| `first_name` | Optional display name |
| `last_name` | Optional display name |
| `is_staff` | `True` grants access to the Django admin site |
| `is_active` | `False` disables the account without deleting it |
| `is_superuser` | `True` bypasses all permission checks in the admin |
| `date_joined` | Auto-set timestamp when the account was created |
| `last_login` | Auto-updated timestamp on each authentication |

> If a user is a guest, their parking sessions are not linked to a user account.

---

### LicensePlate

License plates registered to a user account. A user can register multiple plates; each plate belongs to exactly one user.

| Field&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Description |
| :--- | :--- |
| `user` | The user account that owns this plate |
| `plate_text` | The text of the license plate |
| `is_primary` | Whether this is the user's primary plate |
| `label` | Optional user-side label to identify the plate |

---

### ParkingLot

Each record represents one parking lot.

| Field&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Description |
| :--- | :--- |
| `name` | The name of the parking lot |

---

### LotSettings

Settings for a specific parking lot.

| Field&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Description |
| :--- | :--- |
| `lot` | The parking lot these settings apply to |
| `rate` | Rate per billing unit (hour or minute) in dollars |
| `billing_unit` | Unit of time for the rate (`hour` or `minute`) |
| `grace_period_minutes` | Minutes before a charge is issued |
| `daily_cap_enabled` | Whether to enable the daily charge cap |
| `daily_cap_amount` | Maximum charge per session (the day rate) |
| `image_retention_days` | Days to keep uploaded plate images on disk |
| `confidence_threshold` | Confidence threshold for the CV pipeline |

---

### ParkingSession

The core transactional record — one row per car visit.

| Field&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Description |
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

---

### PlateDetectionEvent

The CV logging system — records every entry and exit event from the CV pipeline.

| Field&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Description |
| :--- | :--- |
| `session` | The parking session this event belongs to |
| `image` | Uploaded plate image file path |
| `raw_plate_text` | Plate text as read by the CV pipeline |
| `confidence_score` | Confidence score from the CV pipeline |
| `event_type` | `entry` or `exit` |
| `is_low_confidence` | Whether score is below the confidence threshold |
| `manually_corrected` | Whether an operator corrected the plate text |
| `corrected_plate` | The manually corrected plate text |
| `bounding_box` | Plate bounding box as a JSON array `[x, y, w, h]` |
| `timestamp` | Time the event was created |

---

## CV Pipeline

```mermaid
flowchart TD
    IMG(["photo path"])

    LOAD["load_image()"]
    BGR["bgr_to_rgb()"]
    RESIZE["resize_for_detector()"]
    NORM["normalize_pixels()"]
    TENSOR["to_tensor()"]

    DETECTOR(["PlateDetectorCNN"])

    CROP["crop_plate_region()"]
    PREP["prepare_for_recognizer()"]

    RECOG(["PlateRecognizerCRNN"])

    IMG --> LOAD --> BGR --> RESIZE --> NORM --> TENSOR --> DETECTOR --> CROP --> PREP --> RECOG

    classDef preproc fill:#1e40af,stroke:#1d4ed8,color:#fff
    classDef model fill:#6d28d9,stroke:#7c3aed,color:#fff
    classDef io fill:#0f766e,stroke:#0d9488,color:#fff

    class LOAD,BGR,RESIZE,NORM,TENSOR,CROP,PREP preproc
    class DETECTOR,RECOG model
    class IMG io
```

<details>
<summary><code>load_image(path)</code></summary>

Loads the image from disk using OpenCV after a security pre-check. Before any pixels are decoded, the resolved path is confirmed to stay inside `MEDIA_ROOT` (path traversal prevention), and Pillow inspects the file header to confirm the format is JPEG, PNG, or WEBP. Images larger than 12 MP (4000×3000) are rejected to prevent decompression bomb attacks. OpenCV then decodes the validated file into a BGR numpy array.

</details>

<details>
<summary><code>bgr_to_rgb(image)</code></summary>

Converts the color channel order from BGR to RGB. OpenCV always loads images in BGR order (blue–green–red), but PyTorch models trained on ImageNet expect RGB (red–green–blue). Without this swap, the model would see the red and blue channels swapped on every image, degrading accuracy. The conversion is done with `cv2.cvtColor` rather than slicing (`img[:, :, ::-1]`) because `cvtColor` produces a contiguous array that avoids a hidden memory copy later in the pipeline.

</details>

<details>
<summary><code>resize_for_detector(image)</code></summary>

Resizes the image to 640×480 pixels — the fixed input resolution the plate detector CNN expects. The resize uses letterboxing (padding the shorter dimension with a neutral fill) to preserve the original aspect ratio. Stretching the image to fit would distort plate shapes and hurt detection accuracy, especially for narrow or wide plates.

</details>

<details>
<summary><code>normalize_pixels(image)</code></summary>

Scales pixel values from the 0–255 integer range down to 0.0–1.0 floats by dividing by 255. Neural networks learn faster and more stably when inputs are in a small, consistent numeric range. Without normalization, large pixel values would cause large gradients and make the model sensitive to overall image brightness rather than plate features.

</details>

<details>
<summary><code>to_tensor(image)</code></summary>

Converts the numpy array to a PyTorch `FloatTensor` and reorders the axes from HWC (Height × Width × Channels) to CHW (Channels × Height × Width). PyTorch's convolutional layers expect channels first. The conversion also moves the data from CPU memory into a tensor that can be transferred to GPU/MPS for inference.

</details>



<details>
<summary><code>crop_plate_region(image, bbox)</code></summary>

Uses the detector's bounding box to crop just the plate area out of the full image. This gives the recognizer a tight view of the plate with minimal background clutter, which significantly improves character recognition accuracy. The crop is clamped to image bounds to handle any slight over-prediction from the detector.

</details>

<details>
<summary><code>prepare_for_recognizer(crop)</code></summary>

Resizes the plate crop to 128×32 pixels and converts it to grayscale. The recognizer operates on grayscale because plate text recognition is a shape task — color carries no useful signal and including it would triple the input size for no accuracy gain. The 128×32 resolution is wide enough to fit the longest plate text while being small enough to keep the encoder fast.

</details>


---
### Plate Detector CNN

`PlateDetectorCNN` is a custom convolutional neural network that finds where the
license plate is in the full image. It does not read the plate text; it predicts
one bounding box so the next model can crop the plate and recognize the
characters.

For a full beginner walkthrough, see
[`docs/plate-detector-cnn-explained.md`](docs/plate-detector-cnn-explained.md).

The detector has 3 main parts:

1. A convolutional backbone which extracts visual patterns from the image.
2. An adaptive pooling layer which reduces those patterns to a fixed `4 x 4` grid.
3. A fully connected regression head which outputs the plate box as `[cx, cy, w, h]`.

All four output values are normalized to `[0, 1]` relative to the image size.
The model uses `Dropout(0.3)` before the final output layer to reduce
overfitting on synthetic training data.

#### Convolutional Backbone

The detector receives input in this tensor shape:

```text
(batch size, 3, height, width)
```

The `3` means RGB color channels. Each convolution block uses:

```text
Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
```

- `Conv2d` learns visual patterns such as edges, corners, borders, and plate-like regions.
- `BatchNorm2d` keeps internal values stable so training is less chaotic.
- `ReLU` keeps useful positive signals and turns negative values into zero.
- `MaxPool2d` shrinks the feature map while keeping the strongest signals.

The three detector blocks learn increasingly abstract patterns:

1. Block 1 learns low-level details such as edges and color changes.
2. Block 2 combines those details into lines, borders, and rectangular outlines.
3. Block 3 learns higher-level plate-like regions with text-like texture inside.

For the normal `480 x 640` detector input, the spatial size changes like this:

```text
480 x 640
-> 240 x 320
-> 120 x 160
-> 60 x 80
```

Then `AdaptiveAvgPool2d((4, 4))` compresses the features to `(batch size, 128, 4, 4)`.
Flattening turns that into `2048` feature values per image.

#### Fully Connected Layer

The fully connected head compresses the `2048` features into `256` values and
then into `4` values:

```text
[cx, cy, w, h]
```

These represent the center position, width, and height of the plate box.
`sigmoid` is applied inside `forward()` so the output always stays in `[0, 1]`.

#### Training the Model

The model receives a batch of images and the expected bounding box for each
image. It predicts boxes, compares them to the expected boxes with
`SmoothL1Loss`, and uses backpropagation with the Adam optimizer to update the
weights. The detector training script is configured from command-line arguments;
the documented target is validation IoU above `0.7` after 50 epochs on
synthetic data.

#### Evaluation the Model

The model is evaluated on a held-out validation set. The training script reports
IoU, or Intersection over Union, which measures how much the predicted box
overlaps the correct box. Higher IoU values are better.

#### Saving the Model

The model is saved to the `apps/cv/weights/detector.pth` file.


### Plate Recognizer CRNN

`PlateRecognizerCRNN` is a custom convolutional-recurrent neural network which contains 3 main parts:
1. The convolutional backbone which looks through the image and extracts patterns,
2. A bidirectional LSTM which reads the image from left to right and right to left,
3. Output the predicted plate text with the help of the CTC loss function.

#### Convolutional Backbone

The convolutional backbone contains 3 layers:
In each layer, the input is processed through the following steps:
1. Conv2d which looks and defines patterns within the image. For example, it might detect the edge of the plate, or where the first letter of the plate is located.
2. BatchNorm2d which keeps the numbers stable during training, this prevents the model from constantly changing the weights of the nodes in the network. The normalize helps the model learn faster and reduces chaos.
3. ReLU which keeps useful positive signals and sets all negative signals to 0. This introduces non-linearity into the network and allows for the model to learn more complex patterns. inplace=True is used to avoid creating a new tensor and just overwriting the existing one each time.
4. MaxPool2d which shrinks the image size by taken the maximum value of the window. kernal size refers to the size of the window, and stride refers to the number of pixels to move the window by each time. In layer 3, the height is kept the same to prevent the model from losing information when trying to predict the plate text. For example, B and 8 would be hard to tell apart if the height was cut in half.

Layer 1 is focused mainly detecting the edges of the plate and possible letters within the plate. Layer 2 is focused on detecting the where the letters are located within the plate. Layer 3 is focused on detecting what the letters are.
The output becomes (batch size, feature channels, height, width) which is then flattened into a 1D tensor with shape (batch size, feature channels * height * width).

#### Bidirectional LSTM
The bidirectional LSTM reads the image from left to right and right to left. This helps the model to understand the context of the plate text and helps to avoid mistakes.
The input is a 1D tensor with shape (batch size, feature channels * height * width) which is given above, the output is a 1D tensor with shape (batch size, hidden size * 2).
The hidden size refers to the number of nodes in the hidden layer, the hidden layer like memory cells that the model uses to store information. The hidden size is doubled because the LSTM is bidirectional, so the output is the sum of the forward and backward hidden states. The output numbers from the hidden layer are turned into 37 scores (26 letters, 10 digits, and 1 blank) which represent the probability of each character which is then passed through a softmax function to get the final probabilities. The final output is a 1D tensor (shape, batch size, 37).

#### Greedy CTC Decoder
The function takes the predicted probabilities and turns them into a plate text. The function collapses consecutive identical tokens and removes blank tokens (index 0). If a two identical letters are preducted next to each other with a blank in between, the blank is removed and the two letters are kept.

#### Training the Model
The model is given a batch of plate images and the expected plate text. The model then outputs the predicted plate text based on the input image. The loss is calculated using the CTC loss function, which calculates the loss between the predicted plate text and the expected plate text. The loss is then backpropagated through the network to update the weights of the nodes in the network. The adam optimizer is used to update the weights of the nodes in the network. The learning rate is set to 0.001 and the batch size is set to 32. The model is trained for 100 epochs.

## Creating the Dataset for Training

### Preprocessing — `apps/cv/preprocessing.py`

Before an image enters the CV pipeline, it is preprocessed to reject bad input early and ensure consistency with deployment conditions. The preprocessing steps are:

1. Open the image using Pillow
2. Verify the compressed file size is under 64 MB
3. Verify the image is smaller than 12 MP (4000×3000)
4. Check the image is not corrupt
5. Confirm the image format is JPEG, PNG, or WEBP — checked from the file's magic bytes, not the file extension

If any check fails, the image is rejected and an error is logged. Otherwise, the loaded image proceeds to the next stage of the pipeline.

### Synthetic Data Generation — `apps/cv/training/synthetic_data.py`

Plates are generated for both Canadian and United States formats at 400×120 pixels. A few setup rules apply to the assets:

- Background images and font files are cached to avoid reloading on each generation.
- Background images must be `.jpg`, `.jpeg`, or `.png`.
- The font must be a TrueType plate font — if the font file is missing or unreadable, Pillow's default font is used as a fallback.

**How a synthetic image is built**

1. **Generate the plate text** randomly, following the format conventions of each country:
   - US plates use one of three formats: `ABC 1234` (most common), `123 ABC`, or `ABC123`.
   - Canadian plates use one of two formats: `ABC 123` or `A1B 2C3` (Ontario-style alphanumeric).
2. **Build the plate background** — a rectangle with a white background that's fully opaque. Canadian plates also have a solid blue strip across the top quarter to visually differentiate them from US plates. A dark border is added to the edges of the plate to help the AI detector model learn the plate's boundaries.
3. **Render the plate text** onto the plate background using the font file. The `textbbox` function determines the center of the plate, and then the text is drawn onto the center of the plate in black ink using the `draw.text` function.
4. **Composite onto a background** — the plate is placed onto a random background image. The background image is resized to 640×480 pixels and then the plate is pasted onto it at a random position and scale. The plate is then rotated between -15 and 15 degrees to simulate the camera angle of the camera in the parking lot. The plate's position is constrained so it always fits fully within the background image. Then the full image is converted to an RGB image and the bounding box coordinates of the plate are returned.

Two builders use this process to produce the training datasets:

- **Detector dataset** — generates 10,000 plate + background images by default. Before generating, any existing `.jpg` and `.txt` files in the output directory are deleted so re-runs do not mix data from previous runs. Images are saved to `data/detector/images` as `.jpg`. A corresponding `.txt` file is generated for each image in `data/detector/labels`, containing the bounding box in YOLO format: class index, normalized center x, normalized center y, normalized width, and normalized height (all values between 0 and 1).
- **Recognizer dataset** — generates 50,000 plates by default; only the cropped plate images are saved, in grayscale. Before generating, any existing `.png` files in the output directory are deleted and `labels.csv` is rewritten from scratch so re-runs do not mix data from previous runs. Plates are saved to `data/recognizer/images` as `.png`. A `labels.csv` file is also generated, containing the filename of the cropped plate image, the plate text, and the country.

Both functions accept an optional `seed` parameter to make the generated dataset reproducible across runs.

### Loading training data — `apps/cv/training/dataset.py`

After `synthetic_data.py` writes files under `data/detector/` and `data/recognizer/`, this module loads them for PyTorch training. Default transforms convert images to tensors with **pixel values in [0, 1]** (not the height/width indices).

**Character encoding (recognizer only)**

- `A→1` … `Z→26`, `0→27` … `9→36`
- Index `0` is reserved for the CTC blank token
- **Spaces are skipped** (not encoded)
- Each `PlateRecognizerDataset` sample returns a **list** of indices; `ctc_collate_fn` builds the batch tensors (below)

**`PlateDetectorDataset`** (`data/detector/`)

1. At startup, scans `images/*.jpg` (skips symlinks) and pairs each file with `labels/<same-stem>.txt`.
2. Each label file is one YOLO line: `0 cx cy w h`. The leading class `0` is dropped; the four floats are the box `[cx, cy, w, h]` (normalized between 0 and 1).
3. `__getitem__` loads the JPG with `safe_open_image`, converts to an RGB tensor `(3, H, W)`, and returns `(image_tensor, bbox_tensor)` where `bbox_tensor` has shape `(4,)`.
4. A `DataLoader` (e.g. batch size 32) uses **default collate** — not using `ctc_collate_fn`. It stacks batches to `(N, 3, H, W)` and `(N, 4)`.

**`PlateRecognizerDataset`** (`data/recognizer/`)

1. At startup, reads `labels.csv` (`filename`, `text`; `country` is stored but not returned per sample).
2. `__getitem__` loads the matching PNG from `images/`, converts to a grayscale tensor `(1, 32, 128)`, encodes text to a list of indices (spaces skipped), and returns `(image_tensor, label_list)`.
3. A `DataLoader` must set `collate_fn=ctc_collate_fn` because label lengths vary. For each batch it:
   - Unzips the list of `(image, label_list)` pairs into separate image and label lists
   - Stacks images with `torch.stack` → `(N, 1, 32, 128)`
   - Concatenates all label lists into one 1D `targets` tensor
   - Builds `target_lengths` (how many indices belong to each sample)
   - Returns `{"images", "targets", "target_lengths"}` for CTC training

### Augmentations — `apps/cv/training/augment.py`

While the model is **learning**, we slightly change each training image (brighter, blurrier, flipped, and so on) so practice pictures feel more like real parking cameras. This file does **not** load images from disk — `dataset.py` does that first; augment only tweaks the numbers already in memory.

**Two modes**

- **`train=True`** — random changes each time (used during training).
- **`train=False`** — no random changes; only **normalize** (rescales pixel numbers for the network). Used when checking accuracy.

**Normalize** means: adjust each pixel with a fixed formula `(pixel - mean) / std` so the model gets inputs in the range it expects. This is not resizing the image.

**Detector** (`DetectorAugment`) — full parking-lot photo, color:

- Random brightness/contrast/color tweaks (different lighting).
- Random slight blur.
- 50% chance to flip the image left–right (car can come from either direction).
- Sometimes turn the image grayscale (10% chance), like a black-and-white security camera.
- Then normalize (ImageNet mean/std — standard for color models).

**Recognizer** (`RecognizerAugment`) — small gray plate crop only:

- Random brightness/contrast tweaks (faded or dirty plates).
- Random slight blur.
- 50% chance of a mild “angled camera” warp (not a full flip).
- Then normalize (simple gray scale: mean 0.5, std 0.5).

**Important:** the recognizer **never** flips the image horizontally. `"ABC 123"` backwards would not match the answer in `labels.csv`. The detector **can** flip because we only care where the plate is, not reading the text yet.

## Web Application

Django 5.1 backend with HTMX for reactive partials and Chart.js for revenue visualization. No Node.js, no React — server-rendered templates with targeted DOM swaps.

All pages require authentication. No public routes.

---

## Docker

The application runs as two containers orchestrated by Docker Compose:

| Container&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Description |
| :--- | :--- |
| `db` | PostgreSQL 16 with a persistent named volume |
| `web` | Django served by Gunicorn on port 8000 |

```bash
# Start all services
docker-compose up --build

# Run migrations
docker-compose exec web python manage.py migrate

# Seed initial data — creates a superuser, default ParkingLot, and LotSettings (safe to run multiple times)
docker-compose exec web python manage.py setup_defaults

# Create an admin user
docker-compose exec web python manage.py createsuperuser

# Run the test suite
docker-compose exec web pytest --cov=apps/accounts --cov=apps/parking --cov-fail-under=80
```
