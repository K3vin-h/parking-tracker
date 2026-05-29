"""
PyTorch Dataset classes for the plate detector and recognizer.

Character encoding (shared by both datasets and the training scripts):

    Index 0 → CTC blank token (reserved — CTCLoss convention)
    A → 1, B → 2, …, Z → 26
    0 → 27, 1 → 28, …, 9 → 36
    VOCAB_SIZE = 37  (26 letters + 10 digits + 1 blank)

Spaces in plate text are intentionally excluded from the encoded label because
the recognizer outputs one token per time-step and CTC alignment handles
variable-length sequences; the space character carries no visual signal.

DataLoader usage:

    Detector (fixed-length labels — default collate works):
        loader = DataLoader(PlateDetectorDataset(root), batch_size=32, shuffle=True)

    Recognizer (variable-length labels — CTC collation required):
        loader = DataLoader(
            PlateRecognizerDataset(root),
            batch_size=32,
            collate_fn=ctc_collate_fn,
        )
"""

import csv
import logging
import string
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from apps.cv.training._image_io import safe_open_image

logger = logging.getLogger(__name__)

# ── Character encoding constants ──────────────────────────────────────────────

BLANK_IDX: int = 0
_CHARS: str = string.ascii_uppercase + string.digits   # A-Z then 0-9

# Assign indices starting at 1 so index 0 is free for the CTC blank
CHAR_TO_IDX: dict[str, int] = {ch: i + 1 for i, ch in enumerate(_CHARS)}
IDX_TO_CHAR: dict[int, str] = {v: k for k, v in CHAR_TO_IDX.items()}
#A->1, B->2, ... Z->26, 0->27, 1->28, ... 9->36

VOCAB_SIZE: int = len(_CHARS) + 1  # 37

# ── Default transforms ────────────────────────────────────────────────────────
#
# These are intentionally minimal (tensor conversion + dtype normalisation only).
# Augmentation is handled separately by DetectorAugment / RecognizerAugment so
# callers can combine transforms freely without order surprises.

_DETECTOR_DEFAULT_TRANSFORM = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

_RECOGNIZER_DEFAULT_TRANSFORM = v2.Compose([
    v2.Grayscale(num_output_channels=1),  # no-op if already L-mode; handles RGB
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])


# ── CTC collate helper ────────────────────────────────────────────────────────

def ctc_collate_fn(
    batch: list[tuple[torch.Tensor, list[int]]],
) -> dict[str, torch.Tensor]:
    """
    Collate variable-length recognizer labels for torch.nn.CTCLoss.

    CTCLoss expects:
        input:          (T, N, C) — encoder output (T time steps, N batch, C vocab)
        targets:        1-D concat of all label sequences
        input_lengths:  (N,) — each item's T (supplied by the encoder at train time)
        target_lengths: (N,) — each item's label length

    This function produces targets and target_lengths from the batch.
    input_lengths must be supplied by the training loop after the encoder runs.

    Args:
        batch: List of (image_tensor, label_indices) from PlateRecognizerDataset.

    Returns:
        dict with keys:
            "images"         — (N, 1, H, W) float32 tensor
            "targets"        — 1-D long tensor (concatenated labels)
            "target_lengths" — (N,) long tensor
    """
    images, labels = zip(*batch)

    # CTCLoss requires target_length >= 1 for every sample; an empty label would
    # cause a silent NaN loss or a cryptic runtime error inside the training loop.
    if any(len(label) == 0 for label in labels):
        raise ValueError(
            "ctc_collate_fn received a sample with an empty label. "
            "Every plate text must encode to at least one character index."
        )

    images_stacked = torch.stack(list(images))
    targets = torch.tensor(
        [idx for label in labels for idx in label], dtype=torch.long
    )
    target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    return {
        "images": images_stacked,
        "targets": targets,
        "target_lengths": target_lengths,
    }


# ── Dataset classes ───────────────────────────────────────────────────────────

class PlateDetectorDataset(Dataset):
    """
    Dataset of composite parking-lot images with YOLO-format bounding box labels.

    Expected on-disk layout (produced by generate_detector_dataset):
        root/images/<name>.jpg
        root/labels/<name>.txt  — one line: "0 cx cy w h" (coords normalised 0–1)

    Args:
        root:      Path to the dataset root directory.
        transform: Optional transform applied to the image tensor.
                   Defaults to ToImage + ToDtype(float32, scale=True).

    Raises:
        FileNotFoundError: root does not exist.
        ValueError:        root/images/ contains no .jpg files.
    """

    def __init__(self, root: Path, transform=None) -> None:
        if not root.exists():
            raise FileNotFoundError(f"Dataset root not found: {root.name}")

        self._img_dir = root / "images"
        self._lbl_dir = root / "labels"
        self._transform = transform or _DETECTOR_DEFAULT_TRANSFORM

        # Skip symlinks: a tampered dataset directory could otherwise point
        # individual ".jpg" entries at arbitrary host files. Real images
        # produced by generate_detector_dataset are plain files.
        self._samples: list[Path] = sorted(
            p for p in self._img_dir.glob("*.jpg") if not p.is_symlink()
        )
        if not self._samples:
            raise ValueError("No .jpg images found in dataset root/images/.")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return (image_tensor, bbox_tensor).

        image_tensor: float32, shape (3, H, W), values in [0, 1] before augmentation.
        bbox_tensor:  float32, shape (4,) — [cx, cy, w, h] normalised 0–1.
        """
        img_path = self._samples[idx]
        lbl_path = self._lbl_dir / (img_path.stem + ".txt")

        img = safe_open_image(img_path).convert("RGB")
        img_t: torch.Tensor = self._transform(img)

        # YOLO label format: "class cx cy w h" — validate before use so a malformed
        # or poisoned label file fails loudly here rather than silently corrupting
        # training with a wrong-shape tensor or a cryptic DataLoader worker crash.
        # An explicit existence check produces a descriptive error naming the
        # missing file rather than the bare "[Errno 2]" trace from read_text.
        if not lbl_path.exists():
            raise FileNotFoundError(
                f"Label file {lbl_path.name} is missing for image "
                f"{img_path.name}. Every image must have a paired label."
            )
        parts = lbl_path.read_text().strip().split()
        if len(parts) != 5:
            raise ValueError(
                f"Malformed label {lbl_path.name}: expected 5 fields, got {len(parts)}."
            )
        bbox = torch.tensor([float(v) for v in parts[1:]], dtype=torch.float32)

        return img_t, bbox


class PlateRecognizerDataset(Dataset):
    """
    Dataset of cropped plate images (128×32 grayscale) with text labels.

    Expected on-disk layout (produced by generate_recognizer_dataset):
        root/images/<name>.png  — 128×32 grayscale
        root/labels.csv         — columns: filename, text, country

    Labels are encoded as lists of CHAR_TO_IDX indices with spaces excluded.
    Use ctc_collate_fn when constructing a DataLoader.

    Args:
        root:      Path to the dataset root directory.
        transform: Optional transform applied to the image tensor.
                   Defaults to Grayscale + ToImage + ToDtype(float32, scale=True).

    Raises:
        FileNotFoundError: root or labels.csv does not exist.
        ValueError:        labels.csv contains no data rows.
    """

    def __init__(self, root: Path, transform=None) -> None:
        if not root.exists():
            raise FileNotFoundError(f"Dataset root not found: {root.name}")

        csv_path = root / "labels.csv"
        if not csv_path.exists():
            raise FileNotFoundError("labels.csv not found in dataset root.")

        self._img_dir = root / "images"
        # Resolve once at init for cheap path-traversal checks in __getitem__
        self._img_dir_resolved = self._img_dir.resolve()
        self._transform = transform or _RECOGNIZER_DEFAULT_TRANSFORM

        self._samples: list[tuple[str, str]] = []  # (filename, plate_text)
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._samples.append((row["filename"], row["text"]))

        if not self._samples:
            raise ValueError("Dataset is empty: labels.csv has no data rows.")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, list[int]]:
        """
        Return (image_tensor, label_indices).

        image_tensor:   float32, shape (1, 32, 128), values in [0, 1].
        label_indices:  list[int] — CHAR_TO_IDX-encoded text, spaces excluded.
        """
        filename, text = self._samples[idx]

        # Guard against path traversal via a tampered labels.csv (e.g. filename
        # containing "../../../config/settings.py"). Resolve and check containment.
        img_path = (self._img_dir / filename).resolve()
        if not img_path.is_relative_to(self._img_dir_resolved):
            raise ValueError(
                "labels.csv contains a path that escapes the dataset directory."
            )

        img = safe_open_image(img_path)
        img_t: torch.Tensor = self._transform(img)

        # Spaces are intentionally excluded (no visual signal for the recognizer).
        # Any other unknown character is a data error — raise rather than silently
        # produce a shorter label, which would corrupt CTCLoss alignment.
        label = []
        for ch in text:
            if ch == " ":
                continue
            if ch not in CHAR_TO_IDX:
                raise ValueError(
                    f"Unrecognised character {ch!r} in plate text {text!r}. "
                    "Add to CHAR_TO_IDX or fix the dataset."
                )
            label.append(CHAR_TO_IDX[ch])

        return img_t, label
