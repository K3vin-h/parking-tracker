"""
Unit tests for apps/cv/training/dataset.py.

Fixtures build minimal synthetic datasets in tmp_path directories so tests
run without any pre-generated training data on disk. No database is touched.
"""
import csv
import string
from pathlib import Path

import pytest
import torch
from PIL import Image

from apps.cv.training.dataset import (
    BLANK_IDX,
    CHAR_TO_IDX,
    IDX_TO_CHAR,
    VOCAB_SIZE,
    PlateDetectorDataset,
    PlateRecognizerDataset,
    ctc_collate_fn,
)


# ── Dataset fixture helpers ───────────────────────────────────────────────────

def _make_detector_root(root: Path, n: int = 3) -> None:
    """Write n synthetic YOLO-format samples to root."""
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    for i in range(n):
        Image.new("RGB", (640, 480), (128, 128, 128)).save(
            root / "images" / f"{i:06d}.jpg"
        )
        (root / "labels" / f"{i:06d}.txt").write_text(
            "0 0.500000 0.500000 0.200000 0.100000\n"
        )


def _make_recognizer_root(root: Path, n: int = 3, text: str = "ABC123") -> None:
    """Write n synthetic recognizer samples to root."""
    (root / "images").mkdir(parents=True)
    rows = []
    for i in range(n):
        filename = f"{i:06d}.png"
        Image.new("L", (128, 32), 200).save(root / "images" / filename)
        rows.append({"filename": filename, "text": text, "country": "US"})
    with (root / "labels.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "text", "country"])
        writer.writeheader()
        writer.writerows(rows)


# ── Character encoding ────────────────────────────────────────────────────────

@pytest.mark.unit
class TestCharacterEncoding:
    def test_blank_idx_is_zero(self):
        """CTC requires blank at index 0 — this is a hard invariant."""
        assert BLANK_IDX == 0

    def test_vocab_size_is_37(self):
        """26 uppercase letters + 10 digits + 1 CTC blank = 37."""
        assert VOCAB_SIZE == 37

    def test_all_uppercase_letters_present(self):
        """Every A–Z must have a unique non-zero index."""
        for ch in string.ascii_uppercase:
            assert ch in CHAR_TO_IDX, f"Missing character {ch!r}"
            assert CHAR_TO_IDX[ch] != BLANK_IDX

    def test_all_digits_present(self):
        """Every 0–9 must have a unique non-zero index."""
        for ch in string.digits:
            assert ch in CHAR_TO_IDX, f"Missing digit {ch!r}"
            assert CHAR_TO_IDX[ch] != BLANK_IDX

    def test_round_trip_all_chars(self):
        """CHAR_TO_IDX → IDX_TO_CHAR must be a perfect round-trip."""
        for ch in string.ascii_uppercase + string.digits:
            assert IDX_TO_CHAR[CHAR_TO_IDX[ch]] == ch

    def test_no_char_maps_to_blank_idx(self):
        """No character should collide with the CTC blank index (0)."""
        assert BLANK_IDX not in CHAR_TO_IDX.values()

    def test_indices_are_contiguous_from_one(self):
        """All assigned indices must be 1, 2, …, VOCAB_SIZE-1 with no gaps."""
        assert sorted(CHAR_TO_IDX.values()) == list(range(1, VOCAB_SIZE))


# ── PlateDetectorDataset ──────────────────────────────────────────────────────

@pytest.mark.unit
class TestPlateDetectorDataset:
    def test_len_matches_sample_count(self, tmp_path: Path):
        """__len__ must equal the number of .jpg images in root/images/."""
        root = tmp_path / "det"
        _make_detector_root(root, n=3)
        assert len(PlateDetectorDataset(root)) == 3

    def test_image_tensor_dtype(self, tmp_path: Path):
        """Image tensor must be float32."""
        root = tmp_path / "det"
        _make_detector_root(root, n=1)
        img, _ = PlateDetectorDataset(root)[0]
        assert img.dtype == torch.float32

    def test_image_tensor_channels(self, tmp_path: Path):
        """Image tensor must have 3 channels (RGB)."""
        root = tmp_path / "det"
        _make_detector_root(root, n=1)
        img, _ = PlateDetectorDataset(root)[0]
        assert img.ndim == 3 and img.shape[0] == 3

    def test_image_tensor_values_in_unit_range(self, tmp_path: Path):
        """Pixel values must be in [0, 1] after default float conversion."""
        root = tmp_path / "det"
        _make_detector_root(root, n=1)
        img, _ = PlateDetectorDataset(root)[0]
        assert img.min() >= 0.0 and img.max() <= 1.0

    def test_bbox_shape(self, tmp_path: Path):
        """Bounding box tensor must have shape (4,)."""
        root = tmp_path / "det"
        _make_detector_root(root, n=1)
        _, bbox = PlateDetectorDataset(root)[0]
        assert bbox.shape == (4,)

    def test_bbox_dtype(self, tmp_path: Path):
        """Bounding box tensor must be float32."""
        root = tmp_path / "det"
        _make_detector_root(root, n=1)
        _, bbox = PlateDetectorDataset(root)[0]
        assert bbox.dtype == torch.float32

    def test_bbox_values_normalised(self, tmp_path: Path):
        """All bounding box coordinates must be in [0, 1]."""
        root = tmp_path / "det"
        _make_detector_root(root, n=1)
        _, bbox = PlateDetectorDataset(root)[0]
        assert (bbox >= 0.0).all() and (bbox <= 1.0).all()

    def test_missing_root_raises_file_not_found(self, tmp_path: Path):
        """FileNotFoundError when root directory does not exist."""
        with pytest.raises(FileNotFoundError):
            PlateDetectorDataset(tmp_path / "nonexistent")

    def test_empty_images_dir_raises_value_error(self, tmp_path: Path):
        """ValueError when root/images/ contains no .jpg files."""
        root = tmp_path / "empty_det"
        (root / "images").mkdir(parents=True)
        (root / "labels").mkdir()
        with pytest.raises(ValueError):
            PlateDetectorDataset(root)

    def test_malformed_label_raises_value_error(self, tmp_path: Path):
        """ValueError when a label file has wrong column count (poisoned dataset guard)."""
        root = tmp_path / "det_bad_label"
        _make_detector_root(root, n=1)
        # Overwrite the label with malformed content
        list((root / "labels").glob("*.txt"))[0].write_text("0 0.5 0.5\n")
        ds = PlateDetectorDataset(root)
        with pytest.raises(ValueError, match="Malformed label"):
            ds[0]

    def test_missing_label_file_raises_descriptive_error(self, tmp_path: Path):
        """
        FileNotFoundError with the missing filename when the label is absent.

        Regression guard: the previous code let ``Path.read_text`` raise a
        bare ``[Errno 2]`` FileNotFoundError with no context about which
        sample was the problem.
        """
        root = tmp_path / "det_no_label"
        (root / "images").mkdir(parents=True)
        (root / "labels").mkdir()
        Image.new("RGB", (640, 480), (128, 128, 128)).save(
            root / "images" / "000000.jpg"
        )
        ds = PlateDetectorDataset(root)
        with pytest.raises(FileNotFoundError, match="000000"):
            ds[0]

    def test_symlinks_in_images_are_skipped(self, tmp_path: Path):
        """
        Symlinks under root/images/ must be excluded from the dataset.

        A tampered dataset directory could otherwise expose arbitrary host
        files as ``.jpg`` entries. The Dataset class strips them at __init__
        time so subsequent ``__getitem__`` calls only see plain files.
        """
        root = tmp_path / "det_symlink"
        (root / "images").mkdir(parents=True)
        (root / "labels").mkdir()
        # Plain file that should remain in the dataset
        Image.new("RGB", (640, 480), (200, 200, 200)).save(
            root / "images" / "real.jpg"
        )
        (root / "labels" / "real.txt").write_text("0 0.5 0.5 0.2 0.1\n")
        # Symlink pointing at a file outside the dataset — must be dropped
        outside = tmp_path / "outside.jpg"
        Image.new("RGB", (640, 480), (10, 10, 10)).save(outside)
        try:
            (root / "images" / "linked.jpg").symlink_to(outside)
        except (NotImplementedError, OSError):
            pytest.skip("Filesystem does not support symlinks")

        ds = PlateDetectorDataset(root)
        assert len(ds) == 1
        # Confirm the surviving sample is the real file, not the symlink
        assert ds._samples[0].name == "real.jpg"


# ── PlateRecognizerDataset ────────────────────────────────────────────────────

@pytest.mark.unit
class TestPlateRecognizerDataset:
    def test_len_matches_csv_rows(self, tmp_path: Path):
        """__len__ must equal the number of data rows in labels.csv."""
        root = tmp_path / "rec"
        _make_recognizer_root(root, n=4)
        assert len(PlateRecognizerDataset(root)) == 4

    def test_image_tensor_shape(self, tmp_path: Path):
        """Image tensor must have shape (1, 32, 128) — grayscale plate crop."""
        root = tmp_path / "rec"
        _make_recognizer_root(root, n=1)
        img, _ = PlateRecognizerDataset(root)[0]
        assert img.shape == (1, 32, 128), f"Unexpected shape {img.shape}"

    def test_image_tensor_dtype(self, tmp_path: Path):
        """Image tensor must be float32."""
        root = tmp_path / "rec"
        _make_recognizer_root(root, n=1)
        img, _ = PlateRecognizerDataset(root)[0]
        assert img.dtype == torch.float32

    def test_label_is_list_of_ints(self, tmp_path: Path):
        """Label must be a list[int] encoded via CHAR_TO_IDX."""
        root = tmp_path / "rec"
        _make_recognizer_root(root, n=1)
        _, label = PlateRecognizerDataset(root)[0]
        assert isinstance(label, list)
        assert all(isinstance(v, int) for v in label)

    def test_label_excludes_spaces(self, tmp_path: Path):
        """Spaces in plate text must be skipped — they have no CHAR_TO_IDX entry."""
        root = tmp_path / "rec_space"
        (root / "images").mkdir(parents=True)
        Image.new("L", (128, 32), 200).save(root / "images" / "000000.png")
        with (root / "labels.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "text", "country"])
            writer.writeheader()
            writer.writerow({"filename": "000000.png", "text": "ABC 123", "country": "US"})

        _, label = PlateRecognizerDataset(root)[0]
        # "ABC 123" has 6 non-space chars → 6 indices
        assert len(label) == 6

    def test_label_indices_are_valid(self, tmp_path: Path):
        """All label indices must be within [1, VOCAB_SIZE - 1]."""
        root = tmp_path / "rec"
        _make_recognizer_root(root, n=1)
        _, label = PlateRecognizerDataset(root)[0]
        for idx in label:
            assert 1 <= idx < VOCAB_SIZE, f"Index {idx} out of valid range"

    def test_missing_root_raises_file_not_found(self, tmp_path: Path):
        """FileNotFoundError when root directory does not exist."""
        with pytest.raises(FileNotFoundError):
            PlateRecognizerDataset(tmp_path / "nonexistent")

    def test_empty_csv_raises_value_error(self, tmp_path: Path):
        """ValueError when labels.csv has a header but no data rows."""
        root = tmp_path / "empty_rec"
        (root / "images").mkdir(parents=True)
        with (root / "labels.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "text", "country"])
            writer.writeheader()
        with pytest.raises(ValueError):
            PlateRecognizerDataset(root)

    def test_unknown_char_in_text_raises(self, tmp_path: Path):
        """ValueError when plate text contains a character not in CHAR_TO_IDX."""
        root = tmp_path / "rec_bad_char"
        (root / "images").mkdir(parents=True)
        Image.new("L", (128, 32), 200).save(root / "images" / "000000.png")
        with (root / "labels.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "text", "country"])
            writer.writeheader()
            # Hyphen is not in CHAR_TO_IDX — must raise, not silently drop
            writer.writerow({"filename": "000000.png", "text": "ABC-123", "country": "US"})
        ds = PlateRecognizerDataset(root)
        with pytest.raises(ValueError, match="Unrecognised character"):
            ds[0]

    def test_path_traversal_in_csv_raises(self, tmp_path: Path):
        """ValueError when labels.csv filename escapes the images directory."""
        root = tmp_path / "rec_traversal"
        (root / "images").mkdir(parents=True)
        with (root / "labels.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "text", "country"])
            writer.writeheader()
            writer.writerow({"filename": "../../../etc/passwd", "text": "ABC123", "country": "US"})
        ds = PlateRecognizerDataset(root)
        with pytest.raises(ValueError, match="escapes the dataset directory"):
            ds[0]


# ── ctc_collate_fn ────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestCtcCollateFn:
    def _batch(self, label_lengths: list[int]) -> list[tuple[torch.Tensor, list[int]]]:
        """Build a batch with grayscale images and labels of given lengths."""
        return [
            (torch.rand(1, 32, 128), list(range(1, n + 1)))
            for n in label_lengths
        ]

    def test_output_keys(self):
        """Collated batch must contain images, targets, and target_lengths."""
        out = ctc_collate_fn(self._batch([3, 2]))
        assert set(out.keys()) == {"images", "targets", "target_lengths"}

    def test_images_batch_dimension(self):
        """images tensor must be (N, C, H, W)."""
        out = ctc_collate_fn(self._batch([3, 3]))
        assert out["images"].shape == (2, 1, 32, 128)

    def test_targets_are_concatenated(self):
        """targets must be the 1-D concatenation of all label sequences."""
        batch = [
            (torch.zeros(1, 32, 128), [1, 2]),
            (torch.zeros(1, 32, 128), [3, 4, 5]),
        ]
        out = ctc_collate_fn(batch)
        assert out["targets"].tolist() == [1, 2, 3, 4, 5]

    def test_target_lengths_are_per_sample(self):
        """target_lengths must hold each sample's individual label length."""
        batch = [
            (torch.zeros(1, 32, 128), [1, 2]),
            (torch.zeros(1, 32, 128), [3, 4, 5]),
        ]
        out = ctc_collate_fn(batch)
        assert out["target_lengths"].tolist() == [2, 3]

    def test_targets_dtype_is_long(self):
        """targets must be torch.long (CTCLoss requirement)."""
        out = ctc_collate_fn(self._batch([2]))
        assert out["targets"].dtype == torch.long

    def test_target_lengths_dtype_is_long(self):
        """target_lengths must be torch.long (CTCLoss requirement)."""
        out = ctc_collate_fn(self._batch([2]))
        assert out["target_lengths"].dtype == torch.long

    def test_empty_label_raises(self):
        """ValueError when any sample has an empty label (CTCLoss requires length >= 1)."""
        batch = [
            (torch.zeros(1, 32, 128), []),   # empty label — invalid for CTC
        ]
        with pytest.raises(ValueError, match="empty label"):
            ctc_collate_fn(batch)
