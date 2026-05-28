"""
Unit tests for apps/cv/training/synthetic_data.py.

All tests are fully isolated — no database, no network, no real background
files. The composite_on_background tests use a fixture that creates a single
solid-colour background image in a tmp_path directory so the function has
something to open without touching production assets.
"""
import csv
import string
from pathlib import Path

import pytest
from PIL import Image

from apps.cv.training.synthetic_data import (
    PLATE_SIZE,
    composite_on_background,
    generate_detector_dataset,
    generate_plate_text,
    generate_recognizer_dataset,
    render_plate_image,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def bg_dir(tmp_path: Path) -> Path:
    """Background directory containing one synthetic solid-colour JPEG."""
    d = tmp_path / "backgrounds"
    d.mkdir()
    Image.new("RGB", (640, 480), (100, 149, 237)).save(d / "bg.jpg")
    return d


# ── generate_plate_text ───────────────────────────────────────────────────────

@pytest.mark.unit
class TestGeneratePlateText:
    _VALID_CHARS = set(string.ascii_uppercase + string.digits + " ")

    def test_returns_two_element_tuple(self):
        """Return value must be a (str, str) 2-tuple."""
        result = generate_plate_text("US")
        assert isinstance(result, tuple) and len(result) == 2
        assert all(isinstance(v, str) for v in result)

    def test_us_country_tag(self):
        """Requesting 'US' must always return country == 'US'."""
        for _ in range(20):
            _, country = generate_plate_text("US")
            assert country == "US"

    def test_ca_country_tag(self):
        """Requesting 'CA' must always return country == 'CA'."""
        for _ in range(20):
            _, country = generate_plate_text("CA")
            assert country == "CA"

    @pytest.mark.parametrize("country", ["US", "CA"])
    def test_text_contains_only_valid_chars(self, country: str):
        """Plate text must use only uppercase letters, digits, and spaces."""
        for _ in range(50):
            text, _ = generate_plate_text(country)
            assert set(text).issubset(self._VALID_CHARS), (
                f"Invalid characters in {text!r}"
            )

    def test_random_mode_produces_both_countries(self):
        """'random' must produce both US and CA across many samples."""
        countries = {generate_plate_text("random")[1] for _ in range(300)}
        assert "US" in countries
        assert "CA" in countries

    def test_unknown_country_raises_value_error(self):
        """An unsupported country code must raise ValueError immediately."""
        with pytest.raises(ValueError, match="Unknown country"):
            generate_plate_text("UK")


# ── render_plate_image ────────────────────────────────────────────────────────

@pytest.mark.unit
class TestRenderPlateImage:
    def test_returns_pil_image(self):
        """Return type must be PIL.Image.Image."""
        text, country = generate_plate_text("US")
        assert isinstance(render_plate_image(text, country), Image.Image)

    def test_size_matches_plate_size_constant(self):
        """Rendered image must be exactly PLATE_SIZE (width × height)."""
        text, country = generate_plate_text("US")
        assert render_plate_image(text, country).size == PLATE_SIZE

    def test_mode_is_rgba(self):
        """Output must be RGBA — alpha channel is required for compositing."""
        for country in ("US", "CA"):
            text, _ = generate_plate_text(country)
            assert render_plate_image(text, country).mode == "RGBA"

    def test_us_background_not_all_black(self):
        """
        US plate centre pixel should not be pure black.

        A completely black centre would indicate the text or background fill
        has written over the entire plate, which would break compositing.
        """
        img = render_plate_image("ABC 1234", "US")
        cx, cy = img.size[0] // 2, img.size[1] // 2
        # Convert to RGB before sampling to normalise away alpha
        r, g, b = img.convert("RGB").getpixel((cx, cy))
        assert max(r, g, b) > 50, "Plate centre appears completely black"

    def test_ca_renders_without_error(self):
        """Canadian plate rendering must complete and return the correct size."""
        img = render_plate_image("ABC 123", "CA")
        assert img.size == PLATE_SIZE

    @pytest.mark.parametrize("text,country", [
        ("ABC 1234", "US"),
        ("123 ABC", "US"),
        ("ABC123", "US"),
        ("ABC 123", "CA"),
        ("A1B 2C3", "CA"),
    ])
    def test_parametrized_formats_render_successfully(self, text: str, country: str):
        """Each supported plate format string must render without error."""
        img = render_plate_image(text, country)
        assert isinstance(img, Image.Image)


# ── composite_on_background ───────────────────────────────────────────────────

@pytest.mark.unit
class TestCompositeOnBackground:
    def test_returns_rgb_image(self, bg_dir: Path):
        """Composite must be an RGB PIL Image (alpha consumed during paste)."""
        plate = render_plate_image("XYZ 123", "US")
        composite, _ = composite_on_background(plate, bg_dir)
        assert composite.mode == "RGB"

    def test_returns_four_element_bbox(self, bg_dir: Path):
        """BBox return value must be a list of exactly 4 integers."""
        plate = render_plate_image("ABC 456", "US")
        _, bbox = composite_on_background(plate, bg_dir)
        assert isinstance(bbox, list) and len(bbox) == 4

    def test_bbox_lies_within_image_bounds(self, bg_dir: Path):
        """Plate must be fully visible — no pixel may fall outside the canvas."""
        plate = render_plate_image("TST 001", "US")
        for _ in range(10):
            composite, (x, y, w, h) = composite_on_background(plate, bg_dir)
            iw, ih = composite.size
            assert x >= 0 and y >= 0, "BBox origin is negative"
            assert x + w <= iw, f"BBox right edge {x + w} exceeds image width {iw}"
            assert y + h <= ih, f"BBox bottom edge {y + h} exceeds image height {ih}"

    def test_target_size_is_respected(self, bg_dir: Path):
        """The composite image must match the requested target_size exactly."""
        plate = render_plate_image("AAA 111", "US")
        composite, _ = composite_on_background(plate, bg_dir, target_size=(800, 600))
        assert composite.size == (800, 600)

    def test_missing_bg_dir_raises(self, tmp_path: Path):
        """FileNotFoundError when bg_dir does not exist."""
        plate = render_plate_image("ERR 000", "US")
        with pytest.raises(FileNotFoundError):
            composite_on_background(plate, tmp_path / "nonexistent")

    def test_empty_bg_dir_raises(self, tmp_path: Path):
        """FileNotFoundError when bg_dir exists but contains no valid images."""
        empty = tmp_path / "empty_bg"
        empty.mkdir()
        plate = render_plate_image("ERR 001", "US")
        with pytest.raises(FileNotFoundError):
            composite_on_background(plate, empty)


# ── generate_detector_dataset ─────────────────────────────────────────────────

@pytest.mark.unit
class TestGenerateDetectorDataset:
    def test_creates_correct_number_of_files(self, tmp_path: Path, bg_dir: Path):
        """n=5 must produce 5 images and 5 label files."""
        out = tmp_path / "detector"
        generate_detector_dataset(n=5, output_dir=out, bg_dir=bg_dir)
        assert len(list((out / "images").glob("*.jpg"))) == 5
        assert len(list((out / "labels").glob("*.txt"))) == 5

    def test_label_is_yolo_format(self, tmp_path: Path, bg_dir: Path):
        """Each label file must be a single line with 5 space-separated values."""
        out = tmp_path / "detector_fmt"
        generate_detector_dataset(n=3, output_dir=out, bg_dir=bg_dir)
        for lbl_path in (out / "labels").glob("*.txt"):
            parts = lbl_path.read_text().strip().split()
            assert len(parts) == 5, f"Expected 5 values, got {len(parts)}: {parts}"
            assert parts[0] == "0", "Class index must be 0"
            cx, cy, w, h = (float(v) for v in parts[1:])
            assert 0.0 <= cx <= 1.0
            assert 0.0 <= cy <= 1.0
            assert 0.0 < w <= 1.0
            assert 0.0 < h <= 1.0

    def test_is_idempotent(self, tmp_path: Path, bg_dir: Path):
        """Re-running generate_detector_dataset must not raise."""
        out = tmp_path / "detector_idem"
        generate_detector_dataset(n=2, output_dir=out, bg_dir=bg_dir)
        generate_detector_dataset(n=2, output_dir=out, bg_dir=bg_dir)


# ── generate_recognizer_dataset ───────────────────────────────────────────────

@pytest.mark.unit
class TestGenerateRecognizerDataset:
    def test_creates_images_and_csv(self, tmp_path: Path):
        """n=5 must produce 5 PNG images and a labels.csv with 5 data rows."""
        out = tmp_path / "recognizer"
        generate_recognizer_dataset(n=5, output_dir=out)
        assert len(list((out / "images").glob("*.png"))) == 5
        with (out / "labels.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 5

    def test_csv_columns(self, tmp_path: Path):
        """labels.csv must contain filename, text, and country columns."""
        out = tmp_path / "recognizer_cols"
        generate_recognizer_dataset(n=2, output_dir=out)
        with (out / "labels.csv").open() as f:
            reader = csv.DictReader(f)
            assert {"filename", "text", "country"}.issubset(set(reader.fieldnames or []))

    def test_images_are_128x32(self, tmp_path: Path):
        """Every saved plate image must be exactly 128×32 pixels."""
        out = tmp_path / "recognizer_size"
        generate_recognizer_dataset(n=3, output_dir=out)
        for img_path in (out / "images").glob("*.png"):
            img = Image.open(img_path)
            assert img.size == (128, 32), f"Wrong size {img.size} for {img_path.name}"

    def test_images_are_grayscale(self, tmp_path: Path):
        """Recognizer images must be grayscale (mode 'L')."""
        out = tmp_path / "recognizer_gray"
        generate_recognizer_dataset(n=2, output_dir=out)
        for img_path in (out / "images").glob("*.png"):
            img = Image.open(img_path)
            assert img.mode == "L", f"Expected mode 'L', got {img.mode!r}"

    def test_is_idempotent(self, tmp_path: Path):
        """Re-running generate_recognizer_dataset must not raise."""
        out = tmp_path / "recognizer_idem"
        generate_recognizer_dataset(n=2, output_dir=out)
        generate_recognizer_dataset(n=2, output_dir=out)
