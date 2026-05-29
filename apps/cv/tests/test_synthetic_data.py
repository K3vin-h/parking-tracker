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

import numpy as np
import pytest
from PIL import Image

from apps.cv.training import synthetic_data
from apps.cv.training import _image_io
from apps.cv.training.synthetic_data import (
    PLATE_SIZE,
    _seed_rng,
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

    @pytest.mark.parametrize("text,country", [
        ("ABC 1234", "US"),
        ("A1B 2C3", "CA"),
        ("X", "US"),
    ])
    def test_text_is_horizontally_centered(self, text: str, country: str):
        """
        Inked text must be horizontally centred within ±8 px of the plate.

        Regression guard for the textbbox origin-offset bug: TrueType bboxes
        carry a non-zero (left, top) bearing, and the rendering code must
        subtract that bearing back out when computing the draw origin or
        every plate ends up off-centre.
        """
        img = render_plate_image(text, country)
        gray = np.array(img.convert("L"))
        h, w = gray.shape
        # Exclude the 4-px black border so it doesn't pollute the column scan
        inner = gray[6:-6, 6:-6]
        # Columns containing any "dark" pixel (text)
        text_cols = np.where(inner.min(axis=0) < 80)[0]
        assert text_cols.size > 0, "No text pixels detected — render failed"
        centroid = (int(text_cols.min()) + int(text_cols.max())) / 2 + 6
        img_centre = w / 2
        assert abs(centroid - img_centre) <= 8, (
            f"Text centroid {centroid:.1f} is more than 8 px from image "
            f"centre {img_centre:.1f} (text={text!r}, country={country!r})"
        )


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

    def test_corrupt_background_raises_oserror(self, tmp_path: Path):
        """
        Corrupt background bytes must surface as OSError, not FileNotFoundError.

        Regression guard for the bare ``except Exception`` that previously
        masked decode failures behind a misleading FileNotFoundError.
        """
        bg_dir = tmp_path / "corrupt_bg"
        bg_dir.mkdir()
        (bg_dir / "broken.jpg").write_bytes(b"This is plainly not a JPEG header.")
        plate = render_plate_image("BAD 000", "US")
        with pytest.raises(OSError) as excinfo:
            composite_on_background(plate, bg_dir)
        # FileNotFoundError is a subclass of OSError — exclude it explicitly so
        # the test fails if the bare-except regression returns.
        assert not isinstance(excinfo.value, FileNotFoundError)

    def test_seeded_runs_are_deterministic(self, tmp_path: Path):
        """
        Same seed + same backgrounds must produce byte-identical composites.

        Regression guard for ``_collect_bg_files`` ordering: ``Path.iterdir``
        returns files in filesystem-dependent order, so without an explicit
        sort the seeded ``random.choice`` would still pick a non-reproducible
        background across machines.
        """
        bg_dir = tmp_path / "two_bgs"
        bg_dir.mkdir()
        Image.new("RGB", (640, 480), (255, 0, 0)).save(bg_dir / "a_red.jpg")
        Image.new("RGB", (640, 480), (0, 0, 255)).save(bg_dir / "b_blue.jpg")
        plate = render_plate_image("DET 111", "US")

        _seed_rng(42)
        composite_a, bbox_a = composite_on_background(plate, bg_dir)
        _seed_rng(42)
        composite_b, bbox_b = composite_on_background(plate, bg_dir)

        assert bbox_a == bbox_b
        # tobytes() compares the raw pixel buffer without going through the
        # deprecated getdata() API; both must match exactly under a fixed seed.
        assert composite_a.tobytes() == composite_b.tobytes()

    def test_oversized_background_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """
        Backgrounds exceeding the pixel cap must be rejected with ValueError.

        Lower the cap to a tiny value so we don't have to write a 12+ MP image
        to disk just to verify the bomb-protection wiring. With the cap at
        10_000 pixels a 200×200 (=40_000 px) image is over-budget, Pillow's
        DecompressionBombWarning fires, and ``safe_open_image`` converts it
        to a ValueError that bubbles up through composite_on_background.
        """
        monkeypatch.setattr(_image_io, "MAX_IMAGE_PIXELS", 10_000)
        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10_000)

        bg_dir = tmp_path / "bomb_bg"
        bg_dir.mkdir()
        Image.new("RGB", (200, 200), (50, 50, 50)).save(bg_dir / "big.jpg")
        plate = render_plate_image("BOM 999", "US")
        with pytest.raises(ValueError):
            composite_on_background(plate, bg_dir)

    def test_invalid_target_size_raises_value_error(self, bg_dir: Path):
        """target_size beyond the per-dim cap must raise ValueError."""
        plate = render_plate_image("SIZ 000", "US")
        with pytest.raises(ValueError, match="target_size"):
            composite_on_background(plate, bg_dir, target_size=(8192, 8192))

    def test_rotated_bbox_matches_visible_plate_pixels(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """
        BBox origin must account for transparent padding added by rotate(expand=True).

        Pillow expands the rotated canvas and offsets the visible plate within it.
        The returned bbox should describe the visible pixels, not the padded canvas.
        """
        bg_colour = np.array([100, 149, 237], dtype=np.uint8)
        bg_dir = tmp_path / "lossless_bg"
        bg_dir.mkdir()
        Image.new("RGB", (640, 480), tuple(int(v) for v in bg_colour)).save(
            bg_dir / "bg.png"
        )

        class FixedRandom:
            def choice(self, values):
                return values[0]

            def uniform(self, start, end):
                if (start, end) == (0.15, 0.40):
                    return 0.30
                if (start, end) == (-15, 15):
                    return 15
                raise AssertionError(f"Unexpected uniform range: {(start, end)}")

            def randint(self, start, end):
                return 25 if start == 0 else end

        monkeypatch.setattr(synthetic_data, "_rng", FixedRandom())
        plate = Image.new("RGBA", PLATE_SIZE, (255, 0, 0, 255))

        composite, (x, y, w, h) = composite_on_background(plate, bg_dir)
        pixels = np.array(composite)
        visible = np.any(pixels != bg_colour, axis=2)
        ys, xs = np.where(visible)

        assert (x, y) == (int(xs.min()), int(ys.min()))
        assert (w, h) == (
            int(xs.max() - xs.min() + 1),
            int(ys.max() - ys.min() + 1),
        )


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

    def test_rerun_with_smaller_n_clears_orphans(
        self, tmp_path: Path, bg_dir: Path
    ):
        """
        Re-running with a smaller n must leave exactly n files behind.

        Without orphan cleanup the second run inherits the first run's images
        and labels, which silently inflates the dataset that PlateDetectorDataset
        would later glob.
        """
        out = tmp_path / "detector_shrink"
        generate_detector_dataset(n=5, output_dir=out, bg_dir=bg_dir)
        generate_detector_dataset(n=2, output_dir=out, bg_dir=bg_dir)
        assert len(list((out / "images").glob("*.jpg"))) == 2
        assert len(list((out / "labels").glob("*.txt"))) == 2

    def test_seed_makes_output_reproducible(self, tmp_path: Path, bg_dir: Path):
        """Same seed must produce byte-identical labels across runs."""
        out_a = tmp_path / "det_seed_a"
        out_b = tmp_path / "det_seed_b"
        generate_detector_dataset(n=3, output_dir=out_a, bg_dir=bg_dir, seed=7)
        generate_detector_dataset(n=3, output_dir=out_b, bg_dir=bg_dir, seed=7)
        for i in range(3):
            label_a = (out_a / "labels" / f"{i:06d}.txt").read_text()
            label_b = (out_b / "labels" / f"{i:06d}.txt").read_text()
            assert label_a == label_b, f"Sample {i} differs between seeded runs"

    def test_invalid_n_raises_value_error(self, tmp_path: Path, bg_dir: Path):
        """n<=0 and n>1_000_000 must raise ValueError before any I/O."""
        out = tmp_path / "det_invalid"
        with pytest.raises(ValueError):
            generate_detector_dataset(n=0, output_dir=out, bg_dir=bg_dir)
        with pytest.raises(ValueError):
            generate_detector_dataset(n=2_000_000, output_dir=out, bg_dir=bg_dir)


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

    def test_rerun_with_smaller_n_clears_orphans(self, tmp_path: Path):
        """Re-running with a smaller n must leave exactly n PNG files behind."""
        out = tmp_path / "recognizer_shrink"
        generate_recognizer_dataset(n=5, output_dir=out)
        generate_recognizer_dataset(n=2, output_dir=out)
        assert len(list((out / "images").glob("*.png"))) == 2
        with (out / "labels.csv").open() as f:
            assert len(list(csv.DictReader(f))) == 2

    def test_seed_makes_output_reproducible(self, tmp_path: Path):
        """Same seed must produce identical CSV contents across runs."""
        out_a = tmp_path / "rec_seed_a"
        out_b = tmp_path / "rec_seed_b"
        generate_recognizer_dataset(n=3, output_dir=out_a, seed=7)
        generate_recognizer_dataset(n=3, output_dir=out_b, seed=7)
        assert (out_a / "labels.csv").read_text() == (out_b / "labels.csv").read_text()

    def test_invalid_n_raises_value_error(self, tmp_path: Path):
        """n<=0 and n>1_000_000 must raise ValueError before any I/O."""
        out = tmp_path / "rec_invalid"
        with pytest.raises(ValueError):
            generate_recognizer_dataset(n=0, output_dir=out)
        with pytest.raises(ValueError):
            generate_recognizer_dataset(n=2_000_000, output_dir=out)
