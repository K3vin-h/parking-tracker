"""
Unit tests for apps/cv/preprocessing.py.

All tests use synthetic numpy arrays or temporary files — no real images
required. Tests are organized by function and follow the Arrange-Act-Assert
pattern throughout.
"""

import os

import numpy as np
import pytest
import torch
import cv2
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from PIL import Image

from apps.cv.preprocessing import (
    load_image,
    bgr_to_rgb,
    resize_for_detector,
    normalize_pixels,
    to_tensor,
    crop_plate_region,
    prepare_for_recognizer,
    _assert_safe_path,
)


def _must_not_decode(*_args, **_kwargs):
    """
    Sentinel patched onto cv2.imdecode to assert the decode path is unreached.

    Used by tests that expect pre-decode validation (path guard, header check,
    pixel-count guard, decompression-bomb guard) to short-circuit before any
    bytes are decoded. Failing loudly here makes a regression — where a
    malicious image slips past validation and reaches the decoder — impossible
    to miss.
    """
    raise AssertionError("cv2.imdecode must not be reached after pre-decode validation")


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_bgr_image(h: int = 100, w: int = 100, fill: int = 128) -> np.ndarray:
    """Return a solid-color uint8 BGR image."""
    return np.full((h, w, 3), fill, dtype=np.uint8)


def make_rgb_image(h: int = 100, w: int = 100, fill: int = 128) -> np.ndarray:
    """Return a solid-color uint8 RGB image."""
    return np.full((h, w, 3), fill, dtype=np.uint8)


class FakeImageHeader:
    """Context manager that mimics Pillow's header-only image object."""

    def __init__(self, size: tuple[int, int], fmt: str = "JPEG"):
        self.size = size
        self.format = fmt

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


@pytest.fixture()
def mock_media_root(tmp_path, monkeypatch):
    """
    Point settings.MEDIA_ROOT at tmp_path for the duration of a test.

    _assert_safe_path checks that image paths are under MEDIA_ROOT.
    Tests that call load_image() with real files need MEDIA_ROOT to match
    tmp_path so the guard passes without disabling security logic.
    """
    monkeypatch.setattr(settings, "MEDIA_ROOT", str(tmp_path))
    return tmp_path


# ── _assert_safe_path ──────────────────────────────────────────────────────────

@pytest.mark.unit
def test_assert_safe_path_rejects_traversal(tmp_path, monkeypatch):
    """Paths outside MEDIA_ROOT must raise ValueError."""
    monkeypatch.setattr(settings, "MEDIA_ROOT", str(tmp_path / "media"))
    with pytest.raises(ValueError, match="outside the permitted"):
        _assert_safe_path("/etc/passwd")


@pytest.mark.unit
def test_assert_safe_path_rejects_dotdot(tmp_path, monkeypatch):
    """Path traversal via '..' must be caught after realpath resolution."""
    media = tmp_path / "media"
    media.mkdir()
    monkeypatch.setattr(settings, "MEDIA_ROOT", str(media))
    with pytest.raises(ValueError, match="outside the permitted"):
        _assert_safe_path(str(media / ".." / "secret.jpg"))


@pytest.mark.unit
def test_assert_safe_path_allows_valid_path(tmp_path, monkeypatch):
    """Paths inside MEDIA_ROOT must not raise."""
    monkeypatch.setattr(settings, "MEDIA_ROOT", str(tmp_path))
    _assert_safe_path(str(tmp_path / "plates" / "img.jpg"))  # must not raise


@pytest.mark.unit
def test_assert_safe_path_rejects_empty_media_root(monkeypatch):
    """An empty MEDIA_ROOT must raise ImproperlyConfigured, not silently allow CWD."""
    monkeypatch.setattr(settings, "MEDIA_ROOT", "")
    with pytest.raises(ImproperlyConfigured):
        _assert_safe_path("/anywhere/file.jpg")


@pytest.mark.unit
def test_assert_safe_path_rejects_cwd_media_root(monkeypatch):
    """MEDIA_ROOT equal to the project working directory must raise."""
    monkeypatch.setattr(settings, "MEDIA_ROOT", os.getcwd())
    with pytest.raises(ImproperlyConfigured):
        _assert_safe_path(os.path.join(os.getcwd(), "file.jpg"))


# ── load_image ─────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_load_image_raises_file_not_found(mock_media_root):
    """Missing file inside MEDIA_ROOT must raise FileNotFoundError."""
    missing = str(mock_media_root / "nonexistent.jpg")
    with pytest.raises(FileNotFoundError, match="Could not load the image"):
        load_image(missing)


@pytest.mark.unit
def test_load_image_rejects_path_outside_media_root(tmp_path, monkeypatch):
    """Paths outside MEDIA_ROOT must be blocked by the path traversal guard."""
    monkeypatch.setattr(settings, "MEDIA_ROOT", str(tmp_path / "media"))
    with pytest.raises(ValueError, match="outside the permitted"):
        load_image("/etc/passwd")


@pytest.mark.unit
def test_load_image_returns_ndarray(mock_media_root):
    """A valid PNG written to MEDIA_ROOT must load as a numpy array."""
    img = make_bgr_image(50, 80)
    img_path = str(mock_media_root / "test.png")
    cv2.imwrite(img_path, img)

    result = load_image(img_path)

    assert isinstance(result, np.ndarray)
    assert result.ndim == 3
    assert result.shape[2] == 3


@pytest.mark.unit
def test_load_image_shape_matches_written(mock_media_root):
    """Loaded image shape must match the dimensions written to disk."""
    img = make_bgr_image(h=60, w=120)
    img_path = str(mock_media_root / "shape_test.png")
    cv2.imwrite(img_path, img)

    result = load_image(img_path)

    assert result.shape == (60, 120, 3)


@pytest.mark.unit
def test_load_image_dtype_is_uint8(mock_media_root):
    """cv2.imread returns uint8 arrays by default."""
    img = make_bgr_image(10, 10)
    img_path = str(mock_media_root / "dtype_test.png")
    cv2.imwrite(img_path, img)

    result = load_image(img_path)

    assert result.dtype == np.uint8


@pytest.mark.unit
def test_load_image_error_message_does_not_leak_path(mock_media_root):
    """FileNotFoundError message must not contain the file path."""
    missing = str(mock_media_root / "secret_internal_path.jpg")
    with pytest.raises(FileNotFoundError) as exc_info:
        load_image(missing)
    assert "secret_internal_path" not in str(exc_info.value)


@pytest.mark.unit
def test_load_image_rejects_disallowed_format(mock_media_root, monkeypatch):
    """Files with an unsupported format (e.g. EPS) must be rejected."""
    img_path = str(mock_media_root / "shell.eps")
    (mock_media_root / "shell.eps").write_bytes(b"fake eps bytes")

    monkeypatch.setattr(
        "apps.cv.preprocessing.Image.open",
        lambda p: FakeImageHeader((100, 100), fmt="EPS"),
    )

    with pytest.raises(ValueError, match="Unsupported image format"):
        load_image(img_path)


@pytest.mark.unit
def test_load_image_rejects_oversized_image_before_decoding(mock_media_root, monkeypatch):
    """Images exceeding the 12 MP pixel cap must fail before cv2.imdecode runs."""
    img_path = str(mock_media_root / "huge.png")
    (mock_media_root / "huge.png").write_bytes(b"fake image bytes")

    monkeypatch.setattr(
        "apps.cv.preprocessing.Image.open",
        lambda src: FakeImageHeader((4001, 3000)),
    )
    monkeypatch.setattr("apps.cv.preprocessing.cv2.imdecode", _must_not_decode)

    with pytest.raises(ValueError, match="exceed"):
        load_image(img_path)


@pytest.mark.unit
def test_load_image_rejects_large_compressed_file_before_header_parse(
    mock_media_root, monkeypatch
):
    """Compressed bytes above the cap must fail before Pillow or OpenCV parse them."""
    img_path = mock_media_root / "large.jpg"
    img_path.write_bytes(b"x" * 11)

    def fail_if_header_parsed(src):
        raise AssertionError("Image.open must not parse oversized compressed files")

    monkeypatch.setattr("apps.cv.preprocessing.MAX_IMAGE_BYTES", 10)
    monkeypatch.setattr("apps.cv.preprocessing.Image.open", fail_if_header_parsed)
    monkeypatch.setattr("apps.cv.preprocessing.cv2.imdecode", _must_not_decode)

    with pytest.raises(ValueError, match="compressed size"):
        load_image(str(img_path))


@pytest.mark.unit
def test_load_image_rejects_uninspectable_image(mock_media_root, monkeypatch):
    """Files whose format cannot be inspected must fail before cv2.imdecode."""
    img_path = str(mock_media_root / "bad.raw")
    (mock_media_root / "bad.raw").write_bytes(b"not an image")

    from PIL import UnidentifiedImageError as UIE

    def fake_open(src):
        raise UIE("cannot identify")

    monkeypatch.setattr("apps.cv.preprocessing.Image.open", fake_open)
    monkeypatch.setattr("apps.cv.preprocessing.cv2.imdecode", _must_not_decode)

    with pytest.raises(FileNotFoundError, match="Could not load the image"):
        load_image(img_path)


@pytest.mark.unit
def test_load_image_rejects_decompression_bomb(mock_media_root, monkeypatch):
    """Pillow's DecompressionBombError must be wrapped, not propagated raw."""
    img_path = str(mock_media_root / "bomb.png")
    (mock_media_root / "bomb.png").write_bytes(b"fake image bytes")

    def fake_open(src):
        raise Image.DecompressionBombError("simulated decompression bomb")

    monkeypatch.setattr("apps.cv.preprocessing.Image.open", fake_open)
    monkeypatch.setattr("apps.cv.preprocessing.cv2.imdecode", _must_not_decode)

    with pytest.raises(FileNotFoundError, match="Could not load the image"):
        load_image(img_path)


@pytest.mark.unit
def test_load_image_allows_image_at_pixel_limit(mock_media_root):
    """Images at exactly 12 MP (4000×3000) must be accepted."""
    img = np.zeros((3000, 4000, 3), dtype=np.uint8)
    img_path = str(mock_media_root / "max.png")
    cv2.imwrite(img_path, img)

    result = load_image(img_path)

    assert result.shape == (3000, 4000, 3)


# ── bgr_to_rgb ─────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_bgr_to_rgb_swaps_first_and_third_channels():
    """
    In a BGR image: channel 0=B, channel 1=G, channel 2=R.
    After conversion: channel 0=R, channel 1=G, channel 2=B.
    """
    img = np.zeros((5, 5, 3), dtype=np.uint8)
    img[:, :, 0] = 10   # B
    img[:, :, 1] = 20   # G
    img[:, :, 2] = 30   # R

    result = bgr_to_rgb(img)

    assert result[0, 0, 0] == 30  # R is now channel 0
    assert result[0, 0, 1] == 20  # G stays in the middle
    assert result[0, 0, 2] == 10  # B is now channel 2


@pytest.mark.unit
def test_bgr_to_rgb_preserves_shape():
    """Color conversion must not alter the image dimensions."""
    img = make_bgr_image(100, 200)
    result = bgr_to_rgb(img)
    assert result.shape == (100, 200, 3)


@pytest.mark.unit
def test_bgr_to_rgb_preserves_dtype():
    """Color conversion must not change the array dtype."""
    img = make_bgr_image()
    result = bgr_to_rgb(img)
    assert result.dtype == np.uint8


@pytest.mark.unit
def test_bgr_to_rgb_is_invertible():
    """Applying BGR→RGB twice must return to the original channel order."""
    img = make_bgr_image()
    img[:, :, 0] = 10
    img[:, :, 2] = 30

    round_tripped = bgr_to_rgb(bgr_to_rgb(img))

    np.testing.assert_array_equal(round_tripped, img)


# ── resize_for_detector ────────────────────────────────────────────────────────

@pytest.mark.unit
def test_resize_for_detector_default_output_shape():
    """Default resize must produce a 640×480 image (width=640, height=480)."""
    img = make_bgr_image(h=1080, w=1920)  # typical 1080p camera frame

    result = resize_for_detector(img)

    # numpy shape is (H, W, C) → (480, 640, 3)
    assert result.shape == (480, 640, 3)


@pytest.mark.unit
def test_resize_for_detector_letterboxes_widescreen_input():
    """16:9 camera frames must preserve aspect ratio with vertical padding."""
    img = make_bgr_image(h=1080, w=1920, fill=255)

    result = resize_for_detector(img)

    assert result.shape == (480, 640, 3)
    assert np.all(result[:60, :, :] == 0)
    assert np.all(result[60:420, :, :] == 255)
    assert np.all(result[420:, :, :] == 0)


@pytest.mark.unit
def test_resize_for_detector_custom_target():
    """Custom target size must be respected."""
    img = make_bgr_image(h=200, w=300)

    result = resize_for_detector(img, target=(160, 120))

    assert result.shape == (120, 160, 3)


@pytest.mark.unit
def test_resize_for_detector_handles_portrait_input():
    """Portrait-orientation images must resize to the correct landscape output."""
    img = make_bgr_image(h=800, w=600, fill=255)

    result = resize_for_detector(img, target=(640, 480))

    assert result.shape == (480, 640, 3)
    assert np.all(result[:, :140, :] == 0)
    assert np.all(result[:, 140:500, :] == 255)
    assert np.all(result[:, 500:, :] == 0)


@pytest.mark.unit
def test_resize_for_detector_handles_upscaling():
    """Images smaller than the target must be upscaled without error."""
    img = make_bgr_image(h=120, w=160)

    result = resize_for_detector(img, target=(640, 480))

    assert result.shape == (480, 640, 3)


@pytest.mark.unit
def test_resize_for_detector_preserves_channels():
    """Resize must not drop or add color channels."""
    img = make_bgr_image(h=200, w=200)
    result = resize_for_detector(img)
    assert result.shape[2] == 3


# ── normalize_pixels ───────────────────────────────────────────────────────────

@pytest.mark.unit
def test_normalize_pixels_dtype_is_float32():
    """Output must be float32 regardless of input dtype."""
    img = make_bgr_image()
    result = normalize_pixels(img)
    assert result.dtype == np.float32


@pytest.mark.unit
def test_normalize_pixels_max_value():
    """uint8 max (255) must map to exactly 1.0."""
    img = np.full((5, 5, 3), 255, dtype=np.uint8)
    result = normalize_pixels(img)
    assert np.allclose(result, 1.0)


@pytest.mark.unit
def test_normalize_pixels_min_value():
    """uint8 zero must map to exactly 0.0."""
    img = np.zeros((5, 5, 3), dtype=np.uint8)
    result = normalize_pixels(img)
    assert np.allclose(result, 0.0)


@pytest.mark.unit
def test_normalize_pixels_range():
    """All pixel values must be in [0.0, 1.0] for arbitrary uint8 input."""
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, size=(50, 50, 3), dtype=np.uint8)

    result = normalize_pixels(img)

    assert result.min() >= 0.0
    assert result.max() <= 1.0


@pytest.mark.unit
def test_normalize_pixels_preserves_shape():
    """Normalization must not change the array shape."""
    img = make_bgr_image(h=30, w=60)
    result = normalize_pixels(img)
    assert result.shape == img.shape


@pytest.mark.unit
def test_normalize_pixels_does_not_mutate_input():
    """normalize_pixels must not modify the original array."""
    img = np.full((5, 5, 3), 200, dtype=np.uint8)
    original_copy = img.copy()

    normalize_pixels(img)

    np.testing.assert_array_equal(img, original_copy)


@pytest.mark.unit
def test_normalize_pixels_rejects_float_input():
    """Passing a float32 array must raise TypeError (not silently produce near-zero values)."""
    img = np.zeros((5, 5, 3), dtype=np.float32)
    with pytest.raises(TypeError, match="uint8"):
        normalize_pixels(img)


# ── to_tensor ──────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_to_tensor_shape_hwc_to_chw():
    """HWC input (480, 640, 3) must become CHW output (3, 480, 640)."""
    img = np.zeros((480, 640, 3), dtype=np.float32)

    result = to_tensor(img)

    assert result.shape == torch.Size([3, 480, 640])


@pytest.mark.unit
def test_to_tensor_dtype_is_float32():
    """Tensor dtype must be float32."""
    img = np.zeros((10, 10, 3), dtype=np.float32)
    result = to_tensor(img)
    assert result.dtype == torch.float32


@pytest.mark.unit
def test_to_tensor_is_contiguous():
    """Result must be contiguous (permute produces a non-contiguous view)."""
    img = np.zeros((100, 100, 3), dtype=np.float32)
    result = to_tensor(img)
    assert result.is_contiguous()


@pytest.mark.unit
def test_to_tensor_values_preserved():
    """Pixel values must not change during axis reordering."""
    img = np.zeros((5, 5, 3), dtype=np.float32)
    img[:, :, 0] = 0.1  # channel 0
    img[:, :, 1] = 0.5  # channel 1
    img[:, :, 2] = 0.9  # channel 2

    result = to_tensor(img)

    assert torch.allclose(result[0], torch.full((5, 5), 0.1))
    assert torch.allclose(result[1], torch.full((5, 5), 0.5))
    assert torch.allclose(result[2], torch.full((5, 5), 0.9))


@pytest.mark.unit
def test_to_tensor_rejects_uint8_input():
    """Passing a uint8 array must raise TypeError, not silently produce a uint8 tensor."""
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(TypeError, match="float32"):
        to_tensor(img)


@pytest.mark.unit
def test_to_tensor_rejects_float64_input():
    """Even float64 must be rejected — only float32 is supported."""
    img = np.zeros((10, 10, 3), dtype=np.float64)
    with pytest.raises(TypeError, match="float32"):
        to_tensor(img)


# ── crop_plate_region ──────────────────────────────────────────────────────────

@pytest.mark.unit
def test_crop_plate_region_correct_shape():
    """
    bbox [0.25, 0.25, 0.5, 0.5] on a 100×100 image should produce a 50×50 crop.
    x_px=25, y_px=25, w_px=50, h_px=50 → image[25:75, 25:75] → (50, 50, 3).
    """
    img = make_rgb_image(h=100, w=100)
    bbox = [0.25, 0.25, 0.5, 0.5]

    result = crop_plate_region(img, bbox)

    assert result.shape == (50, 50, 3)


@pytest.mark.unit
def test_crop_plate_region_accepts_numpy_float_scalars():
    """Detector boxes may arrive as numpy float32 scalars from model output."""
    img = make_rgb_image(h=100, w=100)
    bbox = list(np.array([0.25, 0.25, 0.5, 0.5], dtype=np.float32))

    result = crop_plate_region(img, bbox)

    assert result.shape == (50, 50, 3)


@pytest.mark.unit
def test_crop_plate_region_full_image():
    """bbox [0, 0, 1, 1] must return the entire image."""
    img = make_rgb_image(h=100, w=200)
    bbox = [0.0, 0.0, 1.0, 1.0]

    result = crop_plate_region(img, bbox)

    assert result.shape == (100, 200, 3)


@pytest.mark.unit
def test_crop_plate_region_clamps_out_of_bounds():
    """
    A bbox that extends past the image edge must not raise an error.
    The crop is silently clamped to the image boundary.
    """
    img = make_rgb_image(h=100, w=100)
    bbox = [0.8, 0.8, 0.5, 0.5]  # right and bottom edges exceed 1.0

    result = crop_plate_region(img, bbox)  # must not raise

    assert result.ndim == 3
    assert result.shape[2] == 3


@pytest.mark.unit
def test_crop_plate_region_top_left_corner():
    """A small top-left crop must return pixels from the correct image region."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[0:10, 0:10] = [255, 0, 0]  # mark top-left 10×10 in red
    bbox = [0.0, 0.0, 0.1, 0.1]  # select that exact 10×10 region

    result = crop_plate_region(img, bbox)

    assert result.shape == (10, 10, 3)
    assert np.all(result[:, :, 0] == 255)  # R channel is 255


@pytest.mark.unit
def test_crop_plate_region_preserves_dtype():
    """Crop must not change the array dtype."""
    img = make_rgb_image()
    bbox = [0.1, 0.1, 0.5, 0.5]
    result = crop_plate_region(img, bbox)
    assert result.dtype == np.uint8


@pytest.mark.unit
def test_crop_plate_region_wrong_bbox_length():
    """bbox with wrong number of elements must raise ValueError."""
    img = make_rgb_image()
    with pytest.raises(ValueError, match="4 elements"):
        crop_plate_region(img, [0.1, 0.1, 0.5])


@pytest.mark.unit
def test_crop_plate_region_zero_width_bbox():
    """bbox with zero width must raise ValueError, not return a silent empty array."""
    img = make_rgb_image()
    with pytest.raises(ValueError, match="non-positive"):
        crop_plate_region(img, [0.1, 0.1, 0.0, 0.5])


@pytest.mark.unit
def test_crop_plate_region_degenerate_out_of_bounds():
    """A bbox entirely outside the image must raise ValueError."""
    img = make_rgb_image(h=100, w=100)
    # x=0.95, w=0.1 → x_px=95, x2=min(100,105)=100 → but x1=95 < x2=100 so this is OK
    # Let's test x=1.0 — starts at exact edge with some width
    with pytest.raises(ValueError, match="zero-size"):
        crop_plate_region(img, [1.0, 0.0, 0.5, 0.5])


# ── prepare_for_recognizer ─────────────────────────────────────────────────────

@pytest.mark.unit
def test_prepare_for_recognizer_output_shape():
    """Any input size must produce a (1, 32, 128) tensor."""
    crop = make_rgb_image(h=60, w=200)

    result = prepare_for_recognizer(crop)

    assert result.shape == torch.Size([1, 32, 128])


@pytest.mark.unit
def test_prepare_for_recognizer_dtype_is_float32():
    """Output tensor dtype must be float32."""
    crop = make_rgb_image(h=40, w=120)
    result = prepare_for_recognizer(crop)
    assert result.dtype == torch.float32


@pytest.mark.unit
def test_prepare_for_recognizer_value_range():
    """All values in the output tensor must be in [0.0, 1.0]."""
    crop = np.full((40, 120, 3), 200, dtype=np.uint8)

    result = prepare_for_recognizer(crop)

    assert result.min().item() >= 0.0
    assert result.max().item() <= 1.0


@pytest.mark.unit
def test_prepare_for_recognizer_is_contiguous():
    """Output tensor must be contiguous in memory."""
    crop = make_rgb_image(h=40, w=120)
    result = prepare_for_recognizer(crop)
    assert result.is_contiguous()


@pytest.mark.unit
def test_prepare_for_recognizer_small_input():
    """Input smaller than 128×32 (upscaling path) must work correctly."""
    crop = make_rgb_image(h=10, w=30)

    result = prepare_for_recognizer(crop)

    assert result.shape == torch.Size([1, 32, 128])


@pytest.mark.unit
def test_prepare_for_recognizer_large_input():
    """Input larger than 128×32 (downscaling path) must work correctly."""
    crop = make_rgb_image(h=200, w=600)

    result = prepare_for_recognizer(crop)

    assert result.shape == torch.Size([1, 32, 128])


@pytest.mark.unit
def test_prepare_for_recognizer_rejects_float_input():
    """Already-normalized float crop must raise TypeError, not silently re-normalize."""
    crop = np.zeros((40, 120, 3), dtype=np.float32)
    with pytest.raises(TypeError, match="uint8"):
        prepare_for_recognizer(crop)
