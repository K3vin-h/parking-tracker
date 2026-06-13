"""
Unit tests for PlateRecognitionPipeline.

All tests mock the model I/O — no trained weights or real images needed.
One integration test is skipped unless both weight files are present on disk.

Fixtures
────────
No external files required.  load_image is mocked to return a zero-value
uint8 array; the rest of the preprocessing chain (bgr_to_rgb, resize,
normalize, crop) runs with real numpy/cv2 operations on that array so we
exercise the full data-flow without a real image on disk.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from django.test import override_settings

import apps.cv.pipeline as pipeline_module
from apps.cv.pipeline import LOW_CONFIDENCE_THRESHOLD, PlateRecognitionPipeline, get_pipeline
from apps.cv.training.dataset import BLANK_IDX

# ── Constants ─────────────────────────────────────────────────────────────────

_WEIGHTS_DIR = Path(__file__).parents[3] / "apps" / "cv" / "weights"
_DETECTOR_PATH = str(_WEIGHTS_DIR / "detector.pth")
_RECOGNIZER_PATH = str(_WEIGHTS_DIR / "recognizer.pth")
_WEIGHTS_PRESENT = Path(_DETECTOR_PATH).exists() and Path(_RECOGNIZER_PATH).exists()

# Valid synthetic bbox: cx=0.5, cy=0.5, w=0.4, h=0.2.
# After YOLO→top-left conversion: x=0.3, y=0.4, w=0.4, h=0.2.
# On 640×480: x1=192, y1=192, x2=448, y2=288 — well within bounds.
_VALID_BBOX = [0.5, 0.5, 0.4, 0.2]  # [cx, cy, w, h]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_pipeline() -> PlateRecognitionPipeline:
    """
    Build a pipeline with all model I/O mocked.

    Patches os.path.isfile, torch.load, and both model classes so __init__
    succeeds without any weight files on disk.  The pipeline's .detector and
    .recognizer attributes are MagicMock instances whose methods (predict,
    decode_predictions) can be configured per-test.
    """
    with (
        patch("apps.cv.pipeline.os.path.isfile", return_value=True),
        patch("apps.cv.pipeline.torch.load", return_value={}),
        patch("apps.cv.pipeline.PlateDetectorCNN"),
        patch("apps.cv.pipeline.PlateRecognizerCRNN"),
    ):
        return PlateRecognitionPipeline("det.pth", "rec.pth", device=torch.device("cpu"))


def _run_process(
    pipeline: PlateRecognitionPipeline,
    bbox: list[float],
    log_probs: torch.Tensor,
    plate_text: str = "ABC123",
    image_shape: tuple[int, int, int] = (480, 640, 3),
) -> dict:
    """
    Run pipeline.process() with mocked load_image and model outputs.

    Passes a zero-value uint8 array as the image so the preprocessing chain
    runs normally without a real file on disk.
    """
    pipeline.detector.predict.return_value = torch.tensor([bbox])  # (1, 4)
    pipeline.recognizer.predict.return_value = log_probs
    pipeline.recognizer.decode_predictions.return_value = [plate_text]

    with patch(
        "apps.cv.pipeline.load_image",
        return_value=np.zeros(image_shape, dtype=np.uint8),
    ):
        return pipeline.process("media/fake.jpg")


# ── Init tests ────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestPlateRecognitionPipelineInit:
    """Tests for __init__ weight loading and error handling."""

    def test_raises_if_detector_weights_missing(self) -> None:
        """FileNotFoundError when the detector .pth file does not exist."""
        with pytest.raises(FileNotFoundError, match="Detector model weights"):
            PlateRecognitionPipeline(
                detector_path="/nonexistent/detector.pth",
                recognizer_path="/nonexistent/recognizer.pth",
            )

    def test_raises_if_recognizer_weights_missing(self, tmp_path: Path) -> None:
        """FileNotFoundError when the recognizer .pth file does not exist."""
        det_path = str(tmp_path / "detector.pth")
        Path(det_path).touch()  # detector exists; recognizer does not

        with (
            patch("apps.cv.pipeline.torch.load", return_value={}),
            patch("apps.cv.pipeline.PlateDetectorCNN"),
            pytest.raises(FileNotFoundError, match="Recognizer model weights"),
        ):
            PlateRecognitionPipeline(
                detector_path=det_path,
                recognizer_path="/nonexistent/recognizer.pth",
            )

    def test_error_message_excludes_path(self) -> None:
        """
        The FileNotFoundError message must NOT name the missing file.

        WHY: the raised text may be serialized into an API response by a
        future view's error handler — embedding the filesystem path there
        would disclose server layout (CWE-209).  The full path is logged
        server-side instead.
        """
        with pytest.raises(FileNotFoundError) as exc_info:
            PlateRecognitionPipeline(
                detector_path="/no/such/detector.pth",
                recognizer_path="/no/such/recognizer.pth",
            )
        assert "/no/such/detector.pth" not in str(exc_info.value)

    def test_corrupt_weights_raise_clean_runtime_error(self, tmp_path: Path) -> None:
        """
        A weights file that exists but cannot be loaded raises RuntimeError
        with a generic message (no filesystem path leaked).
        """
        det_path = tmp_path / "detector.pth"
        det_path.write_bytes(b"not a real torch checkpoint")
        rec_path = tmp_path / "recognizer.pth"
        rec_path.write_bytes(b"also not a checkpoint")

        with pytest.raises(RuntimeError, match="detector model weights") as exc_info:
            PlateRecognitionPipeline(
                detector_path=str(det_path),
                recognizer_path=str(rec_path),
            )
        assert str(det_path) not in str(exc_info.value)

    def test_both_models_set_to_eval_mode(self) -> None:
        """detector.eval() and recognizer.eval() are called during init."""
        pipeline = _make_pipeline()
        pipeline.detector.eval.assert_called()
        pipeline.recognizer.eval.assert_called()

    def test_both_models_moved_to_device(self) -> None:
        """detector.to(device) and recognizer.to(device) are called during init."""
        pipeline = _make_pipeline()
        pipeline.detector.to.assert_called_with(torch.device("cpu"))
        pipeline.recognizer.to.assert_called_with(torch.device("cpu"))


# ── Process tests ─────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestPlateRecognitionPipelineProcess:
    """Tests for the process() pipeline method."""

    def test_process_returns_required_keys(self) -> None:
        """Return dict must contain all four expected keys."""
        pipeline = _make_pipeline()
        result = _run_process(pipeline, _VALID_BBOX, torch.zeros(16, 1, 37))
        assert set(result.keys()) == {"plate_text", "confidence", "bounding_box", "is_low_confidence"}

    def test_plate_text_matches_model_output(self) -> None:
        """plate_text is taken from decode_predictions()[0]."""
        pipeline = _make_pipeline()
        result = _run_process(pipeline, _VALID_BBOX, torch.zeros(16, 1, 37), plate_text="XY1234")
        assert result["plate_text"] == "XY1234"

    def test_confidence_is_float_between_0_and_1(self) -> None:
        """confidence is a float in [0.0, 1.0]."""
        pipeline = _make_pipeline()
        result = _run_process(pipeline, _VALID_BBOX, torch.zeros(16, 1, 37))
        conf = result["confidence"]
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    def test_bounding_box_is_top_left_format(self) -> None:
        """
        bounding_box is [x, y, w, h] top-left format, not YOLO center format.

        Detector outputs [cx=0.5, cy=0.5, w=0.4, h=0.2].
        Expected top-left: [x=0.3, y=0.4, w=0.4, h=0.2].
        """
        pipeline = _make_pipeline()
        result = _run_process(pipeline, [0.5, 0.5, 0.4, 0.2], torch.zeros(16, 1, 37))
        x, y, w, h = result["bounding_box"]
        assert abs(x - 0.3) < 1e-5
        assert abs(y - 0.4) < 1e-5
        assert abs(w - 0.4) < 1e-5
        assert abs(h - 0.2) < 1e-5

    def test_bounding_box_unletterboxes_to_original_image(self) -> None:
        """
        Returned bounding_box is normalized to the original upload, not padding.

        A 1920×1080 frame letterboxes into 640×480 as 640×360 with 60 px top
        and bottom padding.  Detector-space y=0.375 therefore maps to original
        y=(180 - 60) / 360 = 0.333..., while x is unchanged because there is no
        horizontal padding.
        """
        pipeline = _make_pipeline()
        result = _run_process(
            pipeline,
            [0.5, 0.5, 0.25, 0.25],
            torch.zeros(16, 1, 37),
            image_shape=(1080, 1920, 3),
        )
        x, y, w, h = result["bounding_box"]
        assert abs(x - 0.375) < 1e-5
        assert abs(y - (1 / 3)) < 1e-5
        assert abs(w - 0.25) < 1e-5
        assert abs(h - (1 / 3)) < 1e-5

    def test_success_log_does_not_include_plate_text(self) -> None:
        """Successful inference logs confidence metadata without raw plate text."""
        pipeline = _make_pipeline()
        with patch("apps.cv.pipeline.logger.debug") as debug_log:
            _run_process(pipeline, _VALID_BBOX, torch.zeros(16, 1, 37), plate_text="SECRET123")

        logged_args = " ".join(str(arg) for call in debug_log.call_args_list for arg in call.args)
        assert "SECRET123" not in logged_args
        assert "Plate recognition complete" in logged_args

    def test_is_low_confidence_false_when_decisive_non_blank(self) -> None:
        """
        is_low_confidence is False when non-blank time-steps have high confidence.

        Construct log-probs where all time-steps have argmax on a non-blank class
        (index 1) with near-certain probability.  The blank-filtered mean is ~1.0.
        """
        # All steps strongly predict class 1 (non-blank).
        log_probs = torch.full((16, 1, 37), -100.0)
        log_probs[:, :, 1] = 0.0  # class 1 dominates every step
        pipeline = _make_pipeline()
        result = _run_process(pipeline, _VALID_BBOX, log_probs)
        assert result["is_low_confidence"] is False

    def test_is_low_confidence_true_when_uniform_distribution(self) -> None:
        """
        is_low_confidence is True when uniform probs (argmax = BLANK_IDX = 0).

        Uniform log_softmax → argmax = 0 (blank) on all steps → fallback to
        full-mean confidence ≈ 1/37 ≈ 0.027 < LOW_CONFIDENCE_THRESHOLD (0.6).
        """
        uniform_log_probs = F.log_softmax(torch.zeros(16, 1, 37), dim=-1)
        pipeline = _make_pipeline()
        result = _run_process(pipeline, _VALID_BBOX, uniform_log_probs)
        assert result["is_low_confidence"] is True

    def test_confidence_excludes_blank_dominated_steps(self) -> None:
        """
        Blank-dominated time-steps are excluded from the confidence mean.

        Steps 0–7: argmax = BLANK_IDX (blank), high blank probability.
        Steps 8–15: argmax = class 1 (non-blank), high character probability.
        Confidence should be close to 1.0 (only character steps counted),
        NOT deflated by the blank steps.
        """
        log_probs = torch.full((16, 1, 37), -100.0)
        log_probs[:8, :, BLANK_IDX] = 0.0    # steps 0–7: blank
        log_probs[8:, :, 1] = 0.0            # steps 8–15: class 1 (character)
        pipeline = _make_pipeline()
        result = _run_process(pipeline, _VALID_BBOX, log_probs)
        assert result["confidence"] > 0.9
        assert result["is_low_confidence"] is False

    def test_small_bbox_width_returns_empty_plate(self) -> None:
        """
        Tiny bbox width → pipeline returns empty plate_text with is_low_confidence=True.

        Simulates a frame where no plate was found.
        """
        pipeline = _make_pipeline()
        pipeline.detector.predict.return_value = torch.tensor([[0.5, 0.5, 0.01, 0.1]])
        with patch(
            "apps.cv.pipeline.load_image",
            return_value=np.zeros((480, 640, 3), dtype=np.uint8),
        ):
            result = pipeline.process("media/fake.jpg")

        assert result["plate_text"] == ""
        assert result["confidence"] == 0.0
        assert result["is_low_confidence"] is True

    def test_small_bbox_height_returns_empty_plate(self) -> None:
        """Tiny bbox height also triggers the no-plate early return."""
        pipeline = _make_pipeline()
        pipeline.detector.predict.return_value = torch.tensor([[0.5, 0.5, 0.3, 0.01]])
        with patch(
            "apps.cv.pipeline.load_image",
            return_value=np.zeros((480, 640, 3), dtype=np.uint8),
        ):
            result = pipeline.process("media/fake.jpg")

        assert result["plate_text"] == ""
        assert result["is_low_confidence"] is True

    def test_degenerate_crop_bbox_returns_empty_plate(self) -> None:
        """
        ValueError from crop_plate_region is caught and returns empty plate.

        A bbox may pass the _MIN_BBOX_SIZE check but produce a zero-area crop
        after integer conversion + clamping (e.g. plate at the image edge).
        The pipeline must not let this ValueError propagate as a 500 error.
        """
        pipeline = _make_pipeline()
        pipeline.detector.predict.return_value = torch.tensor([_VALID_BBOX])
        with (
            patch("apps.cv.pipeline.load_image", return_value=np.zeros((480, 640, 3), dtype=np.uint8)),
            patch("apps.cv.pipeline.crop_plate_region", side_effect=ValueError("degenerate crop")),
        ):
            result = pipeline.process("media/fake.jpg")

        assert result["plate_text"] == ""
        assert result["confidence"] == 0.0
        assert result["is_low_confidence"] is True

    def test_process_propagates_corrupt_image_error(self) -> None:
        """FileNotFoundError from load_image propagates to the caller."""
        pipeline = _make_pipeline()
        with (
            patch("apps.cv.pipeline.load_image", side_effect=FileNotFoundError("bad image")),
            pytest.raises(FileNotFoundError, match="bad image"),
        ):
            pipeline.process("media/corrupt.jpg")

    def test_confidence_constant_threshold(self) -> None:
        """LOW_CONFIDENCE_THRESHOLD is 0.6 — callers can rely on this value."""
        assert LOW_CONFIDENCE_THRESHOLD == 0.6


# ── Singleton tests ───────────────────────────────────────────────────────────

@pytest.mark.unit
class TestGetPipeline:
    """Tests for the get_pipeline() module-level singleton."""

    def test_returns_same_instance_on_repeated_calls(self, monkeypatch) -> None:
        """get_pipeline() returns the same object across multiple calls."""
        monkeypatch.setattr(pipeline_module, "_instance", None)
        with (
            patch("apps.cv.pipeline.os.path.isfile", return_value=True),
            patch("apps.cv.pipeline.torch.load", return_value={}),
            patch("apps.cv.pipeline.PlateDetectorCNN"),
            patch("apps.cv.pipeline.PlateRecognizerCRNN"),
        ):
            p1 = get_pipeline("det.pth", "rec.pth")
            p2 = get_pipeline("det.pth", "rec.pth")

        assert p1 is p2

    def test_new_instance_after_reset(self, monkeypatch) -> None:
        """Resetting _instance to None causes get_pipeline() to create a new one."""
        monkeypatch.setattr(pipeline_module, "_instance", None)
        with (
            patch("apps.cv.pipeline.os.path.isfile", return_value=True),
            patch("apps.cv.pipeline.torch.load", return_value={}),
            patch("apps.cv.pipeline.PlateDetectorCNN"),
            patch("apps.cv.pipeline.PlateRecognizerCRNN"),
        ):
            p1 = get_pipeline("det.pth", "rec.pth")
            monkeypatch.setattr(pipeline_module, "_instance", None)
            p2 = get_pipeline("det.pth", "rec.pth")

        assert p1 is not p2


# ── Integration test ──────────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.skipif(not _WEIGHTS_PRESENT, reason="Trained weight files not present")
class TestPipelineIntegration:
    """End-to-end test using real trained model weights and a synthetic image."""

    def test_integration_end_to_end(self, tmp_path: Path) -> None:
        """
        Smoke test: process() returns a non-empty dict with a valid plate string.

        Uses a synthetic plate image generated from the recognizer dataset
        utilities so no real camera image is needed.  The plate text is not
        asserted — only that the pipeline runs without errors and returns
        the expected dict shape.

        MEDIA_ROOT is overridden to tmp_path so load_image's path-traversal
        guard accepts files in the pytest temporary directory.
        """
        from apps.cv.training.synthetic_data import generate_recognizer_dataset

        generate_recognizer_dataset(n=1, output_dir=str(tmp_path))

        labels_csv = tmp_path / "labels.csv"
        first_line = labels_csv.read_text().strip().splitlines()[1]  # skip header
        filename, _label = first_line.split(",", 1)
        image_path = str(tmp_path / "images" / filename)

        with override_settings(MEDIA_ROOT=str(tmp_path)):
            pipeline = PlateRecognitionPipeline(
                detector_path=_DETECTOR_PATH,
                recognizer_path=_RECOGNIZER_PATH,
            )
            result = pipeline.process(image_path)

        assert set(result.keys()) == {"plate_text", "confidence", "bounding_box", "is_low_confidence"}
        assert isinstance(result["plate_text"], str)
        assert 0.0 <= result["confidence"] <= 1.0
        assert len(result["bounding_box"]) == 4
        assert isinstance(result["is_low_confidence"], bool)
