"""
Unit tests for apps/cv/training/evaluate_detector.py.

WHY stub the pipeline instead of loading real weights: this harness is meant to
run against arbitrary checkpoints on a developer's machine, so its own tests
must not depend on a trained detector.pth/recognizer.pth existing. A stub
pipeline with a controllable process() lets both the "everything matches" and
"nothing matches" paths be asserted deterministically.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

import apps.cv.pipeline as pipeline_module
import apps.cv.training.synthetic_data as synthetic_data_module
from apps.cv.training.evaluate_detector import _iou_xywh, evaluate

# A fixed composite: 640x480 image with the "plate" living at pixel
# box [100, 100, 50, 20] — everything downstream is deterministic from here.
_FIXED_BBOX_PX = [100, 100, 50, 20]
_FIXED_BBOX_NORM = [100 / 640, 100 / 480, 50 / 640, 20 / 480]
_PLATE_TEXT = "ABC123"


class _StubRecognizer:
    """Always decodes to a fixed string, regardless of the input tensor."""

    def __init__(self, text: str) -> None:
        self._text = text

    def predict(self, x):
        return MagicMock()

    def decode_predictions(self, log_probs):
        return [self._text]


class _StubPipeline:
    """Replaces PlateRecognitionPipeline: no weight files, fixed outputs."""

    def __init__(
        self, detector_path, recognizer_path, *, process_text: str, process_bbox
    ):
        self.device = "cpu"
        self.recognizer = _StubRecognizer(process_text)
        self._process_text = process_text
        self._process_bbox = process_bbox

    def process(self, image_path):
        return {
            "plate_text": self._process_text,
            "confidence": 0.9,
            "bounding_box": self._process_bbox,
            "is_low_confidence": False,
        }


def _patch_synthetic_data(monkeypatch, bg_dir: Path) -> None:
    """Fix generate_plate_text/render_plate_image/composite_on_background
    to a known plate text and a known, deterministic composite + bbox."""
    monkeypatch.setattr(
        synthetic_data_module,
        "generate_plate_text",
        lambda country: (_PLATE_TEXT, "fmt"),
    )
    monkeypatch.setattr(
        synthetic_data_module,
        "render_plate_image",
        lambda text, country: Image.new("RGBA", (50, 20), (255, 255, 255, 255)),
    )
    monkeypatch.setattr(
        synthetic_data_module,
        "composite_on_background",
        lambda plate_img, bg_dir_arg, target_size=(640, 480): (
            Image.new("RGB", (640, 480), (128, 128, 128)),
            list(_FIXED_BBOX_PX),
        ),
    )



def test_evaluate_perfect_pipeline_scores_100_percent(monkeypatch, tmp_path):
    _patch_synthetic_data(monkeypatch, tmp_path)
    monkeypatch.setattr(
        pipeline_module,
        "PlateRecognitionPipeline",
        lambda det, rec: _StubPipeline(
            det, rec, process_text=_PLATE_TEXT, process_bbox=list(_FIXED_BBOX_NORM)
        ),
    )

    results = evaluate(
        bg_dir=tmp_path,
        n=3,
        seed=1,
        detector_weights=Path("unused-detector.pth"),
        recognizer_weights=Path("unused-recognizer.pth"),
    )

    assert results["mean_iou"] == pytest.approx(1.0)
    assert results["end_to_end_accuracy"] == pytest.approx(1.0)
    assert results["oracle_bbox_accuracy"] == pytest.approx(1.0)



def test_evaluate_wrong_pipeline_scores_0_percent(monkeypatch, tmp_path):
    _patch_synthetic_data(monkeypatch, tmp_path)
    monkeypatch.setattr(
        pipeline_module,
        "PlateRecognitionPipeline",
        lambda det, rec: _StubPipeline(
            det, rec, process_text="ZZZ999", process_bbox=[0.0, 0.0, 0.01, 0.01]
        ),
    )

    results = evaluate(
        bg_dir=tmp_path,
        n=3,
        seed=1,
        detector_weights=Path("unused-detector.pth"),
        recognizer_weights=Path("unused-recognizer.pth"),
    )

    assert results["mean_iou"] < 0.1
    assert results["end_to_end_accuracy"] == pytest.approx(0.0)
    assert results["oracle_bbox_accuracy"] == pytest.approx(0.0)


class TestIouXywh:
    def test_perfect_overlap_is_one(self) -> None:
        box = [0.2, 0.3, 0.4, 0.1]
        assert _iou_xywh(box, box) == pytest.approx(1.0)

    def test_no_overlap_is_zero(self) -> None:
        assert _iou_xywh([0.0, 0.0, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1]) == 0.0

    def test_partial_overlap_known_value(self) -> None:
        # Two unit squares overlapping in a 0.5x0.5 region: intersection 0.25,
        # union = 1 + 1 - 0.25 = 1.75, IoU = 0.25 / 1.75.
        a = [0.0, 0.0, 1.0, 1.0]
        b = [0.5, 0.5, 1.0, 1.0]
        assert _iou_xywh(a, b) == pytest.approx(0.25 / 1.75)


@pytest.mark.parametrize("fail", [False, True])
def test_evaluation_preserves_existing_files_and_cleans_own_files(monkeypatch, tmp_path, settings, fail):
    settings.MEDIA_ROOT = tmp_path
    existing = tmp_path / "eval_detector_tmp"
    existing.mkdir()
    sentinel = existing / "keep.txt"
    sentinel.write_text("keep")
    _patch_synthetic_data(monkeypatch, tmp_path)
    pipeline = _StubPipeline(None, None, process_text=_PLATE_TEXT, process_bbox=_FIXED_BBOX_NORM)
    if fail:
        pipeline.process = MagicMock(side_effect=RuntimeError("test failure"))
    monkeypatch.setattr(pipeline_module, "PlateRecognitionPipeline", lambda *args: pipeline)
    kwargs = dict(bg_dir=tmp_path, n=1, seed=7, detector_weights=Path("unused"), recognizer_weights=Path("unused"))
    if fail:
        with pytest.raises(RuntimeError, match="test failure"):
            evaluate(**kwargs)
    else:
        evaluate(**kwargs)
    assert sentinel.read_text() == "keep"
    assert list(tmp_path.iterdir()) == [existing]


def test_evaluation_seed_reproduces_samples(monkeypatch, tmp_path, settings):
    settings.MEDIA_ROOT = tmp_path
    observed = []
    original_generate = synthetic_data_module.generate_plate_text
    _patch_synthetic_data(monkeypatch, tmp_path)
    def record(country):
        result = original_generate(country)
        observed.append(result)
        return result
    monkeypatch.setattr(synthetic_data_module, "generate_plate_text", record)
    monkeypatch.setattr(pipeline_module, "PlateRecognitionPipeline", lambda *args: _StubPipeline(None, None, process_text=_PLATE_TEXT, process_bbox=_FIXED_BBOX_NORM))
    for _ in range(2):
        evaluate(tmp_path, 3, 7, Path("unused"), Path("unused"))
    assert observed[:3] == observed[3:]
