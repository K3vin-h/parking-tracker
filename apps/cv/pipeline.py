"""
End-to-end license plate recognition pipeline.

PlateRecognitionPipeline wires together the preprocessing module,
PlateDetectorCNN, and PlateRecognizerCRNN into a single callable:

    result = pipeline.process(image_path)
    # {"plate_text": "ABC123", "confidence": 0.87, "bounding_box": [...], "is_low_confidence": False}

WHY a class rather than a function: both models must be loaded once at
startup — loading weights from disk on every request would add hundreds of
milliseconds of latency. The class holds the loaded models as instance
attributes; get_pipeline() provides a module-level singleton so Django views
share one loaded copy across all requests in the same process.
"""

import logging
import os
import threading
from typing import TypedDict

import torch

from apps.cv.models.plate_detector import PlateDetectorCNN
from apps.cv.models.recognizer import PlateRecognizerCRNN
from apps.cv.preprocessing import (
    bgr_to_rgb,
    crop_plate_region,
    load_image,
    normalize_pixels,
    prepare_for_recognizer,
    resize_for_detector,
    to_tensor,
)
from apps.cv.training.dataset import BLANK_IDX
from apps.cv.utils.device import get_device

logger = logging.getLogger(__name__)

# Confidence below this threshold flags the detection event for manual review
# in the operator dashboard.  Mirrors the default value of
# LotSettings.confidence_threshold so the pipeline's built-in flag aligns
# with the admin-configurable threshold out of the box.
LOW_CONFIDENCE_THRESHOLD: float = 0.6

# Bounding-box width or height below this fraction of image dimensions means
# the detector found nothing meaningful.  A 5 % plate would be ~32 px wide on
# a 640 px image — too small for the recognizer to read reliably.
_MIN_BBOX_SIZE: float = 0.05


class PipelineResult(TypedDict):
    """Return type for PlateRecognitionPipeline.process()."""

    plate_text: str
    confidence: float
    bounding_box: list[float]   # [x, y, w, h] top-left + size, normalised [0, 1]
    is_low_confidence: bool


class PlateRecognitionPipeline:
    """
    End-to-end pipeline: image path → plate text + confidence.

    Loads both CV models once at construction time and keeps them in eval mode
    on the target device.  Subsequent process() calls are stateless and safe
    to use from multiple threads (eval mode, no gradient state is mutated).

    Args:
        detector_path:   Path to PlateDetectorCNN state-dict file (.pth).
        recognizer_path: Path to PlateRecognizerCRNN state-dict file (.pth).
        device:          Target compute device.  Defaults to get_device()
                         which selects MPS → CUDA → CPU automatically.

    Raises:
        FileNotFoundError: If either weight file does not exist.
    """

    def __init__(
        self,
        detector_path: str,
        recognizer_path: str,
        device: torch.device | None = None,
    ) -> None:
        if not os.path.isfile(detector_path):
            raise FileNotFoundError(
                f"Detector weights not found: {detector_path!r}. "
                "Train the detector first: python apps/cv/training/train_detector.py"
            )
        if not os.path.isfile(recognizer_path):
            raise FileNotFoundError(
                f"Recognizer weights not found: {recognizer_path!r}. "
                "Train the recognizer first: python apps/cv/training/train_recognizer.py"
            )

        self.device = device if device is not None else get_device()

        # WHY weights_only=True: prevents arbitrary code execution that would
        # be possible with a pickle-based load of an untrusted .pth file.
        self.detector = PlateDetectorCNN()
        self.detector.load_state_dict(
            torch.load(detector_path, map_location=self.device, weights_only=True)
        )
        self.detector.to(self.device)
        self.detector.eval()

        self.recognizer = PlateRecognizerCRNN()
        self.recognizer.load_state_dict(
            torch.load(recognizer_path, map_location=self.device, weights_only=True)
        )
        self.recognizer.to(self.device)
        self.recognizer.eval()

        logger.info("PlateRecognitionPipeline ready device=%s", self.device)

    def process(self, image_path: str) -> PipelineResult:
        """
        Run the full plate recognition pipeline on a single image.

        Pipeline steps:
        1. Load + preprocess → 640×480 normalised tensor for the detector.
        2. Detector → bounding box [cx, cy, w, h] in the resized image space.
        3. If bbox too small, return early with empty plate_text.
        4. Convert YOLO center bbox to top-left; crop the plate region.
        5. Prepare the crop for the recognizer (128×32 grayscale tensor).
        6. Recognizer → greedy CTC decode → plate_text.
        7. Confidence = mean max-class probability across time-steps.

        WHY crop from the resized image (not the original): the detector was
        trained on 640×480 inputs, so its bbox coordinates are relative to that
        resolution.  Cropping the same resized image ensures the pixel region
        the detector "saw" is what gets passed to the recognizer.

        WHY return bounding_box in top-left format (not YOLO center): the
        PlateDetectionEvent model stores bounding_box as [x, y, w, h] top-left
        + size.  Converting here keeps all callers consistent.

        Args:
            image_path: Path to the uploaded image.  Must be inside MEDIA_ROOT
                        (enforced by load_image's path guard).

        Returns:
            PipelineResult with plate_text, confidence, bounding_box, and
            is_low_confidence.

        Raises:
            FileNotFoundError: If the image cannot be loaded.
            ValueError:        If image_path is outside MEDIA_ROOT.
        """
        # ── Step 1: load and preprocess for the detector ──────────────────
        bgr = load_image(image_path)                          # (H, W, 3) uint8 BGR
        rgb = bgr_to_rgb(bgr)                                 # (H, W, 3) uint8 RGB
        rgb_resized = resize_for_detector(rgb)                # (480, 640, 3) uint8 RGB
        tensor = to_tensor(normalize_pixels(rgb_resized))     # (3, 480, 640) float32

        # ── Step 2: detect plate bounding box ─────────────────────────────
        # unsqueeze(0) adds the batch dimension: (3, 480, 640) → (1, 3, 480, 640)
        detector_input = tensor.unsqueeze(0).to(self.device)
        bbox_tensor = self.detector.predict(detector_input)   # (1, 4) [cx, cy, w, h]
        cx, cy, w, h = bbox_tensor[0].tolist()                # YOLO center format

        # ── Step 3: reject tiny / missing plate ───────────────────────────
        if w < _MIN_BBOX_SIZE or h < _MIN_BBOX_SIZE:
            logger.debug(
                "Plate bbox too small w=%.3f h=%.3f — treating as no plate detected",
                w, h,
            )
            return {
                "plate_text": "",
                "confidence": 0.0,
                "bounding_box": [cx - w / 2, cy - h / 2, w, h],
                "is_low_confidence": True,
            }

        # ── Step 4: crop plate region ─────────────────────────────────────
        # Convert YOLO center [cx, cy, w, h] → top-left [x, y, w, h] so
        # crop_plate_region receives its documented coordinate convention.
        x = cx - w / 2
        y = cy - h / 2
        try:
            crop = crop_plate_region(rgb_resized, [x, y, w, h])  # uint8 RGB
        except ValueError as exc:
            # bbox passed the size check but produces a zero-area crop after
            # integer conversion and clamping (e.g. plate at image edge).
            logger.debug("crop_plate_region raised on edge bbox: %s", exc)
            return {
                "plate_text": "",
                "confidence": 0.0,
                "bounding_box": [x, y, w, h],
                "is_low_confidence": True,
            }
        crop_tensor = prepare_for_recognizer(crop)             # (1, 32, 128) float32

        # ── Step 5: recognise text ────────────────────────────────────────
        recog_input = crop_tensor.unsqueeze(0).to(self.device)  # (1, 1, 32, 128)
        log_probs = self.recognizer.predict(recog_input)         # (T=16, 1, C=37)
        plate_text = self.recognizer.decode_predictions(log_probs)[0]

        # ── Step 6: compute confidence ────────────────────────────────────
        # Confidence is the mean max-class probability over non-blank time-steps.
        #
        # WHY exclude blank positions: the CTC model emits T=16 time-steps for
        # every plate regardless of length.  On a 6-character plate, ~10 steps
        # are blank-dominated (high probability on index 0).  Including them
        # inflates the mean and can mask genuine uncertainty on character steps.
        # Restricting to non-blank argmax positions gives a cleaner signal.
        # Fallback to the full mean when all steps are blank (all-blank output
        # means the model saw nothing; preserving the low value is correct).
        probs = torch.exp(log_probs)                           # (16, 1, 37)
        max_probs = probs.max(dim=-1).values.squeeze(1)        # (16,)
        argmax = log_probs.argmax(dim=-1).squeeze(1)           # (16,)
        char_mask = argmax != BLANK_IDX                        # (16,) bool
        char_probs = max_probs[char_mask]
        confidence: float = (
            char_probs.mean().item() if char_mask.any() else max_probs.mean().item()
        )
        is_low_confidence = confidence < LOW_CONFIDENCE_THRESHOLD

        logger.debug(
            "Detected plate=%r confidence=%.3f low_conf=%s",
            plate_text, confidence, is_low_confidence,
        )

        return {
            "plate_text": plate_text,
            "confidence": confidence,
            "bounding_box": [x, y, w, h],
            "is_low_confidence": is_low_confidence,
        }


# ── Module-level singleton ─────────────────────────────────────────────────────
#
# WHY lazy init (not AppConfig.ready()): AppConfig runs before the first HTTP
# request but also during management commands (migrate, collectstatic) where
# weights are absent and inference is never needed.  A FileNotFoundError at
# startup would prevent running migrations in CI.  Lazy init defers the error
# to the first actual upload request.

_instance: PlateRecognitionPipeline | None = None
_lock = threading.Lock()


def get_pipeline(detector_path: str, recognizer_path: str) -> PlateRecognitionPipeline:
    """
    Return the module-level singleton, creating it on first call.

    Args:
        detector_path:   Path to detector weights — used only on first call;
                         subsequent calls return the cached instance unchanged.
        recognizer_path: Path to recognizer weights — same caveat.

    Returns:
        The shared PlateRecognitionPipeline instance.
    """
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:  # double-checked locking
                _instance = PlateRecognitionPipeline(detector_path, recognizer_path)
    return _instance
