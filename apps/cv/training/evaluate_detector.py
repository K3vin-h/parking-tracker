"""
End-to-end evaluation harness for the plate detector + recognizer pipeline.

WHY THIS SCRIPT EXISTS:
  train_detector.py reports validation IoU against a validation dataset or a random split of the same
  synthetic dataset it trains on. That number does not answer the question
  that actually matters for the kiosk: "given a fresh photo composited on a
  background the model has never seen, does the full pipeline read the plate
  correctly?" This script generates fresh samples from a caller-chosen
  background directory (normally a holdout set never used in training) and
  reports three numbers:

    1. Detector IoU        — mean IoU between the predicted and ground-truth
                              bounding box. Same metric as train_detector.py,
                              but on unseen backgrounds.
    2. End-to-end accuracy — the full PlateRecognitionPipeline.process() output
                              matches the ground-truth plate text exactly.
    3. Oracle-bbox accuracy — the recognizer alone, fed the GROUND-TRUTH crop
                              instead of the detector's predicted crop. This
                              isolates recognizer error from detector error:
                              if this number is low, retraining the detector
                              alone cannot fix end-to-end accuracy.

  Run outside Docker (needs a local torch/torchvision install — see
  requirements-dev.txt):

    SECRET_KEY=eval DEBUG=True DB_PASSWORD=unused PYTHONPATH=. \\
        python apps/cv/training/evaluate_detector.py \\
        --bg-dir data/backgrounds_holdout --n 300 --seed 7 --json eval.json

  config/settings.py refuses to load without SECRET_KEY and DB_PASSWORD, and
  requires a health-check token unless DEBUG=True. This script never serves a
  request or opens a database connection, so throwaway values are fine — but
  they are passed explicitly rather than hard-coded here, so no secret-shaped
  default ever lives in the source.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger("apps.cv.training.evaluate_detector")


def _bootstrap_django() -> None:
    """
    Configure Django settings before importing anything that touches models,
    settings.MEDIA_ROOT, or the ORM. This script never writes to the database,
    but PlateRecognitionPipeline's load_image() enforces a MEDIA_ROOT
    containment check, so settings must be configured to run it at all.
    Required environment variables are documented in the module docstring.
    """
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    import django

    django.setup()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the detector+recognizer pipeline on fresh synthetic samples."
    )
    parser.add_argument(
        "--bg-dir",
        type=Path,
        required=True,
        help="Background directory to composite plates onto (use a holdout set "
        "never seen during training for an honest number).",
    )
    parser.add_argument(
        "--n", type=int, default=300, help="Number of samples to evaluate."
    )
    parser.add_argument(
        "--seed", type=int, default=7, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--detector-weights",
        type=Path,
        default=Path("apps/cv/weights/detector.pth"),
    )
    parser.add_argument(
        "--recognizer-weights",
        type=Path,
        default=Path("apps/cv/weights/recognizer.pth"),
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path to write results as JSON.",
    )
    return parser.parse_args()


def evaluate(
    bg_dir: Path,
    n: int,
    seed: int,
    detector_weights: Path,
    recognizer_weights: Path,
) -> dict[str, float]:
    """
    Generate `n` fresh synthetic samples against `bg_dir` and score the pipeline.

    Isolated as a plain function (no argparse/CLI concerns) so it is directly
    unit-testable with a stub pipeline — see test_evaluate_detector.py.
    """
    import random

    import torch
    from django.conf import settings

    from apps.cv.pipeline import (
        _RECOGNIZER_EVAL_TRANSFORM,
        PlateRecognitionPipeline,
    )
    from apps.cv.preprocessing import (
        bgr_to_rgb, crop_plate_region, load_image, prepare_for_recognizer,
    )
    from apps.cv.training.synthetic_data import (
        _seed_rng,
        _validate_sample_count,
        composite_on_background,
        generate_plate_text,
        render_plate_image,
    )
    from apps.parking.services import normalize_plate

    _validate_sample_count(n)
    _seed_rng(seed)
    pipeline = PlateRecognitionPipeline(str(detector_weights), str(recognizer_weights))

    # Images must live under MEDIA_ROOT — load_image() rejects anything else as
    # a path-traversal attempt (UnsafeImagePathError), by design (see
    # docs/technical/01-cv-pipeline.md#image-preprocessing).
    Path(settings.MEDIA_ROOT).mkdir(parents=True, exist_ok=True)
    temp_dir = tempfile.TemporaryDirectory(prefix="eval_detector_", dir=settings.MEDIA_ROOT)
    tmp_dir = Path(temp_dir.name)

    rng = random.Random(seed)
    ious: list[float] = []
    end_to_end_hits = 0
    oracle_hits = 0

    try:
        for i in range(n):
            country = "US" if rng.random() < 0.6 else "CA"
            text, _ = generate_plate_text(country)
            plate_img = render_plate_image(text, country)
            composite, gt_bbox = composite_on_background(plate_img, bg_dir)
            expected = normalize_plate(text)

            img_path = tmp_dir / f"sample_{i}.jpg"
            composite.save(img_path, quality=92)

            result = pipeline.process(str(img_path))
            gx, gy, gw, gh = gt_bbox
            img_w, img_h = composite.size
            gt_norm = [gx / img_w, gy / img_h, gw / img_w, gh / img_h]
            pred_bbox = result["bounding_box"]
            if pred_bbox:
                ious.append(_iou_xywh(pred_bbox, gt_norm))
            else:
                ious.append(0.0)
            if normalize_plate(result["plate_text"]) == expected:
                end_to_end_hits += 1

            # Oracle path: crop the GROUND-TRUTH box directly (bypassing the
            # detector entirely) and see if the recognizer alone gets it right.
            crop = crop_plate_region(bgr_to_rgb(load_image(str(img_path))), gt_norm)
            crop_tensor = _RECOGNIZER_EVAL_TRANSFORM(prepare_for_recognizer(crop))
            with torch.no_grad():
                log_probs = pipeline.recognizer.predict(
                    crop_tensor.unsqueeze(0).to(pipeline.device)
                )
            oracle_text = pipeline.recognizer.decode_predictions(log_probs)[0]
            if normalize_plate(oracle_text) == expected:
                oracle_hits += 1
    finally:
        temp_dir.cleanup()

    mean_iou = sum(ious) / len(ious) if ious else 0.0
    iou_ge_50 = sum(1 for v in ious if v >= 0.5) / len(ious) if ious else 0.0
    iou_ge_70 = sum(1 for v in ious if v >= 0.7) / len(ious) if ious else 0.0

    return {
        "n": n,
        "mean_iou": mean_iou,
        "iou_ge_0.5": iou_ge_50,
        "iou_ge_0.7": iou_ge_70,
        "end_to_end_accuracy": end_to_end_hits / n if n else 0.0,
        "oracle_bbox_accuracy": oracle_hits / n if n else 0.0,
    }


def _iou_xywh(a: list[float], b: list[float]) -> float:
    """
    IoU between two top-left [x, y, w, h] boxes in the same normalized units.

    A small, dependency-free helper: train_detector.py's _compute_batch_iou
    operates on batched YOLO-center tensors during training. Here a single
    pair of already-decoded top-left boxes is compared per sample, so a plain
    scalar implementation avoids a tensor round-trip for one box at a time.
    """
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, a[2]) * max(0.0, a[3])
    area_b = max(0.0, b[2]) * max(0.0, b[3])
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def main() -> None:
    args = _parse_args()
    if not args.bg_dir.is_dir():
        raise SystemExit(
            f"--bg-dir {args.bg_dir} does not exist or is not a directory."
        )
    _bootstrap_django()
    logging.basicConfig(level=logging.WARNING)
    results = evaluate(
        bg_dir=args.bg_dir,
        n=args.n,
        seed=args.seed,
        detector_weights=args.detector_weights,
        recognizer_weights=args.recognizer_weights,
    )
    print(f"Samples:                {results['n']}")
    print(f"Mean IoU:               {results['mean_iou']:.4f}")
    print(f"IoU >= 0.5:             {results['iou_ge_0.5']:.1%}")
    print(f"IoU >= 0.7:             {results['iou_ge_0.7']:.1%}")
    print(f"End-to-end accuracy:    {results['end_to_end_accuracy']:.1%}")
    print(f"Oracle-bbox accuracy:   {results['oracle_bbox_accuracy']:.1%}")
    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
