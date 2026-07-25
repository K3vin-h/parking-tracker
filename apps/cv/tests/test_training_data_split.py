"""Regression tests that keep stochastic augmentation out of validation data."""

import csv
from pathlib import Path

from PIL import Image

from apps.cv.training.augment import DetectorAugment, RecognizerAugment
from apps.cv.training.train_detector import _build_datasets as build_detector_datasets
from apps.cv.training.train_recognizer import (
    _build_datasets as build_recognizer_datasets,
)


def _detector_root(root: Path, count: int = 5) -> None:
    """Create the smallest valid detector corpus for split wiring tests."""
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    for index in range(count):
        Image.new("RGB", (32, 24)).save(root / "images" / f"{index:06d}.jpg")
        (root / "labels" / f"{index:06d}.txt").write_text("0 0.5 0.5 0.4 0.2")


def _recognizer_root(root: Path, count: int = 5) -> None:
    """Create the smallest valid recognizer corpus for split wiring tests."""
    (root / "images").mkdir(parents=True)
    with (root / "labels.csv").open("w", newline="") as labels_file:
        writer = csv.DictWriter(
            labels_file,
            fieldnames=["filename", "text", "country"],
        )
        writer.writeheader()
        for index in range(count):
            filename = f"{index:06d}.png"
            Image.new("L", (128, 32)).save(root / "images" / filename)
            writer.writerow(
                {"filename": filename, "text": f"ABC12{index}", "country": "US"}
            )


def _assert_split_contract(train_set, val_set, augment_class) -> None:
    """Verify disjoint samples and distinct train/eval transform modes."""
    assert set(train_set.indices).isdisjoint(val_set.indices)
    assert set(train_set.indices) | set(val_set.indices) == set(range(5))
    assert train_set.dataset is not val_set.dataset
    assert isinstance(train_set.dataset._transform, augment_class)
    assert isinstance(val_set.dataset._transform, augment_class)
    assert train_set.dataset._transform._train is True
    assert val_set.dataset._transform._train is False


def test_detector_training_split_uses_train_only_augmentation(tmp_path):
    """Detector validation metrics must remain deterministic and unaugmented."""
    root = tmp_path / "detector"
    _detector_root(root)

    _assert_split_contract(*build_detector_datasets(root, seed=7), DetectorAugment)


def test_recognizer_training_split_uses_train_only_augmentation(tmp_path):
    """Recognizer validation metrics must remain deterministic and unaugmented."""
    root = tmp_path / "recognizer"
    _recognizer_root(root)

    _assert_split_contract(
        *build_recognizer_datasets(root, seed=7),
        RecognizerAugment,
    )
