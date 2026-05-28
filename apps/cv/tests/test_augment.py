"""
Unit tests for apps/cv/training/augment.py.

All tests use synthetic torch tensors — no files, no database, no network.
Tensors are pre-converted to float32 in [0, 1], which is the expected input
for both augmentation classes (ToImage + ToDtype must happen before augment).
"""
import pytest
import torch
from torchvision.transforms import v2

from apps.cv.training.augment import DetectorAugment, RecognizerAugment


# ── Tensor helpers ────────────────────────────────────────────────────────────

def _rgb(h: int = 480, w: int = 640) -> torch.Tensor:
    """Random float32 RGB tensor in [0, 1], shape (3, H, W)."""
    return torch.rand(3, h, w, dtype=torch.float32)


def _gray(h: int = 32, w: int = 128) -> torch.Tensor:
    """Random float32 grayscale tensor in [0, 1], shape (1, H, W)."""
    return torch.rand(1, h, w, dtype=torch.float32)


# ── DetectorAugment ───────────────────────────────────────────────────────────

@pytest.mark.unit
class TestDetectorAugment:
    def test_train_preserves_shape(self):
        """Training augmentation must not alter the tensor shape."""
        aug = DetectorAugment(train=True)
        img = _rgb()
        assert aug(img).shape == img.shape

    def test_eval_preserves_shape(self):
        """Eval augmentation (normalise only) must not alter the tensor shape."""
        aug = DetectorAugment(train=False)
        img = _rgb()
        assert aug(img).shape == img.shape

    def test_output_dtype_is_float32(self):
        """Augmented tensor must remain float32."""
        assert DetectorAugment(train=True)(_rgb()).dtype == torch.float32

    def test_eval_is_deterministic(self):
        """Eval mode must produce bit-identical output for the same input."""
        aug = DetectorAugment(train=False)
        img = _rgb()
        assert torch.allclose(aug(img), aug(img))

    def test_normalisation_shifts_uniform_input(self):
        """
        A uniform 0.5 tensor must be shifted after ImageNet normalisation.

        After Normalize(mean=[0.485, 0.456, 0.406], std=[...]) the output
        is NOT 0.5 — confirming normalisation actually ran.
        """
        aug = DetectorAugment(train=False)
        img = torch.full((3, 480, 640), 0.5)
        out = aug(img)
        assert not torch.allclose(out, img, atol=0.01)

    def test_train_pipeline_has_horizontal_flip(self):
        """
        Detector training must include RandomHorizontalFlip.

        Plates approach from either direction, so the detector must learn to
        find them regardless of horizontal orientation.
        """
        aug = DetectorAugment(train=True)
        transforms_list = getattr(aug._transform, "transforms", [])
        assert any(isinstance(t, v2.RandomHorizontalFlip) for t in transforms_list)


# ── RecognizerAugment ─────────────────────────────────────────────────────────

@pytest.mark.unit
class TestRecognizerAugment:
    def test_train_preserves_shape(self):
        """Training augmentation must not alter the tensor shape."""
        aug = RecognizerAugment(train=True)
        img = _gray()
        assert aug(img).shape == img.shape

    def test_eval_preserves_shape(self):
        """Eval augmentation (normalise only) must not alter the tensor shape."""
        aug = RecognizerAugment(train=False)
        img = _gray()
        assert aug(img).shape == img.shape

    def test_output_dtype_is_float32(self):
        """Augmented tensor must remain float32."""
        assert RecognizerAugment(train=True)(_gray()).dtype == torch.float32

    def test_eval_is_deterministic(self):
        """Eval mode must produce bit-identical output for the same input."""
        aug = RecognizerAugment(train=False)
        img = _gray()
        assert torch.allclose(aug(img), aug(img))

    def test_normalisation_shifts_uniform_input(self):
        """
        A uniform 0.5 tensor must be shifted after single-channel normalisation.

        Normalize(mean=[0.5], std=[0.5]) maps 0.5 → 0.0, confirming it ran.
        """
        aug = RecognizerAugment(train=False)
        img = torch.full((1, 32, 128), 0.5)
        out = aug(img)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-5)

    def test_no_horizontal_flip_in_train_pipeline(self):
        """
        RecognizerAugment must NOT include RandomHorizontalFlip.

        Mirrored plate text ("3BA 4DC") cannot be decoded by a left-to-right
        sequence model and would corrupt the label during training.
        """
        aug = RecognizerAugment(train=True)
        transforms_list = getattr(aug._transform, "transforms", [])
        for t in transforms_list:
            assert not isinstance(t, v2.RandomHorizontalFlip), (
                "RecognizerAugment must not contain RandomHorizontalFlip"
            )

    def test_train_pipeline_has_perspective(self):
        """
        Recognizer training must include RandomPerspective.

        Cameras are rarely perpendicular to the plate; perspective warp
        teaches the recognizer to handle angled views.
        """
        aug = RecognizerAugment(train=True)
        transforms_list = getattr(aug._transform, "transforms", [])
        assert any(isinstance(t, v2.RandomPerspective) for t in transforms_list)
