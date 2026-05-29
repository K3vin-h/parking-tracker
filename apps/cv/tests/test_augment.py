"""
Unit tests for apps/cv/training/augment.py.

All tests use synthetic torch tensors — no files, no database, no network.
Tensors are pre-converted to float32 in [0, 1], which is the expected input
for both augmentation classes (ToImage + ToDtype must happen before augment).
"""
import pytest
import torch

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

    def test_train_pipeline_is_stochastic(self):
        """
        Train mode must produce different outputs under different seeds.

        Stochasticity is what augmentation buys — if the train transform is
        accidentally swapped for the eval transform, this test catches the
        regression without depending on the internal v2.Compose structure.
        """
        aug = DetectorAugment(train=True)
        img = _rgb()
        torch.manual_seed(0)
        out_a = aug(img)
        torch.manual_seed(1)
        out_b = aug(img)
        assert not torch.allclose(out_a, out_b)

    def test_train_pipeline_sometimes_flips(self):
        """
        Detector training must mix flipped and un-flipped samples.

        Behavioural check for RandomHorizontalFlip: feed an image with a sharp
        left/right brightness asymmetry (left half bright, right half dark) and
        compare left-vs-right strip means across many seeds. Without a flip the
        left strip dominates 100% of runs; with p=0.5 the distribution is
        roughly 50/50. Asserting both outcomes appear within 40 seeds confirms
        the flip is in the pipeline without inspecting v2 internals.
        """
        img = torch.zeros(3, 32, 64, dtype=torch.float32)
        img[:, :, :32] = 1.0  # bright left half
        aug = DetectorAugment(train=True)

        left_dominant = 0
        for seed in range(40):
            torch.manual_seed(seed)
            out = aug(img.clone())
            if out[:, :, :32].mean() > out[:, :, 32:].mean():
                left_dominant += 1

        assert 5 < left_dominant < 35, (
            f"Expected mixed flip behaviour over 40 seeds, got left-dominant "
            f"in {left_dominant}/40 — RandomHorizontalFlip may not be firing."
        )

    def test_train_pipeline_flips_bbox_with_image(self):
        """
        Detector horizontal flips must mirror the YOLO bbox centre.

        A plate at cx=0.25 moves to cx=0.75 after a horizontal image flip. This
        guards against image-only detector augmentation that trains against a
        target on the wrong side of the image.
        """
        img = torch.zeros(3, 32, 64, dtype=torch.float32)
        img[:, :, :32] = 1.0  # bright left half
        bbox = torch.tensor([0.25, 0.5, 0.2, 0.1], dtype=torch.float32)
        aug = DetectorAugment(train=True)

        saw_flipped = False
        saw_unflipped = False
        for seed in range(40):
            torch.manual_seed(seed)
            out_img, out_bbox = aug(img.clone(), bbox.clone())
            left_is_brighter = out_img[:, :, :32].mean() > out_img[:, :, 32:].mean()
            if left_is_brighter:
                saw_unflipped = True
                assert torch.allclose(out_bbox, bbox)
            else:
                saw_flipped = True
                expected = torch.tensor([0.75, 0.5, 0.2, 0.1], dtype=torch.float32)
                assert torch.allclose(out_bbox, expected)

        assert saw_flipped and saw_unflipped


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

    def test_train_pipeline_never_flips(self):
        """
        RecognizerAugment must never produce a horizontally mirrored image.

        Mirrored plate text ("3BA 4DC") is undecodable by a left-to-right CTC
        sequence model and would corrupt the label. Behavioural check: feed a
        sharply asymmetric image (left half bright) and assert the left strip
        is always brighter than the right strip across many seeds. 16-px guard
        bands at each end stay outside the ~20% spatial reach of
        distortion_scale=0.2 RandomPerspective, so a flip is the only
        transform that could invert this ordering.
        """
        img = torch.zeros(1, 32, 128, dtype=torch.float32)
        img[:, :, :64] = 1.0  # bright left half
        aug = RecognizerAugment(train=True)

        for seed in range(40):
            torch.manual_seed(seed)
            out = aug(img.clone())
            left = out[:, :, :16].mean()
            right = out[:, :, -16:].mean()
            assert left > right, (
                f"seed {seed}: right strip ({right.item():.3f}) is brighter than "
                f"left strip ({left.item():.3f}) — a horizontal flip leaked into "
                "the recognizer pipeline."
            )

    def test_train_pipeline_is_stochastic(self):
        """
        Recognizer training must produce varying output under different seeds.

        Indirectly verifies that at least one stochastic transform
        (ColorJitter / GaussianBlur / RandomPerspective) is active in the
        train pipeline without depending on the internal v2.Compose layout.
        """
        aug = RecognizerAugment(train=True)
        img = _gray()
        torch.manual_seed(0)
        out_a = aug(img)
        torch.manual_seed(1)
        out_b = aug(img)
        assert not torch.allclose(out_a, out_b)
