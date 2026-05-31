"""
Unit tests for PlateDetectorCNN and the IoU helper used in training.

All tests run on CPU only — no GPU/MPS required in CI.  The model is
deliberately small, so CPU forward passes finish in well under a second.

Fixtures
────────
No external files are needed.  Tests construct random input tensors directly.
"""

import pytest
import torch

from apps.cv.models.plate_detector import PlateDetectorCNN


# ── Helpers ───────────────────────────────────────────────────────────────────

def _random_batch(batch_size: int = 2, height: int = 480, width: int = 640) -> torch.Tensor:
    """Return a random float32 image batch with values in [0, 1]."""
    return torch.rand(batch_size, 3, height, width)


# ── Model architecture tests ─────────────────────────────────────────────────

@pytest.mark.unit
class TestPlateDetectorCNN:
    """Structural and behavioural tests for PlateDetectorCNN."""

    def test_forward_output_shape(self) -> None:
        """Standard 480×640 input should produce a (B, 4) bbox tensor."""
        model = PlateDetectorCNN()
        x = _random_batch(batch_size=4)
        out = model(x)
        assert out.shape == (4, 4), f"Expected (4, 4), got {out.shape}"

    def test_forward_single_image(self) -> None:
        """Batch of 1 should work without squeeze errors."""
        model = PlateDetectorCNN()
        x = _random_batch(batch_size=1)
        out = model(x)
        assert out.shape == (1, 4)

    def test_forward_arbitrary_input_size(self) -> None:
        """AdaptiveAvgPool2d should accept non-standard spatial dimensions."""
        model = PlateDetectorCNN()
        for h, w in [(224, 224), (320, 480), (512, 512), (100, 200)]:
            x = _random_batch(batch_size=2, height=h, width=w)
            out = model(x)
            assert out.shape == (2, 4), f"Failed for input size {h}×{w}"

    def test_predict_range(self) -> None:
        """predict() must clamp all outputs to [0, 1] via sigmoid."""
        model = PlateDetectorCNN()
        x = _random_batch(batch_size=8)
        pred = model.predict(x)
        assert pred.shape == (8, 4)
        assert pred.min().item() >= 0.0, "predict() output below 0"
        assert pred.max().item() <= 1.0, "predict() output above 1"

    def test_predict_does_not_require_grad(self) -> None:
        """predict() uses @torch.no_grad() — gradients must not be tracked."""
        model = PlateDetectorCNN()
        x = _random_batch(batch_size=2)
        pred = model.predict(x)
        assert not pred.requires_grad

    def test_eval_mode_deterministic(self) -> None:
        """Same input in eval mode must produce identical outputs (no dropout)."""
        model = PlateDetectorCNN()
        model.eval()
        x = _random_batch(batch_size=2)
        out1 = model(x)
        out2 = model(x)
        assert torch.allclose(out1, out2), "eval mode gave different outputs for same input"

    def test_train_mode_nondeterministic_with_dropout(self) -> None:
        """
        Dropout should cause at least one differing forward pass in train mode.

        We run 10 passes and assert that not all are identical.  A probabilistic
        test — the chance of 10 identical dropout masks is (0.7^256)^10 ≈ 0.
        """
        model = PlateDetectorCNN()
        model.train()
        torch.manual_seed(0)
        x = _random_batch(batch_size=2)
        outputs = [model(x).detach() for _ in range(10)]
        all_same = all(torch.allclose(outputs[0], o) for o in outputs[1:])
        assert not all_same, "train mode produced identical outputs on 10 passes — dropout may be broken"

    def test_parameter_count_in_expected_range(self) -> None:
        """
        Total trainable parameters should be in the range [500 k, 3 M].

        This is a sanity check — too few params means a layer is missing,
        too many means an accidental architecture change inflated the model.
        """
        model = PlateDetectorCNN()
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert 500_000 <= total <= 3_000_000, (
            f"Unexpected parameter count: {total:,}. "
            "Expected between 500 k and 3 M."
        )

    def test_output_dtype_float32(self) -> None:
        """Output tensor must be float32 to match loss function expectations."""
        model = PlateDetectorCNN()
        x = _random_batch(batch_size=2)
        out = model(x)
        assert out.dtype == torch.float32

    def test_custom_dropout_rate(self) -> None:
        """Constructor dropout parameter should be respected."""
        model_high = PlateDetectorCNN(dropout=0.9)
        # With very high dropout, train-mode outputs should differ almost always
        model_high.train()
        x = _random_batch(batch_size=4)
        outputs = [model_high(x).detach() for _ in range(5)]
        all_same = all(torch.allclose(outputs[0], o) for o in outputs[1:])
        assert not all_same

    def test_forward_outputs_in_unit_interval(self) -> None:
        """
        forward() applies sigmoid internally — outputs must be in [0, 1].

        This guards against the train/inference mismatch that occurs when raw
        logits are used during training but sigmoid is applied only at inference.
        """
        model = PlateDetectorCNN()
        model.eval()
        x = _random_batch(batch_size=4)
        with torch.no_grad():
            out = model(x)
        assert out.min().item() >= 0.0, "forward() output below 0 — sigmoid missing?"
        assert out.max().item() <= 1.0, "forward() output above 1 — sigmoid missing?"

    def test_predict_matches_forward_under_no_grad(self) -> None:
        """
        predict() must return the same values as forward() under no_grad.

        Regression guard: if predict() ever re-applies sigmoid on top of a
        forward() that already applies it, values would be doubly squashed and
        this test would fail.
        """
        model = PlateDetectorCNN()
        model.eval()
        x = _random_batch(batch_size=2)
        with torch.no_grad():
            expected = model(x)
        actual = model.predict(x)
        assert torch.allclose(actual, expected), (
            "predict() and forward() returned different values — "
            "sigmoid is likely being applied twice."
        )


# ── IoU helper tests ─────────────────────────────────────────────────────────
#
# The IoU helper lives in train_detector.py.  We import it directly rather than
# going through the training CLI to keep tests fast and dependency-free.

@pytest.mark.unit
class TestComputeBatchIoU:
    """Tests for the _compute_batch_iou helper in train_detector.py."""

    @staticmethod
    def _iou_fn(pred: torch.Tensor, target: torch.Tensor) -> float:
        from apps.cv.training.train_detector import _compute_batch_iou
        return _compute_batch_iou(pred, target).item()

    def test_perfect_overlap_returns_one(self) -> None:
        """Identical pred and target boxes → IoU = 1.0."""
        box = torch.tensor([[0.5, 0.5, 0.4, 0.3]])
        iou = self._iou_fn(box, box)
        assert abs(iou - 1.0) < 1e-5, f"Expected 1.0, got {iou}"

    def test_no_overlap_returns_zero(self) -> None:
        """Non-overlapping boxes → IoU = 0.0."""
        pred   = torch.tensor([[0.1, 0.1, 0.1, 0.1]])
        target = torch.tensor([[0.9, 0.9, 0.1, 0.1]])
        iou = self._iou_fn(pred, target)
        assert abs(iou - 0.0) < 1e-5, f"Expected 0.0, got {iou}"

    def test_partial_overlap_known_value(self) -> None:
        """
        Two unit squares offset by 0.5 in cx → intersection = 0.5×1 = 0.5,
        union = 1 + 1 − 0.5 = 1.5  →  IoU ≈ 0.3333.

        pred:   cx=0.25, cy=0.5, w=0.5, h=1.0  → x1=0.0, x2=0.5, y1=0.0, y2=1.0
        target: cx=0.75, cy=0.5, w=0.5, h=1.0  → x1=0.5, x2=1.0, y1=0.0, y2=1.0

        These share exactly the edge at x=0.5, so intersection area = 0.
        Use a proper overlap instead:

        pred:   cx=0.4, cy=0.5, w=0.4, h=0.4  → x1=0.2, x2=0.6, y1=0.3, y2=0.7
        target: cx=0.6, cy=0.5, w=0.4, h=0.4  → x1=0.4, x2=0.8, y1=0.3, y2=0.7

        intersection: x [0.4, 0.6] × y [0.3, 0.7] = 0.2 × 0.4 = 0.08
        area_pred   = 0.4 × 0.4 = 0.16
        area_target = 0.4 × 0.4 = 0.16
        union       = 0.16 + 0.16 − 0.08 = 0.24
        IoU         = 0.08 / 0.24 = 1/3 ≈ 0.3333
        """
        pred   = torch.tensor([[0.4, 0.5, 0.4, 0.4]])
        target = torch.tensor([[0.6, 0.5, 0.4, 0.4]])
        iou = self._iou_fn(pred, target)
        assert abs(iou - (1.0 / 3.0)) < 1e-4, f"Expected ~0.333, got {iou}"

    def test_batch_mean_is_correct(self) -> None:
        """Mean IoU over a batch of 2: one perfect, one zero → mean = 0.5."""
        box = torch.tensor([[0.5, 0.5, 0.4, 0.3]])
        pred   = torch.cat([box, torch.tensor([[0.1, 0.1, 0.1, 0.1]])])
        target = torch.cat([box, torch.tensor([[0.9, 0.9, 0.1, 0.1]])])
        iou = self._iou_fn(pred, target)
        assert abs(iou - 0.5) < 1e-4, f"Expected 0.5, got {iou}"

    def test_sigmoid_clamped_preds_still_valid(self) -> None:
        """Large logits (before sigmoid) should not cause NaN IoU."""
        raw_logits = torch.tensor([[10.0, -10.0, 5.0, -5.0]])
        clamped = torch.sigmoid(raw_logits)
        target  = torch.tensor([[0.5, 0.5, 0.4, 0.3]])
        iou = self._iou_fn(clamped, target)
        assert not torch.isnan(torch.tensor(iou)), "IoU is NaN for sigmoid-clamped predictions"
