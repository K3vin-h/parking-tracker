"""
Unit tests for PlateRecognizerCRNN.

All tests run on CPU only — no GPU/MPS required in CI.  The model is
large by CV standards (~18 M params) but forward passes on CPU finish
in well under a second for small batches.

Fixtures
────────
No external files are needed.  Tests construct random input tensors directly.
"""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from apps.cv.models.recognizer import PlateRecognizerCRNN
from apps.cv.training.dataset import BLANK_IDX, VOCAB_SIZE


# ── Helpers ───────────────────────────────────────────────────────────────────

def _random_plate_batch(batch_size: int = 2) -> torch.Tensor:
    """Return a random float32 grayscale plate batch, shape (B, 1, 32, 128)."""
    return torch.rand(batch_size, 1, 32, 128)


# ── Model tests ───────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestPlateRecognizerCRNN:
    """Structural and behavioural tests for PlateRecognizerCRNN."""

    def test_forward_output_shape(self) -> None:
        """Standard plate input should produce (T=16, B, C=37) output."""
        model = PlateRecognizerCRNN()
        x = _random_plate_batch(batch_size=4)
        out = model(x)
        assert out.shape == (16, 4, VOCAB_SIZE), f"Expected (16, 4, 37), got {out.shape}"

    def test_forward_single_image(self) -> None:
        """Batch of 1 should work without squeeze errors."""
        model = PlateRecognizerCRNN()
        x = _random_plate_batch(batch_size=1)
        out = model(x)
        assert out.shape == (16, 1, VOCAB_SIZE)

    def test_output_is_log_softmax(self) -> None:
        """
        forward() applies log_softmax — all output values must be <= 0.

        Log-probabilities are always non-positive (log(p) <= 0 for p in [0,1]).
        Any positive value would mean the output is not a valid log-probability
        and would silently corrupt CTCLoss.
        """
        model = PlateRecognizerCRNN()
        model.eval()
        with torch.no_grad():
            out = model(_random_plate_batch(batch_size=2))
        assert out.max().item() <= 0.0, (
            "forward() returned values > 0 — log_softmax was not applied or "
            "was applied along the wrong dimension."
        )

    def test_output_exp_sums_to_one(self) -> None:
        """
        exp(forward()) should sum to ~1 along the class dimension.

        Verifies that log_softmax is applied over C (dim=-1), not over T or N.
        """
        model = PlateRecognizerCRNN()
        model.eval()
        with torch.no_grad():
            out = model(_random_plate_batch(batch_size=2))
        class_probs = out.exp()
        row_sums = class_probs.sum(dim=-1)  # (T, B)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
            "exp(output).sum(dim=-1) is not close to 1 — log_softmax dim is wrong."
        )

    def test_output_dtype_float32(self) -> None:
        """Output tensor must be float32 to match CTCLoss expectations."""
        model = PlateRecognizerCRNN()
        out = model(_random_plate_batch(batch_size=2))
        assert out.dtype == torch.float32

    def test_forward_docstring_warns_log_softmax(self) -> None:
        """
        forward() docstring must warn callers not to re-apply log_softmax.

        A caller that applies log_softmax again will produce valid-looking but
        incorrect log-probabilities, causing CTCLoss to silently train on garbage.
        The docstring is the only defence against this silent footgun.
        """
        assert "log_softmax" in (PlateRecognizerCRNN.forward.__doc__ or ""), (
            "forward() docstring must mention 'log_softmax' to warn callers "
            "against applying it a second time."
        )

    def test_predict_output_shape(self) -> None:
        """predict() should return the same shape as forward()."""
        model = PlateRecognizerCRNN()
        pred = model.predict(_random_plate_batch(batch_size=3))
        assert pred.shape == (16, 3, VOCAB_SIZE)

    def test_predict_does_not_require_grad(self) -> None:
        """predict() uses @torch.no_grad() — gradients must not be tracked."""
        model = PlateRecognizerCRNN()
        pred = model.predict(_random_plate_batch(batch_size=2))
        assert not pred.requires_grad

    def test_predict_restores_train_state(self) -> None:
        """
        predict() called while model is in training mode must restore that state.

        This allows predict() to be called inside a training loop (e.g. for a
        mid-epoch accuracy sample) without disabling dropout for the rest of the
        epoch.
        """
        model = PlateRecognizerCRNN()
        model.train()
        assert model.training
        model.predict(_random_plate_batch(batch_size=2))
        assert model.training, "predict() left model in eval mode after being called mid-train"

    def test_predict_preserves_eval_state(self) -> None:
        """predict() called while model is in eval mode must leave it in eval mode."""
        model = PlateRecognizerCRNN()
        model.eval()
        model.predict(_random_plate_batch(batch_size=2))
        assert not model.training, "predict() switched model to train mode from eval"

    def test_eval_mode_deterministic(self) -> None:
        """Same input in eval mode must produce identical outputs (dropout disabled)."""
        model = PlateRecognizerCRNN()
        model.eval()
        x = _random_plate_batch(batch_size=2)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2), "eval mode gave different outputs for same input"

    def test_train_mode_nondeterministic(self) -> None:
        """
        Dropout should cause at least one differing forward pass in train mode.

        We run 10 passes and assert that not all are identical.  The chance of
        10 identical dropout masks with p=0.3 across a large BiLSTM is ~0.
        """
        model = PlateRecognizerCRNN()
        model.train()
        torch.manual_seed(0)
        x = _random_plate_batch(batch_size=2)
        outputs = [model(x).detach() for _ in range(10)]
        all_same = all(torch.allclose(outputs[0], o) for o in outputs[1:])
        assert not all_same, (
            "train mode produced identical outputs on 10 passes — dropout may be broken"
        )

    def test_custom_dropout_rate(self) -> None:
        """Constructor dropout parameter should be respected."""
        model = PlateRecognizerCRNN(dropout=0.95)
        model.train()
        x = _random_plate_batch(batch_size=4)
        outputs = [model(x).detach() for _ in range(5)]
        all_same = all(torch.allclose(outputs[0], o) for o in outputs[1:])
        assert not all_same

    def test_parameter_count_range(self) -> None:
        """
        Total trainable parameters should be in the range [5 M, 20 M].

        The BiLSTM with input_size=2048 dominates the parameter count.
        Too few params means a layer is missing; too many means an accidental
        architecture change.
        """
        model = PlateRecognizerCRNN()
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert 5_000_000 <= total <= 20_000_000, (
            f"Unexpected parameter count: {total:,}. Expected between 5 M and 20 M."
        )

    # ── decode_predictions tests ──────────────────────────────────────────────

    def test_decode_predictions_shape(self) -> None:
        """decode_predictions must return a list with one string per batch item."""
        model = PlateRecognizerCRNN()
        model.eval()
        with torch.no_grad():
            out = model(_random_plate_batch(batch_size=4))
        results = model.decode_predictions(out)
        assert isinstance(results, list)
        assert len(results) == 4

    def test_decode_predictions_removes_blank(self) -> None:
        """
        The CTC blank token (index 0) must never appear in the decoded output.

        We construct a (T, 1, C) tensor that is maximally confident for blank at
        every time step and verify the decoded string is empty.
        """
        model = PlateRecognizerCRNN()
        T, C = 16, VOCAB_SIZE
        # All log-probability mass on BLANK_IDX at every time step
        log_probs = torch.full((T, 1, C), fill_value=-1e9)
        log_probs[:, :, BLANK_IDX] = 0.0  # log(1) = 0 → this class wins
        results = model.decode_predictions(log_probs)
        assert results[0] == "", f"Expected empty string for all-blank input, got {results[0]!r}"

    def test_decode_predictions_blank_separates_identical_chars(self) -> None:
        """
        Two identical characters separated by a blank must decode to two chars.

        CTC spec: [A, blank, A] → "AA", NOT "A".
        The blank acts as a separator token that prevents collapsing across it.
        If this collapses to "A", the blank-separator logic is broken.
        """
        from apps.cv.training.dataset import CHAR_TO_IDX

        model = PlateRecognizerCRNN()
        T, C = 16, VOCAB_SIZE
        A_idx = CHAR_TO_IDX["A"]

        # Build: A, blank, A, blank, A, blank, A, blank, A, blank, A, blank × 6
        # Pattern: odd timesteps are blank, even are A → decodes to 8 × "A" = "AAAAAAAA"
        log_probs = torch.full((T, 1, C), fill_value=-1e9)
        for t in range(T):
            if t % 2 == 0:
                log_probs[t, 0, A_idx] = 0.0     # A
            else:
                log_probs[t, 0, BLANK_IDX] = 0.0  # blank separator

        result = model.decode_predictions(log_probs)[0]
        assert result == "A" * 8, (
            f"Expected 'AAAAAAAA' (blanks prevent collapse), got {result!r}"
        )

    def test_decode_predictions_collapse_repeats(self) -> None:
        """
        CTC greedy decode must collapse consecutive identical non-blank tokens.

        Example: [A, A, B, B] → "AB" (not "AABB").
        """
        from apps.cv.training.dataset import CHAR_TO_IDX

        model = PlateRecognizerCRNN()
        T, C = 16, VOCAB_SIZE
        A_idx = CHAR_TO_IDX["A"]
        B_idx = CHAR_TO_IDX["B"]

        # Build a sequence: A repeated 8 times, then B repeated 8 times
        log_probs = torch.full((T, 1, C), fill_value=-1e9)
        for t in range(8):
            log_probs[t, 0, A_idx] = 0.0
        for t in range(8, 16):
            log_probs[t, 0, B_idx] = 0.0

        result = model.decode_predictions(log_probs)[0]
        assert result == "AB", f"Expected 'AB' after collapsing repeats, got {result!r}"

    def test_decode_predictions_returns_strings(self) -> None:
        """Every element returned by decode_predictions must be a str."""
        model = PlateRecognizerCRNN()
        model.eval()
        with torch.no_grad():
            out = model(_random_plate_batch(batch_size=3))
        for s in model.decode_predictions(out):
            assert isinstance(s, str), f"decode_predictions returned non-str: {type(s)}"


@pytest.mark.unit
def test_train_recognizer_script_runs_with_documented_path() -> None:
    """
    The README command uses a file path, so --help must import cleanly.

    WHY this matters:
    Running ``python apps/cv/training/train_recognizer.py`` makes Python put
    apps/cv/training on sys.path instead of the repository root.  This test
    catches regressions where top-level ``apps.cv`` imports break that documented
    command before argument parsing can even display help.
    """
    repo_root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            sys.executable,
            "apps/cv/training/train_recognizer.py",
            "--help",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--data-dir" in result.stdout
