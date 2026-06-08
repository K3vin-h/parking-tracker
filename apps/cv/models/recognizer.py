"""
CRNN plate text recognizer for the parking tracker CV pipeline.

PlateRecognizerCRNN takes a cropped grayscale plate image (128×32) and
outputs a sequence of character log-probabilities suitable for CTC decoding.

Architecture overview
─────────────────────
Three convolutional blocks extract spatial features, progressively compressing
height while keeping width resolution intact.  The resulting feature maps are
reshaped so the width dimension becomes a sequence of time-steps.  A two-layer
bidirectional LSTM reads this sequence left-to-right and right-to-left, then a
linear projection maps each timestep to character log-probabilities.

    Input : (B, 1, 32, 128)  — grayscale plate crop, float32 in [0, 1]

    CNN Block 1 : Conv(1→64,  3×3) + BN + ReLU + MaxPool(2×2)  → (B, 64,  16, 64)
    CNN Block 2 : Conv(64→128,3×3) + BN + ReLU + MaxPool(2×2)  → (B, 128,  8, 32)
    CNN Block 3 : Conv(128→256,3×3)+ BN + ReLU + MaxPool(1×2)  → (B, 256,  8, 16)
    Reshape     : flatten C×H → sequence                         → (T=16, B, 2048)
    BiLSTM      : hidden=256, layers=2, bidirectional            → (T=16, B, 512)
    FC + log_softmax                                             → (T=16, B, 37)

    Output: (T=16, N, C=37)  — log-probabilities per time-step, CTC-ready.

Usage
─────
Training — pass output directly to CTCLoss (move to CPU first on MPS):

    log_probs = model(images)                    # (T, N, C) on device
    loss = ctc_loss(log_probs.cpu(), targets,    # CTCLoss requires CPU on MPS
                    input_lengths, target_lengths)

Inference — call model.predict(x) which handles eval-mode state internally:

    texts = model.decode_predictions(model.predict(image_tensor))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from apps.cv.training.dataset import BLANK_IDX, IDX_TO_CHAR, VOCAB_SIZE


class PlateRecognizerCRNN(nn.Module):
    """
    Convolutional-Recurrent network that reads text from a cropped plate image.

    The CNN backbone extracts spatial features; the BiLSTM models character
    ordering across the width of the plate; the FC layer maps each time-step
    to a probability distribution over the character vocabulary.

    Args:
        dropout: Dropout probability applied between LSTM layers and before the
                 output projection.  0.3 provides regularisation without
                 significantly slowing convergence on synthetic data.

    Input shape:  (B, 1, 32, 128) — batch of float32 grayscale plate crops,
                  pixel values in [0, 1].  Height 32 and width 128 must be exact;
                  use apps.cv.preprocessing.prepare_for_recognizer() to resize.

    Output shape: (T=16, N, C=37) — log-probabilities over VOCAB_SIZE=37 classes
                  (26 letters + 10 digits + 1 CTC blank at index 0) for each of
                  T=16 time-steps.  Pass directly to torch.nn.CTCLoss.
    """

    _DROPOUT: float = 0.3
    _SEQUENCE_LEN: int = 16   # width after CNN blocks — one timestep per column
    _LSTM_INPUT: int = 2048   # C=256 channels × H=8 rows flattened per column

    def __init__(self, dropout: float = _DROPOUT) -> None:
        super().__init__()

        # ── CNN backbone ───────────────────────────────────────────────────
        #
        # WHY three blocks with increasing channel depth (64→128→256):
        # Each block learns more abstract features — Block 1 sees edges and
        # stroke fragments, Block 2 combines them into partial characters,
        # Block 3 assembles those into character-level detectors.  Doubling
        # channels at each stage is the standard ResNet/VGG progression.
        #
        # WHY bias=False before BatchNorm: BatchNorm has a learnable beta
        # (shift) parameter that subsumes the conv bias, so the bias wastes
        # parameters and slows initialisation.

        # Block 1 — low-level features: edges, stroke fragments
        # Input : (B, 1, 32, 128)
        # Output: (B, 64, 16, 64)  — both spatial dims halved by MaxPool(2×2)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2 — mid-level features: partial characters, vertical strokes
        # Input : (B, 64, 16, 64)
        # Output: (B, 128, 8, 32)
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 3 — high-level features: character-level detectors
        # Input : (B, 128, 8, 32)
        # Output: (B, 256, 8, 16)
        #
        # WHY MaxPool((1, 2)) here instead of (2, 2):
        # Halving width from 32 → 16 gives exactly 16 time-steps for the LSTM
        # (one per horizontal position), which is enough to cover plates up to
        # 8 characters with 2 frames per character — sufficient for CTC alignment.
        # Keeping height=8 (not halving it) preserves vertical stroke detail
        # that helps distinguish similar characters like 'I' and '1'.
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )

        # ── Sequence model ─────────────────────────────────────────────────
        #
        # WHY Bidirectional LSTM:
        # Reading the plate left-to-right AND right-to-left simultaneously helps
        # resolve ambiguities: 'D' vs 'O' is easier when you know what letter
        # comes after it.  The outputs from both directions are concatenated,
        # so hidden_size=256 bidirectional produces 512-dimensional representations.
        #
        # WHY two layers: A single LSTM layer learns character-level patterns;
        # the second layer models relationships between adjacent characters (e.g.
        # it becomes unlikely to see 'X' followed by 'X' on a real plate).
        #
        # WHY LSTM over GRU: LSTM's separate forget/input/output gates handle the
        # longer effective sequences produced by padding better than GRU's two
        # gates, especially for 7-8 character plates where the early and late
        # context need to remain distinct.
        #
        # NOTE: the `dropout` parameter applies BETWEEN layers, not after the
        # final layer.  With num_layers=2 this fires once (between layer 1 and 2).
        self.lstm = nn.LSTM(
            input_size=self._LSTM_INPUT,   # 256 channels × 8 height = 2048
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            dropout=dropout,
            batch_first=False,             # input/output shape: (T, B, features)
        )

        # ── Output projection ──────────────────────────────────────────────
        #
        # WHY 512 input features: bidirectional LSTM concatenates forward and
        # backward hidden states → 256 × 2 = 512.
        # VOCAB_SIZE = 37: 26 letters + 10 digits + 1 CTC blank (index 0).
        self.fc = nn.Linear(512, VOCAB_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass and return log-probabilities over the character vocabulary.

        Args:
            x: Float32 tensor, shape (B, 1, 32, 128), pixel values in [0, 1].

        Returns:
            Tensor of shape (T=16, N, C=37) — log-softmax activated.
            Output is already in log-probability space.  Do NOT apply log_softmax
            again — double application silently corrupts CTCLoss by compressing
            probabilities a second time.

        Note:
            Pass the output to torch.nn.CTCLoss.  On MPS, move to CPU first:
            ``loss = ctc_criterion(out.cpu(), targets, input_lengths, target_lengths)``
        """
        B = x.size(0)

        # CNN feature extraction
        x = self.block1(x)   # (B, 64,  16, 64)
        x = self.block2(x)   # (B, 128,  8, 32)
        x = self.block3(x)   # (B, 256,  8, 16)

        # Reshape spatial map into a time sequence.
        # Each vertical column of the feature map becomes one time-step.
        # The 256 channels × 8 rows are flattened into a 2048-dim feature vector.
        # WHY this order (permute width to front): the LSTM reads one column at
        # a time, so the width dimension must be the sequence dimension (dim=0).
        # reshape() instead of view() handles non-contiguous tensors safely.
        # MaxPool2d may leave the tensor non-contiguous; view() would raise a
        # RuntimeError in that case, while reshape() calls contiguous() internally.
        x = x.reshape(B, self._LSTM_INPUT, self._SEQUENCE_LEN)  # (B, 2048, 16)
        x = x.permute(2, 0, 1)                               # (16, B, 2048)

        # Bidirectional LSTM — reads left-to-right and right-to-left
        x, _ = self.lstm(x)   # (16, B, 512)

        # Project each time-step to character log-probabilities
        x = self.fc(x)                         # (16, B, 37)
        return F.log_softmax(x, dim=-1)        # (T=16, N, C=37) — CTC-ready

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run deterministic inference without gradient tracking.

        Temporarily switches the model to eval mode so LSTM and dropout behave
        deterministically, runs a forward pass, then restores the original
        training/eval state.  This makes predict() safe to call at any point —
        mid-training callbacks, validation loops, or standalone inference —
        without side effects on the training loop's dropout behaviour.

        @torch.no_grad() disables gradient tracking for the duration of the call.

        Args:
            x: Float32 tensor, shape (B, 1, 32, 128), pixel values in [0, 1].

        Returns:
            Tensor of shape (T=16, N, C=37) — log-probabilities, same as forward().
        """
        was_training = self.training
        self.eval()
        try:
            return self.forward(x)
        finally:
            if was_training:
                self.train()

    def decode_predictions(self, output: torch.Tensor) -> list[str]:
        """
        Greedy CTC decode: convert log-probability output to plate text strings.

        Greedy decoding takes the most-likely character at each time-step (argmax),
        then applies CTC post-processing: collapse consecutive identical tokens and
        remove blank tokens (index 0).  This is not beam search — it is fast and
        sufficient for synthetic validation data.

        Args:
            output: Tensor of shape (T, N, C) — log-probabilities from forward()
                    or predict().  Does not need to be in log-space; argmax is
                    order-preserving, so softmax probabilities work equally well.

        Returns:
            List of N strings, one per batch item.  Plates with all time-steps
            predicted as blank return an empty string "".
        """
        # Greedy argmax over character dimension at each time step.
        # Shape: (T, N) — one predicted class index per timestep per sample.
        indices = output.argmax(dim=-1)  # (T, N)

        decoded: list[str] = []
        for n in range(indices.size(1)):
            seq = indices[:, n].tolist()   # List[int] of length T

            # CTC post-processing step 1: collapse consecutive identical tokens.
            # WHY: CTC alignment may predict the same character multiple times to
            # span several frames, e.g. [A, A, B] means the character 'A' was
            # spread over 2 frames, not that the plate reads "AAB".
            collapsed: list[int] = []
            for token in seq:
                if not collapsed or token != collapsed[-1]:
                    collapsed.append(token)

            # CTC post-processing step 2: remove blank tokens (BLANK_IDX = 0).
            # Blanks are used by CTC to separate repeated characters and to fill
            # frames between characters — they carry no text content.
            chars = [IDX_TO_CHAR[tok] for tok in collapsed if tok != BLANK_IDX]
            decoded.append("".join(chars))

        return decoded
