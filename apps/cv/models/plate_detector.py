"""
Plate detection CNN for the parking tracker CV pipeline.

PlateDetectorCNN is a convolutional network that takes a full parking-lot
image and predicts where the license plate is located.

Architecture overview
─────────────────────
Five convolutional blocks (conv + batch-norm + relu + max-pool) progressively
reduce the spatial size while learning increasingly abstract features.
AdaptiveAvgPool2d then collapses the spatial dimensions to a fixed 3×4 grid
regardless of the exact input resolution, so the fully-connected head always
receives the same input size. Two dense layers compress those features into
four numbers: the plate's bounding box in YOLO format [cx, cy, w, h] where
all values are normalised to the range [0, 1] relative to the image dimensions.

WHY FIVE BLOCKS, NOT THREE (2026-09-26 retrain): the original three-block
backbone gives each output cell a receptive field of roughly 22px on the
nominal 480×640 input, but a plate at realistic gate-camera scale spans
96–256px. The network could not "see" a whole plate at once from any single
position, which is a first-order explanation for the ~0.43 IoU measured
before this change (target: >0.70 — see docs/technical/01-cv-pipeline.md).
Two extra stride-2 blocks raise the receptive field to roughly 94px, without
changing the model's public input/output contract, so
apps/cv/pipeline.py and every existing weight-loading path continue to work
unchanged (old checkpoints simply no longer match this state dict and fail
closed via strict state-dictionary loading in pipeline.py — they are not
silently mismatched).

Usage
─────
Training — call model(x) to get normalized coordinates, compute a loss against
           normalised YOLO targets:
               loss = criterion(model(images), bboxes)

Inference — call model.predict(x) which applies sigmoid so outputs are
            guaranteed to be in [0, 1]:
               pred_box = model.predict(image_tensor.unsqueeze(0))
"""

import torch
import torch.nn as nn


class PlateDetectorCNN(nn.Module):
    """
    Convolutional network that predicts a license plate bounding box.

    Args:
        dropout: Dropout probability applied before the final output layer.
                 Higher values reduce overfitting on synthetic data but slow
                 convergence.  0.3 is a good starting point.

    Input shape:  (B, 3, H, W) — batch of float32 RGB images, pixel values
                  normalised to [0, 1].  The nominal training size is 480×640
                  (height × width) but AdaptiveAvgPool2d accepts any size.

    Output shape: (B, 4) — normalised [cx, cy, w, h] in [0, 1] (sigmoid applied
                  inside forward so training and inference share the same output space).
    """

    _DROPOUT: float = 0.3

    def __init__(self, dropout: float = _DROPOUT) -> None:
        super().__init__()

        # ── Convolutional backbone ─────────────────────────────────────────
        #
        # Each block follows the canonical pattern:
        #   Conv2d → BatchNorm2d → ReLU → MaxPool2d
        #
        # WHY BatchNorm after every conv: Normalises activations so the network
        # is less sensitive to weight initialisation and allows higher learning
        # rates.  It also acts as a mild regulariser, which helps when training
        # on synthetic data that has less variance than real images.
        #
        # WHY MaxPool(2×2): Halves the spatial dimensions after each block.
        # This gives the next layer a wider receptive field without needing
        # larger (and more expensive) kernels.

        # Block 1 — low-level features: edges, corners, colour gradients
        # Input:  (B, 3, H, W)
        # Output: (B, 32, H/2, W/2)  → 240×320 for the standard 480×640 input
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            # WHY bias=False with BatchNorm: BatchNorm already shifts activations
            # via its learnable beta parameter, so a separate conv bias is redundant
            # and wastes parameters.
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2 — mid-level features: shapes, rectangular outlines
        # Input:  (B, 32, H/2, W/2)
        # Output: (B, 64, H/4, W/4)  → 120×160
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 3 — high-level features: plate-like regions with internal text texture
        # Input:  (B, 64, H/4, W/4)
        # Output: (B, 128, H/8, W/8)  → 60×80
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 4 — wider receptive field: a plate-sized region is now visible
        # to a single output cell, not just a fragment of one.
        # Input:  (B, 128, H/8, W/8)
        # Output: (B, 256, H/16, W/16)  → 30×40
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 5 — scene-level context (where in the frame plate-like texture
        # sits relative to the rest of the parking lot).
        # Input:  (B, 256, H/16, W/16)
        # Output: (B, 256, H/32, W/32)  → 15×20
        self.block5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # ── Spatial aggregation ────────────────────────────────────────────
        #
        # WHY AdaptiveAvgPool2d instead of a fixed flatten:
        # A plain flatten would tie the FC head to the exact input resolution.
        # AdaptiveAvgPool2d computes the pooling stride dynamically so that the
        # output is always a fixed grid regardless of the input H and W.  This
        # means the network can handle images that are slightly different
        # sizes (e.g. after augmentation) and makes it easier to switch input
        # resolutions without rebuilding the model.
        #
        # 3×4 (matching the 3:4 aspect ratio of the 480×640 input) preserves
        # coarse spatial structure — the extra two
        # conv blocks already did most of the work of shrinking the spatial
        # size, so this pool only needs to trim the last bit of resolution,
        # not aggressively collapse it. 3×4 rather than a larger grid like 6×8
        # is deliberate: PyTorch's MPS backend requires the pooled output size
        # to evenly divide the actual input size at this point in the network
        # (15×20), and 3×4 divides it evenly (15/3=5, 20/4=5) — see
        # https://github.com/pytorch/pytorch/issues/96056.
        # Output: (B, 256, 3, 4) → flatten → (B, 3072)
        self.pool = nn.AdaptiveAvgPool2d((3, 4))

        # ── Regression head ────────────────────────────────────────────────
        #
        # Two fully-connected layers compress 3072 features to 4 bbox values.

        # FC1: 3072 → 256  (major compression; most bounding-box information
        # can be captured in ~256 features)
        # WHY Dropout(0.3): Synthetic training data has limited visual variety.
        # Dropout randomly zeros 30 % of activations each forward pass, forcing
        # the network to not rely on any single feature too heavily.  This
        # reduces overfitting and improves generalisation to real plate images.
        self.fc1 = nn.Linear(3072, 256)
        self.relu_fc = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

        # FC2: 256 → 4  (output layer)
        # sigmoid is applied in forward() so the model always outputs [cx, cy, w, h]
        # in [0, 1].  This keeps the training-time and inference-time output spaces
        # identical — the training loss compares normalised [0,1] targets with
        # normalised [0,1] predictions, which is the correct regression setup.
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass and return normalised bounding box predictions.

        Args:
            x: Float32 tensor, shape (B, 3, H, W), pixel values in [0, 1].

        Returns:
            Tensor of shape (B, 4) — sigmoid-activated [cx, cy, w, h] in [0, 1].
            Sigmoid is applied inside this method; do NOT apply it again externally.
        """
        x = self.block1(x)  # (B, 32,  H/2,  W/2)
        x = self.block2(x)  # (B, 64,  H/4,  W/4)
        x = self.block3(x)  # (B, 128, H/8,  W/8)
        x = self.block4(x)  # (B, 256, H/16, W/16)
        x = self.block5(x)  # (B, 256, H/32, W/32)
        x = self.pool(x)  # (B, 256, 3,    4)
        x = x.flatten(1)  # (B, 3072)
        x = self.fc1(x)  # (B, 256)
        x = self.relu_fc(x)
        x = self.dropout(x)
        x = self.fc2(x)  # (B, 4) — raw logits
        return torch.sigmoid(
            x
        )  # normalise to [0, 1] for consistent training + inference

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run deterministic inference without gradient tracking.

        Temporarily switches the model to eval mode so dropout is disabled,
        runs a forward pass, then restores the original training/eval state.
        This makes predict() safe to call at any point — mid-training callbacks,
        validation loops, or standalone inference — without side effects on the
        training loop's dropout behaviour.

        @torch.no_grad() disables gradient tracking — inference does not need
        gradients, and disabling them halves activation memory and speeds up
        the pass.

        Args:
            x: Float32 tensor, shape (B, 3, H, W), pixel values in [0, 1].

        Returns:
            Tensor of shape (B, 4) — normalised [cx, cy, w, h] in [0, 1].
        """
        was_training = self.training
        self.eval()
        try:
            return self.forward(x)
        finally:
            if was_training:
                self.train()
