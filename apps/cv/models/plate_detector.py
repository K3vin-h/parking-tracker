"""
Plate detection CNN for the parking tracker CV pipeline.

PlateDetectorCNN is a lightweight convolutional network that takes a full
parking-lot image and predicts where the license plate is located.

Architecture overview
─────────────────────
Three convolutional blocks (conv + batch-norm + relu + max-pool) progressively
reduce the spatial size while learning increasingly abstract features.
AdaptiveAvgPool2d then collapses the spatial dimensions to a fixed 4×4 grid
regardless of the exact input resolution, so the fully-connected head always
receives the same input size.  Two dense layers compress those features into
four numbers: the plate's bounding box in YOLO format [cx, cy, w, h] where
all values are normalised to the range [0, 1] relative to the image dimensions.

Usage
─────
Training — call model(x) to get raw logits, compute SmoothL1Loss against
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

        # ── Spatial aggregation ────────────────────────────────────────────
        #
        # WHY AdaptiveAvgPool2d instead of a fixed flatten:
        # A plain flatten would tie the FC head to the exact input resolution.
        # AdaptiveAvgPool2d computes the pooling stride dynamically so that the
        # output is always 4×4 regardless of the input H and W.  This means the
        # network can handle images that are slightly different sizes (e.g. after
        # augmentation) and makes it easier to switch input resolutions without
        # rebuilding the model.
        #
        # 4×4 output preserves some spatial structure (vs global average pool
        # which collapses to 1×1) so the FC layers still know roughly where in
        # the image the plate-like features are concentrated.
        # Output: (B, 128, 4, 4) → flatten → (B, 2048)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        # ── Regression head ────────────────────────────────────────────────
        #
        # Two fully-connected layers compress 2048 features to 4 bbox values.

        # FC1: 2048 → 256  (major compression; most bounding-box information
        # can be captured in ~256 features)
        # WHY Dropout(0.3): Synthetic training data has limited visual variety.
        # Dropout randomly zeros 30 % of activations each forward pass, forcing
        # the network to not rely on any single feature too heavily.  This
        # reduces overfitting and improves generalisation to real plate images.
        self.fc1 = nn.Linear(2048, 256)
        self.relu_fc = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

        # FC2: 256 → 4  (output layer)
        # sigmoid is applied in forward() so the model always outputs [cx, cy, w, h]
        # in [0, 1].  This keeps the training-time and inference-time output spaces
        # identical — SmoothL1Loss trains against normalised [0,1] targets with
        # normalised [0,1] predictions, which is the correct regression setup.
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass and return raw bounding box logits.

        Args:
            x: Float32 tensor, shape (B, 3, H, W), pixel values in [0, 1].

        Returns:
            Tensor of shape (B, 4) — raw [cx, cy, w, h] logits.
            Apply torch.sigmoid before using for inference.
        """
        x = self.block1(x)   # (B, 32,  H/2, W/2)
        x = self.block2(x)   # (B, 64,  H/4, W/4)
        x = self.block3(x)   # (B, 128, H/8, W/8)
        x = self.pool(x)     # (B, 128, 4,   4)
        x = x.flatten(1)     # (B, 2048)
        x = self.fc1(x)      # (B, 256)
        x = self.relu_fc(x)
        x = self.dropout(x)
        x = self.fc2(x)           # (B, 4) — raw logits
        return torch.sigmoid(x)   # normalise to [0, 1] for consistent training + inference

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run inference without gradient tracking.

        forward() already applies sigmoid, so this is a thin wrapper that
        disables gradient computation for faster, lower-memory inference.
        Callers are responsible for setting eval mode (model.eval()) before
        calling predict() if they want deterministic outputs (no dropout).

        @torch.no_grad() disables gradient tracking — inference does not need
        gradients, and disabling them halves activation memory and speeds up
        the pass.

        Args:
            x: Float32 tensor, shape (B, 3, H, W), pixel values in [0, 1].

        Returns:
            Tensor of shape (B, 4) — normalised [cx, cy, w, h] in [0, 1].
        """
        return self.forward(x)
