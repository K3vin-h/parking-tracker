"""
Training script for PlateDetectorCNN.

Runs outside of Docker — uses the local Python environment with PyTorch
installed.  On Apple Silicon, the MPS backend is used automatically.

Typical usage
─────────────
    # Generate synthetic data first (requires data/backgrounds/):
    python -c "
    from apps.cv.training.synthetic_data import generate_detector_dataset
    generate_detector_dataset('data/backgrounds', 'data/detector', n=10000)
    "

    # Train:
    python apps/cv/training/train_detector.py \\
        --data-dir data/detector \\
        --epochs 50 \\
        --output apps/cv/weights/detector.pth

    # Smoke test (fast, 10-sample dataset):
    python apps/cv/training/train_detector.py \\
        --data-dir data/detector_smoke \\
        --epochs 2 \\
        --batch-size 4 \\
        --output /tmp/detector_smoke.pth

The script saves the best weights (lowest validation loss) to --output and
prints one summary line per epoch.

IoU target: > 0.7 on the validation split after 50 epochs of synthetic data.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from apps.cv.models.plate_detector import PlateDetectorCNN
from apps.cv.training.dataset import PlateDetectorDataset
from apps.cv.utils.device import get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── IoU helper ────────────────────────────────────────────────────────────────

def _compute_batch_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Compute mean Intersection-over-Union for a batch of YOLO bounding boxes.

    WHY IoU matters for detector evaluation:
    SmoothL1Loss tells you the average coordinate error, but it does not
    directly reflect how much the predicted and ground-truth boxes overlap.
    IoU normalises overlap by the combined area, giving a value in [0, 1]
    that is easy to interpret: 1.0 = perfect match, 0.0 = no overlap.

    YOLO format: [cx, cy, w, h] where all values are normalised to [0, 1]
    relative to the image dimensions.  We first convert to corner format
    [x1, y1, x2, y2] because intersection area is easier to compute that way.

    Args:
        pred:   (B, 4) float32 tensor — predicted [cx, cy, w, h] in [0, 1].
        target: (B, 4) float32 tensor — ground-truth [cx, cy, w, h] in [0, 1].

    Returns:
        Scalar tensor — mean IoU over the batch.
    """
    # Clamp predictions to valid coordinate range as a defensive measure.
    # PlateDetectorCNN.forward() already applies sigmoid so values should
    # already be in [0, 1], but floating-point rounding can produce values
    # marginally outside this range.  Clamping costs nothing and prevents
    # negative areas from corrupting the IoU calculation.
    pred = pred.clamp(0.0, 1.0)

    # Convert YOLO centre format to corner format for both pred and target.
    # [cx, cy, w, h] → [x1, y1, x2, y2]
    # x1 = cx - w/2,  x2 = cx + w/2
    # y1 = cy - h/2,  y2 = cy + h/2
    pred_x1 = pred[:, 0] - pred[:, 2] / 2
    pred_y1 = pred[:, 1] - pred[:, 3] / 2
    pred_x2 = pred[:, 0] + pred[:, 2] / 2
    pred_y2 = pred[:, 1] + pred[:, 3] / 2

    tgt_x1 = target[:, 0] - target[:, 2] / 2
    tgt_y1 = target[:, 1] - target[:, 3] / 2
    tgt_x2 = target[:, 0] + target[:, 2] / 2
    tgt_y2 = target[:, 1] + target[:, 3] / 2

    # Intersection rectangle: take the inner corners.
    # torch.maximum/minimum operate element-wise and handle batches correctly.
    inter_x1 = torch.maximum(pred_x1, tgt_x1)
    inter_y1 = torch.maximum(pred_y1, tgt_y1)
    inter_x2 = torch.minimum(pred_x2, tgt_x2)
    inter_y2 = torch.minimum(pred_y2, tgt_y2)

    # Clamp to zero: if the boxes do not overlap, one dimension of the
    # intersection is negative; clamp makes it zero so area = 0.
    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    inter_area = inter_w * inter_h  # (B,)

    # Individual areas
    pred_area   = (pred_x2 - pred_x1).clamp(min=0.0) * (pred_y2 - pred_y1).clamp(min=0.0)
    target_area = (tgt_x2 - tgt_x1).clamp(min=0.0) * (tgt_y2 - tgt_y1).clamp(min=0.0)

    # Union = sum of areas minus the shared intersection (avoid double-counting).
    # Add a small epsilon to prevent division by zero when both boxes are
    # degenerate (zero area) — can happen early in training with random weights.
    union_area = pred_area + target_area - inter_area + 1e-7

    iou = inter_area / union_area  # (B,)
    return iou.mean()


# ── Training and validation loops ────────────────────────────────────────────

def _train_epoch(
    model: PlateDetectorCNN,
    loader: DataLoader,
    criterion: nn.SmoothL1Loss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Run one full pass over the training set and return the mean loss.

    WHY SmoothL1Loss (Huber loss) for bounding box regression:
    MSE penalises large errors quadratically — a single bad prediction (e.g.
    very early in training when weights are random) creates an enormous gradient
    that destabilises the whole network.  L1 penalises all errors linearly so
    outliers have less impact, but its gradient is constant even near zero,
    which makes convergence choppy.  SmoothL1 (Huber) combines both: it behaves
    like L2 (smooth gradient) for small errors and L1 (robust) for large ones.
    The beta parameter controls the boundary between the two regimes.
    """
    model.train()
    total_loss = 0.0

    for images, bboxes in loader:
        images = images.to(device)
        bboxes = bboxes.to(device)

        optimizer.zero_grad()
        preds = model(images)           # raw logits (B, 4)
        loss  = criterion(preds, bboxes)
        loss.backward()

        # Gradient clipping prevents extremely large weight updates from
        # destabilising training early on when the model is far from convergence.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)  # type: ignore[arg-type]


@torch.no_grad()
def _validate_epoch(
    model: PlateDetectorCNN,
    loader: DataLoader,
    criterion: nn.SmoothL1Loss,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluate the model on the validation set.

    Returns:
        Tuple of (mean_val_loss, mean_iou).
    """
    model.eval()
    total_loss = 0.0
    total_iou  = 0.0

    for images, bboxes in loader:
        images = images.to(device)
        bboxes = bboxes.to(device)

        preds = model(images)
        loss  = criterion(preds, bboxes)
        iou   = _compute_batch_iou(preds.cpu(), bboxes.cpu())

        total_loss += loss.item() * images.size(0)
        total_iou  += iou.item() * images.size(0)

    n = len(loader.dataset)  # type: ignore[arg-type]
    return total_loss / n, total_iou / n


# ── Entry point ───────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PlateDetectorCNN on a synthetic detector dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Root of the detector dataset (must contain images/ and labels/).",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="DataLoader batch size.",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Initial Adam learning rate.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("apps/cv/weights/detector.pth"),
        help="Destination path for the best model weights.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for the train/val split (reproducibility).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # ── Reproducibility ────────────────────────────────────────────────────
    torch.manual_seed(args.seed)

    device = get_device()
    logger.info("Using device: %s", device)

    # ── Data ───────────────────────────────────────────────────────────────
    dataset = PlateDetectorDataset(args.data_dir)
    logger.info("Dataset size: %d samples", len(dataset))

    # 80/20 train/val split.
    # WHY float split (requires PyTorch ≥ 2.0): cleaner than computing integer
    # counts manually, and automatically handles odd-sized datasets.
    rng = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(dataset, [0.8, 0.2], generator=rng)
    logger.info("Train: %d  |  Val: %d", len(train_set), len(val_set))

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # 0 avoids multiprocessing issues with MPS on macOS
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # ── Model, loss, optimiser ─────────────────────────────────────────────
    model = PlateDetectorCNN().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %d", total_params)

    # SmoothL1Loss with beta=1.0 (the default Huber threshold).
    # PlateDetectorCNN.forward() applies sigmoid internally, so both preds
    # and targets are in [0, 1] — the loss and the IoU metric are computed in
    # the same space as inference, so val IoU is a reliable quality signal.
    # Using reduction='mean' averages over all 4 coordinate dimensions AND
    # over the batch — gives a loss value in the same range as a single
    # coordinate error, which is easier to interpret and compare across runs.
    criterion = nn.SmoothL1Loss(beta=1.0, reduction="mean")

    # WHY Adam: Adaptive per-parameter learning rates handle sparse gradients
    # well and require minimal hyperparameter tuning compared to SGD.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # WHY ReduceLROnPlateau: Reduces the learning rate by 0.5 when validation
    # loss plateaus for 5 consecutive epochs.  This allows aggressive initial
    # learning and fine-grained convergence later without manual scheduling.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    # ── Training loop ──────────────────────────────────────────────────────
    best_val_loss = float("inf")

    # Resolve the output path and bound it to the project root so a crafted
    # --output like ../../etc/cron.d/payload cannot create directories outside
    # the repo tree (relevant when this script runs in CI or shared environments).
    output_path  = args.output.resolve()
    _repo_root   = Path(__file__).resolve().parents[3]
    if not output_path.is_relative_to(_repo_root):
        raise SystemExit(
            f"--output path escapes project root.\n"
            f"  Output:       {output_path}\n"
            f"  Project root: {_repo_root}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Starting training for %d epochs …", args.epochs)
    logger.info("%-6s  %-12s  %-12s  %-10s  %-10s", "Epoch", "Train Loss", "Val Loss", "Val IoU", "LR")

    for epoch in range(1, args.epochs + 1):
        train_loss             = _train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_iou      = _validate_epoch(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            "%-6d  %-12.6f  %-12.6f  %-10.4f  %-10.6f",
            epoch, train_loss, val_loss, val_iou, current_lr,
        )

        # Save the best model by validation loss.
        # WHY save only the state_dict and not the full model: state_dict is
        # architecture-independent — it contains only the weights.  Loading it
        # requires reconstructing the model in code, which is explicit about
        # what class is being loaded and avoids pickle-based class loading.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_path)
            logger.info("  ↳ New best (val_loss=%.6f) → saved to %s", best_val_loss, output_path)

    logger.info("Training complete. Best val loss: %.6f  |  Weights: %s", best_val_loss, output_path)
    logger.info("Load with: model.load_state_dict(torch.load(%r, weights_only=True))", str(output_path))


if __name__ == "__main__":
    sys.exit(main())
