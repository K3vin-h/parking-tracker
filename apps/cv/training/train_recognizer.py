"""
Training script for PlateRecognizerCRNN.

Runs outside of Docker — uses the local Python environment with PyTorch
installed.  On Apple Silicon, the MPS backend is used automatically for
the model; CTCLoss runs on CPU regardless (PyTorch MPS does not support it).

Typical usage
─────────────
    # Generate synthetic recognizer data first:
    python -c "
    from apps.cv.training.synthetic_data import generate_recognizer_dataset
    generate_recognizer_dataset(n=5000, output_dir='data/recognizer')
    "

    # Train:
    python apps/cv/training/train_recognizer.py \\
        --data-dir data/recognizer \\
        --epochs 100 \\
        --output apps/cv/weights/recognizer.pth

    # Smoke test (fast, tiny dataset):
    python apps/cv/training/train_recognizer.py \\
        --data-dir data/recognizer_smoke \\
        --epochs 2 \\
        --batch-size 4 \\
        --output apps/cv/weights/recognizer_smoke.pth

The script saves the best weights (lowest validation loss) to --output and
prints one summary line per epoch.

Accuracy targets: >90 % character accuracy, >80 % full-plate exact match
after 100 epochs on synthetic validation data.
"""

import argparse
import logging
import sys
from pathlib import Path

# The documented invocation runs this file directly:
#     python apps/cv/training/train_recognizer.py ...
# In that mode Python places apps/cv/training on sys.path, not the repository
# root, so absolute imports such as apps.cv.models.recognizer would fail before
# main() can run.  Insert the root before importing project modules so the CLI
# works exactly as documented.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from apps.cv.models.recognizer import PlateRecognizerCRNN
from apps.cv.training.dataset import (
    IDX_TO_CHAR,
    PlateRecognizerDataset,
    ctc_collate_fn,
)
from apps.cv.utils.device import get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Training helpers ──────────────────────────────────────────────────────────

def _train_epoch(
    model: PlateRecognizerCRNN,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, list[float]]:
    """
    Run one training epoch and return mean loss + per-batch losses.

    WHY CTCLoss on CPU even when device is MPS:
    PyTorch's MPS backend does not implement CTCLoss as of PyTorch 2.x.
    Moving log_probs to CPU before the loss computation is the standard
    workaround — gradients still flow back through the MPS graph correctly
    because autograd tracks the .cpu() call as part of the compute graph.

    Args:
        model:     The CRNN model.
        loader:    DataLoader using ctc_collate_fn (yields dicts, not tuples).
        criterion: nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True).
        optimizer: Adam optimiser.
        device:    Target device for model inputs (MPS / CUDA / CPU).

    Returns:
        (mean_epoch_loss, per_batch_losses) — per_batch_losses has one entry
        per DataLoader iteration, enabling batch-resolution loss plotting.
    """
    model.train()
    total_loss = 0.0
    batch_losses: list[float] = []

    for batch in loader:
        # ctc_collate_fn returns a dict — not a (images, labels) tuple
        images         = batch["images"].to(device)
        # targets and target_lengths stay on CPU — CTCLoss always requires CPU tensors
        targets        = batch["targets"]
        target_lengths = batch["target_lengths"]

        optimizer.zero_grad()

        log_probs = model(images)              # (T=16, N, C=37) on device
        T, N, _   = log_probs.shape

        # input_lengths: every sample in this batch has the full sequence length T
        input_lengths = torch.full((N,), T, dtype=torch.long)

        # Move log_probs to CPU for CTCLoss — MPS does not support this op
        loss = criterion(log_probs.cpu(), targets, input_lengths, target_lengths)
        loss.backward()

        # Gradient clipping at 5.0 (higher than detector's 1.0 because CTC
        # gradients can be significantly larger, especially early in training
        # when the network produces near-uniform distributions).
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()
        batch_losses.append(loss.item())
        total_loss += loss.item() * N

    mean_loss = total_loss / len(loader.dataset)
    return mean_loss, batch_losses


def _levenshtein(a: str, b: str) -> int:
    """
    Edit distance between two strings (substitutions, insertions, deletions).

    WHY hand-rolled: avoids a third-party dependency for ~15 lines of
    textbook dynamic programming.  Plate strings are ≤ 8 characters, so the
    O(len(a)·len(b)) cost is negligible even across a full validation set.
    """
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            curr.append(min(
                prev[j] + 1,                      # deletion
                curr[j - 1] + 1,                  # insertion
                prev[j - 1] + (ca != cb),         # substitution / match
            ))
        prev = curr
    return prev[-1]


@torch.no_grad()
def _validate_epoch(
    model: PlateRecognizerCRNN,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    device: torch.device,
) -> tuple[float, float, float]:
    """
    Evaluate the model on the validation set.

    Args:
        model:     The CRNN model (will be temporarily set to eval mode).
        loader:    Validation DataLoader (ctc_collate_fn).
        criterion: nn.CTCLoss.
        device:    Target device for model inputs.

    Returns:
        (mean_val_loss, char_accuracy, plate_accuracy)
        - char_accuracy:  1 − character error rate (edit distance / gt length),
                          clamped to [0, 1]
        - plate_accuracy: fraction of plates where every character is correct
    """
    model.eval()
    total_loss      = 0.0
    edit_errors     = 0
    total_chars     = 0
    exact_matches   = 0
    total_plates    = 0

    for batch in loader:
        images         = batch["images"].to(device)
        targets        = batch["targets"]
        target_lengths = batch["target_lengths"]

        log_probs     = model(images)
        T, N, _       = log_probs.shape
        input_lengths = torch.full((N,), T, dtype=torch.long)

        loss = criterion(log_probs.cpu(), targets, input_lengths, target_lengths)
        total_loss += loss.item() * N

        # Decode predicted strings
        pred_texts = model.decode_predictions(log_probs)

        # Reconstruct ground-truth strings by splitting the 1-D targets tensor.
        # WHY torch.split: ctc_collate_fn concatenates all label sequences into
        # a single 1-D tensor; target_lengths records how many indices belong to
        # each sample, so split() reverses this concatenation.
        gt_seqs   = torch.split(targets, target_lengths.tolist())
        gt_texts  = ["".join(IDX_TO_CHAR[i.item()] for i in seq) for seq in gt_seqs]

        for pred, gt in zip(pred_texts, gt_texts):
            total_plates += 1
            if pred == gt:
                exact_matches += 1

            # Character-level accuracy via edit distance (1 − CER).
            # WHY not positional zip() matching: a single shifted character
            # (e.g. pred="XABC12" vs gt="ABC123") makes every position wrong
            # under zip even though 5 of 6 characters are present — the metric
            # would report ~0 % for a near-correct prediction.  Levenshtein
            # distance counts the true number of substitutions / insertions /
            # deletions, so the reported accuracy tracks model quality.
            total_chars += max(len(gt), 1)
            edit_errors += _levenshtein(pred, gt)

    n = len(loader.dataset)
    # Insertions can push edit distance past len(gt); clamp so accuracy ≥ 0.
    char_acc  = max(0.0, 1.0 - edit_errors / max(total_chars, 1))
    plate_acc = exact_matches / max(total_plates, 1)
    return total_loss / n, char_acc, plate_acc


# ── Training-curve helpers ────────────────────────────────────────────────────

def _smooth(values: list[float], weight: float = 0.9) -> list[float]:
    """
    Exponential moving average — same algorithm TensorBoard uses by default.

    Args:
        values: Raw signal (typically per-batch training losses).
        weight: Smoothing factor in [0, 1).  Higher = smoother, more lag.

    Returns:
        List of smoothed values, same length as input.
    """
    if not values:
        return []
    smoothed = []
    last = values[0]
    for v in values:
        last = weight * last + (1.0 - weight) * v
        smoothed.append(last)
    return smoothed


def _plot_training_history(
    history: dict[str, list],
    output_path: Path,
    best_epoch: int,
) -> Path:
    """
    Render and save a dark-themed 4-panel training-progress figure.

    Four stacked rows:
      1. Loss         — per-batch train loss (faint) + EMA (bold) + val markers
      2. Char acc     — validation character accuracy + dashed target at 0.90
      3. Plate acc    — validation exact-match accuracy + dashed target at 0.80
      4. LR           — learning rate as a step plot

    Args:
        history:     Dict with keys train_loss_batch, batches_per_epoch,
                     val_loss, val_char_acc, val_plate_acc, lr.
        output_path: Path to the .pth weights file; PNG saved alongside it.
        best_epoch:  1-based epoch index of the lowest validation loss.

    Returns:
        Path to the saved PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # ── Project design tokens ──────────────────────────────────────────────
    BG_PRIMARY   = "#0f1117"
    BG_SECONDARY = "#1a1d27"
    BORDER       = "#2e3039"
    TEXT_PRIMARY = "#e4e4e7"
    TEXT_MUTED   = "#a1a1aa"
    BLUE         = "#3b82f6"
    GREEN        = "#22c55e"
    YELLOW       = "#eab308"
    RED          = "#ef4444"
    PURPLE       = "#a855f7"

    epochs           = list(range(1, len(history["val_loss"]) + 1))
    n_epochs         = len(epochs)
    batch_losses     = history["train_loss_batch"]
    batches_per_epoch = history["batches_per_epoch"]

    # Map each batch to a fractional epoch coordinate
    batch_x: list[float] = []
    for ep, n_batches in enumerate(batches_per_epoch, start=1):
        for i in range(n_batches):
            batch_x.append(ep - 1.0 + (i + 0.5) / max(n_batches, 1))

    smoothed = _smooth(batch_losses)

    # ── Figure layout — 4 rows ─────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 11), facecolor=BG_PRIMARY)
    gs  = gridspec.GridSpec(
        4, 1, figure=fig,
        hspace=0.06, top=0.92, bottom=0.06, left=0.08, right=0.97,
        height_ratios=[2.2, 1.2, 1.2, 0.8],
    )
    ax_loss  = fig.add_subplot(gs[0])
    ax_char  = fig.add_subplot(gs[1], sharex=ax_loss)
    ax_plate = fig.add_subplot(gs[2], sharex=ax_loss)
    ax_lr    = fig.add_subplot(gs[3], sharex=ax_loss)

    for ax in (ax_loss, ax_char, ax_plate, ax_lr):
        ax.set_facecolor(BG_SECONDARY)
        ax.tick_params(colors=TEXT_MUTED, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        ax.grid(True, color=BORDER, linewidth=0.5, alpha=0.8)
        ax.yaxis.label.set_color(TEXT_PRIMARY)

    plt.setp(ax_loss.get_xticklabels(),  visible=False)
    plt.setp(ax_char.get_xticklabels(),  visible=False)
    plt.setp(ax_plate.get_xticklabels(), visible=False)
    ax_lr.tick_params(axis="x", colors=TEXT_MUTED, labelsize=9)
    ax_lr.set_xlabel("Epoch", color=TEXT_PRIMARY, fontsize=10)

    # Best-epoch vertical marker across all panels
    for ax in (ax_loss, ax_char, ax_plate, ax_lr):
        ax.axvline(best_epoch, color=TEXT_MUTED, linewidth=0.9, linestyle=":", alpha=0.55)

    # ── Row 1: Loss ────────────────────────────────────────────────────────
    ax_loss.plot(batch_x, batch_losses, color=BLUE, linewidth=0.4, alpha=0.18)
    ax_loss.plot(batch_x, smoothed,     color=BLUE, linewidth=1.8, label="Train loss (smoothed)")
    ax_loss.plot(
        epochs, history["val_loss"],
        color=GREEN, linewidth=1.6,
        marker="o", markersize=5,
        markerfacecolor=BG_PRIMARY, markeredgewidth=1.8,
        label="Val loss",
    )
    ax_loss.set_ylabel("Loss", color=TEXT_PRIMARY, fontsize=10)
    ax_loss.legend(
        facecolor=BG_SECONDARY, edgecolor=BORDER,
        labelcolor=TEXT_PRIMARY, fontsize=9, loc="upper right",
    )
    best_val = history["val_loss"][best_epoch - 1]
    x_offset = max(0.8, n_epochs * 0.04)
    ax_loss.annotate(
        f"best  epoch {best_epoch}\n{best_val:.4f}",
        xy=(best_epoch, best_val),
        xytext=(best_epoch + x_offset, best_val),
        color=TEXT_MUTED, fontsize=8,
        arrowprops=dict(arrowstyle="->", color=TEXT_MUTED, lw=0.9),
        va="center",
    )

    # ── Row 2: Character accuracy ──────────────────────────────────────────
    ax_char.axhline(0.90, color=RED, linewidth=1.1, linestyle="--", alpha=0.75, label="Target  90 %")
    ax_char.plot(
        epochs, history["val_char_acc"],
        color=YELLOW, linewidth=1.6,
        marker="s", markersize=4,
        markerfacecolor=BG_PRIMARY, markeredgewidth=1.8,
        label="Char accuracy",
    )
    ax_char.set_ylim(-0.05, 1.08)
    ax_char.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_char.set_ylabel("Char Acc", color=TEXT_PRIMARY, fontsize=10)
    ax_char.legend(
        facecolor=BG_SECONDARY, edgecolor=BORDER,
        labelcolor=TEXT_PRIMARY, fontsize=9, loc="lower right",
    )

    # ── Row 3: Plate (exact-match) accuracy ───────────────────────────────
    ax_plate.axhline(0.80, color=RED, linewidth=1.1, linestyle="--", alpha=0.75, label="Target  80 %")
    ax_plate.plot(
        epochs, history["val_plate_acc"],
        color=PURPLE, linewidth=1.6,
        marker="^", markersize=4,
        markerfacecolor=BG_PRIMARY, markeredgewidth=1.8,
        label="Plate accuracy",
    )
    ax_plate.set_ylim(-0.05, 1.08)
    ax_plate.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_plate.set_ylabel("Plate Acc", color=TEXT_PRIMARY, fontsize=10)
    ax_plate.legend(
        facecolor=BG_SECONDARY, edgecolor=BORDER,
        labelcolor=TEXT_PRIMARY, fontsize=9, loc="lower right",
    )

    # ── Row 4: Learning rate ───────────────────────────────────────────────
    ax_lr.step(
        epochs, history["lr"],
        color=TEXT_MUTED, linewidth=1.3, where="post",
        label="Learning rate",
    )
    ax_lr.set_ylabel("LR", color=TEXT_PRIMARY, fontsize=10)
    ax_lr.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0e}"))
    ax_lr.legend(
        facecolor=BG_SECONDARY, edgecolor=BORDER,
        labelcolor=TEXT_PRIMARY, fontsize=9,
    )

    # ── Title and save ─────────────────────────────────────────────────────
    fig.suptitle(
        "PlateRecognizerCRNN — Training Progress",
        color=TEXT_PRIMARY, fontsize=13, fontweight="bold", y=0.965,
    )
    plot_path = output_path.parent / (output_path.stem + "_training.png")
    fig.savefig(plot_path, dpi=150, facecolor=BG_PRIMARY, bbox_inches="tight")
    plt.close(fig)
    return plot_path


# ── Entry point ───────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PlateRecognizerCRNN on a synthetic recognizer dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Root of the recognizer dataset (must contain images/ and labels.csv).",
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="DataLoader batch size.",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Initial Adam learning rate.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("apps/cv/weights/recognizer.pth"),
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
    if device.type == "mps":
        logger.info("MPS device detected — CTCLoss will run on CPU (MPS limitation)")

    # ── Data ───────────────────────────────────────────────────────────────
    dataset = PlateRecognizerDataset(args.data_dir)
    logger.info("Dataset size: %d samples", len(dataset))

    rng = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(dataset, [0.8, 0.2], generator=rng)
    logger.info("Train: %d  |  Val: %d", len(train_set), len(val_set))

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # 0 avoids multiprocessing issues with MPS on macOS
        pin_memory=(device.type == "cuda"),
        collate_fn=ctc_collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=ctc_collate_fn,
    )

    # ── Model, loss, optimiser ─────────────────────────────────────────────
    model = PlateRecognizerCRNN().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %d", total_params)

    # WHY zero_infinity=True: early in training the network may produce near-zero
    # probabilities for some classes, leading to -inf log-probs and infinite loss.
    # zero_infinity replaces these with 0 and suppresses their gradients, allowing
    # training to recover rather than diverging immediately.
    criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    # ── Output path validation ─────────────────────────────────────────────
    # Bound --output to the project root to prevent path-traversal writes
    # (e.g. --output ../../../../etc/cron.d/payload) in CI or shared environments.
    output_path = args.output.resolve()
    if not output_path.is_relative_to(REPO_ROOT):
        raise SystemExit(
            f"--output path escapes project root.\n"
            f"  Output:       {output_path}\n"
            f"  Project root: {REPO_ROOT}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Training loop ──────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_epoch    = 1

    history: dict[str, list] = {
        "train_loss_epoch": [],
        "train_loss_batch": [],
        "batches_per_epoch": [],
        "val_loss": [],
        "val_char_acc": [],
        "val_plate_acc": [],
        "lr": [],
    }

    logger.info("Starting training for %d epochs …", args.epochs)
    logger.info(
        "%-6s  %-12s  %-12s  %-10s  %-10s  %-10s",
        "Epoch", "Train Loss", "Val Loss", "Char Acc", "Plate Acc", "LR",
    )

    for epoch in range(1, args.epochs + 1):
        train_loss, batch_losses          = _train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_char_acc, val_plate_acc = _validate_epoch(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss_epoch"].append(train_loss)
        history["train_loss_batch"].extend(batch_losses)
        history["batches_per_epoch"].append(len(batch_losses))
        history["val_loss"].append(val_loss)
        history["val_char_acc"].append(val_char_acc)
        history["val_plate_acc"].append(val_plate_acc)
        history["lr"].append(current_lr)

        logger.info(
            "%-6d  %-12.6f  %-12.6f  %-10.4f  %-10.4f  %-10.6f",
            epoch, train_loss, val_loss, val_char_acc, val_plate_acc, current_lr,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            torch.save(model.state_dict(), output_path)
            logger.info("  ↳ New best (val_loss=%.6f) → saved to %s", best_val_loss, output_path)

    logger.info("Training complete. Best val loss: %.6f  |  Weights: %s", best_val_loss, output_path)
    logger.info("Load with: model.load_state_dict(torch.load(%r, weights_only=True))", str(output_path))

    # ── Training curve ─────────────────────────────────────────────────────
    # WHY two separate guards: bundling plot-save and viewer-launch in one
    # try block meant a Popen failure (e.g. `open` does not exist on Linux)
    # was reported as "Could not save training curve" even though the PNG
    # saved fine — and vice versa, a savefig failure after the info log left
    # operators believing a file existed that was never written.
    plot_path = None
    try:
        plot_path = _plot_training_history(history, output_path, best_epoch)
        logger.info("Training curve → %s", plot_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not save training curve: %s", exc)

    # `open` is macOS-only; on Linux/CI skip the viewer instead of failing.
    if plot_path is not None and sys.platform == "darwin":
        import subprocess
        try:
            subprocess.Popen(["open", str(plot_path)])
        except OSError as exc:
            logger.warning("Could not open training curve viewer: %s", exc)


if __name__ == "__main__":
    sys.exit(main())
