"""
Shared plotting/training helpers for the detector and recognizer train scripts.

WHY THIS MODULE EXISTS:
  train_detector.py and train_recognizer.py rendered nearly identical dark-themed
  training-progress figures. The smoothing function, the project design-token
  palette, the batch→epoch x-mapping, the per-axis styling loop, and the save
  step were copy-pasted verbatim across both. Duplicated plotting code drifts:
  a tweak to the theme or a fix to the EMA had to be made twice. This module is
  the single home for the pieces that are genuinely identical; each script keeps
  only its own (divergent) per-panel plotting.

  These helpers are import-light on purpose — matplotlib is imported lazily
  inside import_pyplot() so merely importing this module (or the train scripts)
  does not pull in matplotlib.
"""

from pathlib import Path

# ── Project design tokens (shared with the dashboard CSS variables) ───────────
BG_PRIMARY = "#0f1117"
BG_SECONDARY = "#1a1d27"
BORDER = "#2e3039"
TEXT_PRIMARY = "#e4e4e7"
TEXT_MUTED = "#a1a1aa"
BLUE = "#3b82f6"
GREEN = "#22c55e"
YELLOW = "#eab308"
RED = "#ef4444"
PURPLE = "#a855f7"


def smooth(values: list[float], weight: float = 0.9) -> list[float]:
    """
    Exponential moving average — same algorithm TensorBoard uses by default.

    WHY smooth the batch loss curve: individual batch losses are noisy because
    each mini-batch is a random sample of the training set. An EMA with
    weight=0.9 keeps ~10 epochs of memory, revealing the true trend while
    preserving short-term variation.

    Args:
        values: Raw signal (typically per-batch training losses).
        weight: Smoothing factor in [0, 1). Higher = smoother, more lag.

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


def compute_batch_x(batches_per_epoch: list[int]) -> list[float]:
    """
    Map each batch index to a fractional epoch coordinate.

    WHY: the per-batch loss curve and the per-epoch metrics must share one x
    axis. Batch i of epoch ep is placed at ep-1 + (i+0.5)/n_batches so the batch
    curve lines up with the epoch markers.
    """
    batch_x: list[float] = []
    for ep, n_batches in enumerate(batches_per_epoch, start=1):
        for i in range(n_batches):
            batch_x.append(ep - 1.0 + (i + 0.5) / max(n_batches, 1))
    return batch_x


def import_pyplot():
    """
    Import matplotlib with the headless Agg backend and return (plt, gridspec).

    WHY Agg: it renders to memory/file without a display, which is essential for
    training runs in headless environments (Docker, CI, SSH). matplotlib.use()
    must be called before pyplot is imported, so this is done here in one place.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    return plt, gridspec


def style_axes(axes) -> None:
    """Apply the shared dark theme (facecolor, ticks, spines, grid) to each axis."""
    for ax in axes:
        ax.set_facecolor(BG_SECONDARY)
        ax.tick_params(colors=TEXT_MUTED, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        ax.grid(True, color=BORDER, linewidth=0.5, alpha=0.8)
        ax.yaxis.label.set_color(TEXT_PRIMARY)


def mark_best_epoch(axes, best_epoch: int) -> None:
    """Draw the shared dotted vertical 'best epoch' marker across every panel."""
    for ax in axes:
        ax.axvline(
            best_epoch, color=TEXT_MUTED, linewidth=0.9, linestyle=":", alpha=0.55
        )


def save_training_figure(fig, output_path: Path) -> Path:
    """
    Save the figure beside the weights file as '<stem>_training.png' and close it.

    Returns the PNG path. Closing the figure frees the renderer so repeated runs
    in one process do not leak memory.
    """
    plot_path = output_path.parent / (output_path.stem + "_training.png")
    fig.savefig(plot_path, dpi=150, facecolor=BG_PRIMARY, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return plot_path
