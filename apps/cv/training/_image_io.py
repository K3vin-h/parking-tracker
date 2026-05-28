"""
Hardened Pillow image loading for training-data ingestion.

Both ``synthetic_data.py`` and ``dataset.py`` open image files supplied by the
operator (backgrounds, generated detector images, recognizer crops). A bare
``Image.open()`` leaves three holes that ``apps/cv/preprocessing.py`` already
closes for upload-time validation:

  1. Decompression-bomb DoS — Pillow only emits ``DecompressionBombWarning``
     by default; a worker process that swallows the warning will happily
     allocate gigabytes when fed an attacker-crafted PNG.
  2. Exotic / legacy formats — Pillow's parsers for EPS, PSD, SVG, BMP etc.
     have a richer history of CVEs than the camera formats this pipeline
     actually consumes (JPEG / PNG / WEBP).
  3. Pixel-cap bypass — ``Image.MAX_IMAGE_PIXELS`` defaults to ~178 MP,
     which is well above anything a real parking camera produces.

This helper mirrors the defense-in-depth pattern from ``preprocessing.py`` but
skips the ``MEDIA_ROOT`` containment check, because training data lives
outside ``MEDIA_ROOT`` (and there is no Django settings module available
when training scripts run standalone).
"""

import warnings
from pathlib import Path

from PIL import Image, UnidentifiedImageError

# 12 MP matches the upload-side limit in apps/cv/preprocessing.py. Keeping the
# two limits identical means an image small enough to upload is also small
# enough to feed into a training dataset, which avoids surprise divergence in
# the dev workflow (e.g. an asset that worked in tests but fails at inference).
MAX_IMAGE_PIXELS = 4000 * 3000

# Lower Pillow's global pixel cap to our limit so it is enforced natively
# inside Image.open() during header parsing, independent of the explicit
# warning-to-error promotion below. Idempotent: preprocessing.py sets the
# same value, so importing both modules in either order is safe.
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

# Camera uploads are always JPEG/PNG/WEBP. Refusing anything else shrinks the
# parser attack surface to formats the pipeline actually depends on. BMP is
# deliberately excluded for the same reason it is excluded in preprocessing.py
# (richer history of malformed-header CVEs in both Pillow and OpenCV).
_ALLOWED_IMAGE_FORMATS: frozenset[str] = frozenset({"JPEG", "PNG", "WEBP"})


def safe_open_image(path: Path) -> Image.Image:
    """
    Open an image file with format whitelist, pixel cap, and bomb protection.

    The returned image is fully decoded in memory (``img.load()`` is called
    inside the warning context) so callers can close the underlying file
    handle immediately by letting ``img`` go out of scope. Without the eager
    load, Pillow defers decode until the first pixel access — at which point
    a decompression-bomb warning would fire outside this guard.

    Args:
        path: Filesystem path to an image file.

    Returns:
        A fully-decoded ``PIL.Image.Image`` ready for ``.convert()`` /
        ``.resize()`` / transform pipelines.

    Raises:
        ValueError: The image exceeds the pixel cap, or its detected format
            is not in the JPEG/PNG/WEBP whitelist.
        OSError:    The file is missing, unreadable, corrupt, or unparseable.
    """
    try:
        with warnings.catch_warnings():
            # Convert the warning into a real exception so workers that have
            # filterwarnings("ignore") configured still reject bomb images.
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            img = Image.open(path)
            img.load()
            fmt = img.format
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        # Pillow emits the warning at MAX_IMAGE_PIXELS and the error at 2x
        # MAX_IMAGE_PIXELS; both mean "image is too large" and both surface
        # as a ValueError so callers have a single exception type to handle.
        raise ValueError(
            f"Image {path.name} exceeds the "
            f"{MAX_IMAGE_PIXELS // 1_000_000} MP pixel cap."
        ) from exc
    except (UnidentifiedImageError, OSError) as exc:
        # Re-raise as OSError so callers do not need to import Pillow's
        # exception hierarchy just to handle "could not open this file".
        raise OSError(f"Could not open image {path.name}: {exc}") from exc

    if fmt not in _ALLOWED_IMAGE_FORMATS:
        raise ValueError(
            f"Image {path.name} has unsupported format {fmt!r}. "
            f"Allowed: {sorted(_ALLOWED_IMAGE_FORMATS)}."
        )

    return img
