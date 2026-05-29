"""
Synthetic license plate image generation for detector and recognizer training.

Pipeline:
    generate_plate_text()
        └─► render_plate_image()
                └─► composite_on_background()
                        │
            ┌───────────┴────────────┐
            ▼                        ▼
  generate_detector_dataset()  generate_recognizer_dataset()
  (10k composite images +      (50k cropped 128×32 plates +
   YOLO bounding-box labels)    labels.csv)

Supports US and Canadian plate formats. Synthetic backgrounds are loaded
from a user-supplied directory of real parking-lot photos so the detector
learns to find plates against realistic clutter.
"""

import csv
import logging
import math
import random
import string
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from apps.cv.training._image_io import safe_open_image

logger = logging.getLogger(__name__)

# Module-private RNG — isolated from the process-wide ``random`` module so
# seeding here does not affect reproducibility of other code (tests, workers).
_rng = random.Random()

# Upper bounds for caller-supplied generation parameters. A typo such as
# n=1_000_000_000 would otherwise silently exhaust the disk before any
# obvious failure mode triggers; capping here turns the typo into an
# immediate, descriptive ValueError.
_MAX_SAMPLES = 1_000_000
_MAX_TARGET_DIM = 4096

# ── Character sets ────────────────────────────────────────────────────────────

_LETTERS = string.ascii_uppercase
_DIGITS = string.digits

# ── Plate format templates  (L = letter, D = digit, space = literal space) ───
#
# US formats cover the three most common patterns across states.
# CA formats cover the majority of provinces (ON, BC, AB, QC etc.).
_US_FORMATS = [
    "LLL DDDD",  # ABC 1234 — most common US style
    "DDD LLL",   # 123 ABC — older/vanity style
    "LLLDDD",    # ABC123  — compact / some state vanity plates
]
_CA_FORMATS = [
    "LLL DDD",   # ABC 123 — most Canadian provinces
    "LDL DLD",   # A1B 2C3 — Ontario-style alphanumeric
]

# ── Plate rendering constants ─────────────────────────────────────────────────

PLATE_SIZE = (400, 120)   # (width, height) in pixels

_ASSETS_DIR = Path(__file__).parent / "assets"
_FONT_PATH = _ASSETS_DIR / "plate_font.ttf"
_FONT_SIZE = 64

# ── Background image extensions accepted for compositing ──────────────────────

_BG_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ── Background file collection (cached per bg_dir to avoid 10k dir scans) ────

@lru_cache(maxsize=8)
def _collect_bg_files(bg_dir: Path) -> tuple[Path, ...]:
    """
    Scan bg_dir once and cache the result for the lifetime of the process.

    generate_detector_dataset calls composite_on_background once per sample;
    without caching this would scan the directory 10,000+ times per run.
    lru_cache does not cache exceptions, so a missing dir is re-checked each call.
    maxsize=8 covers the realistic number of distinct background directories a
    single process touches (typically one) without growing unboundedly if a
    caller passes a fresh Path object each call.

    Results are sorted so that ``random.choice`` is reproducible across
    machines once ``random.seed`` is set — ``Path.iterdir`` returns files
    in filesystem-dependent order, which would otherwise leak into the
    sampling decision even under a fixed seed.
    """
    if not bg_dir.exists():
        raise FileNotFoundError("Background directory not found.")
    files = tuple(
        sorted(
            p for p in bg_dir.iterdir()
            if p.suffix.lower() in _BG_EXTENSIONS and not p.is_symlink()
        )
    )
    if not files:
        raise FileNotFoundError("No .jpg/.png images found in background directory.")
    return files


# ── Font loading (cached so 50k render calls don't reload from disk) ──────────

@lru_cache(maxsize=16)
def _load_font(size: int = _FONT_SIZE) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """
    Load the bundled TrueType plate font; fall back to PIL default if absent.

    The LRU cache means the font file is opened once per Python process regardless
    of how many plates are generated. Caching on `size` so different sizes don't
    trample each other if callers vary the parameter; maxsize=16 is plenty for
    the small number of distinct sizes the pipeline ever uses.

    Corrupt or unreadable font files fall through to the PIL default rather
    than aborting the run — a degraded plate font is better than no training
    data at all, and the warning makes the degradation visible.
    """
    if _FONT_PATH.exists():
        try:
            return ImageFont.truetype(str(_FONT_PATH), size=size)
        except (OSError, ValueError) as exc:
            logger.warning(
                "Plate font at %s could not be loaded (%s); "
                "falling back to PIL default.",
                _FONT_PATH,
                exc,
            )
    else:
        logger.warning(
            "Plate font not found — using PIL default (lower fidelity). "
            "See apps/cv/training/assets/README.md to install plate_font.ttf."
        )

    # load_default(size=) accepted in Pillow 10+; fallback handles older builds.
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


# ── Plate text generation ─────────────────────────────────────────────────────

def _apply_format(fmt: str) -> str:
    """Expand a format string where L → random letter, D → random digit."""
    out = []
    for ch in fmt:
        if ch == "L":
            out.append(_rng.choice(_LETTERS))
        elif ch == "D":
            out.append(_rng.choice(_DIGITS))
        else:
            out.append(ch)
    return "".join(out)


def generate_plate_text(country: str = "random") -> tuple[str, str]:
    """
    Generate a random license plate string for the given country.

    Args:
        country: "US", "CA", or "random" (uniform 50/50 split).

    Returns:
        (plate_text, country) — country is always "US" or "CA".

    Raises:
        ValueError: For any country value other than "US", "CA", or "random".
    """
    if country == "random":
        country = _rng.choice(["US", "CA"])

    if country == "US":
        return _apply_format(_rng.choice(_US_FORMATS)), "US"
    if country == "CA":
        return _apply_format(_rng.choice(_CA_FORMATS)), "CA"

    raise ValueError(f"Unknown country {country!r}. Must be 'US', 'CA', or 'random'.")


# ── Plate image rendering ─────────────────────────────────────────────────────

def render_plate_image(text: str, country: str) -> Image.Image:
    """
    Render a synthetic license plate as an RGBA PIL Image.

    Visual style:
        US: white background, black border, black text.
        CA: solid-blue header strip (top quarter) over white, black text.

    The RGBA mode is required so composite_on_background() can use the alpha
    channel as a transparency mask when pasting rotated plates onto backgrounds.

    Args:
        text:    Plate string (e.g. "ABC 1234"). Caller is responsible for validity.
        country: "US" or "CA".

    Returns:
        RGBA PIL Image of size PLATE_SIZE.
    """
    w, h = PLATE_SIZE
    img = Image.new("RGBA", (w, h), (255, 255, 255, 255)) # White background, fully opaque
    draw = ImageDraw.Draw(img)

    if country == "CA":
        # Blue province header strip — a simple visual differentiator from US plates
        draw.rectangle([0, 0, w, h // 4], fill=(0, 0, 180, 255))

    # Thin border to reinforce plate edges (helps detector learn plate boundaries)
    draw.rectangle([0, 0, w - 1, h - 1], outline=(30, 30, 30, 255), width=4)

    font = _load_font()

    # Center text within the plate using textbbox (getsize removed in Pillow 10).
    # textbbox returns the inked rectangle in canvas coordinates *including*
    # any origin offset (bbox[0], bbox[1]) — TrueType side bearings and ascender
    # padding mean bbox[0]/bbox[1] are often non-zero. Subtracting that offset
    # back out is required because draw.text((x, y), ...) interprets (x, y) as
    # the text's drawing origin, not the bbox top-left. Without the subtraction
    # every plate would be shifted right and down by the font's bearing.
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0] #bbox (left, top, right, bottom)
    text_h = bbox[3] - bbox[1]
    x = (w - text_w) // 2 - bbox[0]
    y = (h - text_h) // 2 - bbox[1]

    draw.text((x, y), text, fill=(10, 10, 10, 255), font=font)
    return img


# ── Compositing ───────────────────────────────────────────────────────────────

def composite_on_background(
    plate_img: Image.Image,
    bg_dir: Path,
    target_size: tuple[int, int] = (640, 480),
) -> tuple[Image.Image, list[int]]:
    """
    Paste a plate image onto a randomly chosen background at a random transform.

    Randomisations applied each call:
        - Background: random image from bg_dir
        - Scale:      plate occupies 15–40% of image width (realistic camera framing)
        - Rotation:   ±15° (cameras aren't always perfectly level)
        - Position:   uniform random within image bounds (plate always fully visible)

    Background filenames are never logged or returned to prevent CWE-532
    information disclosure — callers only receive the composite image and bbox.

    Args:
        plate_img:   RGBA Image from render_plate_image().
        bg_dir:      Directory containing .jpg/.png background images.
        target_size: (width, height) of the output image in pixels.

    Returns:
        (composite_rgb, [x, y, w, h]) — RGB PIL Image and ground-truth bounding
        box in pixel coordinates: top-left (x, y), width w, height h.

    Raises:
        FileNotFoundError: If bg_dir is missing or contains no valid images.
        ValueError:        If target_size exceeds the configured per-dim cap,
                           or if the chosen background fails format/pixel checks.
        OSError:           If the chosen background file is corrupt/unreadable.
    """
    tw, th = target_size
    if tw <= 0 or th <= 0 or tw > _MAX_TARGET_DIM or th > _MAX_TARGET_DIM:
        raise ValueError(
            f"target_size {target_size} must be positive and "
            f"<= {_MAX_TARGET_DIM} per dimension."
        )

    bg_files = _collect_bg_files(bg_dir.resolve())  # resolve for consistent cache key

    # Never log the chosen filename — prevents path leaks in production logs.
    # safe_open_image enforces pixel cap, format whitelist, and bomb protection;
    # propagating its OSError/ValueError as-is preserves the real failure cause
    # instead of masking it as FileNotFoundError.
    bg_path = _rng.choice(bg_files)
    bg = safe_open_image(bg_path).convert("RGBA")
    bg = bg.resize(target_size, Image.LANCZOS)

    # Scale plate so it occupies 15–40% of image width — realistic for parking cameras
    scale = _rng.uniform(0.15, 0.40)
    new_w = int(tw * scale)
    new_h = int(new_w * plate_img.height / plate_img.width)
    plate_scaled = plate_img.resize((new_w, new_h), Image.LANCZOS)

    # Rotate ±15°; expand=True grows the canvas to fit rotated corners without clipping
    angle = _rng.uniform(-15, 15)
    plate_rotated = plate_scaled.rotate(
        angle, resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0, 0)
    )

    # Compute the tight AABB of the rotated plate content via the standard formula,
    # explicit rather than reading plate_rotated.size (Pillow's expand=True uses
    # ceil rounding and may add 1–2 px of canvas padding beyond the plate corners).
    rad = math.radians(abs(angle))
    pw = int(new_w * math.cos(rad) + new_h * math.sin(rad))
    ph = int(new_w * math.sin(rad) + new_h * math.cos(rad))

    # Use the actual canvas dimensions for placement bounds so the full rotated
    # image (including any Pillow rounding pixels) always stays within the frame.
    canvas_w, canvas_h = plate_rotated.size

    # Constrain placement so the plate stays fully within the image
    max_x = tw - canvas_w
    max_y = th - canvas_h
    if max_x < 0 or max_y < 0:
        # Safety fallback: plate is larger than image (shouldn't happen at ≤40% scale)
        x, y = max(0, (tw - canvas_w) // 2), max(0, (th - canvas_h) // 2)
    else:
        x = _rng.randint(0, max_x)
        y = _rng.randint(0, max_y)

    # Paste using the plate's own alpha channel as the transparency mask
    bg.paste(plate_rotated, (x, y), mask=plate_rotated)

    return bg.convert("RGB"), [x, y, pw, ph]


# ── Dataset generation ────────────────────────────────────────────────────────

def _validate_sample_count(n: int) -> None:
    """Reject obviously-wrong sample counts before we touch the filesystem."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}.")
    if n > _MAX_SAMPLES:
        raise ValueError(
            f"n={n} exceeds the {_MAX_SAMPLES} sample cap. "
            "Raise _MAX_SAMPLES intentionally if you really need more."
        )


def _seed_rng(seed: int | None) -> None:
    """
    Seed the module-private RNG instance if a seed is supplied.

    Uses the module-level ``_rng`` (a ``random.Random`` instance) rather than
    the process-wide ``random`` module, so seeding here does not affect other
    code running in the same process (parallel workers, test suites).
    """
    if seed is not None:
        _rng.seed(seed)


def _clear_existing(directory: Path, suffixes: tuple[str, ...]) -> None:
    """
    Delete previously-generated dataset files matching the given suffixes.

    Without this, re-running with a smaller ``n`` leaves orphan files behind
    that the Dataset classes will happily ingest at training time — silently
    inflating the dataset and mixing labels across runs. Limiting the deletes
    to a fixed suffix whitelist prevents collateral damage if the operator
    points ``output_dir`` at a populated directory by mistake.
    """
    if not directory.exists():
        return
    for entry in directory.iterdir():
        if entry.is_file() and entry.suffix.lower() in suffixes:
            entry.unlink()


def generate_detector_dataset(
    n: int = 10_000,
    output_dir: Path = Path("data/detector"),
    bg_dir: Path = Path("data/backgrounds"),
    seed: int | None = None,
) -> None:
    """
    Generate n composite images with YOLO-format bounding-box labels.

    Output layout:
        output_dir/images/<index>.jpg
        output_dir/labels/<index>.txt  — "0 cx cy w h" (all values normalized 0–1)

    The YOLO format uses class index 0 for "license plate" and normalises all
    coordinates by the image dimensions so labels are resolution-independent.

    Re-running with a smaller ``n`` clears any prior ``.jpg``/``.txt`` files in
    the output directories first so the resulting dataset has exactly ``n``
    samples — orphan files from a previous larger run would otherwise be
    picked up by ``PlateDetectorDataset`` and pollute training.

    Args:
        n:          Number of training samples to generate (1 .. 1_000_000).
        output_dir: Root output directory (created if missing).
        bg_dir:     Directory of real background images.
        seed:       Optional seed for ``random`` to make output reproducible.
                    Note: this calls ``random.seed(seed)`` on the process-wide
                    RNG, which affects any other code in the same process.

    Raises:
        FileNotFoundError: If bg_dir is missing or empty.
        ValueError:        If n is outside [1, 1_000_000].
    """
    _validate_sample_count(n)
    _seed_rng(seed)

    output_dir = output_dir.resolve()  # canonicalize before any deletion
    bg_dir = bg_dir.resolve()

    img_dir = output_dir / "images"
    lbl_dir = output_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    _clear_existing(img_dir, (".jpg",))
    _clear_existing(lbl_dir, (".txt",))

    skip_count = 0
    for i in range(n):
        try:
            text, country = generate_plate_text()
            plate = render_plate_image(text, country)
            composite, (x, y, w, h) = composite_on_background(plate, bg_dir)

            composite.save(img_dir / f"{i:06d}.jpg", quality=92)

            iw, ih = composite.size
            cx = (x + w / 2) / iw
            cy = (y + h / 2) / ih
            nw = w / iw
            nh = h / ih
            (lbl_dir / f"{i:06d}.txt").write_text(
                f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n"
            )
        except (OSError, ValueError) as exc:
            logger.warning("Detector dataset: skipping sample %d — %s", i, exc)
            skip_count += 1
            continue

        if (i + 1) % 1000 == 0:
            logger.info("Detector dataset: %d / %d generated", i + 1, n)

    if skip_count:
        logger.warning(
            "Detector dataset: skipped %d / %d samples due to errors.", skip_count, n
        )


def generate_recognizer_dataset(
    n: int = 50_000,
    output_dir: Path = Path("data/recognizer"),
    seed: int | None = None,
) -> None:
    """
    Generate n cropped plate images at 128×32 grayscale for recognizer training.

    Output layout:
        output_dir/images/<index>.png  — 128×32, mode "L" (grayscale)
        output_dir/labels.csv          — columns: filename, text, country

    Grayscale is used because plate text recognition is a shape-recognition task;
    colour carries no signal and reducing to 1 channel halves the feature space.

    Re-running with a smaller ``n`` clears any prior ``.png`` files in the
    output directory and rewrites ``labels.csv`` from scratch, so the resulting
    dataset has exactly ``n`` samples — orphan PNGs from a previous larger run
    would otherwise be picked up if the CSV were ever regenerated by hand.

    Args:
        n:          Number of training samples to generate (1 .. 1_000_000).
        output_dir: Root output directory (created if missing).
        seed:       Optional seed for ``random`` to make output reproducible.
                    Note: this calls ``random.seed(seed)`` on the process-wide
                    RNG, which affects any other code in the same process.

    Raises:
        ValueError: If n is outside [1, 1_000_000].
    """
    _validate_sample_count(n)
    _seed_rng(seed)

    output_dir = output_dir.resolve()  # canonicalize before any deletion

    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    _clear_existing(img_dir, (".png",))

    csv_path = output_dir / "labels.csv"
    skip_count = 0
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "text", "country"])

        for i in range(n):
            try:
                text, country = generate_plate_text()
                plate = render_plate_image(text, country)
                # Flatten RGBA onto white before converting to grayscale so the
                # alpha channel is not misinterpreted as black in the luminance
                # formula (safe now since alpha=255, but prevents a silent bug
                # if render_plate_image ever adds partial transparency).
                white_bg = Image.new("RGB", plate.size, (255, 255, 255))
                white_bg.paste(plate, mask=plate.split()[3])
                plate_gray = white_bg.convert("L").resize((128, 32), Image.LANCZOS)

                filename = f"{i:06d}.png"
                plate_gray.save(img_dir / filename)
                writer.writerow([filename, text, country])
            except (OSError, ValueError) as exc:
                logger.warning("Recognizer dataset: skipping sample %d — %s", i, exc)
                skip_count += 1
                continue

            if (i + 1) % 5000 == 0:
                logger.info("Recognizer dataset: %d / %d generated", i + 1, n)

    if skip_count:
        logger.warning(
            "Recognizer dataset: skipped %d / %d samples due to errors.", skip_count, n
        )
