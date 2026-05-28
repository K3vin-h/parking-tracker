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
import random
import string
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

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

@lru_cache(maxsize=None)
def _collect_bg_files(bg_dir: Path) -> tuple[Path, ...]:
    """
    Scan bg_dir once and cache the result for the lifetime of the process.

    generate_detector_dataset calls composite_on_background once per sample;
    without caching this would scan the directory 10,000+ times per run.
    lru_cache does not cache exceptions, so a missing dir is re-checked each call.
    """
    if not bg_dir.exists():
        raise FileNotFoundError("Background directory not found.")
    files = tuple(p for p in bg_dir.iterdir() if p.suffix.lower() in _BG_EXTENSIONS)
    if not files:
        raise FileNotFoundError("No .jpg/.png images found in background directory.")
    return files


# ── Font loading (cached so 50k render calls don't reload from disk) ──────────

@lru_cache(maxsize=None)
def _load_font(size: int = _FONT_SIZE) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """
    Load the bundled TrueType plate font; fall back to PIL default if absent.

    The LRU cache means the font file is opened once per Python process regardless
    of how many plates are generated. Caching on `size` so different sizes don't
    trample each other if callers vary the parameter.
    """
    if _FONT_PATH.exists():
        return ImageFont.truetype(str(_FONT_PATH), size=size)

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
            out.append(random.choice(_LETTERS))
        elif ch == "D":
            out.append(random.choice(_DIGITS))
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
        country = random.choice(["US", "CA"])

    if country == "US":
        return _apply_format(random.choice(_US_FORMATS)), "US"
    if country == "CA":
        return _apply_format(random.choice(_CA_FORMATS)), "CA"

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
    img = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)

    if country == "CA":
        # Blue province header strip — a simple visual differentiator from US plates
        draw.rectangle([0, 0, w, h // 4], fill=(0, 0, 180, 255))

    # Thin border to reinforce plate edges (helps detector learn plate boundaries)
    draw.rectangle([0, 0, w - 1, h - 1], outline=(30, 30, 30, 255), width=4)

    font = _load_font()

    # Center text within the plate using textbbox (getsize removed in Pillow 10)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (w - text_w) // 2
    y = (h - text_h) // 2

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
    """
    bg_files = _collect_bg_files(bg_dir)  # cached; raises FileNotFoundError if missing

    # Never log the chosen filename — prevents path leaks in production logs
    bg_path = random.choice(bg_files)
    try:
        bg = Image.open(bg_path).convert("RGBA")
    except Exception:
        raise FileNotFoundError("Could not open a background image.")
    bg = bg.resize(target_size, Image.LANCZOS)

    tw, th = target_size

    # Scale plate so it occupies 15–40% of image width — realistic for parking cameras
    scale = random.uniform(0.15, 0.40)
    new_w = int(tw * scale)
    new_h = int(new_w * plate_img.height / plate_img.width)
    plate_scaled = plate_img.resize((new_w, new_h), Image.LANCZOS)

    # Rotate ±15°; expand=True grows the canvas to fit rotated corners without clipping
    angle = random.uniform(-15, 15)
    plate_rotated = plate_scaled.rotate(
        angle, resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0, 0)
    )

    # pw, ph are the tight AABB of the rotated plate (Pillow expand=True computes
    # the minimum bounding box of all four rotated corners). This IS the correct
    # axis-aligned bbox for YOLO format — the transparent corners are outside the
    # plate content but within the AABB, which is standard for detection labels.
    pw, ph = plate_rotated.size

    # Constrain placement so the plate stays fully within the image
    max_x = tw - pw
    max_y = th - ph
    if max_x < 0 or max_y < 0:
        # Safety fallback: plate is larger than image (shouldn't happen at ≤40% scale)
        x, y = max(0, (tw - pw) // 2), max(0, (th - ph) // 2)
    else:
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)

    # Paste using the plate's own alpha channel as the transparency mask
    bg.paste(plate_rotated, (x, y), mask=plate_rotated)

    return bg.convert("RGB"), [x, y, pw, ph]


# ── Dataset generation ────────────────────────────────────────────────────────

def generate_detector_dataset(
    n: int = 10_000,
    output_dir: Path = Path("data/detector"),
    bg_dir: Path = Path("data/backgrounds"),
) -> None:
    """
    Generate n composite images with YOLO-format bounding-box labels.

    Output layout:
        output_dir/images/<index>.jpg
        output_dir/labels/<index>.txt  — "0 cx cy w h" (all values normalized 0–1)

    The YOLO format uses class index 0 for "license plate" and normalises all
    coordinates by the image dimensions so labels are resolution-independent.

    Args:
        n:          Number of training samples to generate.
        output_dir: Root output directory (created if missing, safe to re-run).
        bg_dir:     Directory of real background images.

    Raises:
        FileNotFoundError: If bg_dir is missing or empty.
    """
    img_dir = output_dir / "images"
    lbl_dir = output_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n):
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

        if (i + 1) % 1000 == 0:
            logger.info("Detector dataset: %d / %d generated", i + 1, n)


def generate_recognizer_dataset(
    n: int = 50_000,
    output_dir: Path = Path("data/recognizer"),
) -> None:
    """
    Generate n cropped plate images at 128×32 grayscale for recognizer training.

    Output layout:
        output_dir/images/<index>.png  — 128×32, mode "L" (grayscale)
        output_dir/labels.csv          — columns: filename, text, country

    Grayscale is used because plate text recognition is a shape-recognition task;
    colour carries no signal and reducing to 1 channel halves the feature space.

    Args:
        n:          Number of training samples to generate.
        output_dir: Root output directory (created if missing, safe to re-run).
    """
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "labels.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "text", "country"])

        for i in range(n):
            text, country = generate_plate_text()
            plate = render_plate_image(text, country)
            # Convert to grayscale and resize to recognizer input resolution
            plate_gray = plate.convert("L").resize((128, 32), Image.LANCZOS)

            filename = f"{i:06d}.png"
            plate_gray.save(img_dir / filename)
            writer.writerow([filename, text, country])

            if (i + 1) % 5000 == 0:
                logger.info("Recognizer dataset: %d / %d generated", i + 1, n)
