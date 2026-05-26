"""
Image preprocessing for the license plate CV pipeline.

This module transforms raw images (as loaded from disk or uploaded by a user)
into the exact tensor formats that the plate detector and plate recognizer
neural networks expect. It is intentionally free of model code — preprocessing
is a separate concern from inference.

Pipeline overview:
  Raw image on disk
    └─ load_image()            → uint8 BGR numpy array
    └─ bgr_to_rgb()            → uint8 RGB numpy array
    └─ resize_for_detector()   → uint8 RGB numpy array at 640×480
    └─ normalize_pixels()      → float32 RGB numpy array in [0, 1]
    └─ to_tensor()             → float32 CHW tensor        (for detector)

  Detector output (bounding box)
    └─ crop_plate_region()     → uint8 RGB numpy crop
    └─ prepare_for_recognizer() → float32 (1, 32, 128) tensor  (for recognizer)
"""

import logging

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


def load_image(path: str) -> np.ndarray:
    """
    Load an image from disk using OpenCV.

    WHY OpenCV instead of PIL: OpenCV is faster for large images and its
    output is a numpy array — the native format for all subsequent
    preprocessing steps. PIL's Image objects require an extra conversion step.

    WHY explicit None check: cv2.imread returns None on failure (file not
    found, corrupt file, unsupported format) instead of raising an exception.
    Leaving None to propagate would cause a confusing AttributeError or
    TypeError deep in the pipeline. Raising FileNotFoundError here gives
    callers a meaningful error they can catch and handle.

    Args:
        path: Absolute or relative path to the image file.

    Returns:
        uint8 numpy array of shape (H, W, 3) in BGR channel order.

    Raises:
        FileNotFoundError: If the file cannot be loaded by OpenCV.
    """
    img = cv2.imread(path)
    if img is None:
        # Log the path at DEBUG only — do not include it in the exception
        # message, which may surface in API responses and leak internal paths.
        logger.debug("cv2.imread returned None for path=%r", path)
        raise FileNotFoundError(
            "Could not load the image. "
            "The file may not exist, be corrupt, or be an unsupported format."
        )

    # Guard against decompression bombs: a small compressed file can decode
    # to a multi-gigabyte array. A 12 MP cap (4000×3000) is generous for
    # security camera images and prevents runaway memory allocation.
    max_pixels = 4000 * 3000
    h, w = img.shape[:2]
    if h * w > max_pixels:
        raise ValueError(
            f"Image dimensions {w}×{h} exceed the maximum allowed size "
            f"({max_pixels // 1_000_000} MP). Resize before uploading."
        )

    logger.debug("Loaded image shape=%s dtype=%s", img.shape, img.dtype)
    return img  # uint8, shape (H, W, 3), BGR channel order


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert an image from BGR to RGB channel order.

    WHY this is necessary: OpenCV loads images in BGR (Blue-Green-Red) order
    — a legacy from early digital camera sensors that captured in BGR. Every
    other major library (PIL, matplotlib, PyTorch's pretrained models) expects
    RGB (Red-Green-Blue). Passing a BGR image to an RGB-trained model causes
    the model to see 'blue' where it expects 'red', leading to poor predictions.

    WHY cv2.cvtColor instead of array slicing (img[:, :, ::-1]): cvtColor is
    hardware-accelerated and returns a contiguous array. The slice approach
    produces a non-contiguous view, which requires np.ascontiguousarray before
    passing to torch.from_numpy.

    Args:
        image: uint8 numpy array in BGR order, shape (H, W, 3).

    Returns:
        uint8 numpy array in RGB order, shape (H, W, 3).
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_for_detector(
    image: np.ndarray,
    target: tuple[int, int] = (640, 480),
) -> np.ndarray:
    """
    Resize an image to the plate detector's expected input dimensions.

    WHY target=(640, 480): The PlateDetectorCNN is designed for a 640×480
    input (width × height). Consistent input size ensures the fully-connected
    layers receive a fixed-size feature vector after the convolutional blocks.
    The 4:3 aspect ratio matches most security camera outputs.

    WHY adaptive interpolation:
    - INTER_AREA: used when downscaling. It averages the pixels that map to
      each output pixel, preventing moiré patterns and aliasing that appear
      when a large image is naively mapped to fewer pixels. Best quality for
      shrinking images.
    - INTER_LINEAR: used when upscaling. Bilinear interpolation estimates
      new pixel values by blending neighboring pixels. Fast and smooth.
      Upscaling is uncommon in this pipeline (cameras typically produce images
      larger than 640×480) but is handled correctly.

    NOTE on cv2 coordinate convention: cv2.resize takes dsize as (width, height)
    — the opposite of numpy's shape convention of (height, width). This is a
    common source of bugs. The target tuple here follows OpenCV's convention.

    Args:
        image: numpy array, shape (H, W, 3).
        target: (width, height) in pixels. Default matches detector input size.

    Returns:
        Resized numpy array, shape (target_height, target_width, 3).
    """
    src_h, src_w = image.shape[:2]
    target_w, target_h = target

    # Choose interpolation based on net scale direction (by total pixel count).
    # Using `or` on individual dimensions is wrong for mixed cases: an image at
    # 320×600 resizing to 640×480 has one axis shrinking and one growing — the
    # `or` condition would pick INTER_AREA, but OpenCV falls back to nearest-
    # neighbor for INTER_AREA when a dimension is being upscaled, causing blocky
    # artifacts. Comparing total pixel count picks the dominant direction.
    total_src = src_w * src_h
    total_dst = target_w * target_h
    if total_src > total_dst:
        interpolation = cv2.INTER_AREA    # net downscale: average pooling
    else:
        interpolation = cv2.INTER_LINEAR  # net upscale or same size: bilinear

    return cv2.resize(image, target, interpolation=interpolation)


def normalize_pixels(image: np.ndarray) -> np.ndarray:
    """
    Scale pixel values from uint8 [0, 255] to float32 [0.0, 1.0].

    WHY normalization matters for training: Neural networks learn via gradient
    descent, adjusting weights by small increments proportional to the gradient.
    When inputs are in [0, 255], the gradients in the first layer are 255×
    larger than when inputs are in [0, 1]. Large gradients cause large weight
    updates that overshoot the optimum, making training unstable. Normalizing
    to [0, 1] keeps gradients in a stable range and makes the learning rate
    a meaningful hyperparameter.

    WHY float32: PyTorch's default computation dtype is float32. Using float64
    doubles memory usage and compute cost without meaningful precision benefit
    for computer vision tasks.

    WHY astype produces a copy: numpy's astype() with a different dtype always
    returns a new array (never modifies in place). The result is always C-
    contiguous, so it can be passed directly to torch.from_numpy.

    Args:
        image: uint8 numpy array with values in [0, 255].

    Returns:
        float32 numpy array with values in [0.0, 1.0], same shape as input.
    """
    if image.dtype != np.uint8:
        raise TypeError(
            f"normalize_pixels expects a uint8 array, got {image.dtype}. "
            "Call this function before any float conversion."
        )
    return image.astype(np.float32) / 255.0


def to_tensor(image: np.ndarray) -> torch.Tensor:
    """
    Convert a normalized HWC numpy array to a CHW PyTorch tensor.

    WHY HWC → CHW reordering: NumPy and OpenCV store images in
    (Height, Width, Channels) order — the values for all three channels of a
    single pixel are stored together. PyTorch's Conv2d layers expect
    (Channels, Height, Width) — all values for a single channel stored
    together. This 'channel-first' layout is more efficient for GPU memory
    access patterns during convolution.

    permute(2, 0, 1) moves the axis at position 2 (C) to position 0, and
    shifts H and W right: (H=0, W=1, C=2) → (C=0, H=1, W=2).

    WHY .contiguous(): permute() returns a view with non-contiguous memory
    (the strides no longer match the logical layout). Calling .contiguous()
    creates a new tensor with the data laid out in memory to match the logical
    order. This prevents errors in downstream ops that require contiguous tensors
    (e.g. certain CUDA kernels, .numpy() conversion).

    WHY torch.from_numpy: It creates a tensor that shares memory with the numpy
    array — no copy unless .contiguous() triggers one. The float32 dtype from
    normalize_pixels() maps directly to torch.float32.

    Args:
        image: float32 numpy array of shape (H, W, C) with values in [0, 1].

    Returns:
        float32 tensor of shape (C, H, W).
    """
    tensor = torch.from_numpy(image)           # (H, W, C), float32
    return tensor.permute(2, 0, 1).contiguous()  # (C, H, W), contiguous


def crop_plate_region(image: np.ndarray, bbox: list[float]) -> np.ndarray:
    """
    Crop the license plate region from an image using a normalized bounding box.

    WHY normalized coordinates: Storing bbox as fractions of image dimensions
    (rather than pixel values) makes the representation resolution-independent.
    The same bbox [0.3, 0.4, 0.2, 0.1] is valid for 640×480 or 1920×1080 —
    it always means '30% from left, 40% from top, spanning 20% width and 10%
    height'. This is the format output by the PlateDetectorCNN.

    WHY clamping: Floating-point arithmetic can push coordinates slightly
    outside [0, 1]. For example, a bbox edge at 0.999 on a 480px image gives
    0.999 × 480 = 479.52 → rounded to 479, which is fine. But a value of 1.0
    gives exactly 480, which is one past the last valid index (479). Clamping
    with min/max prevents out-of-bounds slice errors without altering the crop
    meaningfully.

    Args:
        image: numpy array of shape (H, W, 3) in RGB order.
        bbox: [x, y, w, h] with all values in [0.0, 1.0].
              (x, y) is the top-left corner; (w, h) are width and height
              as fractions of image dimensions.

    Returns:
        Cropped numpy array, shape (crop_h, crop_w, 3).
    """
    if len(bbox) != 4:
        raise ValueError(f"bbox must have exactly 4 elements [x, y, w, h], got {len(bbox)}")

    x, y, bw, bh = bbox

    if not all(isinstance(v, (int, float)) and np.isfinite(v) for v in bbox):
        raise ValueError(f"bbox contains non-finite values: {bbox}")
    if bw <= 0 or bh <= 0:
        raise ValueError(f"bbox has non-positive dimensions: w={bw}, h={bh}")

    img_h, img_w = image.shape[:2]
    x_px = int(x * img_w)
    y_px = int(y * img_h)
    w_px = int(bw * img_w)
    h_px = int(bh * img_h)

    # Clamp to image dimensions to handle floating-point boundary edge cases
    x1 = max(0, x_px)
    y1 = max(0, y_px)
    x2 = min(img_w, x_px + w_px)
    y2 = min(img_h, y_px + h_px)

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Degenerate bbox {bbox} produced a zero-size crop region "
            f"(x: {x1}→{x2}, y: {y1}→{y2}) on image shape {image.shape}. "
            "The bounding box may be outside the image or have near-zero size."
        )

    return image[y1:y2, x1:x2]


def prepare_for_recognizer(plate_crop: np.ndarray) -> torch.Tensor:
    """
    Prepare a cropped plate region for the CRNN recognizer network.

    Transforms an RGB plate crop of any size into the fixed-format tensor
    that PlateRecognizerCRNN expects: single-channel grayscale at 128×32.

    Step-by-step:

    1. Resize to 128×32 (width × height in OpenCV convention).
       WHY 128 wide: provides enough horizontal resolution to distinguish
       character shapes within a plate (typically 6-8 characters). At 128px,
       each character occupies roughly 16px — enough for the CNN backbone to
       detect vertical strokes, curves, and junctions.
       WHY 32 tall: sufficient character height while keeping the model small.
       The CRNN's CNN blocks reduce height to 8px before the LSTM, giving the
       sequence model 16 horizontal slices to read left-to-right.
       WHY INTER_LINEAR: plate crops are often smaller than 128×32 (the plate
       might be far from the camera), so upscaling is common. INTER_LINEAR's
       bilinear interpolation produces smoother results than nearest-neighbor.

    2. Convert RGB → grayscale using cv2.COLOR_RGB2GRAY.
       WHY grayscale: the recognizer reads text shapes, not colors. 'ABC123'
       in blue-on-white and black-on-yellow should yield the same reading.
       Dropping 2 of 3 channels reduces the CNN's input complexity 3×, making
       the model smaller and faster to train.
       WHY cv2.COLOR_RGB2GRAY (not BGR2GRAY): the input is already in RGB
       order after bgr_to_rgb() was called earlier in the pipeline.
       cv2.cvtColor drops the channel dimension: output shape is (H, W).

    3. Restore the channel dimension: (H, W) → (H, W, 1).
       WHY: permute(2, 0, 1) in step 5 requires a 3-axis array. The channel
       axis represents the single grayscale channel (C=1).

    4. Normalize to [0.0, 1.0] — same rationale as normalize_pixels().

    5. Convert to CHW tensor — same rationale as to_tensor().
       Result shape: (1, 32, 128) — 1 grayscale channel, 32px tall, 128px wide.

    Args:
        plate_crop: RGB numpy array of the plate region, any input size.

    Returns:
        float32 tensor of shape (1, 32, 128).
    """
    # Step 1: resize — cv2 dsize is (width, height)
    resized = cv2.resize(plate_crop, (128, 32), interpolation=cv2.INTER_LINEAR)

    # Step 2: RGB → grayscale; channel dim is dropped, output shape: (H, W)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

    # Step 3: restore channel dim for permute compatibility: (H, W) → (H, W, 1)
    gray = gray[:, :, np.newaxis]

    # Step 4: normalize to [0, 1]
    gray = gray.astype(np.float32) / 255.0

    # Step 5: (H, W, C) → (C, H, W), ensure contiguous memory layout
    return torch.from_numpy(gray).permute(2, 0, 1).contiguous()
