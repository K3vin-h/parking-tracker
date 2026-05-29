"""
Augmentation pipelines for plate detector and recognizer training.

Both classes expose a simple __call__(tensor) → tensor interface so they plug
directly into Dataset constructors as the `transform` argument, or into a
v2.Compose chain after ToImage + ToDtype.

    DetectorAugment   — full parking-lot images (RGB, 640×480)
    RecognizerAugment — cropped plate images (grayscale, 128×32)

Augmentations are chosen to address the main sources of domain shift between
synthetic training data and real parking-lot camera footage.
"""

import torch
from torchvision.transforms import v2


class DetectorAugment:
    """
    Stochastic augmentation pipeline for full parking-lot images.

    train=True transforms (applied randomly per sample):
        ColorJitter       — parking cameras vary in white balance and exposure
        RandomGrayscale   — some CCTV feeds are monochrome; 10% probability
        GaussianBlur      — simulates lens blur and compression artefacts
        Horizontal flip   — vehicles approach from either direction; updates
                            YOLO [cx, cy, w, h] boxes when a bbox is supplied
        Normalize         — ImageNet statistics for compatibility with pre-trained backbones

    train=False: normalise only (no stochastic transforms during validation/inference).

    Args:
        train: True to apply full stochastic augmentations; False for eval mode.
    """

    # ImageNet statistics — standard for models fine-tuned from ImageNet weights
    _MEAN = [0.485, 0.456, 0.406]
    _STD = [0.229, 0.224, 0.225]

    def __init__(self, train: bool = True) -> None:
        self._train = train
        normalize = v2.Normalize(mean=self._MEAN, std=self._STD)

        if train:
            self._transform = v2.Compose([
                v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
                v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
                # Normalize before RandomGrayscale: ImageNet mean/std applied to
                # consistent 3-ch RGB. When grayscale fires, all three channels
                # hold the same luma value — a valid normalized input for 3-ch
                # backbones. Normalizing after grayscale would apply different
                # per-channel offsets to three identical values (wrong statistics).
                normalize,
                v2.RandomGrayscale(p=0.1),
            ])
        else:
            self._transform = normalize

    def __call__(
        self,
        image: torch.Tensor,
        bbox: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Apply augmentation to a float32 image tensor in [0, 1].

        Args:
            image: Tensor of shape (3, H, W) in [0, 1].
            bbox: Optional YOLO-format [cx, cy, w, h] tensor with normalised
                  coordinates. When the train-time horizontal flip fires, cx is
                  mirrored to keep the target aligned with the image.

        Returns:
            Augmented tensor of the same shape, or (image, bbox) when bbox is
            provided.
        """
        if self._train and torch.rand(()) < 0.5:
            image = torch.flip(image, dims=(-1,))
            if bbox is not None:
                bbox = bbox.clone()
                bbox[0] = 1.0 - bbox[0]

        image = self._transform(image)
        if bbox is None:
            return image
        return image, bbox


class RecognizerAugment:
    """
    Stochastic augmentation pipeline for cropped plate images.

    train=True transforms (applied randomly per sample):
        ColorJitter         — faded, dirty, or sun-bleached plates vary in contrast
        GaussianBlur        — camera resolution limits on small plate crops
        RandomPerspective   — camera angle variation at the parking gate entry point

    NOTE: RandomHorizontalFlip is intentionally absent. Mirrored plate text is
    undecodable by a left-to-right sequence model and would poison the labels.

    train=False: normalise only.

    Args:
        train: True to apply full stochastic augmentations; False for eval mode.
    """

    # Single-channel normalisation — zero-centre grayscale values
    _MEAN = [0.5]
    _STD = [0.5]

    def __init__(self, train: bool = True) -> None:
        normalize = v2.Normalize(mean=self._MEAN, std=self._STD)

        if train:
            self._transform = v2.Compose([
                v2.ColorJitter(brightness=0.3, contrast=0.3),
                v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
                # distortion_scale=0.2 is mild — strong perspective breaks OCR accuracy
                v2.RandomPerspective(distortion_scale=0.2, p=0.5),
                normalize,
            ])
        else:
            self._transform = normalize

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation to a float32 grayscale tensor in [0, 1].

        Args:
            image: Tensor of shape (1, H, W) in [0, 1].

        Returns:
            Augmented tensor of the same shape.
        """
        return self._transform(image)
