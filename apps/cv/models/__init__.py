"""
CV model definitions for the parking tracker pipeline.

    PlateDetectorCNN   — CNN that outputs a YOLO bounding box [cx, cy, w, h]
                         for the license plate region in a 640×480 RGB image.

    PlateRecognizerCRNN — CRNN that reads plate text from a cropped plate image,
                         returning (T=16, N, C=37) CTC log-probabilities.
"""

from apps.cv.models.plate_detector import PlateDetectorCNN
from apps.cv.models.recognizer import PlateRecognizerCRNN

__all__ = ["PlateDetectorCNN", "PlateRecognizerCRNN"]
