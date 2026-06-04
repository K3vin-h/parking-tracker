"""
CV model definitions for the parking tracker pipeline.

    PlateDetectorCNN  — CNN that outputs a YOLO bounding box [cx, cy, w, h]
                        for the license plate region in a 640×480 RGB image.

    PlateRecognizerCRNN (Day 5) — CRNN that reads plate text from a cropped
                        plate image.
"""

from apps.cv.models.plate_detector import PlateDetectorCNN

__all__ = ["PlateDetectorCNN"]
