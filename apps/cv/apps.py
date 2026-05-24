"""
App configuration for the 'cv' (computer vision) Django app.

This app will contain:
  - apps/cv/utils/device.py      — MPS/CUDA/CPU auto-detection (Day 2)
  - apps/cv/preprocessing.py     — image loading, resizing, normalizing (Day 2)
  - apps/cv/models/plate_detector.py  — CNN bounding box detector (Day 4)
  - apps/cv/models/recognizer.py      — CRNN plate text recognizer (Day 5)
  - apps/cv/training/             — synthetic data and training scripts (Days 3–5)
  - apps/cv/pipeline.py          — end-to-end inference pipeline (Day 6)
  - apps/cv/weights/              — trained model weights (gitignored)

WHY a separate app for computer vision?
  The CV pipeline is a self-contained subsystem with different concerns:
    - No database models (it doesn't store data, it produces data for apps/parking)
    - Its own training infrastructure (synthetic data, augmentation, training loops)
    - Different dependencies (PyTorch, OpenCV, torchvision)
    - Can be tested and developed independently of the parking logic
  Separating it into its own app enforces a clean interface boundary:
  apps/dashboard calls the pipeline and passes results to apps/parking/services.py.

Day 1: App registered but empty — just the app boundary established.
"""

from django.apps import AppConfig


class CvConfig(AppConfig):
    """Configuration class for the computer vision application."""

    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.cv'
    verbose_name = 'Computer Vision'
