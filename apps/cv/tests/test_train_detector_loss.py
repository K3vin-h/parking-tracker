"""
Unit tests for the pure helpers added to train_detector.py in the 2026-09-26
detector retrain: the combined SmoothL1+GIoU loss and the best-IoU checkpoint
predicate. Both are tested directly, without running an epoch.
"""

import pytest
import torch

from apps.cv.training.train_detector import _detector_loss, _is_new_best


class TestDetectorLoss:
    def test_identical_boxes_have_near_zero_loss(self):
        boxes = torch.tensor([[0.5, 0.5, 0.4, 0.2]])
        smooth_l1 = torch.nn.SmoothL1Loss(beta=1.0, reduction="mean")
        loss = _detector_loss(boxes, boxes, smooth_l1, giou_weight=1.0)
        assert loss.item() == pytest.approx(0.0, abs=1e-4)

    def test_giou_weight_zero_matches_smooth_l1_alone(self):
        preds = torch.tensor([[0.4, 0.4, 0.3, 0.3]])
        targets = torch.tensor([[0.5, 0.5, 0.4, 0.2]])
        smooth_l1 = torch.nn.SmoothL1Loss(beta=1.0, reduction="mean")
        combined = _detector_loss(preds, targets, smooth_l1, giou_weight=0.0)
        plain = smooth_l1(preds, targets)
        assert combined.item() == pytest.approx(plain.item())

    def test_larger_giou_weight_increases_loss_for_non_overlapping_boxes(self):
        preds = torch.tensor([[0.1, 0.1, 0.1, 0.1]])
        targets = torch.tensor([[0.9, 0.9, 0.1, 0.1]])
        smooth_l1 = torch.nn.SmoothL1Loss(beta=1.0, reduction="mean")
        low = _detector_loss(preds, targets, smooth_l1, giou_weight=0.1)
        high = _detector_loss(preds, targets, smooth_l1, giou_weight=2.0)
        assert high.item() > low.item()


class TestIsNewBest:
    def test_higher_iou_is_new_best(self):
        assert _is_new_best(0.5, 0.4) is True

    def test_lower_or_equal_iou_is_not_new_best(self):
        assert _is_new_best(0.4, 0.5) is False
        assert _is_new_best(0.5, 0.5) is False

    def test_first_epoch_beats_sentinel(self):
        assert _is_new_best(0.0, -1.0) is True
