"""
Unit tests for apps/cv/utils/device.py.

These tests verify that get_device() returns a valid torch.device regardless
of what hardware is available in the test environment. We cannot assert which
specific device is returned (that depends on the machine running the tests),
but we can assert the return type and that the device type is one of the three
valid options.
"""

import torch
import pytest

from apps.cv.utils.device import get_device


@pytest.mark.unit
def test_get_device_returns_torch_device():
    """get_device() must return a torch.device instance, not a string."""
    device = get_device()
    assert isinstance(device, torch.device)


@pytest.mark.unit
def test_get_device_valid_device_type():
    """Device type must be one of the three supported backends."""
    device = get_device()
    assert device.type in ("mps", "cuda", "cpu")


@pytest.mark.unit
def test_get_device_is_deterministic():
    """Two consecutive calls must return the same device type."""
    device_a = get_device()
    device_b = get_device()
    assert device_a.type == device_b.type


@pytest.mark.unit
def test_get_device_tensor_can_be_moved_to_device():
    """A small tensor should be moveable to the detected device without error."""
    device = get_device()
    tensor = torch.zeros(2, 2)
    moved = tensor.to(device)
    assert moved.device.type == device.type
