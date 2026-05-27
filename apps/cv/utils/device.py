"""
Device auto-detection for the CV pipeline.

Centralizing device selection here means training scripts, the inference
pipeline, and tests all agree on which hardware to use without duplicating
the detection logic.
"""

import torch


def get_device() -> torch.device:
    """
    Return the best available compute device.

    Priority: MPS → CUDA → CPU

    WHY MPS first: Apple Silicon (M1/M2/M3) uses a unified memory
    architecture where CPU and GPU share the same memory pool. PyTorch's
    MPS backend (Metal Performance Shaders) gives GPU-level throughput for
    tensor operations without copying data across a PCIe bus. This makes it
    ideal for local development and training on Apple hardware.

    WHY CUDA second: NVIDIA GPUs are the standard for production ML workloads.
    CUDA has the most complete PyTorch operator coverage and the highest
    throughput for large-scale training jobs.

    WHY CPU last: Every machine has a CPU. Inference on CPU is slower but
    universally available — ensures the pipeline always runs in environments
    without GPU support (CI runners, minimal cloud instances, Docker without
    GPU passthrough).

    IMPORTANT — MPS operator gaps: MPS does not support all PyTorch ops.
    As of PyTorch 2.x, torch.nn.CTCLoss is not supported on MPS. Training
    scripts that use CTC loss must move the loss computation to CPU while
    keeping the model weights on MPS. Inference-only code (loading a trained
    model and running forward passes) works correctly on MPS.

    torch.backends.mps.is_available() is the correct API — it checks both
    that PyTorch was compiled with MPS support AND that MPS hardware is
    present. torch.backends.mps.is_built() only checks compilation, not
    hardware availability, so it should not be used for device selection.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
