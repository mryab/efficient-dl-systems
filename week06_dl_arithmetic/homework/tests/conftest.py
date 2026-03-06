"""
Shared pytest fixtures and utilities for all tests.
"""

import os
import sys

import pytest
import torch
from torch.autograd.graph import saved_tensors_hooks

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEED = 42


@pytest.fixture(autouse=True)
def seed():
    """Set random seeds for reproducibility."""
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)


@pytest.fixture
def device():
    """CUDA device fixture."""
    return torch.device("cuda")


def bytes_to_mb(num_bytes: int) -> float:
    """Convert bytes to megabytes."""
    return num_bytes / (1024 ** 2)


def measure_saved_tensors_bytes(fn, exclude_tensors=None) -> int:
    """Measure total bytes of new tensors saved for backward pass.
    
    Args:
        fn: Function to measure
        exclude_tensors: List of tensors to exclude (e.g., model parameters)
    """
    exclude_ptrs = {t.data_ptr() for t in (exclude_tensors or [])}
    total = 0

    def pack(t: torch.Tensor):
        nonlocal total
        if t.data_ptr() not in exclude_ptrs:
            total += t.numel() * t.element_size()
        return t

    def unpack(t):
        return t

    with saved_tensors_hooks(pack, unpack):
        fn()
    return total


def measure_peak_memory(fn) -> int:
    """Measure peak GPU memory during fn execution."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    fn()

    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()
