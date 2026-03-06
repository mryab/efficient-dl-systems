"""
Tests for RMSNorm correctness and memory consumption.
"""

import pytest
import torch

from conftest import bytes_to_mb, measure_saved_tensors_bytes
from efficient_model.norm import RMSNorm as EfficientRMSNorm
from model.norm import RMSNorm as BaselineRMSNorm


HIDDEN_DIM = 256
BATCH_SIZE = 4
SEQ_LEN = 1024


@pytest.fixture
def models(device):
    """Create baseline and efficient models with shared weights."""
    def _create_models(dtype):
        baseline = BaselineRMSNorm(HIDDEN_DIM).to(device=device, dtype=dtype)
        efficient = EfficientRMSNorm(HIDDEN_DIM).to(device=device, dtype=dtype)
        efficient.weight.data.copy_(baseline.weight.data)
        return baseline, efficient
    return _create_models


class TestRMSNormCorrectness:
    """Test correctness of efficient RMSNorm implementation."""

    @pytest.mark.parametrize("dtype,atol,rtol", [
        (torch.bfloat16, 2e-2, 2e-2),
    ])
    def test_forward(self, device, models, dtype, atol, rtol):
        """Test that efficient RMSNorm forward matches baseline."""
        baseline, efficient = models(dtype)

        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device=device, dtype=dtype)

        y_baseline = baseline(x)
        y_efficient = efficient(x)

        max_diff = (y_baseline - y_efficient).abs().max().item()
        assert torch.allclose(y_baseline, y_efficient, atol=atol, rtol=rtol), \
            f"Forward mismatch! Max diff: {max_diff}"

    @pytest.mark.parametrize("dtype,rtol", [
        (torch.bfloat16, 2e-2),
    ])
    def test_backward(self, device, models, dtype, rtol):
        """Test that efficient RMSNorm backward matches baseline."""
        baseline, efficient = models(dtype)

        x_baseline = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device=device, dtype=dtype)
        x_baseline.requires_grad = True
        x_efficient = x_baseline.clone().detach().requires_grad_(True)

        y_baseline = baseline(x_baseline)
        y_efficient = efficient(x_efficient)

        grad_output = torch.randn_like(y_baseline)
        y_baseline.backward(grad_output)
        y_efficient.backward(grad_output)

        grad_x_diff = (x_baseline.grad - x_efficient.grad).abs().max().item()
        assert torch.allclose(x_baseline.grad, x_efficient.grad, atol=2e-2, rtol=rtol), \
            f"Backward grad_x mismatch! Max diff: {grad_x_diff}"

        grad_w_diff = (baseline.weight.grad - efficient.weight.grad).abs().max().item()
        assert torch.allclose(baseline.weight.grad, efficient.weight.grad, atol=5e-1, rtol=rtol), \
            f"Backward grad_weight mismatch! Max diff: {grad_w_diff}"


class TestRMSNormMemory:
    """Test memory efficiency of efficient RMSNorm implementation."""

    def test_saved_tensors_reduction(self, device, models):
        """Test that efficient implementation reduces saved tensors by at least 7.9x."""
        dtype = torch.bfloat16
        baseline, efficient = models(dtype)

        x_fwbw = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device=device, dtype=dtype, requires_grad=True)

        def baseline_fwbw_step():
            baseline.zero_grad(set_to_none=True)
            if x_fwbw.grad is not None:
                x_fwbw.grad = None
            y = baseline(x_fwbw)
            y.sum().backward()

        def efficient_fwbw_step():
            efficient.zero_grad(set_to_none=True)
            if x_fwbw.grad is not None:
                x_fwbw.grad = None
            y = efficient(x_fwbw)
            y.sum().backward()

        baseline_saved = measure_saved_tensors_bytes(baseline_fwbw_step, exclude_tensors=list(baseline.parameters()))
        efficient_saved = measure_saved_tensors_bytes(efficient_fwbw_step, exclude_tensors=list(efficient.parameters()))

        reduction = baseline_saved / efficient_saved if efficient_saved > 0 else float("inf")
        min_reduction = 7.9

        assert reduction >= min_reduction, \
            f"Expected at least {min_reduction}x reduction, got {reduction:.2f}x " \
            f"(Baseline: {bytes_to_mb(baseline_saved):.2f} MB, Efficient: {bytes_to_mb(efficient_saved):.2f} MB)"
