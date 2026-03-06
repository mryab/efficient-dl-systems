"""
Tests for SwiGLU Feed-Forward Network correctness and memory consumption.
"""

import pytest
import torch

from conftest import bytes_to_mb, measure_saved_tensors_bytes, measure_peak_memory
from efficient_model.swiglu import SwiGLUFeedForward as EfficientSwiGLU
from model.swiglu import SwiGLUFeedForward as BaselineSwiGLU


HIDDEN_DIM = 256
INTERMEDIATE_DIM = 512
BATCH_SIZE = 4
SEQ_LEN = 1024


@pytest.fixture
def models(device):
    """Create baseline and efficient models with shared weights."""
    def _create_models(dtype):
        baseline = BaselineSwiGLU(HIDDEN_DIM, INTERMEDIATE_DIM).to(device=device, dtype=dtype)
        efficient = EfficientSwiGLU(HIDDEN_DIM, INTERMEDIATE_DIM).to(device=device, dtype=dtype)

        efficient.gate_proj.weight.data.copy_(baseline.gate_proj.weight.data)
        efficient.up_proj.weight.data.copy_(baseline.up_proj.weight.data)
        efficient.down_proj.weight.data.copy_(baseline.down_proj.weight.data)

        return baseline, efficient
    return _create_models


class TestSwiGLUCorrectness:
    """Test correctness of efficient SwiGLU implementation."""

    @pytest.mark.parametrize("dtype,atol,rtol", [
        (torch.float32, 5e-5, 5e-5),
    ])
    def test_forward(self, device, models, dtype, atol, rtol):
        """Test that efficient SwiGLU forward matches baseline."""
        baseline, efficient = models(dtype)

        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device=device, dtype=dtype)

        y_baseline = baseline(x)
        y_efficient = efficient(x)

        max_diff = (y_baseline - y_efficient).abs().max().item()
        assert torch.allclose(y_baseline, y_efficient, atol=atol, rtol=rtol), \
            f"Forward mismatch! Max diff: {max_diff}"

    @pytest.mark.parametrize("dtype,atol,rtol", [
        (torch.float32, 5e-5, 5e-5),
    ])
    def test_backward(self, device, models, dtype, atol, rtol):
        """Test that efficient SwiGLU backward matches baseline."""
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
        assert torch.allclose(x_baseline.grad, x_efficient.grad, atol=atol, rtol=rtol), \
            f"Backward grad_x mismatch! Max diff: {grad_x_diff}"

        grad_gate_diff = (baseline.gate_proj.weight.grad - efficient.gate_proj.weight.grad).abs().max().item()
        assert torch.allclose(baseline.gate_proj.weight.grad, efficient.gate_proj.weight.grad, atol=atol, rtol=rtol), \
            f"Backward grad_gate_proj mismatch! Max diff: {grad_gate_diff}"

        grad_up_diff = (baseline.up_proj.weight.grad - efficient.up_proj.weight.grad).abs().max().item()
        assert torch.allclose(baseline.up_proj.weight.grad, efficient.up_proj.weight.grad, atol=atol, rtol=rtol), \
            f"Backward grad_up_proj mismatch! Max diff: {grad_up_diff}"

        grad_down_diff = (baseline.down_proj.weight.grad - efficient.down_proj.weight.grad).abs().max().item()
        assert torch.allclose(baseline.down_proj.weight.grad, efficient.down_proj.weight.grad, atol=atol, rtol=rtol), \
            f"Backward grad_down_proj mismatch! Max diff: {grad_down_diff}"


class TestSwiGLUMemory:
    """Test memory efficiency of efficient SwiGLU implementation."""

    def test_saved_tensors_reduction(self, device, models):
        """Test that efficient implementation reduces saved tensors by at least 3.4x."""
        dtype = torch.float32
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
        min_reduction = 3.6

        assert reduction >= min_reduction, \
            f"Expected at least {min_reduction}x reduction, got {reduction:.2f}x " \
            f"(Baseline: {bytes_to_mb(baseline_saved):.2f} MB, Efficient: {bytes_to_mb(efficient_saved):.2f} MB)"

    def test_fwbw_peak_memory(self, device, models):
        """Test that efficient forward+backward uses less peak memory."""
        dtype = torch.bfloat16
        baseline, efficient = models(dtype)

        def fwbw_step(model, x):
            y = model(x)
            y.sum().backward()

        x_baseline = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device=device, dtype=dtype, requires_grad=True)
        baseline_peak = measure_peak_memory(
            lambda: fwbw_step(baseline, x_baseline)
        )

        del x_baseline
        baseline.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        x_efficient = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device=device, dtype=dtype, requires_grad=True)
        efficient_peak = measure_peak_memory(
            lambda: fwbw_step(efficient, x_efficient)
        )

        diff_mb = bytes_to_mb(baseline_peak) - bytes_to_mb(efficient_peak)
        min_diff_mb = 20

        assert diff_mb >= min_diff_mb, \
            f"Expected at least {min_diff_mb} MB reduction, got {diff_mb:.2f} MB " \
            f"(Baseline: {bytes_to_mb(baseline_peak):.2f} MB, Efficient: {bytes_to_mb(efficient_peak):.2f} MB)"
