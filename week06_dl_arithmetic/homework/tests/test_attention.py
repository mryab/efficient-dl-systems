"""
Tests for Multi-Head Attention correctness and memory consumption.
"""

import pytest
import torch

from conftest import bytes_to_mb, measure_saved_tensors_bytes, measure_peak_memory
from config import TransformerConfig
from efficient_model.attention import MultiHeadAttention as EfficientAttention
from model.attention import MultiHeadAttention as BaselineAttention


HIDDEN_DIM = 256
NUM_HEADS = 8
BATCH_SIZE = 4
SEQ_LEN = 1024


@pytest.fixture
def config():
    """Create a test config with smaller dimensions."""
    return TransformerConfig(
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        max_seq_len=SEQ_LEN * 2,
        dropout=0.0,
    )


@pytest.fixture
def models(device, config):
    """Create baseline and efficient models with shared weights."""
    def _create_models(dtype):
        baseline = BaselineAttention(config).to(device=device, dtype=dtype)
        efficient = EfficientAttention(config).to(device=device, dtype=dtype)

        with torch.no_grad():
            efficient.qkv_proj.weight.data.copy_(
                torch.cat([
                    baseline.q_proj.weight.data,
                    baseline.k_proj.weight.data,
                    baseline.v_proj.weight.data,
                ], dim=0)
            )
            efficient.out_proj.weight.data.copy_(baseline.out_proj.weight.data)

        baseline.eval()
        efficient.eval()

        return baseline, efficient
    return _create_models


class TestAttentionCorrectness:
    """Test correctness of efficient attention implementation."""

    @pytest.mark.parametrize("dtype,atol,rtol", [
        (torch.bfloat16, 5e-2, 5e-2),
    ])
    def test_forward(self, device, models, dtype, atol, rtol):
        """Test that efficient attention forward matches baseline."""
        baseline, efficient = models(dtype)

        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device=device, dtype=dtype)

        with torch.no_grad():
            y_baseline = baseline(x)
            y_efficient = efficient(x)

        max_diff = (y_baseline - y_efficient).abs().max().item()
        assert torch.allclose(y_baseline, y_efficient, atol=atol, rtol=rtol), \
            f"Forward mismatch! Max diff: {max_diff}"

    @pytest.mark.parametrize("dtype,atol,rtol", [
        (torch.bfloat16, 5e-2, 5e-2),
    ])
    def test_backward(self, device, models, dtype, atol, rtol):
        """Test that efficient attention backward matches baseline."""
        baseline, efficient = models(dtype)
        baseline.train()
        efficient.train()

        x_baseline = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device=device, dtype=dtype)
        x_baseline.requires_grad = True
        x_efficient = x_baseline.clone().detach().requires_grad_(True)

        y_baseline = baseline(x_baseline)
        y_efficient = efficient(x_efficient)

        grad_output = torch.randn_like(y_baseline)
        y_baseline.backward(grad_output)
        y_efficient.backward(grad_output)

        # grad_x_diff = (x_baseline.grad - x_efficient.grad).abs().max().item()
        # assert torch.allclose(x_baseline.grad, x_efficient.grad, atol=atol, rtol=rtol), \
        #     f"Backward grad_x mismatch! Max diff: {grad_x_diff}"

        baseline_qkv_grad = torch.cat([
            baseline.q_proj.weight.grad,
            baseline.k_proj.weight.grad,
            baseline.v_proj.weight.grad,
        ], dim=0)
        grad_qkv_diff = (baseline_qkv_grad - efficient.qkv_proj.weight.grad).abs().max().item()
        assert torch.allclose(baseline_qkv_grad, efficient.qkv_proj.weight.grad, atol=atol, rtol=rtol), \
            f"Backward grad_qkv mismatch! Max diff: {grad_qkv_diff}"

        grad_out_diff = (baseline.out_proj.weight.grad - efficient.out_proj.weight.grad).abs().max().item()
        assert torch.allclose(baseline.out_proj.weight.grad, efficient.out_proj.weight.grad, atol=atol, rtol=rtol), \
            f"Backward grad_out_proj mismatch! Max diff: {grad_out_diff}"


class TestAttentionMemory:
    """Test memory efficiency of efficient attention implementation."""

    def test_saved_tensors_reduction(self, device, models):
        """Test that efficient implementation reduces saved tensors."""
        dtype = torch.bfloat16
        baseline, efficient = models(dtype)
        baseline.train()
        efficient.train()

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
        min_reduction = 6.0

        assert reduction >= min_reduction, \
            f"Expected at least {min_reduction}x reduction, got {reduction:.2f}x " \
            f"(Baseline: {bytes_to_mb(baseline_saved):.2f} MB, Efficient: {bytes_to_mb(efficient_saved):.2f} MB)"

    def test_fwbw_peak_memory(self, device, models):
        """Test that efficient forward+backward uses less peak memory."""
        dtype = torch.bfloat16
        baseline, efficient = models(dtype)
        baseline.train()
        efficient.train()

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
        min_diff_mb = 50

        assert diff_mb >= min_diff_mb, \
            f"Expected at least {min_diff_mb} MB reduction, got {diff_mb:.2f} MB " \
            f"(Baseline: {bytes_to_mb(baseline_peak):.2f} MB, Efficient: {bytes_to_mb(efficient_peak):.2f} MB)"
