"""
Tests for Cross Entropy Loss correctness and memory consumption.
"""

import pytest
import torch

from conftest import bytes_to_mb, measure_saved_tensors_bytes
from efficient_model.loss import CrossEntropyLoss as EfficientCrossEntropyLoss
from model.loss import cross_entropy_loss as baseline_cross_entropy_loss


HIDDEN_DIM = 256
VOCAB_SIZE = 1024
BATCH_SIZE = 4
SEQ_LEN = 1024


class TestCrossEntropyLossCorrectness:
    """Test correctness of efficient CrossEntropyLoss implementation."""

    @pytest.mark.parametrize("dtype,atol,rtol", [
        (torch.bfloat16, 1e-2, 1e-2),
    ])
    def test_forward(self, device, dtype, atol, rtol):
        """Test that efficient CrossEntropyLoss forward matches baseline."""
        hidden_states = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device=device, dtype=dtype)
        lm_head_weight = torch.randn(VOCAB_SIZE, HIDDEN_DIM, device=device, dtype=dtype)
        labels = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)

        logits = hidden_states @ lm_head_weight.T
        loss_baseline = baseline_cross_entropy_loss(logits, labels)

        efficient_loss_fn = EfficientCrossEntropyLoss()
        loss_efficient = efficient_loss_fn(hidden_states, lm_head_weight, labels)
        
        diff = (loss_baseline - loss_efficient).abs().item()
        assert torch.allclose(loss_baseline, loss_efficient, atol=atol, rtol=rtol), \
            f"Forward loss mismatch! Baseline: {loss_baseline.item():.6f}, Efficient: {loss_efficient.item():.6f}, Diff: {diff:.6f}"

    @pytest.mark.parametrize("dtype,atol,rtol", [
        (torch.bfloat16, 1e-1, 1e-2),
    ])
    def test_backward(self, device, dtype, atol, rtol):
        """Test that efficient CrossEntropyLoss backward matches baseline."""
        hidden_baseline = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device=device, dtype=dtype, requires_grad=True)
        lm_head_baseline = torch.randn(VOCAB_SIZE, HIDDEN_DIM, device=device, dtype=dtype, requires_grad=True)
        labels = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)

        hidden_efficient = hidden_baseline.clone().detach().requires_grad_(True)
        lm_head_efficient = lm_head_baseline.clone().detach().requires_grad_(True)

        logits = hidden_baseline @ lm_head_baseline.T
        loss_baseline = baseline_cross_entropy_loss(logits, labels)
        loss_baseline.backward()

        efficient_loss_fn = EfficientCrossEntropyLoss()
        loss_efficient = efficient_loss_fn(hidden_efficient, lm_head_efficient, labels)
        loss_efficient.backward()

        grad_hidden_diff = (hidden_baseline.grad - hidden_efficient.grad.to(dtype)).abs().max().item()
        assert torch.allclose(hidden_baseline.grad, hidden_efficient.grad.to(dtype), atol=atol, rtol=rtol), \
            f"Backward grad_hidden mismatch! Max diff: {grad_hidden_diff}"

        grad_weight_diff = (lm_head_baseline.grad - lm_head_efficient.grad.to(dtype)).abs().max().item()
        assert torch.allclose(lm_head_baseline.grad, lm_head_efficient.grad.to(dtype), atol=atol, rtol=rtol), \
            f"Backward grad_lm_head mismatch! Max diff: {grad_weight_diff}"


class TestCrossEntropyLossMemory:
    """Test memory efficiency of efficient CrossEntropyLoss implementation."""

    def test_saved_tensors_reduction(self, device):
        """Test that efficient implementation reduces saved tensors."""
        dtype = torch.bfloat16
        
        hidden_states = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device=device, dtype=dtype, requires_grad=True)
        lm_head_weight = torch.randn(VOCAB_SIZE, HIDDEN_DIM, device=device, dtype=dtype, requires_grad=True)
        labels = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
        
        efficient_loss_fn = EfficientCrossEntropyLoss()

        def baseline_fwbw_step():
            if hidden_states.grad is not None:
                hidden_states.grad = None
            if lm_head_weight.grad is not None:
                lm_head_weight.grad = None
            logits = hidden_states @ lm_head_weight.T
            loss = baseline_cross_entropy_loss(logits, labels)
            loss.backward()

        def efficient_fwbw_step():
            if hidden_states.grad is not None:
                hidden_states.grad = None
            if lm_head_weight.grad is not None:
                lm_head_weight.grad = None
            loss = efficient_loss_fn(hidden_states, lm_head_weight, labels)
            loss.backward()

        baseline_saved = measure_saved_tensors_bytes(baseline_fwbw_step)
        efficient_saved = measure_saved_tensors_bytes(efficient_fwbw_step)

        reduction = baseline_saved / efficient_saved if efficient_saved > 0 else float("inf")

        assert reduction >= 6.2, \
            f"Efficient should use less memory! Baseline: {bytes_to_mb(baseline_saved):.2f} MB, " \
            f"Efficient: {bytes_to_mb(efficient_saved):.2f} MB, Reduction: {reduction:.2f}x"
