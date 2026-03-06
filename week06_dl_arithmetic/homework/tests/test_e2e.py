"""
End-to-end tests for the full Transformer model.
"""

import pytest
import torch

from conftest import bytes_to_mb, measure_peak_memory
from config import TransformerConfig
from model.transformer import BaselineTransformer
from model.loss import cross_entropy_loss
from efficient_model.transformer import EfficientTransformer


VOCAB_SIZE = 16000
HIDDEN_DIM = 512
NUM_HEADS = 8
NUM_LAYERS = 6
INTERMEDIATE_DIM = 1024
MAX_SEQ_LEN = 4096
BATCH_SIZE = 2
SEQ_LEN = 4096


@pytest.fixture
def config():
    """Create a small test config."""
    return TransformerConfig(
        vocab_size=VOCAB_SIZE,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        intermediate_dim=INTERMEDIATE_DIM,
        max_seq_len=MAX_SEQ_LEN,
        dropout=0.0,
    )


@pytest.fixture
def models(device, config):
    """Create baseline and efficient models with shared weights."""
    def _create_models(dtype):
        baseline = BaselineTransformer(config).to(device=device, dtype=dtype)
        efficient = EfficientTransformer(config).to(device=device, dtype=dtype)

        with torch.no_grad():
            efficient.embedding.weight.data.copy_(baseline.embedding.weight.data)

            for b_layer, e_layer in zip(baseline.layers, efficient.layers):
                e_layer.ln1.weight.data.copy_(b_layer.ln1.weight.data)
                e_layer.ln2.weight.data.copy_(b_layer.ln2.weight.data)

                e_layer.attn.qkv_proj.weight.data.copy_(
                    torch.cat([
                        b_layer.attn.q_proj.weight.data,
                        b_layer.attn.k_proj.weight.data,
                        b_layer.attn.v_proj.weight.data,
                    ], dim=0)
                )
                e_layer.attn.out_proj.weight.data.copy_(b_layer.attn.out_proj.weight.data)

                e_layer.ffn.gate_proj.weight.data.copy_(b_layer.ffn.gate_proj.weight.data)
                e_layer.ffn.up_proj.weight.data.copy_(b_layer.ffn.up_proj.weight.data)
                e_layer.ffn.down_proj.weight.data.copy_(b_layer.ffn.down_proj.weight.data)

            efficient.ln_f.weight.data.copy_(baseline.ln_f.weight.data)
            efficient.lm_head.weight.data.copy_(baseline.lm_head.weight.data)

        return baseline, efficient
    return _create_models


class TestE2ECorrectness:
    """Test end-to-end correctness of the optimized model."""

    @pytest.mark.parametrize("dtype,atol,rtol", [
        (torch.bfloat16, 5e-2, 5e-2),
    ])
    def test_forward(self, device, models, dtype, atol, rtol):
        """Test that efficient model produces same loss as baseline."""
        baseline, efficient = models(dtype)
        baseline.eval()
        efficient.eval()

        input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
        labels = input_ids.clone()

        with torch.no_grad():
            logits_baseline = baseline(input_ids)
            loss_baseline = cross_entropy_loss(logits_baseline, labels)

            loss_efficient = efficient(input_ids, labels=labels)

        loss_diff = (loss_baseline - loss_efficient).abs().item()
        assert torch.allclose(loss_baseline, loss_efficient, atol=atol, rtol=rtol), \
            f"Forward loss mismatch! Baseline: {loss_baseline.item():.4f}, Efficient: {loss_efficient.item():.4f}, Diff: {loss_diff:.4f}"

    @pytest.mark.parametrize("dtype,atol,rtol", [
        (torch.bfloat16, 5e-2, 5e-2),
    ])
    def test_backward(self, device, models, dtype, atol, rtol):
        """Test that efficient model produces same gradients as baseline."""
        baseline, efficient = models(dtype)
        baseline.train()
        efficient.train()

        input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
        labels = input_ids.clone()

        logits_baseline = baseline(input_ids)
        loss_baseline = cross_entropy_loss(logits_baseline, labels)
        loss_baseline.backward()

        loss_efficient = efficient(input_ids, labels=labels)
        loss_efficient.backward()

        emb_grad_diff = (baseline.embedding.weight.grad - efficient.embedding.weight.grad).abs().max().item()
        assert torch.allclose(baseline.embedding.weight.grad, efficient.embedding.weight.grad, atol=atol, rtol=rtol), \
            f"Embedding grad mismatch! Max diff: {emb_grad_diff}"

        lm_grad_diff = (baseline.lm_head.weight.grad - efficient.lm_head.weight.grad).abs().max().item()
        assert torch.allclose(baseline.lm_head.weight.grad, efficient.lm_head.weight.grad, atol=atol, rtol=rtol), \
            f"LM head grad mismatch! Max diff: {lm_grad_diff}"


class TestE2EMemory:
    """Test memory efficiency of the optimized model."""

    def test_peak_memory(self, device, models):
        """Test that efficient model uses less peak memory."""
        dtype = torch.bfloat16
        baseline, efficient = models(dtype)
        baseline.train()
        efficient.train()

        input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
        labels = input_ids.clone()

        def baseline_fwbw():
            baseline.zero_grad(set_to_none=True)
            logits = baseline(input_ids)
            loss = cross_entropy_loss(logits, labels)
            loss.backward()

        def efficient_fwbw():
            efficient.zero_grad(set_to_none=True)
            loss = efficient(input_ids, labels=labels)
            loss.backward()

        baseline_peak = measure_peak_memory(baseline_fwbw)

        baseline.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        efficient_peak = measure_peak_memory(efficient_fwbw)

        ratio = baseline_peak / efficient_peak if efficient_peak > 0 else float("inf")
        min_ratio = 7.76

        assert ratio >= min_ratio, \
            f"Expected at least {min_ratio}x memory reduction, got {ratio:.2f}x " \
            f"(Baseline: {bytes_to_mb(baseline_peak):.2f} MB, Efficient: {bytes_to_mb(efficient_peak):.2f} MB)"
