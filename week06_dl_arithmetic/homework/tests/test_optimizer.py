"""
Tests for optimizer step correctness.
"""

import pytest
import torch
import torch.nn as nn

from optimizer.ademamix import AdEMAMix as AdemamixForloop
from efficient_optimizer.ademamix import AdEMAMix as AdemamixForeach

import torch._dynamo as dynamo
dynamo.config.recompile_limit = 8

HIDDEN_DIM = 16
NUM_LAYERS = 3
NUM_STEPS = 100

def _build_model(device: torch.device, dtype: torch.dtype) -> nn.Module:
    torch.manual_seed(0)
    layers = [nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=True) for _ in range(NUM_LAYERS)]
    return nn.Sequential(*layers).to(device=device, dtype=dtype)


def _assert_models_close(model_a: nn.Module, model_b: nn.Module, step: int, rtol: float=1e-5, atol: float=1e-6) -> None:
    a = dict(model_a.named_parameters())
    b = dict(model_b.named_parameters())
    assert a.keys() == b.keys()

    for name in a.keys():
        pa, pb = a[name].data, b[name].data
        max_diff = (pa - pb).abs().max().item()
        assert torch.allclose(pa, pb, atol=atol, rtol=rtol), (
            f"Param mismatch at step={step}, name={name}, max_diff={max_diff}"
        )


def _apply_random_grads(model_a: nn.Module, model_b: nn.Module) -> None:
    torch.manual_seed(0)
    a = dict(model_a.named_parameters())
    b = dict(model_b.named_parameters())
    for name in a.keys():
        g = torch.randn_like(a[name].data)
        a[name].grad = g
        b[name].grad = g.clone()


class TestCorrectness:
    """Test correctness of efficient AdEMAMix implementation."""
    @pytest.mark.parametrize("dtype,atol,rtol", [
        (torch.float32, 1e-6, 1e-5),
    ])
    def test_steps_match(self, device, dtype, atol, rtol):
        model_baseline = _build_model(device, dtype)
        model_efficient = _build_model(device, dtype)

        opt_baseline = AdemamixForloop(model_baseline.parameters(), lr=1e-2, weight_decay=0.1, alpha_warmup=51, beta3_warmup=51)
        opt_efficient = AdemamixForeach(model_efficient.parameters(), lr=1e-2, weight_decay=0.1, alpha_warmup=51, beta3_warmup=51)

        _assert_models_close(model_baseline, model_efficient, step=0, atol=atol, rtol=rtol)

        for step in range(1, NUM_STEPS + 1):
            _apply_random_grads(model_baseline, model_efficient)
            opt_baseline.step()
            opt_efficient.step()
            _assert_models_close(model_baseline, model_efficient, step=step, atol=atol, rtol=rtol)
