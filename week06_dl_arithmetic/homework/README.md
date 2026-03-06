# Week 6 Home Assignment: Optimizing Transformer Training

This assignment focuses on applying kernel fusion, efficient operators, and distributed training techniques to optimize transformer training pipeline.

## Overview

You are given a **transformer training script** that works correctly but uses non-optimized implementations. Your task is to apply various optimizations covered in the lecture and seminar, verify that model quality is preserved, and analyze the performance improvements.

### Installation

```bash
uv pip install -e ".[flash]"
```

### Project Structure

- `model/` - Clean baseline implementation (read-only reference)
- `efficient_model/` - Your workspace - implement optimizations here
- `train.py` - Baseline training script using `model/`
- `efficient_train.py` - Training script using `efficient_model/`

### Model Inefficiencies (`model/transformer.py`)

1. Separate lm_head + F.cross_entropy (instead of Fused Linear Cross Entropy)
2. Unfused RoPE and Vanilla attention + separate Q/K/V projections (instead of Flash Attention + fused QKV + fused RoPE)
3. Unfused Zero-Centered RMSNorm
4. Unfused gpt-oss SwiGLU (separate clamp, sigmoid, multiply kernels)

### Training Inefficiencies (`train.py`)

1. Non-fused AdEMAMix optimizer
2. DDP instead of FSDP (no parameter sharding)


## Part 1: Model Optimizations (5.5 points)

Apply all optimizations below. Points are awarded only if tests pass.

### 1.1 Fused Linear Cross Entropy (0.5 points)

Recall from lecture: fusing the linear projection with cross entropy avoids materializing the full logits tensor (vocab_size can be huge!). Apply this optimization by replacing separate lm_head + F.cross_entropy with `LigerFusedLinearCrossEntropyLoss` from Liger-Kernels.

**Hint:** Be careful when combining it with FSDP.

**Test:** `pytest tests/test_loss.py`

### 1.2 Efficient Attention (1 point)

Replace vanilla attention with optimized implementation:
- [Flash Attention 2](https://github.com/Dao-AILab/flash-attention) for efficient attention computation
- Fused QKV projection (single matmul instead of three)
- Fused RoPE using [`apply_rotary_emb`](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/layers/rotary.py) from Flash Attention library 

**Test:** `pytest tests/test_attention.py`

### 1.3 Fused & Memory-Efficient Zero-Centered RMSNorm (1.5 points)

Unlike standard RMSNorm (`y = x/rms(x) * weight` with weight init to 1), Zero-Centered uses:
- `y = x/rms(x) * (1 + weight)` with weight initialized to **zeros**
- This is more stable: weight decay pushes toward identity (1) rather than zero

**Your task:** Two separate optimizations:
1. **Fused**: Combine multiple elementwise ops into one kernel
2. **Memory-efficient**: Reduce memory footprint by being smart about what's saved for backward

**Hints:**
- Don't forget `(1 + weight)` must be computed in fp32 to avoid precision loss in bf16
- You can use `@torch.compile` to fuse elementwise ops

**Test:** `pytest tests/test_rmsnorm.py`

### 1.4 Fused & Memory-Efficient gpt-oss SwiGLU (2 points)

The baseline uses gpt-oss variant which differs from standard SwiGLU:
- `gate * sigmoid(gate * alpha)` instead of `silu(gate)`
- `(up + 1) * glu` instead of `up * glu`
- Clamping for numerical stability

**Your task:** Two separate optimizations:
1. **Fused**: Triton kernel for the activation (clamp + sigmoid + multiply)
2. **Memory-efficient**: Reduce memory footprint by being smart about what's saved for backward

If you're new to Triton, start with the [Triton Language Guide](https://triton-lang.org/main/python-api/triton.language.html) for available operations.

**Important:** On the backward pass you are only allowed to recompute **element-wise operations** (like clamp, sigmoid, multiply, etc.). Do not recompute matrix multiplications. Full recomputation with matmuls is too expensive and we don't allow that.

**Hints:**
- Start from Liger's standard SwiGLU kernel as reference
- Use `tl.minimum`/`tl.maximum` for clamping
- Reuse allocations in a smart way (tests check for no unnecessary allocations)

**Test:** `pytest tests/test_swiglu.py`

### 1.5 End to End Test (0.5 points)

After completing all 4 model optimizations, run `pytest tests/test_e2e.py` to verify correctness and memory reduction.

**Hint:** There's a hidden memory inefficiency near the attention in baseline - your memory results won't match targets without finding it.

## Part 2: Training Optimizations (4.5 points)

### 2.1 Fused Optimizer (3.5 points)

Usually, the step time of **element-wise optimizers** is negligible compared to the forward–backward time (for **matrix-wise optimizers** the situation is much more interesting; see the links at the end of this section). However, this efficiency is largely achieved thanks to **horizontal** and **vertical fusing** of operations. If you use a naive for-loop implementation instead, the step time can increase by multiple times, which can already become a problem.

In this assignment, you need to implement an efficient version of the **AdEMAMix** optimizer (https://arxiv.org/abs/2409.03137) in `efficient_optimizer/ademamix.py`.

For this homework, you may assume that gradients either exist for **all** parameters in a `param_group` at the same time, or for none of them. More formally: if `p.grad is not None` for some parameter `p` in the group, then `p.grad is not None` also holds for all other parameters in the same group.

### Tasks

#### 1) Horizontal fusing (0.75 points)
Use `torch._foreach_*` operations to process all tensors at once instead of looping.

#### 2) Vertical fusing (0.75 points)
Use `torch.compile` to fuse operations into efficient kernels. Check that the `torch.compile` graph has no breaks.

#### 3) Reduce the number of kernels (1 point)
Make sure that the number of Triton kernels is **no more than** in `torch.optim.AdamW(foreach=True)` when combined with `torch.compile`.

#### 4) One kernel (1 point)
Analyze and explain why `torch.optim.AdamW(foreach=True)` still produces **more than one** kernel. Taking that observation into account, rewrite your implementation so that it produces **only one** kernel.

**Test:** `pytest tests/test_optimizer.py`

For subtasks 1-2, the test should pass. For subtasks 3-4, use the same test setup to verify how many kernels your implementation produces.

#### Hints
- Look at `torch._foreach_lerp_`, `torch._foreach_mul_`, `torch._foreach_addcmul_`, etc.
- Some `_foreach_` operations may be incompatible with `torch.compile`
- Compile AdamW's step with `torch.compile(opt.step)` for subtasks 3 and 4.
- Think about the relationship between **recompilation** and **input types** for `torch.compile`d functions.
- You may want to inspect the source implementations of different optimizers in PyTorch.
- You may need to modify your implementation a bit to be compatible with FSDP2 from the next task.

To inspect compiled kernels, add before imports:
```python
import os
os.environ["TORCH_LOGS"] = "+output_code"
os.environ["TORCH_LOGS_OUT"] = "compiled_kernels.log"
```

#### Useful links
- [AdamW PyTorch implementation](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/optim/adamw.py#L20)
- [GPU Mode: Optimizing Optimizers](https://www.youtube.com/watch?v=hIop0mWKPHc)

#### Additional materials on efficient use of matrix-wise optimizers for large-scale training
- [Essential AI: Blog on efficient communications for Muon](https://www.essential.ai/research/infra)
- [Efficient Muon implementation in Dion library](https://github.com/microsoft/dion/)

**Test:** `pytest tests/test_optimizer.py`

### 2.2 FSDP (1 point)

Replace DDP with Fully Sharded Data Parallel. Don't forget to ensure Liger kernels and your fused optimizer are compatible with FSDP.

## Part 3: Theoretical Calculators (3 points)

Implement calculators in `calculators/baseline_calculator.py` and `calculators/efficient_calculator.py`.

**Hints for self-validation:**
- You can use `torch.cuda.memory._record_memory_history()` and `torch.cuda.memory._dump_snapshot()` to capture memory snapshots and compare against your calculations
- You can use [torch.utils.flop_counter](https://pytorch.org/docs/stable/torch.utils.flop_counter.html) to verify your FLOP calculations

### 3.1 Memory Calculator (1.5 points)

Calculate peak memory for both implementations:
- **Baseline:** params + gradients + optimizer states + activations (all intermediates saved)
- **Efficient:** params + gradients + optimizer states + activations (selective recomputation)

Compare saved tensors size and peak memory, verify against test assertions, is the difference expected?

### 3.2 Time and Comms Calculator (1.5 points)

Calculate for distributed training (both baseline and efficient implementations):
- **Time:** Roofline model (memory-bound vs compute-bound)
- **Comms:** communication volume (all-reduce, all-gather, reduce-scatter)
- **Overlap analysis:** Account for communication/compute overlap in your time estimates (e.g., estimate the non-overlapped region that adds to total training time, or any other reasonable approach). Think about how you can understand if training is communication-bound.

## Part 4: Report and Analysis (1 point)

Create a report (`report.md` or `report.pdf`) with:

1. **Memory table** - For each operation (SwiGLU, RMSNorm, FA, Loss): calculated peak memory (baseline vs efficient)
2. **Time table** - For each operation: calculated compute time comparison (baseline vs efficient)
3. **Validation** - Compare end-to-end total training time and peak memory of efficient implementation against calculator predictions. Explain any discrepancies.

**Benchmark:** Use `torchrun --nproc_per_node=2 tests/bench_e2e.py` to measure actual training time and memory peak for comparison (the configuration is predefined there).
