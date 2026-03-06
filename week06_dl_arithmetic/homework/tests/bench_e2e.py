"""
End-to-end benchmark comparing baseline vs efficient training.

This benchmark measures:
- Training time per step
- Peak memory consumption
- Throughput (tokens/sec)

Usage:
    torchrun --nproc_per_node=N tests/bench_e2e.py
"""

import argparse
import gc
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TransformerConfig
from conftest import bytes_to_mb
import train as train_module
import efficient_train as efficient_train_module

baseline_train = train_module.train
efficient_train = efficient_train_module.train


def is_master():
    """Check if this is the master process."""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_world_size():
    """Get world size for distributed training."""
    return dist.get_world_size() if dist.is_initialized() else 1


def measure_training(train_fn, args, do_warmup: bool = True) -> tuple[float, int, float]:
    """
    Run training and measure time and peak memory using CUDA events.
    
    If do_warmup is True, runs training twice - first to trigger compilation,
    then to measure actual performance.
    
    Returns:
        (avg_ms_per_step, peak_memory_bytes, total_ms)
    """
    if do_warmup:
        if is_master():
            print("  Warmup run (compiling kernels)...")
        train_fn(args)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if is_master():
            print("  Measurement run...")
    
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    train_fn(args)
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    peak_memory = torch.cuda.max_memory_allocated()
    avg_ms_per_step = total_ms / args.num_steps

    gc.collect()
    torch.cuda.empty_cache()

    return avg_ms_per_step, peak_memory, total_ms


def make_args(
    batch_size: int,
    num_steps: int,
    use_amp: bool = True,
):
    """Create args namespace for train functions."""
    class Args:
        pass
    
    args = Args()
    args.batch_size = batch_size
    args.num_epochs = 1
    args.num_samples = batch_size * num_steps
    args.num_steps = num_steps
    args.learning_rate = 1e-4
    args.weight_decay = 0.1
    args.alpha = 2.0
    args.beta3_warmup = 5
    args.alpha_warmup = 5
    args.grad_clip = 1.0
    args.use_amp = use_amp
    args.seed = 42
    args.num_workers = 0
    args.log_interval = 1000
    return args


def main():
    parser = argparse.ArgumentParser(description="E2E Training Benchmark")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per GPU")
    parser.add_argument("--num-steps", type=int, default=20, help="Number of training steps")
    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup run (faster but includes compilation time)")
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is required"
    
    do_warmup = not args.no_warmup
    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    config = TransformerConfig()

    original_setup = train_module.setup_distributed
    
    def patched_setup():
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            return rank, world_size, local_rank
        return original_setup()
    
    def noop_cleanup():
        pass
    
    train_module.setup_distributed = patched_setup
    train_module.cleanup_distributed = noop_cleanup
    efficient_train_module.setup_distributed = patched_setup
    efficient_train_module.cleanup_distributed = noop_cleanup

    if is_master():
        print("Running baseline training...")
    baseline_args = make_args(
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        use_amp=not args.no_amp,
    )
    baseline_ms, baseline_peak, baseline_total = measure_training(
        baseline_train, baseline_args, do_warmup=do_warmup
    )

    world_size = get_world_size()
    tokens_per_step = args.batch_size * config.max_seq_len * world_size

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    if is_master():
        print("Running efficient training...")
    efficient_args = make_args(
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        use_amp=not args.no_amp,
    )
    efficient_ms, efficient_peak, efficient_total = measure_training(
        efficient_train, efficient_args, do_warmup=do_warmup
    )

    master = is_master()
    
    if dist.is_initialized():
        dist.destroy_process_group()

    if master:
        def ratio(a, b):
            return a / b if b > 0 else float("inf")

        baseline_tps = tokens_per_step / (baseline_ms / 1000)
        efficient_tps = tokens_per_step / (efficient_ms / 1000)

        print(f"\nTokens per step: {tokens_per_step:,} ({world_size} GPU(s))")

        print("\nTime per step:")
        print(f"  Baseline : {baseline_ms:8.2f} ms  ({baseline_tps:,.0f} tokens/s)")
        print(f"  Efficient: {efficient_ms:8.2f} ms  ({efficient_tps:,.0f} tokens/s)")
        print(f"  Speedup  : {ratio(baseline_ms, efficient_ms):.2f}x")

        print("\nTotal training time:")
        print(f"  Baseline : {baseline_total/1000:8.2f} s")
        print(f"  Efficient: {efficient_total/1000:8.2f} s")

        print("\nPeak memory (per GPU):")
        print(f"  Baseline : {bytes_to_mb(baseline_peak):8.1f} MB")
        print(f"  Efficient: {bytes_to_mb(efficient_peak):8.1f} MB")
        print(f"  Reduction: {ratio(baseline_peak, efficient_peak):.2f}x")


if __name__ == "__main__":
    main()
