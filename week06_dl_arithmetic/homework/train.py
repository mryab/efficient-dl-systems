"""
Training Script with DDP

Usage:
    # Single GPU
    python train.py
    
    # Multi-GPU with DDP
    torchrun --nproc_per_node=2 train.py
"""

import os
import time
import argparse
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from config import TransformerConfig
from model import BaselineTransformer
from optimizer.ademamix import AdEMAMix


class SyntheticDataset(Dataset):
    """
    Synthetic dataset generating random token sequences.
    Used for benchmarking.
    """
    
    def __init__(
        self, 
        num_samples: int, 
        seq_len: int, 
        vocab_size: int,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.seed = seed
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        generator = torch.Generator().manual_seed(self.seed + idx)
        tokens = torch.randint(
            0, self.vocab_size, (self.seq_len,), generator=generator
        )
        return tokens


def setup_distributed():
    """Initialize distributed training if available."""
    if 'RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_lr(step: int, warmup_steps: int, max_lr: float, total_steps: int) -> float:
    """Linear warmup followed by cosine decay."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max_lr * 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())


def train(args):
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    is_master = rank == 0

    if is_master:
        print(f"Training with {world_size} GPU(s)")
        print(f"Device: {device}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    config = TransformerConfig()
    if is_master:
        print(f"Model config: {config}")

    model = BaselineTransformer(config).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    raw_model = model.module if world_size > 1 else model
    num_params = sum(p.numel() for p in model.parameters())
    if is_master:
        print(f"Model parameters: {num_params:,}")

    optimizer = AdEMAMix(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999, 0.9999),
        alpha=args.alpha,
        beta3_warmup=args.beta3_warmup,
        alpha_warmup=args.alpha_warmup,
        weight_decay=args.weight_decay,
    )

    dataset = SyntheticDataset(
        num_samples=args.num_samples,
        seq_len=config.max_seq_len,
        vocab_size=config.vocab_size,
        seed=args.seed,
    )

    sampler = DistributedSampler(dataset, shuffle=True) if world_size > 1 else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    total_steps = args.num_epochs * len(dataloader)
    warmup_steps = int(0.1 * total_steps)

    if is_master:
        print(f"\nTraining for {args.num_epochs} epochs ({total_steps} steps)")
        print(f"Batch size: {args.batch_size} x {world_size} = {args.batch_size * world_size}")
        print(f"Sequence length: {config.max_seq_len}")
        print("-" * 60)

    scaler = torch.amp.GradScaler('cuda', enabled=args.use_amp)
    autocast_ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16) if args.use_amp else nullcontext()

    model.train()
    global_step = 0
    total_tokens = 0
    start_time = time.time()
    log_interval_tokens = 0
    log_interval_start = time.time()

    for epoch in range(args.num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_steps = 0

        for batch_idx, input_ids in enumerate(dataloader):
            input_ids = input_ids.to(device)
            labels = input_ids.clone()

            lr = get_lr(global_step, warmup_steps, args.learning_rate, total_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            optimizer.zero_grad()

            with autocast_ctx:
                logits = raw_model(input_ids)
                loss = raw_model.compute_loss(logits, labels)

            scaler.scale(loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            batch_tokens = input_ids.numel() * world_size
            total_tokens += batch_tokens
            log_interval_tokens += batch_tokens
            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            if is_master and global_step % args.log_interval == 0:
                elapsed = time.time() - log_interval_start
                tokens_per_sec = log_interval_tokens / elapsed
                avg_loss = epoch_loss / epoch_steps
                
                print(
                    f"Step {global_step:5d} | "
                    f"Epoch {epoch+1}/{args.num_epochs} | "
                    f"Loss {avg_loss:.4f} | "
                    f"LR {lr:.2e} | "
                    f"Tokens/s {tokens_per_sec:,.0f}"
                )

                log_interval_tokens = 0
                log_interval_start = time.time()
        
        if is_master:
            avg_epoch_loss = epoch_loss / epoch_steps
            print(f"Epoch {epoch+1} complete | Avg Loss: {avg_epoch_loss:.4f}")
    
    total_time = time.time() - start_time

    cleanup_distributed()
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Baseline Transformer Training')

    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size per GPU')
    parser.add_argument('--num-epochs', type=int, default=1,
                        help='Number of training epochs')
    parser.add_argument('--num-samples', type=int, default=1000,
                        help='Number of synthetic samples')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Peak learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.1,
                        help='Weight decay')
    parser.add_argument('--alpha', type=float, default=2.0,
                        help='AdEMAMix alpha coefficient for mixing slow and fast EMAs')
    parser.add_argument('--beta3-warmup', type=int, default=None,
                        help='Number of warmup steps for beta3')
    parser.add_argument('--alpha-warmup', type=int, default=None,
                        help='Number of warmup steps for alpha')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='Gradient clipping (0 to disable)')
    parser.add_argument('--use-amp', action='store_true',
                        help='Use automatic mixed precision')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader workers')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Log every N steps')
    
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
