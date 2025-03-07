import functools
import logging
import os
from functools import partial
from pathlib import Path

import fire
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO)

def trace_handler(prof, trace_dir: str):
    curr_trace_dir_name = "iteration_" + str(prof.step_num)
    curr_trace_dir = os.path.join(trace_dir, curr_trace_dir_name)
    if not os.path.exists(curr_trace_dir):
        os.makedirs(curr_trace_dir, exist_ok=True)
    prof.export_chrome_trace(
        f"{curr_trace_dir}/rank{torch.distributed.get_rank()}_trace.json"
    )


class Block(nn.Module):
    def __init__(self, dim: int, bias=True):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim, bias=bias)
        self.linear1 = nn.Linear(dim, dim * 2, bias=bias)
        self.linear2 = nn.Linear(dim * 2, dim, bias=bias)

    def forward(self, x: torch.Tensor):
        input_x = x
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return input_x + x, {n: p.detach().clone() for n, p in self.named_parameters()}


class Model(nn.Module):
    def __init__(
        self,
        dim: int,
        num_blocks: int,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(Block(dim) for _ in range(num_blocks))

    def forward(self, x: torch.Tensor):
        activations = []
        weights = {}
        for block_id, block in enumerate(self.blocks):
            x, block_weights = block(x)
            activations.append(x.detach().clone())
            weights.update(
                {f"blocks.{block_id}.{k}": v for k, v in block_weights.items()}
            )
        return x, activations, weights


def train(
    framework: str,
    dump_dir: str,
    num_steps: int = 50,
    max_lr: float = 1e-3,
    warmup: float = 0.3,
    dim: int = 256,
    num_blocks: int = 15,
    num_gas_steps: int = 1,
    checkpoint: bool = False,
    eval_interval: int | None = 50,
    master_dtype: torch.dtype = torch.float32,
    param_dtype: torch.dtype = torch.bfloat16,
    reduce_dtype: torch.dtype = torch.float32,
    reshard_after_forward: bool = True,
    profile_freq: int = 10,
    profile_active: int = 1,
    profile_warmup: int = 3,
):
    if framework == "fsdp-2":
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard

        # logging.getLogger("torch.distributed._composable.fsdp").setLevel(logging.DEBUG)
        logging.info("Using FSDP2.")
    elif framework == "hw-fsdp":
        from hw_fsdp import FSDPCommContext, MixedPrecisionPolicy, fully_shard

        # logging.getLogger("hw_fsdp").setLevel(logging.DEBUG)
        logging.info("Using hwFSDP.")

    torch.cuda.set_device(int(os.getenv("LOCAL_RANK")))
    dist.distributed_c10d.init_process_group()
    mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=["dp"])

    if dist.get_rank() == 0:
        torch.cuda.memory._record_memory_history()

    torch.manual_seed(42)
    with torch.device("cuda"):
        model = Model(dim, num_blocks).to(master_dtype)

    if checkpoint:
        apply_activation_checkpointing(model, check_fn=lambda m: m in model.blocks)

    if framework == "hw-fsdp":
        comm_ctx = FSDPCommContext(mesh.device_type)
    fully_shard_module = partial(
        fully_shard,
        mesh=mesh["dp"],
        reshard_after_forward=reshard_after_forward,
        mp_policy=MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
        ),
    )
    for block_id, block in enumerate(model.blocks):
        if framework == "fsdp-2":
            fully_shard_module(block)
        elif framework == "hw-fsdp":
            fully_shard_module(
                block,
                comm_ctx=comm_ctx,
                module_fqn=f"blocks.{block_id}",
            )

    optim = AdamW(model.parameters(), lr=max_lr, fused=True)
    lr_scheduler = OneCycleLR(
        optim,
        max_lr=max_lr,
        total_steps=num_steps,
        anneal_strategy="linear",
        pct_start=warmup,
    )

    dump_dir = Path(dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    loss_file = open(dump_dir / f"{framework.replace("-", "_")}.txt", "w")
    states_dir = dump_dir / "states"
    states_dir.mkdir(parents=True, exist_ok=True)
    mem_snap_dir = dump_dir / "memory_snapshot"
    mem_snap_dir.mkdir(parents=True, exist_ok=True)

    torch.cuda.memory._dump_snapshot(
        mem_snap_dir / f"rank{torch.distributed.get_rank()}_memory_snapshot.pickle"
    )

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=profile_freq - (profile_active + profile_warmup),
            warmup=profile_warmup,
            active=profile_active,
        ),
        on_trace_ready=functools.partial(
            trace_handler,
            trace_dir=dump_dir / "profile_trace",
        ),
    ) as profiler:
        for iter in (pbar := tqdm(range(num_steps))):
            total_loss = 0.0
            for gas_step in range(num_gas_steps):
                batch = torch.full(
                    (2, dim),
                    fill_value=1
                    + 1e-1 * (1 + mesh["dp"].get_local_rank())
                    + 1e-2 * (1 + gas_step),
                    dtype=param_dtype,
                    device="cuda",
                )
                output, activations, weights = model(batch)
                for i, activation in enumerate(activations):
                    torch.save(
                        activation,
                        states_dir / f"act_{iter:03d}_{i:02d}_{dist.get_rank()}.pt",
                    )
                torch.save(
                    weights, states_dir / f"weight_{iter:03d}_{dist.get_rank()}.pt"
                )

                loss = output.sum().sub(1).square()
                loss = loss / num_gas_steps if framework != "ya-fsdp-1" else loss
                loss.backward()
                dist.all_reduce(loss)
                total_loss += loss.item()
                pbar.set_postfix(loss=f"{total_loss:.8f}", gas_step=f"{gas_step}")

            torch.save(
                {n: p.grad for n, p in model.named_parameters()},
                states_dir / f"grad_{iter:03d}_{dist.get_rank()}.pt",
            )

            torch.save(
                {
                    n: (
                        torch.norm(p.grad).full_tensor()
                        if p.grad is not None
                        else None
                    )
                    for n, p in model.named_parameters()
                },
                states_dir / f"ind_grad_norm_{iter:03d}_{dist.get_rank()}.pt",
            )
            total_norm = torch.nn.utils.get_total_norm(
                p.grad for p in model.parameters() if p.grad is not None
            )
            total_norm = total_norm.full_tensor()
            torch.nn.utils.clip_grads_with_norm_(
                model.parameters(), 1.0, total_norm
            )

            torch.save(
                model.state_dict(),
                states_dir / f"model_{iter:03d}_{dist.get_rank()}.pt",
            )
            torch.save(
                optim.state_dict(),
                states_dir / f"optim_{iter:03d}_{dist.get_rank()}.pt",
            )

            optim.step()
            optim.zero_grad()
            lr_scheduler.step()
            profiler.step()

            loss_file.write(f"{total_loss:20.8f} {total_norm:20.8f}\n")

            if eval_interval is not None and iter % eval_interval == 0:
                with torch.no_grad():
                    total_loss = 0.0
                    for gas_step in range(num_gas_steps):
                        batch = torch.full(
                            (2, dim),
                            fill_value=1
                            + 1e-1 * mesh["dp"].get_local_rank()
                            + 1e-2 * gas_step,
                            dtype=param_dtype,
                            device="cuda",
                        )
                        output, activations, weights = model(batch)

                        loss = output.sum().sub(1).square()
                        loss = (
                            loss / num_gas_steps if framework != "ya-fsdp-1" else loss
                        )
                        dist.all_reduce(loss)
                        total_loss += loss.item()

    loss_file.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    fire.Fire(train)
