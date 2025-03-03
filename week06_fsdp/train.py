import functools
import os
import pickle
import time
from functools import partial
from typing import Optional

import fire
import torch
import torch.nn as nn
from torch.distributed import DeviceMesh, init_device_mesh
from torch.optim.lr_scheduler import LambdaLR

import torchtitan.utils as utils
from torchtitan.datasets import build_hf_data_loader, build_tokenizer
from torchtitan.logging import init_logger, logger
from torchtitan.metrics import build_device_memory_monitor
from torchtitan.models import model_name_to_cls, model_name_to_tokenizer, models_config
from torchtitan.optimizer import linear_warmup_linear_decay


def trace_handler(prof, trace_dir: str):
    curr_trace_dir_name = "iteration_" + str(prof.step_num)
    curr_trace_dir = os.path.join(trace_dir, curr_trace_dir_name)
    if not os.path.exists(curr_trace_dir):
        os.makedirs(curr_trace_dir, exist_ok=True)

    logger.info(f"Dumping profiler traces at step {prof.step_num}")
    begin = time.monotonic()
    prof.export_chrome_trace(
        f"{curr_trace_dir}/rank{torch.distributed.get_rank()}_trace.json"
    )
    logger.info(
        f"Finished dumping profiler traces in {time.monotonic() - begin:.2f} seconds"
    )


class MemoryProfiler:
    def __init__(
        self,
        step_num: int,
        freq: int,
        snapshot_dir: str,
        dir_name: Optional[str] = None,
    ):
        self.snapshot_dir = snapshot_dir
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir, exist_ok=True)

        # when resume training, we start from the last step
        self.step_num = step_num
        self.freq = freq

        self.dir_name = dir_name

    def step(self):
        self.step_num += 1
        if self.step_num % self.freq not in [0, self.freq - 1]:
            return
        if self.step_num % self.freq == self.freq - 1:
            torch.cuda.memory._record_memory_history()
            return
        curr_step = self.step_num
        if self.dir_name is None:
            dir_name = f"iteration_{curr_step}"
        else:
            dir_name = self.dir_name
        curr_snapshot_dir = os.path.join(self.snapshot_dir, dir_name)
        if not os.path.exists(curr_snapshot_dir):
            os.makedirs(curr_snapshot_dir, exist_ok=True)
        logger.info(f"Dumping memory snapshot at step {curr_step}")
        begin = time.monotonic()
        with open(
            f"{curr_snapshot_dir}/rank{torch.distributed.get_rank()}_memory_snapshot.pickle",
            "wb",
        ) as output:
            pickle.dump(torch.cuda.memory._snapshot(), output)
        torch.cuda.memory._record_memory_history(None)
        logger.info(
            f"Finished dumping memory snapshot in {time.monotonic() - begin:.2f} seconds"
        )


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    cpu_offload: bool,
    reshard_after_forward: bool,
    hw_fsdp: bool = False,
):
    if hw_fsdp:
        from hw_fsdp import (
            FSDPCommContext,
            MixedPrecisionPolicy,
        )
        from hw_fsdp import (
            fully_shard as hw_fully_shard,
        )
    else:
        from torch.distributed._composable.fsdp import (
            CPUOffloadPolicy,
            MixedPrecisionPolicy,
            fully_shard,
        )

    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    if hw_fsdp:
        comm_ctx = FSDPCommContext(dp_mesh.device_type)
        fully_shard_module = partial(
            hw_fully_shard,
            reshard_after_forward=reshard_after_forward,
            comm_ctx=comm_ctx,
            **fsdp_config,
        )
        fully_shard_module(model.tok_embeddings, module_fqn="tok_embeddings")
        for layer_id, transformer_block in model.layers.items():
            fully_shard_module(transformer_block, module_fqn=f"layers.{layer_id}")
        fully_shard_module(model.norm, module_fqn="norm")
        fully_shard_module(model.output, module_fqn="output")
        model.to(
            torch.device(
                dp_mesh.device_type,
                getattr(torch, dp_mesh.device_type).current_device(),
            )
        )
    else:
        fully_shard_module = partial(
            fully_shard,
            reshard_after_forward=reshard_after_forward,
            **fsdp_config,
        )
        for layer_id, transformer_block in model.layers.items():
            fully_shard_module(transformer_block)
        fully_shard_module(model)


def train(
    lr: float = 8e-4,
    max_norm: float = 1.0,
    training_steps: int = 10,
    warmup_steps: int = 2,
    batch_size: int = 8,
    seq_len: int = 2048,
    model_name: str = "llama3",
    flavor: str = "debugmodel",
    norm_type: str = "rmsnorm",
    enable_cpu_offload: bool = False,
    param_dtype: str = "float32",
    reduce_dtype: str = "float32",
    reshard_after_forward: bool = True,
    reshard_after_forward_degree: int | None = None,
    device_type: str = "cuda",
    log_freq: int = 1,
    gc_freq: int = 50,
    profile_freq: int = 10,
    profile_active: int = 1,
    profile_warmup: int = 3,
    dump_folder: str = ".",
    save_traces_folder: str = "profile_trace",
    save_memory_snapshot_folder: str = "memory_snapshot",
    apply_compile: bool = False,
    num_gas_steps: int = 1,
    reshard_after_backward: bool = True,
    reduce_grads: bool = True,
    seed: int = 42,
    deterministic: bool = True,
    hw_fsdp: bool = False,
):
    decay_steps = training_steps - warmup_steps
    param_dtype = getattr(torch, param_dtype)
    reduce_dtype = getattr(torch, reduce_dtype)
    if reshard_after_forward_degree is not None:
        assert reshard_after_forward
        reshard_after_forward = reshard_after_forward_degree

    init_logger()

    # take control of garbage collection to avoid stragglers
    gc_handler = utils.GarbageCollection(gc_freq=gc_freq)

    # init distributed
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
    torch.cuda.set_device(device)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("cuda:nccl,cpu:gloo")
    # initialize device memory monitor and get peak flops for MFU calculation
    device_memory_monitor = build_device_memory_monitor()
    gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
    logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")

    # build meshes
    world_mesh = init_device_mesh(device_type, (world_size,), mesh_dim_names=("dp",))
    dp_mesh = world_mesh["dp"]
    dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()

    # Set random seed, and maybe enable deterministic mode (mainly for debugging, expect perf loss)
    utils.set_determinism(world_mesh, device, seed, deterministic)

    # build tokenizer
    tokenizer_type = model_name_to_tokenizer[model_name]
    tokenizer = build_tokenizer(
        tokenizer_type, "torchtitan/tests/assets/test_tiktoken.model"
    )
    # build dataloader
    data_loader = build_hf_data_loader(
        "c4_test",
        "torchtitan/tests/assets/c4_test",
        tokenizer,
        batch_size=batch_size,
        seq_len=seq_len,
        world_size=dp_degree,
        rank=dp_rank,
    )

    # build model (using meta init)
    model_cls = model_name_to_cls[model_name]
    model_config = models_config[model_name][flavor]
    model_config.norm_type = norm_type
    model_config.vocab_size = tokenizer.n_words
    model_config.max_seq_len = seq_len

    logger.info(f"Building {model_name} {flavor} with {model_config}")
    memory_profiler = MemoryProfiler(
        profile_freq - 2,
        profile_freq,
        snapshot_dir=os.path.join(dump_folder, save_memory_snapshot_folder),
        dir_name="model_init",
    )
    memory_profiler.step()
    with torch.device("cpu"):
        model = model_cls.from_model_args(model_config)

    # log model size
    model_param_count = utils.get_num_params(model)
    num_flop_per_token = utils.get_num_flop_per_token(
        utils.get_num_params(model, exclude_embedding=True),
        model_config,
        seq_len,
    )
    logger.info(
        f"Model {model_name} {flavor} " f"size: {model_param_count:,} total parameters"
    )

    # loss function
    def loss_fn(pred, labels):
        return torch.nn.functional.cross_entropy(
            pred.flatten(0, 1).float(), labels.flatten(0, 1)
        )

    # move sharded model to CPU/GPU and initialize weights via DTensor
    if enable_cpu_offload:
        init_device = "cpu"
        buffer_device = device_type
    else:
        init_device = device_type
        buffer_device = None

    # apply parallelisms and initialization
    if apply_compile:
        for layer_id, transformer_block in model.layers.named_children():
            transformer_block = torch.compile(transformer_block, fullgraph=True)
            model.layers.register_module(layer_id, transformer_block)
        logger.info("Compiling each TransformerBlock with torch.compile")
    apply_fsdp(
        model,
        dp_mesh=dp_mesh,
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cpu_offload=enable_cpu_offload,
        reshard_after_forward=reshard_after_forward,
        hw_fsdp=hw_fsdp,
    )
    model.train()

    memory_profiler.step()

    device_mem_stats = device_memory_monitor.get_peak_stats()
    logger.info(
        f"{device_type.upper()} memory usage for model: "
        f"{device_mem_stats.max_reserved_gib:.2f}GiB"
        f"({device_mem_stats.max_reserved_pct:.2f}%)"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=True,
    )
    lr_scheduler = LambdaLR(
        optimizer,
        lr_lambda=functools.partial(
            linear_warmup_linear_decay, warmup_steps, decay_steps
        ),
    )

    data_iterator = iter(data_loader)

    train_context = utils.get_train_context(
        enable_loss_parallel=False,
        enable_compiled_autograd=False,
    )

    # variables used to keep info for metrics logging
    step = 0
    ntokens_since_last_log = 0
    data_loading_times = []
    time_last_log = time.perf_counter()
    device_memory_monitor.reset_peak_stats()

    # train loop
    logger.info(
        f"Training starts at step {step + 1}, "
        f"with local batch size {batch_size}, "
        f"global batch size {batch_size * dp_degree}, "
        f"sequence length {seq_len}, "
        f"total steps {training_steps} "
        f"(warmup {warmup_steps})"
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
            trace_handler, trace_dir=os.path.join(dump_folder, save_traces_folder)
        ),
        record_shapes=True,
    ) as torch_profiler:
        while step < training_steps:
            memory_profiler = MemoryProfiler(
                step,
                profile_freq,
                snapshot_dir=os.path.join(dump_folder, save_memory_snapshot_folder),
            )

            step += 1
            gc_handler.run(step)

            optimizer.zero_grad()

            for gas_step in range(num_gas_steps):
                is_last_backward = gas_step == num_gas_steps - 1
                if not hw_fsdp:
                    model.set_is_last_backward(is_last_backward)
                    model.set_reshard_after_backward(
                        reshard_after_backward or is_last_backward
                    )
                    model.set_requires_gradient_sync(reduce_grads or is_last_backward)

                # get batch
                data_load_start = time.perf_counter()
                batch = next(data_iterator)
                input_ids, labels = batch
                ntokens_since_last_log += labels.numel()
                data_loading_times.append(time.perf_counter() - data_load_start)

                input_ids = input_ids.to(device_type)
                labels = labels.to(device_type)

                # Non-PP forward / backward
                with train_context():
                    pred = model(input_ids)
                    loss = loss_fn(pred, labels)
                    # pred.shape=(bs, seq_len, vocab_size)
                    # need to free to before bwd to avoid peaking memory
                    del pred
                    loss.backward()

            # clip gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters()], max_norm
            )

            # optimizer step
            optimizer.step()
            lr_scheduler.step()

            # log metrics
            if step == 1 or step % log_freq == 0:
                loss = loss.detach()
                global_avg_loss = utils.dist_mean(loss, dp_mesh)
                global_grad_norm = grad_norm.full_tensor().item()

                time_delta = time.perf_counter() - time_last_log

                # tokens per second per device, abbreviated as tps
                tps = ntokens_since_last_log / time_delta
                # model FLOPS utilization
                # For its definition and calculation, please refer to the PaLM paper:
                # https://arxiv.org/abs/2204.02311
                mfu = 100 * num_flop_per_token * tps / gpu_peak_flops

                device_mem_stats = device_memory_monitor.get_peak_stats()

                logger.info(
                    f"step: {step:2}  "
                    f"loss: {global_avg_loss:7.4f}  "
                    f"grad norm: {global_grad_norm:7.4f}  "
                    f"memory: {device_mem_stats.max_reserved_gib:5.2f}GiB"
                    f"({device_mem_stats.max_reserved_pct:.2f}%)  "
                    f"tps: {round(tps):,}  "
                    f"mfu: {mfu:.2f}%"
                )

                ntokens_since_last_log = 0
                data_loading_times.clear()
                time_last_log = time.perf_counter()
                device_memory_monitor.reset_peak_stats()

            # signal the profiler that the next profiling step has started
            if torch_profiler:
                torch_profiler.step()
            if memory_profiler:
                memory_profiler.step()

    logger.info("Training completed")

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    fire.Fire(train)
