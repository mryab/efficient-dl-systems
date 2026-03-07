import torch

from edlang.server.api_server import Server
from edlang.entrypoints.config import EngineConfig, ModelConfig
from edlang.managers.scheduler_manager import SchedulerConfig

import argparse

def torch_dtype_from_str(s: str) -> torch.dtype:
    s = s.lower()
    if s in {'float16', 'half', 'torch.float16'}:
        return torch.float16
    elif s in {'float32', 'float', 'torch.float32'}:
        return torch.float32
    elif s in {'bfloat16', 'torch.bfloat16'}:
        return torch.bfloat16
    else:
        raise ValueError(f"Unknown torch_dtype: {s}")

def launch_server(
    model_name: str, device: str, torch_dtype: torch.dtype, max_prompt_length: int, enable_metrics: bool,
    max_batch_size: int, max_waiting_requests: int, prefill_timeout_ms: float
):
    model_config = ModelConfig(
        model_name=model_name,
        device=device,
        torch_dtype=torch_dtype,
        max_prompt_length=max_prompt_length,
    )
    
    engine_config = EngineConfig(model_config=model_config)
    scheduler_config = SchedulerConfig(
        enable_metrics=enable_metrics,
        max_batch_size=max_batch_size,
        max_waiting_requests=max_waiting_requests,
        prefill_timeout_ms=prefill_timeout_ms
    )

    server = Server(engine_config, scheduler_config)
    server.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch edlang API server")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-8B", help="Model name or path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run model on (e.g., cuda:0 or cpu)")
    parser.add_argument("--torch-dtype", type=str, default="float16", help="Torch dtype: float16, bfloat16, float32")
    parser.add_argument("--max-prompt-length", type=int, default=512, help="Maximum prompt length")
    parser.add_argument("--enable-metrics", action="store_true", default=False, help="Enable metrics")
    parser.add_argument("--max-batch-size", type=int, default=20, help="Maximum batch size")
    parser.add_argument("--max-waiting-requests", type=int, default=100, help="Maximum waiting requests")
    parser.add_argument("--prefill-timeout-ms", type=float, default=50.0, help="Prefill timeout in milliseconds")

    args = parser.parse_args()

    torch_dtype = torch_dtype_from_str(args.torch_dtype)

    launch_server(
        model_name=args.model_name,
        device=args.device,
        torch_dtype=torch_dtype,
        max_prompt_length=args.max_prompt_length,
        enable_metrics=args.enable_metrics,
        max_batch_size=args.max_batch_size,
        max_waiting_requests=args.max_waiting_requests,
        prefill_timeout_ms=args.prefill_timeout_ms
    )    