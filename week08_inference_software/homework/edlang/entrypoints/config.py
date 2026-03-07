import torch

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    device: str = "cuda:0"
    torch_dtype: torch.dtype = torch.float16
    max_prompt_length: int = 512

@dataclass(frozen=True)
class EngineConfig:
    model_config: ModelConfig
