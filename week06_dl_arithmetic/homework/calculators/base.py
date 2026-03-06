"""
Base classes and configurations for theoretical calculators.
"""

from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class GPUSpec:
    """GPU hardware specifications."""
    name: str
    memory_bandwidth_gbps: float  # GB/s
    flops_bf16: float             # TFLOP/s for BF16
    interconnect_bandwidth_gbps: float  # GB/s per direction


# H100 GPU spec for example
H100_SXM = GPUSpec(
    name="H100 SXM", 
    memory_bandwidth_gbps=2800,
    flops_bf16=800,
    interconnect_bandwidth_gbps=400,
)

# Put your GPU spec here
GPUS_SPEC = GPUSpec(
    name="...", 
    memory_bandwidth_gbps=...,
    flops_bf16=...,
    interconnect_bandwidth_gbps=...,
)


@dataclass
class ModelConfig:
    """Model configuration for calculations."""
    vocab_size: int
    hidden_dim: int
    num_heads: int
    num_layers: int
    intermediate_dim: int
    max_seq_len: int


@dataclass 
class TrainingConfig:
    """Training configuration."""
    batch_size: int
    seq_len: int
    num_gpus: int
    dtype_bytes: int = 2  # BF16 = 2 bytes


class BaseCalculator(ABC):
    """
    Base class for theoretical calculators.
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        gpu_spec: GPUSpec,
    ):
        self.model = model_config
        self.training = training_config
        self.gpu = gpu_spec

    def roofline_time_ms(self, flops: int, memory_bytes: int) -> float:
        """
        Calculate time using roofline model.
        
        Args:
            flops: Number of floating point operations
            memory_bytes: Memory traffic in bytes
            
        Returns:
            Time in milliseconds (max of compute-bound and memory-bound time)
        """
        compute_time_s = flops / (self.gpu.flops_bf16 * 1e12)  # TFLOP/s -> FLOP/s
        memory_time_s = memory_bytes / (self.gpu.memory_bandwidth_gbps * 1e9)  # GB/s -> B/s
        return max(compute_time_s, memory_time_s) * 1000  # convert to ms

    @abstractmethod
    def calculate_total_params(self) -> int:
        """Calculate total number of model parameters."""
        pass

    @abstractmethod
    def calculate_param_memory(self) -> int:
        """Calculate parameter memory per GPU (bytes)."""
        pass

    @abstractmethod
    def calculate_gradient_memory(self) -> int:
        """Calculate gradient memory per GPU (bytes)."""
        pass

    @abstractmethod
    def calculate_optimizer_memory(self) -> int:
        """Calculate optimizer state memory per GPU (bytes)."""
        pass

    @abstractmethod
    def calculate_activation_memory(self) -> int:
        """Calculate activation memory per GPU (bytes)."""
        pass

    @abstractmethod
    def calculate_peak_memory(self) -> int:
        """Calculate total peak memory per GPU (bytes)."""
        pass

    @abstractmethod
    def time_embedding_ms(self) -> float:
        """Time for embedding lookup (ms)."""
        pass

    @abstractmethod
    def time_rms_norm_ms(self) -> float:
        """Time for single RMSNorm layer (ms)."""
        pass

    @abstractmethod
    def time_attention_ms(self) -> float:
        """Time for single attention layer (ms)."""
        pass

    @abstractmethod
    def time_mlp_ms(self) -> float:
        """Time for single MLP/FFN layer (ms)."""
        pass

    @abstractmethod
    def time_lm_head_ms(self) -> float:
        """Time for language model head (ms)."""
        pass

    @abstractmethod
    def time_loss_ms(self) -> float:
        """Time for loss computation (ms)."""
        pass

    def time_forward_pass_ms(self) -> float:
        """
        Total forward pass time (ms).
        
        Sums up: embedding + (norm + attn + norm + mlp) * num_layers + final_norm + lm_head + loss
        
        Note: residual additions (x = x + layer_out) are omitted - you can account for them if you want.
        """
        total = self.time_embedding_ms()

        for _ in range(self.model.num_layers):
            total += self.time_rms_norm_ms()
            total += self.time_attention_ms()
            total += self.time_rms_norm_ms()
            total += self.time_mlp_ms()

        total += self.time_rms_norm_ms()
        total += self.time_lm_head_ms()
        total += self.time_loss_ms()
        
        return total

    def time_backward_pass_ms(self) -> float:
        """
        Total backward pass time (ms).

        Override if backward has different characteristics.
        """
        return 2.0 * self.time_forward_pass_ms()

    def time_forward_backward_ms(self) -> float:
        """Total forward + backward time (ms)."""
        return self.time_forward_pass_ms() + self.time_backward_pass_ms()

    @abstractmethod
    def calculate_communication_volume(self) -> int:
        """Calculate total communication volume per step (bytes)."""
        pass

    @abstractmethod
    def time_communication_ms(self) -> float:
        """Calculate communication time per step (ms)."""
        pass

    @abstractmethod
    def overlap_efficiency(self) -> float:
        """
        Estimate overlap efficiency (0.0 to 1.0).
        
        Fraction of communication that overlaps with compute.
        """
        pass

    @abstractmethod
    def time_total_step_ms(self) -> float:
        """
        Total step time accounting for compute/comm overlap (ms).
        
        Consider how to combine compute time and communication time
        with overlap efficiency.
        """
        pass
