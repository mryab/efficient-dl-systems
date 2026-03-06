from dataclasses import dataclass


@dataclass
class TransformerConfig:
    vocab_size: int = 32000
    hidden_dim: int = 2048
    num_heads: int = 32
    num_layers: int = 24
    intermediate_dim: int = 5120
    max_seq_len: int = 4096
    dropout: float = 0.0
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    
    def __post_init__(self):
        assert self.hidden_dim % self.num_heads == 0, \
            f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"
