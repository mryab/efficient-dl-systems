"""
Baseline Transformer Model
"""

import torch
import torch.nn as nn

from config import TransformerConfig
from model.norm import RMSNorm
from model.swiglu import SwiGLUFeedForward
from model.attention import MultiHeadAttention
from model.loss import cross_entropy_loss


class TransformerBlock(nn.Module):
    """Single transformer block."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.attn = MultiHeadAttention(config)
        self.ln2 = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.ffn = SwiGLUFeedForward(config.hidden_dim, config.intermediate_dim)
    
    def forward(
        self, 
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attention_mask)
        x = x + self.ffn(self.ln2(x))
        return x


class BaselineTransformer(nn.Module):
    """
    Transformer language model.
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        self.ln_f = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (B, S) token indices
            attention_mask: optional attention mask
            
        Returns:
            logits: (B, S, vocab_size)
        """
        B, S = input_ids.shape
        x = self.embedding(input_ids)

        for layer in self.layers:
            x = layer(x, attention_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits.float()

    def compute_loss(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for language modeling.
        """
        return cross_entropy_loss(logits, labels)
