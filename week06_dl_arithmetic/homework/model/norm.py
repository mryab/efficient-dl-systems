"""
Zero-Centered RMSNorm
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Zero-Centered RMSNorm: y = x/rms(x) * (1 + weight), weight init to zeros.
    """

    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        x_squared = x_fp32 * x_fp32
        mean_squared = x_squared.mean(dim=-1, keepdim=True)
        mean_squared_eps = mean_squared + self.eps
        rsqrt = torch.rsqrt(mean_squared_eps)
        normalized = x_fp32 * rsqrt
        output = normalized * (1.0 + self.weight.float())
        return output.type_as(x)
