"""
Cross Entropy Loss for Causal LM
"""

import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    """Fused Linear Cross Entropy for causal LM."""
    # TODO: Replace with fused linear cross entropy (LigerFusedLinearCrossEntropyLoss)
    # The fused version takes hidden_states + lm_head.weight instead of logits

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self,) -> torch.Tensor:
        # TODO: Implement forward pass
        raise NotImplementedError("TODO: Implement forward pass")
