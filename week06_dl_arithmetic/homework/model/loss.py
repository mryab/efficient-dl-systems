"""
Cross Entropy Loss for Causal LM
"""

import torch
import torch.nn.functional as F


def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Cross entropy loss for causal language modeling.
    
    Shifts logits and labels for next-token prediction:
    - logits[:, :-1] predicts labels[:, 1:]
    
    Args:
        logits: (B, S, vocab_size)
        labels: (B, S)
        ignore_index: label to ignore
    """
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()
    
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )
