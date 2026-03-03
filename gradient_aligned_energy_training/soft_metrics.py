"""
Differentiable structural metrics for gradient-aligned energy training.

These metrics must be differentiable w.r.t. predictions so that their
gradients can serve as alignment targets for the energy function.
"""

import torch


def soft_instance_f1(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute differentiable per-instance F1 score.

    Uses soft counts: tp = Σ y_pred * y_true, etc. When y_pred ∈ {0,1}
    this recovers exact F1. When y_pred ∈ (0,1) it provides a smooth
    relaxation whose gradient indicates which label changes improve F1.

    Args:
        y_pred: Predicted labels in [0, 1], shape (batch, L).
        y_true: Ground truth binary labels, shape (batch, L).
        eps: Small constant for numerical stability.

    Returns:
        Per-instance F1 scores, shape (batch,).
    """
    tp = (y_pred * y_true).sum(dim=-1)
    fp = (y_pred * (1 - y_true)).sum(dim=-1)
    fn = ((1 - y_pred) * y_true).sum(dim=-1)
    f1 = 2 * tp / (2 * tp + fp + fn + eps)
    return f1
