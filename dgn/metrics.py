"""Structural evaluation metrics for DGN.

Two variants:
  - instance_f1: Hard-thresholded F1 for evaluation (non-differentiable).
  - soft_instance_f1: Soft F1 for ES oracle (smooth, gives gradient signal everywhere).
"""

import torch


def instance_f1(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Per-instance F1 score with hard thresholding (for evaluation).

    Args:
        y_pred: (batch, L) continuous predictions in [0, 1].
        y_true: (batch, L) binary ground truth.

    Returns:
        (batch,) F1 score per instance.
    """
    y_hard = (y_pred > 0.5).float()
    tp = (y_hard * y_true).sum(dim=-1)
    pred_pos = y_hard.sum(dim=-1)
    true_pos = y_true.sum(dim=-1)

    precision = tp / (pred_pos + 1e-8)
    recall = tp / (true_pos + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # Handle edge case: both pred and true are all-zero
    both_zero = ((pred_pos == 0) & (true_pos == 0)).float()
    f1 = f1 * (1 - both_zero) + both_zero

    return f1


def soft_instance_f1(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Per-instance soft F1 score (smooth, for ES oracle).

    Uses y_pred directly as soft predictions instead of hard thresholding.
    This gives gradient signal everywhere, not just at the 0.5 boundary.

    Args:
        y_pred: (batch, L) continuous predictions in [0, 1].
        y_true: (batch, L) binary ground truth.

    Returns:
        (batch,) soft F1 score per instance.
    """
    tp = (y_pred * y_true).sum(dim=-1)
    pred_pos = y_pred.sum(dim=-1)
    true_pos = y_true.sum(dim=-1)

    precision = tp / (pred_pos + 1e-8)
    recall = tp / (true_pos + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # Handle edge case: both pred and true are all-zero
    both_zero = ((pred_pos == 0) & (true_pos == 0)).float()
    f1 = f1 * (1 - both_zero) + both_zero

    return f1
