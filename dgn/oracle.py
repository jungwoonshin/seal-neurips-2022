"""Evolution Strategies oracle direction estimation.

Estimates the gradient of a non-differentiable metric S with respect to y_pred
using random Gaussian perturbations (zero-order optimization).

    d* = (1 / (sigma * K)) * sum_k S(y_pred + sigma * eps_k, y_true) * eps_k
"""

from typing import Callable

import torch


@torch.no_grad()
def compute_oracle_direction(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    metric_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    K: int = 8,
    sigma: float = 0.1,
) -> torch.Tensor:
    """Estimate the oracle direction via Evolution Strategies.

    Args:
        y_pred: (batch, L) current predictions in [0, 1].
        y_true: (batch, L) binary ground truth.
        metric_fn: S(y_pred, y_true) -> (batch,) non-differentiable reward.
        K: Number of random perturbation samples.
        sigma: Perturbation scale.

    Returns:
        (batch, L) estimated oracle direction d*.
    """
    batch, L = y_pred.shape
    device = y_pred.device

    d_star = torch.zeros_like(y_pred)  # (batch, L)

    for _ in range(K):
        eps = torch.randn(batch, L, device=device)  # (batch, L)
        y_perturbed = (y_pred + sigma * eps).clamp(0.0, 1.0)
        reward = metric_fn(y_perturbed, y_true)  # (batch,)
        d_star += reward.unsqueeze(-1) * eps  # (batch, L)

    d_star = d_star / (sigma * K)
    return d_star
