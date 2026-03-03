"""
Gradient-Aligned Energy Training (GAET) Loss.

Replaces SEAL's NCE ranking objective for the score_nn (energy function)
with a gradient alignment objective:

    L_E = -cos(∇_ỹ E_Θ(x, ỹ), ∇_ỹ S(ỹ, y*))

where ỹ = F_Φ(x) is the task-net's current prediction, E_Θ is the energy
function, and S is a differentiable structural metric (soft F1).

The key insight: NCE trains the energy to *rank* configurations, but SEAL
uses the energy's *gradient* to train the task-net. An energy that ranks
perfectly can still provide poor gradients. GAET directly optimizes the
energy to produce gradients aligned with the structural metric's gradient,
making it a better "teacher" rather than a better "judge."

This requires second-order derivatives (differentiating through ∇_ỹ E_Θ
w.r.t. Θ), which PyTorch handles via create_graph=True.
"""

import logging
from typing import Dict, Any, Optional

import torch
import torch.nn.functional as F
from seal.modules.loss import Loss
from seal.modules.score_nn import ScoreNN
from seal.modules.oracle_value_function import OracleValueFunction
from .soft_metrics import soft_instance_f1

logger = logging.getLogger(__name__)


@Loss.register("gradient-aligned")
class GradientAlignedLoss(Loss):
    """Gradient-Aligned Energy Training loss for score_nn.

    Trains the energy function so its gradient w.r.t. label predictions
    aligns with the gradient of soft F1. This directly optimizes gradient
    quality rather than ranking quality.

    Integrates as a drop-in replacement for NCE in SEAL configs:
        "loss_fn": {"type": "gradient-aligned", "log_key": "gaet"}
    """

    def __init__(
        self,
        score_nn: Optional[ScoreNN] = None,
        oracle_value_function: Optional[OracleValueFunction] = None,
        reduction: str = "mean",
        normalize_y: bool = False,
        eps: float = 1e-7,
        **kwargs: Any,
    ):
        """
        Args:
            score_nn: The energy function (injected by SEAL framework).
            oracle_value_function: Not used, accepted for interface compat.
            reduction: How to reduce batch losses ("mean", "sum", "none").
            normalize_y: If True, apply sigmoid to y_hat before use.
            eps: Epsilon for cosine similarity numerical stability.
        """
        super().__init__(
            score_nn=score_nn,
            oracle_value_function=oracle_value_function,
            reduction=reduction,
            normalize_y=normalize_y,
            **kwargs,
        )
        self.eps = eps

        if self.score_nn is None:
            raise ValueError("score_nn cannot be None for GradientAlignedLoss")

        logger.info("GradientAlignedLoss initialized (GAET)")

    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(y)

    def _forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],  # (batch, 1, L)
        y_hat: torch.Tensor,             # (batch, 1, L) — task-net prediction
        y_hat_extra: Optional[torch.Tensor],
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute gradient alignment loss.

        L = -cos(∇_ỹ E_Θ(x, ỹ), ∇_ỹ S(ỹ, y*))

        Args:
            x: Input features.
            labels: Ground truth, shape (batch, 1, L).
            y_hat: Task-net prediction (probabilities), shape (batch, 1, L).
            y_hat_extra: Unused.
            buffer: Shared computation buffer.

        Returns:
            Loss tensor, shape depends on reduction.
        """
        assert labels is not None
        assert y_hat.shape[1] == 1

        # Ground truth: (batch, L)
        y_true = labels.squeeze(1).float()

        # Task-net prediction: detach from task-net graph, enable grad for
        # computing ∇_ỹ E. Shape: (batch, L)
        y_pred = y_hat.squeeze(1).detach().clone().requires_grad_(True)

        # === Energy gradient: ∇_ỹ E_Θ(x, ỹ) ===
        # Call score_nn to compute energy. create_graph=True in the subsequent
        # autograd.grad allows backprop through this gradient to update Θ.
        y_for_score = y_pred.unsqueeze(1)  # (batch, 1, L)
        energy = self.score_nn(x, y_for_score, buffer)  # (batch, 1)
        energy = energy.squeeze(1)  # (batch,)

        energy_grad = torch.autograd.grad(
            energy.sum(),
            y_pred,
            create_graph=True,  # needed to differentiate w.r.t. Θ
        )[0]  # (batch, L)

        # === Structural metric gradient: ∇_ỹ S(ỹ, y*) ===
        # Separate computation graph — we only need the gradient direction,
        # not backprop through the metric itself.
        y_pred_for_metric = y_pred.detach().clone().requires_grad_(True)
        f1 = soft_instance_f1(y_pred_for_metric, y_true)  # (batch,)
        metric_grad = torch.autograd.grad(
            f1.sum(),
            y_pred_for_metric,
        )[0]  # (batch, L)
        metric_grad = metric_grad.detach()  # fixed target, no Θ dependence

        # === Gradient alignment loss: -cos(∇E, ∇S) ===
        # Handle degenerate cases where metric gradient is near-zero
        # (e.g., perfect prediction or all-zero labels)
        metric_norm = metric_grad.norm(dim=-1, keepdim=True)
        energy_norm = energy_grad.norm(dim=-1, keepdim=True)
        valid = (metric_norm.squeeze(-1) > self.eps) & (energy_norm.squeeze(-1) > self.eps)

        cos_sim = F.cosine_similarity(
            energy_grad, metric_grad, dim=-1, eps=self.eps
        )  # (batch,)

        # For invalid (degenerate) cases, use zero loss
        loss = torch.where(valid, -cos_sim, torch.zeros_like(cos_sim))

        return loss  # (batch,)
