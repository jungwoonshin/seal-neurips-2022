"""
Core corrective energy alignment algorithm.

L_correct(Theta) = (1/|C|) * sum_{(x,y) in C} [max(alpha * l(F(x), y), min_margin) + E(x,y) - E(x, F(x))]_+

Where C is the critical set within each batch: examples where the energy surface
is inverted AND the task-net is actually making an error.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyCorrector:
    """
    Computes corrective loss on each batch to fix inverted energy regions.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        task_error_metric: str = "f1",
        correct_global_only: bool = False,
        min_margin: float = 0.1,
    ):
        """
        Args:
            alpha: Margin scaling factor in the hinge term.
            task_error_metric: One of 'hamming', 'f1', 'structural'.
            correct_global_only: If True, only correct E_global (label-dependency
                matrix M and scoring vector v), leaving per-label scoring alone.
            min_margin: Floor on the hinge margin so the energy net always
                has to maintain a meaningful gap, even when task error is small.
        """
        self.alpha = alpha
        self.min_margin = min_margin
        self.metric = task_error_metric
        self.correct_global_only = correct_global_only

    # ──────────────────────────────────────────────
    # Corrective loss
    # ──────────────────────────────────────────────

    def batch_corrective_loss(self, energy_net: nn.Module, task_net: nn.Module,
                              x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, int]:
        """
        Compute corrective loss directly on a training batch.

        Checks which examples in this batch are critical and applies the hinge.

        Returns:
            (loss, n_critical) — loss tensor and count of critical examples in batch.
        """
        with torch.no_grad():
            y_pred = task_net(x).detach()
            task_errors = self._compute_error(y_pred, y).detach()

        if self.correct_global_only:
            e_true = energy_net.energy_global(y.float())
            e_pred = energy_net.energy_global(y_pred)
        else:
            e_true = energy_net(x, y.float())
            e_pred = energy_net(x, y_pred)

        margin = torch.clamp(self.alpha * task_errors, min=self.min_margin)
        violation = margin + e_true - e_pred

        # Only keep critical examples: violation > 0 AND task_error > 0
        critical_mask = (violation > 0) & (task_errors > 0)
        n_critical = critical_mask.sum().item()

        if n_critical == 0:
            return torch.tensor(0.0, device=x.device, requires_grad=True), 0

        loss = F.relu(violation[critical_mask]).mean()
        return loss, int(n_critical)

    # ──────────────────────────────────────────────
    # Error metrics
    # ──────────────────────────────────────────────

    def _compute_error(self, y_pred_soft: torch.Tensor,
                       y_true: torch.Tensor) -> torch.Tensor:
        """
        Per-example task error.

        Args:
            y_pred_soft: (batch, num_labels) soft predictions in (0, 1)
            y_true: (batch, num_labels) binary ground truth

        Returns:
            (batch,) tensor of per-example errors.
        """
        if self.metric == "hamming":
            return self._hamming_error(y_pred_soft, y_true)
        elif self.metric == "f1":
            return 1.0 - self._soft_f1(y_pred_soft, y_true)
        elif self.metric == "structural":
            return self._structural_error(y_pred_soft, y_true)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def _hamming_error(self, y_pred_soft: torch.Tensor,
                       y_true: torch.Tensor) -> torch.Tensor:
        y_hard = (y_pred_soft >= 0.5).float()
        return (y_hard != y_true.float()).float().mean(dim=-1)

    def _soft_f1(self, y_pred: torch.Tensor,
                 y_true: torch.Tensor) -> torch.Tensor:
        """Differentiable soft F1 score per example."""
        y_true_f = y_true.float()
        intersection = (y_pred * y_true_f).sum(dim=-1)
        denom = y_pred.sum(dim=-1) + y_true_f.sum(dim=-1)
        return 2.0 * intersection / (denom + 1e-8)

    def _structural_error(self, y_pred: torch.Tensor,
                          y_true: torch.Tensor) -> torch.Tensor:
        """
        Pairwise co-occurrence mismatch.
        Detects broken label dependencies.
        """
        y_hard = (y_pred >= 0.5).float()
        y_true_f = y_true.float()
        # (batch, L, 1) @ (batch, 1, L) -> (batch, L, L)
        co_pred = torch.bmm(y_hard.unsqueeze(2), y_hard.unsqueeze(1))
        co_true = torch.bmm(y_true_f.unsqueeze(2), y_true_f.unsqueeze(1))
        L = y_true.size(-1)
        return (co_pred - co_true).abs().sum(dim=(-1, -2)) / (L * L)
