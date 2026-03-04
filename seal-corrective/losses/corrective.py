"""
Core corrective energy alignment algorithm.

L_correct(Theta) = (1/|C|) * sum_{(x,y) in C} w_i * [alpha * l(F(x), y) + E(x,y) - E(x, F(x))]_+

Where C is the critical set: examples where the energy surface is inverted
(assigns lower energy to incorrect predictions than to ground truth) AND
the task-net is actually making an error.

w_i is a rank-based criticality weight derived from |delta_i| (energy inversion depth).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyCorrector:
    """
    Diagnoses energy surface misalignment and produces a corrective loss
    that surgically fixes inverted regions, weighted by criticality.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        task_error_metric: str = "f1",
        correct_global_only: bool = False,
    ):
        """
        Args:
            alpha: Margin scaling factor in the hinge term.
            task_error_metric: One of 'hamming', 'f1', 'structural'.
            correct_global_only: If True, only correct E_global (label-dependency
                matrix M and scoring vector v), leaving per-label scoring alone.
        """
        self.alpha = alpha
        self.metric = task_error_metric
        self.correct_global_only = correct_global_only
        self.critical_set: list[dict] = []

    # ──────────────────────────────────────────────
    # Diagnose: find the critical misalignment set C
    # ──────────────────────────────────────────────

    def diagnose(self, energy_net: nn.Module, task_net: nn.Module,
                 loaders, device: torch.device) -> int:
        """
        Sweep data loaders to find current critical examples.

        C = {(x, y) | delta < alpha * task_error AND task_error > 0}

        Returns:
            Size of the critical set.
        """
        if not isinstance(loaders, (list, tuple)):
            loaders = [loaders]

        self.critical_set = []
        energy_net.eval()
        task_net.eval()

        with torch.no_grad():
            for loader in loaders:
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    y_pred = task_net(x)

                    e_pred = energy_net(x, y_pred)
                    e_true = energy_net(x, y.float())

                    delta = e_pred - e_true
                    task_error = self._compute_error(y_pred, y)

                    mask = (delta < self.alpha * task_error) & (task_error > 0)

                    for i in range(x.size(0)):
                        if mask[i]:
                            self.critical_set.append({
                                "x": x[i].cpu(),
                                "y": y[i].cpu(),
                                "y_pred": y_pred[i].cpu(),
                                "task_error": task_error[i].cpu(),
                                "delta": delta[i].cpu(),
                            })

        # Compute rank-based criticality from |delta|
        if len(self.critical_set) > 0:
            deltas = torch.tensor([s["delta"].abs().item() for s in self.critical_set])
            ranks = deltas.argsort().argsort().float()
            percentiles = (ranks + 1.0) / len(self.critical_set)
            for i, s in enumerate(self.critical_set):
                s["criticality"] = percentiles[i]

        energy_net.train()
        task_net.train()
        return len(self.critical_set)

    # ──────────────────────────────────────────────
    # Corrective loss
    # ──────────────────────────────────────────────

    def corrective_loss(self, energy_net: nn.Module,
                        batch_from_critical_set: list[dict]) -> torch.Tensor:
        """
        L_correct = (1/B) * sum w_i * [alpha * l(F(x),y) + E(x,y) - E(x,F(x))]_+

        w_i is the rank-based criticality weight (replaces the original task_error
        multiplier to avoid double-counting — task_error already appears inside
        the hinge margin).

        y_pred is detached: we update Theta (energy params) only.
        Energies are computed fresh from the current energy_net.
        """
        if len(batch_from_critical_set) == 0:
            return torch.tensor(0.0, requires_grad=True)

        device = next(energy_net.parameters()).device

        x = torch.stack([s["x"] for s in batch_from_critical_set]).to(device)
        y = torch.stack([s["y"] for s in batch_from_critical_set]).to(device)
        y_pred = torch.stack([s["y_pred"] for s in batch_from_critical_set]).detach().to(device)
        task_errors = torch.stack([s["task_error"] for s in batch_from_critical_set]).detach().to(device)

        # Fresh energy computation from current network state
        if self.correct_global_only:
            e_true = energy_net.energy_global(y.float())
            e_pred = energy_net.energy_global(y_pred)
        else:
            e_true = energy_net(x, y.float())
            e_pred = energy_net(x, y_pred)

        # Hinge: [alpha * l + E(x,y) - E(x,F(x))]_+
        margin = self.alpha * task_errors
        violation = F.relu(margin + e_true - e_pred)

        return violation.mean()

    # ──────────────────────────────────────────────
    # Sampling from the critical set
    # ──────────────────────────────────────────────

    def sample_batch(self, batch_size: int) -> list[dict]:
        """
        Sample from C weighted by criticality rank.

        More critical examples (deeper energy inversions) are sampled
        more often. The loss itself uses no criticality weighting to
        avoid double-counting.
        """
        if len(self.critical_set) == 0:
            return []
        crits = torch.tensor([s["criticality"].item() for s in self.critical_set])
        probs = crits / (crits.sum() + 1e-8)
        n = min(batch_size, len(self.critical_set))
        indices = torch.multinomial(probs, n, replacement=False)
        return [self.critical_set[i] for i in indices]

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
