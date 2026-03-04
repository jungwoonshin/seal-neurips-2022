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

                    required_gap = torch.clamp(self.alpha * task_error, min=self.min_margin)
                    mask = (delta < required_gap) & (task_error > 0)

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
                        batch_from_critical_set: list[dict],
                        task_net: nn.Module = None) -> torch.Tensor:
        """
        L_correct = (1/B) * sum [max(alpha * l, min_margin) + E(x,y) - E(x,F(x))]_+

        If task_net is provided, y_pred and task_error are recomputed fresh
        from the current task net (no staleness). Otherwise falls back to
        the stored snapshots.

        y_pred is detached: we update Theta (energy params) only.
        """
        if len(batch_from_critical_set) == 0:
            return torch.tensor(0.0, requires_grad=True)

        device = next(energy_net.parameters()).device

        x = torch.stack([s["x"] for s in batch_from_critical_set]).to(device)
        y = torch.stack([s["y"] for s in batch_from_critical_set]).to(device)

        if task_net is not None:
            # Fresh y_pred and task_error from current task net
            with torch.no_grad():
                y_pred = task_net(x).detach()
                task_errors = self._compute_error(y_pred, y).detach()
        else:
            y_pred = torch.stack([s["y_pred"] for s in batch_from_critical_set]).detach().to(device)
            task_errors = torch.stack([s["task_error"] for s in batch_from_critical_set]).detach().to(device)

        # Fresh energy computation from current network state
        if self.correct_global_only:
            e_true = energy_net.energy_global(y.float())
            e_pred = energy_net.energy_global(y_pred)
        else:
            e_true = energy_net(x, y.float())
            e_pred = energy_net(x, y_pred)

        # Hinge: [max(alpha * l, min_margin) + E(x,y) - E(x,F(x))]_+
        margin = torch.clamp(self.alpha * task_errors, min=self.min_margin)
        violation = F.relu(margin + e_true - e_pred)

        return violation.mean()

    # ──────────────────────────────────────────────
    # Inline corrective loss (no diagnosis needed)
    # ──────────────────────────────────────────────

    def batch_corrective_loss(self, energy_net: nn.Module, task_net: nn.Module,
                              x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, int]:
        """
        Compute corrective loss directly on a training batch.

        No stored critical set, no diagnosis sweep, no correction_interval.
        Just check which examples in this batch are critical and apply the hinge.

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
