"""
Core corrective energy alignment losses.

Hinge (original):
    L_correct(Θ) = (1/|C|) Σ_{(x,y)∈C} [max(α·ℓ, min_margin) + E(x,y) − E(x,F(x))]₊

Smooth (log-sigmoid with soft weighting):
    L_smooth(Θ) = (1/Σw) Σ_B w(x)·log(1 + exp(α·ℓ + E(x,y) − E(x,F(x))))
    where w(x) = ℓ(x)^γ
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyCorrector:
    """
    Computes corrective loss on each batch to fix inverted energy regions.

    Supports three loss types:
        - "hinge": Original corrective SEAL with hard critical set filtering.
        - "smooth": Log-sigmoid penalty with soft task-error weighting.
        - "theory": Theory-driven loss with curvature-aware margin,
          descent violation weighting, and descent auxiliary loss.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        task_error_metric: str = "f1",
        correct_global_only: bool = False,
        min_margin: float = 0.1,
        loss_type: str = "hinge",
        gamma: float = 1.0,
        eta: float = 0.0,
        kappa: float = 0.0,
        beta2: float = 0.0,
        mu: float = 0.01,
    ):
        """
        Args:
            alpha: Margin scaling factor.
            task_error_metric: One of 'hamming', 'f1', 'structural'.
            correct_global_only: If True, only correct E_global.
            min_margin: Floor on the hinge margin (hinge mode only).
            loss_type: "hinge", "smooth", or "theory".
            gamma: Task-error weighting exponent (smooth/theory).
                   γ=1 → linear weighting, γ>1 → focal-like hard-example focus.
            eta: Curvature correction scale (theory only).
            kappa: Descent violation weighting exponent (theory only).
            beta2: Descent auxiliary loss weight (theory only).
            mu: Descent margin (theory only).
        """
        self.alpha = alpha
        self.min_margin = min_margin
        self.metric = task_error_metric
        self.correct_global_only = correct_global_only
        self.loss_type = loss_type
        self.gamma = gamma
        self.eta = eta
        self.kappa = kappa
        self.beta2 = beta2
        self.mu = mu

    # ──────────────────────────────────────────────
    # Corrective loss (dispatch)
    # ──────────────────────────────────────────────

    def batch_corrective_loss(self, energy_net: nn.Module, task_net: nn.Module,
                              x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, int, dict]:
        if self.loss_type == "theory":
            return self._theory_loss(energy_net, task_net, x, y)
        if self.loss_type == "smooth":
            loss, n = self._smooth_loss(energy_net, task_net, x, y)
            return loss, n, {}
        loss, n = self._hinge_loss(energy_net, task_net, x, y)
        return loss, n, {}

    # ──────────────────────────────────────────────
    # Hinge loss (original)
    # ──────────────────────────────────────────────

    def _hinge_loss(self, energy_net: nn.Module, task_net: nn.Module,
                    x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Original corrective hinge with hard critical set."""
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

        critical_mask = (violation > 0) & (task_errors > 0)
        n_critical = critical_mask.sum().item()

        if n_critical == 0:
            return torch.tensor(0.0, device=x.device, requires_grad=True), 0

        loss = F.relu(violation[critical_mask]).mean()
        return loss, int(n_critical)

    # ──────────────────────────────────────────────
    # Smooth loss (log-sigmoid + soft weighting)
    # ──────────────────────────────────────────────

    def _smooth_loss(self, energy_net: nn.Module, task_net: nn.Module,
                     x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, int]:
        """
        L = (1/Σw) Σ w(x)·log(1 + exp(α·ℓ + E(x,y) − E(x,F(x))))

        where w(x) = ℓ(x)^γ.

        Returns (loss, n_active) where n_active counts examples with ℓ > 0.
        """
        with torch.no_grad():
            y_pred = task_net(x).detach()
            task_errors = self._compute_error(y_pred, y).detach()  # (B,)

        if self.correct_global_only:
            e_true = energy_net.energy_global(y.float())
            e_pred = energy_net.energy_global(y_pred)
        else:
            e_true = energy_net(x, y.float())
            e_pred = energy_net(x, y_pred)

        # Soft weights: w(x) = ℓ(x)^γ
        weights = task_errors.pow(self.gamma)  # (B,)
        sum_w = weights.sum()

        if sum_w < 1e-12:
            return torch.tensor(0.0, device=x.device, requires_grad=True), 0

        # Log-sigmoid penalty: log(1 + exp(v))  where v = α·ℓ + e_true − e_pred
        v = self.alpha * task_errors + e_true - e_pred
        penalty = F.softplus(v)  # numerically stable log(1 + exp(v))

        loss = (weights * penalty).sum() / sum_w

        n_active = (task_errors > 0).sum().item()
        return loss, int(n_active)

    # ──────────────────────────────────────────────
    # Theory loss (theory-driven)
    # ──────────────────────────────────────────────

    def _compute_lipschitz(self, energy_net: nn.Module) -> float:
        """L_lip = ||v||_∞ · ||M||_op² / 4  (Proposition 3)."""
        with torch.no_grad():
            v_inf = energy_net.v.abs().max().item()
            # Spectral norm of M via power iteration (1 step is sufficient for monitoring)
            M_weight = energy_net.M.weight  # (energy_hidden, num_labels)
            M_op = torch.linalg.matrix_norm(M_weight, ord=2).item()
        return v_inf * (M_op ** 2) / 4.0

    def _adaptive_margin(self, task_errors: torch.Tensor,
                         f_x: torch.Tensor, y: torch.Tensor,
                         L_lip: float = None) -> torch.Tensor:
        """α·ℓ + (curvature/2)·||F(x)−y||².

        When L_lip is provided (theory mode), uses the actual Lipschitz
        constant from the energy net.  Falls back to static η otherwise.
        """
        margin = self.alpha * task_errors
        curvature = self.eta * L_lip if L_lip is not None else self.eta
        if curvature > 0:
            residual_sq = (f_x - y.float()).pow(2).sum(dim=-1)
            margin = margin + (curvature / 2.0) * residual_sq
        return margin

    def _descent_loss(self, energy_net: nn.Module, x: torch.Tensor,
                      f_x: torch.Tensor, y: torch.Tensor,
                      task_errors: torch.Tensor) -> tuple[torch.Tensor, float]:
        """
        Descent auxiliary loss (Proposition 2).
        Penalty: w(x)·softplus(−⟨∇E|_{F(x)}, F(x)−y⟩ + μ)

        Returns (loss, descent_sat_frac).
        """
        # Need gradients of E w.r.t. y_input at F(x)
        f_x_grad = f_x.detach().requires_grad_(True)
        if self.correct_global_only:
            e_at_fx = energy_net.energy_global(f_x_grad)
        else:
            e_at_fx = energy_net(x, f_x_grad)
        # Sum over batch to get scalar, then take grad w.r.t. f_x_grad
        grad_E = torch.autograd.grad(
            e_at_fx.sum(), f_x_grad, create_graph=True,
        )[0]  # (B, num_labels)

        # Direction: F(x) - y
        direction = f_x.detach() - y.float()  # (B, num_labels)

        # Inner product ⟨∇E, F(x)−y⟩ per sample
        inner = (grad_E * direction).sum(dim=-1)  # (B,)

        # Descent violation: softplus(−inner + μ)
        violation = F.softplus(-inner + self.mu)  # (B,)

        # Weights: ℓ^γ
        weights = task_errors.pow(self.gamma)
        sum_w = weights.sum()

        if sum_w < 1e-12:
            device = x.device
            return torch.tensor(0.0, device=device, requires_grad=True), 1.0

        loss = (weights * violation).sum() / sum_w

        # Fraction of samples satisfying descent (inner > μ)
        with torch.no_grad():
            sat_mask = (inner > self.mu) & (task_errors > 0)
            n_active = (task_errors > 0).sum().item()
            descent_sat = sat_mask.sum().item() / max(n_active, 1)

        return loss, descent_sat

    def _theory_loss(self, energy_net: nn.Module, task_net: nn.Module,
                        x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, int, dict]:
        """
        Theory-driven loss (theory).

        Combines curvature-aware margin, descent violation weighting,
        and optional descent auxiliary loss.
        """
        with torch.no_grad():
            f_x = task_net(x).detach()
            task_errors = self._compute_error(f_x, y).detach()

        if self.correct_global_only:
            e_true = energy_net.energy_global(y.float())
            e_pred = energy_net.energy_global(f_x)
        else:
            e_true = energy_net(x, y.float())
            e_pred = energy_net(x, f_x)

        # Lipschitz constant from energy net geometry
        L_lip = self._compute_lipschitz(energy_net)

        # Adaptive margin: α·ℓ + (L_lip/2)·||F(x)−y||²
        margin = self._adaptive_margin(task_errors, f_x, y, L_lip=L_lip)

        # Weights: ℓ^γ
        weights = task_errors.pow(self.gamma)

        # Descent violation weighting (κ > 0)
        if self.kappa > 0:
            f_x_no_grad = f_x.detach().requires_grad_(True)
            if self.correct_global_only:
                e_at_fx_ng = energy_net.energy_global(f_x_no_grad)
            else:
                e_at_fx_ng = energy_net(x.detach(), f_x_no_grad)
            grad_E_ng = torch.autograd.grad(
                e_at_fx_ng.sum(), f_x_no_grad, create_graph=False,
            )[0]
            direction_ng = f_x.detach() - y.float()
            inner_ng = (grad_E_ng * direction_ng).sum(dim=-1)
            # ρ = softplus(−inner) as violation magnitude
            rho = F.softplus(-inner_ng).detach()
            weights = weights * rho.pow(self.kappa)

        sum_w = weights.sum()
        if sum_w < 1e-12:
            zero = torch.tensor(0.0, device=x.device, requires_grad=True)
            return zero, 0, {"descent_sat": 1.0, "L_lip": 0.0}

        # Smooth loss: weighted softplus(margin + e_true − e_pred)
        v = margin + e_true - e_pred
        penalty = F.softplus(v)
        loss_smooth = (weights * penalty).sum() / sum_w

        n_active = (task_errors > 0).sum().item()

        # Descent auxiliary loss
        descent_sat = 1.0
        if self.beta2 > 0:
            loss_descent, descent_sat = self._descent_loss(
                energy_net, x, f_x, y, task_errors)
            loss = loss_smooth + self.beta2 * loss_descent
        else:
            loss = loss_smooth

        return loss, int(n_active), {"descent_sat": descent_sat, "L_lip": L_lip}

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
