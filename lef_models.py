"""
Label Energy Fields (LEF): Asymmetric Non-Conservative Vector Fields for
Multi-Label Classification.

This module implements two variants of the LEF framework:
  1. LEF_ODE  — Continuous dynamics via Neural ODE (torchdiffeq)
  2. LEF_DEQ  — Steady-state via Deep Equilibrium Model (Broyden root-finding)

Core Idea:
  Instead of a scalar energy E(x, y) whose gradient gives forces, we define a
  *directed* vector field F(x, y) over label space.  The field is deliberately
  **non-conservative** (curl != 0) because the label-to-label interaction
  matrix alpha is **asymmetric**: alpha_{j->i}(x) != alpha_{i->j}(x).

  This asymmetry is enforced architecturally: each label j has a separate
  "source" embedding (how it influences others) and each label i has a separate
  "target" embedding (how it is influenced), so the pairwise coupling
  alpha_{j->i} is computed from (source_j, target_i, x), which is structurally
  different from (source_i, target_j, x).

Modules:
  - LabelForceField: Core F(x, y) computation.
  - ODEFunc: Wraps LabelForceField for torchdiffeq interface.
  - LEF_ODE: Full ODE model (init -> integrate -> predict).
  - BroydenSolver: Quasi-Newton root finder for F(x, y*) = 0.
  - DEQImplicitFunction: Custom autograd for implicit differentiation.
  - LEF_DEQ: Full DEQ model (init -> root-find -> predict).
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Utility: simple MLP builder
# ---------------------------------------------------------------------------

def _build_mlp(dims: list, activation: str = "relu", dropout: float = 0.0) -> nn.Sequential:
    """Build a simple feedforward MLP.

    Args:
        dims: List of layer dimensions, e.g. [128, 64, 32].
        activation: Activation function name ("relu", "tanh", "gelu").
        dropout: Dropout probability after each hidden layer.

    Returns:
        nn.Sequential module.
    """
    act_fn = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}[activation]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:  # no activation/dropout after last layer
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


# ===================================================================
# CORE MODULE: LabelForceField
# ===================================================================

class LabelForceField(nn.Module):
    """Computes the force vector F(x, y) over label space.

    For each label i:
        F_i(x, y) = F^local_i(x, y) + F^field_i(x, y)

    where:
        F^local_i(x, y) = sigma(MLP_local(x))_i - y_i
            A restoring force that pushes y_i toward the input-only prediction.

        F^field_i(x, y) = sum_{j != i} alpha_{j->i}(x) * y_j
            An asymmetric coupling where label j's current activation pushes
            label i through a directed, input-dependent coefficient.

    ASYMMETRIC ATTENTION MECHANISM (alpha):
    ----------------------------------------
    The key novelty: alpha_{j->i}(x) != alpha_{i->j}(x) by construction.

    Each label has TWO learned embeddings:
        - source_emb[j] in R^d : how label j *emits* influence
        - target_emb[i] in R^d : how label i *receives* influence

    The coupling coefficient is:
        alpha_{j->i}(x) = MLP_alpha( [source_emb[j] || target_emb[i] || x_proj] )

    Since source_emb[j] != target_emb[j] in general, and the roles are not
    symmetric in the concatenation, we get alpha_{j->i} != alpha_{i->j}.

    This is conceptually different from a symmetric energy model where
    E(y) = y^T W y forces W to have symmetric influence.

    Args:
        input_dim: Dimension of input features x.
        num_labels: Number of labels L.
        hidden_dim: Hidden dimension for internal MLPs.
        embed_dim: Dimension of source/target label embeddings.
        dropout: Dropout rate in MLPs.
    """

    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        hidden_dim: int = 256,
        embed_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_labels = num_labels
        self.embed_dim = embed_dim

        # --- Local force: MLP that maps input x -> L logits ---
        self.mlp_local = _build_mlp(
            [input_dim, hidden_dim, hidden_dim, num_labels],
            activation="relu",
            dropout=dropout,
        )

        # --- Asymmetric label embeddings ---
        # source_emb[j]: how label j influences others (its "outgoing" role)
        self.source_emb = nn.Parameter(torch.randn(num_labels, embed_dim) * 0.02)
        # target_emb[i]: how label i is influenced by others (its "incoming" role)
        self.target_emb = nn.Parameter(torch.randn(num_labels, embed_dim) * 0.02)

        # --- EFFICIENT alpha computation via input-dependent bilinear form ---
        #
        # Instead of materializing a (B, L, L, 3d) tensor and running an MLP
        # (which is O(B * L^2 * d) memory and very slow for large L), we use
        # an input-conditioned bilinear factorization:
        #
        #   alpha_{j->i}(x) = source_emb[j]^T @ W(x) @ target_emb[i]
        #
        # where W(x) in R^{d x d} is a low-rank, input-dependent coupling matrix:
        #   W(x) = sum_k  g_k(x) * U_k @ V_k^T    (K rank-1 components)
        #
        # This is computed as:
        #   1. Project x -> gating weights g(x) in R^K  (MLP)
        #   2. Compute weighted sum of rank-1 matrices: W(x) = U diag(g(x)) V^T
        #   3. alpha = source_emb @ W(x) @ target_emb^T   (B, L, L)
        #
        # Complexity: O(B * K * d + B * L * d) instead of O(B * L^2 * d)
        #
        # ASYMMETRY IS PRESERVED because source_emb != target_emb, and U != V,
        # so alpha_{j->i} = src[j]^T @ U @ diag(g(x)) @ V^T @ tgt[i]
        #    alpha_{i->j} = src[i]^T @ U @ diag(g(x)) @ V^T @ tgt[j]
        # These are different whenever source_emb != target_emb (which is the
        # general case since they are separate learned parameters).

        self.alpha_rank = min(embed_dim, 16)  # K: number of rank-1 components
        # U, V: factored coupling matrices (d x K each)
        self.U = nn.Parameter(torch.randn(embed_dim, self.alpha_rank) * 0.02)
        self.V = nn.Parameter(torch.randn(embed_dim, self.alpha_rank) * 0.02)
        # Gating MLP: x -> K weights, bounded by tanh to prevent explosion
        self.alpha_gate = _build_mlp(
            [input_dim, hidden_dim, self.alpha_rank],
            activation="relu",
            dropout=dropout,
        )

        # --- Fixed scale for field force (NOT learnable -- prevents runaway) ---
        # With tanh-bounded gates and small embeddings, 0.1 is safe and lets
        # the field force meaningfully contribute to label interactions.
        self.field_scale = 0.1

    def compute_local_force(self, x: Tensor, z: Tensor) -> Tensor:
        """Compute local restoring force in LOGIT space: MLP(x) - z.

        We operate in unconstrained logit space so the ODE dynamics are
        not fighting against sigmoid saturation. The restoring force
        pulls z toward the raw logits from the input-only MLP.

        Args:
            x: Input features, shape (batch, input_dim).
            z: Current label logits, shape (batch, num_labels).

        Returns:
            Local force, shape (batch, num_labels).
        """
        local_logits = self.mlp_local(x)  # (batch, L) -- raw logits, no sigmoid
        return local_logits - z

    def compute_field_force(self, x: Tensor, z: Tensor) -> Tensor:
        """Compute asymmetric field force EFFICIENTLY via factored bilinear form.

        The field force uses label PROBABILITIES y = sigmoid(z) as activations
        (since "how much label j is on" is naturally a probability), while the
        force itself acts in logit space.

            F^field_i = sum_{j!=i} alpha_{j->i}(x) * sigmoid(z_j)

        where alpha_{j->i}(x) = source[j]^T @ U @ diag(g(x)) @ V^T @ target[i]

        The computation is factored as:
            1. g = gate_MLP(x)                          -> (B, K)
            2. S = source_emb @ U                       -> (L, K)
            3. T = target_emb @ V                       -> (L, K)
            4. S_gated = S * g(x)                       -> (B, L, K)  [broadcast]
            5. field = (S_gated^T @ y) @ T^T             -> (B, L)

        This avoids ever forming the (B, L, L) matrix!
        Complexity: O(B*L*K) instead of O(B*L^2*d).

        WHY THIS IS ASYMMETRIC:
        -----------------------
        alpha_{j->i}(x) = src[j]^T @ U @ diag(g(x)) @ V^T @ tgt[i]
        alpha_{i->j}(x) = src[i]^T @ U @ diag(g(x)) @ V^T @ tgt[j]

        Since source_emb and target_emb are separate parameters (src[j] != tgt[j]),
        these produce different values, breaking symmetry by construction.

        Args:
            x: Input features, shape (batch, input_dim).
            z: Current label logits, shape (batch, num_labels).

        Returns:
            Field force in logit space, shape (batch, num_labels).
        """
        B = x.shape[0]
        L = self.num_labels

        # Convert logits to probabilities for label activations
        y = torch.sigmoid(z)  # (B, L)

        # Step 1: Input-dependent gating weights, bounded by tanh to [-1, 1]
        g = torch.tanh(self.alpha_gate(x))  # (B, K)

        # Step 2: Project label embeddings through factored matrices
        S = self.source_emb @ self.U  # (L, K) -- source projections
        T = self.target_emb @ self.V  # (L, K) -- target projections

        # Step 3: Gate the source projections by input
        S_gated = S.unsqueeze(0) * g.unsqueeze(1)  # (B, L, K)

        # Step 4: Compute field force WITHOUT forming (B, L, L) alpha matrix.
        # F^field_i = sum_j alpha_{j->i} * y_j
        #           = sum_k T[i,k] * [sum_j S_gated[b,j,k] * y[b,j]]
        h = torch.einsum('bj,bjk->bk', y, S_gated)  # (B, K)
        field = h @ T.t()  # (B, L)

        # Remove self-interaction: subtract the diagonal contribution
        diag_alpha = (S_gated * T.unsqueeze(0)).sum(dim=-1)  # (B, L)
        field = field - diag_alpha * y  # remove j=i term

        return self.field_scale * field

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        """Compute total force F(x, z) = F^local + F^field in logit space.

        All dynamics operate on unconstrained logits z. The field force
        internally converts z -> sigmoid(z) for label activations, but the
        force itself pushes in logit space.

        Args:
            x: Input features, shape (batch, input_dim).
            z: Current label logits, shape (batch, num_labels).

        Returns:
            Total force vector in logit space, shape (batch, num_labels).
        """
        f_local = self.compute_local_force(x, z)
        f_field = self.compute_field_force(x, z)
        return f_local + f_field

    def get_initial_logits(self, x: Tensor) -> Tensor:
        """Compute initial logits z(0) = MLP_local(x).

        Args:
            x: Input features, shape (batch, input_dim).

        Returns:
            Initial logits, shape (batch, num_labels).
        """
        return self.mlp_local(x)


# ===================================================================
# TASK 1: Neural ODE Implementation
# ===================================================================

class ODEFunc(nn.Module):
    """Wraps LabelForceField as an ODE right-hand-side for torchdiffeq.

    The ODE is:
        dy/dt = F_Theta(x, y(t))

    where x is fixed (conditioned on the input) and y(t) evolves.

    torchdiffeq requires the signature f(t, y) -> dy/dt.  We store x
    as an attribute set before each integration call.
    """

    def __init__(self, force_field: LabelForceField):
        super().__init__()
        self.force_field = force_field
        self._x: Optional[Tensor] = None  # set externally before odeint

    def set_input(self, x: Tensor):
        """Cache the input features for the current batch."""
        self._x = x

    def forward(self, t: Tensor, y: Tensor) -> Tensor:
        """ODE right-hand-side: dy/dt = F(x, y).

        Args:
            t: Current time (scalar, unused — autonomous ODE).
            y: Current label state, shape (batch, num_labels).

        Returns:
            dy/dt, shape (batch, num_labels).
        """
        return self.force_field(self._x, y)


class LEF_ODE(nn.Module):
    """Label Energy Fields via Neural ODE.

    Forward pass:
        1. Compute y(0) = sigma(MLP_local(x))  — initial state from input.
        2. Integrate dy/dt = F(x, y(t)) from t=0 to t=1 using adjoint method.
        3. Return y(1) as the final multi-label prediction.

    The adjoint method (odeint_adjoint) computes gradients without storing
    intermediate states, making this memory-efficient for deep dynamics.

    Loss: BCELoss between y(1) and ground-truth multi-hot labels.

    Args:
        input_dim: Dimension of input features.
        num_labels: Number of labels.
        hidden_dim: Hidden dimension for force field MLPs.
        embed_dim: Dimension of source/target label embeddings.
        dropout: Dropout rate.
        rtol: Relative tolerance for ODE solver.
        atol: Absolute tolerance for ODE solver.
        method: ODE solver method ('dopri5', 'euler', 'rk4', etc.).
    """

    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        hidden_dim: int = 256,
        embed_dim: int = 32,
        dropout: float = 0.1,
        rtol: float = 1e-3,
        atol: float = 1e-4,
        method: str = "dopri5",
    ):
        super().__init__()
        self.force_field = LabelForceField(
            input_dim, num_labels, hidden_dim, embed_dim, dropout
        )
        self.ode_func = ODEFunc(self.force_field)
        self.rtol = rtol
        self.atol = atol
        self.method = method

        # Integration time span: t=0 to t=1
        self.register_buffer("t_span", torch.tensor([0.0, 1.0]))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: integrate the force field in logit space, then sigmoid.

        The dynamics evolve logits z(t) (unconstrained), not probabilities.
        This avoids the ODE fighting sigmoid saturation and keeps gradients
        healthy. Final output is sigmoid(z(1)) for probabilities.

        Args:
            x: Input features, shape (batch, input_dim).

        Returns:
            Logits z(1) if self.return_logits, else sigmoid(z(1)).
            Shape: (batch, num_labels).
        """
        from torchdiffeq import odeint_adjoint

        # Step 1: Initial logits z(0) = MLP_local(x)
        z0 = self.force_field.get_initial_logits(x)  # (batch, L)

        # Step 2: Set the input for the ODE function
        self.ode_func.set_input(x)

        # Step 3: Integrate dz/dt = F(x, z(t)) from t=0 to t=1
        # For fixed-step solvers (euler, rk4), use step_size=0.1 for 10 substeps
        # so the restoring force has time to dampen overshooting.
        solver_opts = {}
        if self.method in ("euler", "rk4", "midpoint"):
            solver_opts["step_size"] = 0.1

        z_traj = odeint_adjoint(
            self.ode_func,
            z0,
            self.t_span,
            rtol=self.rtol,
            atol=self.atol,
            method=self.method,
            options=solver_opts,
        )

        # Step 4: Extract z(1) and convert to probabilities
        z_final = z_traj[-1]  # (batch, L)
        return z_final  # return logits; use BCEWithLogitsLoss


# ===================================================================
# TASK 2: Deep Equilibrium (DEQ) Implementation
# ===================================================================

# --- Broyden's Method for Root Finding ---

class BroydenSolver:
    """Broyden's method for finding roots of F(y) = 0.

    Broyden's method is a quasi-Newton method that approximates the Jacobian
    inverse using rank-1 updates, avoiding expensive Jacobian computation.

    Given F(y) = 0:
        1. Start with y_0 and approximate inverse Jacobian J_0^{-1} = -I.
        2. At each step:
           - Compute delta_y = -J_k^{-1} F(y_k)
           - Update y_{k+1} = y_k + delta_y
           - Compute delta_F = F(y_{k+1}) - F(y_k)
           - Update J_{k+1}^{-1} via Sherman-Morrison formula:
             J_{k+1}^{-1} = J_k^{-1} + (delta_y - J_k^{-1} delta_F) delta_y^T J_k^{-1}
                             / (delta_y^T J_k^{-1} delta_F)

    We use the "good Broyden" update which maintains the inverse Jacobian
    directly, avoiding any matrix inversion.

    Args:
        max_iter: Maximum number of Broyden iterations.
        tol: Convergence tolerance on ||F(y)||.
        stop_mode: Whether to use 'abs' (absolute) or 'rel' (relative) tolerance.
        ls: Whether to use a simple line-search backtracking.
    """

    def __init__(
        self,
        max_iter: int = 30,
        tol: float = 1e-5,
        stop_mode: str = "abs",
        ls: bool = False,
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.stop_mode = stop_mode
        self.ls = ls

    @torch.no_grad()
    def solve(
        self,
        func,
        y0: Tensor,
    ) -> Tuple[Tensor, dict]:
        """Find y* such that func(y*) ≈ 0 using Broyden's method.

        We flatten the label dimension for Broyden updates and reshape back.

        Args:
            func: Callable y -> F(y), where y is (batch, L) and F(y) is (batch, L).
            y0: Initial guess, shape (batch, L).

        Returns:
            y_star: Approximate root, shape (batch, L).
            info: Dictionary with convergence diagnostics.
        """
        B, L = y0.shape
        device = y0.device

        # Flatten to (B, L) — already flat, just ensure contiguous
        y = y0.clone()
        F_y = func(y)  # (B, L)

        # Initialize inverse Jacobian approximation as -I (standard choice)
        # We store it as (B, L, L) for batched operation
        J_inv = -torch.eye(L, device=device).unsqueeze(0).expand(B, -1, -1).clone()

        best_y = y.clone()
        best_residual = F_y.norm(dim=-1).mean().item()
        nstep = 0

        for k in range(self.max_iter):
            # Step direction: delta_y = -J_inv @ F(y)
            delta_y = -torch.bmm(J_inv, F_y.unsqueeze(-1)).squeeze(-1)  # (B, L)

            # Line search (optional simple backtracking)
            step_size = 1.0
            if self.ls:
                for _ in range(5):
                    y_new = y + step_size * delta_y
                    F_new = func(y_new)
                    if F_new.norm(dim=-1).mean() < F_y.norm(dim=-1).mean():
                        break
                    step_size *= 0.5
                else:
                    y_new = y + step_size * delta_y
                    F_new = func(y_new)
            else:
                y_new = y + delta_y
                F_new = func(y_new)

            # Convergence check
            residual = F_new.norm(dim=-1).mean().item()
            nstep = k + 1

            if residual < best_residual:
                best_residual = residual
                best_y = y_new.clone()

            if residual < self.tol:
                break

            # Broyden rank-1 update of inverse Jacobian (Sherman-Morrison)
            # delta_F = F(y_new) - F(y)
            delta_F = F_new - F_y  # (B, L)

            # numerator = delta_y - J_inv @ delta_F
            J_inv_dF = torch.bmm(J_inv, delta_F.unsqueeze(-1)).squeeze(-1)  # (B, L)
            num = delta_y - J_inv_dF  # (B, L)

            # denominator = delta_y^T @ J_inv @ delta_F
            denom = (delta_y * J_inv_dF).sum(dim=-1, keepdim=True)  # (B, 1)
            denom = denom.unsqueeze(-1)  # (B, 1, 1)

            # Avoid division by zero
            denom = denom + 1e-12 * denom.sign()

            # J_inv update: J_inv += (num @ delta_y^T @ J_inv) / denom
            # num: (B, L, 1), delta_y^T @ J_inv: (B, 1, L)
            dy_J_inv = torch.bmm(delta_y.unsqueeze(1), J_inv)  # (B, 1, L)
            update = torch.bmm(num.unsqueeze(-1), dy_J_inv) / denom  # (B, L, L)
            J_inv = J_inv + update

            # Advance
            y = y_new
            F_y = F_new

        info = {
            "nstep": nstep,
            "residual": best_residual,
            "converged": best_residual < self.tol,
        }

        return best_y, info


# --- Custom Autograd for Implicit Differentiation ---

class DEQImplicitFunction(torch.autograd.Function):
    """Custom autograd function implementing implicit differentiation for DEQ.

    FORWARD PASS:
        Given x, find y* such that F(x, y*) = 0 using Broyden's method.
        Return y* (detached from the root-finding computation graph).

    BACKWARD PASS (Implicit Function Theorem):
        At equilibrium, F(x, y*) = 0.  By the implicit function theorem:

            dL/d_theta = -dL/dy* @ (dF/dy*)^{-1} @ dF/d_theta

        where dL/dy* is the incoming gradient from the loss.

        Implementation:
        1. We receive grad_output = dL/dy* from downstream.
        2. We need to solve: v^T (dF/dy*) = grad_output^T
           i.e., v = (dF/dy*)^{-T} @ grad_output
           This is equivalent to solving (dF/dy*)^T v = grad_output.
        3. We solve this linear system using a fixed-point iteration:
           v_{k+1} = grad_output - (dF/dy*)^T v_k + v_k
           (which converges to v = (I - (dF/dy*)^T)^{-1} grad_output
           when the spectral radius of dF/dy* is < 1).
        4. Then dL/d_theta is obtained by backpropagating v through F.

    In practice, we use a simpler approach: compute the VJP (vector-Jacobian
    product) by calling autograd on F(x, y*) with y* requiring grad, and
    solve the linear system via Neumann series or fixed-point iteration.
    """

    @staticmethod
    def forward(ctx, force_field, x, y0, solver_kwargs):
        """Forward: find equilibrium y* where F(x, y*) = 0.

        Args:
            ctx: Autograd context for saving tensors.
            force_field: LabelForceField module.
            x: Input features (batch, input_dim).
            y0: Initial guess (batch, num_labels).
            solver_kwargs: Dict of Broyden solver parameters.

        Returns:
            y_star: Equilibrium point (batch, num_labels).
        """
        solver = BroydenSolver(**solver_kwargs)

        def func(y):
            return force_field(x, y)

        y_star, info = solver.solve(func, y0)

        # Save for backward: we need x and y* (with grad tracking)
        # We detach y_star because Broyden iterations shouldn't be in the graph
        ctx.save_for_backward(x, y_star.detach())
        ctx.force_field = force_field
        ctx.info = info

        return y_star.detach().requires_grad_(x.requires_grad or any(
            p.requires_grad for p in force_field.parameters()
        ))

    @staticmethod
    def backward(ctx, grad_output):
        """Backward: implicit differentiation via the implicit function theorem.

        At equilibrium F(x, y*) = 0:
            dL/d_theta = -(dF/d_theta)^T @ (dF/dy*)^{-T} @ dL/dy*

        We compute (dF/dy*)^{-T} @ dL/dy* via fixed-point iteration:
            v_{k+1} = dL/dy* - (dF/dy*)^T @ v_k + v_k
                     = dL/dy* + (I - (dF/dy*)^T) @ v_k

        Then we backprop through F(x, y*) using v as the output gradient
        to get gradients w.r.t. theta (force_field parameters) and x.

        Args:
            ctx: Autograd context.
            grad_output: dL/dy*, shape (batch, num_labels).

        Returns:
            Tuple of gradients: (None for force_field, grad_x, None for y0,
                                 None for solver_kwargs).
        """
        x, y_star = ctx.saved_tensors
        force_field = ctx.force_field

        # Enable gradient computation for the linear solve
        x_req = x.detach().requires_grad_(True)
        y_req = y_star.detach().requires_grad_(True)

        # Compute F(x, y*) with grad tracking
        with torch.enable_grad():
            F_val = force_field(x_req, y_req)  # (B, L)

        # Solve for v: (dF/dy*)^T @ v = grad_output
        # Using fixed-point iteration (Anderson/Neumann):
        #   v_{k+1} = grad_output + v_k - JF_y^T @ v_k
        # where JF_y = dF/dy* is the Jacobian of F w.r.t. y at y*.
        #
        # This converges when spectral radius of JF_y < 1 (which is
        # encouraged by the restoring force F^local = sigma(.) - y
        # that contributes -I to the Jacobian).

        v = grad_output.clone()
        max_iter = 25
        tol = 1e-5

        for _ in range(max_iter):
            # Compute JF_y^T @ v via vector-Jacobian product
            # torch.autograd.grad computes v^T @ JF_y, but we want JF_y^T @ v
            # Actually: torch.autograd.grad(F, y, v) = (dF/dy)^T @ v
            # That's exactly what we need!
            with torch.enable_grad():
                JFy_T_v = torch.autograd.grad(
                    F_val, y_req, v, retain_graph=True, create_graph=False
                )[0]

            v_new = grad_output + v - JFy_T_v
            if (v_new - v).norm() < tol * (v.norm() + 1e-8):
                v = v_new
                break
            v = v_new

        # Now backprop through F(x, y*) with v as the surrogate gradient
        # to get parameter gradients and input gradient.
        # We need: dL/d_theta = -v^T @ dF/d_theta
        #          dL/dx = -v^T @ dF/dx
        # The negative sign comes from implicit differentiation.

        # Recompute F with all parameters requiring grad
        x_grad = x.detach().requires_grad_(True)
        y_grad = y_star.detach().requires_grad_(True)

        with torch.enable_grad():
            F_recomp = force_field(x_grad, y_grad)

        # Backprop: compute gradients of sum(F * v) w.r.t. parameters and x
        # This gives dF/d_theta^T @ v for each parameter
        params = [p for p in force_field.parameters() if p.requires_grad]
        all_grads = torch.autograd.grad(
            F_recomp, [x_grad] + params, -v,
            retain_graph=False, allow_unused=True,
        )

        grad_x = all_grads[0]
        param_grads = all_grads[1:]

        # Manually set .grad on parameters (accumulate since autograd expects this)
        for p, g in zip(params, param_grads):
            if g is not None:
                if p.grad is None:
                    p.grad = g.clone()
                else:
                    p.grad = p.grad + g

        # Return: (force_field, x, y0, solver_kwargs) — only x gets a gradient
        return None, grad_x, None, None


class LEF_DEQ(nn.Module):
    """Label Energy Fields via Deep Equilibrium Model.

    Instead of integrating an ODE to a fixed time, we find the steady-state
    y* where the force field vanishes: F(x, y*) = 0.

    Forward pass:
        1. Compute y(0) = sigma(MLP_local(x)) — initial guess.
        2. Use Broyden's method to find y* such that F(x, y*) ≈ 0.
        3. Return y* as the final prediction.

    Backward pass:
        Uses implicit differentiation (via DEQImplicitFunction) to compute
        gradients without backpropagating through the solver iterations.

    Loss: BCELoss between y* and ground-truth labels.

    Args:
        input_dim: Dimension of input features.
        num_labels: Number of labels.
        hidden_dim: Hidden dimension for force field MLPs.
        embed_dim: Dimension of source/target label embeddings.
        dropout: Dropout rate.
        broyden_max_iter: Max Broyden iterations.
        broyden_tol: Convergence tolerance for Broyden.
    """

    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        hidden_dim: int = 256,
        embed_dim: int = 32,
        dropout: float = 0.1,
        broyden_max_iter: int = 30,
        broyden_tol: float = 1e-5,
    ):
        super().__init__()
        self.force_field = LabelForceField(
            input_dim, num_labels, hidden_dim, embed_dim, dropout
        )
        self.solver_kwargs = {
            "max_iter": broyden_max_iter,
            "tol": broyden_tol,
            "stop_mode": "abs",
            "ls": True,
        }

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: find equilibrium logits z* where F(x, z*) = 0.

        Operates in logit space. Returns logits for BCEWithLogitsLoss.

        Args:
            x: Input features, shape (batch, input_dim).

        Returns:
            z_star: Equilibrium logits, shape (batch, num_labels).
        """
        # Step 1: Initial guess in logit space
        z0 = self.force_field.get_initial_logits(x)

        if self.training:
            # Step 2a: Use custom autograd for implicit differentiation
            z_star = DEQImplicitFunction.apply(
                self.force_field, x, z0, self.solver_kwargs
            )
        else:
            # Step 2b: At eval time, just run the solver (no grad needed)
            solver = BroydenSolver(**self.solver_kwargs)
            with torch.no_grad():
                z_star, info = solver.solve(
                    lambda z: self.force_field(x, z), z0
                )

        return z_star  # return logits; use BCEWithLogitsLoss


# ===================================================================
# CONVENIENCE: Combined model factory
# ===================================================================

def create_lef_model(
    variant: str,
    input_dim: int,
    num_labels: int,
    hidden_dim: int = 256,
    embed_dim: int = 32,
    dropout: float = 0.1,
    **kwargs,
) -> nn.Module:
    """Factory function to create LEF models.

    Args:
        variant: "ode" or "deq".
        input_dim: Dimension of input features.
        num_labels: Number of labels.
        hidden_dim: Hidden dim for MLPs.
        embed_dim: Label embedding dim.
        dropout: Dropout rate.
        **kwargs: Extra args passed to the specific variant.

    Returns:
        LEF model instance.
    """
    if variant == "ode":
        return LEF_ODE(
            input_dim, num_labels, hidden_dim, embed_dim, dropout, **kwargs
        )
    elif variant == "deq":
        return LEF_DEQ(
            input_dim, num_labels, hidden_dim, embed_dim, dropout, **kwargs
        )
    else:
        raise ValueError(f"Unknown variant: {variant}. Use 'ode' or 'deq'.")
