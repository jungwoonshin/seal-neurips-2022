"""
Dark Labels: Latent Structural Catalysts for SEAL

Standard SEAL suffers from gradient conflict on rare label combinations:
BCE pushes toward ground truth while the energy loss pushes back (energy
surface penalizes statistically rare-but-correct patterns). The solution
expands the energy function's phase space with K latent unconstrained
variables ("dark labels") that can absorb energy penalties for anomalous
instances, leaving BCE gradients unimpeded.

Usage:
    python dark_labels.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskNetWithCatalysts(nn.Module):
    """Task network that predicts both visible labels y and latent catalysts z.

    Follows SEAL's MultilabelTaskNN pattern: shared feature_network backbone
    followed by a single linear projection, split into bounded y (sigmoid)
    and unconstrained z.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_labels: int,
        num_catalysts: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.num_catalysts = num_catalysts

        # Shared backbone: 2-layer FeedForward with Softplus (matches SEAL pattern)
        self.feature_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Softplus(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Dropout(dropout),
        )

        # Single projection to expanded label space
        self.output_head = nn.Linear(hidden_dim, num_labels + num_catalysts)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, input_dim)
        Returns:
            y_pred: (batch, num_labels) — sigmoid-bounded [0, 1]
            z_pred: (batch, num_catalysts) — unconstrained R^K
        """
        features = self.feature_network(x)
        logits = self.output_head(features)  # (batch, L + K)
        y_pred = torch.sigmoid(logits[:, : self.num_labels])
        z_pred = logits[:, self.num_labels :]
        return y_pred, z_pred


class ExpandedEnergyNet(nn.Module):
    """Energy network over expanded [y, z] space.

    Follows SEAL's Elocal + Eglobal decomposition:
      - Elocal: bilinear interaction between input features and label embeddings,
        weighted by the expanded label vector [y, z].
      - Eglobal: v^T softplus(M @ [y, z]) capturing higher-order structure.
    """

    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        num_catalysts: int,
        hidden_dim: int,
        global_hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.total_dim = num_labels + num_catalysts

        # Energy-net's own feature network (separate from task-net, matches SEAL)
        self.feature_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Softplus(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Dropout(dropout),
        )

        # Elocal: label embeddings for bilinear scoring
        self.label_embeddings = nn.Embedding(self.total_dim, hidden_dim)

        # Eglobal: v^T softplus(M @ [y, z])
        self.global_linear = nn.Linear(self.total_dim, global_hidden_dim)
        self.global_projection = nn.Parameter(
            torch.randn(global_hidden_dim) * (2.0 / global_hidden_dim) ** 0.5
        )

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim)
            y: (batch, num_labels)
            z: (batch, num_catalysts)
        Returns:
            energy: (batch,) scalar energy per instance
        """
        yz = torch.cat([y, z], dim=-1)  # (batch, L + K)

        # Elocal: sum(features @ embeddings.T * yz, dim=-1)
        features = self.feature_network(x)  # (batch, hidden_dim)
        # (batch, hidden_dim) @ (hidden_dim, L+K) -> (batch, L+K)
        scores = torch.matmul(features, self.label_embeddings.weight.T)
        e_local = (scores * yz).sum(dim=-1)  # (batch,)

        # Eglobal: v^T softplus(M @ [y, z])
        e_global = torch.matmul(
            F.softplus(self.global_linear(yz)), self.global_projection
        )  # (batch,)

        return e_local + e_global


def train_step(
    task_net: TaskNetWithCatalysts,
    energy_net: ExpandedEnergyNet,
    optimizer_task: torch.optim.Optimizer,
    optimizer_energy: torch.optim.Optimizer,
    x: torch.Tensor,
    y_star: torch.Tensor,
    margin: float = 1.0,
    lam: float = 0.1,
    z_reg: float = 0.01,
    n_energy_steps: int = 1,
    n_task_steps: int = 1,
) -> dict:
    """Alternating optimization step matching SEAL's GradientDescentMiniMaxTrainer.

    Step A: Update energy-net to separate positive (ground truth) from negative
            (task-net predictions) via hinge loss.
    Step B: Update task-net with BCE + lambda * energy, where catalysts z flow
            gradients only through the energy term.

    Args:
        task_net: Task network predicting (y, z).
        energy_net: Energy network scoring (x, y, z) tuples.
        optimizer_task: Optimizer for task_net parameters.
        optimizer_energy: Optimizer for energy_net parameters.
        x: (batch, input_dim) input features.
        y_star: (batch, num_labels) ground-truth binary labels.
        margin: Hinge loss margin for energy training.
        lam: Weight for energy loss in task-net objective.
        z_reg: L2 regularization coefficient on catalyst norms to prevent
            unbounded growth (Elocal is linear in z).
        n_energy_steps: Number of inner energy-net update steps.
        n_task_steps: Number of outer task-net update steps.

    Returns:
        Dict of scalar metrics for monitoring.
    """
    metrics = {}

    # --- Step A: Energy-Net update ---
    for step_i in range(n_energy_steps):
        optimizer_energy.zero_grad()

        with torch.no_grad():
            y_neg, z_neg = task_net(x)

        # Positive: ground-truth labels + catalyst values from current task-net
        e_pos = energy_net(x, y_star, z_neg)
        # Negative: task-net predictions
        e_neg = energy_net(x, y_neg, z_neg)

        # Hinge: want E(positive) < E(negative) by at least margin
        loss_energy_train = F.relu(e_pos - e_neg + margin).mean()
        loss_energy_train.backward()
        optimizer_energy.step()

    metrics["loss_energy_hinge"] = loss_energy_train.item()

    # --- Step B: Task-Net update ---
    for step_i in range(n_task_steps):
        optimizer_task.zero_grad()

        y_pred, z_pred = task_net(x)

        # BCE on visible labels only
        loss_bce = F.binary_cross_entropy(y_pred, y_star)

        # Energy: z_pred NOT detached -- gradients flow through catalysts
        loss_energy = energy_net(x, y_pred, z_pred).mean()

        # L2 regularization prevents catalysts from growing unboundedly
        # (Elocal is linear in z, so without this z -> -inf to minimize energy)
        loss_z_reg = z_reg * (z_pred ** 2).mean()

        loss_task = loss_bce + lam * loss_energy + loss_z_reg
        loss_task.backward()
        optimizer_task.step()

    metrics["loss_bce"] = loss_bce.item()
    metrics["loss_energy"] = loss_energy.item()
    metrics["loss_z_reg"] = loss_z_reg.item()
    metrics["loss_task"] = loss_task.item()
    metrics["z_norm"] = z_pred.norm(dim=-1).mean().item()

    return metrics


# ============================================================================
# Verification Script
# ============================================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    INPUT_DIM = 32
    HIDDEN_DIM = 64
    NUM_LABELS = 10
    NUM_CATALYSTS = 4
    GLOBAL_HIDDEN_DIM = 16
    BATCH_SIZE = 128

    # ------------------------------------------------------------------
    # Part 1: Gradient Flow Verification
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Part 1: Gradient Flow Verification")
    print("=" * 60)

    task_net = TaskNetWithCatalysts(
        INPUT_DIM, HIDDEN_DIM, NUM_LABELS, NUM_CATALYSTS
    ).to(device)
    energy_net = ExpandedEnergyNet(
        INPUT_DIM, NUM_LABELS, NUM_CATALYSTS, HIDDEN_DIM, GLOBAL_HIDDEN_DIM
    ).to(device)

    x = torch.randn(4, INPUT_DIM, device=device)
    y_star = torch.zeros(4, NUM_LABELS, device=device)
    y_star[:, :3] = 1.0  # sparse labels

    y_pred, z_pred = task_net(x)
    y_pred.retain_grad()
    z_pred.retain_grad()

    # Test 1: BCE should NOT produce gradients on z_pred
    loss_bce = F.binary_cross_entropy(y_pred, y_star)
    loss_bce.backward(retain_graph=True)

    z_grad_after_bce = z_pred.grad
    if z_grad_after_bce is None:
        print(f"  z_pred.grad after BCE: None (as expected)")
    else:
        print(f"  z_pred.grad after BCE: {z_grad_after_bce.abs().sum().item():.6f}")

    y_grad_after_bce = y_pred.grad.abs().sum().item()
    print(f"  y_pred.grad after BCE: {y_grad_after_bce:.6f}")
    assert z_grad_after_bce is None or z_grad_after_bce.abs().sum().item() == 0.0, (
        "FAIL: BCE should not produce gradients on z_pred"
    )
    print("  [PASS] BCE does not affect catalysts z")

    # Test 2: Energy SHOULD produce gradients on z_pred
    task_net.zero_grad()
    energy_net.zero_grad()
    # Reset grads accumulated from BCE
    y_pred.grad = None
    z_pred.grad = None

    loss_energy = energy_net(x, y_pred, z_pred).mean()
    loss_energy.backward()

    z_grad_after_energy = z_pred.grad.abs().sum().item()
    y_grad_after_energy = y_pred.grad.abs().sum().item()
    print(f"\n  z_pred.grad after Energy: {z_grad_after_energy:.6f}")
    print(f"  y_pred.grad after Energy: {y_grad_after_energy:.6f}")
    assert z_grad_after_energy > 0.0, (
        "FAIL: Energy should produce gradients on z_pred"
    )
    print("  [PASS] Energy produces gradients on catalysts z")
    print()

    # ------------------------------------------------------------------
    # Part 2: Training Loop on Synthetic Data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Part 2: Training Loop on Synthetic Data")
    print("=" * 60)

    torch.manual_seed(123)

    # Fresh networks
    task_net = TaskNetWithCatalysts(
        INPUT_DIM, HIDDEN_DIM, NUM_LABELS, NUM_CATALYSTS, dropout=0.0
    ).to(device)
    energy_net = ExpandedEnergyNet(
        INPUT_DIM, NUM_LABELS, NUM_CATALYSTS, HIDDEN_DIM, GLOBAL_HIDDEN_DIM, dropout=0.0
    ).to(device)

    optimizer_task = torch.optim.Adam(task_net.parameters(), lr=1e-3)
    optimizer_energy = torch.optim.Adam(energy_net.parameters(), lr=1e-3)

    # Generate synthetic dataset: random inputs with sparse binary labels
    N_SAMPLES = 512
    x_data = torch.randn(N_SAMPLES, INPUT_DIM, device=device)
    # Sparse labels: each sample has ~30% active labels
    y_data = (torch.rand(N_SAMPLES, NUM_LABELS, device=device) < 0.3).float()

    NUM_EPOCHS = 50
    print(f"\n  Training for {NUM_EPOCHS} epochs, {N_SAMPLES} samples, "
          f"batch_size={BATCH_SIZE}")
    print(f"  {'Epoch':>5}  {'BCE':>8}  {'Energy':>8}  {'Hinge':>8}  {'z_norm':>8}")
    print("  " + "-" * 45)

    history = {"bce": [], "energy": [], "hinge": [], "z_norm": []}

    for epoch in range(NUM_EPOCHS):
        # Shuffle
        perm = torch.randperm(N_SAMPLES, device=device)
        epoch_metrics = {"loss_bce": 0, "loss_energy": 0, "loss_energy_hinge": 0, "z_norm": 0}
        n_batches = 0

        for i in range(0, N_SAMPLES, BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            x_batch = x_data[idx]
            y_batch = y_data[idx]

            m = train_step(
                task_net, energy_net,
                optimizer_task, optimizer_energy,
                x_batch, y_batch,
                margin=1.0, lam=0.01, z_reg=0.1,
                n_energy_steps=1, n_task_steps=1,
            )
            for k in epoch_metrics:
                epoch_metrics[k] += m[k]
            n_batches += 1

        # Average over batches
        for k in epoch_metrics:
            epoch_metrics[k] /= n_batches

        history["bce"].append(epoch_metrics["loss_bce"])
        history["energy"].append(epoch_metrics["loss_energy"])
        history["hinge"].append(epoch_metrics["loss_energy_hinge"])
        history["z_norm"].append(epoch_metrics["z_norm"])

        if epoch % 5 == 0 or epoch == NUM_EPOCHS - 1:
            print(
                f"  {epoch:5d}  "
                f"{epoch_metrics['loss_bce']:8.4f}  "
                f"{epoch_metrics['loss_energy']:8.4f}  "
                f"{epoch_metrics['loss_energy_hinge']:8.4f}  "
                f"{epoch_metrics['z_norm']:8.4f}"
            )

    # ------------------------------------------------------------------
    # Final Verification
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)

    bce_decreased = history["bce"][-1] < history["bce"][0]
    z_nontrivial = history["z_norm"][-1] > 0.01

    print(f"  BCE decreased:  {history['bce'][0]:.4f} -> {history['bce'][-1]:.4f}  "
          f"{'[PASS]' if bce_decreased else '[WARN]'}")
    print(f"  z_norm active:  {history['z_norm'][0]:.4f} -> {history['z_norm'][-1]:.4f}  "
          f"{'[PASS]' if z_nontrivial else '[WARN]'}")
    print()

    if bce_decreased and z_nontrivial:
        print("  All checks passed.")
    else:
        print("  Some checks did not pass -- review output above.")
