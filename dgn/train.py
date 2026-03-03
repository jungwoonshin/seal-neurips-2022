"""Direct Gradient Networks - Training Script.

Alternating training loop:
  Step A: Train G_Theta to predict the ES oracle direction.
  Step B: Train F_Phi by injecting G_Theta's predicted gradient into the backward pass.

Usage:
    python -m dgn.train [--epochs 40] [--device cuda]
"""

import argparse
import json
import os
import time
from datetime import timedelta
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from dgn.data import make_dataloaders
from dgn.gradient_net import GradientNet
from dgn.metrics import instance_f1
from dgn.oracle import compute_oracle_direction


# ---------------------------------------------------------------------------
# Task-Net: simple feedforward multi-label classifier F_Phi(x) -> y_pred
# ---------------------------------------------------------------------------
class TaskNet(nn.Module):
    """Feedforward task network matching SEAL's bibtex architecture."""

    def __init__(self, input_dim: int, label_dim: int, hidden_dim: int = 400):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Softplus(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
        )
        self.classifier = nn.Linear(hidden_dim, label_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns continuous predictions in [0, 1]."""
        features = self.feature_net(x)
        return torch.sigmoid(self.classifier(features))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    task_net: TaskNet,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate on a dataset, returning loss and F1 metrics."""
    task_net.eval()
    total_bce = 0.0
    total_f1 = 0.0
    n = 0

    for x, y_true in loader:
        x, y_true = x.to(device), y_true.to(device)
        y_pred = task_net(x)

        bce = F.binary_cross_entropy(y_pred, y_true, reduction="mean")
        f1 = instance_f1(y_pred, y_true).mean()

        bs = x.shape[0]
        total_bce += bce.item() * bs
        total_f1 += f1.item() * bs
        n += bs

    return {
        "bce": total_bce / n,
        "f1": total_f1 / n,
    }


# ---------------------------------------------------------------------------
# Single training step
# ---------------------------------------------------------------------------
def train_step(
    x: torch.Tensor,
    y_true: torch.Tensor,
    task_net: TaskNet,
    grad_net: GradientNet,
    optimizer_F: torch.optim.Optimizer,
    optimizer_G: torch.optim.Optimizer,
    lambda_1: float,
    lambda_2: float,
    es_K: int,
    es_sigma: float,
) -> Dict[str, float]:
    """One alternating DGN training step.

    Step A: Train G_Theta to match the ES oracle direction.
    Step B: Train F_Phi via gradient injection from G_Theta.

    Returns dict of scalar metrics for logging.
    """
    # ===== Step A: Train the Gradient-Net =====
    grad_net.train()
    task_net.eval()  # freeze task-net dropout during G update

    # Get task-net predictions, detached from F's graph
    with torch.no_grad():
        y_pred_detached = task_net(x)

    # Compute ES oracle direction (no grad)
    d_star = compute_oracle_direction(
        y_pred_detached, y_true, metric_fn=instance_f1, K=es_K, sigma=es_sigma
    )

    # G_Theta predicts the direction
    v_pred = grad_net(x, y_pred_detached, y_true)

    # MSE loss: G should match the oracle
    loss_G = F.mse_loss(v_pred, d_star)

    optimizer_G.zero_grad()
    loss_G.backward()
    optimizer_G.step()

    # ===== Step B: Train the Task-Net via gradient injection =====
    task_net.train()
    grad_net.eval()

    # Forward pass through task-net (attached to F's computation graph)
    y_pred = task_net(x)  # requires_grad through F_Phi

    # Get G's predicted gradient, detached from G's graph
    with torch.no_grad():
        v_inject = grad_net(x, y_pred.detach(), y_true)

    # BCE loss for basic stability
    bce_loss = F.binary_cross_entropy(y_pred, y_true, reduction="mean")

    # Compute BCE gradient w.r.t. y_pred
    bce_grad = torch.autograd.grad(
        bce_loss, y_pred, retain_graph=True, create_graph=False
    )[0]

    # Combine: minimize BCE (positive grad) and maximize structure (negative grad)
    total_gradient_at_y = lambda_2 * bce_grad - lambda_1 * v_inject

    # Inject the custom gradient into the task-net's backward pass
    optimizer_F.zero_grad()
    torch.autograd.backward(tensors=[y_pred], grad_tensors=[total_gradient_at_y])
    optimizer_F.step()

    return {
        "loss_G": loss_G.item(),
        "bce": bce_loss.item(),
        "d_star_norm": d_star.norm(dim=-1).mean().item(),
        "v_pred_norm": v_pred.detach().norm(dim=-1).mean().item(),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(args):
    device = torch.device(args.device)

    # Data
    loaders = make_dataloaders(
        train_pattern=args.train_data,
        val_pattern=args.val_data,
        test_pattern=args.test_data,
        num_labels=args.num_labels,
        batch_size=args.batch_size,
    )

    # Models
    task_net = TaskNet(args.input_dim, args.num_labels, args.hidden_dim).to(device)
    grad_net = GradientNet(
        args.input_dim, args.num_labels, args.grad_hidden_dim, args.grad_num_layers,
    ).to(device)

    task_params = sum(p.numel() for p in task_net.parameters())
    grad_params = sum(p.numel() for p in grad_net.parameters())
    print(f"TaskNet parameters: {task_params:,}")
    print(f"GradientNet parameters: {grad_params:,}")

    # Optimizers
    optimizer_F = torch.optim.AdamW(
        task_net.parameters(), lr=args.lr_task, weight_decay=args.weight_decay,
    )
    optimizer_G = torch.optim.AdamW(
        grad_net.parameters(), lr=args.lr_grad, weight_decay=args.weight_decay,
    )

    # LR scheduler for task-net
    scheduler_F = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_F, T_max=args.epochs, eta_min=1e-5,
    )

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "epoch_results.txt")
    config_path = os.path.join(args.output_dir, "config.json")

    # Save config
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    # Training
    best_val_f1 = 0.0
    best_epoch = -1
    start_time = time.time()

    with open(log_path, "w", encoding="utf-8") as log_f:
        log_f.write("=" * 100 + "\n")
        log_f.write("Direct Gradient Networks (DGN) - Bibtex Training Results\n")
        log_f.write("=" * 100 + "\n\n")

        header = (
            f"{'Epoch':>5}  {'Loss_G':>10}  {'BCE':>10}  "
            f"{'|d*|':>10}  {'|v|':>10}  "
            f"{'Val BCE':>10}  {'Val F1':>10}  "
            f"{'Best Ep':>7}  {'Best F1':>10}  {'Time':>10}"
        )
        log_f.write(header + "\n")
        log_f.write("-" * len(header) + "\n")
        log_f.flush()

        print(header)
        print("-" * len(header))

        for epoch in range(args.epochs):
            epoch_start = time.time()

            # --- Training epoch ---
            task_net.train()
            grad_net.train()
            epoch_metrics = {"loss_G": 0, "bce": 0, "d_star_norm": 0, "v_pred_norm": 0}
            n_batches = 0

            for x, y_true in loaders["train"]:
                x, y_true = x.to(device), y_true.to(device)

                step_metrics = train_step(
                    x, y_true, task_net, grad_net,
                    optimizer_F, optimizer_G,
                    lambda_1=args.lambda_1,
                    lambda_2=args.lambda_2,
                    es_K=args.es_K,
                    es_sigma=args.es_sigma,
                )
                for k in epoch_metrics:
                    epoch_metrics[k] += step_metrics[k]
                n_batches += 1

            for k in epoch_metrics:
                epoch_metrics[k] /= n_batches

            # --- Validation ---
            val_metrics = evaluate(task_net, loaders["val"], device)
            scheduler_F.step()

            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_epoch = epoch
                torch.save({
                    "task_net": task_net.state_dict(),
                    "grad_net": grad_net.state_dict(),
                    "epoch": epoch,
                    "val_f1": best_val_f1,
                }, os.path.join(args.output_dir, "best.pt"))

            elapsed = timedelta(seconds=int(time.time() - epoch_start))

            line = (
                f"{epoch:>5}  {epoch_metrics['loss_G']:>10.6f}  {epoch_metrics['bce']:>10.6f}  "
                f"{epoch_metrics['d_star_norm']:>10.4f}  {epoch_metrics['v_pred_norm']:>10.4f}  "
                f"{val_metrics['bce']:>10.6f}  {val_metrics['f1']:>10.6f}  "
                f"{best_epoch:>7}  {best_val_f1:>10.6f}  {str(elapsed):>10}"
            )
            print(line)
            log_f.write(line + "\n")
            log_f.flush()

            # Save per-epoch metrics JSON
            epoch_data = {
                "epoch": epoch,
                "training_loss_G": epoch_metrics["loss_G"],
                "training_bce": epoch_metrics["bce"],
                "training_d_star_norm": epoch_metrics["d_star_norm"],
                "training_v_pred_norm": epoch_metrics["v_pred_norm"],
                "validation_bce": val_metrics["bce"],
                "validation_f1": val_metrics["f1"],
                "best_epoch": best_epoch,
                "best_validation_f1": best_val_f1,
                "lr_task": optimizer_F.param_groups[0]["lr"],
            }
            with open(os.path.join(args.output_dir, f"metrics_epoch_{epoch}.json"), "w") as mf:
                json.dump(epoch_data, mf, indent=2)

        # --- Final summary ---
        total_time = timedelta(seconds=int(time.time() - start_time))

        log_f.write("\n" + "=" * 100 + "\n")
        log_f.write(f"Training complete in {total_time}\n")
        log_f.write(f"Best validation F1: {best_val_f1:.6f} at epoch {best_epoch}\n")

        # --- Test evaluation ---
        if "test" in loaders:
            checkpoint = torch.load(os.path.join(args.output_dir, "best.pt"))
            task_net.load_state_dict(checkpoint["task_net"])
            test_metrics = evaluate(task_net, loaders["test"], device)
            log_f.write(f"Test BCE: {test_metrics['bce']:.6f}\n")
            log_f.write(f"Test F1: {test_metrics['f1']:.6f}\n")
            print(f"\nTest F1: {test_metrics['f1']:.6f} (best model from epoch {best_epoch})")

        log_f.write("=" * 100 + "\n")

    print(f"\nResults written to {log_path}")


def main():
    parser = argparse.ArgumentParser(description="DGN Training")

    # Data
    parser.add_argument("--train-data", default="./data/bibtex_stratified10folds_meka/Bibtex-fold@(1|2|3|4|5|6).arff")
    parser.add_argument("--val-data", default="./data/bibtex_stratified10folds_meka/Bibtex-fold@(7|8).arff")
    parser.add_argument("--test-data", default="./data/bibtex_stratified10folds_meka/Bibtex-fold@(9|10).arff")
    parser.add_argument("--num-labels", type=int, default=159)
    parser.add_argument("--input-dim", type=int, default=1836)
    parser.add_argument("--batch-size", type=int, default=32)

    # Task-Net
    parser.add_argument("--hidden-dim", type=int, default=400)
    parser.add_argument("--lr-task", type=float, default=1e-3)

    # Gradient-Net
    parser.add_argument("--grad-hidden-dim", type=int, default=512)
    parser.add_argument("--grad-num-layers", type=int, default=3)
    parser.add_argument("--lr-grad", type=float, default=1e-3)

    # DGN hyperparameters
    parser.add_argument("--lambda-1", type=float, default=1.0, help="Weight for structural gradient injection")
    parser.add_argument("--lambda-2", type=float, default=1.0, help="Weight for BCE gradient")
    parser.add_argument("--es-K", type=int, default=8, help="Number of ES perturbation samples")
    parser.add_argument("--es-sigma", type=float, default=0.1, help="ES perturbation scale")

    # Training
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="output/dgn_run")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
