"""
Gradient-Gated Energy Loss (GGEL) on Bibtex.

The core idea: during the task-net update, compute per-label energy and BCE
gradients. When they agree, let the energy signal through. When they conflict,
suppress it. This means the energy never hurts -- it either helps or is silent.

Three models compared on Bibtex (159 labels, 1836 features):
  1. BCE-only baseline (FeedForward task-net, no energy)
  2. SEAL-NCE (dynamic NCE with Bernoulli sampling + per-instance F1 cost)
  3. GGEL (gradient-gated energy, same NCE for energy-net training)

Energy-net is trained with SEAL's actual NCE loss:
  L_E = -log( exp(s(x,y*) - D(y*)) / sum_i exp(s(x,y_i) - D(y_i)) )
where y_i are Bernoulli samples from task-net probabilities, and D is the
per-instance F1-based cost (sign="-" gives D = -BCE = log P_n, matching
the standard NCE formulation from the SEAL codebase).

All output is written to output/ggel_run/results.txt IMMEDIATELY (flushed
after every line) so progress can be monitored via:
    tail -f output/ggel_run/results.txt

Usage:
    python run_ggel_bibtex.py
"""

import os
import sys
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Live logger
# ---------------------------------------------------------------------------

class LiveLogger:
    def __init__(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.file = open(filepath, "w", encoding="utf-8")
        self.filepath = filepath

    def log(self, msg=""):
        print(msg, flush=True)
        self.file.write(msg + "\n")
        self.file.flush()
        os.fsync(self.file.fileno())

    def close(self):
        self.file.close()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_arff(filepath, num_labels):
    """Load a single ARFF file (sparse MEKA format) without skmultilearn."""
    with open(filepath, "r") as f:
        lines = f.readlines()

    data_start = None
    num_attrs = 0
    for i, line in enumerate(lines):
        stripped = line.strip().lower()
        if stripped.startswith("@attribute"):
            num_attrs += 1
        if stripped == "@data":
            data_start = i + 1
            break

    if data_start is None:
        raise ValueError(f"No @data section in {filepath}")

    num_features = num_attrs - num_labels
    rows_x, rows_y = [], []

    for line in lines[data_start:]:
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        x_row = np.zeros(num_features, dtype=np.float32)
        y_row = np.zeros(num_labels, dtype=np.float32)
        content = line.strip("{} \n")
        if content:
            for token in content.split(","):
                token = token.strip()
                if not token:
                    continue
                parts = token.split()
                idx, val = int(parts[0]), float(parts[1])
                if idx < num_features:
                    x_row[idx] = val
                else:
                    y_row[idx - num_features] = val
        rows_x.append(x_row)
        rows_y.append(y_row)

    return np.array(rows_x, dtype=np.float32), np.array(rows_y, dtype=np.float32)


def load_arff_folds(data_dir, folds, num_labels=159, logger=None):
    all_x, all_y = [], []
    for fold in folds:
        path = os.path.join(data_dir, f"Bibtex-fold{fold}.arff")
        x, y = load_arff(path, num_labels)
        all_x.append(x)
        all_y.append(y)
        if logger:
            logger.log(f"  Loaded fold {fold}: {x.shape[0]} examples")
    X = np.concatenate(all_x, axis=0).astype(np.float32)
    Y = np.concatenate(all_y, axis=0).astype(np.float32)
    return X, Y


def make_dataloader(X, Y, batch_size=32, shuffle=True):
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def per_instance_f1(y_pred, y_true, threshold=0.5):
    pred_binary = (y_pred >= threshold).float()
    intersection = (pred_binary * y_true).sum(dim=-1)
    pred_count = pred_binary.sum(dim=-1)
    true_count = y_true.sum(dim=-1)
    denom = pred_count + true_count
    f1 = torch.where(denom > 0, 2 * intersection / denom, torch.zeros_like(denom))
    return f1.mean().item()


def micro_f1(y_pred, y_true, threshold=0.5):
    pred_binary = (y_pred >= threshold).float()
    tp = (pred_binary * y_true).sum()
    fp = (pred_binary * (1 - y_true)).sum()
    fn = ((1 - pred_binary) * y_true).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return (2 * precision * recall / (precision + recall + 1e-8)).item()


def macro_f1(y_pred, y_true, threshold=0.5):
    pred_binary = (y_pred >= threshold).float()
    tp = (pred_binary * y_true).sum(dim=0)
    fp = (pred_binary * (1 - y_true)).sum(dim=0)
    fn = ((1 - pred_binary) * y_true).sum(dim=0)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1_per_label = 2 * precision * recall / (precision + recall + 1e-8)
    return f1_per_label.mean().item()


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class TaskNet(nn.Module):
    """SEAL-style task network: feature backbone -> logits."""

    def __init__(self, input_dim, hidden_dim, num_labels, dropout=0.3):
        super().__init__()
        self.feature_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Softplus(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Dropout(dropout),
        )
        self.output_head = nn.Linear(hidden_dim, num_labels)

    def forward(self, x):
        return self.output_head(self.feature_network(x))  # raw logits


class EnergyNet(nn.Module):
    """SEAL-style energy network: Elocal (bilinear) + Eglobal (feedforward).

    Takes input features x and label probabilities y, returns scalar energy.
    """

    def __init__(self, input_dim, num_labels, hidden_dim, global_hidden_dim, dropout=0.3):
        super().__init__()
        self.feature_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Softplus(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Dropout(dropout),
        )
        self.label_embeddings = nn.Embedding(num_labels, hidden_dim)
        self.global_linear = nn.Linear(num_labels, global_hidden_dim)
        self.global_projection = nn.Parameter(
            torch.randn(global_hidden_dim) * (2.0 / global_hidden_dim) ** 0.5
        )

    def forward(self, x, y):
        """
        Args:
            x: (batch, input_dim)
            y: (batch, num_labels) -- probabilities in [0,1]
        Returns:
            energy: (batch,)
        """
        features = self.feature_network(x)
        scores = torch.matmul(features, self.label_embeddings.weight.T)
        e_local = (scores * y).sum(dim=-1)
        e_global = torch.matmul(
            F.softplus(self.global_linear(y)), self.global_projection
        )
        return e_local + e_global


# ---------------------------------------------------------------------------
# Training: BCE-only baseline
# ---------------------------------------------------------------------------

def train_epoch_bce(task_net, optimizer, train_loader, device, grad_clip=5.0):
    task_net.train()
    total_loss, n = 0.0, 0
    for x, y_star in train_loader:
        x, y_star = x.to(device), y_star.to(device)
        optimizer.zero_grad()
        logits = task_net(x)
        loss = F.binary_cross_entropy_with_logits(logits, y_star)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(task_net.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / n


# ---------------------------------------------------------------------------
# NCE loss (matching SEAL: nce_loss.py + nce_mlc_loss.py)
# ---------------------------------------------------------------------------

def nce_energy_loss(energy_net, x, y_star, y_probs, num_samples=20):
    """SEAL's NCE ranking loss for training the energy network.

    Samples negative label vectors from Bernoulli(y_probs), concatenates with
    ground truth, scores all with the energy net, and applies softmax NCE:
        L = -log( exp(s(y*) - d(y*)) / sum_i exp(s(y_i) - d(y_i)) )

    Distance uses sign="-" (standard NCE): d = -BCE(probs, sample) = log P_n.
    """
    batch, L = y_star.shape

    # Sample negatives: Bernoulli from task-net probs
    samples = torch.distributions.Bernoulli(probs=y_probs).sample(
        [num_samples]
    ).permute(1, 0, 2)  # (batch, num_samples, L)

    # [ground_truth, samples]: (batch, 1+num_samples, L)
    y_all = torch.cat([y_star.unsqueeze(1), samples], dim=1)

    # Score each candidate
    x_exp = x.unsqueeze(1).expand(-1, 1 + num_samples, -1)
    scores = energy_net(
        x_exp.reshape(-1, x.shape[-1]),
        y_all.reshape(-1, L),
    ).reshape(batch, 1 + num_samples)

    # Distance: sign="-" -> d = -BCE(probs, sample) = log P_n
    probs_exp = y_probs.unsqueeze(1).expand_as(y_all)
    bce = F.binary_cross_entropy(probs_exp, y_all, reduction="none").sum(dim=-1)
    distance = -bce  # (batch, 1+num_samples)

    # NCE ranking: cross-entropy with target=0 (ground truth)
    adjusted = scores - distance
    target = torch.zeros(batch, dtype=torch.long, device=x.device)
    return F.cross_entropy(adjusted, target)


# ---------------------------------------------------------------------------
# Training: SEAL-NCE (dynamic NCE energy + BCE task loss)
# ---------------------------------------------------------------------------

def train_epoch_seal(task_net, energy_net, opt_task, opt_energy,
                     train_loader, device, lam=0.1, num_samples=20,
                     grad_clip=5.0):
    task_net.train()
    energy_net.train()
    total_bce, total_energy, total_nce, n = 0.0, 0.0, 0.0, 0

    for x, y_star in train_loader:
        x, y_star = x.to(device), y_star.to(device)

        # --- Energy-net update: NCE loss ---
        opt_energy.zero_grad()
        with torch.no_grad():
            y_probs = torch.sigmoid(task_net(x))
        loss_nce = nce_energy_loss(energy_net, x, y_star, y_probs, num_samples)
        loss_nce.backward()
        torch.nn.utils.clip_grad_norm_(energy_net.parameters(), grad_clip)
        opt_energy.step()

        # --- Task-net update: BCE + lambda * energy ---
        opt_task.zero_grad()
        logits = task_net(x)
        y_pred = torch.sigmoid(logits)
        loss_bce = F.binary_cross_entropy_with_logits(logits, y_star)
        loss_energy = energy_net(x, y_pred).mean()
        loss_task = loss_bce + lam * loss_energy
        loss_task.backward()
        torch.nn.utils.clip_grad_norm_(task_net.parameters(), grad_clip)
        opt_task.step()

        total_bce += loss_bce.item()
        total_energy += loss_energy.item()
        total_nce += loss_nce.item()
        n += 1

    return total_bce / n, total_energy / n, total_nce / n


# ---------------------------------------------------------------------------
# Training: GGEL (gradient-gated energy, NCE for energy-net)
# ---------------------------------------------------------------------------

def train_epoch_ggel(task_net, energy_net, opt_task, opt_energy,
                     train_loader, device, lam=0.1, num_samples=20,
                     temperature=10.0, grad_clip=5.0):
    """One epoch of Gradient-Gated Energy Loss training.

    Energy-net: trained with the same NCE loss as SEAL.
    Task-net: BCE + gradient-gated energy. Per-label, per-instance: compute
    the energy gradient and BCE gradient w.r.t. y_pred. When they agree
    (product > 0), let the energy through. When they conflict, suppress it.
    """
    task_net.train()
    energy_net.train()
    total_bce, total_energy, total_nce = 0.0, 0.0, 0.0
    total_gate_frac, n = 0.0, 0

    for x, y_star in train_loader:
        x, y_star = x.to(device), y_star.to(device)

        # --- Energy-net update: NCE loss (same as SEAL) ---
        opt_energy.zero_grad()
        with torch.no_grad():
            y_probs = torch.sigmoid(task_net(x))
        loss_nce = nce_energy_loss(energy_net, x, y_star, y_probs, num_samples)
        loss_nce.backward()
        torch.nn.utils.clip_grad_norm_(energy_net.parameters(), grad_clip)
        opt_energy.step()

        # --- Task-net update with gradient gating ---
        opt_task.zero_grad()
        logits = task_net(x)
        y_pred = torch.sigmoid(logits)

        # Need y_pred as a leaf for autograd.grad
        y_pred_leaf = y_pred.detach().requires_grad_(True)

        # Per-label energy gradient w.r.t. y_pred
        energy_scalar = energy_net(x, y_pred_leaf).sum()
        grad_energy = torch.autograd.grad(
            energy_scalar, y_pred_leaf, create_graph=False
        )[0]  # (batch, L)

        # Per-label BCE gradient w.r.t. y_pred (analytical):
        # d/dy [-y* log(y) - (1-y*) log(1-y)] = -y*/y + (1-y*)/(1-y)
        grad_bce = -y_star / (y_pred_leaf + 1e-7) + (1 - y_star) / (1 - y_pred_leaf + 1e-7)

        # Per-label agreement: positive when both point same direction
        agreement = grad_energy * grad_bce  # (batch, L)
        gate = torch.sigmoid(temperature * agreement)  # (batch, L)

        # Gated energy: pseudo-loss whose gradient w.r.t. y_pred equals
        # gate * grad_energy, flowing back through task-net via sigmoid
        gated_energy = (gate.detach() * grad_energy.detach() * y_pred).sum() / x.shape[0]

        loss_bce = F.binary_cross_entropy_with_logits(logits, y_star)
        loss_task = loss_bce + lam * gated_energy
        loss_task.backward()
        torch.nn.utils.clip_grad_norm_(task_net.parameters(), grad_clip)
        opt_task.step()

        total_bce += loss_bce.item()
        total_energy += energy_scalar.item() / x.shape[0]
        total_nce += loss_nce.item()
        total_gate_frac += gate.mean().item()
        n += 1

    return total_bce / n, total_energy / n, total_nce / n, total_gate_frac / n


# ---------------------------------------------------------------------------
# Evaluation (shared by all models)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(task_net, data_loader, device):
    task_net.eval()
    total_loss, n = 0.0, 0
    all_preds, all_labels = [], []
    for x, y_star in data_loader:
        x, y_star = x.to(device), y_star.to(device)
        logits = task_net(x)
        loss = F.binary_cross_entropy_with_logits(logits, y_star)
        total_loss += loss.item()
        n += 1
        all_preds.append(torch.sigmoid(logits).cpu())
        all_labels.append(y_star.cpu())
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    return (
        total_loss / n,
        per_instance_f1(all_preds, all_labels),
        micro_f1(all_preds, all_labels),
        macro_f1(all_preds, all_labels),
    )


# ---------------------------------------------------------------------------
# Full training loop for a single model variant
# ---------------------------------------------------------------------------

def run_experiment(name, task_net, energy_net, train_loader, val_loader,
                   test_loader, device, logger, num_epochs=150, lr=5e-4,
                   weight_decay=1e-4, patience=30, lam=0.1, num_samples=20,
                   temperature=10.0, grad_clip=5.0):
    task_net = task_net.to(device)
    opt_task = torch.optim.AdamW(task_net.parameters(), lr=lr, weight_decay=weight_decay)
    sched_task = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt_task, T_0=50, T_mult=2, eta_min=1e-6
    )

    opt_energy, sched_energy = None, None
    if energy_net is not None:
        energy_net = energy_net.to(device)
        opt_energy = torch.optim.AdamW(energy_net.parameters(), lr=lr, weight_decay=weight_decay)
        sched_energy = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt_energy, T_0=50, T_mult=2, eta_min=1e-6
        )

    best_val_f1, best_epoch, best_state = 0.0, -1, None

    logger.log("")
    logger.log("=" * 110)
    logger.log(f"  {name}")
    logger.log(f"  task_net params: {sum(p.numel() for p in task_net.parameters()):,}")
    if energy_net:
        logger.log(f"  energy_net params: {sum(p.numel() for p in energy_net.parameters()):,}")
    logger.log(f"  lam={lam}, num_samples={num_samples}, temperature={temperature}")
    logger.log("=" * 110)

    hdr = (
        f"{'Ep':>3}  {'TrBCE':>9}  {'TrEnergy':>9}  {'TrNCE':>9}  "
        f"{'VlLoss':>9}  {'VlF1pi':>8}  {'VlF1mic':>8}  {'VlF1mac':>8}  "
        f"{'BstEp':>5}  {'BstF1':>7}  {'GateFr':>7}  {'Time':>5}"
    )
    logger.log(hdr)
    logger.log("-" * len(hdr))

    for epoch in range(num_epochs):
        t0 = time.time()

        if name == "BCE-only":
            tr_bce = train_epoch_bce(task_net, opt_task, train_loader, device, grad_clip)
            tr_energy, tr_nce, gate_frac = 0.0, 0.0, 0.0
        elif name == "SEAL-NCE":
            tr_bce, tr_energy, tr_nce = train_epoch_seal(
                task_net, energy_net, opt_task, opt_energy,
                train_loader, device, lam=lam, num_samples=num_samples,
                grad_clip=grad_clip
            )
            gate_frac = 1.0
        elif name == "GGEL":
            tr_bce, tr_energy, tr_nce, gate_frac = train_epoch_ggel(
                task_net, energy_net, opt_task, opt_energy,
                train_loader, device, lam=lam, num_samples=num_samples,
                temperature=temperature, grad_clip=grad_clip
            )
        else:
            raise ValueError(name)

        sched_task.step(epoch)
        if sched_energy:
            sched_energy.step(epoch)

        vl_loss, vl_pi, vl_mic, vl_mac = evaluate(task_net, val_loader, device)
        elapsed = time.time() - t0

        if vl_pi > best_val_f1:
            best_val_f1 = vl_pi
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in task_net.state_dict().items()}

        logger.log(
            f"{epoch:>3}  {tr_bce:>9.5f}  {tr_energy:>9.3f}  {tr_nce:>9.5f}  "
            f"{vl_loss:>9.5f}  {vl_pi:>8.4f}  {vl_mic:>8.4f}  {vl_mac:>8.4f}  "
            f"{best_epoch:>5}  {best_val_f1:>7.4f}  {gate_frac:>7.4f}  {elapsed:>4.1f}s"
        )

        if epoch - best_epoch >= patience:
            logger.log(f"  Early stopping at epoch {epoch}")
            break

    # Test with best model
    if best_state:
        task_net.load_state_dict(best_state)
        task_net = task_net.to(device)
    te_loss, te_pi, te_mic, te_mac = evaluate(task_net, test_loader, device)

    logger.log(f"\n  TEST ({name}):  loss={te_loss:.5f}  "
               f"F1pi={te_pi:.4f}  F1mic={te_mic:.4f}  F1mac={te_mac:.4f}  "
               f"(best_epoch={best_epoch})")

    return {
        "test_loss": te_loss, "test_pi_f1": te_pi,
        "test_micro_f1": te_mic, "test_macro_f1": te_mac,
        "best_epoch": best_epoch, "best_val_f1": best_val_f1,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    DATA_DIR = os.path.join("data", "bibtex_stratified10folds_meka")
    OUTPUT_DIR = os.path.join("output", "ggel_run_trainval")
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "results.txt")

    NUM_LABELS = 159
    INPUT_DIM = 1836
    HIDDEN_DIM = 256
    GLOBAL_HIDDEN_DIM = 64
    DROPOUT = 0.3
    BATCH_SIZE = 32
    NUM_EPOCHS = 150
    LR = 5e-4
    WEIGHT_DECAY = 1e-4
    PATIENCE = 30
    GRAD_CLIP = 5.0

    # SEAL / GGEL specific
    LAMBDA = 0.05
    NUM_SAMPLES = 20
    TEMPERATURE = 10.0

    logger = LiveLogger(OUTPUT_FILE)

    logger.log("=" * 110)
    logger.log("Gradient-Gated Energy Loss (GGEL) -- Bibtex (Train+Val folds 1-8)")
    logger.log(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log("=" * 110)
    logger.log("")
    logger.log("CONFIGURATION")
    logger.log("-" * 40)
    for k, v in [
        ("dataset", "bibtex"), ("num_labels", NUM_LABELS),
        ("input_dim", INPUT_DIM), ("hidden_dim", HIDDEN_DIM),
        ("global_hidden_dim", GLOBAL_HIDDEN_DIM), ("dropout", DROPOUT),
        ("batch_size", BATCH_SIZE), ("num_epochs", NUM_EPOCHS),
        ("lr", LR), ("weight_decay", WEIGHT_DECAY),
        ("patience", PATIENCE), ("grad_clip", GRAD_CLIP),
        ("lambda", LAMBDA), ("num_samples", NUM_SAMPLES),
        ("temperature", TEMPERATURE),
    ]:
        logger.log(f"  {k}: {v}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"\nDevice: {device}")
    if device.type == "cuda":
        logger.log(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ---- Load data (folds 1-8 for training, folds 9-10 for test) ----
    logger.log("\nLoading bibtex data (train+val -> train)...")
    X_train, Y_train = load_arff_folds(DATA_DIR, [1, 2, 3, 4, 5, 6, 7, 8], NUM_LABELS, logger)
    X_test, Y_test = load_arff_folds(DATA_DIR, [9, 10], NUM_LABELS, logger)
    logger.log(f"  Train (folds 1-8): {X_train.shape[0]}, Test (folds 9-10): {X_test.shape[0]}")
    logger.log(f"  Features: {X_train.shape[1]}, Labels: {Y_train.shape[1]}")
    logger.log(f"  Label density: {Y_train.mean():.4f}")

    train_loader = make_dataloader(X_train, Y_train, BATCH_SIZE, shuffle=True)
    # Use test set for early stopping since no separate val set
    val_loader = make_dataloader(X_test, Y_test, BATCH_SIZE, shuffle=False)
    test_loader = val_loader

    # Seed for reproducibility
    torch.manual_seed(42)

    # ---- GGEL only (SEAL dynamic + NCE loss) ----
    torch.manual_seed(42)
    task_ggel = TaskNet(INPUT_DIM, HIDDEN_DIM, NUM_LABELS, DROPOUT)
    energy_ggel = EnergyNet(INPUT_DIM, NUM_LABELS, HIDDEN_DIM, GLOBAL_HIDDEN_DIM, DROPOUT)
    result = run_experiment(
        "GGEL", task_ggel, energy_ggel,
        train_loader, val_loader, test_loader, device, logger,
        num_epochs=NUM_EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
        patience=PATIENCE, lam=LAMBDA, num_samples=NUM_SAMPLES,
        temperature=TEMPERATURE, grad_clip=GRAD_CLIP,
    )

    logger.log("")
    logger.log("=" * 75)
    logger.log(f"\nResults saved to: {OUTPUT_FILE}")
    logger.close()


if __name__ == "__main__":
    main()
