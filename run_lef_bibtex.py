"""
Train Label Energy Fields (LEF) on Bibtex multi-label classification.

Trains both LEF-ODE and LEF-DEQ variants on the Bibtex dataset (159 labels,
1836 features) and logs epoch-level results to output/lef_run/results.txt.

All output is written to the log file IMMEDIATELY (flushed after every line)
so progress can be monitored in real time via `tail -f output/lef_run/results.txt`.

Usage:
    python run_lef_bibtex.py
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
# Focal Loss: solves class imbalance BY DESIGN
# ---------------------------------------------------------------------------

class FocalLossWithLogits(nn.Module):
    """Sigmoid Focal Loss for multi-label classification.

    Standard BCE treats every prediction equally, so abundant easy negatives
    (98.5% of labels in bibtex) dominate the gradient. Focal loss fixes this
    by multiplying each sample's loss by (1 - p_t)^gamma, where p_t is the
    model's estimated probability for the correct class.

    Effect:
      - Easy negatives (model correctly predicts ~0): p_t ~ 1, so
        (1 - p_t)^gamma -> 0. Their loss contribution vanishes.
      - Hard positives (model wrongly predicts ~0 for a true label): p_t ~ 0,
        so (1 - p_t)^gamma -> 1. Full loss signal preserved.

    This automatically focuses learning on the minority positive class and
    hard-to-classify examples without any label frequency computation.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

    Args:
        gamma: Focusing parameter. Higher = more focus on hard examples.
               gamma=0 reduces to standard BCE. Typical: 1.0-3.0.
        reduction: 'mean' or 'sum'.
    """

    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Numerically stable BCE per element (no reduction)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # p_t = probability assigned to the correct class
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Focal modulation: downweight easy examples
        focal_weight = (1 - p_t) ** self.gamma

        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ---------------------------------------------------------------------------
# Live logger: writes to both stdout and a file, flushing immediately
# ---------------------------------------------------------------------------

class LiveLogger:
    """Dual-output logger that writes to stdout AND a file, flushing instantly."""

    def __init__(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.file = open(filepath, "w", encoding="utf-8")
        self.filepath = filepath

    def log(self, msg=""):
        """Write a line to both stdout and the log file, flush immediately."""
        print(msg, flush=True)
        self.file.write(msg + "\n")
        self.file.flush()
        os.fsync(self.file.fileno())

    def close(self):
        self.file.close()


# ---------------------------------------------------------------------------
# Data loading (direct ARFF reading, no AllenNLP dependency)
# ---------------------------------------------------------------------------

def load_arff_folds(data_dir, folds, num_labels=159, logger=None):
    """Load and concatenate bibtex ARFF folds into numpy arrays."""
    from skmultilearn.dataset import load_from_arff

    all_x, all_y = [], []
    for fold in folds:
        path = os.path.join(data_dir, f"Bibtex-fold{fold}.arff")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing fold file: {path}")
        x, y, _, _ = load_from_arff(
            path, label_count=num_labels, return_attribute_definitions=True
        )
        all_x.append(x.toarray())
        all_y.append(y.toarray())
        if logger:
            logger.log(f"  Loaded fold {fold}: {x.shape[0]} examples")

    X = np.concatenate(all_x, axis=0).astype(np.float32)
    Y = np.concatenate(all_y, axis=0).astype(np.float32)
    return X, Y


def make_dataloader(X, Y, batch_size=32, shuffle=True):
    """Create a PyTorch DataLoader from numpy arrays."""
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def per_instance_f1(y_pred, y_true, threshold=0.5):
    """Per-instance F1 (macro-averaged over instances)."""
    pred_binary = (y_pred >= threshold).float()
    intersection = (pred_binary * y_true).sum(dim=-1)
    pred_count = pred_binary.sum(dim=-1)
    true_count = y_true.sum(dim=-1)
    denom = pred_count + true_count
    f1 = torch.where(denom > 0, 2 * intersection / denom, torch.zeros_like(denom))
    return f1.mean().item()


def micro_f1(y_pred, y_true, threshold=0.5):
    """Micro-averaged F1."""
    pred_binary = (y_pred >= threshold).float()
    tp = (pred_binary * y_true).sum()
    fp = (pred_binary * (1 - y_true)).sum()
    fn = ((1 - pred_binary) * y_true).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1.item()


def macro_f1(y_pred, y_true, threshold=0.5):
    """Macro-averaged F1 (averaged over labels)."""
    pred_binary = (y_pred >= threshold).float()
    tp = (pred_binary * y_true).sum(dim=0)
    fp = (pred_binary * (1 - y_true)).sum(dim=0)
    fn = ((1 - pred_binary) * y_true).sum(dim=0)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1_per_label = 2 * precision * recall / (precision + recall + 1e-8)
    return f1_per_label.mean().item()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, train_loader, optimizer, criterion, device, grad_clip=10.0):
    """Train for one epoch. Returns average loss.

    Models output raw LOGITS. Criterion is BCEWithLogitsLoss.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(x_batch)  # raw logits
        loss = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, data_loader, criterion, device):
    """Evaluate model. Returns (loss, per_instance_f1, micro_f1, macro_f1).

    Models output raw LOGITS. We apply sigmoid for F1 metric computation.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_labels = []

    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        total_loss += loss.item()
        n_batches += 1
        all_preds.append(torch.sigmoid(logits).cpu())  # convert to probs for F1
        all_labels.append(y_batch.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    avg_loss = total_loss / max(n_batches, 1)
    pi_f1 = per_instance_f1(all_preds, all_labels)
    mi_f1 = micro_f1(all_preds, all_labels)
    ma_f1 = macro_f1(all_preds, all_labels)

    return avg_loss, pi_f1, mi_f1, ma_f1


# ---------------------------------------------------------------------------
# Main training procedure -- logs each epoch to file immediately
# ---------------------------------------------------------------------------

def train_model(model_name, model, train_loader, val_loader, test_loader,
                device, logger, num_epochs=20, lr=1e-3,
                weight_decay=1e-5, patience=40, focal_gamma=2.0):
    """Full training loop with focal loss, cosine annealing, and early stopping.

    Args:
        patience: Early stopping patience (epochs without improvement).
        focal_gamma: Focal loss gamma (0 = standard BCE, 2 = strong focusing).
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Cosine annealing with warm restarts: T_0=50 epochs per cycle, decays by 2x
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6,
    )

    criterion = FocalLossWithLogits(gamma=focal_gamma)

    results = []
    best_val_f1 = 0.0
    best_epoch = -1
    best_state = None

    logger.log("")
    logger.log("=" * 100)
    logger.log(f"Training {model_name}")
    logger.log(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.log(f"  Device: {device}")
    logger.log(f"  Loss: FocalLoss(gamma={focal_gamma})")
    logger.log(f"  Scheduler: CosineAnnealingWarmRestarts(T_0=50, T_mult=2)")
    logger.log(f"  Early stopping patience: {patience}")
    logger.log("=" * 100)

    header = (
        f"{'Ep':>3}  {'Train Loss':>11}  {'Val Loss':>11}  "
        f"{'Val F1(pi)':>11}  {'Val F1(mic)':>11}  {'Val F1(mac)':>11}  "
        f"{'BestEp':>6}  {'BestF1':>8}  {'LR':>10}  {'Time':>6}"
    )
    logger.log(header)
    logger.log("-" * len(header))

    for epoch in range(num_epochs):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_pi_f1, val_mi_f1, val_ma_f1 = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step(epoch)
        elapsed = time.time() - t0

        if val_pi_f1 > best_val_f1:
            best_val_f1 = val_pi_f1
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        cur_lr = optimizer.param_groups[0]["lr"]
        epoch_result = {
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_pi_f1": val_pi_f1, "val_micro_f1": val_mi_f1,
            "val_macro_f1": val_ma_f1, "best_epoch": best_epoch,
            "best_val_f1": best_val_f1, "lr": cur_lr, "time_sec": elapsed,
        }
        results.append(epoch_result)

        line = (
            f"{epoch:>3}  {train_loss:>11.6f}  {val_loss:>11.6f}  "
            f"{val_pi_f1:>11.4f}  {val_mi_f1:>11.4f}  {val_ma_f1:>11.4f}  "
            f"{best_epoch:>6}  {best_val_f1:>8.4f}  {cur_lr:>10.2e}  {elapsed:>5.1f}s"
        )
        logger.log(line)

        # Early stopping
        if epoch - best_epoch >= patience:
            logger.log(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # Load best model and evaluate on test
    logger.log(f"\nLoading best model from epoch {best_epoch}...")
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    # Use unweighted BCE for test evaluation (fair comparison)
    test_criterion = nn.BCEWithLogitsLoss()
    test_loss, test_pi_f1, test_mi_f1, test_ma_f1 = evaluate(
        model, test_loader, test_criterion, device
    )

    logger.log(f"TEST RESULTS ({model_name}):")
    logger.log(f"  Loss:              {test_loss:.6f}")
    logger.log(f"  Per-instance F1:   {test_pi_f1:.4f}")
    logger.log(f"  Micro F1:          {test_mi_f1:.4f}")
    logger.log(f"  Macro F1:          {test_ma_f1:.4f}")

    test_results = {
        "test_loss": test_loss, "test_pi_f1": test_pi_f1,
        "test_micro_f1": test_mi_f1, "test_macro_f1": test_ma_f1,
        "best_epoch": best_epoch, "best_val_f1": best_val_f1,
    }
    return results, test_results


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def smoke_test(device, logger):
    """Quick forward/backward on random data for both models."""
    from lef_models import LEF_ODE, LEF_DEQ

    logger.log("")
    logger.log("=" * 60)
    logger.log("SMOKE TEST: Verifying gradient flow on random data")
    logger.log("=" * 60)

    input_dim, num_labels = 64, 10
    batch_size = 4

    x = torch.randn(batch_size, input_dim, device=device)
    y_true = torch.randint(0, 2, (batch_size, num_labels), device=device).float()
    criterion = nn.BCEWithLogitsLoss()

    for name, ModelClass, kwargs in [
        ("LEF-ODE", LEF_ODE, {"method": "euler", "rtol": 1e-2, "atol": 1e-2}),
        ("LEF-DEQ", LEF_DEQ, {"broyden_max_iter": 15, "broyden_tol": 1e-3}),
    ]:
        logger.log(f"\n--- {name} ---")
        model = ModelClass(
            input_dim=input_dim, num_labels=num_labels,
            hidden_dim=64, embed_dim=16, dropout=0.0, **kwargs,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for step in range(3):
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y_true)
            loss.backward()
            optimizer.step()

            probs = torch.sigmoid(logits)
            has_nan = torch.isnan(logits).any().item()
            has_inf = torch.isinf(logits).any().item()
            n_grads = sum(
                1 for p in model.parameters()
                if p.grad is not None and p.grad.abs().sum() > 0
            )
            n_params = sum(1 for p in model.parameters() if p.requires_grad)

            logger.log(
                f"  Step {step}: loss={loss.item():.4f}, "
                f"prob range=[{probs.min().item():.3f}, {probs.max().item():.3f}], "
                f"grads={n_grads}/{n_params}, nan={has_nan}, inf={has_inf}"
            )

            if has_nan or has_inf:
                logger.log(f"  WARNING: {name} produced NaN/Inf!")
                return False

    logger.log("\nSmoke test PASSED -- both models compile, run, and flow gradients.\n")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    DATA_DIR = os.path.join("data", "bibtex_stratified10folds_meka")
    OUTPUT_DIR = os.path.join("output", "lef_run")
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "results.txt")

    NUM_LABELS = 159
    INPUT_DIM = 1836
    HIDDEN_DIM = 128
    EMBED_DIM = 24
    DROPOUT = 0.4
    BATCH_SIZE = 32
    NUM_EPOCHS = 300
    LR = 5e-4
    WEIGHT_DECAY = 1e-3

    ODE_METHOD = "euler"
    ODE_RTOL = 1e-2
    ODE_ATOL = 1e-3

    DEQ_MAX_ITER = 25
    DEQ_TOL = 1e-4

    FOCAL_GAMMA = 2.0  # focal loss focusing parameter
    PATIENCE = 40       # early stopping patience

    # ---- Create live logger (writes to file + stdout, flushes every line) ----
    logger = LiveLogger(OUTPUT_FILE)

    logger.log("=" * 100)
    logger.log("Label Energy Fields (LEF) -- Bibtex Training Results")
    logger.log(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log("=" * 100)
    logger.log("")
    logger.log("CONFIGURATION")
    logger.log("-" * 40)
    for k, v in [
        ("dataset", "bibtex"), ("num_labels", NUM_LABELS),
        ("input_dim", INPUT_DIM), ("hidden_dim", HIDDEN_DIM),
        ("embed_dim", EMBED_DIM), ("dropout", DROPOUT),
        ("batch_size", BATCH_SIZE), ("num_epochs", NUM_EPOCHS),
        ("lr", LR), ("weight_decay", WEIGHT_DECAY),
        ("ode_method", ODE_METHOD), ("ode_rtol", ODE_RTOL),
        ("ode_atol", ODE_ATOL), ("deq_max_iter", DEQ_MAX_ITER),
        ("deq_tol", DEQ_TOL), ("focal_gamma", FOCAL_GAMMA),
        ("patience", PATIENCE),
    ]:
        logger.log(f"  {k}: {v}")

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"\nUsing device: {device}")
    if device.type == "cuda":
        logger.log(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ---- Smoke test ----
    if not smoke_test(device, logger):
        logger.log("Smoke test FAILED. Aborting.")
        logger.close()
        sys.exit(1)

    # ---- Load data ----
    logger.log("Loading bibtex data...")
    logger.log("  Train folds: 1-6")
    X_train, Y_train = load_arff_folds(DATA_DIR, [1, 2, 3, 4, 5, 6], NUM_LABELS, logger)
    logger.log("  Val folds: 7-8")
    X_val, Y_val = load_arff_folds(DATA_DIR, [7, 8], NUM_LABELS, logger)
    logger.log("  Test folds: 9-10")
    X_test, Y_test = load_arff_folds(DATA_DIR, [9, 10], NUM_LABELS, logger)

    logger.log(f"\n  Train: {X_train.shape[0]} examples, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    logger.log(f"  Features: {X_train.shape[1]}, Labels: {Y_train.shape[1]}")
    logger.log(f"  Label density: {Y_train.mean():.4f}")

    train_loader = make_dataloader(X_train, Y_train, BATCH_SIZE, shuffle=True)
    val_loader = make_dataloader(X_val, Y_val, BATCH_SIZE, shuffle=False)
    test_loader = make_dataloader(X_test, Y_test, BATCH_SIZE, shuffle=False)

    from lef_models import LEF_ODE, LEF_DEQ

    # ---- Train LEF-ODE ----
    ode_model = LEF_ODE(
        input_dim=INPUT_DIM, num_labels=NUM_LABELS, hidden_dim=HIDDEN_DIM,
        embed_dim=EMBED_DIM, dropout=DROPOUT,
        rtol=ODE_RTOL, atol=ODE_ATOL, method=ODE_METHOD,
    )
    ode_results, ode_test = train_model(
        "LEF-ODE", ode_model, train_loader, val_loader, test_loader,
        device, logger, num_epochs=NUM_EPOCHS, lr=LR,
        weight_decay=WEIGHT_DECAY, patience=PATIENCE, focal_gamma=FOCAL_GAMMA,
    )

    # ---- Train LEF-DEQ ----
    deq_model = LEF_DEQ(
        input_dim=INPUT_DIM, num_labels=NUM_LABELS, hidden_dim=HIDDEN_DIM,
        embed_dim=EMBED_DIM, dropout=DROPOUT,
        broyden_max_iter=DEQ_MAX_ITER, broyden_tol=DEQ_TOL,
    )
    deq_results, deq_test = train_model(
        "LEF-DEQ", deq_model, train_loader, val_loader, test_loader,
        device, logger, num_epochs=NUM_EPOCHS, lr=LR,
        weight_decay=WEIGHT_DECAY, patience=PATIENCE, focal_gamma=FOCAL_GAMMA,
    )

    # ---- Final comparison table ----
    logger.log("")
    logger.log("=" * 100)
    logger.log("COMPARISON SUMMARY")
    logger.log("=" * 100)
    logger.log("")
    logger.log(f"{'Metric':<25} {'LEF-ODE':>12} {'LEF-DEQ':>12}")
    logger.log("-" * 50)
    logger.log(f"{'Best Val F1 (pi)':.<25} {ode_test['best_val_f1']:>12.4f} {deq_test['best_val_f1']:>12.4f}")
    logger.log(f"{'Best Epoch':.<25} {ode_test['best_epoch']:>12} {deq_test['best_epoch']:>12}")
    logger.log(f"{'Test Loss':.<25} {ode_test['test_loss']:>12.6f} {deq_test['test_loss']:>12.6f}")
    logger.log(f"{'Test F1 (per-instance)':.<25} {ode_test['test_pi_f1']:>12.4f} {deq_test['test_pi_f1']:>12.4f}")
    logger.log(f"{'Test F1 (micro)':.<25} {ode_test['test_micro_f1']:>12.4f} {deq_test['test_micro_f1']:>12.4f}")
    logger.log(f"{'Test F1 (macro)':.<25} {ode_test['test_macro_f1']:>12.4f} {deq_test['test_macro_f1']:>12.4f}")
    logger.log("=" * 100)

    logger.log(f"\nAll done. Results at: {OUTPUT_FILE}")
    logger.close()


if __name__ == "__main__":
    main()
