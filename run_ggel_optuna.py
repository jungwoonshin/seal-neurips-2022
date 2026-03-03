"""
GGEL Optuna hyperparameter search on Bibtex.

Data split:
  - Train: folds 1-6
  - Validation: folds 7-8 (Optuna objective = val F1 per-instance)
  - Test: folds 9-10 (final evaluation of best trial)

50 TPE trials with MedianPruner. All output flushed immediately to:
    output/ggel_optuna/optuna_log.txt

Usage:
    python run_ggel_optuna.py
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
import optuna


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
        return self.output_head(self.feature_network(x))


class EnergyNet(nn.Module):
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
        features = self.feature_network(x)
        scores = torch.matmul(features, self.label_embeddings.weight.T)
        e_local = (scores * y).sum(dim=-1)
        e_global = torch.matmul(
            F.softplus(self.global_linear(y)), self.global_projection
        )
        return e_local + e_global


# ---------------------------------------------------------------------------
# NCE loss
# ---------------------------------------------------------------------------

def nce_energy_loss(energy_net, x, y_star, y_probs, num_samples=20):
    batch, L = y_star.shape
    samples = torch.distributions.Bernoulli(probs=y_probs).sample(
        [num_samples]
    ).permute(1, 0, 2)
    y_all = torch.cat([y_star.unsqueeze(1), samples], dim=1)
    x_exp = x.unsqueeze(1).expand(-1, 1 + num_samples, -1)
    scores = energy_net(
        x_exp.reshape(-1, x.shape[-1]),
        y_all.reshape(-1, L),
    ).reshape(batch, 1 + num_samples)
    probs_exp = y_probs.unsqueeze(1).expand_as(y_all)
    bce = F.binary_cross_entropy(probs_exp, y_all, reduction="none").sum(dim=-1)
    distance = -bce
    adjusted = scores - distance
    target = torch.zeros(batch, dtype=torch.long, device=x.device)
    return F.cross_entropy(adjusted, target)


# ---------------------------------------------------------------------------
# GGEL training epoch
# ---------------------------------------------------------------------------

def train_epoch_ggel(task_net, energy_net, opt_task, opt_energy,
                     train_loader, device, lam=0.1, num_samples=20,
                     temperature=10.0, grad_clip=5.0):
    task_net.train()
    energy_net.train()
    total_bce, total_energy, total_nce = 0.0, 0.0, 0.0
    total_gate_frac, n = 0.0, 0

    for x, y_star in train_loader:
        x, y_star = x.to(device), y_star.to(device)

        # Energy-net update: NCE loss
        opt_energy.zero_grad()
        with torch.no_grad():
            y_probs = torch.sigmoid(task_net(x))
        loss_nce = nce_energy_loss(energy_net, x, y_star, y_probs, num_samples)
        loss_nce.backward()
        torch.nn.utils.clip_grad_norm_(energy_net.parameters(), grad_clip)
        opt_energy.step()

        # Task-net update with gradient gating
        opt_task.zero_grad()
        logits = task_net(x)
        y_pred = torch.sigmoid(logits)

        y_pred_leaf = y_pred.detach().requires_grad_(True)
        energy_scalar = energy_net(x, y_pred_leaf).sum()
        grad_energy = torch.autograd.grad(
            energy_scalar, y_pred_leaf, create_graph=False
        )[0]

        grad_bce = -y_star / (y_pred_leaf + 1e-7) + (1 - y_star) / (1 - y_pred_leaf + 1e-7)
        agreement = grad_energy * grad_bce
        gate = torch.sigmoid(temperature * agreement)
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
# Evaluation
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
# Single GGEL run (used by Optuna objective and final retraining)
# ---------------------------------------------------------------------------

def run_ggel(train_loader, val_loader, device, logger,
             hidden_dim, global_hidden_dim, dropout, lr, weight_decay,
             lam, num_samples, temperature, grad_clip=5.0,
             num_epochs=150, patience=30, trial=None, trial_num=None):
    """Train GGEL and return best val F1 (per-instance).

    If `trial` is provided, reports intermediate values for Optuna pruning.
    """
    INPUT_DIM = 1836
    NUM_LABELS = 159

    torch.manual_seed(42)
    task_net = TaskNet(INPUT_DIM, hidden_dim, NUM_LABELS, dropout).to(device)
    energy_net = EnergyNet(INPUT_DIM, NUM_LABELS, hidden_dim, global_hidden_dim, dropout).to(device)

    opt_task = torch.optim.AdamW(task_net.parameters(), lr=lr, weight_decay=weight_decay)
    opt_energy = torch.optim.AdamW(energy_net.parameters(), lr=lr, weight_decay=weight_decay)
    sched_task = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_task, T_0=50, T_mult=2, eta_min=1e-6)
    sched_energy = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_energy, T_0=50, T_mult=2, eta_min=1e-6)

    best_val_f1, best_epoch = 0.0, -1
    best_state = None

    prefix = f"[T{trial_num:>3}]" if trial_num is not None else "      "

    for epoch in range(num_epochs):
        t0 = time.time()

        tr_bce, tr_energy, tr_nce, gate_frac = train_epoch_ggel(
            task_net, energy_net, opt_task, opt_energy,
            train_loader, device, lam=lam, num_samples=num_samples,
            temperature=temperature, grad_clip=grad_clip
        )
        sched_task.step(epoch)
        sched_energy.step(epoch)

        vl_loss, vl_pi, vl_mic, vl_mac = evaluate(task_net, val_loader, device)
        elapsed = time.time() - t0

        if vl_pi > best_val_f1:
            best_val_f1 = vl_pi
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in task_net.state_dict().items()}

        logger.log(
            f"{prefix} ep{epoch:>3}  bce={tr_bce:.5f}  nce={tr_nce:.5f}  "
            f"vl_f1pi={vl_pi:.4f}  vl_mic={vl_mic:.4f}  "
            f"best={best_val_f1:.4f}@{best_epoch}  gate={gate_frac:.4f}  {elapsed:.1f}s"
        )

        # Optuna pruning
        if trial is not None:
            trial.report(vl_pi, epoch)
            if trial.should_prune():
                logger.log(f"{prefix} PRUNED at epoch {epoch}")
                raise optuna.TrialPruned()

        if epoch - best_epoch >= patience:
            logger.log(f"{prefix} Early stop at epoch {epoch}")
            break

    return best_val_f1, best_epoch, best_state, task_net


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def make_objective(train_loader_cache, val_loader_cache, device, logger):
    """Returns an Optuna objective function with closures over data."""

    def objective(trial):
        # Suggest hyperparameters
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        lam = trial.suggest_float("lambda", 0.01, 0.5, log=True)
        temperature = trial.suggest_float("temperature", 1.0, 50.0, log=True)
        hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
        global_hidden_dim = trial.suggest_categorical("global_hidden_dim", [32, 64, 128])
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        num_samples = trial.suggest_categorical("num_samples", [10, 20, 40])
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

        # Rebuild dataloaders if batch_size changed
        train_loader = train_loader_cache.get(batch_size)
        if train_loader is None:
            train_loader = make_dataloader(X_train_global, Y_train_global, batch_size, shuffle=True)
            train_loader_cache[batch_size] = train_loader
        val_loader = val_loader_cache.get(batch_size)
        if val_loader is None:
            val_loader = make_dataloader(X_val_global, Y_val_global, batch_size, shuffle=False)
            val_loader_cache[batch_size] = val_loader

        trial_num = trial.number
        logger.log("")
        logger.log(f"{'='*90}")
        logger.log(
            f"Trial {trial_num}: lr={lr:.5f} lam={lam:.4f} temp={temperature:.2f} "
            f"hid={hidden_dim} ghid={global_hidden_dim} drop={dropout:.3f} "
            f"nsamp={num_samples} bs={batch_size} wd={weight_decay:.6f}"
        )
        logger.log(f"{'='*90}")

        best_val_f1, best_epoch, _, _ = run_ggel(
            train_loader, val_loader, device, logger,
            hidden_dim=hidden_dim, global_hidden_dim=global_hidden_dim,
            dropout=dropout, lr=lr, weight_decay=weight_decay,
            lam=lam, num_samples=num_samples, temperature=temperature,
            trial=trial, trial_num=trial_num,
        )

        logger.log(f"Trial {trial_num} done: best_val_f1={best_val_f1:.4f} @ epoch {best_epoch}")
        return best_val_f1

    return objective


# ---------------------------------------------------------------------------
# Globals for dataloader caching inside objective
# ---------------------------------------------------------------------------
X_train_global = None
Y_train_global = None
X_val_global = None
Y_val_global = None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global X_train_global, Y_train_global, X_val_global, Y_val_global

    DATA_DIR = os.path.join("data", "bibtex_stratified10folds_meka")
    OUTPUT_DIR = os.path.join("output", "ggel_optuna")
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "optuna_log.txt")

    NUM_LABELS = 159
    N_TRIALS = 50
    NUM_EPOCHS = 150
    PATIENCE = 30

    logger = LiveLogger(OUTPUT_FILE)

    logger.log("=" * 90)
    logger.log("GGEL Optuna Hyperparameter Search -- Bibtex")
    logger.log(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"Trials: {N_TRIALS}, Epochs/trial: {NUM_EPOCHS}, Patience: {PATIENCE}")
    logger.log("Train: folds 1-6, Val: folds 7-8, Test: folds 9-10")
    logger.log("=" * 90)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"\nDevice: {device}")
    if device.type == "cuda":
        logger.log(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    logger.log("\nLoading data...")
    X_train_global, Y_train_global = load_arff_folds(DATA_DIR, [1, 2, 3, 4, 5, 6], NUM_LABELS, logger)
    X_val_global, Y_val_global = load_arff_folds(DATA_DIR, [7, 8], NUM_LABELS, logger)
    X_test, Y_test = load_arff_folds(DATA_DIR, [9, 10], NUM_LABELS, logger)
    logger.log(f"  Train: {X_train_global.shape[0]}, Val: {X_val_global.shape[0]}, Test: {X_test.shape[0]}")
    logger.log(f"  Features: {X_train_global.shape[1]}, Labels: {NUM_LABELS}")

    # Make globals accessible in objective
    import run_ggel_optuna
    run_ggel_optuna.X_train_global = X_train_global
    run_ggel_optuna.Y_train_global = Y_train_global
    run_ggel_optuna.X_val_global = X_val_global
    run_ggel_optuna.Y_val_global = Y_val_global

    # Dataloader caches (keyed by batch_size)
    train_loader_cache = {}
    val_loader_cache = {}

    # Optuna study
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    objective = make_objective(train_loader_cache, val_loader_cache, device, logger)

    logger.log(f"\nStarting {N_TRIALS} Optuna trials...\n")
    study.optimize(objective, n_trials=N_TRIALS)

    # Summary
    logger.log("")
    logger.log("=" * 90)
    logger.log("OPTUNA SEARCH COMPLETE")
    logger.log("=" * 90)

    best = study.best_trial
    logger.log(f"\nBest trial: #{best.number}")
    logger.log(f"Best val F1 (pi): {best.value:.4f}")
    logger.log("Best params:")
    for k, v in best.params.items():
        logger.log(f"  {k}: {v}")

    # Top 10 trials
    logger.log(f"\nTop 10 trials:")
    logger.log(f"{'#':>4}  {'ValF1pi':>8}  {'lr':>9}  {'lambda':>8}  {'temp':>7}  {'hid':>4}  {'ghid':>4}  {'drop':>5}  {'nsamp':>5}  {'bs':>4}  {'wd':>9}")
    logger.log("-" * 85)
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else -1, reverse=True)
    for t in sorted_trials[:10]:
        if t.value is None:
            continue
        p = t.params
        logger.log(
            f"{t.number:>4}  {t.value:>8.4f}  {p['lr']:>9.6f}  {p['lambda']:>8.5f}  "
            f"{p['temperature']:>7.2f}  {p['hidden_dim']:>4}  {p['global_hidden_dim']:>4}  "
            f"{p['dropout']:>5.3f}  {p['num_samples']:>5}  {p['batch_size']:>4}  {p['weight_decay']:>9.7f}"
        )

    # Retrain best on train (folds 1-6), evaluate on test (folds 9-10)
    logger.log("")
    logger.log("=" * 90)
    logger.log("RETRAINING BEST CONFIG & EVALUATING ON TEST SET")
    logger.log("=" * 90)

    bp = best.params
    bs = bp["batch_size"]
    train_loader = make_dataloader(X_train_global, Y_train_global, bs, shuffle=True)
    val_loader = make_dataloader(X_val_global, Y_val_global, bs, shuffle=False)
    test_loader = make_dataloader(X_test, Y_test, bs, shuffle=False)

    best_val_f1, best_epoch, best_state, task_net = run_ggel(
        train_loader, val_loader, device, logger,
        hidden_dim=bp["hidden_dim"], global_hidden_dim=bp["global_hidden_dim"],
        dropout=bp["dropout"], lr=bp["lr"], weight_decay=bp["weight_decay"],
        lam=bp["lambda"], num_samples=bp["num_samples"],
        temperature=bp["temperature"],
        trial_num=999,
    )

    # Load best checkpoint and evaluate on test
    if best_state is not None:
        task_net.load_state_dict(best_state)
        task_net = task_net.to(device)
    te_loss, te_pi, te_mic, te_mac = evaluate(task_net, test_loader, device)

    logger.log("")
    logger.log("=" * 90)
    logger.log("FINAL TEST RESULTS (best Optuna config)")
    logger.log("=" * 90)
    logger.log(f"  Val F1 (pi):   {best_val_f1:.4f} @ epoch {best_epoch}")
    logger.log(f"  Test loss:     {te_loss:.5f}")
    logger.log(f"  Test F1 (pi):  {te_pi:.4f}")
    logger.log(f"  Test F1 (mic): {te_mic:.4f}")
    logger.log(f"  Test F1 (mac): {te_mac:.4f}")
    logger.log("")
    logger.log(f"Results saved to: {OUTPUT_FILE}")
    logger.close()


if __name__ == "__main__":
    main()
