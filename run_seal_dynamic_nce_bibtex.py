"""
SEAL Dynamic + NCE Loss on Bibtex — Original paper's approach.

Reproduces the SEAL (Structured Energy Network As a Loss) training from:
  "Structured Energy Network As a Loss" (NeurIPS 2022)

Architecture (matching paper's config bibtex_strat_SEAL-static-NCE):
  - Task network: 2-layer MLP (1836 -> 400 -> 400) + label embeddings (400d)
  - Score/Energy network: same task_nn architecture + feedforward global score
  - NCE loss with Bernoulli sampling, per-instance F1 cost
  - Minimax training: score_nn and task_nn updated jointly

Data splits (10-fold stratified, MEKA format):
  - Train: folds 1-6
  - Validation: folds 7-8
  - Test: folds 9-10

Usage:
    python run_seal_dynamic_nce_bibtex.py

Monitor progress:
    tail -f output/seal_dynamic_nce/results.txt
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
# Data loading (ARFF sparse MEKA format)
# ---------------------------------------------------------------------------

def load_arff(filepath, num_labels):
    """Load a single ARFF file (sparse MEKA format)."""
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
    """Per-instance F1 (the paper's primary metric 'fixed_f1')."""
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
# Networks (matching paper's architecture)
# ---------------------------------------------------------------------------

class TaskNet(nn.Module):
    """SEAL task network: 2-layer MLP feature backbone + label embeddings.

    Matches the paper config:
      feature_network: 1836 -> 400 (softplus, dropout=0.5) -> 400 (softplus)
      label_embeddings: 159 x 400
      output: dot product of features and label embeddings
    """

    def __init__(self, input_dim=1836, hidden_dim=400, num_labels=159, dropout=0.5):
        super().__init__()
        self.feature_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Softplus(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
        )
        self.label_embeddings = nn.Embedding(num_labels, hidden_dim)
        self.num_labels = num_labels

    def forward(self, x):
        """Returns logits (batch, num_labels)."""
        features = self.feature_network(x)  # (batch, hidden_dim)
        # Dot product with each label embedding
        logits = torch.matmul(features, self.label_embeddings.weight.T)
        return logits


class ScoreNN(nn.Module):
    """SEAL score/energy network: task_nn features + global feedforward score.

    Matches the paper config (multilabel_classification_score_nn.py):
      - Local score: dot(task_nn_logits, y) — compute logits ONCE, broadcast
      - Global score: feedforward on y -> scalar (softplus activation, 200 hidden)

    The original computes score_nn.task_nn(x) once and broadcasts logits
    across all y samples. This avoids re-running dropout on the same x.
    """

    def __init__(self, input_dim=1836, hidden_dim=400, num_labels=159,
                 global_hidden_dim=200, dropout=0.5):
        super().__init__()
        # Score_nn's own task_nn (separate weights from inference task_nn)
        self.feature_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Softplus(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
        )
        self.label_embeddings = nn.Embedding(num_labels, hidden_dim)

        # Global score: feedforward on label vector
        self.global_score_ff = nn.Sequential(
            nn.Linear(num_labels, global_hidden_dim),
            nn.Softplus(),
        )
        self.global_projection = nn.Linear(global_hidden_dim, 1, bias=False)

    def forward(self, x, y):
        """
        Args:
            x: (batch, input_dim) input features
            y: (batch, num_samples, num_labels) label vectors
        Returns:
            score: (batch, num_samples) scalar energy per instance per sample
        """
        # Local score: compute logits ONCE, broadcast across all samples
        features = self.feature_network(x)  # (batch, hidden)
        logits = torch.matmul(features, self.label_embeddings.weight.T)  # (batch, num_labels)
        e_local = torch.sum(logits.unsqueeze(1) * y, dim=-1)  # (batch, num_samples)

        # Global score: feedforward on label structure
        e_global = self.global_projection(self.global_score_ff(y)).squeeze(-1)  # (batch, num_samples)

        return e_local + e_global


# ---------------------------------------------------------------------------
# NCE Loss (matching paper: nce_loss.py + nce_mlc_loss.py)
# ---------------------------------------------------------------------------

def nce_ranking_loss(score_nn, x, y_star, y_probs, num_samples=10):
    """SEAL's NCE ranking loss for training the score network.

    Matches nce_loss.py NCERankingLoss.compute_loss():
      1. sample(y_hat) -> Bernoulli negatives from task_nn probs
      2. y = cat(labels, samples) -> ground truth at index 0
      3. distance = mul * sum(BCE(probs, y)) with sign="-" -> distance = ln P_n(y)
      4. score = score_nn(x, y)
      5. adjusted = score - distance
      6. loss = CrossEntropy(adjusted, target=0)
    """
    batch, L = y_star.shape

    # Sample negatives from Bernoulli(task_nn_probs)
    samples = torch.distributions.Bernoulli(probs=y_probs).sample(
        [num_samples]
    ).permute(1, 0, 2)  # (batch, num_samples, L)

    # Concatenate: [ground_truth, samples] — ground truth at index 0
    y_all = torch.cat([y_star.unsqueeze(1), samples], dim=1)  # (batch, 1+K, L)

    # Score all candidates — score_nn now takes (batch, 1+K, L) directly
    scores = score_nn(x, y_all)  # (batch, 1+K)

    # Distance: sign="-", mul=-1 -> distance = -1 * BCE = ln P_n(y)
    probs_exp = y_probs.unsqueeze(1).expand_as(y_all)
    bce = F.binary_cross_entropy(
        probs_exp.clamp(1e-6, 1 - 1e-6), y_all, reduction="none"
    ).sum(dim=-1)
    distance = -bce  # (batch, 1+K)

    # NCE ranking: cross-entropy with target=0 (ground truth at index 0)
    adjusted = scores - distance
    target = torch.zeros(batch, dtype=torch.long, device=x.device)
    return F.cross_entropy(adjusted, target)


# ---------------------------------------------------------------------------
# Training loop: SEAL Dynamic NCE
# ---------------------------------------------------------------------------

def train_epoch(task_nn, score_nn, opt_task, opt_score,
                train_loader, device, score_loss_weight=0.006780,
                num_samples=10, num_steps_score=1, num_steps_task=10,
                grad_clip_task=10.0):
    """One epoch of SEAL dynamic training (minimax).

    Matches gradient_descent_minimax_trainer.py _train_epoch():
      inner_mode = score_nn, num_steps = {score_nn: 1, task_nn: 10}

      For each batch, the trainer runs num_outer_steps (=task_nn=10) iterations.
      Each iteration does num_inner_steps (=score_nn=1) inner steps then 1 outer step:

        for outer_step in range(10):       # num_steps[task_nn]
            for inner_step in range(1):    # num_steps[score_nn]
                score_nn_step(batch)
            task_nn_step(batch)

      Total: 10 score_nn updates + 10 task_nn updates per batch, alternating.
    """
    task_nn.train()
    score_nn.train()
    total_nce, total_task_loss, n = 0.0, 0.0, 0

    for x, y_star in train_loader:
        x, y_star = x.to(device), y_star.to(device)
        batch_nce, batch_task = 0.0, 0.0

        # Alternating minimax: num_outer_steps pairs of (inner + outer)
        for _outer in range(num_steps_task):

            # --- INNER: Score network update (NCE loss) ---
            for _inner in range(num_steps_score):
                opt_score.zero_grad()
                with torch.no_grad():
                    y_probs = torch.sigmoid(task_nn(x)).clamp(1e-6, 1 - 1e-6)
                loss_nce = nce_ranking_loss(score_nn, x, y_star, y_probs, num_samples)
                loss_nce.backward()
                opt_score.step()

            # --- OUTER: Task network update (BCE + score-based loss) ---
            opt_task.zero_grad()
            logits = task_nn(x)
            y_pred = torch.sigmoid(logits)

            # BCE loss (BCEWithLogitsLoss on raw logits, matching original)
            loss_bce = F.binary_cross_entropy_with_logits(logits, y_star)

            # Score-based loss: -score_nn(x, sigmoid(logits))
            # Original: normalize_y=True in score_loss means apply sigmoid to
            # raw logits — NOT L1 normalization. y_pred = sigmoid(logits) is correct.
            # score_nn params frozen during task_nn update (no_grad_for_other_mode)
            for p in score_nn.parameters():
                p.requires_grad_(False)
            y_for_score = y_pred.unsqueeze(1)  # (batch, 1, num_labels)
            score_val = score_nn(x, y_for_score).mean()
            loss_score = -score_val
            for p in score_nn.parameters():
                p.requires_grad_(True)

            # Combined loss (matching paper: 0.00678 * score_loss + 1.0 * bce)
            loss_task = score_loss_weight * loss_score + loss_bce
            loss_task.backward()
            torch.nn.utils.clip_grad_norm_(task_nn.parameters(), grad_clip_task)
            opt_task.step()

            batch_nce += loss_nce.item()
            batch_task += loss_task.item()

        total_nce += batch_nce / num_steps_task
        total_task_loss += batch_task / num_steps_task
        n += 1

    return total_nce / n, total_task_loss / n


@torch.no_grad()
def evaluate(task_nn, data_loader, device):
    task_nn.eval()
    total_loss, n = 0.0, 0
    all_preds, all_labels = [], []
    for x, y_star in data_loader:
        x, y_star = x.to(device), y_star.to(device)
        logits = task_nn(x)
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
# Main
# ---------------------------------------------------------------------------

def run_phase(phase_name, task_nn, score_nn, train_loader, val_loader,
              device, logger, num_epochs, lr_task, lr_score, weight_decay,
              patience, grad_clip_task, num_samples, num_steps_score,
              num_steps_task, score_loss_weight, lr_reduce_factor,
              lr_reduce_patience):
    """Run one training phase (pretraining or fine-tuning)."""

    opt_task = torch.optim.AdamW(
        task_nn.parameters(), lr=lr_task, weight_decay=weight_decay
    )
    opt_score = torch.optim.AdamW(
        score_nn.parameters(), lr=lr_score, weight_decay=weight_decay
    )
    scheduler_task = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_task, mode='max', factor=lr_reduce_factor,
        patience=lr_reduce_patience,
    )

    best_val_f1, best_epoch = 0.0, -1
    best_task_state, best_score_state = None, None

    logger.log("")
    hdr = (
        f"{'Ep':>3}  {'NCE':>9}  {'TaskLoss':>9}  "
        f"{'VlLoss':>9}  {'VlF1pi':>8}  {'VlF1mic':>8}  {'VlF1mac':>8}  "
        f"{'BstEp':>5}  {'BstF1':>7}  {'LR':>10}  {'Time':>5}"
    )
    logger.log(hdr)
    logger.log("-" * len(hdr))

    for epoch in range(num_epochs):
        t0 = time.time()

        nce_loss, task_loss = train_epoch(
            task_nn, score_nn, opt_task, opt_score,
            train_loader, device,
            score_loss_weight=score_loss_weight,
            num_samples=num_samples,
            num_steps_score=num_steps_score,
            num_steps_task=num_steps_task,
            grad_clip_task=grad_clip_task,
        )

        vl_loss, vl_pi, vl_mic, vl_mac = evaluate(task_nn, val_loader, device)
        elapsed = time.time() - t0

        scheduler_task.step(vl_pi)
        current_lr = opt_task.param_groups[0]['lr']

        if vl_pi > best_val_f1:
            best_val_f1 = vl_pi
            best_epoch = epoch
            best_task_state = {k: v.cpu().clone() for k, v in task_nn.state_dict().items()}
            best_score_state = {k: v.cpu().clone() for k, v in score_nn.state_dict().items()}

        logger.log(
            f"{epoch:>3}  {nce_loss:>9.5f}  {task_loss:>9.5f}  "
            f"{vl_loss:>9.5f}  {vl_pi:>8.4f}  {vl_mic:>8.4f}  {vl_mac:>8.4f}  "
            f"{best_epoch:>5}  {best_val_f1:>7.4f}  {current_lr:>10.2e}  {elapsed:>4.1f}s"
        )

        if epoch - best_epoch >= patience:
            logger.log(f"\n  Early stopping at epoch {epoch} (patience={patience})")
            break

    return best_task_state, best_score_state, best_val_f1, best_epoch


def main():
    DATA_DIR = os.path.join("data", "bibtex_stratified10folds_meka")
    OUTPUT_DIR = os.path.join("output", "seal_dynamic_nce")
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "results.txt")

    # Architecture
    NUM_LABELS = 159
    INPUT_DIM = 1836
    HIDDEN_DIM = 400
    GLOBAL_HIDDEN_DIM = 200
    DROPOUT = 0.5
    BATCH_SIZE = 32
    WEIGHT_DECAY = 1e-05
    SEED = 42

    # =====================================================================
    # Phase 1: NCE Pretraining (bibtex_strat_nce_64bujahw config)
    #   Strong score_loss_weight, many score_nn steps, trains a good energy fn
    # =====================================================================
    P1_NUM_EPOCHS = 300
    P1_LR_TASK = 0.0011682627302272157
    P1_LR_SCORE = 4.512859464505083e-05
    P1_PATIENCE = 20
    P1_GRAD_CLIP_TASK = 10
    P1_NUM_SAMPLES = 20
    P1_NUM_STEPS_SCORE = 12
    P1_NUM_STEPS_TASK = 5
    P1_SCORE_LOSS_WEIGHT = 5.514710814981766
    P1_LR_REDUCE_FACTOR = 0.5
    P1_LR_REDUCE_PATIENCE = 5

    # =====================================================================
    # Phase 2: SEAL Fine-tuning (bibtex_strat_SEAL-static-NCE config)
    #   Load pretrained score_nn, low score weight, focus on task_nn
    # =====================================================================
    P2_NUM_EPOCHS = 300
    P2_LR_TASK = 0.0015178255756827371
    P2_LR_SCORE = 4.512859464505083e-05
    P2_PATIENCE = 20
    P2_GRAD_CLIP_TASK = 10
    P2_NUM_SAMPLES = 10
    P2_NUM_STEPS_SCORE = 1
    P2_NUM_STEPS_TASK = 10
    P2_SCORE_LOSS_WEIGHT = 0.006780079477675936
    P2_LR_REDUCE_FACTOR = 0.5
    P2_LR_REDUCE_PATIENCE = 5

    logger = LiveLogger(OUTPUT_FILE)

    logger.log("=" * 100)
    logger.log("SEAL Dynamic + NCE Loss -- Bibtex (Two-Phase: Pretrain + Fine-tune)")
    logger.log(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log("=" * 100)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"\nDevice: {device}")
    if device.type == "cuda":
        logger.log(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ---- Load data ----
    logger.log("\nLoading bibtex data...")
    X_train, Y_train = load_arff_folds(DATA_DIR, [1, 2, 3, 4, 5, 6], NUM_LABELS, logger)
    X_val, Y_val = load_arff_folds(DATA_DIR, [7, 8], NUM_LABELS, logger)
    X_test, Y_test = load_arff_folds(DATA_DIR, [9, 10], NUM_LABELS, logger)

    logger.log(f"\n  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    logger.log(f"  Features: {X_train.shape[1]}, Labels: {Y_train.shape[1]}")

    train_loader = make_dataloader(X_train, Y_train, BATCH_SIZE, shuffle=True)
    val_loader = make_dataloader(X_val, Y_val, BATCH_SIZE, shuffle=False)
    test_loader = make_dataloader(X_test, Y_test, BATCH_SIZE, shuffle=False)

    # ---- Build models ----
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    task_nn = TaskNet(INPUT_DIM, HIDDEN_DIM, NUM_LABELS, DROPOUT).to(device)
    score_nn_model = ScoreNN(INPUT_DIM, HIDDEN_DIM, NUM_LABELS, GLOBAL_HIDDEN_DIM, DROPOUT).to(device)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    task_nn.apply(init_weights)
    score_nn_model.apply(init_weights)

    logger.log(f"  task_nn params:  {sum(p.numel() for p in task_nn.parameters()):,}")
    logger.log(f"  score_nn params: {sum(p.numel() for p in score_nn_model.parameters()):,}")

    # ================================================================
    # PHASE 1: NCE Pretraining
    # ================================================================
    logger.log("")
    logger.log("=" * 100)
    logger.log("PHASE 1: NCE Pretraining (bibtex_strat_nce config)")
    logger.log(f"  score_loss_weight={P1_SCORE_LOSS_WEIGHT}, "
               f"steps=(score:{P1_NUM_STEPS_SCORE}, task:{P1_NUM_STEPS_TASK}), "
               f"samples={P1_NUM_SAMPLES}")
    logger.log(f"  lr_task={P1_LR_TASK:.6f}, lr_score={P1_LR_SCORE:.6e}")
    logger.log("=" * 100)

    p1_task_state, p1_score_state, p1_val_f1, p1_best_epoch = run_phase(
        "Phase1-NCE", task_nn, score_nn_model, train_loader, val_loader,
        device, logger,
        num_epochs=P1_NUM_EPOCHS, lr_task=P1_LR_TASK, lr_score=P1_LR_SCORE,
        weight_decay=WEIGHT_DECAY, patience=P1_PATIENCE,
        grad_clip_task=P1_GRAD_CLIP_TASK, num_samples=P1_NUM_SAMPLES,
        num_steps_score=P1_NUM_STEPS_SCORE, num_steps_task=P1_NUM_STEPS_TASK,
        score_loss_weight=P1_SCORE_LOSS_WEIGHT,
        lr_reduce_factor=P1_LR_REDUCE_FACTOR,
        lr_reduce_patience=P1_LR_REDUCE_PATIENCE,
    )

    # Save Phase 1 pretrained checkpoint
    pretrained_path = os.path.join(OUTPUT_DIR, "pretrained_nce.pt")
    torch.save({'task_nn': p1_task_state, 'score_nn': p1_score_state}, pretrained_path)
    logger.log(f"\n  Phase 1 best val F1: {p1_val_f1:.4f} (epoch {p1_best_epoch})")
    logger.log(f"  Pretrained weights saved to: {pretrained_path}")

    # Evaluate Phase 1 on test
    task_nn.load_state_dict(p1_task_state)
    task_nn = task_nn.to(device)
    p1_te = evaluate(task_nn, test_loader, device)
    logger.log(f"  Phase 1 test F1 (pi): {p1_te[1]:.4f}, micro: {p1_te[2]:.4f}, macro: {p1_te[3]:.4f}")

    # ================================================================
    # PHASE 2: SEAL Fine-tuning with pretrained score_nn
    # ================================================================
    logger.log("")
    logger.log("=" * 100)
    logger.log("PHASE 2: SEAL Fine-tuning (bibtex_strat_SEAL-static-NCE config)")
    logger.log(f"  score_loss_weight={P2_SCORE_LOSS_WEIGHT}, "
               f"steps=(score:{P2_NUM_STEPS_SCORE}, task:{P2_NUM_STEPS_TASK}), "
               f"samples={P2_NUM_SAMPLES}")
    logger.log(f"  lr_task={P2_LR_TASK:.6f}, lr_score={P2_LR_SCORE:.6e}")
    logger.log("  Initializing score_nn from Phase 1 pretrained weights")
    logger.log("  Reinitializing task_nn from scratch (matching paper)")
    logger.log("=" * 100)

    # Reset task_nn to fresh weights, keep pretrained score_nn
    torch.manual_seed(SEED + 1)  # different seed for phase 2
    task_nn_p2 = TaskNet(INPUT_DIM, HIDDEN_DIM, NUM_LABELS, DROPOUT).to(device)
    task_nn_p2.apply(init_weights)

    # Load pretrained score_nn weights
    score_nn_model.load_state_dict(p1_score_state)
    score_nn_model = score_nn_model.to(device)

    p2_task_state, p2_score_state, p2_val_f1, p2_best_epoch = run_phase(
        "Phase2-SEAL", task_nn_p2, score_nn_model, train_loader, val_loader,
        device, logger,
        num_epochs=P2_NUM_EPOCHS, lr_task=P2_LR_TASK, lr_score=P2_LR_SCORE,
        weight_decay=WEIGHT_DECAY, patience=P2_PATIENCE,
        grad_clip_task=P2_GRAD_CLIP_TASK, num_samples=P2_NUM_SAMPLES,
        num_steps_score=P2_NUM_STEPS_SCORE, num_steps_task=P2_NUM_STEPS_TASK,
        score_loss_weight=P2_SCORE_LOSS_WEIGHT,
        lr_reduce_factor=P2_LR_REDUCE_FACTOR,
        lr_reduce_patience=P2_LR_REDUCE_PATIENCE,
    )

    # ---- Test with best Phase 2 model ----
    task_nn_p2.load_state_dict(p2_task_state)
    task_nn_p2 = task_nn_p2.to(device)
    te_loss, te_pi, te_mic, te_mac = evaluate(task_nn_p2, test_loader, device)

    logger.log("")
    logger.log("=" * 100)
    logger.log("FINAL TEST RESULTS (Phase 2: SEAL Dynamic + NCE)")
    logger.log("-" * 60)
    logger.log(f"  Phase 1 best val F1: {p1_val_f1:.4f} (epoch {p1_best_epoch})")
    logger.log(f"  Phase 2 best val F1: {p2_val_f1:.4f} (epoch {p2_best_epoch})")
    logger.log(f"  Test loss:           {te_loss:.5f}")
    logger.log(f"  Test F1 (pi):        {te_pi:.4f}")
    logger.log(f"  Test F1 (micro):     {te_mic:.4f}")
    logger.log(f"  Test F1 (macro):     {te_mac:.4f}")
    logger.log("=" * 100)

    # Save final model
    model_path = os.path.join(OUTPUT_DIR, "best_model.pt")
    torch.save({
        'task_nn': p2_task_state,
        'score_nn': p2_score_state,
        'phase1_val_f1': p1_val_f1,
        'phase2_val_f1': p2_val_f1,
        'best_epoch': p2_best_epoch,
        'test_metrics': {
            'loss': te_loss, 'f1_pi': te_pi,
            'f1_micro': te_mic, 'f1_macro': te_mac,
        },
    }, model_path)
    logger.log(f"\nModel saved to: {model_path}")
    logger.log(f"Results saved to: {OUTPUT_FILE}")
    logger.close()


if __name__ == "__main__":
    main()
