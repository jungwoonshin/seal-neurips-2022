"""
Self-Supervised Learning Framework for BibtEx Multi-Label Classification.

Loss = feature2feature + feature2label

- feature2feature: sequentially mask one feature at a time, predict it from the rest
- feature2label: sequentially mask one feature at a time, predict labels
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from skmultilearn.dataset import load_from_arff

# ─── Config ──────────────────────────────────────────────────────────────────

NUM_FEATURES = 1836
NUM_LABELS = 159
HIDDEN_DIM = 256
LABEL_EMB_DIM = 256  # must match HIDDEN_DIM for bilinear scoring
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_FILE = "ssl_bibtex_log.txt"

DATA_DIR = "data/bibtex_stratified10folds_meka"


# ─── Dataset ─────────────────────────────────────────────────────────────────

class BibtexDataset(Dataset):
    def __init__(self, arff_path, num_labels=NUM_LABELS):
        x, y, _, _ = load_from_arff(
            arff_path, label_count=num_labels, return_attribute_definitions=True
        )
        self.x = torch.tensor(x.toarray(), dtype=torch.float32)
        self.y = torch.tensor(y.toarray(), dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def load_folds(data_dir, fold_ids):
    """Load and concatenate multiple stratified folds."""
    datasets = []
    for fid in fold_ids:
        path = os.path.join(data_dir, f"Bibtex-fold{fid}.arff")
        datasets.append(BibtexDataset(path))
    return ConcatDataset(datasets)


# ─── Model ───────────────────────────────────────────────────────────────────

class SSLModel(nn.Module):
    """
    Shared encoder with two heads:
    1. Feature2Feature: masked feature reconstruction
    2. Feature2Label: predict labels from feature representation
    """

    def __init__(self, num_features, num_labels, hidden_dim, label_emb_dim):
        super().__init__()
        assert hidden_dim == label_emb_dim, "hidden_dim must equal label_emb_dim"

        # Learnable mask token: replaces masked feature positions
        self.mask_token = nn.Parameter(torch.zeros(num_features))

        # Shared feature encoder: raw features -> hidden representation
        self.feature_encoder = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Head 1: Feature2Feature – reconstruct masked features from hidden repr
        self.feature_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_features),
        )

        # Head 2: Feature2Label – label embeddings for bilinear scoring
        self.label_embeddings = nn.Embedding(num_labels, label_emb_dim)

    def encode_features(self, x):
        """Encode raw features to hidden representation."""
        return self.feature_encoder(x)  # (B, hidden_dim)

    def decode_features(self, h):
        """Reconstruct features from hidden representation."""
        return self.feature_decoder(h)  # (B, num_features)

    def predict_labels(self, h):
        """Predict label logits via dot product with label embeddings."""
        logits = torch.matmul(h, self.label_embeddings.weight.T)  # (B, num_labels)
        return logits


# ─── Loss Functions ──────────────────────────────────────────────────────────

def mask_feature(model, x, feat_idx):
    """Replace feature at feat_idx with the learned mask token value."""
    x_masked = x.clone()
    x_masked[:, feat_idx] = model.mask_token[feat_idx]
    return x_masked


def feature2feature_loss(model, x, feat_idx):
    """
    Mask a single feature at feat_idx with learned mask token, encode,
    decode, and compute reconstruction loss on that one masked position.
    """
    x_masked = mask_feature(model, x, feat_idx)

    h = model.encode_features(x_masked)
    x_recon = model.decode_features(h)

    loss = F.binary_cross_entropy_with_logits(
        x_recon[:, feat_idx], x[:, feat_idx]
    )
    return loss


def feature2label_loss_masked(model, x, y, feat_idx):
    """Mask a single feature at feat_idx with learned mask token, predict labels."""
    x_masked = mask_feature(model, x, feat_idx)

    h = model.encode_features(x_masked)
    logits = model.predict_labels(h)
    loss = F.binary_cross_entropy_with_logits(logits, y)
    return loss


def feature2label_loss(model, x, y):
    """Standard BCE loss: encode full features -> predict labels."""
    h = model.encode_features(x)
    logits = model.predict_labels(h)
    loss = F.binary_cross_entropy_with_logits(logits, y)
    return loss


# ─── Instance-level F1 ──────────────────────────────────────────────────────

def instance_f1(y_true, y_pred):
    """
    Compute instance-level F1 with 0.5 threshold.
    For each instance: F1 = 2*|intersection| / (|pred| + |true|)
    Then average over all instances.
    """
    intersection = (y_true * y_pred).sum(axis=1)
    pred_sum = y_pred.sum(axis=1)
    true_sum = y_true.sum(axis=1)
    denom = pred_sum + true_sum
    f1_per_instance = np.where(denom > 0, 2.0 * intersection / denom, 1.0)
    return f1_per_instance.mean()


# ─── Evaluation ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    total_f2f, total_f2l_m, total_f2l = 0.0, 0.0, 0.0
    n_batches = 0
    eval_feat_idx = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        total_f2f += feature2feature_loss(model, x, eval_feat_idx).item()
        total_f2l_m += feature2label_loss_masked(model, x, y, eval_feat_idx).item()
        total_f2l += feature2label_loss(model, x, y).item()
        n_batches += 1
        eval_feat_idx = (eval_feat_idx + 1) % NUM_FEATURES

        # Predictions use full (unmasked) features
        h = model.encode_features(x)
        logits = model.predict_labels(h)
        preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    inst_f1 = instance_f1(all_labels, all_preds)

    return {
        'f2f_loss': total_f2f / n_batches,
        'f2l_m_loss': total_f2l_m / n_batches,
        'f2l_loss': total_f2l / n_batches,
        'total_loss': (total_f2f + total_f2l_m + total_f2l) / n_batches,
        'instance_f1': inst_f1,
    }


# ─── Logging ─────────────────────────────────────────────────────────────────

def log(msg, log_file):
    """Print and immediately flush to log file."""
    print(msg, flush=True)
    with open(log_file, 'a') as f:
        f.write(msg + '\n')
        f.flush()
        os.fsync(f.fileno())


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    with open(LOG_FILE, 'w') as f:
        f.write('')

    log(f"Device: {DEVICE}", LOG_FILE)
    log(f"Config: hidden_dim={HIDDEN_DIM}, lr={LR}, batch_size={BATCH_SIZE}, "
        f"epochs={EPOCHS}, sequential masking over {NUM_FEATURES} features", LOG_FILE)
    log("=" * 100, LOG_FILE)

    # Load stratified folds: train=1-6, val=7-8, test=9-10
    log("Loading stratified folds...", LOG_FILE)
    train_ds = load_folds(DATA_DIR, [1, 2, 3, 4, 5, 6])
    val_ds = load_folds(DATA_DIR, [7, 8])
    test_ds = load_folds(DATA_DIR, [9, 10])

    log(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}", LOG_FILE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = SSLModel(NUM_FEATURES, NUM_LABELS, HIDDEN_DIM, LABEL_EMB_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    total_params = sum(p.numel() for p in model.parameters())
    log(f"Model parameters: {total_params:,}", LOG_FILE)
    log("=" * 100, LOG_FILE)

    best_val_f1 = 0.0
    feat_idx = 0  # sequential feature index counter

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_f2f, epoch_f2l_m, epoch_f2l = 0.0, 0.0, 0.0
        n_batches = 0
        t_start = time.time()

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            loss_f2f = feature2feature_loss(model, x, feat_idx)
            loss_f2l_m = feature2label_loss_masked(model, x, y, feat_idx)
            loss_f2l = feature2label_loss(model, x, y)
            loss = loss_f2f + loss_f2l_m + loss_f2l

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_f2f += loss_f2f.item()
            epoch_f2l_m += loss_f2l_m.item()
            epoch_f2l += loss_f2l.item()
            n_batches += 1

            # Advance to next feature
            feat_idx = (feat_idx + 1) % NUM_FEATURES

        elapsed = time.time() - t_start

        train_f2f = epoch_f2f / n_batches
        train_f2l_m = epoch_f2l_m / n_batches
        train_f2l = epoch_f2l / n_batches
        train_total = train_f2f + train_f2l_m + train_f2l

        val_metrics = evaluate(model, val_loader)

        is_best = val_metrics['instance_f1'] > best_val_f1
        if is_best:
            best_val_f1 = val_metrics['instance_f1']

        msg = (
            f"Epoch {epoch:3d}/{EPOCHS} ({elapsed:.1f}s) | "
            f"Train  f2f={train_f2f:.4f}  f2l_m={train_f2l_m:.4f}  f2l={train_f2l:.4f}  total={train_total:.4f} | "
            f"Val  f2f={val_metrics['f2f_loss']:.4f}  f2l_m={val_metrics['f2l_m_loss']:.4f}  "
            f"f2l={val_metrics['f2l_loss']:.4f}  total={val_metrics['total_loss']:.4f} | "
            f"Val  instance_f1={val_metrics['instance_f1']:.4f}"
            f"{'  *best*' if is_best else ''}"
        )
        log(msg, LOG_FILE)

    # Final test evaluation
    log("=" * 100, LOG_FILE)
    test_metrics = evaluate(model, test_loader)
    log(
        f"Test Results | "
        f"f2f={test_metrics['f2f_loss']:.4f}  f2l_m={test_metrics['f2l_m_loss']:.4f}  "
        f"f2l={test_metrics['f2l_loss']:.4f}  total={test_metrics['total_loss']:.4f} | "
        f"instance_f1={test_metrics['instance_f1']:.4f}",
        LOG_FILE,
    )
    log(f"Best Val instance_f1: {best_val_f1:.4f}", LOG_FILE)


if __name__ == '__main__':
    main()
