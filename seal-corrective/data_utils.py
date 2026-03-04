"""
Data loading utilities for multi-label classification datasets.
Supports sparse ARFF format (bibtex, etc.).
"""

import torch
from torch.utils.data import DataLoader, TensorDataset


def load_arff_sparse(filepath: str, num_features: int, num_labels: int):
    """
    Load a sparse ARFF file into dense feature and label tensors.

    Args:
        filepath: Path to the .arff file.
        num_features: Number of feature attributes (first num_features columns).
        num_labels: Number of label attributes (last num_labels columns).

    Returns:
        X: (n_samples, num_features) float tensor
        Y: (n_samples, num_labels) float tensor
    """
    total_attrs = num_features + num_labels
    rows_x = []
    rows_y = []

    in_data = False
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line == "@data":
                in_data = True
                continue
            if not in_data or not line or line.startswith("%"):
                continue

            # Parse sparse format: {idx val, idx val, ...}
            x = torch.zeros(num_features)
            y = torch.zeros(num_labels)

            content = line.strip("{}")
            if content:
                for pair in content.split(","):
                    pair = pair.strip()
                    if not pair:
                        continue
                    parts = pair.split()
                    idx = int(parts[0])
                    val = float(parts[1])
                    if idx < num_features:
                        x[idx] = val
                    else:
                        y[idx - num_features] = val

            rows_x.append(x)
            rows_y.append(y)

    X = torch.stack(rows_x)
    Y = torch.stack(rows_y)
    return X, Y


def load_bibtex(data_dir: str, batch_size: int = 32):
    """
    Load bibtex dataset using stratified 10-fold splits from meka format.

    Fixed split: folds 1-6 for training, 7-8 for validation, 9-10 for test.

    Args:
        data_dir: Path to bibtex_stratified10folds_meka/ directory.
        batch_size: Batch size for DataLoaders.

    Returns:
        train_loader, val_loader, test_loader, input_dim, num_labels, pos_weight
    """
    import os

    NUM_FEATURES = 1836
    NUM_LABELS = 159

    TRAIN_FOLDS = list(range(1, 7))   # folds 1-6
    VAL_FOLDS = [7, 8]                # folds 7-8
    TEST_FOLDS = [9, 10]              # folds 9-10

    # Load all 10 folds
    folds = {}
    for i in range(1, 11):
        path = os.path.join(data_dir, f"Bibtex-fold{i}.arff")
        folds[i] = load_arff_sparse(path, NUM_FEATURES, NUM_LABELS)

    train_x = torch.cat([folds[i][0] for i in TRAIN_FOLDS], dim=0)
    train_y = torch.cat([folds[i][1] for i in TRAIN_FOLDS], dim=0)
    val_x = torch.cat([folds[i][0] for i in VAL_FOLDS], dim=0)
    val_y = torch.cat([folds[i][1] for i in VAL_FOLDS], dim=0)
    test_x = torch.cat([folds[i][0] for i in TEST_FOLDS], dim=0)
    test_y = torch.cat([folds[i][1] for i in TEST_FOLDS], dim=0)

    print(f"Bibtex stratified split: train=folds{TRAIN_FOLDS}, "
          f"val=folds{VAL_FOLDS}, test=folds{TEST_FOLDS}")
    print(f"  Sizes: train={train_x.shape[0]}, val={val_x.shape[0]}, test={test_x.shape[0]}")
    print(f"  Features: {NUM_FEATURES}, Labels: {NUM_LABELS}")
    print(f"  Avg labels/sample: train={train_y.sum(dim=-1).mean():.1f}, "
          f"val={val_y.sum(dim=-1).mean():.1f}, test={test_y.sum(dim=-1).mean():.1f}")

    train_loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(val_x, val_y), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=False)

    # Compute positive class weights for imbalanced labels
    pos_freq = train_y.mean(dim=0)  # (num_labels,)
    pos_weight = ((1.0 - pos_freq) / (pos_freq + 1e-8)).clamp(max=50.0)
    print(f"  Pos weight range: [{pos_weight.min():.1f}, {pos_weight.max():.1f}]")

    return train_loader, val_loader, test_loader, NUM_FEATURES, NUM_LABELS, pos_weight


def compute_instance_f1(y_pred: torch.Tensor, y_true: torch.Tensor,
                        threshold: float = 0.5) -> float:
    """
    Compute instance-level F1 score (averaged over samples).

    Args:
        y_pred: (n, L) soft predictions
        y_true: (n, L) binary ground truth
        threshold: Binarization threshold

    Returns:
        Mean instance-level F1 score.
    """
    y_hard = (y_pred >= threshold).float()
    y_true_f = y_true.float()

    intersection = (y_hard * y_true_f).sum(dim=-1)
    pred_sum = y_hard.sum(dim=-1)
    true_sum = y_true_f.sum(dim=-1)

    denom = pred_sum + true_sum
    f1 = 2.0 * intersection / (denom + 1e-8)

    # Handle case where both pred and true are all zeros
    both_zero = (pred_sum == 0) & (true_sum == 0)
    f1 = torch.where(both_zero, torch.ones_like(f1), f1)

    return f1.mean().item()
