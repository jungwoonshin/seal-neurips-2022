"""
Data loading utilities for multi-label classification datasets.
Supports sparse and dense ARFF formats.
"""

import os

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
            if line.lower() == "@data":
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


def load_arff_dense(filepath: str, num_features: int, num_labels: int,
                    skip_nominal: bool = False):
    """
    Load a dense ARFF file into feature and label tensors.

    Handles both numeric and YES/NO valued attributes.

    Args:
        filepath: Path to the .arff file.
        num_features: Number of feature columns (after skipping nominal).
        num_labels: Number of label columns (last num_labels columns).
        skip_nominal: If True, skip the first column (e.g. protein name in genbase).
    """
    rows_x = []
    rows_y = []

    in_data = False
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.lower() == "@data":
                in_data = True
                continue
            if not in_data or not line or line.startswith("%"):
                continue

            parts = line.split(",")
            if skip_nominal:
                parts = parts[1:]  # drop nominal first column

            # Convert YES/NO to 1/0, otherwise parse as float
            vals = []
            for p in parts:
                p = p.strip()
                if p == "YES":
                    vals.append(1.0)
                elif p == "NO":
                    vals.append(0.0)
                else:
                    vals.append(float(p))

            x = torch.tensor(vals[:num_features], dtype=torch.float32)
            y = torch.tensor(vals[num_features:num_features + num_labels], dtype=torch.float32)
            rows_x.append(x)
            rows_y.append(y)

    return torch.stack(rows_x), torch.stack(rows_y)


def _load_folds(data_dir, fold_prefix, num_features, num_labels, batch_size,
                loader_fn, loader_kwargs=None):
    """Generic 10-fold loader with 6/2/2 train/val/test split."""
    loader_kwargs = loader_kwargs or {}
    TRAIN_FOLDS = list(range(1, 7))
    VAL_FOLDS = [7, 8]
    TEST_FOLDS = [9, 10]

    folds = {}
    for i in range(1, 11):
        path = os.path.join(data_dir, f"{fold_prefix}-fold{i}.arff")
        folds[i] = loader_fn(path, num_features, num_labels, **loader_kwargs)

    train_x = torch.cat([folds[i][0] for i in TRAIN_FOLDS])
    train_y = torch.cat([folds[i][1] for i in TRAIN_FOLDS])
    val_x = torch.cat([folds[i][0] for i in VAL_FOLDS])
    val_y = torch.cat([folds[i][1] for i in VAL_FOLDS])
    test_x = torch.cat([folds[i][0] for i in TEST_FOLDS])
    test_y = torch.cat([folds[i][1] for i in TEST_FOLDS])

    print(f"{fold_prefix} stratified split: train=folds{TRAIN_FOLDS}, "
          f"val=folds{VAL_FOLDS}, test=folds{TEST_FOLDS}")
    print(f"  Sizes: train={train_x.shape[0]}, val={val_x.shape[0]}, test={test_x.shape[0]}")
    print(f"  Features: {num_features}, Labels: {num_labels}")
    print(f"  Avg labels/sample: train={train_y.sum(dim=-1).mean():.1f}, "
          f"val={val_y.sum(dim=-1).mean():.1f}, test={test_y.sum(dim=-1).mean():.1f}")

    train_loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(val_x, val_y), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=False)

    pos_freq = train_y.mean(dim=0)
    pos_weight = ((1.0 - pos_freq) / (pos_freq + 1e-8)).clamp(max=50.0)
    print(f"  Pos weight range: [{pos_weight.min():.1f}, {pos_weight.max():.1f}]")

    return train_loader, val_loader, test_loader, num_features, num_labels, pos_weight


def _load_folds_normalized(data_dir, fold_prefix, num_features, num_labels,
                           batch_size, loader_fn, loader_kwargs=None,
                           file_suffix="-normalised"):
    """10-fold loader variant that loads *-normalised.arff files when available."""
    loader_kwargs = loader_kwargs or {}
    TRAIN_FOLDS = list(range(1, 7))
    VAL_FOLDS = [7, 8]
    TEST_FOLDS = [9, 10]

    folds = {}
    for i in range(1, 11):
        path = os.path.join(
            data_dir, f"{fold_prefix}-fold{i}{file_suffix}.arff"
        )
        folds[i] = loader_fn(path, num_features, num_labels, **loader_kwargs)

    train_x = torch.cat([folds[i][0] for i in TRAIN_FOLDS])
    train_y = torch.cat([folds[i][1] for i in TRAIN_FOLDS])
    val_x = torch.cat([folds[i][0] for i in VAL_FOLDS])
    val_y = torch.cat([folds[i][1] for i in VAL_FOLDS])
    test_x = torch.cat([folds[i][0] for i in TEST_FOLDS])
    test_y = torch.cat([folds[i][1] for i in TEST_FOLDS])

    print(f"{fold_prefix} stratified split (normalized): train=folds{TRAIN_FOLDS}, "
          f"val=folds{VAL_FOLDS}, test=folds{TEST_FOLDS}")
    print(f"  Sizes: train={train_x.shape[0]}, val={val_x.shape[0]}, test={test_x.shape[0]}")
    print(f"  Features: {num_features}, Labels: {num_labels}")
    print(f"  Avg labels/sample: train={train_y.sum(dim=-1).mean():.1f}, "
          f"val={val_y.sum(dim=-1).mean():.1f}, test={test_y.sum(dim=-1).mean():.1f}")

    train_loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(val_x, val_y), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=False)

    pos_freq = train_y.mean(dim=0)
    pos_weight = ((1.0 - pos_freq) / (pos_freq + 1e-8)).clamp(max=50.0)
    print(f"  Pos weight range: [{pos_weight.min():.1f}, {pos_weight.max():.1f}]")

    return train_loader, val_loader, test_loader, num_features, num_labels, pos_weight


def load_bibtex(data_dir: str, batch_size: int = 32):
    """Load bibtex dataset (1836 features, 159 labels, sparse ARFF, 10 folds)."""
    return _load_folds(data_dir, "Bibtex", 1836, 159, batch_size, load_arff_sparse)


def load_delicious(data_dir: str, batch_size: int = 32):
    """Load delicious dataset (500 features, 983 labels, sparse ARFF, 10 folds)."""
    return _load_folds(data_dir, "Delicious", 500, 983, batch_size, load_arff_sparse)


def load_genbase(data_dir: str, batch_size: int = 32):
    """Load genbase dataset (1185 YES/NO features, 27 labels, dense ARFF, 10 folds).
    First attribute (protein name) is skipped."""
    return _load_folds(data_dir, "Genbase", 1185, 27, batch_size,
                       load_arff_dense, loader_kwargs={"skip_nominal": True})


def load_eurlex_ev(data_dir: str, batch_size: int = 32):
    """Load Eurlex-ev dataset (5000 features, 3993 labels, sparse ARFF, 10 folds, normalized)."""
    return _load_folds_normalized(
        data_dir, "Eurlex-ev", 5000, 3993, batch_size, load_arff_sparse
    )


def load_cal500(data_dir: str, batch_size: int = 32):
    """Load CAL500 dataset (68 features, 174 labels, dense ARFF, 10 folds, normalized)."""
    return _load_folds_normalized(
        data_dir, "CAL500", 68, 174, batch_size, load_arff_dense
    )


def load_expr_fun(data_dir: str, batch_size: int = 32):
    """Load expr_fun dataset (561 features, 500 labels, dense numeric ARFF, pre-split)."""
    NUM_FEATURES = 561
    NUM_LABELS = 500

    train_x, train_y = load_arff_dense(
        os.path.join(data_dir, "train-normalized.arff"), NUM_FEATURES, NUM_LABELS)
    val_x, val_y = load_arff_dense(
        os.path.join(data_dir, "dev-normalized.arff"), NUM_FEATURES, NUM_LABELS)
    test_x, test_y = load_arff_dense(
        os.path.join(data_dir, "test-normalized.arff"), NUM_FEATURES, NUM_LABELS)

    print(f"expr_fun pre-split: train/dev/test")
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

    pos_freq = train_y.mean(dim=0)
    pos_weight = ((1.0 - pos_freq) / (pos_freq + 1e-8)).clamp(max=50.0)
    print(f"  Pos weight range: [{pos_weight.min():.1f}, {pos_weight.max():.1f}]")

    return train_loader, val_loader, test_loader, NUM_FEATURES, NUM_LABELS, pos_weight


def load_spo_fun(data_dir: str, batch_size: int = 32):
    """Load spo_fun dataset (86 features, 500 labels, dense numeric ARFF, pre-split, normalized)."""
    NUM_FEATURES = 86
    NUM_LABELS = 500

    train_x, train_y = load_arff_dense(
        os.path.join(data_dir, "train-normalized.arff"), NUM_FEATURES, NUM_LABELS)
    val_x, val_y = load_arff_dense(
        os.path.join(data_dir, "dev-normalized.arff"), NUM_FEATURES, NUM_LABELS)
    test_x, test_y = load_arff_dense(
        os.path.join(data_dir, "test-normalized.arff"), NUM_FEATURES, NUM_LABELS)

    print(f"spo_fun pre-split (normalized): train/dev/test")
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

    pos_freq = train_y.mean(dim=0)
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
