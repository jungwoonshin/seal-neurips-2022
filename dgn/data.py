"""Standalone ARFF data loader for multi-label classification.

Loads bibtex-style ARFF files without depending on AllenNLP.
"""

import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def load_arff(filepath: str, num_labels: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load a single ARFF file into dense numpy arrays.

    Args:
        filepath: Path to .arff file.
        num_labels: Number of label attributes (from the end).

    Returns:
        X: (N, D) float32 feature matrix.
        Y: (N, L) float32 label matrix.
    """
    try:
        from skmultilearn.dataset import load_from_arff
        from scipy.sparse import issparse

        x_raw, y_raw, _, _ = load_from_arff(
            filepath, label_count=num_labels, load_sparse=False,
            return_attribute_definitions=True,
        )
        X = x_raw.toarray().astype(np.float32) if issparse(x_raw) else np.array(x_raw, dtype=np.float32)
        Y = y_raw.toarray().astype(np.float32) if issparse(y_raw) else np.array(y_raw, dtype=np.float32)
        return X, Y
    except ImportError:
        pass

    # Fallback: manual sparse ARFF parsing
    with open(filepath, "r") as f:
        lines = f.readlines()

    # Find @data section
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
        raise ValueError(f"No @data section found in {filepath}")

    num_features = num_attrs - num_labels

    rows_x = []
    rows_y = []
    for line in lines[data_start:]:
        line = line.strip()
        if not line or line.startswith("%"):
            continue

        # Parse sparse format: {idx val, idx val, ...}
        x_row = np.zeros(num_features, dtype=np.float32)
        y_row = np.zeros(num_labels, dtype=np.float32)

        content = line.strip("{} \n")
        if not content:
            rows_x.append(x_row)
            rows_y.append(y_row)
            continue

        for token in content.split(","):
            token = token.strip()
            if not token:
                continue
            parts = token.split()
            idx = int(parts[0])
            val = float(parts[1])
            if idx < num_features:
                x_row[idx] = val
            else:
                y_row[idx - num_features] = val

        rows_x.append(x_row)
        rows_y.append(y_row)

    return np.array(rows_x, dtype=np.float32), np.array(rows_y, dtype=np.float32)


def load_arff_glob(pattern: str, num_labels: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load multiple ARFF files matching a fold pattern.

    Supports patterns like: ./data/dir/Bibtex-fold@(1|2|3).arff
    which expands to fold1, fold2, fold3.

    Args:
        pattern: Glob pattern with @(...) fold syntax.
        num_labels: Number of label attributes.

    Returns:
        X: (N_total, D) concatenated features.
        Y: (N_total, L) concatenated labels.
    """
    # Parse fold pattern: path/Bibtex-fold@(1|2|3).arff
    match = re.search(r"@\(([^)]+)\)", pattern)
    if match:
        fold_ids = match.group(1).split("|")
        base = pattern[: match.start()] + "{}" + pattern[match.end() :]
        files = [base.format(fid) for fid in fold_ids]
    else:
        files = [pattern]

    all_x, all_y = [], []
    for fp in files:
        p = Path(fp)
        if not p.exists():
            raise FileNotFoundError(f"Data file not found: {fp}")
        x, y = load_arff(str(p), num_labels)
        all_x.append(x)
        all_y.append(y)

    return np.concatenate(all_x, axis=0), np.concatenate(all_y, axis=0)


class MultiLabelDataset(Dataset):
    """Simple multi-label dataset wrapping numpy arrays."""

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def make_dataloaders(
    train_pattern: str,
    val_pattern: str,
    num_labels: int,
    batch_size: int = 32,
    test_pattern: str = None,
) -> dict:
    """Build train/val/test DataLoaders from ARFF fold patterns.

    Returns:
        Dict with keys "train", "val", and optionally "test".
    """
    X_train, Y_train = load_arff_glob(train_pattern, num_labels)
    X_val, Y_val = load_arff_glob(val_pattern, num_labels)

    loaders = {
        "train": DataLoader(
            MultiLabelDataset(X_train, Y_train),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        ),
        "val": DataLoader(
            MultiLabelDataset(X_val, Y_val),
            batch_size=batch_size,
            shuffle=False,
        ),
    }

    if test_pattern:
        X_test, Y_test = load_arff_glob(test_pattern, num_labels)
        loaders["test"] = DataLoader(
            MultiLabelDataset(X_test, Y_test),
            batch_size=batch_size,
            shuffle=False,
        )

    print(f"Data loaded: train={len(X_train)}, val={len(X_val)}"
          + (f", test={len(X_test)}" if test_pattern else ""))

    return loaders
