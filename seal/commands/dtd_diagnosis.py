"""
Offline dependency diagnosis for DTD (Dependency Type Decomposition).

Analyzes label co-occurrence in training data to:
1. Compute a co-occurrence matrix and SVD label embeddings
2. Estimate conditional lift D_{ik} and its variance across bootstrap samples
3. Classify label pairs into Type A (deterministic), Type B (input-mediated), Type C (spurious)
4. Initialize per-label type vectors q_i and Type A pair parameters

Usage:
    python -m seal.commands.dtd_diagnosis \
        --data-pattern "./data/bibtex_stratified10folds_meka/Bibtex-fold@(1|2|3|4|5|6).arff" \
        --num-labels 159 \
        --svd-rank 32 \
        --num-type-a-pairs 50 \
        --output diagnosis.json
"""

import argparse
import json
import logging
import sys
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


def load_label_matrix_from_arff(
    data_pattern: str, num_labels: int
) -> np.ndarray:
    """Load binary label matrix Y (N, L) from ARFF files matching a glob pattern."""
    from skmultilearn.dataset import load_from_arff
    from wcmatch import glob

    all_y = []
    for file_ in glob.glob(data_pattern, flags=glob.EXTGLOB):
        logger.info(f"Reading {file_}")
        x, y, feature_names, label_names = load_from_arff(
            file_,
            label_count=num_labels,
            return_attribute_definitions=True,
        )
        all_y.append(y.toarray())
    if not all_y:
        raise ValueError(f"No files matched pattern: {data_pattern}")
    Y = np.vstack(all_y).astype(np.float64)
    logger.info(f"Loaded label matrix: {Y.shape[0]} instances, {Y.shape[1]} labels")
    return Y


def compute_cooccurrence_matrix(Y: np.ndarray) -> np.ndarray:
    """Compute pointwise mutual information style co-occurrence: P(y_i=1, y_k=1) - P(y_i=1)*P(y_k=1)."""
    N = Y.shape[0]
    # P(y_i=1)
    p = Y.mean(axis=0)  # (L,)
    # P(y_i=1, y_k=1) = Y^T Y / N
    joint = (Y.T @ Y) / N  # (L, L)
    # Outer product of marginals
    outer = np.outer(p, p)  # (L, L)
    C = joint - outer
    return C


def compute_svd_embeddings(
    C: np.ndarray, svd_rank: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute top-d SVD embeddings from co-occurrence matrix.

    Returns:
        embeddings: (L, svd_rank) — label embeddings (U * sqrt(S))
        singular_values: (svd_rank,) — top singular values
    """
    from numpy.linalg import svd

    # Symmetric matrix, use full SVD and take top-d
    U, S, Vt = svd(C, full_matrices=False)
    d = min(svd_rank, len(S))
    embeddings = U[:, :d] * np.sqrt(S[:d][np.newaxis, :])
    return embeddings, S[:d]


def compute_conditional_lift(Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute D_{ik} = P(y_k=1|y_i=1) - P(y_k=1|y_i=0) for all pairs.

    Returns:
        D_mean: (L, L) — conditional lift matrix
        D_valid: (L, L) — boolean mask where both conditionals are well-defined
    """
    N, L = Y.shape
    eps = 1e-10

    # Count occurrences
    n_pos = Y.sum(axis=0)  # (L,) how many times each label is 1
    n_neg = N - n_pos  # (L,) how many times each label is 0

    # Co-occurrence: n_pos_ik = number of instances where both y_i=1 and y_k=1
    cooccur = Y.T @ Y  # (L, L)

    # P(y_k=1 | y_i=1) = cooccur[i,k] / n_pos[i]
    # P(y_k=1 | y_i=0) = (n_pos[k] - cooccur[i,k]) / n_neg[i]
    with np.errstate(divide="ignore", invalid="ignore"):
        p_k_given_i_pos = cooccur / (n_pos[:, np.newaxis] + eps)  # (L, L)
        p_k_given_i_neg = (
            n_pos[np.newaxis, :] - cooccur
        ) / (n_neg[:, np.newaxis] + eps)  # (L, L)

    D_mean = p_k_given_i_pos - p_k_given_i_neg

    # Valid where both conditionals have sufficient support
    min_support = 5
    D_valid = (n_pos[:, np.newaxis] >= min_support) & (
        n_neg[:, np.newaxis] >= min_support
    )

    return D_mean, D_valid


def bootstrap_lift_variance(
    Y: np.ndarray, num_bootstrap: int = 50, seed: int = 42
) -> np.ndarray:
    """Estimate variance of conditional lift across bootstrap samples.

    Returns:
        D_var: (L, L) — variance of conditional lift
    """
    rng = np.random.RandomState(seed)
    N, L = Y.shape
    lifts = []

    for _ in range(num_bootstrap):
        idx = rng.choice(N, size=N, replace=True)
        Y_b = Y[idx]
        D_b, _ = compute_conditional_lift(Y_b)
        lifts.append(D_b)

    lifts = np.stack(lifts, axis=0)  # (num_bootstrap, L, L)
    D_var = np.var(lifts, axis=0)  # (L, L)
    return D_var


def classify_dependencies(
    D_mean: np.ndarray,
    D_var: np.ndarray,
    D_valid: np.ndarray,
    num_type_a_pairs: int,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Classify label dependencies and select Type A pairs.

    Returns:
        q_init: (L, 3) — initial type logits [A, B, C]
        type_a_pairs: list of dicts with pair info and initialization
    """
    L = D_mean.shape[0]

    # Compute per-pair signal-to-noise ratio
    eps = 1e-10
    snr = np.abs(D_mean) / (np.sqrt(D_var) + eps)  # (L, L)
    snr[~D_valid] = 0.0
    # Zero out diagonal
    np.fill_diagonal(snr, 0.0)
    np.fill_diagonal(D_mean, 0.0)

    # Adaptive thresholds based on distribution percentiles
    valid_snr = snr[D_valid & ~np.eye(L, dtype=bool)]
    valid_var = D_var[D_valid & ~np.eye(L, dtype=bool)]
    valid_abs_lift = np.abs(D_mean[D_valid & ~np.eye(L, dtype=bool)])

    if len(valid_snr) == 0:
        logger.warning("No valid label pairs found, using defaults")
        q_init = np.zeros((L, 3), dtype=np.float64)
        q_init[:, 1] = 1.0  # Default to Type B
        return q_init, []

    # Type A: high SNR (low variance relative to mean) and strong lift
    snr_threshold = np.percentile(valid_snr, 90)
    lift_threshold = np.percentile(valid_abs_lift, 75)

    # Type C: low absolute lift (near-zero signal)
    lift_c_threshold = np.percentile(valid_abs_lift, 25)

    # Select Type A pairs: ranked by SNR, above thresholds
    type_a_mask = (snr >= snr_threshold) & (np.abs(D_mean) >= lift_threshold) & D_valid
    np.fill_diagonal(type_a_mask, False)

    # Get candidate pairs sorted by SNR
    candidates_i, candidates_k = np.where(type_a_mask)
    candidate_snrs = snr[type_a_mask]
    sorted_idx = np.argsort(-candidate_snrs)  # Descending

    # Take top num_type_a_pairs
    n_pairs = min(num_type_a_pairs, len(sorted_idx))
    selected_idx = sorted_idx[:n_pairs]

    type_a_pairs = []
    type_a_labels = set()
    for idx in selected_idx:
        i, k = int(candidates_i[idx]), int(candidates_k[idx])
        d_val = float(D_mean[i, k])
        type_a_pairs.append({
            "i": i,
            "k": k,
            "w_init": float(np.sign(d_val) * min(abs(d_val) * 2.0, 5.0)),
            "mu_init": 0.0,
            "log_tau_init": 0.0,
            "D_mean": d_val,
            "D_var": float(D_var[i, k]),
            "snr": float(snr[i, k]),
        })
        type_a_labels.add(i)
        type_a_labels.add(k)

    # Per-label type assignment based on aggregate statistics
    q_init = np.zeros((L, 3), dtype=np.float64)
    for label_idx in range(L):
        # Check if this label participates in Type A pairs
        label_snrs = snr[label_idx, :]
        label_snrs[label_idx] = 0.0
        max_snr = label_snrs.max() if D_valid[label_idx].any() else 0.0

        label_lifts = np.abs(D_mean[label_idx, :])
        label_lifts[label_idx] = 0.0
        max_lift = label_lifts.max() if D_valid[label_idx].any() else 0.0

        if label_idx in type_a_labels:
            # Strong Type A signal
            q_init[label_idx] = [2.0, 0.0, -1.0]
        elif max_lift < lift_c_threshold and max_snr < np.percentile(valid_snr, 50):
            # Weak signal — likely Type C
            q_init[label_idx] = [-1.0, 0.0, 2.0]
        else:
            # Default: Type B (input-mediated)
            q_init[label_idx] = [0.0, 1.0, 0.0]

    return q_init, type_a_pairs


def run_diagnosis(
    data_pattern: str,
    num_labels: int,
    svd_rank: int = 32,
    num_type_a_pairs: int = 50,
    num_bootstrap: int = 50,
    output_path: str = "dtd_diagnosis.json",
) -> Dict[str, Any]:
    """Run the full DTD diagnosis pipeline."""
    logger.info("Phase 0: DTD Dependency Diagnosis")

    # Step 1: Load data
    logger.info("Step 1: Loading label matrix...")
    Y = load_label_matrix_from_arff(data_pattern, num_labels)
    N, L = Y.shape
    assert L == num_labels, f"Expected {num_labels} labels, got {L}"

    # Step 2: Co-occurrence matrix
    logger.info("Step 2: Computing co-occurrence matrix...")
    C = compute_cooccurrence_matrix(Y)

    # Step 3: SVD embeddings
    logger.info(f"Step 3: SVD decomposition (rank={svd_rank})...")
    embeddings, singular_values = compute_svd_embeddings(C, svd_rank)

    # Step 4: Conditional lift
    logger.info("Step 4: Computing conditional lift...")
    D_mean, D_valid = compute_conditional_lift(Y)

    # Step 5: Bootstrap variance
    logger.info(f"Step 5: Bootstrap variance estimation ({num_bootstrap} samples)...")
    D_var = bootstrap_lift_variance(Y, num_bootstrap=num_bootstrap)

    # Step 6: Classify and select pairs
    logger.info(f"Step 6: Classifying dependencies (max {num_type_a_pairs} Type A pairs)...")
    q_init, type_a_pairs = classify_dependencies(
        D_mean, D_var, D_valid, num_type_a_pairs
    )

    # Summary
    type_counts = {
        "A": int((q_init.argmax(axis=1) == 0).sum()),
        "B": int((q_init.argmax(axis=1) == 1).sum()),
        "C": int((q_init.argmax(axis=1) == 2).sum()),
    }
    logger.info(
        f"Type classification: A={type_counts['A']}, B={type_counts['B']}, C={type_counts['C']}"
    )
    logger.info(f"Selected {len(type_a_pairs)} Type A pairs")

    # Build output
    result = {
        "num_labels": num_labels,
        "svd_rank": int(embeddings.shape[1]),
        "num_instances": N,
        "type_counts": type_counts,
        "q_init": q_init.tolist(),
        "svd_embeddings": embeddings.tolist(),
        "svd_singular_values": singular_values.tolist(),
        "type_a_pairs": type_a_pairs,
    }

    # Save
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Diagnosis saved to {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="DTD Dependency Diagnosis for SEAL"
    )
    parser.add_argument(
        "--data-pattern",
        type=str,
        required=True,
        help="Glob pattern for ARFF training data files",
    )
    parser.add_argument(
        "--num-labels", type=int, required=True, help="Number of labels"
    )
    parser.add_argument(
        "--svd-rank", type=int, default=32, help="SVD embedding dimension"
    )
    parser.add_argument(
        "--num-type-a-pairs",
        type=int,
        default=50,
        help="Maximum number of Type A pairs to select",
    )
    parser.add_argument(
        "--num-bootstrap",
        type=int,
        default=50,
        help="Number of bootstrap samples for variance estimation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dtd_diagnosis.json",
        help="Output JSON file path",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_diagnosis(
        data_pattern=args.data_pattern,
        num_labels=args.num_labels,
        svd_rank=args.svd_rank,
        num_type_a_pairs=args.num_type_a_pairs,
        num_bootstrap=args.num_bootstrap,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
