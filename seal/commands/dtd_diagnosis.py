"""
DTD Diagnostic Script: Analyze label dependencies in multi-label datasets.

Classifies label-pair dependencies into:
  - Type A (deterministic): strong, input-independent co-occurrence patterns
  - Type B (input-mediated): dependencies that vary with input
  - Type C (spurious): low mutual information beyond marginals

Outputs a JSON file with initialization artifacts for DTD energy modules.

Usage:
    python -m seal.commands.dtd_diagnosis \
        --train_data "./data/bibtex_stratified10folds_meka/Bibtex-fold@(1|2|3|4|5|6).arff" \
        --num_labels 159 \
        --output data/bibtex_dtd_diagnosis.json \
        --svd_rank 32 \
        --num_type_a_pairs 200 \
        --num_bootstrap 50
"""

import argparse
import json
import logging
import sys
from typing import List, Tuple, Dict, Any

import numpy as np
from scipy import sparse
from skmultilearn.dataset import load_from_arff
from wcmatch import glob

logger = logging.getLogger(__name__)


def load_label_matrix(file_path: str, num_labels: int) -> np.ndarray:
    """Load label matrix Y of shape (N, L) from ARFF files matching glob pattern."""
    all_y = []
    for file_ in glob.glob(file_path, flags=glob.EXTGLOB):
        logger.info(f"Reading {file_}")
        _, y, _, _ = load_from_arff(
            file_, label_count=num_labels, return_attribute_definitions=True
        )
        y_dense = y.toarray() if sparse.issparse(y) else np.array(y)
        all_y.append(y_dense)

    Y = np.concatenate(all_y, axis=0)  # (N, L)
    # Filter out examples with no labels (matching ARFFReader behavior)
    mask = Y.sum(axis=1) > 0
    Y = Y[mask]
    logger.info(f"Loaded label matrix: {Y.shape[0]} examples, {Y.shape[1]} labels")
    return Y


def compute_cooccurrence_matrix(Y: np.ndarray) -> np.ndarray:
    """Compute label co-occurrence deviation matrix C.

    C_ik = P(y_i=1, y_k=1) - P(y_i=1) * P(y_k=1)
    """
    N = Y.shape[0]
    marginals = Y.mean(axis=0)  # (L,)
    joint = (Y.T @ Y) / N  # (L, L) = P(y_i=1, y_k=1)
    C = joint - np.outer(marginals, marginals)
    return C


def compute_conditional_lift(Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute conditional lift D_ik = P(y_k=1|y_i=1) - P(y_k=1|y_i=0).

    Returns:
        D_mean: (L, L) mean conditional lift
        D_var: (L, L) variance of conditional lift across bootstrap samples
    """
    N, L = Y.shape
    eps = 1e-10

    # Compute full-data conditional lift
    pos_counts = Y.sum(axis=0)  # (L,)
    neg_counts = N - pos_counts  # (L,)
    joint_pos = Y.T @ Y  # (L, L) count of both positive

    # P(y_k=1|y_i=1) for each (i, k)
    p_k_given_i = joint_pos / (pos_counts[:, None] + eps)  # (L, L)

    # P(y_k=1|y_i=0): need count of y_k=1 when y_i=0
    joint_neg = (1 - Y).T @ Y  # (L, L) count of y_k=1 when y_i=0
    p_k_given_not_i = joint_neg / (neg_counts[:, None] + eps)  # (L, L)

    D_mean = p_k_given_i - p_k_given_not_i  # (L, L)

    return D_mean, np.zeros_like(D_mean)  # variance computed via bootstrap


def bootstrap_conditional_lift_variance(
    Y: np.ndarray, num_bootstrap: int = 50, seed: int = 42
) -> np.ndarray:
    """Estimate variance of conditional lift across bootstrap samples."""
    rng = np.random.RandomState(seed)
    N, L = Y.shape
    eps = 1e-10

    D_samples = []
    for _ in range(num_bootstrap):
        indices = rng.choice(N, size=N, replace=True)
        Y_b = Y[indices]
        pos_counts = Y_b.sum(axis=0)
        neg_counts = N - pos_counts
        joint_pos = Y_b.T @ Y_b
        joint_neg = (1 - Y_b).T @ Y_b

        p_k_given_i = joint_pos / (pos_counts[:, None] + eps)
        p_k_given_not_i = joint_neg / (neg_counts[:, None] + eps)
        D_b = p_k_given_i - p_k_given_not_i
        D_samples.append(D_b)

    D_stack = np.stack(D_samples, axis=0)  # (num_bootstrap, L, L)
    D_var = D_stack.var(axis=0)  # (L, L)
    return D_var


def classify_dependencies(
    D_mean: np.ndarray,
    D_var: np.ndarray,
    C: np.ndarray,
    num_type_a_pairs: int = 200,
    type_a_lift_percentile: float = 90.0,
    type_c_mi_percentile: float = 20.0,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Classify label pairs into Type A/B/C.

    Uses data-adaptive percentile thresholds rather than absolute thresholds
    to ensure robustness across datasets with different marginal rates.

    Returns:
        type_a_pairs: list of (i, k) index pairs for Type A
        q_init: (L, 3) initial gate logits [type_A, type_B, type_C]
    """
    L = D_mean.shape[0]

    # Get upper triangle values (excluding diagonal)
    triu_i, triu_k = np.triu_indices(L, k=1)
    abs_D = np.abs(D_mean)
    abs_D_upper = abs_D[triu_i, triu_k]
    D_var_upper = D_var[triu_i, triu_k]

    # Adaptive threshold: Type A candidates must have lift above the percentile
    lift_threshold = np.percentile(abs_D_upper, type_a_lift_percentile)

    # Score each pair: high lift, low variance -> Type A candidate
    # Use signal-to-noise ratio: |D| / (sqrt(var) + eps)
    snr = abs_D_upper / (np.sqrt(D_var_upper) + 1e-8)

    # Filter by lift threshold, then rank by SNR
    type_a_scores = []
    for idx in range(len(triu_i)):
        i, k = int(triu_i[idx]), int(triu_k[idx])
        if abs_D_upper[idx] > lift_threshold:
            type_a_scores.append((i, k, float(snr[idx])))

    # Sort by SNR descending, take top pairs
    type_a_scores.sort(key=lambda x: -x[2])
    type_a_pairs = [(i, k) for i, k, _ in type_a_scores[:num_type_a_pairs]]

    logger.info(
        f"Type A selection: lift_threshold={lift_threshold:.4f}, "
        f"candidates above threshold={len(type_a_scores)}, selected={len(type_a_pairs)}"
    )

    # Determine per-label type affinity for gate initialization
    # Labels involved in many Type A pairs get higher Type A affinity
    type_a_label_counts = np.zeros(L)
    for i, k in type_a_pairs:
        type_a_label_counts[i] += 1
        type_a_label_counts[k] += 1

    # Type C: labels with low total co-occurrence signal (adaptive threshold)
    marginal_signal = np.abs(C).sum(axis=1)  # total co-occurrence signal per label
    type_c_threshold = np.percentile(marginal_signal, type_c_mi_percentile)
    type_c_mask = marginal_signal < type_c_threshold

    # Initialize q logits: (L, 3) = [type_A, type_B, type_C]
    q_init = np.zeros((L, 3))
    for i in range(L):
        if type_c_mask[i]:
            q_init[i] = [-1.0, 0.0, 1.0]  # favor Type C
        elif type_a_label_counts[i] > 0:
            a_strength = min(type_a_label_counts[i] / 5.0, 1.0)
            q_init[i] = [a_strength, 1.0 - a_strength * 0.5, -0.5]
        else:
            q_init[i] = [-0.5, 1.0, -0.5]  # default to Type B

    logger.info(
        f"Type distribution: A-affiliated labels={int((type_a_label_counts > 0).sum())}, "
        f"C labels={int(type_c_mask.sum())}, B labels={int(L - (type_a_label_counts > 0).sum() - type_c_mask.sum())}"
    )

    return type_a_pairs, q_init


def compute_type_a_params(
    D_mean: np.ndarray, type_a_pairs: List[Tuple[int, int]]
) -> List[Dict[str, Any]]:
    """Initialize w, mu, tau parameters for Type A pairs."""
    params = []
    for i, k in type_a_pairs:
        d_ik = D_mean[i, k]
        params.append({
            "i": int(i),
            "k": int(k),
            "w_init": float(np.sign(d_ik) * min(abs(d_ik), 2.0)),
            "mu_init": float(np.sign(d_ik) * 0.5),
            "log_tau_init": 0.0,  # tau = 1.0
        })
    return params


def run_diagnosis(
    train_data: str,
    num_labels: int,
    output: str,
    svd_rank: int = 32,
    num_type_a_pairs: int = 200,
    num_bootstrap: int = 50,
) -> Dict[str, Any]:
    """Run full DTD diagnosis pipeline."""
    logger.info("Loading label matrix...")
    Y = load_label_matrix(train_data, num_labels)
    N, L = Y.shape
    assert L == num_labels, f"Expected {num_labels} labels, got {L}"

    logger.info("Computing co-occurrence matrix...")
    C = compute_cooccurrence_matrix(Y)

    logger.info("Computing SVD embeddings...")
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    U_d = U[:, :svd_rank]  # (L, svd_rank)
    S_d = S[:svd_rank]

    logger.info("Computing conditional lift...")
    D_mean, _ = compute_conditional_lift(Y)

    logger.info(f"Bootstrap variance estimation ({num_bootstrap} samples)...")
    D_var = bootstrap_conditional_lift_variance(Y, num_bootstrap=num_bootstrap)

    logger.info("Classifying dependencies...")
    type_a_pairs, q_init = classify_dependencies(
        D_mean, D_var, C, num_type_a_pairs=num_type_a_pairs
    )
    logger.info(f"Found {len(type_a_pairs)} Type A pairs")

    logger.info("Computing Type A parameters...")
    type_a_params = compute_type_a_params(D_mean, type_a_pairs)

    # Assemble output
    result = {
        "num_labels": L,
        "num_examples": N,
        "svd_rank": svd_rank,
        "cooccurrence_matrix": C.tolist(),
        "svd_embeddings": U_d.tolist(),
        "svd_singular_values": S_d.tolist(),
        "q_init": q_init.tolist(),
        "type_a_pairs": type_a_params,
        "num_type_a_pairs": len(type_a_pairs),
        "label_marginals": Y.mean(axis=0).tolist(),
    }

    logger.info(f"Writing diagnosis to {output}")
    with open(output, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("Done.")
    return result


def main():
    parser = argparse.ArgumentParser(description="DTD Label Dependency Diagnosis")
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="Glob pattern for training ARFF files",
    )
    parser.add_argument(
        "--num_labels", type=int, required=True, help="Number of labels"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/bibtex_dtd_diagnosis.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--svd_rank", type=int, default=32, help="SVD embedding dimension"
    )
    parser.add_argument(
        "--num_type_a_pairs",
        type=int,
        default=200,
        help="Max number of Type A pairs",
    )
    parser.add_argument(
        "--num_bootstrap",
        type=int,
        default=50,
        help="Number of bootstrap samples for variance estimation",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    run_diagnosis(
        train_data=args.train_data,
        num_labels=args.num_labels,
        output=args.output,
        svd_rank=args.svd_rank,
        num_type_a_pairs=args.num_type_a_pairs,
        num_bootstrap=args.num_bootstrap,
    )


if __name__ == "__main__":
    main()
