"""
Synthetic Constrained Sequence Dataset v2 for G-SEAL Validation.

Designed to test constraints that CRFs CANNOT express:

Structural constraints:
  1. GLOBAL COUNT: Each sequence must contain exactly K A's (default K=2)
  2. SEPARATION: A's cannot be adjacent
  3. LOCAL PREDICTION: B, C, D are predicted from input features (learnable)

Why this design:
  - A feedforward model can learn per-position B/C/D predictions but can't
    enforce "exactly 2 A's" globally
  - A CRF can discourage adjacent A's (pairwise) but CANNOT enforce exact
    count (global constraint)
  - G-SEAL's transformer structure network CAN learn to count A's and
    penalize wrong counts — this is the genuine advantage

Tags: A=0, B=1, C=2, D=3
"""

import torch
import numpy as np
from typing import Tuple, Dict, List


TAGS = {"A": 0, "B": 1, "C": 2, "D": 3}
NUM_TAGS = 4
REQUIRED_A_COUNT = 2  # Global counting constraint


def count_violations(sequences: torch.Tensor) -> Dict[str, int]:
    """Count structural violations in a batch of tag sequences.

    Returns dict with:
      - count_violations: sequences with wrong number of A's
      - adjacency_violations: sequences with adjacent A's
      - any_violation: sequences with either violation
      - total_count_off: sum of |num_A - REQUIRED_A_COUNT| across batch
    """
    batch, seq_len = sequences.shape
    count_viols = 0
    adj_viols = 0
    any_viols = 0
    total_count_off = 0

    for i in range(batch):
        seq = sequences[i]
        a_positions = (seq == TAGS["A"]).nonzero(as_tuple=True)[0]
        num_a = len(a_positions)

        has_count_viol = num_a != REQUIRED_A_COUNT
        has_adj_viol = False
        for j in range(len(a_positions) - 1):
            if a_positions[j + 1] - a_positions[j] == 1:
                has_adj_viol = True
                break

        if has_count_viol:
            count_viols += 1
        if has_adj_viol:
            adj_viols += 1
        if has_count_viol or has_adj_viol:
            any_viols += 1
        total_count_off += abs(num_a - REQUIRED_A_COUNT)

    return {
        "count_violations": count_viols,
        "adjacency_violations": adj_viols,
        "any_violation": any_viols,
        "total_count_off": total_count_off,
    }


def violation_rate(sequences: torch.Tensor) -> Dict[str, float]:
    """Fraction of sequences with each violation type."""
    stats = count_violations(sequences)
    n = sequences.shape[0]
    return {
        "count_viol_rate": stats["count_violations"] / n,
        "adj_viol_rate": stats["adjacency_violations"] / n,
        "any_viol_rate": stats["any_violation"] / n,
        "avg_count_off": stats["total_count_off"] / n,
    }


def _generate_valid_sequence(seq_len: int, rng: np.random.RandomState) -> np.ndarray:
    """Generate a single valid tag sequence satisfying all constraints.

    Places exactly REQUIRED_A_COUNT A's at non-adjacent positions,
    fills the rest with B, C, D.
    """
    while True:
        # Choose positions for A's
        a_positions = sorted(rng.choice(seq_len, REQUIRED_A_COUNT, replace=False))

        # Check adjacency constraint
        valid = True
        for j in range(len(a_positions) - 1):
            if a_positions[j + 1] - a_positions[j] == 1:
                valid = False
                break
        if valid:
            break

    tags = np.zeros(seq_len, dtype=np.int64)
    tags[a_positions] = TAGS["A"]

    # Fill non-A positions with B, C, D based on position features
    non_a_tags = [TAGS["B"], TAGS["C"], TAGS["D"]]
    for t in range(seq_len):
        if t not in a_positions:
            tags[t] = rng.choice(non_a_tags)

    return tags


def generate_dataset(
    num_samples: int,
    seq_len: int = 10,
    input_dim: int = 16,
    noise_scale: float = 2.0,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """Generate synthetic constrained sequence dataset.

    Input features correlate with the correct tag at each position (so
    per-position prediction is learnable) but do NOT encode global count
    or adjacency information.

    Args:
        num_samples: number of sequences
        seq_len: length of each sequence
        input_dim: input feature dimensionality
        noise_scale: noise magnitude (higher = harder per-position task)
        seed: random seed for data generation

    Returns:
        dict with "inputs" (num_samples, seq_len, input_dim),
                   "labels" (num_samples, seq_len)
    """
    rng = np.random.RandomState(seed)

    # Fixed prototypes shared across all splits (seed=0)
    proto_rng = np.random.RandomState(0)
    tag_prototypes = proto_rng.randn(NUM_TAGS, input_dim).astype(np.float32)

    all_inputs = []
    all_labels = []

    for _ in range(num_samples):
        tags = _generate_valid_sequence(seq_len, rng)
        inputs = tag_prototypes[tags] + rng.randn(seq_len, input_dim).astype(np.float32) * noise_scale
        all_inputs.append(inputs)
        all_labels.append(tags)

    return {
        "inputs": torch.tensor(np.stack(all_inputs), dtype=torch.float32),
        "labels": torch.tensor(np.stack(all_labels), dtype=torch.long),
    }


def corrupt_sequence(
    labels: torch.Tensor, corruption_type: str = "count", rng=None
) -> torch.Tensor:
    """Create corrupted sequences that violate specific constraints.

    corruption_type:
      - "count": Change the number of A's (add or remove one)
      - "adjacency": Make two A's adjacent
      - "random": Fully random tags
      - "mixed": Random mix of count and adjacency corruptions
    """
    if rng is None:
        rng = np.random.RandomState()

    batch, seq_len = labels.shape
    corrupted = labels.clone()

    if corruption_type == "random":
        return torch.randint(0, NUM_TAGS, (batch, seq_len), dtype=torch.long)

    for i in range(batch):
        seq = corrupted[i].numpy().copy()
        a_positions = list(np.where(seq == TAGS["A"])[0])
        non_a_positions = list(np.where(seq != TAGS["A"])[0])

        if corruption_type == "count" or (corruption_type == "mixed" and rng.random() < 0.5):
            # Violate count constraint: add or remove an A
            if rng.random() < 0.5 and len(non_a_positions) > 0:
                # Add an extra A
                pos = rng.choice(non_a_positions)
                seq[pos] = TAGS["A"]
            elif len(a_positions) > 0:
                # Remove an A (replace with random non-A)
                pos = rng.choice(a_positions)
                seq[pos] = rng.choice([TAGS["B"], TAGS["C"], TAGS["D"]])

        elif corruption_type == "adjacency" or corruption_type == "mixed":
            # Violate adjacency: move an A next to another A
            if len(a_positions) >= 2:
                # Pick an A and move it next to another A
                src_idx = rng.randint(0, len(a_positions))
                src_pos = a_positions[src_idx]
                # Find another A
                other_positions = [p for p in a_positions if p != src_pos]
                tgt = rng.choice(other_positions)
                # Move src next to tgt
                new_pos = tgt + 1 if tgt + 1 < seq_len and tgt + 1 != src_pos else tgt - 1
                if 0 <= new_pos < seq_len and new_pos != src_pos:
                    seq[src_pos] = rng.choice([TAGS["B"], TAGS["C"], TAGS["D"]])
                    seq[new_pos] = TAGS["A"]
            elif len(a_positions) == 1:
                # Only one A — add another adjacent to it
                pos = a_positions[0]
                adj = pos + 1 if pos + 1 < seq_len else pos - 1
                if 0 <= adj < seq_len:
                    seq[adj] = TAGS["A"]

        corrupted[i] = torch.tensor(seq, dtype=torch.long)

    return corrupted


def labels_to_onehot(labels: torch.Tensor) -> torch.Tensor:
    """Convert integer labels to one-hot. (batch, seq_len) -> (batch, seq_len, NUM_TAGS)"""
    return torch.nn.functional.one_hot(labels, NUM_TAGS).float()


if __name__ == "__main__":
    data = generate_dataset(1000, seq_len=10, seed=42)
    print(f"Inputs shape: {data['inputs'].shape}")
    print(f"Labels shape: {data['labels'].shape}")

    # Verify all generated sequences are valid
    stats = count_violations(data["labels"])
    print(f"Count violations: {stats['count_violations']} (should be 0)")
    print(f"Adjacency violations: {stats['adjacency_violations']} (should be 0)")

    # Check A count distribution
    labels = data["labels"].numpy()
    a_counts = (labels == 0).sum(axis=1)
    print(f"A counts per sequence: min={a_counts.min()}, max={a_counts.max()}, "
          f"mean={a_counts.mean():.1f} (should all be {REQUIRED_A_COUNT})")

    # Tag distribution
    for tag_name, tag_id in TAGS.items():
        print(f"  {tag_name}: {(labels == tag_id).sum()}")

    # Test corruptions
    for ctype in ["count", "adjacency", "random", "mixed"]:
        corrupted = corrupt_sequence(data["labels"][:200], ctype)
        vr = violation_rate(corrupted)
        print(f"\n{ctype} corruption: count_viol={vr['count_viol_rate']:.3f}, "
              f"adj_viol={vr['adj_viol_rate']:.3f}, any={vr['any_viol_rate']:.3f}")
