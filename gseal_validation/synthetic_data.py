"""
Synthetic Constrained Sequence Dataset for G-SEAL Validation.

Structural constraints:
  - Tag A (0) must be immediately followed by tag B (1)
  - Tag C (2) must be immediately followed by tag D (3)
  - B and D can be followed by anything
  - Last position has no constraint on what follows

Input features correlate with correct tags but do NOT encode transition constraints,
so a feedforward model can learn per-position predictions (~70-80%) but will violate
structural constraints (~15-25% of sequences).
"""

import torch
import numpy as np
from typing import Tuple, Dict


TAGS = {"A": 0, "B": 1, "C": 2, "D": 3}
NUM_TAGS = 4

# Transition constraints: which tags MUST follow a given tag
# None means any tag is allowed
FORCED_NEXT = {0: 1, 2: 3}  # A->B, C->D


def count_violations(sequences: torch.Tensor) -> Tuple[int, int]:
    """Count structural violations in a batch of tag sequences.

    Args:
        sequences: (batch, seq_len) integer tag indices

    Returns:
        (total_violations, num_sequences_with_violations)
    """
    batch, seq_len = sequences.shape
    total_violations = 0
    sequences_with_violations = 0

    for i in range(batch):
        has_violation = False
        for t in range(seq_len - 1):
            tag = sequences[i, t].item()
            next_tag = sequences[i, t + 1].item()
            if tag in FORCED_NEXT and next_tag != FORCED_NEXT[tag]:
                total_violations += 1
                has_violation = True
        if has_violation:
            sequences_with_violations += 1

    return total_violations, sequences_with_violations


def violation_rate(sequences: torch.Tensor) -> float:
    """Fraction of sequences that contain at least one violation."""
    _, num_violated = count_violations(sequences)
    return num_violated / sequences.shape[0]


def generate_dataset(
    num_samples: int,
    seq_len: int = 10,
    input_dim: int = 16,
    noise_scale: float = 2.0,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """Generate synthetic constrained sequence dataset.

    Each input feature vector x_t is constructed to correlate with the correct
    tag y_t via a fixed projection matrix, but does NOT encode transition info.

    Args:
        num_samples: number of sequences to generate
        seq_len: length of each sequence
        input_dim: dimensionality of input features
        noise_scale: noise added to input features (higher = harder per-position task)
        seed: random seed

    Returns:
        dict with keys: "inputs" (num_samples, seq_len, input_dim),
                        "labels" (num_samples, seq_len) integer tags
    """
    rng = np.random.RandomState(seed)

    # Fixed projection from tags to input features (not learned, just for data generation)
    # IMPORTANT: Use a fixed seed (0) for prototypes so train/val/test share the same
    # feature-to-label mapping. Only data generation randomness varies by seed.
    proto_rng = np.random.RandomState(0)
    tag_prototypes = proto_rng.randn(NUM_TAGS, input_dim).astype(np.float32)
    # Scale prototypes so per-position classification is learnable but imperfect.
    # With noise_scale=1.0 (default) and prototype_scale=1.0, the MLP should get
    # ~70-80% per-position accuracy, creating 15-25% sequence violation rate.
    tag_prototypes *= 1.0

    all_inputs = []
    all_labels = []

    for _ in range(num_samples):
        tags = []
        for t in range(seq_len):
            if t > 0 and tags[t - 1] in FORCED_NEXT:
                # Forced transition
                tag = FORCED_NEXT[tags[t - 1]]
            else:
                # Free choice — weighted by a simple heuristic based on position
                # Use position-dependent bias to create variety
                weights = np.ones(NUM_TAGS)
                # Increase probability of A and C to ensure constraints are exercised
                weights[0] = 1.5  # A
                weights[2] = 1.5  # C
                # Avoid A/C at second-to-last position to prevent forced last tag
                if t == seq_len - 2:
                    weights[0] = 0.5
                    weights[2] = 0.5
                weights /= weights.sum()
                tag = rng.choice(NUM_TAGS, p=weights)
            tags.append(tag)

        tags = np.array(tags)

        # Generate input features: prototype of correct tag + noise
        inputs = tag_prototypes[tags] + rng.randn(seq_len, input_dim).astype(np.float32) * noise_scale

        all_inputs.append(inputs)
        all_labels.append(tags)

    inputs_tensor = torch.tensor(np.stack(all_inputs), dtype=torch.float32)
    labels_tensor = torch.tensor(np.stack(all_labels), dtype=torch.long)

    return {"inputs": inputs_tensor, "labels": labels_tensor}


def corrupt_sequence(
    labels: torch.Tensor, corruption_type: str = "structural", rng=None
) -> torch.Tensor:
    """Create corrupted versions of valid sequences.

    Args:
        labels: (batch, seq_len) valid tag sequences
        corruption_type: "structural" (violate constraints), "random" (random tags)
        rng: numpy random state

    Returns:
        corrupted: (batch, seq_len) corrupted sequences
    """
    if rng is None:
        rng = np.random.RandomState()

    batch, seq_len = labels.shape
    corrupted = labels.clone()

    if corruption_type == "random":
        return torch.randint(0, NUM_TAGS, (batch, seq_len), dtype=torch.long)

    elif corruption_type == "structural":
        # Specifically violate the A->B and C->D constraints
        for i in range(batch):
            # Find positions where constraints exist
            constrained_positions = []
            for t in range(seq_len - 1):
                tag = corrupted[i, t].item()
                if tag in FORCED_NEXT:
                    constrained_positions.append(t)

            if len(constrained_positions) == 0:
                # No constraints to violate — make a random swap
                t = rng.randint(0, seq_len)
                corrupted[i, t] = rng.randint(0, NUM_TAGS)
            else:
                # Violate 1-2 constraints
                num_to_violate = min(len(constrained_positions), rng.randint(1, 3))
                positions = rng.choice(constrained_positions, num_to_violate, replace=False)
                for t in positions:
                    tag = corrupted[i, t].item()
                    correct_next = FORCED_NEXT[tag]
                    # Pick any tag EXCEPT the correct one
                    wrong_tags = [j for j in range(NUM_TAGS) if j != correct_next]
                    corrupted[i, t + 1] = rng.choice(wrong_tags)

        return corrupted

    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")


def labels_to_onehot(labels: torch.Tensor) -> torch.Tensor:
    """Convert integer labels to one-hot soft representation.

    Args:
        labels: (batch, seq_len) integer tags

    Returns:
        onehot: (batch, seq_len, num_tags) float tensor
    """
    return torch.nn.functional.one_hot(labels, NUM_TAGS).float()


if __name__ == "__main__":
    # Quick sanity check
    data = generate_dataset(1000, seq_len=10, seed=42)
    print(f"Inputs shape: {data['inputs'].shape}")
    print(f"Labels shape: {data['labels'].shape}")

    # Verify all generated sequences are valid
    violations, violated_seqs = count_violations(data["labels"])
    print(f"Violations in generated data: {violations} (should be 0)")
    print(f"Sequences with violations: {violated_seqs} (should be 0)")

    # Check constraint frequency
    labels = data["labels"].numpy()
    a_count = (labels == 0).sum()
    b_count = (labels == 1).sum()
    c_count = (labels == 2).sum()
    d_count = (labels == 3).sum()
    print(f"Tag distribution: A={a_count}, B={b_count}, C={c_count}, D={d_count}")

    # Test corruption
    corrupted = corrupt_sequence(data["labels"][:100], "structural")
    violations, violated_seqs = count_violations(corrupted)
    print(f"\nAfter structural corruption:")
    print(f"Violations: {violations}, Sequences with violations: {violated_seqs}/100")

    corrupted_random = corrupt_sequence(data["labels"][:100], "random")
    violations, violated_seqs = count_violations(corrupted_random)
    print(f"\nAfter random corruption:")
    print(f"Violations: {violations}, Sequences with violations: {violated_seqs}/100")
