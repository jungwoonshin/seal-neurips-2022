"""DTD Type Gate: learnable per-label type assignment.

Maintains a (L, 3) parameter tensor whose softmax gives per-label probabilities
for Type A (deterministic), Type B (input-mediated), and Type C (spurious).
"""

from typing import Optional
import torch
import torch.nn as nn


class DTDTypeGate(nn.Module):
    """Learnable gate that assigns each label a soft type distribution.

    Parameters: 3 * num_labels (e.g., 477 for bibtex with 159 labels).
    """

    TYPE_A = 0
    TYPE_B = 1
    TYPE_C = 2

    def __init__(self, num_labels: int, q_init: Optional[torch.Tensor] = None):
        super().__init__()
        if q_init is not None:
            assert q_init.shape == (num_labels, 3)
            self.q = nn.Parameter(q_init.clone())
        else:
            self.q = nn.Parameter(torch.zeros(num_labels, 3))

    @property
    def type_probs(self) -> torch.Tensor:
        """(L, 3) soft type probabilities via softmax."""
        return torch.softmax(self.q, dim=-1)

    def gate_a(self, i_indices: torch.Tensor, k_indices: torch.Tensor) -> torch.Tensor:
        """Compute Type A gate for given label pairs.

        Args:
            i_indices: (num_pairs,) label indices
            k_indices: (num_pairs,) label indices

        Returns:
            (num_pairs,) gate values = t_i^A * t_k^A
        """
        probs = self.type_probs  # (L, 3)
        t_a_i = probs[i_indices, self.TYPE_A]  # (num_pairs,)
        t_a_k = probs[k_indices, self.TYPE_A]  # (num_pairs,)
        return t_a_i * t_a_k

    def gate_b_vector(self) -> torch.Tensor:
        """Return Type B probability per label.

        Returns:
            (L,) vector of t^B values
        """
        return self.type_probs[:, self.TYPE_B]
