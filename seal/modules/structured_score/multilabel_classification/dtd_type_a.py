"""DTD Type A Energy: deterministic label-pair dependencies.

Computes:
    E^A(y) = sum_{(i,k) in A} g^A_ik * w_ik * softplus((y_i - y_k - mu_ik) / tau_ik)

where g^A_ik is the Type A gate value for the pair.
"""

from typing import Dict, Any, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from seal.modules.structured_score import StructuredScore
from .dtd_type_gate import DTDTypeGate


@StructuredScore.register("dtd-type-a")
class DTDTypeAEnergy(StructuredScore):
    """Structured energy for deterministic label-pair dependencies.

    Parameters: 3 * |A| (w, mu, log_tau per pair).
    """

    def __init__(
        self,
        type_gate: DTDTypeGate,
        pair_indices: List[Tuple[int, int]],
        w_init: List[float],
        mu_init: List[float],
        log_tau_init: List[float],
    ):
        super().__init__()
        self.type_gate = type_gate
        num_pairs = len(pair_indices)

        # Store pair indices as buffers (not parameters)
        i_idx = torch.tensor([p[0] for p in pair_indices], dtype=torch.long)
        k_idx = torch.tensor([p[1] for p in pair_indices], dtype=torch.long)
        self.register_buffer("i_indices", i_idx)
        self.register_buffer("k_indices", k_idx)

        # Learnable parameters per pair
        self.w = nn.Parameter(torch.tensor(w_init, dtype=torch.float))
        self.mu = nn.Parameter(torch.tensor(mu_init, dtype=torch.float))
        self.log_tau = nn.Parameter(torch.tensor(log_tau_init, dtype=torch.float))

        assert self.w.shape[0] == num_pairs
        assert self.mu.shape[0] == num_pairs
        assert self.log_tau.shape[0] == num_pairs

    def forward(
        self,
        y: torch.Tensor,  # (batch, num_samples, num_labels)
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        batch, num_samples, num_labels = y.shape

        # Extract label values for each pair
        y_i = y[:, :, self.i_indices]  # (batch, num_samples, num_pairs)
        y_k = y[:, :, self.k_indices]  # (batch, num_samples, num_pairs)

        # Compute softplus penalty per pair
        tau = torch.exp(self.log_tau)  # (num_pairs,)
        diff = (y_i - y_k - self.mu) / (tau + 1e-8)  # (batch, num_samples, num_pairs)
        penalty = F.softplus(diff)  # (batch, num_samples, num_pairs)

        # Apply gate and weight
        gate = self.type_gate.gate_a(self.i_indices, self.k_indices)  # (num_pairs,)
        weighted = gate * self.w * penalty  # (batch, num_samples, num_pairs)

        # Sum over pairs -> (batch, num_samples)
        energy = weighted.sum(dim=-1)
        return energy
