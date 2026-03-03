"""
DTD Type A Energy: Soft Constraints for Deterministic Dependencies.

For deterministic label dependencies (e.g., "Dog" -> "Animal"), enforces
directional constraints with learnable tolerance:

    E^A(y) = sum_{(i,k) in A} g^A_{ik} * w_{ik} * softplus((y_i - y_k - mu_{ik}) / tau_{ik})

Parameters per pair:
    w_{ik}: constraint strength (initialized from mean conditional lift)
    mu_{ik}: learned offset for exception tolerance
    tau_{ik}: learned temperature for enforcement sharpness

Gradient:
    dE^A/dy_i = (g^A_{ik} * w_{ik} / tau_{ik}) * sigmoid((y_i - y_k - mu_{ik}) / tau_{ik})

Always finite, smoothly varying, targeted to specific constraint violations.
"""

from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from seal.modules.structured_score import StructuredScore
from .dtd_type_gate import DTDTypeGate


@StructuredScore.register("dtd-type-a")
class DTDTypeAEnergy(StructuredScore):
    """Type A energy: soft constraint enforcement for deterministic label dependencies."""

    def __init__(
        self,
        type_gate: DTDTypeGate,
        pair_indices_i: List[int],
        pair_indices_k: List[int],
        w_init: Optional[List[float]] = None,
        mu_init: Optional[List[float]] = None,
        log_tau_init: Optional[List[float]] = None,
    ):
        """
        Args:
            type_gate: Shared DTDTypeGate module for computing g^A gates.
            pair_indices_i: Source label indices for Type A pairs.
            pair_indices_k: Target label indices for Type A pairs.
            w_init: Initial constraint weights per pair.
            mu_init: Initial offset values per pair.
            log_tau_init: Initial log-temperature values per pair.
        """
        super().__init__()
        self.type_gate = type_gate
        num_pairs = len(pair_indices_i)
        assert len(pair_indices_k) == num_pairs

        # Register pair indices as buffers (not parameters)
        self.register_buffer(
            "pair_i", torch.tensor(pair_indices_i, dtype=torch.long)
        )
        self.register_buffer(
            "pair_k", torch.tensor(pair_indices_k, dtype=torch.long)
        )

        # Learnable parameters per pair
        if w_init is not None:
            assert len(w_init) == num_pairs
            self.w = nn.Parameter(torch.tensor(w_init, dtype=torch.float32))
        else:
            self.w = nn.Parameter(torch.ones(num_pairs))

        if mu_init is not None:
            assert len(mu_init) == num_pairs
            self.mu = nn.Parameter(torch.tensor(mu_init, dtype=torch.float32))
        else:
            self.mu = nn.Parameter(torch.zeros(num_pairs))

        if log_tau_init is not None:
            assert len(log_tau_init) == num_pairs
            self.log_tau = nn.Parameter(
                torch.tensor(log_tau_init, dtype=torch.float32)
            )
        else:
            self.log_tau = nn.Parameter(torch.zeros(num_pairs))

    @property
    def num_pairs(self) -> int:
        return self.pair_i.shape[0]

    def forward(
        self,
        y: torch.Tensor,  # (batch, num_samples, num_labels)
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute Type A energy for all pairs.

        E^A(y) = sum_{(i,k)} g^A_{ik} * w_{ik} * softplus((y_i - y_k - mu_{ik}) / tau_{ik})

        Args:
            y: Label predictions, shape (batch, num_samples, num_labels)
            buffer: Shared buffer dict.

        Returns:
            Energy tensor of shape (batch, num_samples)
        """
        if self.num_pairs == 0:
            return torch.zeros(
                y.shape[0], y.shape[1], device=y.device, dtype=y.dtype
            )

        # Extract label values for each pair
        # y shape: (batch, num_samples, num_labels)
        y_i = y[:, :, self.pair_i]  # (batch, num_samples, num_pairs)
        y_k = y[:, :, self.pair_k]  # (batch, num_samples, num_pairs)

        # Compute temperature (always positive)
        tau = self.log_tau.exp().clamp(min=1e-4)  # (num_pairs,)

        # Constraint violation: softplus((y_i - y_k - mu) / tau)
        violation = (y_i - y_k - self.mu) / tau  # (batch, num_samples, num_pairs)
        penalty = F.softplus(violation)  # (batch, num_samples, num_pairs)

        # Type A gates from the shared gate module
        g_a = self.type_gate.gate_a(self.pair_i, self.pair_k)  # (num_pairs,)

        # Weighted sum over pairs
        # w * g_a: (num_pairs,), broadcast over batch and samples
        weighted_penalty = penalty * (self.w * g_a)  # (batch, num_samples, num_pairs)

        # Sum over pairs -> (batch, num_samples)
        energy = weighted_penalty.sum(dim=-1)

        return energy
