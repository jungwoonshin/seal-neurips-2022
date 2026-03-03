"""
DTD Type Gate: Learnable per-label type assignment.

Each label i has a learnable type vector t_i = softmax(q_i) in R^3,
representing soft assignment to Type A (deterministic), Type B (input-mediated),
and Type C (spurious).

Pair-level gates are derived through composition rules:
    g^A_{ik} = t_i^A * t_k^A       (both must be Type A)
    g^C_{ik} = max(t_i^C, t_k^C)   (either being spurious contaminates)
    g^B_{ik} = 1 - g^A_{ik} - g^C_{ik}
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


TYPE_A = 0  # Deterministic
TYPE_B = 1  # Input-mediated
TYPE_C = 2  # Spurious


class DTDTypeGate(nn.Module):
    """Learnable per-label soft type assignment with pair-level gate derivation."""

    def __init__(
        self,
        num_labels: int,
        q_init: Optional[torch.Tensor] = None,
        ema_decay: float = 0.99,
    ):
        """
        Args:
            num_labels: Number of labels L.
            q_init: Optional (L, 3) tensor for initialization. If None, defaults to uniform.
            ema_decay: Decay rate for EMA statistics used in dynamic refinement.
        """
        super().__init__()
        self.num_labels = num_labels

        if q_init is not None:
            assert q_init.shape == (num_labels, 3)
            self.q = nn.Parameter(q_init.float())
        else:
            # Default: slight bias toward Type B
            init = torch.zeros(num_labels, 3)
            init[:, TYPE_B] = 1.0
            self.q = nn.Parameter(init)

        self.ema_decay = ema_decay
        # Running statistics for dynamic refinement
        self.register_buffer(
            "ema_attn_variance", torch.zeros(num_labels)
        )  # \bar{a}_k
        self.register_buffer(
            "ema_info_gain", torch.zeros(num_labels)
        )  # \bar{c}_k
        self.register_buffer(
            "ema_initialized", torch.tensor(False)
        )

    @property
    def type_probs(self) -> torch.Tensor:
        """Per-label type probabilities: softmax(q) -> (L, 3)."""
        return F.softmax(self.q, dim=-1)

    def gate_a(
        self,
        i_indices: torch.Tensor,
        k_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Type A gate for specific label pairs.

        g^A_{ik} = t_i^A * t_k^A

        Args:
            i_indices: (num_pairs,) source label indices
            k_indices: (num_pairs,) target label indices

        Returns:
            (num_pairs,) gate values
        """
        probs = self.type_probs  # (L, 3)
        t_a_i = probs[i_indices, TYPE_A]  # (num_pairs,)
        t_a_k = probs[k_indices, TYPE_A]  # (num_pairs,)
        return t_a_i * t_a_k

    def gate_b_vector(self) -> torch.Tensor:
        """Per-label Type B gate values.

        Returns:
            (L,) — t^B for each label
        """
        return self.type_probs[:, TYPE_B]

    def gate_c(
        self,
        i_indices: torch.Tensor,
        k_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Type C gate for specific label pairs.

        g^C_{ik} = max(t_i^C, t_k^C)

        Args:
            i_indices: (num_pairs,) source label indices
            k_indices: (num_pairs,) target label indices

        Returns:
            (num_pairs,) gate values
        """
        probs = self.type_probs  # (L, 3)
        t_c_i = probs[i_indices, TYPE_C]
        t_c_k = probs[k_indices, TYPE_C]
        return torch.max(t_c_i, t_c_k)

    def update_ema(
        self,
        attn_variance: torch.Tensor,
        info_gain: torch.Tensor,
    ) -> None:
        """Update running EMA statistics for dynamic type refinement.

        Args:
            attn_variance: (L,) — per-label attention variance (low -> Type A)
            info_gain: (L,) — per-label information gain (low -> Type C)
        """
        alpha = self.ema_decay
        if not self.ema_initialized:
            self.ema_attn_variance.copy_(attn_variance)
            self.ema_info_gain.copy_(info_gain)
            self.ema_initialized.fill_(True)
        else:
            self.ema_attn_variance.mul_(alpha).add_(attn_variance, alpha=1 - alpha)
            self.ema_info_gain.mul_(alpha).add_(info_gain, alpha=1 - alpha)

    def refine_types(
        self,
        tau_a: float = 1.0,
        tau_c: float = 1.0,
    ) -> None:
        """Refine type logits based on accumulated EMA statistics.

        q_i^A proportional to exp(-attn_variance_i / tau_a)  (low variance -> Type A)
        q_i^C proportional to exp(-info_gain_i / tau_c)      (low info gain -> Type C)
        q_i^B is the residual
        """
        with torch.no_grad():
            # Type A: low attention variance -> high A logit
            q_a = -self.ema_attn_variance / (tau_a + 1e-8)
            # Type C: low info gain -> high C logit
            q_c = -self.ema_info_gain / (tau_c + 1e-8)
            # Type B: residual
            q_b = torch.zeros_like(q_a)

            # Soft update: blend with current values
            blend = 0.1
            self.q.data[:, TYPE_A] = (1 - blend) * self.q.data[:, TYPE_A] + blend * q_a
            self.q.data[:, TYPE_B] = (1 - blend) * self.q.data[:, TYPE_B] + blend * q_b
            self.q.data[:, TYPE_C] = (1 - blend) * self.q.data[:, TYPE_C] + blend * q_c

    def get_type_distribution_summary(self) -> Dict[str, float]:
        """Return summary statistics of type assignments for logging."""
        with torch.no_grad():
            probs = self.type_probs  # (L, 3)
            hard_assign = probs.argmax(dim=-1)
            return {
                "type_a_count": float((hard_assign == TYPE_A).sum()),
                "type_b_count": float((hard_assign == TYPE_B).sum()),
                "type_c_count": float((hard_assign == TYPE_C).sum()),
                "type_a_mean_prob": float(probs[:, TYPE_A].mean()),
                "type_b_mean_prob": float(probs[:, TYPE_B].mean()),
                "type_c_mean_prob": float(probs[:, TYPE_C].mean()),
                "type_entropy": float(
                    -(probs * (probs + 1e-10).log()).sum(dim=-1).mean()
                ),
            }
