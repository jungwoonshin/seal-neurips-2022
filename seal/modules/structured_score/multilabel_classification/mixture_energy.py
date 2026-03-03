"""
Mixture of Dependency Mechanisms: End-to-end learned energy decomposition.

Replaces the DTD approach (diagnosis + discrete type gates + hand-designed rules)
with a principled mixture of three mechanisms with distinct inductive biases:

    E(x, y) = pi_1(x,y) * E^(1)(y) + pi_2(x,y) * E^(2)(x,y) + pi_3(x,y) * 0

where:
    Mechanism 1 (Invariant): E^(1)(y) = ||R^T y||^2  — input-independent
    Mechanism 2 (Conditional): E^(2)(x,y) = ||Z(x)^T y||^2  — input-dependent
    Mechanism 3 (Null): zero energy  — captures noise/independence

Mixture weights (pi_1, pi_2, pi_3) are instance-level, learned via softmax
over a linear function of [task_features || stats(y)].

Everything is learned end-to-end through the SEAL NCE objective.
"""

import json
import logging
import math
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from seal.modules.structured_score import StructuredScore

logger = logging.getLogger(__name__)


class MechanismInvariant(nn.Module):
    """Input-independent energy via low-rank projection (replaces Type A).

    E^(1)(y) = ||R^T y||^2

    where R in R^{L x r_1}. This is architecturally constrained to be
    input-independent: R has no dependency on x.
    """

    def __init__(self, num_labels: int, rank: int = 16):
        super().__init__()
        self.R = nn.Parameter(torch.randn(num_labels, rank) * 0.01)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Compute invariant energy.

        Args:
            y: (batch, num_samples, L)

        Returns:
            Energy of shape (batch, num_samples)
        """
        # y @ R -> (batch, num_samples, r_1)
        proj = y @ self.R
        # ||proj||^2 -> (batch, num_samples)
        return (proj * proj).sum(dim=-1)


class MechanismConditional(nn.Module):
    """Input-conditional energy via factored parameterization (replaces Type B).

    E^(2)(x, y) = ||Z(x)^T y||^2

    where z_i(x) = P * (e_i . (Q * T_E(x)))

    Same factored parameterization as DTDTypeBEnergy but without the gate
    weighting (y * t^B). The mixture weights handle soft routing instead.
    """

    def __init__(
        self,
        num_labels: int,
        input_feature_dim: int,
        svd_rank: int = 32,
        conditional_rank: int = 16,
        svd_embeddings_init: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.svd_rank = svd_rank
        self.conditional_rank = conditional_rank

        # Q: shared input projection (h -> d)
        self.input_proj = nn.Linear(input_feature_dim, svd_rank, bias=False)
        nn.init.kaiming_uniform_(self.input_proj.weight, a=math.sqrt(5))

        # P: shared rank projection (d -> r_2)
        self.rank_proj = nn.Linear(svd_rank, conditional_rank, bias=False)
        nn.init.kaiming_uniform_(self.rank_proj.weight, a=math.sqrt(5))

        # Per-label embeddings e_i in R^d
        if svd_embeddings_init is not None:
            assert svd_embeddings_init.shape == (num_labels, svd_rank), (
                f"Expected ({num_labels}, {svd_rank}), got {svd_embeddings_init.shape}"
            )
            self.label_embeddings = nn.Parameter(svd_embeddings_init.float())
        else:
            self.label_embeddings = nn.Parameter(
                torch.randn(num_labels, svd_rank) * 0.01
            )

    def forward(
        self,
        y: torch.Tensor,
        task_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute conditional energy.

        Args:
            y: (batch, num_samples, L)
            task_features: (batch, h) from task network

        Returns:
            Energy of shape (batch, num_samples)
        """
        # Step 1: Q * T_E(x) -> (batch, d)
        q_x = self.input_proj(task_features)

        # Step 2: e_i . q(x) for all labels, then project through P
        # label_embeddings: (L, d), q_x: (batch, d) -> (batch, L, d)
        modulated = self.label_embeddings.unsqueeze(0) * q_x.unsqueeze(1)

        # Apply rank projection P: (batch, L, d) -> (batch, L, r_2)
        Z = self.rank_proj(modulated)

        # Step 3: Z(x)^T @ y^T -> (batch, r_2, num_samples)
        # Z: (batch, L, r_2), y: (batch, num_samples, L)
        agg = torch.bmm(
            Z.transpose(1, 2),       # (batch, r_2, L)
            y.transpose(1, 2),        # (batch, L, num_samples)
        ).transpose(1, 2)            # (batch, num_samples, r_2)

        # Step 4: ||agg||^2 -> (batch, num_samples)
        return (agg * agg).sum(dim=-1)


@StructuredScore.register("dtd-mixture")
class MixtureOfMechanismsEnergy(StructuredScore):
    """Mixture of Dependency Mechanisms energy.

    E(x, y) = pi_1(x,y) * E^(1)(y) + pi_2(x,y) * E^(2)(x,y) + pi_3(x,y) * 0

    where (pi_1, pi_2, pi_3) = softmax(W_pi * [T_E(x) || stats(y)] + b_pi)
    and stats(y) = [mean(y), ||y||, var(y)].

    Registered as "dtd-mixture" for use as global_score in SEAL configs with
    score_nn type "multi-label-classification-dtd".
    """

    def __init__(
        self,
        num_labels: int,
        input_feature_dim: int,
        invariant_rank: int = 16,
        svd_rank: int = 32,
        conditional_rank: int = 16,
        diagnosis_file: Optional[str] = None,
    ):
        """
        Args:
            num_labels: Number of labels L.
            input_feature_dim: Dimension h of task network hidden features.
            invariant_rank: Rank r_1 for the invariant mechanism.
            svd_rank: SVD embedding dimension d for the conditional mechanism.
            conditional_rank: Rank r_2 for the conditional mechanism.
            diagnosis_file: Optional path to DTD diagnosis JSON. Only
                svd_embeddings, num_labels, and svd_rank are read.
        """
        super().__init__()

        # Load SVD embeddings from diagnosis if provided
        svd_embeddings_init = None
        if diagnosis_file is not None:
            logger.info(f"Loading SVD embeddings from {diagnosis_file}")
            with open(diagnosis_file, "r") as f:
                diagnosis = json.load(f)

            diag_num_labels = diagnosis["num_labels"]
            diag_svd_rank = diagnosis["svd_rank"]

            assert diag_num_labels == num_labels, (
                f"Diagnosis num_labels={diag_num_labels} != config num_labels={num_labels}"
            )

            actual_svd_rank = min(svd_rank, diag_svd_rank)
            if actual_svd_rank != svd_rank:
                logger.warning(
                    f"Requested svd_rank={svd_rank} but diagnosis has {diag_svd_rank}, "
                    f"using {actual_svd_rank}"
                )
                svd_rank = actual_svd_rank

            svd_embeddings = torch.tensor(
                diagnosis["svd_embeddings"], dtype=torch.float32
            )
            if svd_embeddings.shape[1] > svd_rank:
                svd_embeddings = svd_embeddings[:, :svd_rank]
            elif svd_embeddings.shape[1] < svd_rank:
                pad = torch.zeros(num_labels, svd_rank - svd_embeddings.shape[1])
                svd_embeddings = torch.cat([svd_embeddings, pad], dim=1)
            svd_embeddings_init = svd_embeddings

        # Mechanism 1: Invariant (input-independent)
        self.mechanism_invariant = MechanismInvariant(num_labels, invariant_rank)

        # Mechanism 2: Conditional (input-dependent)
        self.mechanism_conditional = MechanismConditional(
            num_labels=num_labels,
            input_feature_dim=input_feature_dim,
            svd_rank=svd_rank,
            conditional_rank=conditional_rank,
            svd_embeddings_init=svd_embeddings_init,
        )

        # Mixture weight layer: [task_features || stats(y)] -> 3 logits
        # stats(y) = [mean(y), ||y||, var(y)] -> 3 scalars
        num_stats = 3
        self.mixture_layer = nn.Linear(input_feature_dim + num_stats, 3)

        logger.info(
            f"MixtureOfMechanisms initialized: {num_labels} labels, "
            f"invariant_rank={invariant_rank}, svd_rank={svd_rank}, "
            f"conditional_rank={conditional_rank}"
        )

    def forward(
        self,
        y: torch.Tensor,  # (batch, num_samples, num_labels)
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute mixture-of-mechanisms energy.

        E(x, y) = pi_1 * E^(1)(y) + pi_2 * E^(2)(x, y) + pi_3 * 0

        Args:
            y: Label predictions, shape (batch, num_samples, num_labels).
            buffer: Must contain "task_features" of shape (batch, h).

        Returns:
            Energy tensor of shape (batch, num_samples).
        """
        task_features = buffer["task_features"]  # (batch, h)

        # Compute y statistics: (batch, num_samples, 3)
        y_mean = y.mean(dim=-1, keepdim=True)        # (batch, num_samples, 1)
        y_norm = y.norm(dim=-1, keepdim=True)         # (batch, num_samples, 1)
        y_var = y.var(dim=-1, keepdim=True)            # (batch, num_samples, 1)
        y_stats = torch.cat([y_mean, y_norm, y_var], dim=-1)  # (batch, num_samples, 3)

        # Expand task_features to match num_samples: (batch, num_samples, h)
        task_expanded = task_features.unsqueeze(1).expand(
            -1, y.shape[1], -1
        )

        # Mixture input: (batch, num_samples, h+3)
        mixture_input = torch.cat([task_expanded, y_stats], dim=-1)

        # Mixture weights: (batch, num_samples, 3)
        pi = F.softmax(self.mixture_layer(mixture_input), dim=-1)

        # Mechanism energies
        e_invariant = self.mechanism_invariant(y)                           # (batch, num_samples)
        e_conditional = self.mechanism_conditional(y, task_features)        # (batch, num_samples)

        # Combine: pi_1 * E^(1) + pi_2 * E^(2) + pi_3 * 0
        energy = pi[:, :, 0] * e_invariant + pi[:, :, 1] * e_conditional

        return energy
