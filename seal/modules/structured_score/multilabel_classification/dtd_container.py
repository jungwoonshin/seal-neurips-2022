"""
DTD Combined Energy: Top-level container that replaces SEAL's monolithic E^global.

Composes Type A, Type B energies with the type gate, loads initialization from
the diagnosis JSON, and handles dynamic type refinement during training.

    E^DTD(x, y) = type_a_weight * E^A(y) + type_b_weight * E^B(x, y)

Type C dependencies contribute zero energy through the gate mechanism.

This is a drop-in replacement for the "multi-label-feedforward" StructuredScore
in SEAL configs — only the global_score entry in the score_nn config changes.
"""

import json
import logging
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
from seal.modules.structured_score import StructuredScore
from .dtd_type_gate import DTDTypeGate
from .dtd_type_a import DTDTypeAEnergy
from .dtd_type_b import DTDTypeBEnergy

logger = logging.getLogger(__name__)


@StructuredScore.register("dtd-combined")
class DTDCombinedEnergy(StructuredScore):
    """Combined DTD energy replacing SEAL's monolithic E^global.

    Loads initialization from a diagnosis JSON file and instantiates:
        - DTDTypeGate: per-label type assignment
        - DTDTypeAEnergy: soft constraint enforcement
        - DTDTypeBEnergy: input-conditional interactions

    Type C dependencies are excluded via the gate mechanism (zero cost).
    """

    def __init__(
        self,
        diagnosis_file: str,
        input_feature_dim: int,
        svd_rank: int = 32,
        low_rank_dim: int = 16,
        type_a_weight: float = 1.0,
        type_b_weight: float = 1.0,
        ema_decay: float = 0.99,
        refine_interval: int = 100,
        tau_a: float = 1.0,
        tau_c: float = 1.0,
    ):
        """
        Args:
            diagnosis_file: Path to JSON from dtd_diagnosis.py.
            input_feature_dim: Dimension h of task network hidden features.
            svd_rank: SVD embedding dimension d (must match diagnosis).
            low_rank_dim: Low-rank projection dimension r for Type B.
            type_a_weight: Scalar weight for Type A energy.
            type_b_weight: Scalar weight for Type B energy.
            ema_decay: EMA decay rate for dynamic type refinement.
            refine_interval: Refine types every P steps.
            tau_a: Temperature for Type A refinement.
            tau_c: Temperature for Type C refinement.
        """
        super().__init__()

        # Load diagnosis
        logger.info(f"Loading DTD diagnosis from {diagnosis_file}")
        with open(diagnosis_file, "r") as f:
            diagnosis = json.load(f)

        num_labels = diagnosis["num_labels"]
        diag_svd_rank = diagnosis["svd_rank"]

        # Use the smaller of requested and available SVD rank
        actual_svd_rank = min(svd_rank, diag_svd_rank)
        if actual_svd_rank != svd_rank:
            logger.warning(
                f"Requested svd_rank={svd_rank} but diagnosis has {diag_svd_rank}, "
                f"using {actual_svd_rank}"
            )

        # Initialize type gate
        q_init = torch.tensor(diagnosis["q_init"], dtype=torch.float32)
        self.type_gate = DTDTypeGate(
            num_labels=num_labels,
            q_init=q_init,
            ema_decay=ema_decay,
        )

        # Initialize Type A energy
        type_a_pairs = diagnosis["type_a_pairs"]
        pair_i = [p["i"] for p in type_a_pairs]
        pair_k = [p["k"] for p in type_a_pairs]
        w_init = [p["w_init"] for p in type_a_pairs]
        mu_init = [p["mu_init"] for p in type_a_pairs]
        log_tau_init = [p["log_tau_init"] for p in type_a_pairs]

        self.type_a_energy = DTDTypeAEnergy(
            type_gate=self.type_gate,
            pair_indices_i=pair_i,
            pair_indices_k=pair_k,
            w_init=w_init,
            mu_init=mu_init,
            log_tau_init=log_tau_init,
        )

        # Initialize Type B energy
        svd_embeddings = torch.tensor(
            diagnosis["svd_embeddings"], dtype=torch.float32
        )
        # Truncate or pad to actual_svd_rank
        if svd_embeddings.shape[1] > actual_svd_rank:
            svd_embeddings = svd_embeddings[:, :actual_svd_rank]
        elif svd_embeddings.shape[1] < actual_svd_rank:
            pad = torch.zeros(num_labels, actual_svd_rank - svd_embeddings.shape[1])
            svd_embeddings = torch.cat([svd_embeddings, pad], dim=1)

        self.type_b_energy = DTDTypeBEnergy(
            type_gate=self.type_gate,
            num_labels=num_labels,
            input_feature_dim=input_feature_dim,
            svd_rank=actual_svd_rank,
            low_rank_dim=low_rank_dim,
            svd_embeddings_init=svd_embeddings,
        )

        # Weights
        self.type_a_weight = type_a_weight
        self.type_b_weight = type_b_weight

        # Dynamic refinement settings
        self.refine_interval = refine_interval
        self.tau_a = tau_a
        self.tau_c = tau_c
        self.register_buffer("step_counter", torch.tensor(0, dtype=torch.long))

        logger.info(
            f"DTD initialized: {num_labels} labels, "
            f"{len(type_a_pairs)} Type A pairs, "
            f"svd_rank={actual_svd_rank}, low_rank_dim={low_rank_dim}"
        )

    def forward(
        self,
        y: torch.Tensor,  # (batch, num_samples, num_labels)
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute combined DTD energy.

        E^DTD(x, y) = type_a_weight * E^A(y) + type_b_weight * E^B(x, y)

        Args:
            y: Label predictions, shape (batch, num_samples, num_labels).
            buffer: Must contain "task_features" for Type B energy.

        Returns:
            Energy tensor of shape (batch, num_samples).
        """
        # Type A: soft constraints (input-independent)
        energy_a = self.type_a_energy(y, buffer, **kwargs)  # (batch, num_samples)

        # Type B: input-conditional interactions
        energy_b = self.type_b_energy(y, buffer, **kwargs)  # (batch, num_samples)

        # Combined
        energy = self.type_a_weight * energy_a + self.type_b_weight * energy_b

        # Increment step counter for refinement scheduling
        if self.training:
            self.step_counter += 1

        return energy

    def maybe_refine_types(self) -> bool:
        """Check if it's time to refine type classification. Call from callback.

        Returns:
            True if refinement was performed.
        """
        if self.step_counter > 0 and self.step_counter % self.refine_interval == 0:
            self.type_gate.refine_types(tau_a=self.tau_a, tau_c=self.tau_c)
            return True
        return False
