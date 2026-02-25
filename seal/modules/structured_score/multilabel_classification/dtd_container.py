"""DTD Combined Energy Container.

Owns the shared DTDTypeGate, DTDTypeAEnergy, and DTDTypeBEnergy.
Loads all initialization from the diagnosis JSON file produced by
seal.commands.dtd_diagnosis.
"""

from typing import Dict, Any
import json
import torch
from seal.modules.structured_score import StructuredScore
from .dtd_type_gate import DTDTypeGate
from .dtd_type_a import DTDTypeAEnergy
from .dtd_type_b import DTDTypeBEnergy


@StructuredScore.register("dtd-combined")
class DTDCombinedEnergy(StructuredScore):
    """Combined DTD energy: weighted sum of Type A and Type B energies.

    Loads initialization artifacts from a diagnosis JSON file and constructs
    the shared type gate and per-type energy modules.

    Args:
        diagnosis_file: Path to JSON from dtd_diagnosis.py
        input_feature_dim: Dimension of task network hidden features
        svd_rank: SVD embedding dimension (must match diagnosis)
        low_rank_dim: Low-rank projection dimension for Type B
        type_a_weight: Weight for Type A energy term
        type_b_weight: Weight for Type B energy term
    """

    def __init__(
        self,
        diagnosis_file: str,
        input_feature_dim: int,
        svd_rank: int = 32,
        low_rank_dim: int = 16,
        type_a_weight: float = 1.0,
        type_b_weight: float = 1.0,
    ):
        super().__init__()

        # Load diagnosis artifacts
        with open(diagnosis_file, "r") as f:
            diag = json.load(f)

        num_labels = diag["num_labels"]
        assert diag["svd_rank"] == svd_rank, (
            f"SVD rank mismatch: diagnosis has {diag['svd_rank']}, config has {svd_rank}"
        )

        # Shared type gate
        q_init = torch.tensor(diag["q_init"], dtype=torch.float)
        self.type_gate = DTDTypeGate(num_labels, q_init=q_init)

        # Type A energy
        type_a_pairs_data = diag["type_a_pairs"]
        if len(type_a_pairs_data) > 0:
            pair_indices = [(p["i"], p["k"]) for p in type_a_pairs_data]
            w_init = [p["w_init"] for p in type_a_pairs_data]
            mu_init = [p["mu_init"] for p in type_a_pairs_data]
            log_tau_init = [p["log_tau_init"] for p in type_a_pairs_data]

            self.type_a = DTDTypeAEnergy(
                type_gate=self.type_gate,
                pair_indices=pair_indices,
                w_init=w_init,
                mu_init=mu_init,
                log_tau_init=log_tau_init,
            )
            self.has_type_a = True
        else:
            self.has_type_a = False

        # Type B energy
        svd_embeddings = torch.tensor(diag["svd_embeddings"], dtype=torch.float)
        self.type_b = DTDTypeBEnergy(
            type_gate=self.type_gate,
            num_labels=num_labels,
            input_feature_dim=input_feature_dim,
            svd_rank=svd_rank,
            low_rank_dim=low_rank_dim,
            svd_embeddings_init=svd_embeddings,
        )

        self.type_a_weight = type_a_weight
        self.type_b_weight = type_b_weight

    def forward(
        self,
        y: torch.Tensor,  # (batch, num_samples, num_labels)
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        energy = y.new_zeros(y.shape[0], y.shape[1])

        if self.has_type_a:
            energy = energy + self.type_a_weight * self.type_a(y, buffer, **kwargs)

        energy = energy + self.type_b_weight * self.type_b(y, buffer, **kwargs)

        return energy
