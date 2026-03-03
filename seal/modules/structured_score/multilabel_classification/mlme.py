"""
Masked Label Model as Energy (MLME): Pseudolikelihood-based structured energy.

Replaces the monolithic E^global = v^T σ(My) with a pseudolikelihood energy
that predicts each label from all other labels and the input (Besag, 1975):

    E^cross(x, y) = -Σ_k [y_k log σ(ŷ_k) + (1-y_k) log(1 - σ(ŷ_k))]

where ŷ_k = a_k · ReLU(W · [c(x) || s_{-k}]) + b_k
and   s_{-k} = Σ_{j≠k} y_j · e_j  (leave-one-out label context)

This provides per-label conditional gradient signals and is a principled
replacement grounded in pseudolikelihood theory.
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


@StructuredScore.register("mlme")
class MLMEEnergy(StructuredScore):
    """Masked Label Model as Energy.

    Computes a pseudolikelihood cross-entropy energy over label configurations:
    each label k is predicted from all other labels (leave-one-out) and the
    input features, then the total BCE is used as the energy.

    Higher energy = more inconsistent configuration.
    Registered as "mlme" for use as global_score in SEAL configs with
    score_nn type "multi-label-classification-dtd".
    """

    def __init__(
        self,
        num_labels: int,
        input_feature_dim: int,
        label_embed_dim: int = 32,
        hidden_dim: int = 128,
        diagnosis_file: Optional[str] = None,
    ):
        """
        Args:
            num_labels: Number of labels L.
            input_feature_dim: Dimension h of task network hidden features.
            label_embed_dim: Dimension d for per-label embeddings e_k.
            hidden_dim: Dimension d' for the shared hidden layer.
            diagnosis_file: Optional path to DTD diagnosis JSON for
                initializing label embeddings from SVD embeddings.
        """
        super().__init__()

        self.num_labels = num_labels
        self.label_embed_dim = label_embed_dim
        self.hidden_dim = hidden_dim

        # Per-label embeddings e_k: (L, d)
        svd_init = None
        if diagnosis_file is not None:
            logger.info(f"Loading SVD embeddings from {diagnosis_file}")
            with open(diagnosis_file, "r") as f:
                diagnosis = json.load(f)

            diag_num_labels = diagnosis["num_labels"]
            assert diag_num_labels == num_labels, (
                f"Diagnosis num_labels={diag_num_labels} != config num_labels={num_labels}"
            )

            svd_embeddings = torch.tensor(
                diagnosis["svd_embeddings"], dtype=torch.float32
            )
            if svd_embeddings.shape[1] > label_embed_dim:
                svd_embeddings = svd_embeddings[:, :label_embed_dim]
            elif svd_embeddings.shape[1] < label_embed_dim:
                pad = torch.zeros(num_labels, label_embed_dim - svd_embeddings.shape[1])
                svd_embeddings = torch.cat([svd_embeddings, pad], dim=1)
            svd_init = svd_embeddings

        if svd_init is not None:
            self.label_embeddings = nn.Parameter(svd_init)
        else:
            self.label_embeddings = nn.Parameter(
                torch.randn(num_labels, label_embed_dim) * 0.01
            )

        # Shared hidden layer split into input and label parts:
        #   W_input: (h -> d'), W_label: (d -> d')
        # So that hidden = ReLU(W_input(c_x) + W_label(s_{-k}))
        self.W_input = nn.Linear(input_feature_dim, hidden_dim)
        self.W_label = nn.Linear(label_embed_dim, hidden_dim, bias=False)

        # Per-label output projections a_k: (L, d')
        self.per_label_a = nn.Parameter(
            torch.randn(num_labels, hidden_dim) * (1.0 / math.sqrt(hidden_dim))
        )

        # Per-label biases b_k: (L,)
        self.per_label_b = nn.Parameter(torch.zeros(num_labels))

        logger.info(
            f"MLMEEnergy initialized: {num_labels} labels, "
            f"label_embed_dim={label_embed_dim}, hidden_dim={hidden_dim}, "
            f"input_feature_dim={input_feature_dim}"
        )

    def forward(
        self,
        y: torch.Tensor,  # (batch, num_samples, num_labels)
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute pseudolikelihood cross-entropy energy.

        E^cross(x, y) = -Σ_k [y_k log σ(ŷ_k) + (1-y_k) log(1-σ(ŷ_k))]

        Args:
            y: Label configurations, shape (batch, num_samples, L).
            buffer: Must contain "task_features" of shape (batch, h).

        Returns:
            Energy tensor of shape (batch, num_samples).
        """
        task_features = buffer["task_features"]  # (batch, h)
        batch_size, num_samples, L = y.shape

        # Step 1: Precompute input-side hidden contribution (done once)
        # base: (batch, d') -> (batch, 1, 1, d') for broadcasting
        base = self.W_input(task_features)  # (batch, d')
        base = base.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, d')

        # Step 2: Global label context s = y @ E
        # y: (batch, num_samples, L), label_embeddings: (L, d)
        # s: (batch, num_samples, d)
        s = y @ self.label_embeddings

        # Step 3: Compute all leave-one-out contexts at once
        # Per-label contribution: y[:,:,k] * e_k for all k simultaneously
        # contributions: (batch, num_samples, L, d)
        contributions = y.unsqueeze(-1) * self.label_embeddings.unsqueeze(0).unsqueeze(0)

        # s_leave_one_out[..., k, :] = s - y[:,:,k]*e_k
        # (batch, num_samples, 1, d) - (batch, num_samples, L, d) = (batch, num_samples, L, d)
        s_leave_one_out = s.unsqueeze(2) - contributions

        # Step 4: Apply W_label to all leave-one-out contexts
        # (batch, num_samples, L, d) @ (d, d') -> (batch, num_samples, L, d')
        label_hidden = self.W_label(s_leave_one_out)

        # Step 5: Combine with input-side hidden, apply ReLU
        # base: (batch, 1, 1, d'), label_hidden: (batch, num_samples, L, d')
        hidden = F.relu(base + label_hidden)  # (batch, num_samples, L, d')

        # Step 6: Per-label output projections
        # hidden: (batch, num_samples, L, d'), per_label_a: (L, d')
        # Element-wise multiply and sum over d' dimension
        logits = (hidden * self.per_label_a).sum(dim=-1) + self.per_label_b
        # logits: (batch, num_samples, L)

        # Step 7: Compute BCE energy (sum over labels)
        # BCE = -Σ_k [y_k log σ(ŷ_k) + (1-y_k) log(1-σ(ŷ_k))]
        energy = F.binary_cross_entropy_with_logits(
            logits, y, reduction='none'
        ).sum(dim=-1)  # (batch, num_samples)

        return energy
