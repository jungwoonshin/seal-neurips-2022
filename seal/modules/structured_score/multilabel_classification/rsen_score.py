"""
Representation-Space Energy Networks (RSEN) for Multi-Label Classification.

Instead of defining the energy function over the output space E_Θ(x, ỹ),
RSEN defines it over the task-net's internal representation space:

    E_Θ(x, R_Φ(x), ỹ) → ℝ

where R_Φ(x) ∈ ℝ^{L×d} is the matrix of per-label internal representations.
"""

from typing import List, Tuple, Union, Dict, Any, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.modules.feedforward import FeedForward
from seal.modules.structured_score import StructuredScore


@StructuredScore.register("rsen-alignment")
class RSENAlignmentEnergy(StructuredScore):
    """Representation alignment energy.

    E^align = -∑_{i,j} A_{ij} · sim(R_i, R_j) · ỹ_i · ỹ_j

    Assigns low energy when co-occurring labels have aligned representations.
    """

    def __init__(
        self,
        num_labels: int,
        init_scale: float = 0.01,
    ):
        super().__init__()
        self.num_labels = num_labels
        # Learned label affinity matrix A ∈ ℝ^{L×L}
        # Parameterized as a raw matrix; sigmoid applied during forward for [0,1] range
        self.affinity_logits = nn.Parameter(
            torch.randn(num_labels, num_labels) * init_scale
        )

    def forward(
        self,
        y: torch.Tensor,  # (batch, num_samples, num_labels)
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Args:
            y: Predicted labels, shape (batch, num_samples, num_labels)
            buffer: Must contain 'representations' of shape (batch, num_labels, hidden_dim)

        Returns:
            Energy scores of shape (batch, num_samples)
        """
        representations = buffer.get("representations")
        if representations is None:
            return torch.zeros(
                y.shape[0], y.shape[1], device=y.device, dtype=y.dtype
            )

        # representations: (batch, num_labels, hidden_dim)
        # Compute pairwise cosine similarity: (batch, num_labels, num_labels)
        rep_norm = F.normalize(representations, p=2, dim=-1)  # (batch, L, d)
        sim_matrix = torch.bmm(rep_norm, rep_norm.transpose(1, 2))  # (batch, L, L)

        # Affinity matrix A ∈ [0, 1]
        A = torch.sigmoid(self.affinity_logits)  # (L, L)

        # Weighted similarity: (batch, L, L)
        weighted_sim = A.unsqueeze(0) * sim_matrix  # (batch, L, L)

        # y: (batch, num_samples, L)
        # Compute ỹ_i · ỹ_j for all pairs: (batch, num_samples, L, L)
        y_outer = y.unsqueeze(-1) * y.unsqueeze(-2)  # (batch, S, L, L)

        # Energy: -∑_{i,j} A_{ij} · sim(R_i, R_j) · ỹ_i · ỹ_j
        # weighted_sim: (batch, L, L) -> (batch, 1, L, L)
        energy = -(weighted_sim.unsqueeze(1) * y_outer).sum(dim=(-2, -1))  # (batch, S)

        return energy


@StructuredScore.register("rsen-distinctiveness")
class RSENDistinctivenessEnergy(StructuredScore):
    """Representation distinctiveness energy.

    E^distinct = ∑_{i,j} (1 - A_{ij}) · max(0, sim(R_i, R_j) - δ) · (1 - ỹ_i · ỹ_j)

    Penalizes when non-co-occurring labels have similar representations.
    """

    def __init__(
        self,
        num_labels: int,
        margin: float = 0.1,
        init_scale: float = 0.01,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.margin = margin
        # Shared affinity logits (or can be separate; here we use separate for flexibility)
        self.affinity_logits = nn.Parameter(
            torch.randn(num_labels, num_labels) * init_scale
        )

    def forward(
        self,
        y: torch.Tensor,  # (batch, num_samples, num_labels)
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        representations = buffer.get("representations")
        if representations is None:
            return torch.zeros(
                y.shape[0], y.shape[1], device=y.device, dtype=y.dtype
            )

        # representations: (batch, num_labels, hidden_dim)
        rep_norm = F.normalize(representations, p=2, dim=-1)
        sim_matrix = torch.bmm(rep_norm, rep_norm.transpose(1, 2))  # (batch, L, L)

        A = torch.sigmoid(self.affinity_logits)  # (L, L)
        one_minus_A = 1.0 - A  # (L, L)

        # max(0, sim - δ)
        sim_margin = torch.relu(sim_matrix - self.margin)  # (batch, L, L)

        # Weighted by (1 - A): (batch, L, L)
        weighted_sim = one_minus_A.unsqueeze(0) * sim_margin

        # (1 - ỹ_i · ỹ_j): penalize only when labels are NOT co-occurring
        y_outer = y.unsqueeze(-1) * y.unsqueeze(-2)  # (batch, S, L, L)
        non_cooccur = 1.0 - y_outer  # (batch, S, L, L)

        # Energy: ∑_{i,j} (1 - A_{ij}) · max(0, sim - δ) · (1 - ỹ_i · ỹ_j)
        energy = (weighted_sim.unsqueeze(1) * non_cooccur).sum(dim=(-2, -1))  # (batch, S)

        return energy


@StructuredScore.register("rsen-conditioned")
class RSENConditionedEnergy(StructuredScore):
    """Output-conditioned representation energy.

    E^cond = MLP(Attention(R, R, R) || ỹ)

    Self-attention over per-label representations, conditioned on predicted outputs,
    followed by a learned MLP for higher-order interactions.
    """

    def __init__(
        self,
        num_labels: int,
        hidden_dim: int,
        num_attention_heads: int = 4,
        attention_dropout: float = 0.1,
        mlp_hidden_dim: int = 128,
        mlp_num_layers: int = 2,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim

        # Multi-head self-attention over label representations
        # Note: PyTorch 1.8 does not support batch_first, so we transpose manually
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
        )
        self.attn_layer_norm = nn.LayerNorm(hidden_dim)

        # MLP that takes concatenated attention output and predicted labels
        # Input: pooled attention output (hidden_dim) + predicted labels (num_labels)
        mlp_input_dim = hidden_dim + num_labels
        layers = []
        in_dim = mlp_input_dim
        for i in range(mlp_num_layers - 1):
            layers.append(nn.Linear(in_dim, mlp_hidden_dim))
            layers.append(nn.ReLU())
            in_dim = mlp_hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

        # Initialize final layer to small values for stable training start
        nn.init.normal_(self.mlp[-1].weight, std=math.sqrt(2.0 / in_dim) * 0.1)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        y: torch.Tensor,  # (batch, num_samples, num_labels)
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        representations = buffer.get("representations")
        if representations is None:
            return torch.zeros(
                y.shape[0], y.shape[1], device=y.device, dtype=y.dtype
            )

        # representations: (batch, num_labels, hidden_dim)
        batch_size = representations.shape[0]
        num_samples = y.shape[1]

        # Self-attention: Attention(R, R, R)
        # PyTorch MHA expects (seq_len, batch, embed_dim) without batch_first
        rep_t = representations.transpose(0, 1)  # (L, batch, hidden_dim)
        attn_out_t, _ = self.attention(rep_t, rep_t, rep_t)  # (L, batch, hidden_dim)
        attn_out = attn_out_t.transpose(0, 1)  # (batch, L, hidden_dim)
        attn_out = self.attn_layer_norm(attn_out + representations)  # residual

        # Pool over labels via mean
        pooled = attn_out.mean(dim=1)  # (batch, hidden_dim)

        # Expand for num_samples: (batch, num_samples, hidden_dim)
        pooled_expanded = pooled.unsqueeze(1).expand(-1, num_samples, -1)

        # Concatenate with predicted labels: (batch, num_samples, hidden_dim + num_labels)
        mlp_input = torch.cat([pooled_expanded, y], dim=-1)

        # MLP output: (batch, num_samples, 1) -> (batch, num_samples)
        energy = self.mlp(mlp_input).squeeze(-1)

        return energy


@StructuredScore.register("rsen-combined")
class RSENCombinedEnergy(StructuredScore):
    """Combined RSEN energy: alignment + distinctiveness + conditioned.

    E_Θ(x, R_Φ(x), ỹ) = w_1 · E^align + w_2 · E^distinct + w_3 · E^cond
    """

    def __init__(
        self,
        num_labels: int,
        hidden_dim: int,
        alignment_weight: float = 1.0,
        distinctiveness_weight: float = 1.0,
        conditioned_weight: float = 1.0,
        distinctiveness_margin: float = 0.1,
        num_attention_heads: int = 4,
        attention_dropout: float = 0.1,
        mlp_hidden_dim: int = 128,
        mlp_num_layers: int = 2,
        affinity_init_scale: float = 0.01,
        share_affinity: bool = True,
    ):
        super().__init__()
        self.alignment_weight = alignment_weight
        self.distinctiveness_weight = distinctiveness_weight
        self.conditioned_weight = conditioned_weight

        # Alignment energy
        self.alignment = RSENAlignmentEnergy(
            num_labels=num_labels,
            init_scale=affinity_init_scale,
        )

        # Distinctiveness energy
        self.distinctiveness = RSENDistinctivenessEnergy(
            num_labels=num_labels,
            margin=distinctiveness_margin,
            init_scale=affinity_init_scale,
        )

        # Share affinity matrix between alignment and distinctiveness
        if share_affinity:
            self.distinctiveness.affinity_logits = self.alignment.affinity_logits

        # Conditioned energy
        self.conditioned = RSENConditionedEnergy(
            num_labels=num_labels,
            hidden_dim=hidden_dim,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            mlp_hidden_dim=mlp_hidden_dim,
            mlp_num_layers=mlp_num_layers,
        )

    def forward(
        self,
        y: torch.Tensor,  # (batch, num_samples, num_labels)
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        energy = torch.zeros(y.shape[0], y.shape[1], device=y.device, dtype=y.dtype)

        if self.alignment_weight > 0:
            energy = energy + self.alignment_weight * self.alignment(y, buffer, **kwargs)

        if self.distinctiveness_weight > 0:
            energy = energy + self.distinctiveness_weight * self.distinctiveness(y, buffer, **kwargs)

        if self.conditioned_weight > 0:
            energy = energy + self.conditioned_weight * self.conditioned(y, buffer, **kwargs)

        return energy
