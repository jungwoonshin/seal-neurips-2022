"""Label Self-Attention (LSA) module for BiStruct.

Refines initial per-label predictions by allowing labels to attend to each other.
Each label has a learnable embedding; these are concatenated with the initial
prediction scalar, projected to an attention-compatible dimension, and passed
through one self-attention layer, then projected back to a scalar prediction
per label.
"""

import torch
import torch.nn as nn


class LabelSelfAttention(nn.Module):
    """Label Self-Attention module.

    Args:
        num_labels: Number of labels L.
        label_embed_dim: Dimension d_l of each label embedding.
        attn_dim: Internal dimension for self-attention (must be divisible by num_heads).
        num_heads: Number of attention heads.
        dropout: Dropout rate for attention.
    """

    def __init__(
        self,
        num_labels: int,
        label_embed_dim: int = 64,
        attn_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.label_embed_dim = label_embed_dim
        assert attn_dim % num_heads == 0, (
            f"attn_dim ({attn_dim}) must be divisible by num_heads ({num_heads})"
        )

        # Learnable label embeddings E ∈ R^{L × d_l}
        # Shared with backward model
        self.label_embeddings = nn.Parameter(
            torch.randn(num_labels, label_embed_dim) * 0.02
        )

        # Project [E_i; ŷ_i] ∈ R^{d_l + 1} -> R^{attn_dim}
        self.input_projection = nn.Linear(label_embed_dim + 1, attn_dim)

        # Self-attention layer
        self.self_attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(attn_dim)
        self.ffn = nn.Sequential(
            nn.Linear(attn_dim, attn_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(attn_dim * 4, attn_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(attn_dim)

        # Project back to scalar prediction per label
        self.output_projection = nn.Linear(attn_dim, 1)

    def forward(
        self,
        y_hat: torch.Tensor,  # (batch, num_labels) initial predictions in [0,1]
    ) -> torch.Tensor:
        """Refine predictions using label self-attention.

        Args:
            y_hat: Initial sigmoid predictions, shape (batch, num_labels).

        Returns:
            Refined predictions (logits, pre-sigmoid), shape (batch, num_labels).
        """
        batch_size = y_hat.shape[0]

        # Expand label embeddings: (1, L, d_l) -> (batch, L, d_l)
        label_emb = self.label_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        # Concatenate: [E_i; ŷ_i] -> (batch, L, d_l + 1)
        y_hat_expanded = y_hat.unsqueeze(-1)  # (batch, L, 1)
        tokens = torch.cat([label_emb, y_hat_expanded], dim=-1)

        # Project to attention dimension: (batch, L, d_l + 1) -> (batch, L, attn_dim)
        tokens = self.input_projection(tokens)

        # Self-attention with residual + layer norm
        attn_out, _ = self.self_attn(tokens, tokens, tokens)
        tokens = self.attn_norm(tokens + attn_out)

        # FFN with residual + layer norm
        ffn_out = self.ffn(tokens)
        tokens = self.ffn_norm(tokens + ffn_out)

        # Project to scalar per label: (batch, L, 1) -> (batch, L)
        refined_logits = self.output_projection(tokens).squeeze(-1)

        return refined_logits
