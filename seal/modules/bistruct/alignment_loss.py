"""Symmetric Alignment Loss for BiStruct.

Implements BYOL/SimSiam-style symmetric alignment:
    L_align = ||sg(h) - h_hat||^2 + ||h - sg(h_hat)||^2

where sg(·) is stop-gradient. This prevents representation collapse and
ensures both the encoder and backward model improve.
"""

import torch
import torch.nn as nn


class SymmetricAlignmentLoss(nn.Module):
    """Symmetric alignment loss between forward features and backward reconstruction."""

    def forward(
        self,
        h: torch.Tensor,  # (batch, feature_dim) forward features
        h_hat: torch.Tensor,  # (batch, feature_dim) backward reconstruction
    ) -> torch.Tensor:
        """Compute symmetric alignment loss.

        Args:
            h: Forward path features from encoder, shape (batch, d).
            h_hat: Backward path reconstructed features, shape (batch, d).

        Returns:
            Scalar alignment loss.
        """
        # ||sg(h) - h_hat||^2 : gradient flows only through h_hat (trains backward model)
        loss_backward = torch.mean((h.detach() - h_hat) ** 2)

        # ||h - sg(h_hat)||^2 : gradient flows only through h (trains encoder)
        loss_forward = torch.mean((h - h_hat.detach()) ** 2)

        return loss_backward + loss_forward
