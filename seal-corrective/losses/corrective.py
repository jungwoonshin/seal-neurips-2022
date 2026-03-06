"""
C-SEAL: Corrective Energy Alignment.

Energy loss: L_E = log(1 + exp(E(x, y*) - E(x, F(x))))
Pushes E(x, y*) below E(x, F(x)) — ground truth gets lower energy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyCorrector:
    """
    Computes corrective energy loss on all examples.
    """

    def energy_loss(
        self,
        energy_net: nn.Module,
        task_net: nn.Module,
        x: torch.Tensor,
        y_star: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Logistic margin energy loss over all examples.

        Args:
            energy_net: Energy network.
            task_net: Task network.
            x: (B, D) input features.
            y_star: (B, L) binary ground truth labels.

        Returns:
            (loss, info_dict) with diagnostics.
        """
        B = x.size(0)

        with torch.no_grad():
            y_hat = task_net(x).detach()

        e_true = energy_net(x, y_star.float())  # (B,)
        e_pred = energy_net(x, y_hat)            # (B,)
        losses = F.softplus(e_true - e_pred)      # (B,)
        loss = losses.mean()

        info = {"batch_size": B, "loss": loss.item()}
        return loss, info
