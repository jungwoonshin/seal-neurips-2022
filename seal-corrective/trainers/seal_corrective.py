"""
C-SEAL training loop.

Each batch step:
1. Update Theta (energy net) with NCE loss on all examples
2. Update Phi (task net) with lambda1 * E(x, F(x)) + lambda2 * BCE on full batch
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from losses.corrective import EnergyCorrector


class SEALCorrectiveTrainer:
    """
    C-SEAL trainer: NCE energy alignment.
    """

    def __init__(
        self,
        config: dict,
        task_net: nn.Module,
        energy_net: nn.Module,
        corrector: EnergyCorrector,
        train_loader,
        val_loader,
        device: torch.device,
        pos_weight: torch.Tensor = None,
    ):
        self.config = config
        self.task_net = task_net
        self.energy_net = energy_net
        self.corrector = corrector
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.step = 0
        self.pos_weight = pos_weight.to(device) if pos_weight is not None else None

        wd = config.get("weight_decay", 1e-4)

        self.opt_theta = AdamW(energy_net.parameters(), lr=config["lr_energy"], weight_decay=wd)
        self.opt_phi = AdamW(task_net.parameters(), lr=config["lr_task"], weight_decay=wd)

        patience = config.get("lr_patience", 10)
        factor = config.get("lr_factor", 0.5)
        self.sched_theta = ReduceLROnPlateau(self.opt_theta, mode="max", patience=patience, factor=factor, min_lr=1e-6)
        self.sched_phi = ReduceLROnPlateau(self.opt_phi, mode="max", patience=patience, factor=factor, min_lr=1e-6)

        self.best_task_state = None
        self.best_energy_state = None

    def save_best(self):
        self.best_task_state = copy.deepcopy(self.task_net.state_dict())
        self.best_energy_state = copy.deepcopy(self.energy_net.state_dict())

    def restore_best(self):
        if self.best_task_state is not None:
            self.task_net.load_state_dict(self.best_task_state)
            self.energy_net.load_state_dict(self.best_energy_state)

    def step_schedulers(self, val_metric: float):
        self.sched_theta.step(val_metric)
        self.sched_phi.step(val_metric)

    def _compute_bce(self, y_pred, y):
        bce_raw = F.binary_cross_entropy(y_pred, y.float(), reduction="none")
        if self.pos_weight is not None:
            weight = torch.where(
                y > 0, self.pos_weight.unsqueeze(0), torch.ones_like(bce_raw))
            bce_raw = bce_raw * weight
        return bce_raw.mean(dim=-1)

    def train_one_epoch(self, epoch: int):
        self.task_net.train()
        self.energy_net.train()

        epoch_loss_theta = 0.0
        epoch_loss_phi = 0.0
        epoch_total = 0
        n_batches = 0

        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            B = x.size(0)

            # Step 1: Update energy net on all examples
            self.opt_theta.zero_grad()
            loss_E, info = self.corrector.energy_loss(
                self.energy_net, self.task_net, x, y,
            )
            loss_theta = self.config["beta"] * loss_E
            loss_theta.backward()
            torch.nn.utils.clip_grad_norm_(self.energy_net.parameters(), 1.0)
            self.opt_theta.step()

            # Step 2: Update task net on full batch
            self.opt_phi.zero_grad()
            y_pred = self.task_net(x)
            energy = self.energy_net(x, y_pred)
            bce = self._compute_bce(y_pred, y)
            loss_phi = (
                self.config["lambda1"] * energy + self.config["lambda2"] * bce
            ).mean()
            loss_phi.backward()
            torch.nn.utils.clip_grad_norm_(self.task_net.parameters(), 1.0)
            self.opt_phi.step()

            epoch_loss_theta += loss_theta.item()
            epoch_loss_phi += loss_phi.item()
            epoch_total += B
            n_batches += 1
            self.step += 1

        if n_batches > 0:
            avg_theta = epoch_loss_theta / n_batches
            avg_phi = epoch_loss_phi / n_batches
            lr_t = self.opt_theta.param_groups[0]["lr"]
            lr_p = self.opt_phi.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch + 1}: "
                f"L_theta = {avg_theta:.4f}, "
                f"L_phi = {avg_phi:.4f}, "
                f"lr_task = {lr_p:.6f}, lr_energy = {lr_t:.6f}"
            )
            return avg_theta, avg_phi
        return 0.0, 0.0

    def train(self):
        for epoch in range(self.config["epochs"]):
            self.train_one_epoch(epoch)
