"""
Training loop integrating the corrective energy alignment algorithm.

Each batch step:
1. Update Theta (energy net) with energy loss on batch data + corrective loss on critical-sampled data
2. Update Phi (task net) with BCE on batch data + BCE on critical-sampled data
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
    SEAL trainer with validation-guided energy correction.
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

        # Separate optimizers with weight decay
        self.opt_theta = AdamW(energy_net.parameters(), lr=config["lr_energy"], weight_decay=wd)
        self.opt_phi = AdamW(task_net.parameters(), lr=config["lr_task"], weight_decay=wd)

        # ReduceLROnPlateau: reduce LR when val metric stops improving
        patience = config.get("lr_patience", 10)
        factor = config.get("lr_factor", 0.5)
        self.sched_theta = ReduceLROnPlateau(self.opt_theta, mode="max", patience=patience, factor=factor, min_lr=1e-6)
        self.sched_phi = ReduceLROnPlateau(self.opt_phi, mode="max", patience=patience, factor=factor, min_lr=1e-6)

        # Best model checkpointing
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
        """Step LR schedulers with validation metric."""
        self.sched_theta.step(val_metric)
        self.sched_phi.step(val_metric)

    def _compute_bce(self, y_pred, y):
        """Weighted BCE loss, returns per-sample mean."""
        bce_raw = F.binary_cross_entropy(y_pred, y.float(), reduction="none")
        if self.pos_weight is not None:
            weight = torch.where(
                y > 0, self.pos_weight.unsqueeze(0), torch.ones_like(bce_raw))
            bce_raw = bce_raw * weight
        return bce_raw.mean(dim=-1)

    def train_one_epoch(self, epoch: int):
        """Train for one epoch. Returns average losses."""
        if self.config.get("no_correction_interval", False):
            return self._train_one_epoch_inline(epoch)
        return self._train_one_epoch_periodic(epoch)

    def _train_one_epoch_inline(self, epoch: int):
        """Inline mode: corrective loss computed directly on each batch."""
        self.task_net.train()
        self.energy_net.train()

        epoch_loss_theta = 0.0
        epoch_loss_phi = 0.0
        epoch_n_critical = 0
        n_batches = 0

        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)

            # ── Step 1: Update Theta (energy net) ──
            self.opt_theta.zero_grad()
            loss_correct, n_crit = self.corrector.batch_corrective_loss(
                self.energy_net, self.task_net, x, y)
            loss_theta = self.config["beta"] * loss_correct
            loss_theta.backward()
            torch.nn.utils.clip_grad_norm_(self.energy_net.parameters(), 1.0)
            self.opt_theta.step()

            # ── Step 2: Update Phi (task net) ──
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
            epoch_n_critical += n_crit
            n_batches += 1
            self.step += 1

        if n_batches > 0:
            avg_theta = epoch_loss_theta / n_batches
            avg_phi = epoch_loss_phi / n_batches
            lr_t = self.opt_theta.param_groups[0]["lr"]
            lr_p = self.opt_phi.param_groups[0]["lr"]
            total = len(self.train_loader.dataset) if hasattr(self.train_loader, "dataset") else n_batches
            print(
                f"  Epoch {epoch + 1}: "
                f"L_theta = {avg_theta:.4f}, "
                f"L_phi = {avg_phi:.4f}, "
                f"n_critical = {epoch_n_critical}/{total}, "
                f"lr_task = {lr_p:.6f}, lr_energy = {lr_t:.6f}"
            )
            return avg_theta, avg_phi
        return 0.0, 0.0

    def _train_one_epoch_periodic(self, epoch: int):
        """Periodic mode: diagnose every correction_interval steps."""
        self.task_net.train()
        self.energy_net.train()

        epoch_loss_theta = 0.0
        epoch_loss_phi = 0.0
        n_batches = 0

        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            bs = x.size(0)

            # ── Periodic diagnosis ──
            if self.step % self.config["correction_interval"] == 0:
                n_critical = self.corrector.diagnose(
                    self.energy_net, self.task_net,
                    self.train_loader, self.device,
                )
                diagnostics = self._compute_diagnostics()
                print(
                    f"  Step {self.step}: |C| = {n_critical}, "
                    f"frac = {diagnostics['frac_inverted']:.4f}, "
                    f"mean_delta = {diagnostics['mean_delta_on_C']:.4f}, "
                    f"mean_error = {diagnostics['mean_task_error_on_C']:.4f}"
                )

            # Sample critical batch (shared for both updates)
            c_batch = self.corrector.sample_batch(bs)

            # ── Step 1: Update Theta (energy net) ──
            self.opt_theta.zero_grad()

            loss_correct = self.corrector.corrective_loss(self.energy_net, c_batch, task_net=self.task_net)

            loss_theta = self.config["beta"] * loss_correct
            loss_theta.backward()
            torch.nn.utils.clip_grad_norm_(self.energy_net.parameters(), 1.0)
            self.opt_theta.step()

            # ── Step 2: Update Phi (task net) ──
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

    def _energy_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            y_neg = (self.task_net(x) >= 0.5).float()

        e_true = self.energy_net(x, y.float())
        e_neg = self.energy_net(x, y_neg)

        margin = (y_neg != y.float()).float().mean(dim=-1)
        return F.relu(margin - e_neg + e_true).mean()

    def _compute_diagnostics(self) -> dict:
        C = self.corrector.critical_set
        if len(C) == 0:
            return {
                "frac_inverted": 0.0,
                "mean_delta_on_C": 0.0,
                "mean_task_error_on_C": 0.0,
            }

        deltas = torch.stack([s["delta"] for s in C])
        errors = torch.stack([s["task_error"] for s in C])

        total = len(self.train_loader.dataset) if hasattr(self.train_loader, "dataset") else len(C)

        return {
            "frac_inverted": len(C) / max(total, 1),
            "mean_delta_on_C": deltas.mean().item(),
            "mean_task_error_on_C": errors.mean().item(),
        }
