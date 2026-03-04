"""
Unit tests for EnergyCorrector.

Tests:
1. diagnose: finds known inversions on toy data
2. corrective_loss gradient direction: pushes E(x, y_true) down, E(x, F(x)) up
3. empty critical set: returns 0.0, no division by zero
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models import TaskNet, EnergyNet
from losses import EnergyCorrector


def make_toy_data(n=100, input_dim=10, num_labels=5, seed=0):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, input_dim, generator=gen)
    y = (torch.rand(n, num_labels, generator=gen) > 0.5).float()
    return x, y


class InvertedEnergyNet(nn.Module):
    """Energy net that intentionally inverts energy for ~30% of examples."""

    def __init__(self, input_dim, num_labels, invert_mask):
        super().__init__()
        self.linear = nn.Linear(input_dim + num_labels, 1)
        self.invert_mask = invert_mask  # (n,) bool tensor

    def forward(self, x, y):
        # Simple energy: just a linear function
        cat = torch.cat([x, y], dim=-1)
        e = self.linear(cat).squeeze(-1)
        return e

    def energy_global(self, y):
        return torch.zeros(y.size(0))


def test_diagnose_finds_known_inversions():
    """
    Build a toy energy net that inverts energy for exactly 30% of validation set.
    Verify diagnose finds those examples.
    """
    torch.manual_seed(42)
    n, input_dim, num_labels = 100, 10, 5
    x, y = make_toy_data(n, input_dim, num_labels)

    task_net = TaskNet(input_dim, 20, num_labels)

    # Pre-compute task net predictions and errors
    with torch.no_grad():
        y_pred = task_net(x)
        y_hard = (y_pred >= 0.5).float()
        has_error = (y_hard != y).float().mean(dim=-1) > 0

    # Pre-compute energy values: 30% inverted
    n_inverted = 30
    e_true_vals = torch.ones(n) * 1.0
    e_pred_vals = torch.ones(n) * 2.0  # correct: pred higher energy
    e_pred_vals[:n_inverted] = 0.5      # inverted: pred LOWER energy

    # Expected: inverted AND has_error
    expected_critical = sum(
        1 for i in range(n)
        if e_pred_vals[i] < e_true_vals[i] and has_error[i]
    )

    # Energy net that uses call-order tracking (pred first, then true, per batch)
    class PrecomputedEnergyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(1))
            self._e_true = e_true_vals
            self._e_pred = e_pred_vals
            self._global_idx = 0  # tracks position across all forward calls
            self._call_parity = 0  # 0 = pred call, 1 = true call

        def forward(self, x, y):
            bs = x.size(0)
            start = self._global_idx
            end = start + bs

            if self._call_parity == 0:
                # First call in diagnose loop: energy_net(x, y_pred)
                result = self._e_pred[start:end]
                self._call_parity = 1
            else:
                # Second call in diagnose loop: energy_net(x, y.float())
                result = self._e_true[start:end]
                self._call_parity = 0
                self._global_idx = end  # advance after both calls

            return result

        def energy_global(self, y):
            return torch.zeros(y.size(0))

    energy_net = PrecomputedEnergyNet()
    dataset = TensorDataset(x, y)
    val_loader = DataLoader(dataset, batch_size=20, shuffle=False)

    corrector = EnergyCorrector(alpha=1.0, task_error_metric="hamming")
    n_found = corrector.diagnose(energy_net, task_net, val_loader, torch.device("cpu"))

    print(f"  Expected critical: {expected_critical}, Found: {n_found}")
    assert n_found == expected_critical, (
        f"Expected {expected_critical} critical examples, found {n_found}"
    )
    print("  PASSED: diagnose finds known inversions")


def test_corrective_loss_gradient_direction():
    """
    Given a known critical set, verify that one gradient step on L_correct
    increases the gap E(x, F(x)) - E(x, y_true).
    """
    torch.manual_seed(42)
    input_dim, num_labels = 10, 5

    energy_net = EnergyNet(input_dim, 20, num_labels, 20)
    opt = torch.optim.SGD(energy_net.parameters(), lr=0.1)

    # Create a synthetic critical set
    x = torch.randn(8, input_dim)
    y = (torch.rand(8, num_labels) > 0.5).float()
    y_pred = torch.sigmoid(torch.randn(8, num_labels))  # soft predictions
    task_errors = torch.rand(8) * 0.5 + 0.1  # all > 0

    critical_batch = [
        {"x": x[i], "y": y[i], "y_pred": y_pred[i], "task_error": task_errors[i],
         "delta": torch.tensor(-0.5), "criticality": torch.tensor((i + 1) / 8.0)}
        for i in range(8)
    ]

    corrector = EnergyCorrector(alpha=1.0, task_error_metric="hamming")

    # Measure gap before
    with torch.no_grad():
        e_true_before = energy_net(x, y).clone()
        e_pred_before = energy_net(x, y_pred).clone()
        gap_before = (e_pred_before - e_true_before).mean().item()

    # One gradient step
    opt.zero_grad()
    loss = corrector.corrective_loss(energy_net, critical_batch)
    loss.backward()
    opt.step()

    # Measure gap after
    with torch.no_grad():
        e_true_after = energy_net(x, y)
        e_pred_after = energy_net(x, y_pred)
        gap_after = (e_pred_after - e_true_after).mean().item()

    print(f"  Gap before: {gap_before:.4f}, Gap after: {gap_after:.4f}")
    assert gap_after > gap_before, (
        f"Gap should increase. Before: {gap_before}, After: {gap_after}"
    )
    print("  PASSED: corrective loss pushes gap in correct direction")


def test_empty_critical_set():
    """Verify corrective_loss returns 0.0 with no division by zero."""
    torch.manual_seed(42)
    energy_net = EnergyNet(10, 20, 5, 20)
    corrector = EnergyCorrector(alpha=1.0)

    loss = corrector.corrective_loss(energy_net, [])
    assert loss.item() == 0.0, f"Expected 0.0, got {loss.item()}"

    # Verify it's differentiable (no crash on backward)
    loss.backward()
    print("  PASSED: empty critical set returns 0.0, no crash")


def test_sample_batch():
    """Verify sample_batch handles empty and non-empty sets."""
    corrector = EnergyCorrector(alpha=1.0)

    # Empty
    batch = corrector.sample_batch(10)
    assert len(batch) == 0

    # Non-empty
    corrector.critical_set = [
        {"x": torch.randn(5), "y": torch.ones(3), "y_pred": torch.zeros(3),
         "task_error": torch.tensor(0.5), "delta": torch.tensor(-0.1),
         "criticality": torch.tensor((i + 1) / 20.0)}
        for i in range(20)
    ]
    batch = corrector.sample_batch(5)
    assert len(batch) == 5

    batch = corrector.sample_batch(100)  # more than available
    assert len(batch) == 20

    print("  PASSED: sample_batch works correctly")


if __name__ == "__main__":
    print("Test 1: diagnose finds known inversions")
    test_diagnose_finds_known_inversions()
    print()

    print("Test 2: corrective loss gradient direction")
    test_corrective_loss_gradient_direction()
    print()

    print("Test 3: empty critical set")
    test_empty_critical_set()
    print()

    print("Test 4: sample_batch")
    test_sample_batch()
    print()

    print("All tests passed!")
