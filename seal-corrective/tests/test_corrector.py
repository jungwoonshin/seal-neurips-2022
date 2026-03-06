"""
Unit tests for C-SEAL EnergyCorrector.

Tests:
1. Energy loss shape and values
2. Logistic margin loss gradient direction
3. Full pipeline differentiability
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from models import TaskNet, EnergyNet
from losses import EnergyCorrector


def make_toy_data(n=32, input_dim=10, num_labels=5, seed=0):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, input_dim, generator=gen)
    y = (torch.rand(n, num_labels, generator=gen) > 0.5).float()
    return x, y


def test_energy_loss_shape():
    torch.manual_seed(42)
    input_dim, num_labels = 10, 5

    energy_net = EnergyNet(input_dim, 20, num_labels, 20)
    task_net = TaskNet(input_dim, 20, num_labels)
    corrector = EnergyCorrector()

    x, y = make_toy_data(16, input_dim, num_labels)
    loss, info = corrector.energy_loss(energy_net, task_net, x, y)

    assert loss.shape == (), f"Expected scalar loss, got {loss.shape}"
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert info["batch_size"] == 16
    assert info["loss"] > 0.0
    print(f"  Loss: {info['loss']:.4f}")
    print("  PASSED: energy loss has correct shape and is valid")


def test_loss_gradient_direction():
    """One gradient step should push E(y*) below E(F(x))."""
    torch.manual_seed(42)
    input_dim, num_labels = 10, 5

    energy_net = EnergyNet(input_dim, 20, num_labels, 20)
    task_net = TaskNet(input_dim, 20, num_labels)
    opt = torch.optim.SGD(energy_net.parameters(), lr=0.1)
    corrector = EnergyCorrector()

    x, y = make_toy_data(16, input_dim, num_labels)

    loss_before, _ = corrector.energy_loss(energy_net, task_net, x, y)
    loss_val_before = loss_before.item()

    opt.zero_grad()
    loss, _ = corrector.energy_loss(energy_net, task_net, x, y)
    loss.backward()
    opt.step()

    with torch.no_grad():
        loss_after, _ = corrector.energy_loss(energy_net, task_net, x, y)

    print(f"  Loss before: {loss_val_before:.4f}, after: {loss_after.item():.4f}")
    assert loss_after.item() < loss_val_before, (
        f"Expected loss to decrease. Before: {loss_val_before}, After: {loss_after.item()}")
    print("  PASSED: logistic margin loss decreases after gradient step")


def test_differentiability():
    torch.manual_seed(42)
    input_dim, num_labels = 10, 5

    energy_net = EnergyNet(input_dim, 20, num_labels, 20)
    task_net = TaskNet(input_dim, 20, num_labels)
    corrector = EnergyCorrector()

    x, y = make_toy_data(32, input_dim, num_labels)
    loss, info = corrector.energy_loss(energy_net, task_net, x, y)

    print(f"  Loss: {info['loss']:.4f}")
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in energy_net.parameters())
    assert has_grad
    print("  PASSED: energy loss is differentiable w.r.t. energy net")


if __name__ == "__main__":
    tests = [
        ("Test 1: energy loss shape", test_energy_loss_shape),
        ("Test 2: logistic margin loss gradient direction", test_loss_gradient_direction),
        ("Test 3: differentiability", test_differentiability),
    ]

    for name, fn in tests:
        print(name)
        fn()
        print()

    print("All tests passed!")
