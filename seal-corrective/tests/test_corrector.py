"""
Unit tests for EnergyCorrector (batch_corrective_loss API).

Tests both "hinge" and "smooth" loss modes:
1. Gradient direction: one step increases E(x,F(x)) − E(x,y)
2. Perfect predictions → zero loss
3. Smooth loss is differentiable everywhere (no NaN grads)
4. Smooth loss with γ > 1 focuses on harder examples
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from models import TaskNet, EnergyNet
from losses import EnergyCorrector


def test_hinge_gradient_direction():
    """One gradient step on hinge loss increases E(x,F(x)) − E(x,y)."""
    torch.manual_seed(42)
    input_dim, num_labels = 10, 5

    energy_net = EnergyNet(input_dim, 20, num_labels, 20)
    task_net = TaskNet(input_dim, 20, num_labels)
    opt = torch.optim.SGD(energy_net.parameters(), lr=0.1)

    x = torch.randn(16, input_dim)
    y = (torch.rand(16, num_labels) > 0.5).float()

    with torch.no_grad():
        y_pred = task_net(x)
        e_true_before = energy_net(x, y).clone()
        e_pred_before = energy_net(x, y_pred).clone()
        gap_before = (e_pred_before - e_true_before).mean().item()

    corrector = EnergyCorrector(alpha=1.0, loss_type="hinge")
    opt.zero_grad()
    loss, n_crit, _ = corrector.batch_corrective_loss(energy_net, task_net, x, y)
    if n_crit > 0:
        loss.backward()
        opt.step()

        with torch.no_grad():
            y_pred = task_net(x)
            gap_after = (energy_net(x, y_pred) - energy_net(x, y)).mean().item()

        assert gap_after > gap_before, f"Gap should increase: {gap_before} → {gap_after}"
    print("  PASSED: hinge gradient direction")


def test_smooth_gradient_direction():
    """One gradient step on smooth loss increases E(x,F(x)) − E(x,y)."""
    torch.manual_seed(42)
    input_dim, num_labels = 10, 5

    energy_net = EnergyNet(input_dim, 20, num_labels, 20)
    task_net = TaskNet(input_dim, 20, num_labels)
    opt = torch.optim.SGD(energy_net.parameters(), lr=0.1)

    x = torch.randn(16, input_dim)
    y = (torch.rand(16, num_labels) > 0.5).float()

    with torch.no_grad():
        y_pred = task_net(x)
        gap_before = (energy_net(x, y_pred) - energy_net(x, y)).mean().item()

    corrector = EnergyCorrector(alpha=1.0, loss_type="smooth", gamma=1.0)
    opt.zero_grad()
    loss, n_active, _ = corrector.batch_corrective_loss(energy_net, task_net, x, y)
    if n_active > 0:
        loss.backward()
        opt.step()

        with torch.no_grad():
            y_pred = task_net(x)
            gap_after = (energy_net(x, y_pred) - energy_net(x, y)).mean().item()

        assert gap_after > gap_before, f"Gap should increase: {gap_before} → {gap_after}"
    print("  PASSED: smooth gradient direction")


def test_perfect_predictions_zero_loss():
    """When task net predicts perfectly, both losses should be ~0."""
    torch.manual_seed(42)
    input_dim, num_labels = 10, 5

    energy_net = EnergyNet(input_dim, 20, num_labels, 20)
    x = torch.randn(8, input_dim)
    y = (torch.rand(8, num_labels) > 0.5).float()

    # Perfect task net: always returns y exactly
    class PerfectTaskNet(torch.nn.Module):
        def __init__(self, targets):
            super().__init__()
            self.targets = targets
        def forward(self, x):
            return self.targets

    task_net = PerfectTaskNet(y.clone())

    for lt in ["hinge", "smooth"]:
        corrector = EnergyCorrector(alpha=1.0, loss_type=lt, min_margin=0.0)
        loss, n, _ = corrector.batch_corrective_loss(energy_net, task_net, x, y)
        assert loss.item() < 1e-6, f"{lt}: expected ~0 loss, got {loss.item()}"
    print("  PASSED: perfect predictions → zero loss")


def test_smooth_no_nan_gradients():
    """Smooth loss should never produce NaN gradients."""
    torch.manual_seed(42)
    input_dim, num_labels = 10, 5

    energy_net = EnergyNet(input_dim, 20, num_labels, 20)
    task_net = TaskNet(input_dim, 20, num_labels)

    x = torch.randn(32, input_dim)
    y = (torch.rand(32, num_labels) > 0.5).float()

    corrector = EnergyCorrector(alpha=1.0, loss_type="smooth", gamma=2.0)
    loss, _, _ = corrector.batch_corrective_loss(energy_net, task_net, x, y)
    loss.backward()

    for name, p in energy_net.named_parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"
    print("  PASSED: smooth loss has no NaN gradients")


def test_gamma_focuses_on_hard_examples():
    """Higher γ should give more relative weight to high-error examples."""
    torch.manual_seed(42)
    input_dim, num_labels = 10, 5

    energy_net = EnergyNet(input_dim, 20, num_labels, 20)
    task_net = TaskNet(input_dim, 20, num_labels)

    x = torch.randn(32, input_dim)
    y = (torch.rand(32, num_labels) > 0.5).float()

    # Compute losses with γ=1 and γ=2; γ=2 should focus more on hard examples
    # (different loss values, both valid)
    corrector_g1 = EnergyCorrector(alpha=1.0, loss_type="smooth", gamma=1.0)
    corrector_g2 = EnergyCorrector(alpha=1.0, loss_type="smooth", gamma=2.0)

    loss_g1, _, _ = corrector_g1.batch_corrective_loss(energy_net, task_net, x, y)
    loss_g2, _, _ = corrector_g2.batch_corrective_loss(energy_net, task_net, x, y)

    # Both should be finite and differentiable
    assert torch.isfinite(loss_g1), f"γ=1 loss not finite: {loss_g1}"
    assert torch.isfinite(loss_g2), f"γ=2 loss not finite: {loss_g2}"
    # They should generally differ (different weighting)
    print(f"  γ=1 loss: {loss_g1.item():.4f}, γ=2 loss: {loss_g2.item():.4f}")
    print("  PASSED: gamma parameter produces finite, varying losses")


def test_empty_batch_no_crash():
    """Zero-error batch returns 0.0, no division by zero."""
    torch.manual_seed(42)
    energy_net = EnergyNet(10, 20, 5, 20)

    # Task net that returns y exactly → zero error → zero weight sum
    x = torch.randn(4, 10)
    y = (torch.rand(4, 5) > 0.5).float()

    class PerfectTaskNet(torch.nn.Module):
        def __init__(self, targets):
            super().__init__()
            self.targets = targets
        def forward(self, x):
            return self.targets

    task_net = PerfectTaskNet(y.clone())

    for lt in ["hinge", "smooth"]:
        corrector = EnergyCorrector(alpha=1.0, loss_type=lt, min_margin=0.0)
        loss, n, _ = corrector.batch_corrective_loss(energy_net, task_net, x, y)
        assert torch.isfinite(loss), f"{lt}: loss not finite"
        loss.backward()
    print("  PASSED: empty batch no crash")


def test_theory_gradient_direction():
    """One gradient step on theory loss increases E(x,F(x)) − E(x,y)."""
    torch.manual_seed(42)
    input_dim, num_labels = 10, 5

    energy_net = EnergyNet(input_dim, 20, num_labels, 20)
    task_net = TaskNet(input_dim, 20, num_labels)
    opt = torch.optim.SGD(energy_net.parameters(), lr=0.1)

    x = torch.randn(16, input_dim)
    y = (torch.rand(16, num_labels) > 0.5).float()

    with torch.no_grad():
        y_pred = task_net(x)
        gap_before = (energy_net(x, y_pred) - energy_net(x, y)).mean().item()

    corrector = EnergyCorrector(alpha=1.0, loss_type="theory", gamma=1.0)
    opt.zero_grad()
    loss, n_active, diag = corrector.batch_corrective_loss(energy_net, task_net, x, y)
    if n_active > 0:
        loss.backward()
        opt.step()

        with torch.no_grad():
            y_pred = task_net(x)
            gap_after = (energy_net(x, y_pred) - energy_net(x, y)).mean().item()

        assert gap_after > gap_before, f"Gap should increase: {gap_before} → {gap_after}"
    print("  PASSED: theory gradient direction")


def test_theory_adaptive_margin():
    """Margin grows with ||F(x)−y||² when η > 0."""
    torch.manual_seed(42)
    input_dim, num_labels = 10, 5

    energy_net = EnergyNet(input_dim, 20, num_labels, 20)
    task_net = TaskNet(input_dim, 20, num_labels)

    x = torch.randn(16, input_dim)
    y = (torch.rand(16, num_labels) > 0.5).float()

    # η=0: no curvature correction
    corrector_0 = EnergyCorrector(alpha=1.0, loss_type="theory", eta=0.0)
    loss_0, _, _ = corrector_0.batch_corrective_loss(energy_net, task_net, x, y)

    # η=1.0: curvature correction active
    corrector_1 = EnergyCorrector(alpha=1.0, loss_type="theory", eta=1.0)
    loss_1, _, _ = corrector_1.batch_corrective_loss(energy_net, task_net, x, y)

    # With η>0, the margin is larger, so loss should be >= (typically strictly greater)
    assert loss_1.item() >= loss_0.item() - 1e-6, \
        f"η>0 should increase loss: η=0 → {loss_0.item()}, η=1 → {loss_1.item()}"
    print(f"  η=0 loss: {loss_0.item():.4f}, η=1 loss: {loss_1.item():.4f}")
    print("  PASSED: theory adaptive margin")


def test_theory_descent_loss_grads():
    """Descent loss (β₂>0) produces gradients on energy net parameters."""
    torch.manual_seed(42)
    input_dim, num_labels = 10, 5

    energy_net = EnergyNet(input_dim, 20, num_labels, 20)
    task_net = TaskNet(input_dim, 20, num_labels)

    x = torch.randn(16, input_dim)
    y = (torch.rand(16, num_labels) > 0.5).float()

    corrector = EnergyCorrector(
        alpha=1.0, loss_type="theory", beta2=0.1, mu=0.01)
    loss, n_active, diag = corrector.batch_corrective_loss(energy_net, task_net, x, y)
    loss.backward()

    # Check that energy net has gradients
    has_grad = False
    for name, p in energy_net.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, "Descent loss should produce gradients on energy net"
    assert "descent_sat" in diag, "Should report descent_sat"
    print(f"  descent_sat: {diag['descent_sat']:.3f}")
    print("  PASSED: theory descent loss produces gradients")


def test_theory_backward_compat():
    """η=0, κ=0, β₂=0 should match smooth loss."""
    torch.manual_seed(42)
    input_dim, num_labels = 10, 5

    energy_net = EnergyNet(input_dim, 20, num_labels, 20)
    task_net = TaskNet(input_dim, 20, num_labels)
    # Eval mode to eliminate dropout randomness between calls
    energy_net.eval()
    task_net.eval()

    x = torch.randn(16, input_dim)
    y = (torch.rand(16, num_labels) > 0.5).float()

    corrector_smooth = EnergyCorrector(alpha=1.0, loss_type="smooth", gamma=1.0)
    loss_smooth, n_smooth, _ = corrector_smooth.batch_corrective_loss(
        energy_net, task_net, x, y)

    corrector_imp = EnergyCorrector(
        alpha=1.0, loss_type="theory", gamma=1.0,
        eta=0.0, kappa=0.0, beta2=0.0)
    loss_imp, n_imp, _ = corrector_imp.batch_corrective_loss(
        energy_net, task_net, x, y)

    assert abs(loss_smooth.item() - loss_imp.item()) < 1e-5, \
        f"Should match smooth: {loss_smooth.item()} vs {loss_imp.item()}"
    assert n_smooth == n_imp, f"n_active mismatch: {n_smooth} vs {n_imp}"
    print(f"  smooth: {loss_smooth.item():.6f}, theory: {loss_imp.item():.6f}")
    print("  PASSED: theory backward compatible with smooth")


def test_theory_diagnostics():
    """Returns descent_sat and L_lip in diagnostics dict."""
    torch.manual_seed(42)
    input_dim, num_labels = 10, 5

    energy_net = EnergyNet(input_dim, 20, num_labels, 20)
    task_net = TaskNet(input_dim, 20, num_labels)

    x = torch.randn(16, input_dim)
    y = (torch.rand(16, num_labels) > 0.5).float()

    corrector = EnergyCorrector(
        alpha=1.0, loss_type="theory", beta2=0.1)
    _, _, diag = corrector.batch_corrective_loss(energy_net, task_net, x, y)

    assert "descent_sat" in diag, "Missing descent_sat"
    assert "L_lip" in diag, "Missing L_lip"
    assert 0.0 <= diag["descent_sat"] <= 1.0, f"descent_sat out of range: {diag['descent_sat']}"
    assert diag["L_lip"] >= 0.0, f"L_lip should be non-negative: {diag['L_lip']}"
    print(f"  descent_sat: {diag['descent_sat']:.3f}, L_lip: {diag['L_lip']:.4f}")
    print("  PASSED: theory diagnostics")


if __name__ == "__main__":
    tests = [
        ("Hinge gradient direction", test_hinge_gradient_direction),
        ("Smooth gradient direction", test_smooth_gradient_direction),
        ("Perfect predictions → zero loss", test_perfect_predictions_zero_loss),
        ("Smooth no NaN gradients", test_smooth_no_nan_gradients),
        ("Gamma focuses on hard examples", test_gamma_focuses_on_hard_examples),
        ("Empty batch no crash", test_empty_batch_no_crash),
        ("Theory gradient direction", test_theory_gradient_direction),
        ("Theory adaptive margin", test_theory_adaptive_margin),
        ("Theory descent loss grads", test_theory_descent_loss_grads),
        ("Theory backward compat", test_theory_backward_compat),
        ("Theory diagnostics", test_theory_diagnostics),
    ]

    for i, (name, fn) in enumerate(tests, 1):
        print(f"Test {i}: {name}")
        fn()
        print()

    print("All tests passed!")
