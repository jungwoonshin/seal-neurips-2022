# SEAL Corrective Energy Alignment

## 1. Problem Setting

Multi-label classification: given input **x**, predict a binary label vector **y ∈ {0,1}^L** (e.g., L = 159 for Bibtex). Labels exhibit co-occurrence structure and dependencies that a naive per-label BCE classifier ignores.

## 2. Architecture

The system uses two separate networks trained via alternating optimization.

### Task Network F_φ(x) → [0,1]^L

A 3-layer MLP (input → 768 → ReLU → dropout → 768 → ReLU → dropout → L → sigmoid) that directly predicts soft label probabilities.

### Energy Network E_θ(x, y) → ℝ

Scores how compatible an (input, label) pair is; lower energy means better fit. It decomposes into two terms:

- **E_local(x, y)** = Σ_i y_i · b_i^T h(x) — per-label compatibility with a learned feature embedding h(x).
- **E_global(y)** = v^T softplus(M·y) — label-label dependencies via a mixing matrix M ∈ ℝ^{d×L}, independent of x.

## 3. Training Procedure

For each mini-batch (x, y), the procedure alternates between two steps.

### Step 1: Update θ (energy net), φ frozen

The energy net is trained via a **corrective loss** targeting examples where the energy surface is misaligned with the task net's errors. Two loss formulations are supported: a hinge loss (original) and a smooth probabilistic loss.

**Task error ℓ(F(x), y)** is computed by `_compute_error` (corrective.py:136–155). By default (with `task_error_metric="f1"`):

```
ℓ(F(x), y) = 1 − soft_F1(F(x), y)
```

where soft F1 (corrective.py:162–168) is:

```
soft_F1 = 2 · (F(x) · y) / (sum(F(x)) + sum(y) + ε)
```

This uses soft predictions directly (no thresholding), so it is differentiable. The other supported metrics are:

- **"hamming":** fraction of labels that differ (uses hard 0.5 threshold).
- **"structural":** pairwise co-occurrence mismatch between predicted and true label matrices.

The task error ℓ plays the same role as Δ(ỹ, y) in the SEAL margin loss — it measures how wrong the prediction is — but it is computed from the task net's actual output rather than from an adversarially chosen ỹ.

The total energy loss is L_θ = β · L_correct (or L_smooth).

### Step 2: Update φ (task net), θ frozen

```
L_φ = λ₁ · E_θ(x, F_φ(x)) + λ₂ · BCE_weighted(F_φ(x), y*)
```

The energy term (λ₁ = 0.01) transfers label-structure knowledge from the energy net. The weighted BCE term (λ₂ = 1.0) provides standard per-label supervision with pos_weight = (1 − freq) / freq, clamped at 50.

## 4. Loss Formulations

The system supports two corrective loss formulations, selected via the `loss_type` parameter.

### 4a. Hinge Loss (`loss_type="hinge"`, original)

**Critical set definition:**

```
C = { (x, y) | δ(x) < max(α · ℓ(F(x), y), min_margin) AND ℓ(F(x), y) > 0 }
```

where δ(x) = E_θ(x, F(x)) − E_θ(x, y) is the energy gap.

A sample is critical when the task net is making an error **and** the energy net fails to assign sufficiently lower energy to the ground truth relative to the prediction.

**Corrective hinge loss:**

```
L_correct(θ) = (1/|C|) Σ_{(x,y)∈C} [ max(α·ℓ(F(x),y), min_margin) + E_θ(x, y) − E_θ(x, F(x)) ]₊
```

The sum is over the critical set C and normalized by |C|, not the full batch. The hinge pushes ground-truth energy below prediction energy by a margin proportional to the task error.

**Properties:**
- Hard binary filtering: examples are either in C or out.
- The hinge `[·]₊` gives zero loss when the margin is satisfied, linear penalty when violated.
- Requires `min_margin` hyperparameter to prevent margin collapse when task error is small.

### 4b. Smooth Loss (`loss_type="smooth"`)

Derived from a probabilistic interpretation: maximize the log-probability that the energy net correctly ranks ground truth below the prediction, weighted by task error severity.

**Smooth corrective loss:**

```
L_smooth(θ) = (1/Σw) Σ_{(x,y)∈B} w(x) · log(1 + exp(α·ℓ + E_θ(x,y) − E_θ(x,F(x))))
```

where:
- **w(x) = ℓ(x)^γ** — soft task-error weighting. γ=1 gives linear weighting; γ>1 provides focal-loss-like focus on hard examples.
- The **log-sigmoid penalty** `log(1 + exp(v))` is implemented as `F.softplus(v)` for numerical stability.
- **No hard critical set filtering** — every example in the batch contributes, weighted by its task error.
- **No `min_margin` needed** — soft weighting naturally down-weights near-zero-error examples.
- **Normalized by Σw** instead of |C|, keeping the loss scale invariant to error distribution.

**Derivation:** Model the probability that the energy ranking is correct as P(y ≻ F(x) | x) = σ(δ(x) − α·ℓ), where σ is the sigmoid. Minimize the negative log-likelihood: −log σ(−v) = log(1 + exp(v)). This is the softplus with temperature 1, giving the loss a principled probabilistic interpretation.

**Gradient analysis for a single example:**

```
∂L/∂θ = w(x) · σ(α·ℓ + E(x,y) − E(x,F(x))) · [∂E(x,y)/∂θ − ∂E(x,F(x))/∂θ]
```

The sigmoid factor σ(·) acts as a **soft gate**: when the margin is well satisfied (large negative argument), σ → 0 and the gradient vanishes. When the margin is badly violated (large positive argument), σ → 1 and you get the full gradient. Compare with the hinge, where this gate is binary.

**Properties:**
- Smooth gradients everywhere — no discontinuities at the margin boundary.
- Examples near the boundary contribute small but nonzero gradients, providing a continuous signal.
- One fewer hyperparameter than hinge (no `min_margin`), one additional (γ).

### Comparison of Loss Formulations

| Component | Hinge | Smooth |
| --- | --- | --- |
| Penalty function | `[v]₊` (ReLU) | `log(1 + exp(v))` (softplus) |
| Example filtering | Hard critical set C | Soft weighting w(x) = ℓ^γ |
| Margin | `max(α·ℓ, min_margin)` | `α·ℓ` (linear, no floor) |
| Normalization | `1/\|C\|` | `1/Σw` |
| Gradient at boundary | Discontinuous (binary gate) | Smooth (sigmoid gate) |
| Hyperparameters | α, min_margin | α, γ |

## 5. Key Design Choices

| Component | Choice | Rationale |
| --- | --- | --- |
| Gradient isolation | Separate AdamW optimizers for θ and φ | Prevents entangled updates |
| y_pred detached in Step 1 | `torch.no_grad()` on task net | Energy net learns from current predictions without backprop through them |
| Adaptive margin | Proportional to task error ℓ | Larger errors demand larger energy gaps |
| LR scheduling | ReduceLROnPlateau on val F1 | Patience=10, factor=0.5 for both networks |
| Checkpointing | Restore best val F1 model | Avoids overfitting in later epochs |

## 6. Comparison with SEAL Paper's Margin Loss

The original SEAL (NeurIPS 2022) learns an energy model directly on the entire dataset using the classic SSVM loss:

```
L_E_margin = Σ_{x,y} max_{ỹ} [ Δ(ỹ, y) − E_θ(x, ỹ) + E_θ(x, y) ]₊
```

| Aspect | SEAL Paper (L_E_margin) | Corrective Loss |
| --- | --- | --- |
| **Negative label** | max_ỹ — adversarial search over all ỹ | F(x) — the task net's current prediction |
| **Margin term** | Δ(ỹ, y) — fixed task loss | α·ℓ (scaled task error, optionally with floor) |
| **Summation** | Over all (x, y) | Critical set (hinge) or weighted full batch (smooth) |
| **Normalization** | Sum (no averaging) | Mean over critical set or weight-normalized |

**Core conceptual shift:** SEAL's margin loss performs adversarial search to find the worst-case violator ỹ across the entire label space for every training example, shaping a globally correct energy surface. The corrective loss eliminates this search entirely — it uses the task net's own prediction F(x), and normalizes by the effective set size.

### Eliminating Explicit Negative Sampling

SEAL's margin loss requires an explicit procedure to produce ỹ — whether GBI (gradient-based inference), random corruption, or InfNet — searching the combinatorial label space for hard negatives. The corrective loss requires no such step. F(x) is not a "negative sample" found by search; it is simply the task net's forward pass, which already happens as part of training.

This provides three benefits:

1. **No combinatorial search** over the label space — no GBI, no proposal distribution, no sampling.
2. **Automatically hard negatives** — as the task net improves, its mistakes become increasingly informative contrasts without extra machinery.
3. **Simpler objective** — the only contrast is ground truth vs. current prediction, directly targeting actual mistakes.

### Main Advantages over Original SEAL

- **Targeted updates:** Corrects only where the energy surface is misaligned (via critical set or soft weighting), rather than pushing the energy surface everywhere.
- **Explicit separation of prediction and structure:** The task net handles prediction via BCE; the energy net handles structural consistency. Each can be debugged and ablated independently.
- **Avoids over-fitting the energy:** No corrective gradient when the task is correct (hinge excludes them; smooth down-weights them to near zero).
- **Error-scaled margins:** The required energy gap is proportional to the task error, making correction proportionally aggressive where the task is most wrong.

## 7. Concrete Examples

Setup: 4 labels, ground truth y = [1, 1, 0, 0], task net predicts F(x) = [0.8, 0.3, 0.1, 0.2]. The task error is ℓ = 1 − soft_F1 = 0.353.

### Case A: Energy surface correct, sufficient margin (not critical)

E(x, y) = −2.0, E(x, F(x)) = −1.2. The gap δ = 0.8 exceeds the required margin of 0.353. The energy gradient naturally helps — no correction needed.

- Hinge: violation = 0.353 + (−2.0) − (−1.2) = −0.447 → loss = 0 (not in C).
- Smooth: softplus(−0.447) = 0.48, weighted by 0.353^γ → small contribution.

### Case B: Energy surface inverted (critical)

E(x, y) = −1.0, E(x, F(x)) = −1.5. The energy net thinks the wrong prediction is better (lower energy). The energy gradient actively reinforces the mistake.

- Hinge: [0.353 + (−1.0) − (−1.5)]₊ = 0.853 → strong correction.
- Smooth: softplus(0.853) = 1.21, weighted by 0.353^γ → strong correction.

### Case C: Energy surface correct but margin too small (critical)

E(x, y) = −1.5, E(x, F(x)) = −1.4. The ranking is technically correct, but the gap of 0.1 is too weak relative to the error of 0.353.

- Hinge: [0.353 + (−1.5) − (−1.4)]₊ = 0.253 → moderate correction.
- Smooth: softplus(0.253) = 0.84, weighted by 0.353^γ → moderate correction.

### Case D: Task net correct (never critical)

If F(x) ≈ y then ℓ ≈ 0. Hinge excludes this example (ℓ > 0 fails). Smooth gives it near-zero weight (0^γ ≈ 0).

| Case | Task wrong? | Energy ranking | Hinge | Smooth |
| --- | --- | --- | --- | --- |
| A | Yes | Correct, large gap | Loss = 0 | Negligible (softplus decays) |
| B | Yes | Inverted | Strong correction | Strong correction |
| C | Yes | Correct, small gap | Moderate correction | Moderate correction |
| D | No | Irrelevant | Excluded (ℓ = 0) | Weight ≈ 0 |

## 8. Intuition

The system implements a **diagnose-then-correct loop:**

1. The energy net should assign lower energy to ground truth than to wrong predictions.
2. When this property is violated (the energy surface is inverted or has insufficient margin), the corrective loss fixes those regions — either surgically (hinge) or smoothly (softplus).
3. Once corrected, the energy surface provides a useful gradient signal to the task net, pushing predictions toward lower-energy (more compatible) label configurations.
4. As the task net improves, different examples become critical, and the energy net adapts.

This creates a **virtuous cycle** where the energy net learns label-space structure and the task net exploits that structure.

## 9. Hyperparameters

| Param | Default | Role |
| --- | --- | --- |
| `lr_energy` | 0.001 | Energy net learning rate |
| `lr_task` | 0.001 | Task net learning rate |
| `lambda1` | 0.01 | Weight of energy signal in task loss |
| `lambda2` | 1.0 | Weight of BCE in task loss |
| `beta` | 0.1 | Weight of corrective loss |
| `alpha` | 1.0 | Margin scaling factor |
| `min_margin` | 0.1 | Floor on hinge margin (hinge only) |
| `loss_type` | "smooth" | Loss formulation: "hinge" or "smooth" |
| `gamma` | 1.0 | Task-error weighting exponent (smooth only) |
| `hidden_dim` | 768 | MLP width for both networks |
| `energy_hidden` | 768 | Dimension of global label-mixing layer |

## 10. Results

### Test Instance-F1 (%) Comparison

| Method | Bibtex | Delicious | Genbase | CAL500 | Eurlex-ev | Expr_fun | SPO_fun |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Cross-entropy (CE)** | 42.40 | 29.89 | 47.37 | 33.58 | 42.19 | 37.50 | 27.81 |
| | | | | | | | |
| **Energy network with GBI** | | | | | | | |
| SPEN | 42.99 | 24.20 | 32.13 | 37.24 | 41.86 | 36.74 | 27.95 |
| DVN | **45.95** | 24.87 | 77.92 | **47.74** | 25.49 | 31.47 | 29.35 |
| NCE ranking | 12.95 | 12.69 | 12.40 | 33.89 | 0.19 | 27.74 | 18.06 |
| | | | | | | | |
| **SEAL-static** | | | | | | | |
| margin | 43.11 | 28.08 | 57.45 | 33.91 | 42.15 | 38.13 | 28.15 |
| regression | 42.29 | 30.09 | 96.68 | 37.63 | 42.18 | 33.12 | 28.42 |
| NCE ranking | 43.03 | 30.08 | 37.82 | 37.82 | 42.11 | 37.78 | 28.29 |
| | | | | | | | |
| **SEAL-dynamic** | | | | | | | |
| margin (InfNet) | 42.86 | 29.75 | 96.53 | 36.69 | 41.83 | 37.81 | 28.43 |
| regression | 43.74 | 29.79 | 96.95 | 37.97 | 41.67 | 37.95 | 28.83 |
| regression-s | 44.53 | 29.87 | 97.81 | 38.95 | 42.17 | 37.95 | — |
| NCE ranking | 44.76 | 34.67 | 97.32 | 41.62 | 41.62 | **38.28** | 28.83 |
| | | | | | | | |
| **Ours (Hinge)** | 44.35 | **36.05** | **98.99** | 41.35 | **47.95** | 37.32 | **31.78** |
| **Ours (Smooth, γ=1)** | 44.08 | 35.51 | — | **43.22** | — | 37.58 | — |

Bold indicates best result per dataset.

### Dataset Statistics

| Dataset | Features | Labels | Train | Val | Test |
| --- | --- | --- | --- | --- | --- |
| Bibtex | 1,836 | 159 | 4,407 | 1,491 | 1,497 |
| CAL500 | 68 | 174 | 283 | 105 | 114 |
| Delicious | 500 | 983 | 9,698 | 3,210 | 3,197 |
| Eurlex-ev | 5,000 | 3,993 | 11,581 | 3,883 | 3,884 |
| Expr_fun | 561 | 500 | 1,639 | 849 | 1,291 |
| Genbase | 1,185 | 27 | 398 | 132 | 132 |
| SPO_fun | 86 | 500 | 1,600 | 837 | 1,266 |

### Training Time Comparison (sec/epoch)

Baseline values from the original paper (Appendix D, Table 19), measured on TitanX GPU. Ours measured on local hardware.

| Method | Bibtex | CAL500 | Delicious | Eurlex-ev | Expr_fun | Genbase | SPO_fun |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CE | 22.67 | 1.00 | 33.01 | 114.84 | 6.29 | 2.12 | 5.46 |
| SPEN | 28.04 | 2.53 | 37.31 | 136.10 | 10.99 | 3.57 | 11.01 |
| DVN | 32.12 | 1.87 | 55.24 | 129.94 | 10.95 | 2.79 | 10.02 |
| NCE (Energy Net) | 13.97 | 2.82 | 22.03 | 89.33 | 3.71 | 3.68 | 3.83 |
| SEAL-margin | 212.94 | 5.78 | 44.60 | 260.10 | 27.96 | 8.73 | 35.24 |
| SEAL-regression | 27.78 | 8.07 | 143.90 | 352.63 | 46.32 | 13.91 | 56.97 |
| SEAL-regression-s | 179.97 | 8.48 | 45.33 | 221.33 | 42.87 | 10.79 | 77.92 |
| SEAL-NCE | 131.73 | 16.58 | 158.97 | 431.92 | 72.43 | 17.77 | 40.34 |
| SEAL-Ranking | 218.37 | 7.04 | 317.63 | 408.79 | 41.24 | 10.74 | 26.83 |
| **Ours (Corrective)** | 1.93 | 0.13 | 3.63 | 5.05 | 0.81 | 0.19 | 0.80 |

### Detailed Results: Hinge Loss

| Dataset | Val F1 | Test F1 | Best Epoch | Epochs | Total Time | sec/epoch |
| --- | --- | --- | --- | --- | --- | --- |
| Bibtex | 0.4549 | 0.4435 | 32 | 70 | 135.4s | 1.93 |
| CAL500 | 0.4171 | 0.4135 | 59 | 70 | 8.9s | 0.13 |
| Delicious | 0.3621 | 0.3605 | 14 | 70 | 254.3s | 3.63 |
| Eurlex-ev | 0.4827 | 0.4795 | 171 | 300 | 1516.0s | 5.05 |
| Expr_fun | 0.3760 | 0.3732 | 60 | 70 | 56.9s | 0.81 |
| Genbase | 0.9932 | 0.9899 | 18 | 70 | 13.2s | 0.19 |
| SPO_fun | 0.3042 | 0.3178 | 22 | 70 | 55.9s | 0.80 |

### Detailed Results: Smooth Loss (γ=1.0)

| Dataset | Val F1 | Test F1 | Best Epoch | Epochs | Total Time |
| --- | --- | --- | --- | --- | --- |
| Bibtex | 0.4569 | 0.4408 | 42 | 300 | 780.1s |
| CAL500 | 0.4191 | 0.4322 | 27 | 300 | 46.6s |
| Delicious | 0.3601 | 0.3551 | 14 | 300 | 2354.4s |
| Expr_fun | 0.3729 | 0.3758 | 38 | 300 | 412.8s |

### Configuration

Shared: lr_energy = lr_task = 0.001, β = 0.1, α = 1.0, λ₁ = 0.01, λ₂ = 1.0, hidden_dim = energy_hidden = 768, batch_size = 32, AdamW (weight decay 1e-4), ReduceLROnPlateau (patience 10, factor 0.5).

Hinge-specific: min_margin = 0.1, epochs = 70 (eurlex_ev: 300).

Smooth-specific: γ = 1.0, epochs = 300. No min_margin.
