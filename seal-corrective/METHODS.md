# SEAL: Corrective Energy Alignment for Multi-Label Classification

## 1. Overview

This work introduces **Corrective Energy Alignment**, a framework for structured energy-based multi-label classification. The central insight is that energy-based models (EBMs) can develop **energy surface inversions** — pathological states where the energy function assigns lower energy to incorrect predictions than to ground truth labels — even when the task network produces erroneous outputs. The proposed method corrects these inversions through a targeted hinge-based loss applied per batch.

The framework consists of three tightly coupled components:

1. A **task network** $F_\Phi(x)$ that maps inputs to soft multi-label predictions.
2. An **energy network** $E_\Theta(x, y)$ that scores input–label compatibility.
3. A **corrective mechanism** that identifies energy surface inversions within each batch and applies targeted corrections via a hinge-based loss.

---

## 2. Model Architecture

### 2.1 Task Network $F_\Phi$

The task network is a two-layer feedforward network with ReLU activations, dropout regularization ($p = 0.3$), and a sigmoid output layer:

$$
F_\Phi(x) = \sigma\bigl(W_3 \cdot \text{ReLU}(W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2) + b_3\bigr)
$$

The output $\hat{y} = F_\Phi(x) \in (0, 1)^L$ gives soft predictions over $L$ labels.

### 2.2 Energy Network $E_\Theta$

The energy function decomposes into **local** and **global** terms:

$$
E_\Theta(x, y) = E_{\text{local}}(x, y) + E_{\text{global}}(y)
$$

**Local energy** measures per-label compatibility between features and labels:

$$
E_{\text{local}}(x, y) = \sum_{i=1}^{L} y_i \cdot b_i^\top h(x)
$$

where $h(x)$ is a learned feature representation (two-layer MLP with dropout $p = 0.2$) and $b_i$ are per-label scoring vectors parameterized by a linear layer $B$.

**Global energy** captures label–label dependencies via a bilinear interaction:

$$
E_{\text{global}}(y) = v^\top \text{softplus}(M y)
$$

where $M \in \mathbb{R}^{d_e \times L}$ is a label mixing matrix (no bias) and $v \in \mathbb{R}^{d_e}$ is a global scoring vector. The softplus nonlinearity allows smooth, non-negative interaction modeling. This term enables the energy function to encode label co-occurrence patterns independently of input features.

---

## 3. Corrective Energy Alignment

### 3.1 The Energy Inversion Problem

In a well-calibrated energy model, the energy of ground truth labels should be lower than the energy of incorrect predictions:

$$
E_\Theta(x, y_{\text{true}}) < E_\Theta(x, \hat{y}) \quad \text{whenever} \quad \ell(\hat{y}, y_{\text{true}}) > 0
$$

In practice, co-training the energy and task networks can lead to **inversions** where $E_\Theta(x, \hat{y}) < E_\Theta(x, y_{\text{true}})$ despite nonzero task error. These inversions represent a fundamental misalignment between the energy landscape and the task objective.

### 3.2 Inline Critical Detection

For each training batch, the method identifies **critical examples** on-the-fly — examples where the energy surface is misaligned with the task network's errors. An example $(x, y)$ is critical when:

1. The task network is making an error: $\ell(\hat{y}, y) > 0$
2. The energy gap is insufficient: $\Delta(x, y) < \max(\alpha \cdot \ell(\hat{y}, y),\; m)$

where:
- $\Delta(x, y) = E_\Theta(x, \hat{y}) - E_\Theta(x, y)$ is the energy gap
- $\ell(\hat{y}, y)$ is the task error (see Section 3.4)
- $\alpha$ is a margin scaling factor controlling the required gap relative to task error
- $m$ is a minimum margin floor enforcing meaningful energy separation even for small errors

Critical detection runs inline within each batch without requiring any full-dataset sweep, making it computationally efficient and always up-to-date with the current model state.

### 3.3 Corrective Loss

The corrective loss applies a hinge penalty to enforce proper energy ordering on the critical examples found in each batch:

$$
\mathcal{L}_{\text{correct}}(\Theta) = \frac{1}{|\mathcal{B}|} \sum_{(x, y) \in \mathcal{B}_{\text{crit}}} \bigl[\max(\alpha \cdot \ell(\hat{y}, y),\; m) + E_\Theta(x, y) - E_\Theta(x, \hat{y})\bigr]_+
$$

where $\mathcal{B}_{\text{crit}} \subseteq \mathcal{B}$ is the subset of critical examples in the current batch, and $[\cdot]_+$ denotes the ReLU hinge.

This loss drives $E_\Theta(x, y_{\text{true}})$ down and $E_\Theta(x, \hat{y})$ up, restoring proper energy ranking. The margin term $\max(\alpha \cdot \ell, m)$ ensures the correction is proportional to the task error, with a floor $m$ for stability.

An optional **global-only correction** mode restricts gradient updates to $E_{\text{global}}(y)$, leaving the local energy term fixed. This can be useful when label correlations are the primary source of misalignment.

### 3.4 Task Error Metrics

Three task error metrics $\ell(\hat{y}, y)$ are supported:

**Hamming error** (label-wise accuracy):
$$
\ell_{\text{hamming}}(\hat{y}, y) = \frac{1}{L} \sum_{i=1}^{L} \mathbb{1}[\hat{y}_i \neq y_i]
$$

**Differentiable soft F1 error** (default):
$$
\ell_{F_1}(\hat{y}, y) = 1 - \frac{2 \sum_i \hat{y}_i \cdot y_i}{\sum_i \hat{y}_i + \sum_i y_i + \epsilon}
$$

This uses the soft predictions directly (without thresholding), enabling gradient flow through the task error.

**Structural error** (label co-occurrence mismatch):
$$
\ell_{\text{struct}}(\hat{y}, y) = \frac{1}{L^2} \sum_{i,j} \bigl|\hat{y}_i \hat{y}_j - y_i y_j\bigr|
$$

This penalizes discrepancies in pairwise label co-occurrence patterns, capturing higher-order label structure.

---

## 4. Training Procedure

### 4.1 Bilevel Optimization

Training alternates between two coupled updates per batch:

**Step 1 — Energy network update (parameters $\Theta$):**
$$
\mathcal{L}_\Theta = \beta \cdot \mathcal{L}_{\text{correct}}(\Theta)
$$

The corrective loss is computed inline on the current batch. Critical examples are identified, and the hinge loss is applied to the critical subset. If no examples in the batch are critical, the energy network receives no gradient.

**Step 2 — Task network update (parameters $\Phi$):**
$$
\mathcal{L}_\Phi = \lambda_1 \cdot \mathbb{E}_{x}\bigl[E_\Theta(x, F_\Phi(x))\bigr] + \lambda_2 \cdot \text{BCE}_w\bigl(F_\Phi(x),\; y\bigr)
$$

where $\text{BCE}_w$ is a weighted binary cross-entropy with per-label positive class weights:

$$
w_l = \min\!\Bigl(\frac{1 - f_l}{f_l + \epsilon},\; 50\Bigr), \quad f_l = \text{frequency of label } l
$$

The task network loss combines two signals: (1) minimizing the energy of its predictions under the current energy model (energy-guided refinement), and (2) supervised cross-entropy against ground truth labels.

Gradients are detached appropriately: the corrective loss updates only $\Theta$ (task network predictions are treated as fixed targets), and the energy term in $\mathcal{L}_\Phi$ does not backpropagate through $\Theta$.

### 4.2 Key Design Choices

| Component | Choice | Rationale |
| --- | --- | --- |
| Gradient isolation | Separate AdamW optimizers for $\Theta$ and $\Phi$ | Prevents entangled updates |
| $\hat{y}$ detached in Step 1 | `torch.no_grad()` on task net | Energy net learns from current predictions without backprop through them |
| Adaptive margin | $\max(\alpha \cdot \ell, m)$ | Larger errors demand larger energy gaps; $m$ prevents collapse |
| LR scheduling | ReduceLROnPlateau on val F1 | Patience=10, factor=0.5 for both networks |
| Checkpointing | Restore best val F1 model | Avoids overfitting in later epochs |

### 4.3 Optimization Details

- **Optimizer:** AdamW (weight decay $10^{-4}$) for both networks
- **Learning rate schedule:** ReduceLROnPlateau (patience 10, factor 0.5, min LR $10^{-6}$), stepped on validation instance-F1
- **Gradient clipping:** Max norm 1.0 for both networks
- **Checkpointing:** Best model restored from checkpoint selected by validation instance-F1
- **Evaluation metric:** Instance-averaged F1 at threshold 0.5

---

## 5. Comparison with SEAL Paper's Margin Loss

The original SEAL (NeurIPS 2022) learns an energy model directly on the entire dataset using the classic SSVM loss:

$$
\mathcal{L}_{E}^{\text{margin}} = \sum_{\tilde{y}} \bigl[\Delta(\tilde{y}, y) - E_\Theta(x, \tilde{y}) + E_\Theta(x, y)\bigr]_+
$$

The corrective approach differs in several fundamental ways:

| Aspect | SEAL Paper (Eq. 6) | Corrective Loss |
| --- | --- | --- |
| **Margin term** | $\Delta(\tilde{y}, y)$ — fixed cost function (e.g., Hamming distance) | $\alpha \cdot \ell(F(x), y)$ — actual task error of the current prediction |
| **Negative sample $\tilde{y}$** | Found by loss-augmented GBI or arbitrary samples | Always $F(x)$ — the task net's current prediction |
| **When loss applies** | All training examples unconditionally | Only critical examples where task net is wrong AND energy is misaligned |
| **Adaptive margin** | No — $\Delta$ is static | Yes — scales with prediction error, with $m$ floor |
| **Scope** | Global energy surface shaping | Targeted correction of misaligned regions only |

**Core conceptual shift:** The SEAL paper asks "is the energy gap at least as large as the label-space distance?" and tries to learn a globally correct energy surface. The corrective loss asks "for the predictions the task net is currently getting wrong, does the energy net correctly rank ground truth below those predictions?" — it targets and fixes only misaligned regions.

### Eliminating Explicit Negative Sampling

The original SEAL requires explicit negative label configurations sampled from a proposal distribution over the combinatorial label space. The corrective approach uses the task net's own prediction as the negative, which provides three benefits:

1. **No combinatorial sampling** over the label space — the model's own prediction serves as the negative.
2. **Automatically hard negatives** — as the task net improves, its predictions become increasingly informative hard negatives.
3. **Simpler objective** — the only contrast is true vs. current prediction, directly targeting actual mistakes.

### Main Advantages over Original SEAL

- **Targeted updates:** Only corrects on critical examples rather than pushing the energy surface everywhere, focusing on truly problematic regions.
- **Explicit separation of prediction and structure:** The task net handles prediction via BCE; the energy net handles structural consistency. Each can be debugged and ablated independently.
- **Avoids over-fitting the energy:** No corrective gradient when the task is correct, even if the energy surface has minor imperfections.
- **Error-scaled margins:** The required energy gap is proportional to the task error, making correction proportionally aggressive where the task is most wrong.
- **Inline per-batch detection:** Always up-to-date with the current model state, no stale critical sets.

---

## 6. Concrete Examples

Setup: 4 labels, ground truth $y^* = [1, 1, 0, 0]$, task net predicts $F(x) = [0.8, 0.3, 0.1, 0.2]$. The task error is $\ell = 1 - \text{soft } F_1 = 0.353$.

### Case A: Energy surface correct, sufficient margin (not critical)

$E(x, y^*) = -2.0$, $E(x, F(x)) = -1.2$. The gap $\Delta = 0.8$ exceeds the required margin of $0.353$. The energy gradient naturally helps — no correction needed.

### Case B: Energy surface inverted (critical)

$E(x, y^*) = -1.0$, $E(x, F(x)) = -1.5$. The energy net thinks the wrong prediction is better (lower energy). The energy gradient actively reinforces the mistake. Corrective hinge loss: $[0.353 + (-1.0) - (-1.5)]_+ = 0.853$, forcing the energy net to fix this inversion.

### Case C: Energy surface correct but margin too small (critical)

$E(x, y^*) = -1.5$, $E(x, F(x)) = -1.4$. The ranking is technically correct, but the gap of $0.1$ is too weak relative to the error of $0.353$. The corrective loss widens the gap to provide a useful gradient signal.

### Case D: Task net correct (never critical)

If $F(x) \approx y^*$ then $\ell \approx 0$ and the example is excluded regardless of the energy surface.

| Case | Task wrong? | Energy ranking | Critical? | Effect |
| --- | --- | --- | --- | --- |
| A | Yes | Correct, large gap | No | Energy gradient naturally helps |
| B | Yes | Inverted | **Yes** | Energy gradient hurts — corrective loss fixes it |
| C | Yes | Correct, small gap | **Yes** | Energy gradient too weak — corrective loss widens gap |
| D | No | Irrelevant | No | Task net already correct, skip |

---

## 7. Role of the Hinge vs. the Error Metric

Two distinct components serve different roles in the corrective loss:

$$
\mathcal{L}_{\text{correct}} = \bigl[\underbrace{\max(\alpha \cdot \ell,\; m)}_{\text{margin (error metric)}} + E(x, y^*) - E(x, F(x))\bigr]\underbrace{_+}_{\text{hinge}}
$$

- **Error metric (Hamming, soft F1, or structural):** Answers "how wrong is the prediction?" — determines the required margin size.
- **Hinge $[\cdot]_+$:** Answers "is the energy gap sufficient?" — if yes, loss is zero; if not, loss equals the violation amount.

With **soft F1** (the default), the $\ell > 0$ condition is nearly always satisfied (sigmoid never outputs exactly 0 or 1), so the hinge alone performs the real filtering. With **Hamming error**, $\ell$ can be exactly zero for correct hard predictions, making the $\ell > 0$ condition meaningful.

The corrective loss targets the **mismatch** between task error magnitude and energy gap size. A moderately wrong prediction with a tiny energy gap is critical; a very wrong prediction with a large energy gap is not. It is not about absolute prediction badness but about whether the energy net is doing its job for that example.

---

## 8. Intuition

The system implements an **identify-and-correct loop** within each batch:

1. The energy net should assign lower energy to ground truth than to wrong predictions.
2. When this property is violated (the energy surface is inverted or has insufficient margin), the corrective loss fixes those regions.
3. Once corrected, the energy surface provides a useful gradient signal to the task net, pushing predictions toward lower-energy (more compatible) label configurations.
4. As the task net improves, different examples become critical, and the energy net adapts.

This creates a **virtuous cycle** where the energy net learns label-space structure and the task net exploits that structure.

---

## 9. Hyperparameters

| Symbol | Parameter | Default | Description |
|--------|-----------|---------|-------------|
| $\alpha$ | `alpha` | 1.0 | Margin scaling relative to task error |
| $m$ | `min_margin` | 0.1 | Minimum energy gap floor |
| $\beta$ | `beta` | 0.1 | Weight on corrective loss for energy net |
| $\lambda_1$ | `lambda1` | 0.01 | Energy term weight in task net loss |
| $\lambda_2$ | `lambda2` | 1.0 | BCE term weight in task net loss |
| — | `lr_energy` | 0.001 | Energy network learning rate |
| — | `lr_task` | 0.001 | Task network learning rate |
| — | `hidden_dim` | 512 | Shared hidden dimension |
| — | `energy_hidden` | 512 | Global energy interaction dimension |
| — | `batch_size` | 32 | Training batch size |
| — | `epochs` | 300 | Maximum training epochs |
| — | `task_error_metric` | f1 | Task error metric (f1, hamming, structural) |

All runs use the default configuration: `lr_energy = lr_task = 0.001`, `beta = 0.1`, `alpha = 1.0`, `min_margin = 0.1`, `lambda1 = 0.01`, `lambda2 = 1.0`, `hidden_dim = energy_hidden = 512`, `batch_size = 32`, `epochs = 300`, inline mode, AdamW (weight decay 1e-4), ReduceLROnPlateau (patience 10, factor 0.5).

---

## 10. Evaluation

### Instance-Level F1

The primary evaluation metric is **instance-averaged F1**:

$$
F_1^{\text{instance}} = \frac{1}{N} \sum_{n=1}^{N} \frac{2 \sum_i \hat{y}_i^{(n)} y_i^{(n)}}{\sum_i \hat{y}_i^{(n)} + \sum_i y_i^{(n)} + \epsilon}
$$

where predictions are binarized at threshold 0.5. The edge case where both prediction and ground truth are all-zero is handled by defining $F_1 = 1.0$.

---

## 11. Datasets

| Dataset | Features | Labels | Train | Val | Test | Avg Labels/Sample | Format |
|---------|----------|--------|-------|-----|------|-------------------|--------|
| Bibtex | 1,836 | 159 | 4,407 | 1,491 | 1,497 | 2.4 | Sparse ARFF |
| Delicious | 500 | 983 | 9,698 | 3,210 | 3,197 | 19.0 | Sparse ARFF |
| Genbase | 1,185 | 27 | — | — | — | — | Dense ARFF |
| CAL500 | 68 | 174 | — | — | — | — | Dense ARFF |
| Eurlex-ev | 5,000 | 3,993 | — | — | — | — | Sparse ARFF |
| Expr_fun | 561 | 500 | 1,639 | 849 | 1,291 | 9.7 | Dense ARFF |
| SPO_fun | 86 | 500 | — | — | — | — | Dense ARFF |

Bibtex, Delicious, Genbase, CAL500, and Eurlex-ev use stratified 10-fold cross-validation splits (folds 1–6 for training, 7–8 for validation, 9–10 for test). Expr_fun and SPO_fun use pre-defined train/dev/test splits.

---

## 12. Theoretical Discussion

The corrective energy alignment framework addresses a gap in energy-based structured prediction: while standard energy-based training (e.g., contrastive divergence, noise-contrastive estimation) focuses on the energy surface globally, it can fail to enforce the critical local property that **energy must respect task-level ranking** — i.e., better predictions should have lower energy.

The inline corrective mechanism provides a targeted approach: rather than uniformly shaping the energy surface, it identifies within each batch the specific examples where energy–task alignment fails and applies focused corrections. The margin-scaled hinge avoids over-correcting minor discrepancies while ensuring proportionally aggressive correction where the task is most wrong.

The bilevel structure — where the energy net learns to properly rank while the task net learns to exploit energy guidance — creates a virtuous cycle: as the task net improves, fewer examples are critical; as the energy surface improves, the energy term in the task loss provides better gradient signal.
