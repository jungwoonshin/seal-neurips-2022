# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SEAL (Structured Energy Network As a Loss) — official implementation of the NeurIPS 2022 paper. Built on top of **AllenNLP 2.5** as an `--include-package` plugin. All models, modules, dataset readers, and trainers are registered via AllenNLP's registry system and configured through JSON config files.

## Environment Setup

```bash
# Create virtualenv and install dependencies
bash setup_env.sh          # creates .venv_seal, installs requirements, runs wandb init
source .venv_seal/bin/activate

# Download datasets
bash download_datasets.sh  # downloads into ./data/

# Environment variables
export CUDA_DEVICE=0       # 0 for GPU, -1 for CPU
export DATA_DIR="./data/"
export TEST=1              # 1 for dry-run (no wandb upload), 0 for real runs
```

**Python 3.8**, PyTorch <1.9.0 (>=1.7.0), AllenNLP 2.5.0.

## Training Commands

Training uses `allennlp train` with `--include-package seal`:

```bash
# Train a model using a config from best_models_configs/
allennlp train best_models_configs/bibtex_strat_cross-entropy_ezllp30k/config.json \
  -s run_output_dir --include-package seal \
  --overrides '{"trainer.cuda_device":-1}'

# Run tests via nox
nox --session=tests -- -x -v
```

## Architecture

### Core Abstractions (all AllenNLP Registrable)

The framework implements a **score-based structured prediction** paradigm with these key components:

- **`ScoreBasedLearningModel`** (`seal/models/base.py`) — Central model class. Orchestrates two alternating training phases via `ModelMode`:
  - `UPDATE_TASK_NN` — runs the inference module to produce predictions and update the task network
  - `UPDATE_SCORE_NN` — generates samples, then computes the score-based loss to update the score network

- **`TaskNN`** (`seal/modules/task_nn.py`) — Feature network that maps input x to output logits. Task-specific subclasses (e.g., `MultilabelClassificationTaskNN`).

- **`ScoreNN`** (`seal/modules/score_nn.py`) — Computes energy/score for (x, y) pairs. Combines a `TaskNN` (local score) with an optional `StructuredScore` (global score capturing label dependencies).

- **`Sampler`** (`seal/modules/sampler/sampler.py`) — Generates output samples during training. Has two context modes: `"sample"` (for generating training samples) and `"inference"` (for test-time prediction). Variants include:
  - `BasicSampler` — wraps a TaskNN
  - `InferenceNetSampler` — learned inference network
  - `GradientBasedInferenceSampler` — gradient-based optimization in output space
  - `GroundTruthSampler` — returns ground truth labels
  - `AppendingSamplerContainer` — combines multiple samplers

- **`Loss`** (`seal/modules/loss/loss.py`) — Loss functions for training ScoreNN. Key implementations:
  - `DVNLoss` / `DVNScoreLoss` — Deep Value Network losses
  - `NCELoss` / `NCERankingLoss` — Noise Contrastive Estimation
  - `MultiLabelBCELoss` — standard cross-entropy baseline
  - `CombinationLoss` — weighted sum of multiple losses

- **`OracleValueFunction`** (`seal/modules/oracle_value_function/`) — Computes ground-truth quality of predictions (e.g., per-instance F1, Hamming loss).

- **`GradientDescentMinimaxTrainer`** (`seal/training/trainer/gradient_descent_minimax_trainer.py`) — Custom trainer that alternates optimization between TaskNN and ScoreNN parameters using a `MiniMaxOptimizer` with separate optimizers for each parameter group.

### Task-Specific Modules

Modules are organized by task under subdirectories:
- `seal/modules/*/multilabel_classification/` — Multi-label classification (primary task)
- `seal/modules/*/sequence_tagging/` — Sequence tagging
- `seal/modules/*/weizmann_horse_seg/` — Image segmentation

### Registration Pattern

All components use `@ClassName.register("type-name")` decorators. The JSON configs reference these type names. When adding new components, register them and import them in the relevant `__init__.py` so AllenNLP discovers them via `--include-package seal`.

### Config Structure

Configs in `best_models_configs/` define the full experiment: dataset reader, model (with nested sampler, loss, score_nn, task_nn, oracle_value_function), trainer (with optimizer, scheduler, num_epochs), and data paths. The `multiple_seeds_config.yaml` files run the same config across multiple random seeds.

### Logging

Custom `LoggingMixin` (`seal/modules/logging.py`) provides hierarchical metric logging through `logging_children` and `logging_buffer` — used by Loss, Sampler, and Model classes.
