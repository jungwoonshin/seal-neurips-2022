"""
Train the Mixture of Dependency Mechanisms model on bibtex for 40 epochs.

Runs allennlp train with the dtd-mixture global_score and logs epoch-level
results to output/mixture_run/epoch_results.txt.
"""

import json
import os
import subprocess
import sys
import glob


SERIALIZATION_DIR = os.path.join("output", "mixture_run")

CONFIG = {
    "dataset_reader": {
        "type": "arff",
        "num_labels": 159,
    },
    "model": {
        "type": "multi-label-classification-with-infnet",
        "inference_module": {
            "type": "multi-label-inference-net-normalized",
            "log_key": "inference_module",
            "loss_fn": {
                "type": "combination-loss",
                "constituent_losses": [
                    {
                        "log_key": "neg.nce_score",
                        "normalize_y": True,
                        "reduction": "none",
                        "type": "multi-label-score-loss",
                    },
                    {
                        "log_key": "bce",
                        "reduction": "none",
                        "type": "multi-label-bce",
                    },
                ],
                "log_key": "loss",
                "loss_weights": [5.514710814981766, 1],
                "reduction": "mean",
            },
        },
        "initializer": {
            "regexes": [
                [
                    ".*_linear_layers.*weight",
                    {"nonlinearity": "relu", "type": "kaiming_uniform"},
                ],
                [".*linear_layers.*bias", {"type": "zero"}],
            ]
        },
        "loss_fn": {
            "type": "multi-label-nce-ranking-with-discrete-sampling",
            "log_key": "nce",
            "num_samples": 20,
            "sign": "-",
        },
        "oracle_value_function": {
            "type": "per-instance-f1",
            "differentiable": False,
        },
        "sampler": {
            "type": "appending-container",
            "constituent_samplers": [],
            "log_key": "sampler",
        },
        "score_nn": {
            "type": "multi-label-classification-dtd",
            "global_score": {
                "type": "dtd-mixture",
                "num_labels": 159,
                "input_feature_dim": 400,
                "invariant_rank": 16,
                "svd_rank": 32,
                "conditional_rank": 16,
            },
            "task_nn": {
                "type": "multi-label-classification",
                "feature_network": {
                    "activations": ["softplus", "softplus"],
                    "dropout": [0.5, 0],
                    "hidden_dims": 400,
                    "input_dim": 1836,
                    "num_layers": 2,
                },
                "label_embeddings": {
                    "embedding_dim": 400,
                    "vocab_namespace": "labels",
                },
            },
        },
        "task_nn": {
            "type": "multi-label-classification",
            "feature_network": {
                "activations": ["softplus", "softplus"],
                "dropout": [0.5, 0],
                "hidden_dims": 400,
                "input_dim": 1836,
                "num_layers": 2,
            },
            "label_embeddings": {
                "embedding_dim": 400,
                "vocab_namespace": "labels",
            },
        },
    },
    "train_data_path": "./data/bibtex_stratified10folds_meka/Bibtex-fold@(1|2|3|4|5|6).arff",
    "validation_data_path": "./data/bibtex_stratified10folds_meka/Bibtex-fold@(7|8).arff",
    "test_data_path": "./data/bibtex_stratified10folds_meka/Bibtex-fold@(9|10).arff",
    "trainer": {
        "type": "gradient_descent_minimax",
        "callbacks": [],
        "checkpointer": {"keep_most_recent_by_count": 1},
        "cuda_device": 0,
        "grad_norm": {"task_nn": 10},
        "inner_mode": "score_nn",
        "learning_rate_schedulers": {
            "task_nn": {
                "type": "reduce_on_plateau",
                "factor": 0.5,
                "mode": "max",
                "patience": 5,
                "verbose": True,
            }
        },
        "num_epochs": 40,
        "num_steps": {"score_nn": 12, "task_nn": 5},
        "optimizer": {
            "optimizers": {
                "score_nn": {
                    "type": "adamw",
                    "lr": 4.512859464505083e-05,
                    "weight_decay": 1e-05,
                },
                "task_nn": {
                    "type": "adamw",
                    "lr": 0.0011682627302272157,
                    "weight_decay": 1e-05,
                },
            }
        },
        "patience": None,
        "validation_metric": "+fixed_f1",
    },
    "data_loader": {"batch_size": 32, "shuffle": True},
    "evaluate_on_test": True,
    "validation_dataset_reader": {"type": "arff", "num_labels": 159},
}


def write_config():
    """Write the config JSON to the serialization directory."""
    os.makedirs(SERIALIZATION_DIR, exist_ok=True)
    config_path = os.path.join(SERIALIZATION_DIR, "mixture_config.json")
    with open(config_path, "w") as f:
        json.dump(CONFIG, f, indent=2)
    print(f"Config written to {config_path}")
    return config_path


def run_training(config_path):
    """Launch allennlp train as a subprocess."""
    cmd = [
        sys.executable, "-m", "allennlp", "train",
        config_path,
        "-s", SERIALIZATION_DIR,
        "--include-package", "seal",
        "-f",
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    return result.returncode


def collect_epoch_results():
    """Read metrics_epoch_*.json files and write a summary text file."""
    output_path = os.path.join(SERIALIZATION_DIR, "epoch_results.txt")

    # Find all epoch metric files
    pattern = os.path.join(SERIALIZATION_DIR, "metrics_epoch_*.json")
    metric_files = sorted(glob.glob(pattern), key=lambda f: int(
        os.path.basename(f).replace("metrics_epoch_", "").replace(".json", "")
    ))

    if not metric_files:
        print("No epoch metric files found.")
        return

    # Keys of interest
    keys = [
        "epoch",
        "training_loss",
        "validation_loss",
        "validation_fixed_f1",
        "validation_micro_f1",
        "validation_macro_f1",
        "best_epoch",
        "best_validation_fixed_f1",
        "training_duration",
    ]

    with open(output_path, "w") as out:
        out.write("=" * 100 + "\n")
        out.write("Mixture of Dependency Mechanisms — Bibtex Training Results (40 epochs)\n")
        out.write("=" * 100 + "\n\n")

        # Write header
        header = f"{'Epoch':>5}  {'Train Loss':>12}  {'Val Loss':>12}  "
        header += f"{'Val F1(fix)':>12}  {'Val F1(mic)':>12}  {'Val F1(mac)':>12}  "
        header += f"{'Best Ep':>7}  {'Best F1(fix)':>12}"
        out.write(header + "\n")
        out.write("-" * len(header) + "\n")

        for mf in metric_files:
            with open(mf, "r") as f:
                m = json.load(f)

            epoch = m.get("epoch", "?")
            train_loss = m.get("training_loss", float("nan"))
            val_loss = m.get("validation_loss", float("nan"))
            val_f1_fix = m.get("validation_fixed_f1", float("nan"))
            val_f1_mic = m.get("validation_micro_f1", float("nan"))
            val_f1_mac = m.get("validation_macro_f1", float("nan"))
            best_ep = m.get("best_epoch", "?")
            best_f1 = m.get("best_validation_fixed_f1", float("nan"))

            line = f"{epoch:>5}  {train_loss:>12.6f}  {val_loss:>12.6f}  "
            line += f"{val_f1_fix:>12.6f}  {val_f1_mic:>12.6f}  {val_f1_mac:>12.6f}  "
            line += f"{best_ep:>7}  {best_f1:>12.6f}"
            out.write(line + "\n")

        # Write final summary from last epoch
        out.write("\n" + "=" * 100 + "\n")
        out.write("Final Metrics (last epoch):\n")
        out.write("-" * 40 + "\n")
        with open(metric_files[-1], "r") as f:
            last = json.load(f)
        for k in sorted(last.keys()):
            out.write(f"  {k}: {last[k]}\n")
        out.write("=" * 100 + "\n")

    print(f"\nEpoch results written to {output_path}")


def main():
    config_path = write_config()
    returncode = run_training(config_path)

    if returncode != 0:
        print(f"Training failed with exit code {returncode}")
        # Still try to collect whatever results exist
        collect_epoch_results()
        sys.exit(returncode)

    collect_epoch_results()
    print("Done.")


if __name__ == "__main__":
    main()
