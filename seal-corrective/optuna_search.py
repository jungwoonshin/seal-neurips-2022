"""
Optuna grid search over hidden_dim and energy_hidden.

Runs all 9 combinations of hidden_dim x energy_hidden with 70 epochs each,
using inline corrective loss (no correction interval).
Logs final table to seal-corrective/result/.
"""

import json
import os
import sys
import time
from datetime import datetime

import optuna
import torch
from torch.utils.data import DataLoader

from models import TaskNet, EnergyNet
from losses import EnergyCorrector
from trainers import SEALCorrectiveTrainer
from data_utils import load_bibtex, compute_instance_f1

BASE_CONFIG = {
    "lr_energy": 0.001,
    "lr_task": 0.001,
    "epochs": 70,
    "lambda1": 0.01,
    "lambda2": 1.0,
    "beta": 0.1,
    "alpha": 1.0,
    "correction_interval": 25,
    "task_error_metric": "f1",
    "correct_global_only": False,
    "batch_size": 32,
    "min_margin": 0.1,
    "no_correction_interval": True,
}

HIDDEN_DIMS = [64, 256, 512]
ENERGY_HIDDENS = [64, 256, 512]


@torch.no_grad()
def evaluate_silent(task_net, loader, device):
    task_net.eval()
    all_pred, all_true = [], []
    for x, y in loader:
        x = x.to(device)
        all_pred.append(task_net(x).cpu())
        all_true.append(y)
    all_pred = torch.cat(all_pred, 0)
    all_true = torch.cat(all_true, 0)
    f1 = compute_instance_f1(all_pred, all_true, threshold=0.5)
    task_net.train()
    return f1


def run_trial(hidden_dim, energy_hidden, train_loader, val_loader, test_loader,
              input_dim, num_labels, pos_weight, device):
    config = BASE_CONFIG.copy()
    config["hidden_dim"] = hidden_dim
    config["energy_hidden"] = energy_hidden

    task_net = TaskNet(input_dim, hidden_dim, num_labels).to(device)
    energy_net = EnergyNet(input_dim, hidden_dim, num_labels, energy_hidden).to(device)

    corrector = EnergyCorrector(
        alpha=config["alpha"],
        task_error_metric=config["task_error_metric"],
        correct_global_only=config["correct_global_only"],
        min_margin=config["min_margin"],
    )

    trainer = SEALCorrectiveTrainer(
        config, task_net, energy_net, corrector,
        train_loader, val_loader, device,
        pos_weight=pos_weight,
    )

    best_val_f1 = 0.0
    best_test_f1 = 0.0
    best_epoch = 0

    start = time.time()
    for epoch in range(config["epochs"]):
        trainer.train_one_epoch(epoch)
        val_f1 = evaluate_silent(task_net, val_loader, device)
        test_f1 = evaluate_silent(task_net, test_loader, device)
        trainer.step_schedulers(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_test_f1 = test_f1
            best_epoch = epoch + 1
            trainer.save_best()

    elapsed = time.time() - start

    # Restore best and re-evaluate
    trainer.restore_best()
    final_val = evaluate_silent(task_net, val_loader, device)
    final_test = evaluate_silent(task_net, test_loader, device)

    return {
        "hidden_dim": hidden_dim,
        "energy_hidden": energy_hidden,
        "best_val_f1": round(best_val_f1, 4),
        "best_test_f1": round(final_test, 4),
        "best_epoch": best_epoch,
        "time_s": round(elapsed, 1),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "bibtex_stratified10folds_meka")
    train_loader, val_loader, test_loader, input_dim, num_labels, pos_weight = load_bibtex(
        data_dir, BASE_CONFIG["batch_size"])

    print(f"Device: {device}")
    print(f"Grid: hidden_dim={HIDDEN_DIMS} x energy_hidden={ENERGY_HIDDENS}")
    print(f"Epochs per trial: {BASE_CONFIG['epochs']}")
    print()

    # Suppress trainer prints during search
    class SilentWriter:
        def write(self, msg): pass
        def flush(self): pass

    original_stdout = sys.stdout
    results = []

    # Use Optuna with GridSampler for exhaustive search
    search_space = {
        "hidden_dim": HIDDEN_DIMS,
        "energy_hidden": ENERGY_HIDDENS,
    }
    sampler = optuna.samplers.GridSampler(search_space)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial):
        hidden_dim = trial.suggest_categorical("hidden_dim", HIDDEN_DIMS)
        energy_hidden = trial.suggest_categorical("energy_hidden", ENERGY_HIDDENS)

        original_stdout.write(
            f"Running: hidden_dim={hidden_dim}, energy_hidden={energy_hidden} ... ")
        original_stdout.flush()

        sys.stdout = SilentWriter()
        result = run_trial(
            hidden_dim, energy_hidden,
            train_loader, val_loader, test_loader,
            input_dim, num_labels, pos_weight, device,
        )
        sys.stdout = original_stdout

        results.append(result)
        print(f"val={result['best_val_f1']:.4f}, test={result['best_test_f1']:.4f}, "
              f"epoch={result['best_epoch']}, time={result['time_s']}s")

        return result["best_val_f1"]

    study.optimize(objective, n_trials=len(HIDDEN_DIMS) * len(ENERGY_HIDDENS))

    # Sort results for display
    results.sort(key=lambda r: r["best_val_f1"], reverse=True)

    # Print table
    print("\n" + "=" * 80)
    print("RESULTS TABLE")
    print("=" * 80)
    header = f"{'hidden_dim':>12} {'energy_hidden':>14} {'val_f1':>10} {'test_f1':>10} {'epoch':>7} {'time':>8}"
    print(header)
    print("-" * 80)
    for r in results:
        print(f"{r['hidden_dim']:>12} {r['energy_hidden']:>14} "
              f"{r['best_val_f1']:>10.4f} {r['best_test_f1']:>10.4f} "
              f"{r['best_epoch']:>7} {r['time_s']:>7.1f}s")
    print("=" * 80)

    # Save to result/
    result_dir = os.path.join(os.path.dirname(__file__), "result")
    os.makedirs(result_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON
    json_path = os.path.join(result_dir, f"grid_search_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump({"config": BASE_CONFIG, "results": results}, f, indent=2)

    # Save readable table
    table_path = os.path.join(result_dir, f"grid_search_{timestamp}.txt")
    with open(table_path, "w") as f:
        f.write("SEAL Corrective Energy Alignment - Grid Search Results\n")
        f.write(f"Date: {timestamp}\n")
        f.write(f"Epochs: {BASE_CONFIG['epochs']}, Mode: inline (no correction interval)\n")
        f.write(f"Grid: hidden_dim={HIDDEN_DIMS} x energy_hidden={ENERGY_HIDDENS}\n\n")
        f.write(header + "\n")
        f.write("-" * 80 + "\n")
        for r in results:
            f.write(f"{r['hidden_dim']:>12} {r['energy_hidden']:>14} "
                    f"{r['best_val_f1']:>10.4f} {r['best_test_f1']:>10.4f} "
                    f"{r['best_epoch']:>7} {r['time_s']:>7.1f}s\n")

    print(f"\nResults saved to: {json_path}")
    print(f"Table saved to:   {table_path}")


if __name__ == "__main__":
    main()
