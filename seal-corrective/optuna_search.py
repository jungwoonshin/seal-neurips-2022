"""
Optuna grid search over lr and architecture for all datasets.

Search space: lr_energy x lr_task x (hidden_dim, energy_hidden)
Saves intermediate + final results per dataset to seal-corrective/result/.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import optuna
import torch

from models import TaskNet, EnergyNet
from losses import EnergyCorrector
from trainers import SEALCorrectiveTrainer
from data_utils import (
    load_bibtex, load_delicious, load_genbase, load_expr_fun,
    load_eurlex_ev, load_cal500, load_spo_fun, compute_instance_f1,
)

BASE_CONFIG = {
    "lr_energy": 0.001,
    "lr_task": 0.001,
    "epochs": 75,
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

DATASETS = {
    "bibtex": {"loader": load_bibtex, "data_dir": "bibtex_stratified10folds_meka", "epochs": 75},
    "cal500": {"loader": load_cal500, "data_dir": "cal500-stratified10folds-meka", "epochs": 75},
    "delicious": {"loader": load_delicious, "data_dir": "delicious-stratified10folds-meka", "epochs": 75},
    "eurlex_ev": {"loader": load_eurlex_ev, "data_dir": "eurlex-ev-stratified10folds-meka", "epochs": 200},
    "expr_fun": {"loader": load_expr_fun, "data_dir": "expr_fun", "epochs": 75},
    "genbase": {"loader": load_genbase, "data_dir": "genbase-stratified10folds-meka", "epochs": 75},
    "spo_fun": {"loader": load_spo_fun, "data_dir": "spo_fun", "epochs": 75},
}

LR_ENERGYS = [1e-3, 5e-3, 1e-2]
LR_TASKS = [1e-3, 5e-3, 1e-2]
ARCH_CONFIGS = [(512, 512), (768, 768)]


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


def run_trial(hidden_dim, energy_hidden, lr_energy, lr_task, epochs,
              train_loader, val_loader, test_loader,
              input_dim, num_labels, pos_weight, device):
    config = BASE_CONFIG.copy()
    config["hidden_dim"] = hidden_dim
    config["energy_hidden"] = energy_hidden
    config["lr_energy"] = lr_energy
    config["lr_task"] = lr_task
    config["epochs"] = epochs

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
    best_epoch = 0

    start = time.time()
    for epoch in range(epochs):
        trainer.train_one_epoch(epoch)
        val_f1 = evaluate_silent(task_net, val_loader, device)
        test_f1 = evaluate_silent(task_net, test_loader, device)
        trainer.step_schedulers(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            trainer.save_best()

    elapsed = time.time() - start

    trainer.restore_best()
    final_test = evaluate_silent(task_net, test_loader, device)

    return {
        "hidden_dim": hidden_dim,
        "energy_hidden": energy_hidden,
        "lr_energy": lr_energy,
        "lr_task": lr_task,
        "best_val_f1": round(best_val_f1, 4),
        "best_test_f1": round(final_test, 4),
        "best_epoch": best_epoch,
        "time_s": round(elapsed, 1),
    }


def run_dataset(dataset_name, device):
    ds = DATASETS[dataset_name]
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", ds["data_dir"])
    epochs = ds["epochs"]

    print(f"\n{'='*80}")
    print(f"DATASET: {dataset_name} (epochs={epochs})")
    print(f"{'='*80}")

    # Load data (suppress prints)
    train_loader, val_loader, test_loader, input_dim, num_labels, pos_weight = ds["loader"](
        data_dir, BASE_CONFIG["batch_size"])

    # Build search space
    arch_choices = list(range(len(ARCH_CONFIGS)))
    search_space = {
        "arch": arch_choices,
        "lr_energy": LR_ENERGYS,
        "lr_task": LR_TASKS,
    }
    n_total = len(arch_choices) * len(LR_ENERGYS) * len(LR_TASKS)
    print(f"Arch configs: {ARCH_CONFIGS}")
    print(f"LR grid: lr_energy={LR_ENERGYS} x lr_task={LR_TASKS}")
    print(f"Total trials: {n_total}")

    class SilentWriter:
        def write(self, msg): pass
        def flush(self): pass

    original_stdout = sys.stdout
    results = []

    result_dir = os.path.join(os.path.dirname(__file__), "result")
    os.makedirs(result_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    intermediate_path = os.path.join(result_dir, f"grid_{dataset_name}_{timestamp}.txt")

    def save_intermediate():
        sorted_results = sorted(results, key=lambda r: r["best_val_f1"], reverse=True)
        with open(intermediate_path, "w") as f:
            f.write(f"Dataset: {dataset_name} ({len(results)}/{n_total} trials, epochs={epochs})\n\n")
            f.write(f"{'hidden':>8} {'energy':>8} {'lr_e':>10} {'lr_t':>10} {'val_f1':>10} {'test_f1':>10} {'epoch':>7} {'time':>8}\n")
            f.write("-" * 80 + "\n")
            for r in sorted_results:
                f.write(f"{r['hidden_dim']:>8} {r['energy_hidden']:>8} "
                        f"{r['lr_energy']:>10.5f} {r['lr_task']:>10.5f} "
                        f"{r['best_val_f1']:>10.4f} {r['best_test_f1']:>10.4f} "
                        f"{r['best_epoch']:>7} {r['time_s']:>7.1f}s\n")

    sampler = optuna.samplers.GridSampler(search_space)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial):
        arch_idx = trial.suggest_categorical("arch", arch_choices)
        lr_energy = trial.suggest_categorical("lr_energy", LR_ENERGYS)
        lr_task = trial.suggest_categorical("lr_task", LR_TASKS)
        hidden_dim, energy_hidden = ARCH_CONFIGS[arch_idx]

        original_stdout.write(
            f"  [{len(results)+1}/{n_total}] dim={hidden_dim}, "
            f"lr_e={lr_energy}, lr_t={lr_task} ... ")
        original_stdout.flush()

        sys.stdout = SilentWriter()
        result = run_trial(
            hidden_dim, energy_hidden, lr_energy, lr_task, epochs,
            train_loader, val_loader, test_loader,
            input_dim, num_labels, pos_weight, device,
        )
        sys.stdout = original_stdout

        results.append(result)
        print(f"val={result['best_val_f1']:.4f}, test={result['best_test_f1']:.4f}, "
              f"ep={result['best_epoch']}, {result['time_s']}s")
        save_intermediate()

        return result["best_val_f1"]

    study.optimize(objective, n_trials=n_total)

    # Final save
    save_intermediate()

    # Best result
    best = max(results, key=lambda r: r["best_val_f1"])
    print(f"\nBest for {dataset_name}: val={best['best_val_f1']:.4f}, "
          f"test={best['best_test_f1']:.4f}, dim={best['hidden_dim']}, "
          f"lr_e={best['lr_energy']}, lr_t={best['lr_task']}")

    # Save JSON
    json_path = os.path.join(result_dir, f"grid_{dataset_name}_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump({"dataset": dataset_name, "epochs": epochs,
                   "config": BASE_CONFIG, "results": results}, f, indent=2)

    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None,
                        choices=list(DATASETS.keys()),
                        help="Run single dataset (default: all)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    datasets_to_run = [args.dataset] if args.dataset else list(DATASETS.keys())
    all_best = {}

    for ds_name in datasets_to_run:
        best = run_dataset(ds_name, device)
        all_best[ds_name] = best

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY - Best config per dataset")
    print(f"{'='*80}")
    print(f"{'dataset':>12} {'dim':>6} {'lr_e':>10} {'lr_t':>10} {'val_f1':>10} {'test_f1':>10}")
    print("-" * 65)
    for ds_name, best in all_best.items():
        print(f"{ds_name:>12} {best['hidden_dim']:>6} "
              f"{best['lr_energy']:>10.5f} {best['lr_task']:>10.5f} "
              f"{best['best_val_f1']:>10.4f} {best['best_test_f1']:>10.4f}")


if __name__ == "__main__":
    main()
