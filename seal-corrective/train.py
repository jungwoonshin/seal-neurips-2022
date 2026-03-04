"""
Entry point for SEAL corrective energy alignment training.

Usage:
    python train.py --dataset bibtex --epochs 3
    python train.py --dataset synthetic --epochs 200
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader, TensorDataset

from models import TaskNet, EnergyNet
from losses import EnergyCorrector
from trainers import SEALCorrectiveTrainer
from data_utils import load_bibtex, compute_instance_f1


DEFAULT_CONFIG = {
    "lr_energy": 0.001,
    "lr_task": 0.001,
    "epochs": 300,
    "lambda1": 0.01,
    "lambda2": 1.0,
    "beta": 0.1,
    "alpha": 1.0,
    "correction_interval": 25,
    "task_error_metric": "f1",
    "correct_global_only": False,
    "batch_size": 32,
    "hidden_dim": 512,
    "energy_hidden": 512,
    "min_margin": 0.1,
}


class Logger:
    """Dual output to stdout and log file with immediate flush."""

    def __init__(self, log_path: str):
        self.log_file = open(log_path, "w")
        self.stdout = sys.stdout

    def write(self, msg: str):
        self.stdout.write(msg)
        self.stdout.flush()
        self.log_file.write(msg)
        self.log_file.flush()

    def flush(self):
        self.stdout.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


def make_synthetic_data(n_samples: int, input_dim: int, num_labels: int,
                        seed: int = 42) -> TensorDataset:
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n_samples, input_dim, generator=gen)
    W = torch.randn(input_dim, num_labels, generator=gen) * 0.3
    latent = torch.randn(n_samples, 1, generator=gen)
    correlation = torch.randn(1, num_labels, generator=gen) * 0.5
    logits = x @ W + latent * correlation
    y = (torch.sigmoid(logits) > 0.5).float()
    return TensorDataset(x, y)


@torch.no_grad()
def evaluate(task_net, loader, device, split_name="val"):
    """Evaluate instance-level F1 at 0.5 threshold."""
    task_net.eval()
    all_pred = []
    all_true = []
    for x, y in loader:
        x = x.to(device)
        y_pred = task_net(x)
        all_pred.append(y_pred.cpu())
        all_true.append(y)
    all_pred = torch.cat(all_pred, dim=0)
    all_true = torch.cat(all_true, dim=0)
    f1 = compute_instance_f1(all_pred, all_true, threshold=0.5)
    task_net.train()
    print(f"  {split_name} instance-F1 (threshold=0.5): {f1:.4f}")
    return f1


def main():
    parser = argparse.ArgumentParser(description="SEAL Corrective Energy Alignment")
    parser.add_argument("--dataset", type=str, default="bibtex",
                        choices=["bibtex", "synthetic"])
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--correction-interval", type=int, default=None)
    parser.add_argument("--metric", type=str, default=None,
                        choices=["hamming", "f1", "structural"])
    parser.add_argument("--global-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr-energy", type=float, default=None)
    parser.add_argument("--lr-task", type=float, default=None)
    parser.add_argument("--no-correction-interval", action="store_true",
                        help="Compute corrective loss inline per batch (no periodic diagnosis)")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Directory for log files")
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.beta is not None:
        config["beta"] = args.beta
    if args.alpha is not None:
        config["alpha"] = args.alpha
    if args.correction_interval is not None:
        config["correction_interval"] = args.correction_interval
    if args.metric is not None:
        config["task_error_metric"] = args.metric
    if args.global_only:
        config["correct_global_only"] = True
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.lr_energy is not None:
        config["lr_energy"] = args.lr_energy
    if args.lr_task is not None:
        config["lr_task"] = args.lr_task
    if args.no_correction_interval:
        config["no_correction_interval"] = True
    # ── Logging setup ──
    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, f"train_{args.dataset}_{timestamp}.log")
    results_path = os.path.join(args.log_dir, f"results_{args.dataset}_{timestamp}.jsonl")
    logger = Logger(log_path)
    sys.stdout = logger

    # Results file (JSONL) for structured logging
    results_file = open(results_path, "w")

    def log_result(entry: dict):
        results_file.write(json.dumps(entry) + "\n")
        results_file.flush()

    print(f"Logging to: {log_path}")
    print(f"Results to: {results_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Data ──
    test_loader = None
    pos_weight = None
    if args.dataset == "bibtex":
        data_dir = args.data_dir or os.path.join(
            os.path.dirname(__file__), "..", "data", "bibtex_stratified10folds_meka")
        train_loader, val_loader, test_loader, input_dim, num_labels, pos_weight = load_bibtex(
            data_dir, config["batch_size"])
    else:
        input_dim, num_labels = 50, 10
        train_ds = make_synthetic_data(500, input_dim, num_labels, seed=42)
        val_ds = make_synthetic_data(100, input_dim, num_labels, seed=123)
        train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)

    print(f"Device: {device}")
    print(f"Config: {config}")
    print(f"Input dim: {input_dim}, Num labels: {num_labels}")

    log_result({"type": "config", "config": config, "dataset": args.dataset,
                "input_dim": input_dim, "num_labels": num_labels})

    # ── Models ──
    task_net = TaskNet(input_dim, config["hidden_dim"], num_labels).to(device)
    energy_net = EnergyNet(
        input_dim, config["hidden_dim"], num_labels, config["energy_hidden"],
    ).to(device)

    # ── Corrector ──
    corrector = EnergyCorrector(
        alpha=config["alpha"],
        task_error_metric=config["task_error_metric"],
        correct_global_only=config["correct_global_only"],
        min_margin=config["min_margin"],
    )

    # ── Train ──
    trainer = SEALCorrectiveTrainer(
        config, task_net, energy_net, corrector,
        train_loader, val_loader, device,
        pos_weight=pos_weight,
    )

    # Evaluate before training
    print("\n── Before training ──")
    val_f1 = evaluate(task_net, val_loader, device, "val")
    test_f1 = evaluate(task_net, test_loader, device, "test") if test_loader else None
    log_result({"type": "eval", "epoch": 0, "val_f1": val_f1, "test_f1": test_f1})

    best_val_f1 = val_f1
    best_test_f1 = test_f1
    best_epoch = 0
    start_time = time.time()

    # Train with per-epoch evaluation
    for epoch in range(config["epochs"]):
        epoch_start = time.time()
        print(f"\n── Epoch {epoch + 1}/{config['epochs']} ──")
        loss_theta, loss_phi = trainer.train_one_epoch(epoch)
        epoch_time = time.time() - epoch_start

        print(f"  End of epoch {epoch + 1}:")
        val_f1 = evaluate(task_net, val_loader, device, "val")
        test_f1 = evaluate(task_net, test_loader, device, "test") if test_loader else None

        trainer.step_schedulers(val_f1)

        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            best_test_f1 = test_f1
            best_epoch = epoch + 1
            trainer.save_best()

        log_result({
            "type": "eval",
            "epoch": epoch + 1,
            "val_f1": val_f1,
            "test_f1": test_f1,
            "loss_theta": loss_theta,
            "loss_phi": loss_phi,
            "critical_set_size": len(corrector.critical_set),
            "epoch_time_s": round(epoch_time, 2),
            "best": is_best,
        })

        if is_best:
            print(f"  ** New best val F1: {val_f1:.4f} **")

    # Restore best model and do final evaluation
    trainer.restore_best()
    print("\n── Final evaluation (best model from epoch {}) ──".format(best_epoch))
    final_val_f1 = evaluate(task_net, val_loader, device, "val")
    final_test_f1 = evaluate(task_net, test_loader, device, "test") if test_loader else None

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.1f}s.")
    print(f"Best val F1: {best_val_f1:.4f} at epoch {best_epoch}")
    print(f"Final test F1 (best model): {final_test_f1:.4f}" if final_test_f1 else "")

    log_result({"type": "summary", "best_val_f1": best_val_f1,
                "best_test_f1": final_test_f1,
                "best_epoch": best_epoch, "total_time_s": round(total_time, 2)})

    results_file.close()
    sys.stdout = logger.stdout
    logger.close()

    print(f"Logs saved to: {log_path}")
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
