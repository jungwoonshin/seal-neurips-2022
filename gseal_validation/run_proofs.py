"""
G-SEAL Validation: Proofs 1-3 on Synthetic Constrained Sequences.

Proof 1: Structure network can distinguish valid from invalid structures (>95% acc)
Proof 2: Structure network gradient provides useful signal to task-net
         (G-SEAL violation rate < baseline, gradient ratio > 0.01)
Proof 3: Knowledge transfers — violations stay low after structure net removal

Usage:
    python gseal_validation/run_proofs.py --proof 1
    python gseal_validation/run_proofs.py --proof 2
    python gseal_validation/run_proofs.py --proof 3
    python gseal_validation/run_proofs.py --proof all
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from synthetic_data import (
    NUM_TAGS,
    FORCED_NEXT,
    generate_dataset,
    corrupt_sequence,
    labels_to_onehot,
    count_violations,
    violation_rate,
)
from models import (
    TaskNet,
    StructureNetwork,
    OracleStructureLoss,
)


LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


class TeeStream:
    """Write to both stdout and a log file."""

    def __init__(self, log_path):
        self.terminal = sys.stdout
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log_file = open(log_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


def setup_logging(proof_id: str) -> Path:
    """Set up logging to both console and file under seal-v1/logs/.

    Returns the log file path.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"proof{proof_id}_{timestamp}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    tee = TeeStream(log_path)
    sys.stdout = tee
    return log_path


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _evaluate(task_net, data, device):
    """Evaluate task-net on accuracy and violation rate."""
    task_net.eval()
    all_preds = []
    correct = 0
    total = 0

    with torch.no_grad():
        for start in range(0, len(data["labels"]), 256):
            x = data["inputs"][start : start + 256].to(device)
            labels = data["labels"][start : start + 256].to(device)
            preds = task_net.predict(x)
            correct += (preds == labels).sum().item()
            total += labels.numel()
            all_preds.append(preds.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    viol_rate = violation_rate(all_preds)
    violations, _ = count_violations(all_preds)

    return {
        "accuracy": correct / total,
        "violation_rate": viol_rate,
        "total_violations": violations,
    }


# ============================================================================
# Proof 1: Can the structure network learn structural constraints?
# ============================================================================


def run_proof_1(args):
    """Train structure network to classify valid vs invalid sequences."""
    print("=" * 70)
    print("PROOF 1: Can the structure network distinguish valid from invalid?")
    print("=" * 70)
    set_seed(args.seed)

    device = torch.device(args.device)

    train_data = generate_dataset(5000, seq_len=args.seq_len, seed=42)
    val_data = generate_dataset(1000, seq_len=args.seq_len, seed=123)

    struct_net = StructureNetwork(
        num_tags=NUM_TAGS,
        d_model=args.struct_d_model,
        num_heads=args.struct_num_heads,
        num_layers=args.struct_num_layers,
        dim_feedforward=args.struct_d_model * 2,
        use_embeddings=False,
    ).to(device)

    param_count = sum(p.numel() for p in struct_net.parameters())
    print(f"Structure network parameters: {param_count:,}")

    optimizer = optim.Adam(struct_net.parameters(), lr=1e-3)
    rng = np.random.RandomState(42)

    best_val_acc = 0.0
    history = []

    for epoch in range(args.proof1_epochs):
        struct_net.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        perm = torch.randperm(len(train_data["labels"]))

        for start in range(0, len(perm), args.batch_size):
            idx = perm[start : start + args.batch_size]
            valid_labels = train_data["labels"][idx].to(device)
            batch = valid_labels.shape[0]

            valid_onehot = labels_to_onehot(valid_labels).to(device)

            neg_structural = corrupt_sequence(valid_labels.cpu(), "structural", rng)
            neg_random = corrupt_sequence(valid_labels.cpu(), "random", rng)

            neg_structural_oh = labels_to_onehot(neg_structural).to(device)
            neg_random_oh = labels_to_onehot(neg_random).to(device)

            score_valid = struct_net(valid_onehot)
            score_neg_struct = struct_net(neg_structural_oh)
            score_neg_random = struct_net(neg_random_oh)

            target = torch.ones(batch, device=device)
            loss = (
                F.margin_ranking_loss(score_valid, score_neg_struct, target, margin=1.0)
                + F.margin_ranking_loss(score_valid, score_neg_random, target, margin=1.0)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch
            correct_struct = (score_valid > score_neg_struct).sum().item()
            correct_random = (score_valid > score_neg_random).sum().item()
            epoch_correct += correct_struct + correct_random
            epoch_total += 2 * batch

        train_acc = epoch_correct / epoch_total

        # Validation
        struct_net.eval()
        val_correct = 0
        val_total = 0
        val_score_gaps = []

        with torch.no_grad():
            for start in range(0, len(val_data["labels"]), args.batch_size):
                valid_labels = val_data["labels"][start : start + args.batch_size].to(device)
                batch = valid_labels.shape[0]

                valid_onehot = labels_to_onehot(valid_labels).to(device)
                neg_struct_oh = labels_to_onehot(
                    corrupt_sequence(valid_labels.cpu(), "structural", rng)
                ).to(device)
                neg_random_oh = labels_to_onehot(
                    corrupt_sequence(valid_labels.cpu(), "random", rng)
                ).to(device)

                s_valid = struct_net(valid_onehot)
                s_struct = struct_net(neg_struct_oh)
                s_random = struct_net(neg_random_oh)

                val_correct += (s_valid > s_struct).sum().item()
                val_correct += (s_valid > s_random).sum().item()
                val_total += 2 * batch
                val_score_gaps.extend(
                    (s_valid - torch.min(s_struct, s_random)).cpu().tolist()
                )

        val_acc = val_correct / val_total
        mean_gap = np.mean(val_score_gaps)
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        history.append({
            "epoch": epoch, "train_loss": epoch_loss / len(perm),
            "train_acc": train_acc, "val_acc": val_acc, "mean_score_gap": mean_gap,
        })

        print(
            f"Epoch {epoch:3d} | Loss: {epoch_loss / len(perm):.4f} | "
            f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
            f"Score Gap: {mean_gap:.3f}"
        )

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    result = "PASS" if best_val_acc > 0.95 else "FAIL"
    print(f"Proof 1 result: {result} (threshold: >0.95)")

    output = {
        "proof": 1, "result": result, "best_val_acc": best_val_acc,
        "history": history,
        "config": {
            "d_model": args.struct_d_model, "num_heads": args.struct_num_heads,
            "num_layers": args.struct_num_layers, "epochs": args.proof1_epochs,
            "param_count": param_count,
        },
    }
    save_results(output, "proof1_results.json")
    return output


# ============================================================================
# Proof 2: Structure network gradient provides useful signal to task-net
# ============================================================================


def _train_task_net_ce(task_net, train_data, val_data, args, device, oracle_loss_fn=None, label=""):
    """Train task-net with CE loss only (optionally + oracle structure loss)."""
    optimizer = optim.Adam(task_net.parameters(), lr=args.task_lr)
    history = []

    for epoch in range(args.gseal_epochs):
        task_net.train()
        perm = torch.randperm(len(train_data["labels"]))
        epoch_loss = 0.0
        num_batches = 0

        for start in range(0, len(perm), args.batch_size):
            idx = perm[start : start + args.batch_size]
            x = train_data["inputs"][idx].to(device)
            labels = train_data["labels"][idx].to(device)

            logits, _ = task_net(x)
            ce_loss = F.cross_entropy(logits.view(-1, NUM_TAGS), labels.view(-1))
            total_loss = ce_loss

            if oracle_loss_fn is not None:
                soft = F.softmax(logits, dim=-1)
                lam = args.oracle_weight
                if epoch < args.warmup_epochs:
                    lam *= epoch / args.warmup_epochs
                total_loss = ce_loss + lam * oracle_loss_fn(soft).mean()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            num_batches += 1

        val_metrics = _evaluate(task_net, val_data, device)
        history.append({
            "epoch": epoch, "train_loss": epoch_loss / num_batches,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })
        print(
            f"[{label}] Epoch {epoch:3d} | Loss: {epoch_loss / num_batches:.4f} | "
            f"ValAcc: {val_metrics['accuracy']:.4f} | ValViol: {val_metrics['violation_rate']:.4f}"
        )

    return history


def train_gseal(train_data, val_data, args, device):
    """Train task-net with G-SEAL: alternating optimization.

    Key design choices:
    - Structure network trains on ONE-HOT inputs only (valid vs corrupted/random).
      This avoids the soft-vs-onehot domain mismatch that killed contrastive learning.
    - Task-net loss includes structure score on SOFT outputs, so gradients flow
      through softmax back to task-net parameters.
    - Lambda warmup prevents random structure gradients in early epochs.
    """
    print("\n--- Training: G-SEAL ---")
    set_seed(args.seed)

    task_net = TaskNet(
        input_dim=train_data["inputs"].shape[-1],
        hidden_dim=args.task_hidden_dim,
        num_tags=NUM_TAGS,
    ).to(device)

    struct_net = StructureNetwork(
        num_tags=NUM_TAGS,
        d_model=args.struct_d_model,
        num_heads=args.struct_num_heads,
        num_layers=args.struct_num_layers,
        dim_feedforward=args.struct_d_model * 2,
        use_embeddings=args.use_embeddings,
        embedding_dim=args.task_hidden_dim,
    ).to(device)

    task_optimizer = optim.Adam(task_net.parameters(), lr=args.task_lr)
    struct_optimizer = optim.Adam(struct_net.parameters(), lr=args.struct_lr)
    rng = np.random.RandomState(args.seed)
    history = []

    for epoch in range(args.gseal_epochs):
        task_net.train()
        struct_net.train()

        # Lambda warmup
        if epoch < args.warmup_epochs:
            lambda_struct = args.lambda_struct * (epoch / args.warmup_epochs)
        else:
            lambda_struct = args.lambda_struct

        perm = torch.randperm(len(train_data["labels"]))
        epoch_metrics = defaultdict(float)
        num_batches = 0

        for start in range(0, len(perm), args.batch_size):
            idx = perm[start : start + args.batch_size]
            x = train_data["inputs"][idx].to(device)
            labels = train_data["labels"][idx].to(device)
            batch = x.shape[0]
            labels_oh = labels_to_onehot(labels).to(device)

            # ==== Phase 1: Update structure network ====
            # Use ONLY one-hot inputs to avoid soft-vs-onehot domain mismatch
            # Generate task-net hard predictions as additional negatives
            with torch.no_grad():
                task_logits, task_emb = task_net(x)
                task_preds = task_logits.argmax(dim=-1)  # hard predictions
                task_preds_oh = labels_to_onehot(task_preds).to(device)

            neg_struct = corrupt_sequence(labels.cpu(), "structural", rng)
            neg_struct_oh = labels_to_onehot(neg_struct).to(device)
            neg_random = corrupt_sequence(labels.cpu(), "random", rng)
            neg_random_oh = labels_to_onehot(neg_random).to(device)

            emb = task_emb if args.use_embeddings else None

            pos_score = struct_net(labels_oh, embeddings=emb)

            neg_scores = []
            for neg_oh in [task_preds_oh, neg_struct_oh, neg_random_oh]:
                neg_scores.append(struct_net(neg_oh, embeddings=emb))

            target = torch.ones(batch, device=device)
            struct_loss = sum(
                F.margin_ranking_loss(pos_score, ns, target, margin=1.0)
                for ns in neg_scores
            )

            struct_optimizer.zero_grad()
            struct_loss.backward()
            struct_optimizer.step()

            # Structure network diagnostics
            with torch.no_grad():
                all_neg = torch.stack(neg_scores, dim=1)
                best_neg = all_neg.max(dim=1).values
                contrastive_acc = (pos_score > best_neg).float().mean().item()
                score_gap = (pos_score - best_neg).mean().item()

            # ==== Phase 2: Update task-net ====
            # Structure network is frozen (no grad for struct params)
            for p in struct_net.parameters():
                p.requires_grad_(False)

            task_logits, task_emb = task_net(x)
            task_soft = F.softmax(task_logits, dim=-1)

            ce_loss = F.cross_entropy(task_logits.view(-1, NUM_TAGS), labels.view(-1))

            # Structure score on soft output — gradient flows through softmax to task-net
            struct_score = struct_net(
                task_soft,
                embeddings=task_emb if args.use_embeddings else None,
            )
            struct_task_loss = -struct_score.mean()

            total_loss = ce_loss + lambda_struct * struct_task_loss

            task_optimizer.zero_grad()
            total_loss.backward()

            # Gradient diagnostics (efficient: single backward, decompose norms)
            task_grad_norm = sum(
                p.grad.norm().item() ** 2
                for p in task_net.parameters() if p.grad is not None
            ) ** 0.5

            task_optimizer.step()

            # Re-enable struct_net gradients for next phase 1
            for p in struct_net.parameters():
                p.requires_grad_(True)

            # Compute separate CE and struct gradient norms every 10 batches
            ce_grad_norm = 0.0
            struct_grad_norm = 0.0
            grad_ratio = 0.0
            if num_batches % 10 == 0:
                # CE-only gradient
                task_optimizer.zero_grad()
                logits_tmp, emb_tmp = task_net(x)
                ce_tmp = F.cross_entropy(logits_tmp.view(-1, NUM_TAGS), labels.view(-1))
                ce_tmp.backward(retain_graph=True)
                ce_grad_norm = sum(
                    p.grad.norm().item() ** 2
                    for p in task_net.parameters() if p.grad is not None
                ) ** 0.5

                # Struct-only gradient
                task_optimizer.zero_grad()
                soft_tmp = F.softmax(logits_tmp, dim=-1)
                for p in struct_net.parameters():
                    p.requires_grad_(False)
                s_tmp = struct_net(soft_tmp, embeddings=emb_tmp if args.use_embeddings else None)
                (lambda_struct * -s_tmp.mean()).backward()
                for p in struct_net.parameters():
                    p.requires_grad_(True)
                struct_grad_norm = sum(
                    p.grad.norm().item() ** 2
                    for p in task_net.parameters() if p.grad is not None
                ) ** 0.5
                task_optimizer.zero_grad()

                grad_ratio = struct_grad_norm / (ce_grad_norm + 1e-10)

            epoch_metrics["ce_loss"] += ce_loss.item()
            epoch_metrics["struct_task_loss"] += struct_task_loss.item()
            epoch_metrics["struct_contrastive_loss"] += struct_loss.item()
            epoch_metrics["contrastive_acc"] += contrastive_acc
            epoch_metrics["score_gap"] += score_gap
            epoch_metrics["task_grad_norm"] += task_grad_norm
            epoch_metrics["ce_grad_norm"] += ce_grad_norm
            epoch_metrics["struct_grad_norm"] += struct_grad_norm
            epoch_metrics["grad_ratio"] += grad_ratio
            num_batches += 1

        for k in epoch_metrics:
            epoch_metrics[k] /= num_batches

        val_metrics = _evaluate(task_net, val_data, device)
        epoch_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
        epoch_metrics["epoch"] = epoch
        epoch_metrics["lambda_struct"] = lambda_struct
        history.append(dict(epoch_metrics))

        print(
            f"Epoch {epoch:3d} | CE: {epoch_metrics['ce_loss']:.4f} | "
            f"Struct: {epoch_metrics['struct_task_loss']:.3f} | "
            f"ContrAcc: {epoch_metrics['contrastive_acc']:.3f} | "
            f"Gap: {epoch_metrics['score_gap']:.2f} | "
            f"GradR: {epoch_metrics['grad_ratio']:.4f} | "
            f"Acc: {val_metrics['accuracy']:.4f} | "
            f"Viol: {val_metrics['violation_rate']:.4f} | "
            f"lam: {lambda_struct:.2f}"
        )

        # Kill conditions
        if epoch > args.warmup_epochs + 3:
            if epoch_metrics["grad_ratio"] < 0.001 and epoch_metrics["grad_ratio"] > 0:
                print("  WARNING: Gradient ratio collapsed below 0.001!")
            if epoch_metrics["contrastive_acc"] < 0.6 and epoch > args.warmup_epochs + 5:
                print("  WARNING: Structure network not learning (contrastive acc < 0.6)")

    return task_net, struct_net, history


def run_proof_2(args):
    """Compare Baseline vs Oracle vs G-SEAL."""
    print("=" * 70)
    print("PROOF 2: Structure network gradient provides useful signal")
    print("=" * 70)

    device = torch.device(args.device)

    train_data = generate_dataset(5000, seq_len=args.seq_len, seed=42)
    val_data = generate_dataset(1000, seq_len=args.seq_len, seed=123)
    test_data = generate_dataset(1000, seq_len=args.seq_len, seed=456)

    # A: CE Baseline
    print("\n--- Training: CE Baseline ---")
    set_seed(args.seed)
    baseline_net = TaskNet(
        input_dim=train_data["inputs"].shape[-1],
        hidden_dim=args.task_hidden_dim, num_tags=NUM_TAGS,
    ).to(device)
    baseline_history = _train_task_net_ce(
        baseline_net, train_data, val_data, args, device, label="Baseline"
    )
    baseline_test = _evaluate(baseline_net, test_data, device)
    print(f"\nBaseline test: Acc={baseline_test['accuracy']:.4f}, Viol={baseline_test['violation_rate']:.4f}")

    # B: CE + Oracle
    print("\n--- Training: CE + Oracle Structure Loss ---")
    set_seed(args.seed)
    oracle_net = TaskNet(
        input_dim=train_data["inputs"].shape[-1],
        hidden_dim=args.task_hidden_dim, num_tags=NUM_TAGS,
    ).to(device)
    oracle_loss = OracleStructureLoss(penalty_weight=args.oracle_weight).to(device)
    oracle_history = _train_task_net_ce(
        oracle_net, train_data, val_data, args, device,
        oracle_loss_fn=oracle_loss, label="Oracle"
    )
    oracle_test = _evaluate(oracle_net, test_data, device)
    print(f"Oracle test:   Acc={oracle_test['accuracy']:.4f}, Viol={oracle_test['violation_rate']:.4f}")

    # C: G-SEAL
    gseal_net, struct_net, gseal_history = train_gseal(train_data, val_data, args, device)
    gseal_test = _evaluate(gseal_net, test_data, device)
    print(f"G-SEAL test:   Acc={gseal_test['accuracy']:.4f}, Viol={gseal_test['violation_rate']:.4f}")

    # Results
    print("\n" + "=" * 70)
    print("PROOF 2 RESULTS:")
    print(f"  Baseline violation rate: {baseline_test['violation_rate']:.4f}")
    print(f"  Oracle violation rate:   {oracle_test['violation_rate']:.4f}")
    print(f"  G-SEAL violation rate:   {gseal_test['violation_rate']:.4f}")
    print(f"  Baseline accuracy:       {baseline_test['accuracy']:.4f}")
    print(f"  Oracle accuracy:         {oracle_test['accuracy']:.4f}")
    print(f"  G-SEAL accuracy:         {gseal_test['accuracy']:.4f}")

    gseal_beats_baseline = gseal_test["violation_rate"] < baseline_test["violation_rate"]
    print(f"\n  G-SEAL < Baseline violations: {gseal_beats_baseline}")

    # Check gradient ratio from G-SEAL history (only from batches that measured it)
    grad_ratios = [h["grad_ratio"] for h in gseal_history[-10:] if h["grad_ratio"] > 0]
    avg_final_ratio = np.mean(grad_ratios) if grad_ratios else 0.0
    grad_ok = avg_final_ratio > 0.01
    print(f"  Avg final gradient ratio: {avg_final_ratio:.4f} (>0.01: {grad_ok})")

    result = "PASS" if (gseal_beats_baseline and grad_ok) else "FAIL"
    print(f"\n  Proof 2 result: {result}")
    print("=" * 70)

    output = {
        "proof": 2, "result": result,
        "test_metrics": {
            "baseline": baseline_test, "oracle": oracle_test, "gseal": gseal_test,
        },
        "gseal_beats_baseline": gseal_beats_baseline,
        "avg_final_grad_ratio": avg_final_ratio,
        "baseline_history": baseline_history,
        "oracle_history": oracle_history,
        "gseal_history": gseal_history,
    }
    save_results(output, "proof2_results.json")
    return output


# ============================================================================
# Proof 3: Knowledge transfers after structure network removal
# ============================================================================


def run_proof_3(args):
    """Train with G-SEAL, remove structure net, check if knowledge persists."""
    print("=" * 70)
    print("PROOF 3: Knowledge transfer after structure network removal")
    print("=" * 70)

    device = torch.device(args.device)

    train_data = generate_dataset(5000, seq_len=args.seq_len, seed=42)
    val_data = generate_dataset(1000, seq_len=args.seq_len, seed=123)
    test_data = generate_dataset(1000, seq_len=args.seq_len, seed=456)

    # Phase 1: Train with G-SEAL
    print("\n=== Phase 1: Training with G-SEAL ===")
    gseal_net, struct_net, gseal_history = train_gseal(train_data, val_data, args, device)
    gseal_test = _evaluate(gseal_net, test_data, device)
    print(f"\nG-SEAL test: Acc={gseal_test['accuracy']:.4f}, Viol={gseal_test['violation_rate']:.4f}")

    # Phase 2: Evaluate standalone (structure net removed — just don't use it)
    print("\n=== Phase 2: Structure network removed ===")
    standalone_test = _evaluate(gseal_net, test_data, device)
    print(f"Standalone test: Acc={standalone_test['accuracy']:.4f}, Viol={standalone_test['violation_rate']:.4f}")

    # Phase 3: Continue with CE only
    print("\n=== Phase 3: Continue training with CE only ===")
    ce_optimizer = optim.Adam(gseal_net.parameters(), lr=args.task_lr)
    continued_history = []

    for epoch in range(args.continued_epochs):
        gseal_net.train()
        perm = torch.randperm(len(train_data["labels"]))
        epoch_loss = 0.0
        num_batches = 0

        for start in range(0, len(perm), args.batch_size):
            idx = perm[start : start + args.batch_size]
            x = train_data["inputs"][idx].to(device)
            labels = train_data["labels"][idx].to(device)

            logits, _ = gseal_net(x)
            loss = F.cross_entropy(logits.view(-1, NUM_TAGS), labels.view(-1))
            ce_optimizer.zero_grad()
            loss.backward()
            ce_optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        val_metrics = _evaluate(gseal_net, val_data, device)
        continued_history.append({
            "epoch": epoch, "train_loss": epoch_loss / num_batches,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })
        print(
            f"CE-only Epoch {epoch:3d} | Loss: {epoch_loss / num_batches:.4f} | "
            f"ValAcc: {val_metrics['accuracy']:.4f} | ValViol: {val_metrics['violation_rate']:.4f}"
        )

    final_test = _evaluate(gseal_net, test_data, device)
    print(f"\nFinal (after CE continuation): Acc={final_test['accuracy']:.4f}, Viol={final_test['violation_rate']:.4f}")

    # Baseline comparison
    print("\n=== Baseline (CE from scratch) ===")
    set_seed(args.seed)
    baseline_net = TaskNet(
        input_dim=train_data["inputs"].shape[-1],
        hidden_dim=args.task_hidden_dim, num_tags=NUM_TAGS,
    ).to(device)
    baseline_history = _train_task_net_ce(
        baseline_net, train_data, val_data, args, device, label="Baseline"
    )
    baseline_test = _evaluate(baseline_net, test_data, device)
    print(f"Baseline test: Acc={baseline_test['accuracy']:.4f}, Viol={baseline_test['violation_rate']:.4f}")

    # Results
    print("\n" + "=" * 70)
    print("PROOF 3 RESULTS:")
    print(f"  Baseline violation rate:     {baseline_test['violation_rate']:.4f}")
    print(f"  G-SEAL (during training):    {gseal_test['violation_rate']:.4f}")
    print(f"  After struct net removal:    {standalone_test['violation_rate']:.4f}")
    print(f"  After continued CE:          {final_test['violation_rate']:.4f}")

    transfer = standalone_test["violation_rate"] < baseline_test["violation_rate"]
    persistence = final_test["violation_rate"] < baseline_test["violation_rate"]

    print(f"\n  Knowledge transferred (removal < baseline): {transfer}")
    print(f"  Knowledge persists (continued < baseline):  {persistence}")

    result = "PASS" if (transfer and persistence) else "FAIL"
    print(f"\n  Proof 3 result: {result}")
    print("=" * 70)

    output = {
        "proof": 3, "result": result,
        "test_metrics": {
            "baseline": baseline_test,
            "gseal_with_struct": gseal_test,
            "standalone_after_removal": standalone_test,
            "after_continued_ce": final_test,
        },
        "transfer_success": transfer,
        "persistence": persistence,
        "gseal_history": gseal_history,
        "continued_history": continued_history,
        "baseline_history": baseline_history,
    }
    save_results(output, "proof3_results.json")
    return output


# ============================================================================
# Utilities
# ============================================================================


def save_results(output, filename):
    """Save JSON results to both gseal_validation/results/ and logs/."""
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    for out_dir in [Path(__file__).parent / "results", LOG_DIR]:
        out_dir.mkdir(parents=True, exist_ok=True)
        filepath = out_dir / filename
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2, default=convert)
        print(f"Results saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="G-SEAL Validation Proofs")
    parser.add_argument("--proof", type=str, default="all", choices=["1", "2", "3", "all"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=10)

    # Task-net config
    parser.add_argument("--task_hidden_dim", type=int, default=64)
    parser.add_argument("--task_lr", type=float, default=1e-3)

    # Structure network config
    parser.add_argument("--struct_d_model", type=int, default=64)
    parser.add_argument("--struct_num_heads", type=int, default=4)
    parser.add_argument("--struct_num_layers", type=int, default=2)
    parser.add_argument("--struct_lr", type=float, default=1e-3)

    # G-SEAL training config
    parser.add_argument("--lambda_struct", type=float, default=1.0)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--oracle_weight", type=float, default=1.0)
    parser.add_argument("--use_embeddings", action="store_true")

    # Proof-specific epochs
    parser.add_argument("--proof1_epochs", type=int, default=30)
    parser.add_argument("--gseal_epochs", type=int, default=40)
    parser.add_argument("--continued_epochs", type=int, default=20)

    args = parser.parse_args()

    # Set up logging to seal-v1/logs/
    log_path = setup_logging(args.proof)
    print(f"Logging to: {log_path}")
    print(f"Args: {vars(args)}\n")

    if args.proof in ("1", "all"):
        run_proof_1(args)
    if args.proof in ("2", "all"):
        run_proof_2(args)
    if args.proof in ("3", "all"):
        run_proof_3(args)

    # Restore stdout
    if isinstance(sys.stdout, TeeStream):
        sys.stdout.close()
        sys.stdout = sys.stdout.terminal


if __name__ == "__main__":
    main()
