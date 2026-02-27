"""
G-SEAL Validation v2: Global Counting Constraints.

Tests the setting where G-SEAL has genuine advantage over CRFs:
  - Exactly 2 A's per sequence (global count — CRF can't express this)
  - No adjacent A's (pairwise — CRF can express this)

Proof 1: Structure network distinguishes valid from invalid (count + adjacency)
Proof 2: G-SEAL reduces violations vs Baseline and CRF-style oracle
Proof 3: Knowledge transfers after structure network removal

Usage:
    python gseal_validation/run_proofs_v2.py --proof 1
    python gseal_validation/run_proofs_v2.py --proof 2
    python gseal_validation/run_proofs_v2.py --proof 3
    python gseal_validation/run_proofs_v2.py --proof all
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from synthetic_data_v2 import (
    NUM_TAGS,
    REQUIRED_A_COUNT,
    generate_dataset,
    corrupt_sequence,
    labels_to_onehot,
    count_violations,
    violation_rate,
)
from models import TaskNet, StructureNetwork

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


class Logger:
    """Write to both stdout and a log file with immediate flushing."""

    def __init__(self, log_path: Path):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.terminal = sys.__stdout__
        self.log_file = open(log_path, "w")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


def log(msg: str):
    """Print and flush immediately."""
    print(msg, flush=True)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ============================================================================
# Oracle losses (upper bounds)
# ============================================================================


class PairwiseOracleLoss(torch.nn.Module):
    """CRF-equivalent oracle: penalizes adjacent A's only (pairwise).
    CANNOT enforce global count — this is the CRF's limitation."""

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, y_soft: torch.Tensor) -> torch.Tensor:
        """y_soft: (batch, seq_len, num_tags)"""
        p_a = y_soft[:, :, 0]  # P(tag=A) at each position
        # Penalize P(A_t) * P(A_{t+1}) — adjacent A's
        adj_penalty = (p_a[:, :-1] * p_a[:, 1:]).sum(dim=1)
        return self.weight * adj_penalty


class GlobalOracleLoss(torch.nn.Module):
    """Full oracle: penalizes wrong A count AND adjacent A's.
    This is the upper bound — a perfect structure loss."""

    def __init__(self, count_weight: float = 1.0, adj_weight: float = 1.0):
        super().__init__()
        self.count_weight = count_weight
        self.adj_weight = adj_weight

    def forward(self, y_soft: torch.Tensor) -> torch.Tensor:
        """y_soft: (batch, seq_len, num_tags)"""
        p_a = y_soft[:, :, 0]

        # Count penalty: (sum(P(A)) - REQUIRED_A_COUNT)^2
        expected_count = p_a.sum(dim=1)
        count_penalty = (expected_count - REQUIRED_A_COUNT) ** 2

        # Adjacency penalty
        adj_penalty = (p_a[:, :-1] * p_a[:, 1:]).sum(dim=1)

        return self.count_weight * count_penalty + self.adj_weight * adj_penalty


# ============================================================================
# Evaluation
# ============================================================================


def evaluate(task_net, data, device) -> dict:
    """Evaluate task-net on accuracy and all violation types."""
    task_net.eval()
    all_preds = []
    correct = 0
    total = 0

    with torch.no_grad():
        for s in range(0, len(data["labels"]), 256):
            x = data["inputs"][s:s + 256].to(device)
            labels = data["labels"][s:s + 256].to(device)
            preds = task_net.predict(x)
            correct += (preds == labels).sum().item()
            total += labels.numel()
            all_preds.append(preds.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    vr = violation_rate(all_preds)

    return {"accuracy": correct / total, **vr}


# ============================================================================
# Proof 1
# ============================================================================


def run_proof_1(args):
    log("=" * 70)
    log("PROOF 1: Can the structure network distinguish valid from invalid?")
    log("  (Global counting + adjacency constraints)")
    log("=" * 70)
    set_seed(args.seed)
    device = torch.device(args.device)

    train_data = generate_dataset(5000, seq_len=args.seq_len, seed=42)
    val_data = generate_dataset(1000, seq_len=args.seq_len, seed=123)

    struct_net = StructureNetwork(
        num_tags=NUM_TAGS, d_model=args.struct_d_model,
        num_heads=args.struct_num_heads, num_layers=args.struct_num_layers,
        dim_feedforward=args.struct_d_model * 2, use_embeddings=False,
    ).to(device)
    log(f"Structure network params: {sum(p.numel() for p in struct_net.parameters()):,}")

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

        for s in range(0, len(perm), args.batch_size):
            idx = perm[s:s + args.batch_size]
            valid = train_data["labels"][idx].to(device)
            batch = valid.shape[0]
            valid_oh = labels_to_onehot(valid).to(device)

            # 3 types of negatives: count violation, adjacency violation, random
            neg_count_oh = labels_to_onehot(corrupt_sequence(valid.cpu(), "count", rng)).to(device)
            neg_adj_oh = labels_to_onehot(corrupt_sequence(valid.cpu(), "adjacency", rng)).to(device)
            neg_rand_oh = labels_to_onehot(corrupt_sequence(valid.cpu(), "random", rng)).to(device)

            s_valid = struct_net(valid_oh)
            s_count = struct_net(neg_count_oh)
            s_adj = struct_net(neg_adj_oh)
            s_rand = struct_net(neg_rand_oh)

            target = torch.ones(batch, device=device)
            loss = (
                F.margin_ranking_loss(s_valid, s_count, target, margin=1.0)
                + F.margin_ranking_loss(s_valid, s_adj, target, margin=1.0)
                + F.margin_ranking_loss(s_valid, s_rand, target, margin=1.0)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch
            # Accuracy: valid > all negatives
            worst_neg = torch.min(torch.min(s_count, s_adj), s_rand)
            epoch_correct += (s_valid > worst_neg).sum().item()
            epoch_total += batch

        # Validation
        struct_net.eval()
        val_correct = 0
        val_total = 0
        val_acc_count = 0
        val_acc_adj = 0

        with torch.no_grad():
            for s in range(0, len(val_data["labels"]), args.batch_size):
                valid = val_data["labels"][s:s + args.batch_size].to(device)
                batch = valid.shape[0]
                valid_oh = labels_to_onehot(valid).to(device)
                neg_c = labels_to_onehot(corrupt_sequence(valid.cpu(), "count", rng)).to(device)
                neg_a = labels_to_onehot(corrupt_sequence(valid.cpu(), "adjacency", rng)).to(device)

                sv = struct_net(valid_oh)
                sc = struct_net(neg_c)
                sa = struct_net(neg_a)

                val_acc_count += (sv > sc).sum().item()
                val_acc_adj += (sv > sa).sum().item()
                val_correct += ((sv > sc) & (sv > sa)).sum().item()
                val_total += batch

        val_acc = val_correct / val_total
        va_count = val_acc_count / val_total
        va_adj = val_acc_adj / val_total
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        entry = {
            "epoch": epoch, "train_loss": epoch_loss / len(perm),
            "train_acc": epoch_correct / epoch_total,
            "val_acc": val_acc, "val_acc_count": va_count, "val_acc_adj": va_adj,
        }
        history.append(entry)

        log(f"Epoch {epoch:3d} | Loss: {entry['train_loss']:.4f} | "
            f"TrainAcc: {entry['train_acc']:.4f} | "
            f"ValAcc: {val_acc:.4f} (count: {va_count:.4f}, adj: {va_adj:.4f})")

    log(f"\nBest val accuracy: {best_val_acc:.4f}")
    result = "PASS" if best_val_acc > 0.95 else "FAIL"
    log(f"Proof 1: {result} (threshold: >0.95)")

    save_results({"proof": 1, "result": result, "best_val_acc": best_val_acc,
                  "history": history}, "proof1_v2_results.json")


# ============================================================================
# Proof 2
# ============================================================================


def train_ce_baseline(train_data, val_data, args, device):
    """CE-only baseline."""
    log("\n--- CE Baseline ---")
    set_seed(args.seed)
    net = TaskNet(input_dim=train_data["inputs"].shape[-1],
                  hidden_dim=args.task_hidden_dim, num_tags=NUM_TAGS).to(device)
    opt = optim.Adam(net.parameters(), lr=args.task_lr)
    history = _train_ce(net, opt, train_data, val_data, args, device, "Baseline")
    return net, history


def train_pairwise_oracle(train_data, val_data, args, device):
    """CE + pairwise oracle (CRF equivalent). Can't enforce count."""
    log("\n--- CE + Pairwise Oracle (CRF-equivalent) ---")
    set_seed(args.seed)
    net = TaskNet(input_dim=train_data["inputs"].shape[-1],
                  hidden_dim=args.task_hidden_dim, num_tags=NUM_TAGS).to(device)
    oracle = PairwiseOracleLoss(weight=args.oracle_weight)
    opt = optim.Adam(net.parameters(), lr=args.task_lr)
    history = _train_ce(net, opt, train_data, val_data, args, device, "PairOracle", oracle)
    return net, history


def train_global_oracle(train_data, val_data, args, device):
    """CE + global oracle (count + adjacency). Upper bound."""
    log("\n--- CE + Global Oracle (upper bound) ---")
    set_seed(args.seed)
    net = TaskNet(input_dim=train_data["inputs"].shape[-1],
                  hidden_dim=args.task_hidden_dim, num_tags=NUM_TAGS).to(device)
    oracle = GlobalOracleLoss(count_weight=args.oracle_weight, adj_weight=args.oracle_weight)
    opt = optim.Adam(net.parameters(), lr=args.task_lr)
    history = _train_ce(net, opt, train_data, val_data, args, device, "GlobOracle", oracle)
    return net, history


def _train_ce(net, opt, train_data, val_data, args, device, label, oracle=None):
    history = []
    for epoch in range(args.gseal_epochs):
        net.train()
        perm = torch.randperm(len(train_data["labels"]))
        total_loss = 0.0
        nb = 0
        for s in range(0, len(perm), args.batch_size):
            idx = perm[s:s + args.batch_size]
            x = train_data["inputs"][idx].to(device)
            labels = train_data["labels"][idx].to(device)
            logits, _ = net(x)
            ce = F.cross_entropy(logits.view(-1, NUM_TAGS), labels.view(-1))
            loss = ce
            if oracle is not None:
                lam = args.oracle_weight
                if epoch < args.warmup_epochs:
                    lam *= epoch / args.warmup_epochs
                loss = ce + lam * oracle(F.softmax(logits, dim=-1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            nb += 1

        m = evaluate(net, val_data, device)
        entry = {"epoch": epoch, "train_loss": total_loss / nb, **{f"val_{k}": v for k, v in m.items()}}
        history.append(entry)
        log(f"[{label}] Epoch {epoch:3d} | Loss: {total_loss/nb:.4f} | "
            f"Acc: {m['accuracy']:.4f} | CountV: {m['count_viol_rate']:.4f} | "
            f"AdjV: {m['adj_viol_rate']:.4f} | AnyV: {m['any_viol_rate']:.4f}")
    return history


def train_gseal(train_data, val_data, args, device):
    """G-SEAL: CE + learned structure score with alternating optimization."""
    log("\n--- G-SEAL ---")
    set_seed(args.seed)

    task_net = TaskNet(input_dim=train_data["inputs"].shape[-1],
                       hidden_dim=args.task_hidden_dim, num_tags=NUM_TAGS).to(device)
    struct_net = StructureNetwork(
        num_tags=NUM_TAGS, d_model=args.struct_d_model,
        num_heads=args.struct_num_heads, num_layers=args.struct_num_layers,
        dim_feedforward=args.struct_d_model * 2,
        use_embeddings=False,
    ).to(device)

    task_opt = optim.Adam(task_net.parameters(), lr=args.task_lr)
    struct_opt = optim.Adam(struct_net.parameters(), lr=args.struct_lr)
    rng = np.random.RandomState(args.seed)
    history = []

    for epoch in range(args.gseal_epochs):
        task_net.train()
        struct_net.train()

        lam = args.lambda_struct * min(1.0, epoch / args.warmup_epochs) if args.warmup_epochs > 0 else args.lambda_struct
        perm = torch.randperm(len(train_data["labels"]))
        metrics = defaultdict(float)
        nb = 0

        for s in range(0, len(perm), args.batch_size):
            idx = perm[s:s + args.batch_size]
            x = train_data["inputs"][idx].to(device)
            labels = train_data["labels"][idx].to(device)
            batch = x.shape[0]
            labels_oh = labels_to_onehot(labels).to(device)

            # ==== Phase 1: Update structure network ====
            with torch.no_grad():
                task_logits, _ = task_net(x)
                task_preds_oh = labels_to_onehot(task_logits.argmax(dim=-1)).to(device)

            neg_count_oh = labels_to_onehot(corrupt_sequence(labels.cpu(), "count", rng)).to(device)
            neg_adj_oh = labels_to_onehot(corrupt_sequence(labels.cpu(), "adjacency", rng)).to(device)
            neg_rand_oh = labels_to_onehot(corrupt_sequence(labels.cpu(), "random", rng)).to(device)

            pos_score = struct_net(labels_oh)
            neg_scores = [struct_net(n) for n in [task_preds_oh, neg_count_oh, neg_adj_oh, neg_rand_oh]]

            target = torch.ones(batch, device=device)
            struct_loss = sum(F.margin_ranking_loss(pos_score, ns, target, margin=1.0) for ns in neg_scores)

            struct_opt.zero_grad()
            struct_loss.backward()
            struct_opt.step()

            with torch.no_grad():
                all_neg = torch.stack(neg_scores, dim=1)
                best_neg = all_neg.max(dim=1).values
                contr_acc = (pos_score > best_neg).float().mean().item()
                gap = (pos_score - best_neg).mean().item()

            # ==== Phase 2: Update task-net ====
            for p in struct_net.parameters():
                p.requires_grad_(False)

            task_logits, _ = task_net(x)
            task_soft = F.softmax(task_logits, dim=-1)
            ce = F.cross_entropy(task_logits.view(-1, NUM_TAGS), labels.view(-1))
            struct_score = struct_net(task_soft)
            total_loss = ce + lam * (-struct_score.mean())

            task_opt.zero_grad()
            total_loss.backward()

            # Gradient diagnostics (every 10 batches)
            grad_ratio = 0.0
            if nb % 10 == 0:
                task_grad = sum(p.grad.norm().item()**2 for p in task_net.parameters() if p.grad is not None)**0.5
                task_opt.zero_grad()
                logits_tmp, _ = task_net(x)
                F.cross_entropy(logits_tmp.view(-1, NUM_TAGS), labels.view(-1)).backward(retain_graph=True)
                ce_gn = sum(p.grad.norm().item()**2 for p in task_net.parameters() if p.grad is not None)**0.5
                task_opt.zero_grad()
                soft_tmp = F.softmax(logits_tmp, dim=-1)
                (lam * -struct_net(soft_tmp).mean()).backward()
                st_gn = sum(p.grad.norm().item()**2 for p in task_net.parameters() if p.grad is not None)**0.5
                task_opt.zero_grad()
                grad_ratio = st_gn / (ce_gn + 1e-10)
                # Redo actual update
                logits_final, _ = task_net(x)
                soft_final = F.softmax(logits_final, dim=-1)
                loss_final = F.cross_entropy(logits_final.view(-1, NUM_TAGS), labels.view(-1)) + lam * (-struct_net(soft_final).mean())
                loss_final.backward()

            task_opt.step()

            for p in struct_net.parameters():
                p.requires_grad_(True)

            metrics["ce"] += ce.item()
            metrics["struct_loss"] += struct_loss.item()
            metrics["contr_acc"] += contr_acc
            metrics["gap"] += gap
            metrics["grad_ratio"] += grad_ratio
            nb += 1

        for k in metrics:
            metrics[k] /= nb

        m = evaluate(task_net, val_data, device)
        entry = {"epoch": epoch, "lambda": lam, **{k: v for k, v in metrics.items()},
                 **{f"val_{k}": v for k, v in m.items()}}
        history.append(entry)

        log(f"Epoch {epoch:3d} | CE: {metrics['ce']:.4f} | "
            f"ContrAcc: {metrics['contr_acc']:.3f} | Gap: {metrics['gap']:.2f} | "
            f"GradR: {metrics['grad_ratio']:.4f} | "
            f"Acc: {m['accuracy']:.4f} | CountV: {m['count_viol_rate']:.4f} | "
            f"AdjV: {m['adj_viol_rate']:.4f} | AnyV: {m['any_viol_rate']:.4f} | "
            f"lam: {lam:.2f}")

    return task_net, struct_net, history


def run_proof_2(args):
    log("=" * 70)
    log("PROOF 2: G-SEAL vs Baseline vs CRF-oracle vs Global-oracle")
    log("  Key test: G-SEAL should beat CRF-oracle on COUNT violations")
    log("=" * 70)

    device = torch.device(args.device)
    train = generate_dataset(5000, seq_len=args.seq_len, seed=42)
    val = generate_dataset(1000, seq_len=args.seq_len, seed=123)
    test = generate_dataset(1000, seq_len=args.seq_len, seed=456)

    # A: CE baseline
    baseline_net, baseline_hist = train_ce_baseline(train, val, args, device)
    baseline_test = evaluate(baseline_net, test, device)
    log(f"\nBaseline test: {_fmt(baseline_test)}")

    # B: CRF-equivalent oracle (pairwise only)
    pair_net, pair_hist = train_pairwise_oracle(train, val, args, device)
    pair_test = evaluate(pair_net, test, device)
    log(f"Pairwise oracle test: {_fmt(pair_test)}")

    # C: Global oracle (count + adjacency)
    glob_net, glob_hist = train_global_oracle(train, val, args, device)
    glob_test = evaluate(glob_net, test, device)
    log(f"Global oracle test: {_fmt(glob_test)}")

    # D: G-SEAL
    gseal_net, struct_net, gseal_hist = train_gseal(train, val, args, device)
    gseal_test = evaluate(gseal_net, test, device)
    log(f"G-SEAL test: {_fmt(gseal_test)}")

    # Results
    log("\n" + "=" * 70)
    log("PROOF 2 RESULTS:")
    log(f"{'Model':<25} {'Acc':>7} {'CountV':>8} {'AdjV':>8} {'AnyV':>8} {'AvgOff':>8}")
    log("-" * 70)
    for name, m in [("CE Baseline", baseline_test), ("Pairwise Oracle (CRF)", pair_test),
                    ("Global Oracle (upper)", glob_test), ("G-SEAL", gseal_test)]:
        log(f"{name:<25} {m['accuracy']:>7.4f} {m['count_viol_rate']:>8.4f} "
            f"{m['adj_viol_rate']:>8.4f} {m['any_viol_rate']:>8.4f} {m['avg_count_off']:>8.4f}")

    # Key checks
    gseal_beats_baseline = gseal_test["any_viol_rate"] < baseline_test["any_viol_rate"]
    gseal_beats_crf_on_count = gseal_test["count_viol_rate"] < pair_test["count_viol_rate"]

    log(f"\nG-SEAL < Baseline (any violations): {gseal_beats_baseline}")
    log(f"G-SEAL < CRF-oracle (count violations): {gseal_beats_crf_on_count}")

    result = "PASS" if (gseal_beats_baseline and gseal_beats_crf_on_count) else "FAIL"
    log(f"\nProof 2: {result}")
    log("=" * 70)

    save_results({
        "proof": 2, "result": result,
        "test": {"baseline": baseline_test, "pairwise_oracle": pair_test,
                 "global_oracle": glob_test, "gseal": gseal_test},
        "gseal_beats_baseline": gseal_beats_baseline,
        "gseal_beats_crf_on_count": gseal_beats_crf_on_count,
        "baseline_history": baseline_hist, "pairwise_history": pair_hist,
        "global_history": glob_hist, "gseal_history": gseal_hist,
    }, "proof2_v2_results.json")


# ============================================================================
# Proof 3
# ============================================================================


def run_proof_3(args):
    log("=" * 70)
    log("PROOF 3: Knowledge transfer after structure network removal")
    log("=" * 70)

    device = torch.device(args.device)
    train = generate_dataset(5000, seq_len=args.seq_len, seed=42)
    val = generate_dataset(1000, seq_len=args.seq_len, seed=123)
    test = generate_dataset(1000, seq_len=args.seq_len, seed=456)

    # Phase 1: G-SEAL training
    log("\n=== Phase 1: G-SEAL training ===")
    gseal_net, struct_net, gseal_hist = train_gseal(train, val, args, device)
    gseal_test = evaluate(gseal_net, test, device)
    log(f"\nG-SEAL test: {_fmt(gseal_test)}")

    # Phase 2: Remove structure network
    log("\n=== Phase 2: Structure network removed ===")
    standalone_test = evaluate(gseal_net, test, device)
    log(f"Standalone: {_fmt(standalone_test)}")

    # Phase 3: Continue CE-only
    log("\n=== Phase 3: Continue CE-only ===")
    ce_opt = optim.Adam(gseal_net.parameters(), lr=args.task_lr)
    cont_hist = []
    for epoch in range(args.continued_epochs):
        gseal_net.train()
        perm = torch.randperm(len(train["labels"]))
        tl = 0.0
        nb = 0
        for s in range(0, len(perm), args.batch_size):
            idx = perm[s:s + args.batch_size]
            x = train["inputs"][idx].to(device)
            labels = train["labels"][idx].to(device)
            logits, _ = gseal_net(x)
            loss = F.cross_entropy(logits.view(-1, NUM_TAGS), labels.view(-1))
            ce_opt.zero_grad()
            loss.backward()
            ce_opt.step()
            tl += loss.item()
            nb += 1
        m = evaluate(gseal_net, val, device)
        cont_hist.append({"epoch": epoch, **{f"val_{k}": v for k, v in m.items()}})
        log(f"CE-only {epoch:3d} | Acc: {m['accuracy']:.4f} | CountV: {m['count_viol_rate']:.4f} | "
            f"AdjV: {m['adj_viol_rate']:.4f} | AnyV: {m['any_viol_rate']:.4f}")

    final_test = evaluate(gseal_net, test, device)
    log(f"\nFinal (after CE continuation): {_fmt(final_test)}")

    # Baseline comparison
    log("\n=== Baseline (CE from scratch) ===")
    baseline_net, baseline_hist = train_ce_baseline(train, val, args, device)
    baseline_test = evaluate(baseline_net, test, device)
    log(f"Baseline: {_fmt(baseline_test)}")

    # Results
    log("\n" + "=" * 70)
    log("PROOF 3 RESULTS:")
    log(f"{'Stage':<30} {'CountV':>8} {'AdjV':>8} {'AnyV':>8}")
    log("-" * 60)
    for name, m in [("Baseline", baseline_test), ("G-SEAL (during training)", gseal_test),
                    ("After struct net removal", standalone_test), ("After CE continuation", final_test)]:
        log(f"{name:<30} {m['count_viol_rate']:>8.4f} {m['adj_viol_rate']:>8.4f} {m['any_viol_rate']:>8.4f}")

    transfer = standalone_test["any_viol_rate"] < baseline_test["any_viol_rate"]
    persist = final_test["any_viol_rate"] < baseline_test["any_viol_rate"]
    log(f"\nKnowledge transferred: {transfer}")
    log(f"Knowledge persists: {persist}")
    result = "PASS" if (transfer and persist) else "FAIL"
    log(f"Proof 3: {result}")
    log("=" * 70)

    save_results({
        "proof": 3, "result": result,
        "test": {"baseline": baseline_test, "gseal": gseal_test,
                 "standalone": standalone_test, "final": final_test},
        "transfer": transfer, "persist": persist,
        "gseal_history": gseal_hist, "continued_history": cont_hist,
        "baseline_history": baseline_hist,
    }, "proof3_v2_results.json")


# ============================================================================
# Utilities
# ============================================================================


def _fmt(m: dict) -> str:
    return (f"Acc={m['accuracy']:.4f} CountV={m['count_viol_rate']:.4f} "
            f"AdjV={m['adj_viol_rate']:.4f} AnyV={m['any_viol_rate']:.4f}")


def save_results(output, filename):
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")

    for d in [Path(__file__).parent / "results", LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        with open(d / filename, "w") as f:
            json.dump(output, f, indent=2, default=convert)
        log(f"Results saved to {d / filename}")


def main():
    parser = argparse.ArgumentParser(description="G-SEAL v2 Validation")
    parser.add_argument("--proof", type=str, default="all", choices=["1", "2", "3", "all"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--task_hidden_dim", type=int, default=64)
    parser.add_argument("--task_lr", type=float, default=1e-3)
    parser.add_argument("--struct_d_model", type=int, default=64)
    parser.add_argument("--struct_num_heads", type=int, default=4)
    parser.add_argument("--struct_num_layers", type=int, default=2)
    parser.add_argument("--struct_lr", type=float, default=1e-3)
    parser.add_argument("--lambda_struct", type=float, default=1.0)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--oracle_weight", type=float, default=1.0)
    parser.add_argument("--proof1_epochs", type=int, default=30)
    parser.add_argument("--gseal_epochs", type=int, default=40)
    parser.add_argument("--continued_epochs", type=int, default=20)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"proof{args.proof}_v2_{timestamp}.log"
    logger = Logger(log_path)
    sys.stdout = logger

    log(f"Logging to: {log_path}")
    log(f"Args: {vars(args)}\n")

    if args.proof in ("1", "all"):
        run_proof_1(args)
    if args.proof in ("2", "all"):
        run_proof_2(args)
    if args.proof in ("3", "all"):
        run_proof_3(args)

    sys.stdout = logger.terminal
    logger.close()


if __name__ == "__main__":
    main()
