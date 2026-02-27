"""Standalone training script for BiStruct on Bibtex.

Logs per-epoch results instantly to logs/bistruct_bibtex/.
No wandb dependency required.

Usage:
    python train_bistruct.py [--cuda DEVICE] [--epochs N] [--alpha A] [--beta B]
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, average_precision_score
from skmultilearn.dataset import load_from_arff
from wcmatch import glob as wglob

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("bistruct")


# ── Data Loading ──────────────────────────────────────────────────────────────


def load_arff_data(file_pattern: str, num_labels: int = 159):
    """Load multi-label data from ARFF files matching a glob pattern."""
    all_x, all_y = [], []
    for fpath in sorted(wglob.glob(file_pattern, flags=wglob.EXTGLOB)):
        logger.info(f"Loading {fpath}")
        x, y, _, _ = load_from_arff(
            fpath, label_count=num_labels, return_attribute_definitions=True
        )
        all_x.append(x.toarray())
        all_y.append(y.toarray())
    X = np.vstack(all_x).astype(np.float32)
    Y = np.vstack(all_y).astype(np.float32)
    # Filter examples with no labels
    mask = Y.sum(axis=1) > 0
    return X[mask], Y[mask]


def make_dataloader(X, Y, batch_size=32, shuffle=True):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ── Model ─────────────────────────────────────────────────────────────────────


class BiStructModel(torch.nn.Module):
    """BiStruct model (pure PyTorch, no AllenNLP dependency)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_labels: int,
        label_embed_dim: int = 64,
        lsa_attn_dim: int = 64,
        lsa_num_heads: int = 4,
        lsa_dropout: float = 0.1,
        backward_bottleneck_dim: int = 128,
        encoder_dropout: float = 0.4,
    ):
        super().__init__()
        self.num_labels = num_labels

        # Encoder: 2-layer feedforward
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.Softplus(),
            torch.nn.Dropout(encoder_dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Softplus(),
        )

        # Classifier label embeddings G ∈ R^{L × d}
        self.classifier_embeddings = torch.nn.Parameter(
            torch.randn(num_labels, hidden_dim) * 0.02
        )

        # ── LSA ───────────────────────────────────────────────────────────
        # Learnable label embeddings E ∈ R^{L × d_l} (shared with backward)
        self.label_embeddings = torch.nn.Parameter(
            torch.randn(num_labels, label_embed_dim) * 0.02
        )

        assert lsa_attn_dim % lsa_num_heads == 0
        self.lsa_input_proj = torch.nn.Linear(label_embed_dim + 1, lsa_attn_dim)
        self.lsa_self_attn = torch.nn.MultiheadAttention(
            embed_dim=lsa_attn_dim,
            num_heads=lsa_num_heads,
            dropout=lsa_dropout,
            batch_first=True,
        )
        self.lsa_attn_norm = torch.nn.LayerNorm(lsa_attn_dim)
        self.lsa_ffn = torch.nn.Sequential(
            torch.nn.Linear(lsa_attn_dim, lsa_attn_dim * 4),
            torch.nn.GELU(),
            torch.nn.Dropout(lsa_dropout),
            torch.nn.Linear(lsa_attn_dim * 4, lsa_attn_dim),
            torch.nn.Dropout(lsa_dropout),
        )
        self.lsa_ffn_norm = torch.nn.LayerNorm(lsa_attn_dim)
        self.lsa_output_proj = torch.nn.Linear(lsa_attn_dim, 1)

        # ── Backward Model ────────────────────────────────────────────────
        self.backward_mlp = torch.nn.Sequential(
            torch.nn.Linear(label_embed_dim, backward_bottleneck_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(backward_bottleneck_dim, hidden_dim),
        )

    def forward_lsa(self, y_hat: torch.Tensor) -> torch.Tensor:
        """Label Self-Attention: refine predictions."""
        B = y_hat.shape[0]
        emb = self.label_embeddings.unsqueeze(0).expand(B, -1, -1)  # (B, L, d_l)
        tokens = torch.cat([emb, y_hat.unsqueeze(-1)], dim=-1)  # (B, L, d_l+1)
        tokens = self.lsa_input_proj(tokens)  # (B, L, attn_dim)

        attn_out, _ = self.lsa_self_attn(tokens, tokens, tokens)
        tokens = self.lsa_attn_norm(tokens + attn_out)

        ffn_out = self.lsa_ffn(tokens)
        tokens = self.lsa_ffn_norm(tokens + ffn_out)

        return self.lsa_output_proj(tokens).squeeze(-1)  # (B, L)

    def forward_backward(self, y: torch.Tensor) -> torch.Tensor:
        """Backward model: reconstruct features from labels."""
        z = y.unsqueeze(-1) * self.label_embeddings.unsqueeze(0)  # (B, L, d_l)
        num_active = y.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1)
        z_pooled = z.sum(dim=1) / num_active  # (B, d_l)
        return self.backward_mlp(z_pooled)  # (B, d)

    def forward(self, x, y=None):
        # Encode
        h = self.encoder(x)  # (B, d)

        # Classify
        logits = h @ self.classifier_embeddings.T  # (B, L)
        y_hat = torch.sigmoid(logits)

        # Refine via LSA
        refined_logits = self.forward_lsa(y_hat)  # (B, L)

        if y is None:
            return {"refined_logits": refined_logits, "logits": logits}

        # Losses
        loss_task = F.binary_cross_entropy_with_logits(logits, y)
        loss_refine = F.binary_cross_entropy_with_logits(refined_logits, y)

        # Backward path
        h_hat = self.forward_backward(y)
        loss_align = (
            torch.mean((h.detach() - h_hat) ** 2)
            + torch.mean((h - h_hat.detach()) ** 2)
        )

        return {
            "logits": logits,
            "refined_logits": refined_logits,
            "loss_task": loss_task,
            "loss_refine": loss_refine,
            "loss_align": loss_align,
        }


# ── Evaluation ────────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    n_batches = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        out = model(x, y)
        total_loss += (out["loss_task"] + out["loss_refine"]).item()
        n_batches += 1

        preds = torch.sigmoid(out["refined_logits"]).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(y.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # Fixed-threshold F1
    binary_preds = (all_preds >= 0.5).astype(float)
    f1_scores = [
        f1_score(all_labels[i], binary_preds[i]) for i in range(len(all_labels))
    ]
    fixed_f1 = float(np.mean(f1_scores))

    # MAP
    maps = []
    for i in range(len(all_labels)):
        if all_labels[i].sum() > 0:
            maps.append(average_precision_score(all_labels[i], all_preds[i]))
    mean_ap = float(np.mean(maps)) if maps else 0.0

    model.train()
    return {
        "loss": total_loss / max(n_batches, 1),
        "fixed_f1": fixed_f1,
        "MAP": mean_ap,
    }


# ── Training ──────────────────────────────────────────────────────────────────


def train(args):
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Set up file logging
    file_handler = logging.FileHandler(log_dir / "training.log")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    logger.info(f"Config: {vars(args)}")

    # Device
    device = torch.device(
        f"cuda:{args.cuda}" if args.cuda >= 0 and torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Device: {device}")

    # Load data
    data_root = "./data/bibtex_stratified10folds_meka"
    train_X, train_Y = load_arff_data(f"{data_root}/Bibtex-fold@(1|2|3|4|5|6).arff")
    val_X, val_Y = load_arff_data(f"{data_root}/Bibtex-fold@(7|8).arff")
    test_X, test_Y = load_arff_data(f"{data_root}/Bibtex-fold@(9|10).arff")
    logger.info(
        f"Data: train={len(train_X)}, val={len(val_X)}, test={len(test_X)}"
    )

    train_loader = make_dataloader(train_X, train_Y, batch_size=args.batch_size)
    val_loader = make_dataloader(val_X, val_Y, batch_size=args.batch_size, shuffle=False)
    test_loader = make_dataloader(
        test_X, test_Y, batch_size=args.batch_size, shuffle=False
    )

    num_labels = train_Y.shape[1]
    input_dim = train_X.shape[1]
    logger.info(f"Input dim: {input_dim}, Num labels: {num_labels}")

    # Model
    model = BiStructModel(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_labels=num_labels,
        label_embed_dim=args.label_embed_dim,
        lsa_attn_dim=args.lsa_attn_dim,
        lsa_num_heads=args.lsa_num_heads,
        lsa_dropout=args.lsa_dropout,
        backward_bottleneck_dim=args.backward_bottleneck_dim,
        encoder_dropout=args.encoder_dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    classifier_params = model.classifier_embeddings.numel()
    lsa_overhead = (
        model.label_embeddings.numel()
        + sum(p.numel() for p in model.lsa_input_proj.parameters())
        + sum(p.numel() for p in model.lsa_self_attn.parameters())
        + sum(p.numel() for p in model.lsa_attn_norm.parameters())
        + sum(p.numel() for p in model.lsa_ffn.parameters())
        + sum(p.numel() for p in model.lsa_ffn_norm.parameters())
        + sum(p.numel() for p in model.lsa_output_proj.parameters())
    )
    backward_params = sum(p.numel() for p in model.backward_mlp.parameters())
    logger.info(
        f"Params: total={total_params:,} | encoder={encoder_params:,} "
        f"| classifier={classifier_params:,} | LSA={lsa_overhead:,} "
        f"| backward={backward_params:,}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, verbose=True
    )

    # Training state
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    all_epoch_results = []

    # Save config
    with open(log_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    logger.info("=" * 70)
    logger.info("Starting training")
    logger.info("=" * 70)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()

        train_loss_task = 0.0
        train_loss_refine = 0.0
        train_loss_align = 0.0
        n_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x, y)

            loss = (
                out["loss_task"]
                + args.alpha * out["loss_align"]
                + args.beta * out["loss_refine"]
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            optimizer.step()

            train_loss_task += out["loss_task"].item()
            train_loss_refine += out["loss_refine"].item()
            train_loss_align += out["loss_align"].item()
            n_batches += 1

        # Average training losses
        train_loss_task /= n_batches
        train_loss_refine /= n_batches
        train_loss_align /= n_batches
        train_loss_total = (
            train_loss_task + args.alpha * train_loss_align + args.beta * train_loss_refine
        )

        # Evaluate
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["fixed_f1"])

        epoch_time = time.time() - epoch_start

        # Build epoch result
        epoch_result = {
            "epoch": epoch,
            "time_sec": round(epoch_time, 2),
            "lr": optimizer.param_groups[0]["lr"],
            "train": {
                "loss_total": round(train_loss_total, 6),
                "loss_task": round(train_loss_task, 6),
                "loss_refine": round(train_loss_refine, 6),
                "loss_align": round(train_loss_align, 6),
            },
            "val": {
                "loss": round(val_metrics["loss"], 6),
                "fixed_f1": round(val_metrics["fixed_f1"], 6),
                "MAP": round(val_metrics["MAP"], 6),
            },
        }

        # Check for improvement
        is_best = val_metrics["fixed_f1"] > best_val_f1
        if is_best:
            best_val_f1 = val_metrics["fixed_f1"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), log_dir / "best_model.pt")
            epoch_result["is_best"] = True
        else:
            patience_counter += 1

        all_epoch_results.append(epoch_result)

        # ── Instant logging ───────────────────────────────────────────────
        # Write per-epoch result immediately
        with open(log_dir / f"epoch_{epoch:03d}.json", "w") as f:
            json.dump(epoch_result, f, indent=2)

        # Also append to a single results file for convenience
        with open(log_dir / "all_epochs.jsonl", "a") as f:
            f.write(json.dumps(epoch_result) + "\n")

        # Console + file log
        best_marker = " *BEST*" if is_best else ""
        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} ({epoch_time:.1f}s) | "
            f"Train loss: {train_loss_total:.4f} "
            f"(task={train_loss_task:.4f} align={train_loss_align:.4f} refine={train_loss_refine:.4f}) | "
            f"Val F1: {val_metrics['fixed_f1']:.4f}  MAP: {val_metrics['MAP']:.4f}{best_marker}"
        )

        # Early stopping
        if patience_counter >= args.patience:
            logger.info(
                f"Early stopping at epoch {epoch}. "
                f"Best val F1: {best_val_f1:.4f} at epoch {best_epoch}"
            )
            break

    # ── Final evaluation on test set ──────────────────────────────────────
    logger.info("=" * 70)
    logger.info(f"Training complete. Best val F1: {best_val_f1:.4f} at epoch {best_epoch}")
    logger.info("Loading best model for test evaluation...")

    model.load_state_dict(torch.load(log_dir / "best_model.pt", map_location=device))
    test_metrics = evaluate(model, test_loader, device)

    logger.info(
        f"TEST results: F1={test_metrics['fixed_f1']:.4f}  MAP={test_metrics['MAP']:.4f}"
    )

    # Save final summary
    summary = {
        "best_epoch": best_epoch,
        "best_val_f1": round(best_val_f1, 6),
        "test": {
            "fixed_f1": round(test_metrics["fixed_f1"], 6),
            "MAP": round(test_metrics["MAP"], 6),
        },
        "total_epochs": epoch,
        "config": vars(args),
    }
    with open(log_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"All results saved to {log_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Train BiStruct on Bibtex")
    # Architecture
    parser.add_argument("--hidden-dim", type=int, default=400)
    parser.add_argument("--label-embed-dim", type=int, default=64)
    parser.add_argument("--lsa-attn-dim", type=int, default=64)
    parser.add_argument("--lsa-num-heads", type=int, default=4)
    parser.add_argument("--lsa-dropout", type=float, default=0.1)
    parser.add_argument("--backward-bottleneck-dim", type=int, default=128)
    parser.add_argument("--encoder-dropout", type=float, default=0.4)
    # Loss weights
    parser.add_argument("--alpha", type=float, default=0.1, help="Alignment loss weight")
    parser.add_argument("--beta", type=float, default=1.0, help="Refinement loss weight")
    # Training
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--grad-norm", type=float, default=10.0)
    parser.add_argument("--cuda", type=int, default=-1, help="CUDA device (-1=CPU)")
    # Logging
    parser.add_argument(
        "--log-dir", type=str, default="logs/bistruct_bibtex",
        help="Directory for epoch logs",
    )

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
