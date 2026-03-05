"""BERT-adapter on AAPD (Arxiv Academic Paper Dataset) for multi-label classification.

Following the SEAL paper (Appendix F): BERT with Pfeiffer adapter (BERT frozen,
only adapter + classifier trained). Cross-entropy baseline.

Usage:
    python train_bert_aapd.py [--cuda 0] [--epochs 20] [--lr 1e-4] [--batch-size 16]
"""

import argparse
import csv
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, average_precision_score
import adapters
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("bert_aapd")

NUM_LABELS = 54
MAX_SEQ_LEN = 512


# ── Dataset ──────────────────────────────────────────────────────────────────


class AAPDDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer: BertTokenizer, max_len: int = MAX_SEQ_LEN):
        texts = []
        labels = []

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)  # skip header
            for row in reader:
                texts.append(row[0])
                labels.append([int(x) for x in row[1:]])

        self.labels = np.array(labels, dtype=np.float32)
        logger.info(f"Loaded {len(texts)} examples from {csv_path}, pre-tokenizing...")

        # Pre-tokenize all texts at once (much faster than per-item)
        encodings = tokenizer(
            texts, max_length=max_len, padding="max_length", truncation=True, return_tensors="pt"
        )
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        logger.info(f"Pre-tokenization done. Shape: {self.input_ids.shape}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


# ── Model ────────────────────────────────────────────────────────────────────


class BertAdapterMultiLabelClassifier(nn.Module):
    """BERT with Pfeiffer adapter (frozen BERT) + classifier for multi-label classification."""

    def __init__(self, model_name: str = "bert-base-uncased", num_labels: int = NUM_LABELS,
                 dropout: float = 0.1, adapter_config: str = "pfeiffer"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        adapters.init(self.bert)
        self.bert.add_adapter("task_adapter", config=adapter_config)
        self.bert.set_active_adapters("task_adapter")
        self.bert.train_adapter("task_adapter")  # freezes BERT, only adapter trainable

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output  # [CLS] token
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


# ── Evaluation ───────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        total_loss += loss.item()
        n_batches += 1

        preds = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # Per-instance F1
    binary_preds = (all_preds >= 0.5).astype(float)
    f1_scores = [f1_score(all_labels[i], binary_preds[i], zero_division=0)
                 for i in range(len(all_labels))]
    fixed_f1 = float(np.mean(f1_scores))

    # Micro F1
    micro_f1 = f1_score(all_labels, binary_preds, average="micro", zero_division=0)

    # Macro F1
    macro_f1 = f1_score(all_labels, binary_preds, average="macro", zero_division=0)

    # MAP
    maps = []
    for i in range(len(all_labels)):
        if all_labels[i].sum() > 0:
            maps.append(average_precision_score(all_labels[i], all_preds[i]))
    mean_ap = float(np.mean(maps)) if maps else 0.0

    model.train()
    return {
        "loss": total_loss / max(n_batches, 1),
        "instance_f1": fixed_f1,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "MAP": mean_ap,
    }


# ── Training ─────────────────────────────────────────────────────────────────


def train(args):
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_dir / "training.log")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)
    logger.info(f"Config: {vars(args)}")

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.cuda}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # Tokenizer & datasets
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    data_dir = Path(args.data_dir)

    train_dataset = AAPDDataset(data_dir / "train.csv", tokenizer, max_len=args.max_seq_len)
    val_dataset = AAPDDataset(data_dir / "dev.csv", tokenizer, max_len=args.max_seq_len)
    test_dataset = AAPDDataset(data_dir / "test.csv", tokenizer, max_len=args.max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    logger.info(f"Data: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    # Model
    model = BertAdapterMultiLabelClassifier(
        model_name=args.model_name,
        num_labels=NUM_LABELS,
        dropout=args.dropout,
        adapter_config=args.adapter_config,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Params: total={total_params:,}, trainable={trainable_params:,}")

    # Optimizer: only adapter + classifier params (BERT is frozen)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # Linear warmup scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Training state
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    all_epoch_results = []

    with open(log_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    logger.info("=" * 70)
    logger.info("Starting BERT-adapter training on AAPD (BERT frozen)")
    logger.info("=" * 70)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()

        train_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = F.binary_cross_entropy_with_logits(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            n_batches += 1

            if n_batches % args.log_every == 0:
                logger.info(
                    f"  Epoch {epoch} step {n_batches}/{len(train_loader)} | "
                    f"loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}"
                )

        train_loss /= n_batches

        # Evaluate
        val_metrics = evaluate(model, val_loader, device)
        epoch_time = time.time() - epoch_start

        epoch_result = {
            "epoch": epoch,
            "time_sec": round(epoch_time, 2),
            "lr": scheduler.get_last_lr()[0],
            "train_loss": round(train_loss, 6),
            "val": {k: round(v, 6) for k, v in val_metrics.items()},
        }

        is_best = val_metrics["micro_f1"] > best_val_f1
        if is_best:
            best_val_f1 = val_metrics["micro_f1"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), log_dir / "best_model.pt")
            epoch_result["is_best"] = True
        else:
            patience_counter += 1

        all_epoch_results.append(epoch_result)

        with open(log_dir / f"epoch_{epoch:03d}.json", "w") as f:
            json.dump(epoch_result, f, indent=2)
        with open(log_dir / "all_epochs.jsonl", "a") as f:
            f.write(json.dumps(epoch_result) + "\n")

        best_marker = " *BEST*" if is_best else ""
        logger.info(
            f"Epoch {epoch:2d}/{args.epochs} ({epoch_time:.1f}s) | "
            f"Train loss: {train_loss:.4f} | "
            f"Val inst_F1: {val_metrics['instance_f1']:.4f}  "
            f"micro_F1: {val_metrics['micro_f1']:.4f}  "
            f"macro_F1: {val_metrics['macro_f1']:.4f}  "
            f"MAP: {val_metrics['MAP']:.4f}{best_marker}"
        )

        if patience_counter >= args.patience:
            logger.info(f"Early stopping at epoch {epoch}. Best micro_F1: {best_val_f1:.4f} at epoch {best_epoch}")
            break

    # ── Test evaluation ──────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info(f"Training complete. Best val micro_F1: {best_val_f1:.4f} at epoch {best_epoch}")
    logger.info("Loading best model for test evaluation...")

    model.load_state_dict(torch.load(log_dir / "best_model.pt", map_location=device))
    test_metrics = evaluate(model, test_loader, device)

    logger.info(
        f"TEST: inst_F1={test_metrics['instance_f1']:.4f}  "
        f"micro_F1={test_metrics['micro_f1']:.4f}  "
        f"macro_F1={test_metrics['macro_f1']:.4f}  "
        f"MAP={test_metrics['MAP']:.4f}"
    )

    summary = {
        "best_epoch": best_epoch,
        "best_val_micro_f1": round(best_val_f1, 6),
        "test": {k: round(v, 6) for k, v in test_metrics.items()},
        "total_epochs": epoch,
        "config": vars(args),
    }
    with open(log_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"All results saved to {log_dir}/")


def main():
    parser = argparse.ArgumentParser(description="BERT-adapter on AAPD")
    # Model
    parser.add_argument("--model-name", type=str, default="bert-base-uncased")
    parser.add_argument("--adapter-config", type=str, default="pfeiffer")
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    # Training
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100, help="Log every N steps")
    # Data
    parser.add_argument("--data-dir", type=str, default="data/aapd")
    # Hardware
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, mps, cuda:0")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device index (used when device=auto)")
    # Logging
    parser.add_argument("--log-dir", type=str, default="logs/bert_adapter_aapd")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
