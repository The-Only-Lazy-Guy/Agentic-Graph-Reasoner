"""Train the task-complexity classifier (Option C: embedding MLP).

Reads labeled data from the distillation corpus, embeds each question,
trains a small MLP, and saves the checkpoint.

Usage:
    python scripts/train_classifier.py
    python scripts/train_classifier.py --corpus data/distillation_corpus/sessions.jsonl
    python scripts/train_classifier.py --labeled data/classifier/labeled.jsonl  # pre-labeled
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from embedder import encode_one
from reasoning.task_classifier import (
    ComplexityMLP,
    LEVEL_TO_IDX,
    IDX_TO_LEVEL,
    LEVELS,
    build_training_set,
    DEFAULT_CHECKPOINT_PATH,
)


def embed_questions(questions: list[str]) -> np.ndarray:
    """Embed all questions (sequential; ~50ms each)."""
    print(f"  embedding {len(questions)} questions...")
    embs = []
    for i, q in enumerate(questions):
        embs.append(encode_one(q))
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(questions)}")
    return np.stack(embs)


def train(
    questions: list[str],
    labels: list[str],
    *,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    val_split: float = 0.15,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
) -> dict:
    """Train the MLP and save the checkpoint. Returns eval metrics."""
    X = embed_questions(questions)
    y = np.array([LEVEL_TO_IDX[l] for l in labels], dtype=np.int64)

    # Train/val split (deterministic shuffle)
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(X))
    n_val = max(1, int(len(X) * val_split))
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    X_train = torch.from_numpy(X[train_idx]).float()
    y_train = torch.from_numpy(y[train_idx]).long()
    X_val = torch.from_numpy(X[val_idx]).float()
    y_val = torch.from_numpy(y[val_idx]).long()

    print(f"  train: {len(X_train)}  val: {len(X_val)}")
    print(f"  label distribution (train): {dict(zip(*np.unique(y[train_idx], return_counts=True)))}")
    print(f"  label distribution (val):   {dict(zip(*np.unique(y[val_idx], return_counts=True)))}")

    model = ComplexityMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    best_val_acc = 0.0
    best_state = None
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            logits = model(X_val)
            preds = logits.argmax(dim=-1)
            acc = float((preds == y_val).float().mean().item())
        if acc > best_val_acc:
            best_val_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    epoch {epoch+1:3d}  loss={total_loss/len(loader):.4f}  val_acc={acc:.3f}  best={best_val_acc:.3f}")

    # Save best checkpoint
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, str(checkpoint_path))
    print(f"\n  checkpoint saved: {checkpoint_path}  (val_acc={best_val_acc:.3f})")

    # Per-class metrics
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val).argmax(dim=-1).numpy()
    y_val_np = y_val.numpy()
    per_class = {}
    for idx, name in IDX_TO_LEVEL.items():
        mask_true = y_val_np == idx
        mask_pred = val_preds == idx
        tp = int((mask_true & mask_pred).sum())
        fp = int((~mask_true & mask_pred).sum())
        fn = int((mask_true & ~mask_pred).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[name] = {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)}
    print(f"\n  per-class (val):")
    for name, m in per_class.items():
        print(f"    {name:10s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")

    return {
        "train_size": len(X_train),
        "val_size": len(X_val),
        "best_val_acc": best_val_acc,
        "per_class": per_class,
        "checkpoint": str(checkpoint_path),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="data/distillation_corpus/sessions.jsonl",
                    help="path to the distillation corpus")
    ap.add_argument("--labeled", default=None,
                    help="pre-labeled JSONL (each line: {question, label})")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT_PATH))
    args = ap.parse_args()

    if args.labeled:
        print(f"Loading pre-labeled data from {args.labeled}")
        with open(args.labeled, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        print(f"Building training set from corpus: {args.corpus}")
        rows = build_training_set(Path(args.corpus))

    if not rows:
        print("No training data. Run some v4 sessions first to populate the corpus.")
        sys.exit(1)

    questions = [r["question"] for r in rows]
    labels = [r["label"] for r in rows]
    print(f"  {len(rows)} labeled examples")

    results = train(
        questions, labels,
        epochs=args.epochs,
        lr=args.lr,
        checkpoint_path=Path(args.checkpoint),
    )
    print(f"\nDone. Results: {json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()
