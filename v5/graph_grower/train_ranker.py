"""Train a retrieval RANKER (contrastive bi-encoder) on (query -> support node) pairs.

Raw embedding leaves a lot on the table for code (Hit@5 0.27) -- the query is a
bug-symptom issue, the target a `def` signature with little lexical overlap. A
contrastive fine-tune teaches the embedding space to put a query near the symbols
its fix actually touched. Gold = (question, support_ids) from SWE (and/or STEM);
node text resolved from the add_node candidates. MultipleNegativesRanking uses the
rest of the batch as negatives. After training, re-run `retrieval_eval --model <out>`
to measure the lift over the off-the-shelf embedder.

GPU + sentence-transformers -> run on a Linux box (segfaults on Windows). See
SWE_DATA_PIPELINE.md / V5_V2_DESIGN.md (the GNN-as-ranker starts here as a bi-encoder).

    python -m v5.graph_grower.train_ranker \
      --gold data/swe/retrieval_gold_code.jsonl \
      --nodes artifacts/graph_growth/swe_code_candidates.jsonl \
      --base Qwen/Qwen3-Embedding-0.6B --out models/ranker-code
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def load_node_texts(paths: Sequence[str]) -> Dict[str, str]:
    """id -> text from add_node candidate jsonls (or MemoryGraph-style {nodes:[...]})."""
    id2text: Dict[str, str] = {}
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            e = r.get("raw_edit", r)
            if e.get("op") == "add_node" and e.get("node_id"):
                id2text.setdefault(e["node_id"], e.get("text", "") or "")
    return id2text


def build_examples(gold_paths: Sequence[str], id2text: Dict[str, str]):
    from sentence_transformers import InputExample
    ex = []
    n_q = n_pair = 0
    for gp in gold_paths:
        for line in Path(gp).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            q = str(r.get("question", "")).strip()
            if not q:
                continue
            n_q += 1
            for sid in r.get("support_ids", []) or []:
                t = id2text.get(sid)
                if t:
                    ex.append(InputExample(texts=[q, t])); n_pair += 1
    print(f"  {n_q} queries -> {n_pair} (query, support) training pairs")
    return ex


def train(gold_paths, node_paths, base: str, out: str, *, epochs: int = 2,
          batch_size: int = 32, lr: float = 2e-5, max_seq: int = 256) -> str:
    from torch.utils.data import DataLoader
    from sentence_transformers import SentenceTransformer, losses

    id2text = load_node_texts(node_paths)
    print(f"node texts: {len(id2text)}")
    examples = build_examples(gold_paths, id2text)
    if not examples:
        raise ValueError("no training pairs (check gold support_ids vs node ids)")

    try:
        model = SentenceTransformer(base, trust_remote_code=True)
    except TypeError:
        model = SentenceTransformer(base)
    model.max_seq_length = max_seq
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size, drop_last=True)
    loss = losses.MultipleNegativesRankingLoss(model)   # in-batch negatives, InfoNCE-style
    warmup = int(0.1 * len(loader) * epochs)
    model.fit(train_objectives=[(loader, loss)], epochs=epochs, warmup_steps=warmup,
              optimizer_params={"lr": lr}, show_progress_bar=True, output_path=out)
    model.save(out)
    print(f"saved ranker -> {out}")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Train a contrastive retrieval ranker on query->support pairs.")
    ap.add_argument("--gold", nargs="+", required=True, help="gold jsonl(s): {question, support_ids}")
    ap.add_argument("--nodes", nargs="+", required=True, help="add_node candidate jsonl(s) for node text")
    ap.add_argument("--base", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--out", default="models/ranker-code")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args(argv)
    train(args.gold, args.nodes, args.base, args.out,
          epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
