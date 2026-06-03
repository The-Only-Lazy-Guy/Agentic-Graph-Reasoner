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


def load_gold_rows(gold_paths: Sequence[str]) -> List[dict]:
    """Slim gold {question, support_ids} -> rows (no brief -> in-batch negatives only)."""
    rows = []
    for gp in gold_paths:
        for line in Path(gp).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and str(json.loads(line).get("question", "")).strip():
                rows.append(json.loads(line))
    return rows


def load_trace_rows(trace_paths: Sequence[str]) -> List[dict]:
    """grounded_traces -> {question, support_ids, brief_ids}; brief enables HARD negatives."""
    rows = []
    for tp in trace_paths:
        for line in Path(tp).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            v = (json.loads(line).get("v2_grounding") or {})
            q = str(v.get("task", "")).strip()
            sup = v.get("support_ids") or []
            if q and sup:
                rows.append({"question": q, "support_ids": sup,
                             "brief_ids": (v.get("brief") or {}).get("retrieved_ids") or []})
    return rows


def build_examples(rows: Sequence[dict], id2text: Dict[str, str], num_hard: int = 2):
    """(query, support[, hard_neg]) examples. Hard negatives = non-support symbols from
    the SAME touched files (brief - support) -> the ranker learns this file's OTHER
    functions aren't the fix. MultipleNegativesRanking treats a 3rd text as a hard neg
    (plus in-batch negatives). Falls back to (query, support) pairs when no brief."""
    import random
    from sentence_transformers import InputExample
    rng = random.Random(0)
    ex, n_hard_used = [], 0
    for r in rows:
        q = str(r.get("question", "")).strip()
        sup = set(r.get("support_ids", []) or [])
        hard_pool = [b for b in (r.get("brief_ids") or []) if b not in sup and b in id2text]
        for sid in r.get("support_ids", []) or []:
            pt = id2text.get(sid)
            if not pt:
                continue
            if num_hard > 0 and hard_pool:
                for neg in rng.sample(hard_pool, min(num_hard, len(hard_pool))):
                    ex.append(InputExample(texts=[q, pt, id2text[neg]])); n_hard_used += 1
            else:
                ex.append(InputExample(texts=[q, pt]))
    print(f"  {len(rows)} queries -> {len(ex)} examples ({n_hard_used} with hard negatives)")
    return ex


def train(rows: List[dict], node_paths, base: str, out: str, *, epochs: int = 2,
          batch_size: int = 32, lr: float = 2e-5, max_seq: int = 256, num_hard: int = 0,
          eval_frac: float = 0.15, heldout_out: str = "data/swe/retrieval_gold_heldout.jsonl") -> str:
    import random
    from torch.utils.data import DataLoader
    from sentence_transformers import SentenceTransformer, losses

    id2text = load_node_texts(node_paths)
    print(f"node texts: {len(id2text)}")
    # leakage-safe internal split: hold out a fraction of QUERIES for eval
    random.seed(0); random.shuffle(rows)
    n_hold = int(len(rows) * eval_frac) if eval_frac > 0 else 0
    held, train_rows = rows[:n_hold], rows[n_hold:]
    if held and heldout_out:
        Path(heldout_out).parent.mkdir(parents=True, exist_ok=True)
        with open(heldout_out, "w", encoding="utf-8") as h:
            for r in held:   # held-out gold = slim {question, support_ids}
                h.write(json.dumps({"question": r["question"],
                                    "support_ids": r["support_ids"]}, ensure_ascii=False) + "\n")
        print(f"held out {len(held)} eval queries -> {heldout_out} "
              f"(eval the trained ranker on this for a leakage-free number)")
    examples = build_examples(train_rows, id2text, num_hard=num_hard)
    if not examples:
        raise ValueError("no training examples (check support_ids vs node ids)")

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
    ap.add_argument("--traces", nargs="*", default=[],
                    help="grounded_traces jsonl(s) -> enables HARD negatives from the brief (preferred)")
    ap.add_argument("--gold", nargs="*", default=[],
                    help="slim gold jsonl(s) {question, support_ids} -> in-batch negatives only")
    ap.add_argument("--nodes", nargs="+", required=True, help="add_node candidate jsonl(s) for node text")
    ap.add_argument("--base", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--out", default="models/ranker-code")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--num-hard", type=int, default=0,
                    help="hard negatives per positive (needs --traces). LOCKED default 0: the A40 A/B "
                         "(2026-06-03) showed hard negs gave NO clean win over in-batch (bottleneck = "
                         "query<->symbol semantic gap, not negative quality). Set >0 to re-enable.")
    ap.add_argument("--eval-frac", type=float, default=0.15, help="held-out query fraction (0=off)")
    ap.add_argument("--heldout-out", default="data/swe/retrieval_gold_heldout.jsonl")
    args = ap.parse_args(argv)
    rows = load_trace_rows(args.traces) if args.traces else load_gold_rows(args.gold)
    if not rows:
        raise SystemExit("no rows: pass --traces (with brief, for hard negs) or --gold")
    train(rows, args.nodes, args.base, args.out, epochs=args.epochs,
          batch_size=args.batch_size, lr=args.lr, num_hard=args.num_hard,
          eval_frac=args.eval_frac, heldout_out=args.heldout_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
