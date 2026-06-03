"""Retrieval-quality eval — does the right node come up for a query?

INFERENCE ONLY (no training). Gold pairs are (question -> support node ids) taken
from the corpus (`outputs.answer_support_ids` / `v2_grounding.support_ids`), i.e.
the nodes the V4 answer actually rested on. We embed every graph node + every
query, rank nodes by cosine, and score retrieval with Recall@k / Hit@k / MRR /
nDCG against the gold support nodes.

Embedder-pluggable so the SAME gold + graph can A/B:
  - mpnet            : all-mpnet-base-v2 (768, retrieval-trained)   [baseline, local]
  - causal-hidden    : raw Qwen hidden state mean-pool (aligned, NOT retrieval-trained)
  - st-embed         : a sentence-transformers embedding model (e.g. Qwen3-Embedding)

Usage:
  python -m v5.graph_grower.retrieval_eval --graph graphs/grown_graph4.json \
      --embedder mpnet
  python -m v5.graph_grower.retrieval_eval --graph graphs/grown_graph4.json \
      --embedder causal-hidden --model Qwen/Qwen3.5-4B          # on the rented GPU
  python -m v5.graph_grower.retrieval_eval --graph graphs/grown_graph4.json \
      --embedder st-embed --model Qwen/Qwen3-Embedding-0.6B     # on the rented GPU
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

DEFAULT_CORPUS_GLOBS = [
    "data/corpus_shards/*.jsonl",
    "data/distillation_corpus/sessions.jsonl",
]


# ── gold (question -> support node ids) ──────────────────────────────────────
def _row_support(row: dict) -> List[str]:
    out = row.get("outputs", {}) or {}
    ids = out.get("answer_support_ids")
    if not ids:
        ids = (row.get("v2_grounding", {}) or {}).get("support_ids")
    return list(ids or [])


def _row_question(row: dict) -> str:
    return ((row.get("input", {}) or {}).get("question")
            or (row.get("v2_grounding", {}) or {}).get("task") or "")


def load_gold(globs: Sequence[str], node_ids: set) -> Dict[str, List[str]]:
    """question -> union of support node ids that EXIST in the graph (drop empties)."""
    gold: Dict[str, set] = {}
    for pat in globs:
        for path in glob.glob(pat):
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                q = _row_question(row).strip()
                sup = [s for s in _row_support(row) if s in node_ids]
                if q and sup:
                    gold.setdefault(q, set()).update(sup)
    return {q: sorted(s) for q, s in gold.items()}


# ── embedders (inference only) ───────────────────────────────────────────────
def _l2(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.clip(n, 1e-9, None)


def embed_mpnet(texts: List[str], model_name: str, device) -> np.ndarray:
    from transformers import AutoTokenizer, AutoModel
    import torch
    repo = model_name or "sentence-transformers/all-mpnet-base-v2"
    tok = AutoTokenizer.from_pretrained(repo)
    mdl = AutoModel.from_pretrained(repo).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            batch = texts[i:i + 32]
            enc = tok(batch, padding=True, truncation=True, max_length=256,
                      return_tensors="pt").to(device)
            h = mdl(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            out.append(emb.cpu().numpy())
    return _l2(np.concatenate(out, 0))


def embed_causal_hidden(texts: List[str], model_name: str, device) -> np.ndarray:
    """Raw decoder-LM representation: mean-pooled last hidden state. Aligned to the
    LM but NOT trained for cosine retrieval -- the honest 'option 2' to measure."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name, output_hidden_states=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 8):
            batch = texts[i:i + 8]
            enc = tok(batch, padding=True, truncation=True, max_length=512,
                      return_tensors="pt").to(device)
            h = mdl(**enc).hidden_states[-1]            # [B, T, D]
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            out.append(emb.float().cpu().numpy())
    return _l2(np.concatenate(out, 0))


def embed_st(texts: List[str], model_name: str, device) -> np.ndarray:
    """A sentence-transformers embedding model (retrieval-trained), e.g.
    Qwen3-Embedding -- 'option 3': family-aligned AND trained for retrieval."""
    from sentence_transformers import SentenceTransformer
    mdl = SentenceTransformer(model_name, device=str(device))
    emb = mdl.encode(texts, batch_size=32, normalize_embeddings=True,
                     show_progress_bar=False)
    return np.asarray(emb, dtype=np.float32)


EMBEDDERS = {"mpnet": embed_mpnet, "causal-hidden": embed_causal_hidden, "st-embed": embed_st}


# ── metrics ──────────────────────────────────────────────────────────────────
def score(query_emb: np.ndarray, node_emb: np.ndarray, node_ids: List[str],
          gold: Dict[str, List[str]], queries: List[str], ks=(1, 5, 10, 20)) -> dict:
    sims = query_emb @ node_emb.T                      # [Q, N] cosine
    order = np.argsort(-sims, axis=1)                  # ranked node indices per query
    idx_of = {nid: i for i, nid in enumerate(node_ids)}
    recall = {k: [] for k in ks}
    hit = {k: [] for k in ks}
    rr = []
    for qi, q in enumerate(queries):
        gold_idx = {idx_of[g] for g in gold[q] if g in idx_of}
        if not gold_idx:
            continue
        ranked = order[qi]
        first = next((r for r, ni in enumerate(ranked) if ni in gold_idx), None)
        rr.append(1.0 / (first + 1) if first is not None else 0.0)
        for k in ks:
            topk = set(ranked[:k].tolist())
            inter = len(topk & gold_idx)
            recall[k].append(inter / len(gold_idx))
            hit[k].append(1.0 if inter > 0 else 0.0)
    n = len(rr)
    return {
        "queries_scored": n,
        "MRR": round(float(np.mean(rr)), 4) if n else 0.0,
        "Recall@k": {k: round(float(np.mean(recall[k])), 4) for k in ks},
        "Hit@k": {k: round(float(np.mean(hit[k])), 4) for k in ks},
    }


def main(argv=None) -> int:
    import torch
    from graph_core import MemoryGraph
    ap = argparse.ArgumentParser(description="Node-retrieval quality eval (inference only).")
    ap.add_argument("--graph", default="graphs/grown_graph4.json")
    ap.add_argument("--embedder", choices=list(EMBEDDERS), default="mpnet")
    ap.add_argument("--model", default="", help="HF repo for the embedder (blank = mpnet default)")
    ap.add_argument("--corpus-globs", nargs="*", default=DEFAULT_CORPUS_GLOBS)
    ap.add_argument("--out", default="artifacts/graph_growth/retrieval_eval.json")
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = MemoryGraph.load_json(args.graph)
    node_ids = list(g.nodes.keys())
    node_texts = [getattr(g.nodes[n], "text", "") or "" for n in node_ids]
    gold = load_gold(args.corpus_globs, set(node_ids))
    queries = sorted(gold)
    print(f"graph={args.graph} nodes={len(node_ids)} | gold queries={len(queries)} "
          f"| embedder={args.embedder} model={args.model or '(default)'}", flush=True)
    if not queries:
        print("NO gold (question->support) pairs found; cannot score.")
        return 1

    fn = EMBEDDERS[args.embedder]
    node_emb = fn(node_texts, args.model, device)
    query_emb = fn(queries, args.model, device)
    result = score(query_emb, node_emb, node_ids, gold, queries)
    result["embedder"] = args.embedder
    result["model"] = args.model or "default"
    result["graph"] = args.graph
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
