"""Topology rerank eval — does graph structure beat flat bi-encoder retrieval?

The #2 ranker plateaued at the query<->symbol semantic gap (issue = behavior text,
target = bare `def` signature). #3 minted STRATEGY/atom nodes whose TEXT is
behavior-level (matches the issue) and which `leveraged`-link to the support
symbols. This eval tests the thesis: a behavior-level node that matches the query
can LIFT its linked symbol above flat cosine -- one-hop message passing, the
lite "GNN-as-ranker".

Run on the MIXED pool (grown_graph4 STEM/algo/cs + code symbols + strategy nodes)
so the number reflects the REAL generalist deploy condition (distractors present),
not a sanitized code-only pool. base = flat cosine; rerank = base + alpha * (best
bridge-neighbor's query score). Reports both, side by side, on the same gold.

  base[v]    = cos(q, v)
  prop[v]    = max over bridge-neighbors u of base[u]
  rerank[v]  = base[v] + alpha * prop[v]

sentence-transformers segfaults on Windows -> run st-embed on the Linux GPU box.
Use --embedder hash for an offline logic smoke (meaningless scores, proves the
pipeline + that rerank actually moves ranks).

  python -m v5.graph_grower.topo_rerank_eval \
    --graph graphs/grown_graph4.json \
    --nodes-file artifacts/graph_growth/swe_code_candidates.jsonl \
                 artifacts/graph_growth/swe_code_candidates_verified.jsonl \
                 artifacts/graph_growth/swe_strategy_candidates_clean.jsonl \
    --edges-file artifacts/graph_growth/swe_strategy_candidates_clean.jsonl \
    --gold-file data/swe/retrieval_gold_code.jsonl \
    --embedder st-embed --model Qwen/Qwen3-Embedding-0.6B --alpha 0.5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from v5.graph_grower.retrieval_eval import EMBEDDERS, load_gold_file, score


# ── pool + edges ─────────────────────────────────────────────────────────────
def load_nodes(graph: str, nodes_files: Sequence[str]) -> Tuple[List[str], List[str], Dict[str, str]]:
    """Merge graph nodes + candidate add_node jsonls into one deduped pool.
    Returns (node_ids, node_texts, id2type)."""
    ids: List[str] = []
    texts: List[str] = []
    id2type: Dict[str, str] = {}
    seen = set()

    def _add(nid: str, text: str, ntype: str):
        if nid in seen:
            return
        seen.add(nid)
        ids.append(nid); texts.append(text or "")
        id2type[nid] = ntype or "?"

    if graph:
        from graph_core import MemoryGraph
        g = MemoryGraph.load_json(graph)
        for nid, node in g.nodes.items():
            _add(nid, getattr(node, "text", "") or "", getattr(node, "node_type", "?"))
    for f in nodes_files or []:
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            e = r.get("raw_edit", r)
            if e.get("op") == "add_node" and e.get("node_id"):
                _add(e["node_id"], e.get("text", "") or "", e.get("node_type", "?"))
    return ids, texts, id2type


def load_bridge_edges(graph: str, edges_files: Sequence[str], id2type: Dict[str, str],
                      relations: set) -> List[Tuple[str, str]]:
    """Collect (src,dst) edges whose relation is a bridge relation and whose endpoints
    are BOTH real (in the pool) and NEITHER is a hub (hubs would smear scores)."""
    edges: List[Tuple[str, str]] = []

    def _ok(u: str, v: str) -> bool:
        return (u in id2type and v in id2type
                and id2type[u] != "hub" and id2type[v] != "hub")

    if graph:
        from graph_core import MemoryGraph
        g = MemoryGraph.load_json(graph)
        for e in getattr(g, "relations", getattr(g, "edges", [])) or []:
            src = getattr(e, "src", None) or getattr(e, "source", None)
            dst = getattr(e, "dst", None) or getattr(e, "target", None)
            rel = getattr(e, "relation", None) or getattr(e, "rel", None)
            if rel in relations and src and dst and _ok(src, dst):
                edges.append((src, dst))
    for f in edges_files or []:
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            e = json.loads(line).get("raw_edit", {})
            if e.get("op") == "add_edge" and e.get("relation") in relations:
                u, v = e.get("src"), e.get("dst")
                if u and v and _ok(u, v):
                    edges.append((u, v))
    return edges


# ── rerank ───────────────────────────────────────────────────────────────────
def topo_rerank(sims: np.ndarray, node_ids: List[str], edges: List[Tuple[str, str]],
                alpha: float) -> np.ndarray:
    """rerank[q,v] = sims[q,v] + alpha * max over bridge-neighbors u of sims[q,u].
    Bidirectional (a matching strategy lifts its symbol AND vice-versa)."""
    idx = {nid: i for i, nid in enumerate(node_ids)}
    prop = np.zeros_like(sims)
    for u, v in edges:
        iu, iv = idx.get(u), idx.get(v)
        if iu is None or iv is None:
            continue
        # v gets u's column; u gets v's column (max-pool over neighbors)
        np.maximum(prop[:, iv], sims[:, iu], out=prop[:, iv])
        np.maximum(prop[:, iu], sims[:, iv], out=prop[:, iu])
    return sims + alpha * prop


def main(argv=None) -> int:
    import torch
    ap = argparse.ArgumentParser(description="Topology rerank vs flat retrieval (mixed pool).")
    ap.add_argument("--graph", default="graphs/grown_graph4.json",
                    help="base graph nodes+edges (the mixed/general substrate); '' to skip")
    ap.add_argument("--nodes-file", nargs="*", default=[],
                    help="extra add_node candidate jsonls (code symbols + strategy nodes)")
    ap.add_argument("--edges-file", nargs="*", default=[],
                    help="add_edge candidate jsonls (strategy leveraged/transfers_to/...)")
    ap.add_argument("--gold-file", default="data/swe/retrieval_gold_code.jsonl")
    ap.add_argument("--embedder", choices=list(EMBEDDERS) + ["hash"], default="st-embed")
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--alpha", type=float, default=0.5, help="bridge boost weight")
    ap.add_argument("--relations", default="leveraged,transfers_to,supports,chain_step",
                    help="comma-list of bridge relations to propagate along")
    ap.add_argument("--out", default="artifacts/graph_growth/topo_rerank_eval.json")
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    relations = {r.strip() for r in args.relations.split(",") if r.strip()}

    node_ids, node_texts, id2type = load_nodes(args.graph, args.nodes_file)
    edges = load_bridge_edges(args.graph, args.edges_file, id2type, relations)
    gold = load_gold_file(args.gold_file, set(node_ids))
    queries = sorted(gold)
    print(f"pool nodes={len(node_ids)} | bridge edges={len(edges)} "
          f"| gold queries={len(queries)} | embedder={args.embedder} alpha={args.alpha}",
          flush=True)
    if not queries:
        print("NO gold pairs land in this pool; cannot score."); return 1

    if args.embedder == "hash":
        node_emb = _hash_embed(node_texts); query_emb = _hash_embed(queries)
    else:
        fn = EMBEDDERS[args.embedder]
        node_emb = fn(node_texts, args.model, device)
        query_emb = fn(queries, args.model, device)

    sims = query_emb @ node_emb.T
    base = score(query_emb, node_emb, node_ids, gold, queries)
    reranked = topo_rerank(sims, node_ids, edges, args.alpha)
    rr = _score_sims(reranked, node_ids, gold, queries)

    out = {"pool_nodes": len(node_ids), "bridge_edges": len(edges),
           "queries_scored": base["queries_scored"], "alpha": args.alpha,
           "relations": sorted(relations), "base": base, "rerank": rr}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n{'metric':10} {'base':>8} {'rerank':>8}")
    for k in ("1", "5", "10", "20"):
        print(f"Hit@{k:<6} {base['Hit@k'][int(k)]:>8} {rr['Hit@k'][int(k)]:>8}")
    print(f"{'MRR':10} {base['MRR']:>8} {rr['MRR']:>8}")
    return 0


def _score_sims(sims: np.ndarray, node_ids, gold, queries, ks=(1, 5, 10, 20)) -> dict:
    """Same metrics as retrieval_eval.score but on a precomputed sims matrix."""
    order = np.argsort(-sims, axis=1)
    idx_of = {nid: i for i, nid in enumerate(node_ids)}
    recall = {k: [] for k in ks}; hit = {k: [] for k in ks}; rr = []
    for qi, q in enumerate(queries):
        gidx = {idx_of[g] for g in gold[q] if g in idx_of}
        if not gidx:
            continue
        ranked = order[qi]
        first = next((r for r, ni in enumerate(ranked) if ni in gidx), None)
        rr.append(1.0 / (first + 1) if first is not None else 0.0)
        for k in ks:
            inter = len(set(ranked[:k].tolist()) & gidx)
            recall[k].append(inter / len(gidx)); hit[k].append(1.0 if inter else 0.0)
    n = len(rr)
    return {"queries_scored": n,
            "MRR": round(float(np.mean(rr)), 4) if n else 0.0,
            "Recall@k": {k: round(float(np.mean(recall[k])), 4) for k in ks},
            "Hit@k": {k: round(float(np.mean(hit[k])), 4) for k in ks}}


def _hash_embed(texts: Sequence[str], dim: int = 256) -> np.ndarray:
    """Offline deterministic char-3gram hashing embedder (numpy only). For LOGIC
    smoke without torch/sentence-transformers -- scores are meaningless but it proves
    the pool/edge/rerank pipeline runs and that rerank perturbs ranks."""
    m = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        s = (t or "").lower()
        for j in range(len(s) - 2):
            h = hash(s[j:j + 3]) % dim
            m[i, h] += 1.0
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.clip(n, 1e-9, None)


if __name__ == "__main__":
    raise SystemExit(main())
