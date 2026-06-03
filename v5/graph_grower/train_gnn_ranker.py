"""GNN-as-ranker (stage 4.3) — does graph TOPOLOGY beat the bi-encoder once it is
LEARNED-gated? The #2 bi-encoder (Hit@5 0.39 / MRR 0.28 on the mixed pool) ignores
edges; the naive untrained topo heuristic HURT. This trains the real R-GCN
(`v5.gnn_encoder.RGCNEncoder`) so a symbol's embedding absorbs its `leveraged` /
strategy neighbors, with learning deciding which edges matter.

Design rule — REFINE, don't replace: node_repr = proj(concat[ frozen Qwen-embed(1024),
gnn_topology(256) ]). Topology is an additive delta on the strong text features, so it
can only help (replacing 1024-d through the 256 bottleneck would lose to the 0.6B
bi-encoder). Query: issue -> frozen Qwen-embed -> projector -> same space. Contrastive
(issue->support, full-softmax negatives over all nodes), Qwen frozen, R-GCN + 2
projectors trained. Eval on the SAME held-out as #2 -> directly comparable to 0.39.

GPU + sentence-transformers + torch_geometric -> Linux box. See V5_V2_DESIGN §7 / READ_THIS.

  python -m v5.graph_grower.train_gnn_ranker \
    --graph graphs/grown_graph4.json \
    --nodes-file artifacts/graph_growth/swe_code_candidates.jsonl \
                 artifacts/graph_growth/swe_code_candidates_verified.jsonl \
                 <strategy_clean.jsonl> \
    --edges-file <strategy_clean.jsonl> \
    --traces <grounded_traces*.jsonl> \
    --heldout-file data/swe/retrieval_gold_heldout.jsonl \
    --emb Qwen/Qwen3-Embedding-0.6B
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from v5.graph_grower.retrieval_eval import embed_st, load_gold_file
from v5.graph_grower.topo_rerank_eval import load_nodes
from v5.gnn_encoder import (
    RGCNEncoder, GraphEncoderInputs, _node_type_id, _relation_type_id,
    _epistemic_status_id,
)


def load_all_edges(graph: str, edges_files: Sequence[str], idx: Dict[str, int]
                   ) -> Tuple[List[int], List[int], List[int]]:
    """Every edge with both endpoints in the pool -> (src_idx, dst_idx, rel_id)."""
    src, dst, rel = [], [], []

    def _add(u, v, r):
        iu, iv = idx.get(u), idx.get(v)
        if iu is not None and iv is not None:
            src.append(iu); dst.append(iv); rel.append(_relation_type_id(r or "related"))

    if graph:
        from graph_core import MemoryGraph
        g = MemoryGraph.load_json(graph)
        for e in g.edges:
            _add(getattr(e, "src", None), getattr(e, "dst", None), getattr(e, "relation", None))
    for f in edges_files or []:
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            e = json.loads(line).get("raw_edit", {})
            if e.get("op") == "add_edge":
                _add(e.get("src"), e.get("dst"), e.get("relation"))
    return src, dst, rel


def load_trace_queries(trace_paths: Sequence[str], node_ids: set) -> Dict[str, List[str]]:
    """grounded_traces -> {issue: [support symbol ids in pool]}."""
    gold: Dict[str, set] = {}
    for tp in trace_paths:
        for line in Path(tp).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            v = (json.loads(line).get("v2_grounding") or {})
            q = str(v.get("task", "")).strip()
            sup = [s for s in (v.get("support_ids") or []) if s in node_ids]
            if q and sup:
                gold.setdefault(q, set()).update(sup)
    return {q: sorted(s) for q, s in gold.items()}


class GNNRanker:
    def __init__(self, text_dim: int, proj_dim: int, device):
        import torch.nn as nn
        self.gnn = RGCNEncoder(text_embed_dim=text_dim).to(device)
        self.node_proj = nn.Linear(text_dim + 256, proj_dim).to(device)
        self.query_proj = nn.Linear(text_dim, proj_dim).to(device)
        self.device = device

    def params(self):
        return list(self.gnn.parameters()) + list(self.node_proj.parameters()) \
            + list(self.query_proj.parameters())

    def node_repr(self, inputs, text_emb):
        import torch.nn.functional as F
        gnn = self.gnn(inputs)                          # [N, 256]
        rep = self.node_proj(np_cat(text_emb, gnn))     # [N, D]
        return F.normalize(rep, dim=-1)

    def query_repr(self, q_emb):
        import torch.nn.functional as F
        return F.normalize(self.query_proj(q_emb), dim=-1)


def np_cat(a, b):
    import torch
    return torch.cat([a, b], dim=-1)


def main(argv=None) -> int:
    import torch
    import torch.nn.functional as F
    ap = argparse.ArgumentParser(description="Train the R-GCN GNN-as-ranker (topology, learned).")
    ap.add_argument("--graph", default="graphs/grown_graph4.json")
    ap.add_argument("--nodes-file", nargs="*", default=[])
    ap.add_argument("--edges-file", nargs="*", default=[])
    ap.add_argument("--traces", nargs="+", required=True)
    ap.add_argument("--heldout-file", default="data/swe/retrieval_gold_heldout.jsonl")
    ap.add_argument("--emb", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--proj-dim", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="artifacts/graph_growth/cloud_results/gnn_ranker.json")
    args = ap.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── pool + edges ────────────────────────────────────────────────────────
    node_ids, node_texts, id2type = load_nodes(args.graph, args.nodes_file)
    idx = {nid: i for i, nid in enumerate(node_ids)}
    src, dst, rel = load_all_edges(args.graph, args.edges_file, idx)
    N = len(node_ids)
    print(f"pool nodes={N} | edges={len(src)}", flush=True)

    # ── frozen Qwen features ────────────────────────────────────────────────
    print("embedding nodes (frozen Qwen)...", flush=True)
    text_emb = torch.tensor(embed_st(node_texts, args.emb, device), dtype=torch.float32, device=device)
    text_dim = text_emb.shape[1]

    inputs = GraphEncoderInputs(
        node_ids=node_ids,
        text_embeddings=text_emb,
        node_type_ids=torch.tensor([_node_type_id(id2type.get(n, "unknown")) for n in node_ids],
                                   dtype=torch.long, device=device),
        epistemic_status_ids=torch.full((N,), _epistemic_status_id("unknown"),
                                        dtype=torch.long, device=device),
        confidences=torch.ones((N, 1), dtype=torch.float32, device=device),
        edge_index=(torch.tensor([src, dst], dtype=torch.long, device=device)
                    if src else torch.zeros((2, 0), dtype=torch.long, device=device)),
        edge_type=(torch.tensor(rel, dtype=torch.long, device=device)
                   if rel else torch.zeros((0,), dtype=torch.long, device=device)),
    )

    # ── train / held-out split (SAME held-out as #2 -> comparable) ──────────
    gold_all = load_trace_queries(args.traces, set(node_ids))
    held = load_gold_file(args.heldout_file, set(node_ids)) if Path(args.heldout_file).exists() else {}
    train_q = [q for q in gold_all if q not in held]
    print(f"train queries={len(train_q)} | held-out={len(held)}", flush=True)

    q_text = train_q
    q_emb = torch.tensor(embed_st(q_text, args.emb, device), dtype=torch.float32, device=device)
    # one positive support index per train query (first in-pool support)
    pos_idx = [idx[gold_all[q][0]] for q in q_text]
    pos_t = torch.tensor(pos_idx, dtype=torch.long, device=device)

    model = GNNRanker(text_dim, args.proj_dim, device)
    opt = torch.optim.AdamW(model.params(), lr=args.lr)
    nq = len(q_text)
    for ep in range(args.epochs):
        perm = torch.randperm(nq, device=device)
        tot = 0.0
        for i in range(0, nq, args.batch_size):
            b = perm[i:i + args.batch_size]
            node_rep = model.node_repr(inputs, text_emb)        # [N, D]  (GNN fwd / step)
            q_rep = model.query_repr(q_emb[b])                  # [B, D]
            logits = q_rep @ node_rep.T                         # [B, N]
            loss = F.cross_entropy(logits, pos_t[b])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(b)
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d}  loss {tot / nq:.4f}", flush=True)

    # ── eval on held-out (rank support among all N) ─────────────────────────
    model.gnn.eval()
    with torch.no_grad():
        node_rep = model.node_repr(inputs, text_emb)
        hq = sorted(held)
        if not hq:
            print("no held-out gold in pool; skip eval"); return 0
        hq_emb = torch.tensor(embed_st(hq, args.emb, device), dtype=torch.float32, device=device)
        q_rep = model.query_repr(hq_emb)
        sims = (q_rep @ node_rep.T).cpu().numpy()
    res = _score(sims, node_ids, held, hq)
    res.update({"pool_nodes": N, "edges": len(src), "proj_dim": args.proj_dim,
                "epochs": args.epochs, "bar_biencoder": {"Hit@5": 0.386, "MRR": 0.285}})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\nGNN-ranker held-out (n={res['queries_scored']}): "
          f"Hit@1 {res['Hit@k'][1]} Hit@5 {res['Hit@k'][5]} Hit@10 {res['Hit@k'][10]} "
          f"Hit@20 {res['Hit@k'][20]} MRR {res['MRR']}")
    print(f"bi-encoder #2 bar: Hit@5 0.386 / MRR 0.285  -> GNN {'BEATS' if res['Hit@k'][5] > 0.386 else 'does NOT beat'} it")
    return 0


def _score(sims: np.ndarray, node_ids, gold, queries, ks=(1, 5, 10, 20)) -> dict:
    order = np.argsort(-sims, axis=1)
    idx_of = {nid: i for i, nid in enumerate(node_ids)}
    hit = {k: [] for k in ks}; recall = {k: [] for k in ks}; rr = []
    for qi, q in enumerate(queries):
        gidx = {idx_of[g] for g in gold[q] if g in idx_of}
        if not gidx:
            continue
        ranked = order[qi]
        first = next((r for r, ni in enumerate(ranked) if ni in gidx), None)
        rr.append(1.0 / (first + 1) if first is not None else 0.0)
        for k in ks:
            inter = len(set(ranked[:k].tolist()) & gidx)
            hit[k].append(1.0 if inter else 0.0); recall[k].append(inter / len(gidx))
    n = len(rr)
    return {"queries_scored": n,
            "MRR": round(float(np.mean(rr)), 4) if n else 0.0,
            "Hit@k": {k: round(float(np.mean(hit[k])), 4) for k in ks},
            "Recall@k": {k: round(float(np.mean(recall[k])), 4) for k in ks}}


if __name__ == "__main__":
    raise SystemExit(main())
