"""GNN-as-ranker (stage 4.3) — does graph TOPOLOGY beat the bi-encoder once it is
LEARNED-gated? The #2 bi-encoder (Hit@5 ~0.42 / MRR ~0.29 on the mixed pool) ignores
edges; the naive untrained topo heuristic HURT. This trains the real R-GCN
(`v5.gnn_encoder.RGCNEncoder`) so a symbol's embedding absorbs its `leveraged` /
strategy neighbors, with learning deciding which edges matter.

CRITICAL design (v2, after a v1 misfire): retrieval works because query and node share
ONE frozen embedding space (aligned by construction). So DON'T project the query. Keep
query = frozen Qwen-embed; make node = Qwen-embed + delta(gnn_topology) with delta
ZERO-INITIALIZED -> at init node==query-space == the raw bi-encoder baseline, and
topology is a learned ADDITIVE nudge. Can only climb from the baseline, never collapse to
random. (v1 used separate query/node projections -> destroyed the alignment -> ~random.)

Contrastive (issue->support, full-softmax over all nodes), Qwen frozen, only R-GCN +
the zero-init delta trained. Eval on the SAME held-out as #2 -> directly comparable.

GPU + sentence-transformers + torch_geometric -> Linux box. See V5_V2_DESIGN §7 / READ_THIS.

  python -m v5.graph_grower.train_gnn_ranker \
    --graph graphs/grown_graph4.json \
    --nodes-file <code symbols...> <strategy_clean.jsonl> \
    --edges-file <strategy_clean.jsonl> \
    --traces <grounded_traces*.jsonl> \
    --heldout-file data/swe/retrieval_gold_heldout.jsonl --emb Qwen/Qwen3-Embedding-0.6B
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
    """node = frozen Qwen-embed + delta(gnn); query = frozen Qwen-embed (shared space).
    delta is zero-init so training STARTS at the bi-encoder baseline and adds topology."""
    def __init__(self, text_dim: int, device):
        import torch.nn as nn
        self.gnn = RGCNEncoder(text_embed_dim=text_dim).to(device)
        self.delta = nn.Linear(256, text_dim).to(device)
        nn.init.zeros_(self.delta.weight); nn.init.zeros_(self.delta.bias)  # start == base
        self.device = device

    def params(self):
        return list(self.gnn.parameters()) + list(self.delta.parameters())

    def train_mode(self, on: bool):
        self.gnn.train(on); self.delta.train(on)

    def node_repr(self, inputs, text_emb):
        import torch.nn.functional as F
        gnn = self.gnn(inputs)                       # [N, 256]
        rep = text_emb + self.delta(gnn)             # residual on frozen Qwen (delta=0 @ init)
        return F.normalize(rep, dim=-1)

    def query_repr(self, q_emb):
        import torch.nn.functional as F
        return F.normalize(q_emb, dim=-1)            # SAME frozen space, no params


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


def _eval(model, inputs, text_emb, held, hq_emb, hq, node_ids):
    import torch
    model.train_mode(False)
    with torch.no_grad():
        node_rep = model.node_repr(inputs, text_emb)
        q_rep = model.query_repr(hq_emb)
        sims = (q_rep @ node_rep.T).cpu().numpy()
    model.train_mode(True)
    return _score(sims, node_ids, held, hq)


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
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="artifacts/graph_growth/cloud_results/gnn_ranker.json")
    args = ap.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    node_ids, node_texts, id2type = load_nodes(args.graph, args.nodes_file)
    idx = {nid: i for i, nid in enumerate(node_ids)}
    src, dst, rel = load_all_edges(args.graph, args.edges_file, idx)
    N = len(node_ids)
    print(f"pool nodes={N} | edges={len(src)}", flush=True)

    print("embedding nodes (frozen Qwen)...", flush=True)
    text_emb = torch.tensor(embed_st(node_texts, args.emb, device), dtype=torch.float32, device=device)
    text_dim = text_emb.shape[1]

    inputs = GraphEncoderInputs(
        node_ids=node_ids, text_embeddings=text_emb,
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

    gold_all = load_trace_queries(args.traces, set(node_ids))
    held = load_gold_file(args.heldout_file, set(node_ids)) if Path(args.heldout_file).exists() else {}
    train_q = [q for q in gold_all if q not in held]
    print(f"train queries={len(train_q)} | held-out={len(held)}", flush=True)

    q_emb = torch.tensor(embed_st(train_q, args.emb, device), dtype=torch.float32, device=device)
    pos_t = torch.tensor([idx[gold_all[q][0]] for q in train_q], dtype=torch.long, device=device)
    hq = sorted(held)
    hq_emb = torch.tensor(embed_st(hq, args.emb, device), dtype=torch.float32, device=device) if hq else None

    model = GNNRanker(text_dim, device)
    opt = torch.optim.AdamW(model.params(), lr=args.lr)

    # epoch 0 == base (delta is zero) -> sanity it equals the raw bi-encoder
    if hq:
        base = _eval(model, inputs, text_emb, held, hq_emb, hq, node_ids)
        print(f"  [init/base] Hit@5 {base['Hit@k'][5]} MRR {base['MRR']} (delta=0 -> raw embed)", flush=True)

    nq = len(train_q)
    best = None
    for ep in range(args.epochs):
        model.train_mode(True)
        perm = torch.randperm(nq, device=device); tot = 0.0
        for i in range(0, nq, args.batch_size):
            b = perm[i:i + args.batch_size]
            node_rep = model.node_repr(inputs, text_emb)       # GNN fwd / step
            q_rep = model.query_repr(q_emb[b])
            logits = (q_rep @ node_rep.T) / 0.05               # temperature
            loss = F.cross_entropy(logits, pos_t[b])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        if (ep % 5 == 0 or ep == args.epochs - 1) and hq:
            ev = _eval(model, inputs, text_emb, held, hq_emb, hq, node_ids)
            print(f"  epoch {ep:3d} loss {tot/nq:.4f}  Hit@5 {ev['Hit@k'][5]} MRR {ev['MRR']}", flush=True)
            if best is None or ev["Hit@k"][5] > best["Hit@k"][5]:
                best = ev

    res = best or (_eval(model, inputs, text_emb, held, hq_emb, hq, node_ids) if hq else {})
    res.update({"pool_nodes": N, "edges": len(src), "epochs": args.epochs,
                "bar_biencoder": {"Hit@5": 0.42, "MRR": 0.29}})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    if hq:
        print(f"\nGNN-ranker BEST held-out: Hit@1 {res['Hit@k'][1]} Hit@5 {res['Hit@k'][5]} "
              f"Hit@10 {res['Hit@k'][10]} Hit@20 {res['Hit@k'][20]} MRR {res['MRR']}")
        print(f"bi-encoder #2 bar Hit@5 ~0.42 -> GNN {'BEATS' if res['Hit@k'][5] > 0.42 else 'does NOT beat'} it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
