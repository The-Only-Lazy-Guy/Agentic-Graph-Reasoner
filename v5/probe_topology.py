"""Topology=computation probe (no LM). Answers 3 questions about the injected graph:

Q1  Are two freshly-constructed RGCNEncoder()s identical? (i.e. is the eval-GNN the
    same random net the adapter trained against, or an independent draw?)
Q2  Holding nodes fixed, how much does the GNN's per-node output (the K/V the LM reads)
    change under edge conditions: real / none / shuffled-wiring / shuffled-relation?
Q3  What share of the node embedding is node-content (survives via residual) vs
    edge-mixing (the random message-passing)?

Run:  PYTHONPATH=E:\\PROJECT\\graph_v5 python -m v5.probe_topology
"""
from __future__ import annotations

import torch

from v5.gnn_encoder import (RGCNEncoder, GraphEncoderInputs, NUM_RELATIONS,
                            NUM_NODE_TYPES, TEXT_EMBED_DIM)


def _cos(a, b):
    a = a / (a.norm(dim=-1, keepdim=True) + 1e-9)
    b = b / (b.norm(dim=-1, keepdim=True) + 1e-9)
    return (a * b).sum(-1)               # [N] per-node cosine


def make_graph(N=60, E=120, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    text = torch.randn(N, TEXT_EMBED_DIM, generator=g)
    ntype = torch.randint(0, NUM_NODE_TYPES, (N,), generator=g)
    epi = torch.randint(0, 5, (N,), generator=g)
    conf = torch.rand(N, 1, generator=g)
    # realistic-ish wiring: a contains-tree backbone + a few cross edges
    src = torch.randint(0, N, (E,), generator=g)
    dst = torch.randint(0, N, (E,), generator=g)
    etype = torch.randint(0, NUM_RELATIONS, (E,), generator=g)
    inp = GraphEncoderInputs(
        node_ids=[f"n{i}" for i in range(N)],
        text_embeddings=text.to(device), node_type_ids=ntype.to(device),
        epistemic_status_ids=epi.to(device), confidences=conf.to(device),
        edge_index=torch.stack([src, dst]).to(device), edge_type=etype.to(device),
    )
    return inp, g


def variant(inp, kind, g):
    """Return a copy of inp with edges mutated."""
    E = inp.edge_index.shape[1]
    N = inp.num_nodes
    dev = inp.device
    if kind == "real":
        ei, et = inp.edge_index, inp.edge_type
    elif kind == "none":
        ei = torch.zeros((2, 0), dtype=torch.long, device=dev)
        et = torch.zeros((0,), dtype=torch.long, device=dev)
    elif kind == "shuffle_wiring":        # destroy topology, keep edge count + relation multiset
        src = torch.randint(0, N, (E,), generator=g).to(dev)
        dst = torch.randint(0, N, (E,), generator=g).to(dev)
        ei, et = torch.stack([src, dst]), inp.edge_type
    elif kind == "shuffle_relation":      # keep wiring, randomize relation labels
        ei = inp.edge_index
        et = torch.randint(0, NUM_RELATIONS, (E,), generator=g).to(dev)
    else:
        raise ValueError(kind)
    return GraphEncoderInputs(
        node_ids=inp.node_ids, text_embeddings=inp.text_embeddings,
        node_type_ids=inp.node_type_ids, epistemic_status_ids=inp.epistemic_status_ids,
        confidences=inp.confidences, edge_index=ei, edge_type=et,
    )


@torch.no_grad()
def main():
    torch.manual_seed(123)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Q1: two fresh GNNs identical? ────────────────────────────────────────
    g1 = RGCNEncoder().to(dev).eval()
    g2 = RGCNEncoder().to(dev).eval()
    diffs = [(g1.state_dict()[k] - g2.state_dict()[k]).abs().max().item()
             for k in g1.state_dict() if g1.state_dict()[k].is_floating_point()]
    print("== Q1: two freshly-constructed RGCNEncoder()s ==")
    print(f"  max |param diff| across all tensors: {max(diffs):.4e}")
    print(f"  -> {'IDENTICAL (deterministic init)' if max(diffs) < 1e-9 else 'DIFFERENT random nets'}")
    inp, gg = make_graph(device=dev)
    out1 = g1.forward(inp); out2 = g2.forward(inp)
    print(f"  same graph, two GNNs: mean per-node cosine(out1,out2) = {_cos(out1, out2).mean().item():.4f}")
    print(f"  (if eval-GNN != train-GNN, the adapter reads K/V from a net it never saw)\n")

    # ── Q2 + Q3: edge ablation on ONE fixed GNN ──────────────────────────────
    gnn = g1
    real = gnn.forward(variant(inp, "real", gg))
    print("== Q2: per-node output change vs REAL edges (one fixed GNN) ==")
    for kind in ["none", "shuffle_wiring", "shuffle_relation"]:
        out = gnn.forward(variant(inp, kind, gg))
        c = _cos(real, out)
        rel = ((real - out).norm(dim=-1) / (real.norm(dim=-1) + 1e-9))
        print(f"  {kind:18s}: mean cos={c.mean().item():.4f}  min cos={c.min().item():.4f}  "
              f"mean ||delta||/||real||={rel.mean().item():.4f}")

    print("\n== Q3: node-content vs edge-mixing share ==")
    none = gnn.forward(variant(inp, "none", gg))   # edges off => pure content path (+random convs on x)
    c_cn = _cos(real, none).mean().item()
    print(f"  cos(real, none) = {c_cn:.4f}  -> edges move the embedding by ~{(1-c_cn)*100:.1f}% (cosine)")
    print(f"  interpretation: high cos => topology barely alters the K/V the LM reads.")


if __name__ == "__main__":
    main()
