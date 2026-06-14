"""Adapter-level topology probe. Loads the REAL trained adapter and asks:
does its injected steering (h_updated) depend on the random GNN's message-passing,
or is it stable across GNN conditions (=> content-driven, GNN-robust)?

Conditions (same nodes, same h, same goal; only the node_embeddings source changes):
  gnnA_real   random GNN #1, real edges        (the actual eval regime)
  gnnA_none   random GNN #1, edges removed
  gnnB_real   random GNN #2, real edges        (different random net => train!=eval test)
  content     input_proj only (no convs)        (clean content, no random mixing)

Compares planning + evidence h_updated across conditions (cosine). If gnnA_real ~ gnnB_real
~ content, the adapter ignores the random topology mixing -> grounding is content-driven.

Run: PYTHONPATH=E:\\PROJECT\\graph_v5 PYTHONIOENCODING=utf-8 \
     python -m v5.probe_adapter --adapter artifacts/stage_cache/adapter_code_sr.pt
"""
from __future__ import annotations

import argparse
import torch

from v5.gnn_encoder import (RGCNEncoder, GraphEncoderInputs, NUM_RELATIONS,
                            NUM_NODE_TYPES, TEXT_EMBED_DIM)
from v5.cross_attention import V5AttentionAdapter
from v5.goal_encoder import GoalEncoder, encode_task_frame
from v5.subgraph import GraphMemoryKV


def _cos(a, b):
    a = a / (a.norm(dim=-1, keepdim=True) + 1e-9)
    b = b / (b.norm(dim=-1, keepdim=True) + 1e-9)
    return (a * b).sum(-1).mean().item()


def make_inputs(N=40, E=90, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    return GraphEncoderInputs(
        node_ids=[f"n{i}" for i in range(N)],
        text_embeddings=torch.randn(N, TEXT_EMBED_DIM, generator=g).to(device),
        node_type_ids=torch.randint(0, NUM_NODE_TYPES, (N,), generator=g).to(device),
        epistemic_status_ids=torch.randint(0, 5, (N,), generator=g).to(device),
        confidences=torch.rand(N, 1, generator=g).to(device),
        edge_index=torch.stack([torch.randint(0, N, (E,), generator=g),
                                torch.randint(0, N, (E,), generator=g)]).to(device),
        edge_type=torch.randint(0, NUM_RELATIONS, (E,), generator=g).to(device),
    )


def content_only(gnn, inp):
    """input_proj output only — clean node content, no message-passing."""
    type_emb = gnn.node_type_embed(inp.node_type_ids)
    epi_emb = gnn.epistemic_embed(inp.epistemic_status_ids)
    x = torch.cat([inp.text_embeddings, type_emb, epi_emb, inp.confidences], dim=-1)
    return gnn.input_proj(x)


def make_kv(node_embs, inp):
    N = node_embs.shape[0]
    dev = node_embs.device
    return GraphMemoryKV(
        node_embeddings=node_embs,
        node_ids=inp.node_ids, node_types=["fact"] * N,
        planning_mask=torch.ones(N, dtype=torch.bool, device=dev),
        evidence_mask=torch.ones(N, dtype=torch.bool, device=dev),
        invalidator_flags=torch.zeros(N, device=dev),
        slot_relevance=torch.ones(N, device=dev),
    )


def _infer_lm_dim(sd):
    # proj.norm = LayerNorm(lm_hidden_dim) -> its weight is exactly lm_hidden_dim.
    for k, v in sd.items():
        if k.endswith("proj.norm.weight"):
            return v.shape[0]
    return 2560


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="artifacts/stage_cache/adapter_code_sr.pt")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(7)

    sd = torch.load(a.adapter, map_location=dev)
    lm_dim = _infer_lm_dim(sd)
    print(f"adapter={a.adapter}  inferred lm_hidden_dim={lm_dim}", flush=True)
    adapter = V5AttentionAdapter(r_plan=3, r_evidence=4, lm_hidden_dim=lm_dim).to(dev)
    adapter.load_state_dict(sd); adapter.eval()
    goal_enc = GoalEncoder().to(dev).eval()

    tf = {"task_family": "code_fix", "required_slots": []}
    goal = encode_task_frame(tf, dev, goal_enc)
    h = torch.randn(1, lm_dim, device=dev) * 0.02      # fixed prefill-anchor stand-in

    inp = make_inputs(device=dev)
    gnnA = RGCNEncoder().to(dev).eval()
    gnnB = RGCNEncoder().to(dev).eval()

    inp_none = GraphEncoderInputs(
        node_ids=inp.node_ids, text_embeddings=inp.text_embeddings,
        node_type_ids=inp.node_type_ids, epistemic_status_ids=inp.epistemic_status_ids,
        confidences=inp.confidences,
        edge_index=torch.zeros((2, 0), dtype=torch.long, device=dev),
        edge_type=torch.zeros((0,), dtype=torch.long, device=dev),
    )

    embs = {
        "gnnA_real": gnnA.forward(inp),
        "gnnA_none": gnnA.forward(inp_none),
        "gnnB_real": gnnB.forward(inp),
        "content":   content_only(gnnA, inp),
    }

    out = {}
    for name, ne in embs.items():
        kv = make_kv(ne, inp)
        hp, _, _ = adapter.run_planning(h, goal, kv, inp.node_ids, r_max=3, task_frame=tf)
        he, _, _ = adapter.run_evidence(hp, goal, kv, inp.node_ids, r_max=4, task_frame=tf)
        out[name] = (hp, he)

    ref = "gnnA_real"
    print(f"\n== adapter steering stability vs {ref} (cosine of h_updated) ==")
    print(f"  {'condition':12s}  {'planning':>9s}  {'evidence':>9s}")
    for name in embs:
        cp = _cos(out[name][0], out[ref][0])
        ce = _cos(out[name][1], out[ref][1])
        print(f"  {name:12s}  {cp:9.4f}  {ce:9.4f}")
    print("\n  gnnB_real ~ gnnA_real => adapter robust to GNN randomness (eval-GNN != train-GNN ok)")
    print("  content   ~ gnnA_real => random message-passing irrelevant; grounding is content-driven")

    # also: how different is the cold (no-injection) h from the steered h? (sanity: injection moves it)
    print(f"\n  cos(h_init, gnnA_real planning) = {_cos(h, out['gnnA_real'][0]):.4f}  "
          f"(low => injection actually moves the hidden state)")


if __name__ == "__main__":
    main()
