"""Stage 1 on the real V4 corpus: real mpnet embeddings + frozen-Qwen h_init.

End-to-end real-data run of Stage 1 (heads-only, frozen loop projections) on the
Phase 15 corpus, using the persisted-graph neighborhood, real mpnet node
embeddings, and real frozen-Qwen prefill h_init. Trains the heads that actually
have corpus labels (evidence / slot / epistemic / shortcut); planning is skipped
because corpus planning coverage is still 0% (substrate gap).

This is the first training of V5 heads on real graph states + real LM hidden
states — the synthetic trainability test proved capacity; this proves the real
pipeline trains.

    $env:KMP_DUPLICATE_LIB_OK="TRUE"; python -u -m v5.training.stage1_real
    ... --model Qwen/Qwen2.5-0.5B-Instruct   # lighter LM
"""
from __future__ import annotations

import argparse

import torch

import os

from v5.cross_attention import V5AttentionAdapter
from v5.gnn_encoder import RGCNEncoder
from v5.training.bridge import corpus_to_stage1_examples, DEFAULT_PERSISTED_GRAPH
from v5.training.providers import RealEmbedder, FrozenQwenHInitProvider, ANCHOR_LAYER
from v5.training.stage1 import Stage1Trainer, Stage1Config

CORPUS = "artifacts/phase15/phase15_corpus.jsonl"
SUBSTRATE_GRAPH = "graphs/merged_graph_substrate.json"


def _graph_path() -> str:
    """Prefer the substrate-enriched graph (planning coverage > 0) when present."""
    return SUBSTRATE_GRAPH if os.path.exists(SUBSTRATE_GRAPH) else DEFAULT_PERSISTED_GRAPH


def run(corpus_path: str = CORPUS, model_name: str = "Qwen/Qwen2.5-1.5B",
        device_str: str = None, epochs: int = 150):
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}  lm={model_name}  anchor_layer={ANCHOR_LAYER}")

    # real providers
    print("loading frozen Qwen h_init provider...")
    h_provider = FrozenQwenHInitProvider(model_name, anchor_layer=ANCHOR_LAYER, device=device)
    lm_dim = h_provider.hidden_size
    print(f"  LM hidden_size = {lm_dim}")
    print("loading mpnet embedder...")
    embedder = RealEmbedder(device)

    # frozen GNN
    gnn = RGCNEncoder().to(device).eval()
    for p in gnn.parameters():
        p.requires_grad_(False)

    # build real corpus examples (persisted neighborhood + real embeddings + real h_init)
    gpath = _graph_path()
    print(f"building Stage1Examples from the real corpus (graph={gpath}; runs Qwen prefills + mpnet)...")
    examples = corpus_to_stage1_examples(
        corpus_path, gnn=gnn, embedder=embedder, h_init_provider=h_provider,
        device=device, lm_dim=lm_dim, graph_path=gpath, hops=1)
    n_plan = sum(1 for e in examples if e.plan_anchor is not None)
    print(f"  built {len(examples)} examples  (avg nodes={sum(len(e.node_ids) for e in examples)/max(1,len(examples)):.1f}"
          f", planning-labeled: {n_plan}/{len(examples)})")

    # adapter sized to the LM
    adapter = V5AttentionAdapter(r_plan=3, r_evidence=3, lm_hidden_dim=lm_dim).to(device)
    trainer = Stage1Trainer(adapter, Stage1Config(epochs=epochs, lr=1e-3))
    print(f"trainable param tensors: {len(trainer.params)} (cross-attn projections frozen)")

    before = trainer.evaluate(examples)
    print("\n=== BEFORE (untrained heads) ===")
    _show(before)
    print("\ntraining Stage 1 on covered heads (evidence / slot / epistemic / shortcut)...")
    trainer.train(examples)
    after = trainer.evaluate(examples)
    print("\n=== AFTER ===")
    _show(after)

    print("\n=== DELTA (covered heads) ===")
    for k in ("plan_acc", "evid_acc", "slot_acc", "epi_acc", "sc_acc"):
        if k in before and k in after:
            b, a = before[k], after[k]
            if a == a and b == b:   # not NaN
                print(f"  {k:10s} {b:.2f} -> {a:.2f}  ({a-b:+.2f})")
    print("\nReal graph states + real LM h_init. With the substrate-enriched graph,")
    print("planning is now supervised (substrate planning nodes as labeled anchors).")
    print("\nREAL-CORPUS STAGE 1 COMPLETE")
    return after


def _show(m):
    for k in ("plan_acc", "evid_acc", "slot_acc", "epi_acc", "inv_acc", "sc_acc",
              "fallback_applicable", "fallback_blocked"):
        if k in m:
            v = m[k]
            print(f"  {k:20s} {'n/a' if v != v else f'{v:.2f}'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=150)
    a = ap.parse_args()
    run(a.corpus, a.model, a.device, a.epochs)
