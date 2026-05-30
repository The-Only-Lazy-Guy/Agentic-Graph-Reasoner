"""Stage 2A on the real V4 corpus + perturbation re-check.

Order (per the staging plan; Stage 2B held until 2A proves out):
  1. build real corpus Stage1Examples (substrate graph + mpnet + frozen-Qwen h_init)
  2. Stage 2A: train Q/K/V only (W_o + gate frozen, gate small) — learn to LOOK
  3. perturbation re-check with the trained adapter: catastrophic rate must stay
     <= random baseline, hooks 1/1, drift not exploding

Since W_o + gate are frozen (gate ~0.02), Stage 2A should improve attention
routing WITHOUT much changing generation — exactly the safety property we check.

    $env:KMP_DUPLICATE_LIB_OK="TRUE"; python -u -m v5.training.stage2_real
"""
from __future__ import annotations

import argparse
import os

import torch

from v5.adapter import GraphAttentionInjector
from v5.cross_attention import V5AttentionAdapter
from v5.gnn_encoder import RGCNEncoder
from v5.goal_encoder import GoalEncoder
from v5.perturbation_baseline import evaluate_injection
from v5.training.bridge import corpus_to_stage1_examples, load_persisted_graph
from v5.training.providers import RealEmbedder, FrozenQwenHInitProvider
from v5.training.stage1_real import _graph_path
from v5.training.stage2 import Stage2Trainer, Stage2Config, GATE_INIT

CORPUS = "artifacts/phase15/phase15_corpus.jsonl"
DEFAULT_LM = "Qwen/Qwen2.5-1.5B"


def run(corpus_path=CORPUS, model_name=DEFAULT_LM, device_str=None, epochs=150, n_perturb=20):
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}  lm={model_name}")

    print("loading frozen Qwen (h_init provider + generation)...")
    h_provider = FrozenQwenHInitProvider(model_name, device=device)
    model, tok = h_provider.model, h_provider.tok           # reuse the same frozen LM
    lm_dim = h_provider.hidden_size
    embedder = RealEmbedder(device)
    gnn = RGCNEncoder().to(device).eval()
    for p in gnn.parameters():
        p.requires_grad_(False)
    goal_enc = GoalEncoder().to(device).eval()

    gpath = _graph_path()
    print(f"building real corpus examples (graph={gpath})...")
    examples = corpus_to_stage1_examples(
        corpus_path, gnn=gnn, embedder=embedder, h_init_provider=h_provider,
        device=device, lm_dim=lm_dim, graph_path=gpath, hops=1)
    n_plan = sum(1 for e in examples if e.plan_anchor is not None)
    print(f"  {len(examples)} examples (planning-labeled: {n_plan})")

    # adapter with small gate; Stage 2A trains Q/K/V only
    adapter = V5AttentionAdapter(r_plan=3, r_evidence=4, lm_hidden_dim=lm_dim, gate_init=GATE_INIT).to(device)
    trainer = Stage2Trainer(adapter, Stage2Config(sub_stage="2A", epochs=epochs, lr=2e-4))
    print(f"\nStage 2A trainable tensors: {len(trainer.params)} (W_o/gate/heads frozen)")
    before = trainer.evaluate(examples)
    print("BEFORE:", {k: round(v, 3) for k, v in before.items() if v == v})
    trainer.train(examples)
    after = trainer.evaluate(examples)
    print("AFTER :", {k: round(v, 3) for k, v in after.items() if v == v})
    print("\nattention precision delta: plan "
          f"{before['plan_attn_precision']:.2f}->{after['plan_attn_precision']:.2f}  "
          f"evid {before['evid_attn_precision']:.2f}->{after['evid_attn_precision']:.2f}")

    # ── perturbation re-check with the TRAINED adapter ───────────────────────
    print("\n" + "=" * 60)
    print(f"PERTURBATION RE-CHECK (Stage-2A adapter, {n_perturb} questions):")
    graph = load_persisted_graph(gpath)
    injector = GraphAttentionInjector(adapter.eval(), gnn, goal_enc, device=device)
    from v5.training.dataset import Phase15Dataset
    samples = Phase15Dataset(corpus_path).samples[:n_perturb]
    agg = evaluate_injection(model, tok, embedder, injector, graph, samples, device, max_new_tokens=60)
    print("\nAGGREGATE:")
    for k, v in agg.items():
        print(f"  {k:22s} {v:.3f}" if isinstance(v, float) else f"  {k:22s} {v}")

    print("\nSuccess (2A): attention routing improves; catastrophic rate stays low")
    print(f"(={agg['catastrophic']}/{agg['n']}); hooks {agg['hooks_ok']}/{agg['n']}; W_o/gate frozen so")
    print("generation barely changes. If green, Stage 2B (learn to write) is justified.")
    return after, agg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--model", default=DEFAULT_LM)
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--n-perturb", type=int, default=20)
    a = ap.parse_args()
    run(a.corpus, a.model, a.device, a.epochs, a.n_perturb)
