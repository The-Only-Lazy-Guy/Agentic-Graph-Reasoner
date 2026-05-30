"""Integrated Stage 1 -> 2A -> 2B on ONE adapter (honest completion of Stage 2).

The standalone Stage 2B run used fresh random heads, so fallback was 1.0
everywhere. This runs the full pipeline on a single adapter so heads, routing,
and write are all trained together, then checks the integrated gates — including
the fallback behavior (drop for applicable, stay for blocked/negative).

  Stage 1 : train heads only (projections frozen)
  Stage 2A: freeze heads + W_o + gate, train Q/K/V routing
  Stage 2B: keep heads + routing, train W_o + gate (Q/K/V lower LR) + penalties

Integrated gates:
  head metrics retained · routing retained · write bounded (~0.01-0.10) ·
  negatives write least · catastrophic <= baseline · fallback drops for
  applicable · fallback stays for blocked/negative.

    $env:KMP_DUPLICATE_LIB_OK="TRUE"; python -u -m v5.training.stage_integrated
"""
from __future__ import annotations

import argparse

import torch

from v5.adapter import GraphAttentionInjector
from v5.cross_attention import V5AttentionAdapter
from v5.gnn_encoder import RGCNEncoder
from v5.goal_encoder import GoalEncoder
from v5.perturbation_baseline import evaluate_injection
from v5.training.bridge import corpus_to_stage1_examples, load_persisted_graph
from v5.training.dataset import Phase15Dataset
from v5.training.providers import RealEmbedder, FrozenQwenHInitProvider
from v5.training.stage1 import Stage1Trainer, Stage1Config
from v5.training.stage1_real import _graph_path
from v5.training.stage2 import Stage2Trainer, Stage2Config, GATE_INIT
from v5.training.stage2b_real import make_real_negatives, _write_and_fallback_by_tag

CORPUS = "artifacts/phase15/phase15_corpus.jsonl"
DEFAULT_LM = "Qwen/Qwen2.5-1.5B"


def run(model_name=DEFAULT_LM, device_str=None, e1=200, e2a=120, e2b=120, n_perturb=20):
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}  lm={model_name}")

    provider = FrozenQwenHInitProvider(model_name, device=device)
    model, tok = provider.model, provider.tok
    lm_dim = provider.hidden_size
    embedder = RealEmbedder(device)
    gnn = RGCNEncoder().to(device).eval()
    for p in gnn.parameters():
        p.requires_grad_(False)
    goal_enc = GoalEncoder().to(device).eval()

    gpath = _graph_path()
    graph = load_persisted_graph(gpath)
    pos = corpus_to_stage1_examples(CORPUS, gnn=gnn, embedder=embedder,
                                    h_init_provider=provider, device=device,
                                    lm_dim=lm_dim, graph_path=gpath, hops=1)
    negs = make_real_negatives(provider, embedder, gnn, graph, device, lm_dim, pos[0])
    examples = pos + negs
    print(f"examples: {len(pos)} positives + {len(negs)} negatives")

    adapter = V5AttentionAdapter(r_plan=3, r_evidence=4, lm_hidden_dim=lm_dim, gate_init=GATE_INIT).to(device)

    fb0 = _write_and_fallback_by_tag(adapter, examples)   # pre-training fallback

    print("\n--- Stage 1: train heads only (positives; negatives have no head labels) ---")
    s1 = Stage1Trainer(adapter, Stage1Config(epochs=e1, lr=1e-3))
    s1.train(pos)
    head_after_s1 = s1.evaluate(examples)
    print("head metrics after S1:", {k: round(v, 2) for k, v in head_after_s1.items() if v == v})

    print("\n--- Stage 2A: train Q/K/V routing (heads frozen) ---")
    Stage2Trainer(adapter, Stage2Config(sub_stage="2A", epochs=e2a, lr=2e-4)).train(examples)

    print("\n--- Stage 2B: train W_o + gate (heads + routing kept) ---")
    t2b = Stage2Trainer(adapter, Stage2Config(sub_stage="2B", epochs=e2b, lr=1e-4,
                                              lambda_delta=1.0, qkv_lr_scale=0.3))
    t2b.train(examples)

    # ── retained head metrics + routing ──────────────────────────────────────
    head_final = s1.evaluate(examples)
    routing = t2b.evaluate(examples)
    print("\nhead metrics retained (after 2B):", {k: round(v, 2) for k, v in head_final.items() if v == v})
    print(f"routing retained: plan {routing['plan_attn_precision']:.2f}  evid {routing['evid_attn_precision']:.2f}")

    # ── per-case-type fallback + write ───────────────────────────────────────
    fb1 = _write_and_fallback_by_tag(adapter, examples)
    print("\n=== integrated per-case-type (write_ratio | fallback before->after) ===")
    print(f"{'tag':12s} {'n':>3s} {'write':>7s} {'fb_before':>10s} {'fb_after':>9s}")
    wr_by = {}
    for tag in ("applicable", "blocked", "negative"):
        if tag in fb1:
            d, db = fb1[tag], fb0[tag]
            wr = sum(d["wr"]) / max(1, len(d["wr"]))
            wr_by[tag] = wr
            print(f"{tag:12s} {d['n']:>3d} {wr:>7.3f} {db['fb']/max(1,db['n']):>10.2f} {d['fb']/max(1,d['n']):>9.2f}")

    # ── perturbation re-check ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"PERTURBATION (integrated adapter, {n_perturb} q):")
    injector = GraphAttentionInjector(adapter.eval(), gnn, goal_enc, device=device)
    samples = Phase15Dataset(CORPUS).samples[:n_perturb]
    pa = evaluate_injection(model, tok, embedder, injector, graph, samples, device, max_new_tokens=60)
    print("AGGREGATE:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in pa.items()})

    # ── integrated gates ─────────────────────────────────────────────────────
    appl_fb = fb1["applicable"]["fb"] / max(1, fb1["applicable"]["n"]) if "applicable" in fb1 else 1.0
    blk_fb = fb1["blocked"]["fb"] / max(1, fb1["blocked"]["n"]) if "blocked" in fb1 else 0.0
    neg_fb = fb1["negative"]["fb"] / max(1, fb1["negative"]["n"]) if "negative" in fb1 else 0.0
    print("\n=== INTEGRATED GATES ===")
    gates = {
        "head metrics retained (slot/epi/shortcut >=0.8)": min(
            head_final.get("slot_acc", 0), head_final.get("epi_acc", 0), head_final.get("sc_acc", 0)) >= 0.8,
        "routing retained (>=0.8)": min(routing["plan_attn_precision"], routing["evid_attn_precision"]) >= 0.8,
        "write bounded (<=0.20)": routing["mean_write_ratio"] <= 0.20,
        "negatives write <= positives": wr_by.get("negative", 1) <= wr_by.get("applicable", 0) + 0.05,
        "catastrophic <= baseline (~5%)": pa["catastrophic"] <= 1,
        "fallback applicable LOW (<=0.5)": appl_fb <= 0.5,
        "fallback blocked HIGH (>=0.5)": blk_fb >= 0.5,
        "fallback negative HIGH (>=0.5)": neg_fb >= 0.5,
    }
    for k, v in gates.items():
        print(f"  [{'OK' if v else 'FAIL'}] {k}  ")
    n_ok = sum(gates.values())
    print(f"\n{n_ok}/{len(gates)} integrated gates pass.")
    print(f"fallback: applicable {appl_fb:.2f} | blocked {blk_fb:.2f} | negative {neg_fb:.2f}")
    return gates, fb1, pa


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_LM)
    ap.add_argument("--device", default=None)
    ap.add_argument("--e1", type=int, default=200)
    ap.add_argument("--e2a", type=int, default=120)
    ap.add_argument("--e2b", type=int, default=120)
    ap.add_argument("--n-perturb", type=int, default=20)
    a = ap.parse_args()
    run(a.model, a.device, a.e1, a.e2a, a.e2b, a.n_perturb)
