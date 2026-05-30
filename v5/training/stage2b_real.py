"""Stage 2B on the real V4 corpus — controlled WRITE-SAFETY milestone.

NOT an answer-quality milestone. The question: can the adapter WRITE graph signal
into the LM residual stream without breaking generation stability or fallback?

Protocol (per the staging plan):
  - frozen: base LM, GNN, heads, overlay, (LoRA)
  - train:  W_o + gate (full LR + decay), Q/K/V (lower LR; routing already good)
  - losses: attention CE + gate^2 residual penalty + negative diffuse-attention
  - data:   real corpus positives + a few real negatives (no-graph questions)
  - 2A first (learn to look), then 2B (learn to write)

Pass gates (re-run perturbation harness):
  catastrophic rate <= random baseline (ideally ~0); hooks 20/20; no length
  collapse; no gibberish; semantic sim not collapsing; write ratio bounded
  (~0.01-0.15); fallback retained (ON for blocked/negative, low for applicable);
  negatives have low write ratio.

    $env:KMP_DUPLICATE_LIB_OK="TRUE"; python -u -m v5.training.stage2b_real
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import torch

from v5.adapter import GraphAttentionInjector
from v5.cross_attention import V5AttentionAdapter
from v5.exit_condition import fallback_needed
from v5.gnn_encoder import RGCNEncoder
from v5.goal_encoder import GoalEncoder
from v5.perturbation_baseline import evaluate_injection
from v5.training.bridge import (corpus_to_stage1_examples, load_persisted_graph,
                                _neighborhood, _goal_for)
from v5.training.dataset import Phase15Dataset
from v5.training.providers import RealEmbedder, FrozenQwenHInitProvider
from v5.training.stage1 import Stage1Example
from v5.training.stage1_real import _graph_path
from v5.training.stage2 import Stage2Trainer, Stage2Config, GATE_INIT

CORPUS = "artifacts/phase15/phase15_corpus.jsonl"
DEFAULT_LM = "Qwen/Qwen2.5-1.5B"

NO_GRAPH_QUESTIONS = [
    "Hello, how are you today?",
    "Write a short haiku about the ocean.",
    "What is 2 + 2?",
    "Translate 'good morning' into French.",
    "Tell me a fun fact about cats.",
]


def make_real_negatives(provider, embedder, gnn, graph, device, lm_dim, sample):
    """Real negatives: no-graph questions with real h_init over an irrelevant
    subgraph. Correct behavior: diffuse attention, small write, fallback."""
    node_ids = _neighborhood(graph, sample.node_ids, hops=1, max_nodes=18)
    text_emb = embedder.embed_nodes({nid: (getattr(graph.nodes[nid], "text", "") or "") for nid in node_ids})
    from v5.subgraph import build_active_subgraph
    asg = build_active_subgraph(graph, node_ids, text_emb, device, sample.task_frame)
    with torch.no_grad():
        kv = gnn.encode_to_kv(asg.encoder_inputs, asg)
    negs = []
    for q in NO_GRAPH_QUESTIONS:
        h = provider(q, sample.task_frame)
        negs.append(Stage1Example(
            h_init=h, graph_kv=kv, goal=_goal_for(sample.task_frame, device),
            node_ids=node_ids, task_frame=sample.task_frame,
            plan_anchor=None, evid_anchor=None, slot_target=None,
            epi_target=None, inv_target=None, shortcut_target=None,
            tag="negative",
        ))
    return negs


@torch.no_grad()
def _write_and_fallback_by_tag(adapter, examples):
    """Per-tag mean write ratio + fallback rate (heads frozen, but h changed by 2B)."""
    adapter.eval()
    agg = defaultdict(lambda: {"wr": [], "fb": 0, "n": 0})
    for ex in examples:
        _, ps, _ = adapter.run_planning(ex.h_init, ex.goal, ex.graph_kv, ex.node_ids, task_frame=ex.task_frame)
        _, es, _ = adapter.run_evidence(ps.h_r, ex.goal, ex.graph_kv, ex.node_ids, task_frame=ex.task_frame)
        wr = (ps.write_ratios or []) + (es.write_ratios or [])
        d = agg[ex.tag]
        d["wr"] += wr
        d["fb"] += int(fallback_needed(es, ex.task_frame))
        d["n"] += 1
    return agg


def run(corpus_path=CORPUS, model_name=DEFAULT_LM, device_str=None,
        epochs_2a=120, epochs_2b=120, n_perturb=20):
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}  lm={model_name}")

    print("loading frozen Qwen (h_init + generation)...")
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
    print(f"building real corpus examples (graph={gpath})...")
    pos = corpus_to_stage1_examples(corpus_path, gnn=gnn, embedder=embedder,
                                    h_init_provider=provider, device=device,
                                    lm_dim=lm_dim, graph_path=gpath, hops=1)
    negs = make_real_negatives(provider, embedder, gnn, graph, device, lm_dim, pos[0])
    examples = pos + negs
    print(f"  {len(pos)} positives + {len(negs)} negatives = {len(examples)}")

    adapter = V5AttentionAdapter(r_plan=3, r_evidence=4, lm_hidden_dim=lm_dim, gate_init=GATE_INIT).to(device)

    print("\n--- Stage 2A (learn to look) ---")
    Stage2Trainer(adapter, Stage2Config(sub_stage="2A", epochs=epochs_2a, lr=2e-4)).train(examples)

    print("\n--- Stage 2B (learn to write; W_o+gate full LR, Q/K/V lower) ---")
    fb_before = _write_and_fallback_by_tag(adapter, examples)
    t2b = Stage2Trainer(adapter, Stage2Config(sub_stage="2B", epochs=epochs_2b, lr=1e-4,
                                              lambda_delta=1.0, qkv_lr_scale=0.3))
    t2b.train(examples)
    after = t2b.evaluate(examples)

    # ── per-case-type report ─────────────────────────────────────────────────
    fb_after = _write_and_fallback_by_tag(adapter, examples)
    print("\n=== per-case-type report (after 2B) ===")
    print(f"{'tag':12s} {'n':>3s} {'write_ratio':>11s} {'fallback_before':>15s} {'fallback_after':>14s}")
    for tag in ("applicable", "blocked", "negative"):
        if tag in fb_after:
            d, db = fb_after[tag], fb_before[tag]
            wr = sum(d["wr"]) / max(1, len(d["wr"]))
            print(f"{tag:12s} {d['n']:>3d} {wr:>11.3f} "
                  f"{db['fb']/max(1,db['n']):>15.2f} {d['fb']/max(1,d['n']):>14.2f}")
    print(f"\ngates plan/evid: {after['plan_gate']:.3f} / {after['evid_gate']:.3f}  "
          f"overall mean_write_ratio: {after['mean_write_ratio']:.3f}")

    # ── perturbation re-check (write-safety) ─────────────────────────────────
    print("\n" + "=" * 60)
    print(f"PERTURBATION RE-CHECK (Stage-2B adapter, {n_perturb} questions):")
    injector = GraphAttentionInjector(adapter.eval(), gnn, goal_enc, device=device)
    samples = Phase15Dataset(corpus_path).samples[:n_perturb]
    pa = evaluate_injection(model, tok, embedder, injector, graph, samples, device, max_new_tokens=60)
    print("\nAGGREGATE:")
    for k, v in pa.items():
        print(f"  {k:22s} {v:.3f}" if isinstance(v, float) else f"  {k:22s} {v}")

    # ── pass gates ───────────────────────────────────────────────────────────
    neg_wr = sum(fb_after["negative"]["wr"]) / max(1, len(fb_after["negative"]["wr"])) if "negative" in fb_after else 0.0
    print("\n=== WRITE-SAFETY GATES ===")
    gates = {
        "catastrophic ~0": pa["catastrophic"] <= 1,
        "hooks 20/20": pa["hooks_ok"] == pa["n"],
        "no injected gibberish": pa["injected_gibberish"] == 0,
        "sim not collapsing (>=0.5)": pa["mean_sim"] >= 0.5,
        "write ratio bounded (<=0.20)": after["mean_write_ratio"] <= 0.20,
        "negatives low write (<= positives)": neg_wr <= after["mean_write_ratio"] + 0.05,
    }
    for k, v in gates.items():
        print(f"  [{'OK' if v else 'FAIL'}] {k}")
    print("\nClaim if green: Stage 2B trains the residual write path on real corpus")
    print("states while preserving generation stability + fallback safety.")
    print("NOT a quality claim (needs held-out eval).")
    return after, pa, gates


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_LM)
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs-2a", type=int, default=120)
    ap.add_argument("--epochs-2b", type=int, default=120)
    ap.add_argument("--n-perturb", type=int, default=20)
    a = ap.parse_args()
    run(model_name=a.model, device_str=a.device, epochs_2a=a.epochs_2a,
        epochs_2b=a.epochs_2b, n_perturb=a.n_perturb)
