"""V5 random-injection perturbation baseline (pre-Stage-2).

Goal: NOT to show improvement. To show the adapter is *usually non-catastrophic*
with UNTRAINED (random-init) cross-attention projections, before we spend effort
training W_q/W_o/K/V in Stage 2. If random injection rarely breaks generation,
Stage 2 starts from a stable injected-generation baseline.

For each corpus question: baseline generate vs V5-injected generate (with that
question's substrate-graph neighborhood). Measures per question:
  hook counts (expect planning=1, evidence=1)   fallback_needed
  baseline / injected length                     gibberish flags
  semantic similarity (mpnet cosine)             catastrophic = injected broke

Aggregates a non-catastrophic rate. "Catastrophic" = injected output is
degenerate (empty / repetition / mostly non-text) while baseline was fine.

    $env:KMP_DUPLICATE_LIB_OK="TRUE"; python -u -m v5.perturbation_baseline
    ... --n 30 --model Qwen/Qwen2.5-0.5B-Instruct
"""
from __future__ import annotations

import argparse
from collections import Counter

import torch

from v5.adapter import GraphAttentionInjector
from v5.cross_attention import V5AttentionAdapter
from v5.gnn_encoder import RGCNEncoder
from v5.goal_encoder import GoalEncoder
from v5.training.bridge import load_persisted_graph, _neighborhood
from v5.training.dataset import Phase15Dataset
from v5.training.providers import RealEmbedder, FrozenQwenHInitProvider  # noqa: F401 (reuse)
from v5.training.substrate import DEFAULT_OUT as SUBSTRATE_GRAPH

CORPUS = "artifacts/phase15/phase15_corpus.jsonl"
DEFAULT_LM = "Qwen/Qwen2.5-1.5B"


def is_degenerate(text: str) -> bool:
    """Heuristic gibberish/format-break detector."""
    t = (text or "").strip()
    if len(t) < 5:
        return True
    alpha = sum(c.isalpha() or c.isspace() for c in t)
    if alpha / max(1, len(t)) < 0.45:          # mostly non-text
        return True
    words = t.split()
    if len(words) >= 8 and len(set(words)) / len(words) < 0.35:   # word repetition
        return True
    if len(words) >= 12:                        # repeated 3-gram
        grams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
        if Counter(grams).most_common(1)[0][1] >= 4:
            return True
    return False


@torch.no_grad()
def _gen(model, tok, q, device, max_new_tokens):
    enc = tok(q, return_tensors="pt").to(device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def evaluate_injection(model, tok, embedder, injector, graph, samples, device, max_new_tokens=60):
    """Baseline vs injected generation over `samples`, using a (possibly trained)
    injector/adapter. Returns the aggregate dict. Shared by the random-init
    baseline and the post-Stage-2 check."""
    catastrophic = hooks_ok = 0
    sims, rows = [], []
    for i, s in enumerate(samples):
        node_ids = _neighborhood(graph, s.node_ids, hops=1, max_nodes=24)
        node_ids = node_ids + [nid for nid in s.substrate_nodes
                               if nid in graph.nodes and nid not in set(node_ids)]
        text_emb = embedder.embed_nodes({nid: (getattr(graph.nodes[nid], "text", "") or "") for nid in node_ids})
        injector.prepare_session(graph, node_ids, text_emb, s.task_frame, r_plan=3, r_evidence=4)
        base = _gen(model, tok, s.question, device, max_new_tokens)
        with injector.inject(model):
            inj = _gen(model, tok, s.question, device, max_new_tokens)
        hc = injector.get_hook_call_counts()
        hooks_ok += int(hc == {"planning": 1, "evidence": 1})
        bg, ig = is_degenerate(base), is_degenerate(inj)
        if ig and not bg:
            catastrophic += 1
        em = embedder.embed_nodes({"b": base or ".", "i": inj or "."})
        vb, vi = torch.tensor(em["b"]), torch.tensor(em["i"])
        sims.append(float(torch.dot(vb, vi) / (vb.norm() * vi.norm() + 1e-9)))
        rows.append((i, len(base), len(inj), bg, ig, sims[-1]))
        print(f"[{i:2d}] base_len={len(base):4d} inj_len={len(inj):4d} "
              f"base_gib={bg!s:5} inj_gib={ig!s:5} sim={sims[-1]:.2f}")
    N = len(samples)
    return {
        "n": N, "hooks_ok": hooks_ok,
        "baseline_gibberish": sum(r[3] for r in rows),
        "injected_gibberish": sum(r[4] for r in rows),
        "catastrophic": catastrophic,
        "non_catastrophic_rate": (N - catastrophic) / max(1, N),
        "mean_base_len": sum(r[1] for r in rows) / max(1, N),
        "mean_inj_len": sum(r[2] for r in rows) / max(1, N),
        "mean_sim": sum(sims) / max(1, N),
    }


def run(corpus_path=CORPUS, model_name=DEFAULT_LM, device_str=None, n=20, max_new_tokens=60):
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}  lm={model_name}  n={n}")

    import os
    graph_path = SUBSTRATE_GRAPH if os.path.exists(SUBSTRATE_GRAPH) else "graphs/merged_graph.json"
    graph = load_persisted_graph(graph_path)
    print(f"graph={graph_path} ({len(graph.nodes)} nodes)")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("loading frozen LM...")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    lm_dim = model.config.hidden_size

    embedder = RealEmbedder(device)
    adapter = V5AttentionAdapter(r_plan=3, r_evidence=4, lm_hidden_dim=lm_dim).to(device).eval()
    gnn = RGCNEncoder().to(device).eval()
    goal_enc = GoalEncoder().to(device).eval()
    injector = GraphAttentionInjector(adapter, gnn, goal_enc, device=device)

    ds = Phase15Dataset(corpus_path)
    samples = ds.samples[:n]

    rows = []
    catastrophic = 0
    hooks_ok = 0
    sims = []
    print(f"\nrunning {len(samples)} questions (baseline + injected each)...\n")
    for i, s in enumerate(samples):
        node_ids = _neighborhood(graph, s.node_ids, hops=1, max_nodes=24)
        node_ids = node_ids + [nid for nid in s.substrate_nodes if nid in graph.nodes and nid not in set(node_ids)]
        text_emb = embedder.embed_nodes({nid: (getattr(graph.nodes[nid], "text", "") or "") for nid in node_ids})
        injector.prepare_session(graph, node_ids, text_emb, s.task_frame, r_plan=3, r_evidence=4)

        base = _gen(model, tok, s.question, device, max_new_tokens)
        with injector.inject(model):
            inj = _gen(model, tok, s.question, device, max_new_tokens)

        hc = injector.get_hook_call_counts()
        hooks_ok += int(hc == {"planning": 1, "evidence": 1})
        bg, ig = is_degenerate(base), is_degenerate(inj)
        if ig and not bg:
            catastrophic += 1
        # semantic drift via mpnet cosine
        em = embedder.embed_nodes({"b": base or ".", "i": inj or "."})
        vb, vi = torch.tensor(em["b"]), torch.tensor(em["i"])
        sim = float(torch.dot(vb, vi) / (vb.norm() * vi.norm() + 1e-9))
        sims.append(sim)
        rows.append((i, len(base), len(inj), bg, ig, sim, injector.get_fallback_needed()))
        print(f"[{i:2d}] base_len={len(base):4d} inj_len={len(inj):4d} "
              f"base_gib={bg!s:5} inj_gib={ig!s:5} sim={sim:.2f} hooks={hc}")

    N = len(samples)
    print("\n" + "=" * 60)
    print("AGGREGATE:")
    print(f"  questions                : {N}")
    print(f"  hook control ok (1/1)    : {hooks_ok}/{N}")
    print(f"  baseline gibberish       : {sum(r[3] for r in rows)}/{N}")
    print(f"  injected gibberish       : {sum(r[4] for r in rows)}/{N}")
    print(f"  CATASTROPHIC (inj broke) : {catastrophic}/{N}  ({catastrophic/N:.0%})")
    print(f"  non-catastrophic rate    : {(N-catastrophic)/N:.0%}")
    print(f"  mean baseline length     : {sum(r[1] for r in rows)/N:.0f} chars")
    print(f"  mean injected length     : {sum(r[2] for r in rows)/N:.0f} chars")
    print(f"  mean semantic sim        : {sum(sims)/N:.2f}  (1=identical, lower=more drift)")
    print("\nGoal was non-catastrophic, NOT improvement. A high non-catastrophic")
    print("rate means Stage 2 starts from a stable injected-generation baseline.")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--model", default=DEFAULT_LM)
    ap.add_argument("--device", default=None)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=60)
    a = ap.parse_args()
    run(a.corpus, a.model, a.device, a.n, a.max_new_tokens)
