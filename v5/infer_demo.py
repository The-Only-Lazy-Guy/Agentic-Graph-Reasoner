"""V5 generation demo: baseline Qwen vs V5-injected, same question.

Actually runs model.generate() with the V5 adapter live (hooks at L8/L20) and
prints the generated text both with and without graph injection, side by side.

HONEST EXPECTATION: the cross-attention projections (W_q/W_o) are at RANDOM init
— Stage 1 trains only the aux heads and freezes the projections. So the loop
writes a random-projected vector into the residual stream; injection may PERTURB
or DEGRADE generation. That is the expected pre-Stage-2 result. The point of this
demo is to confirm generation runs end-to-end with hooks active and to SEE the
effect, not to claim improved answers (that needs Stage 2 projection training +
LoRA).

    $env:KMP_DUPLICATE_LIB_OK="TRUE"; python -u -m v5.infer_demo
"""
from __future__ import annotations

import argparse
import torch

from v5.adapter import GraphAttentionInjector, PLANNING_LAYER, EVIDENCE_LAYER
from v5.cross_attention import V5AttentionAdapter
from v5.gnn_encoder import RGCNEncoder
from v5.goal_encoder import GoalEncoder
from v5.realstack_test import build_test_graph, RealEmbedder

DEFAULT_LM = "Qwen/Qwen2.5-1.5B"


@torch.no_grad()
def _generate(model, tok, question, device, max_new_tokens=80):
    inputs = tok(question, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip()


def run(model_name: str = DEFAULT_LM, device_str: str = None, max_new_tokens: int = 80):
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}  lm={model_name}")

    # graph + injected substrate (binary-search applicability scenario)
    g, node_ids, question, task_frame = build_test_graph("graphs/algo3_binary_search.json")
    print(f"\nquestion: {question}")
    print(f"subgraph: {len(node_ids)} nodes")

    print("\nloading frozen LM...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    lm_dim = model.config.hidden_size

    print("embedding nodes (mpnet)...")
    embedder = RealEmbedder(device)
    text_emb = embedder.embed_nodes({nid: g.nodes[nid].text for nid in node_ids})

    # ── baseline generation (no adapter) ─────────────────────────────────
    print("\n" + "=" * 70)
    print("BASELINE (no V5 adapter):")
    base = _generate(model, tok, question, device, max_new_tokens)
    print(f"  {base!r}")

    # ── V5-injected generation ───────────────────────────────────────────
    adapter = V5AttentionAdapter(r_plan=3, r_evidence=4, lm_hidden_dim=lm_dim).to(device).eval()
    gnn = RGCNEncoder().to(device).eval()
    goal_enc = GoalEncoder().to(device).eval()
    injector = GraphAttentionInjector(adapter, gnn, goal_enc, device=device)
    injector.prepare_session(g, node_ids, text_emb, task_frame, r_plan=3, r_evidence=4)

    print("\n" + "=" * 70)
    print("V5-INJECTED (untrained projections — expect perturbation):")
    with injector.inject(model):
        inj = _generate(model, tok, question, device, max_new_tokens)
    print(f"  {inj!r}")
    print(f"\nhook call counts: {injector.get_hook_call_counts()}")
    print(f"fallback_needed: {injector.get_fallback_needed()}")

    print("\n" + "=" * 70)
    print("DIFF:", "outputs differ" if base != inj else "IDENTICAL (hooks had no effect!)")
    print("\nInterpretation: generation runs end-to-end with V5 hooks active. The")
    print("injected output reflects RANDOM-init cross-attention projections, so any")
    print("change is perturbation, not learned improvement — that needs Stage 2")
    print("(train projections) + LoRA. This demo proves the generation path works.")
    return base, inj


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_LM)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    a = ap.parse_args()
    run(a.model, a.device, a.max_new_tokens)
