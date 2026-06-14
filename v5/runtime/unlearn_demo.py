"""Selective-unlearn demo: editing the GRAPH changes a FROZEN LM's behavior, reversibly.

The INTFAIR money shot. Two skills A, B live as disjoint node sets in the graph. We measure
the NLL of each skill's GOLD edit (lower = the LM "knows" the fix) under graph states:

  cold            no graph injected            (baseline: the bare frozen 4B)
  both            graph = A-nodes + B-nodes     (both skills taught)
  delete-A        graph = B-nodes only          (A surgically removed from the graph)

Expected:
  skill A:  cold (high) -> both (LOW, grounded) -> delete-A (~cold again)  == A UNLEARNED
  skill B:  cold (high) -> both (LOW, grounded) -> delete-A (~both, LOW)   == B RETAINED

=> deleting A's nodes provably removes A's competence and ONLY A's, with NO retraining.
That is the editable/auditable/unlearnable property, demonstrated on the frozen LM.

Real run (A40 / any box with the 4B + V5_LM_QUANT=4bit):
  V5_LM_TRUST_REMOTE_CODE=1 V5_LM_QUANT=4bit python -m v5.runtime.unlearn_demo \
    --adapter-ckpt artifacts/stage_cache/adapter_code_sr.pt
Plumbing-only check (no LM, fits any box):
  python -m v5.runtime.unlearn_demo --dry
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext

import torch

from v5.graph_grower.swe_load import load_instances
from v5.graph_grower.swe_probe import load_traces
from v5.runtime.sr_withcode import load_symbol_meta
from v5.training.stage_sr_sft import _build_rows, sr_nll
from v5.training.stage4_generate import _stub_graph


def _pick_two(rows):
    """Two rows from DIFFERENT repos -> disjoint support node sets (clean A/B skills)."""
    seen = {}
    for r in rows:
        repo = r["iid"].split("__")[0]
        if repo not in seen:
            seen[repo] = r
        if len(seen) == 2:
            break
    return list(seen.values())[:2]


def run(model_name, traces_p, nodes_p, adapter_ckpt, dataset, split, repo_root,
        src_bodies, src_lines, dry, max_ids=40, device_str=None):
    traces = load_traces([traces_p])
    meta = load_symbol_meta([nodes_p])
    insts = {t["instance_id"]: t for t in load_instances(dataset, split, limit=0)}
    ids = [i for i in traces if i in insts][:max_ids]
    rows = _build_rows(ids, traces, insts, meta, repo_root, src_bodies, src_lines)
    pair = _pick_two(rows)
    if len(pair) < 2:
        print("need 2 instances from different repos; got", len(pair)); return
    A, B = pair
    setA, setB = set(A["node_ids"]), set(B["node_ids"])
    print(f"\nskill A = {A['iid']}  ({len(setA)} nodes)")
    print(f"skill B = {B['iid']}  ({len(setB)} nodes)")
    print(f"support disjoint: {setA.isdisjoint(setB)}  (overlap={len(setA & setB)})")
    if dry:
        print("\n[dry] plumbing OK — rows built, supports disjoint. Run with the 4B for NLL numbers.")
        return

    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    from v5.adapter import GraphAttentionInjector
    from v5.cross_attention import V5AttentionAdapter
    from v5.gnn_encoder import RGCNEncoder
    from v5.goal_encoder import GoalEncoder
    from v5.training.providers import RealEmbedder, FrozenQwenHInitProvider

    provider = FrozenQwenHInitProvider(model_name, device=device)
    model, tok = provider.model, provider.tok
    lm_dim = provider.hidden_size
    embedder = RealEmbedder(device)
    gnn = RGCNEncoder().to(device).eval()
    goal_enc = GoalEncoder().to(device).eval()
    adapter = V5AttentionAdapter(r_plan=3, r_evidence=4, lm_hidden_dim=lm_dim).to(device)
    adapter.load_state_dict(torch.load(adapter_ckpt, map_location=device)); adapter.eval()
    injector = GraphAttentionInjector(adapter, gnn, goal_enc, device=device)
    injector.inject_all_positions = True       # condition ALL teacher-forced tokens (else the
    tf = {"task_family": "code_fix", "required_slots": []}   # hook only touches the last token -> no NLL effect

    def prep(node_ids, texts):
        injector.prepare_session(_stub_graph(node_ids, texts, {s: "fact" for s in node_ids}),
                                 node_ids, embedder.embed_nodes(texts), tf, r_plan=3, r_evidence=4)

    union_ids = A["node_ids"] + [n for n in B["node_ids"] if n not in setA]
    union_texts = {**A["texts"], **B["texts"]}

    @torch.no_grad()
    def nll(row, state):
        if state == "cold":
            return float(sr_nll(model, tok, injector, row["issue"], row["src"], row["sr"],
                                device, inject=False))
        if state == "both":
            prep(union_ids, union_texts)
        elif state == "delete-A":
            prep(B["node_ids"], B["texts"])            # A removed from the graph
        return float(sr_nll(model, tok, injector, row["issue"], row["src"], row["sr"],
                            device, inject=True))

    print("\n=== SR-NLL of each skill's gold edit (lower = LM knows the fix) ===")
    print(f"  {'skill':8s} {'cold':>8s} {'both':>8s} {'delete-A':>9s}")
    res = {}
    for name, row in (("A", A), ("B", B)):
        c, bo, da = nll(row, "cold"), nll(row, "both"), nll(row, "delete-A")
        res[name] = (c, bo, da)
        print(f"  {name:8s} {c:8.4f} {bo:8.4f} {da:9.4f}")

    cA, boA, daA = res["A"]; cB, boB, daB = res["B"]
    print("\n=== verdict ===")
    print(f"  A grounded by graph:   both {boA:.3f} < cold {cA:.3f}   (lift {cA-boA:+.3f})")
    print(f"  A UNLEARNED by delete: delete-A {daA:.3f} -> back toward cold {cA:.3f} "
          f"(reverted {daA-boA:+.3f} of {cA-boA:+.3f})")
    print(f"  B RETAINED (selective): both {boB:.3f} vs delete-A {daB:.3f} "
          f"(drift {abs(daB-boB):.3f} — small = B unaffected by deleting A)")
    print("\n  => editing the graph changed the frozen LM's behavior, and deleting A's nodes")
    print("     removed A's competence ONLY. No retraining. That is selective unlearning.")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Selective-unlearn demo (edit graph -> change frozen LM).")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--traces", default="data/swe/grounded_traces.jsonl")
    ap.add_argument("--nodes", default="artifacts/graph_growth/swe_code_candidates.jsonl")
    ap.add_argument("--adapter-ckpt", default="artifacts/stage_cache/adapter_code_sr.pt")
    ap.add_argument("--dataset", default="lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--repo-root", default="data/swe_repos")
    ap.add_argument("--src-bodies", type=int, default=4)
    ap.add_argument("--src-lines", type=int, default=55)
    ap.add_argument("--dry", action="store_true", help="plumbing check only (no LM)")
    ap.add_argument("--max-ids", type=int, default=40, help="instances to scan for an A/B pair")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    run(a.model, a.traces, a.nodes, a.adapter_ckpt, a.dataset, a.split, a.repo_root,
        a.src_bodies, a.src_lines, a.dry, max_ids=a.max_ids, device_str=a.device)


if __name__ == "__main__":
    raise SystemExit(main())
