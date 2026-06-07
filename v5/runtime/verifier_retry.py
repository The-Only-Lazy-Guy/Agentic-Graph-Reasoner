"""Verifier-retry loop (V4's loop, for code) — TIER 1: apply-feedback (no verifier).

The single-shot 4B writes SR edits whose SEARCH often doesn't match the file (search_in_file
~0.2-0.33) -> unapplyable. The loop fixes that by ITERATING: generate -> check the SEARCH against
the real file -> if it doesn't match, hand back the exact code and retry. The model's #1 failure
(inexact SEARCH) is exactly what a tiny feedback step repairs. No tests, no Docker -- pure
file-matching. Measures applyable@1 vs applyable@k (does iteration lift the applyable rate).

TIER 2 (test-pass) is pluggable: once a SEARCH-matching patch exists, apply_sr + run the tests
(swe_verify / sb-cli) and feed the failure back. Stubbed here behind --verify.

This is V4's reasoning_loop shape (retrieve -> act -> verify -> retry), code action-set.

  V5_LM_TRUST_REMOTE_CODE=1 python -m v5.runtime.verifier_retry \
    --adapter-ckpt artifacts/stage_cache/adapter_code_s3.pt --n-eval 15 --max-retries 3 \
    --dump artifacts/retry_dump.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from v5.adapter import GraphAttentionInjector
from v5.cross_attention import V5AttentionAdapter
from v5.gnn_encoder import RGCNEncoder
from v5.goal_encoder import GoalEncoder
from v5.training.providers import RealEmbedder, FrozenQwenHInitProvider
from v5.training.stage4_generate import _gen, _stub_graph
from v5.graph_grower.swe_load import load_instances, patch_files, checkout_repo
from v5.graph_grower.swe_probe import load_traces, _symbol_name
from v5.runtime.search_replace import SR_SYS, parse_sr, apply_sr
from v5.runtime.sr_withcode import load_symbol_meta, read_body, _file_text
from v5.graph_grower.swe_verify import write_predictions

import subprocess


def _user(issue, src_ctx, feedback=""):
    s = f"ISSUE:\n{issue[:1400]}\n\n"
    if src_ctx:                              # empty under --no-graph (cold baseline)
        s += f"RELEVANT SOURCE (the bug is in here):\n{src_ctx}\n\n"
    if feedback:
        s += f"PREVIOUS ATTEMPT FAILED:\n{feedback}\n\n"
    return (s + "Find the exact line(s) causing the bug and fix them. Output ONLY search/replace "
            "blocks: SEARCH must copy the source EXACTLY (character-for-character); REPLACE must "
            "DIFFER from SEARCH. Keep it minimal.")


def _unmatched(blocks, dest):
    out = []
    for b in blocks:
        s = (b.get("search") or "").strip()
        if s and s not in _file_text(dest, b.get("file")):
            out.append(b)
    return out


def solve(model, tok, injector, issue, src_ctx, dest, max_retries, max_new, inject_on=True):
    """Iterate generate -> check SEARCH matches file -> feedback -> retry. Returns
    (applyable, attempts_used, blocks, history). inject_on=False = cold baseline (--no-graph)."""
    feedback = ""
    history = []
    blocks = []
    for attempt in range(max_retries):
        gen = _gen(model, tok, [{"role": "system", "content": SR_SYS},
                                {"role": "user", "content": _user(issue, src_ctx, feedback)}],
                   model.device if hasattr(model, "device") else next(model.parameters()).device,
                   injector, inject_on, max_new)
        blocks = parse_sr(gen)
        um = _unmatched(blocks, dest)
        applyable = bool(blocks) and not um
        history.append({"attempt": attempt + 1, "n_blocks": len(blocks), "applyable": applyable})
        if applyable:
            return True, attempt + 1, blocks, history
        # apply-feedback: tell it exactly what failed
        if not blocks:
            feedback = "You produced NO valid search/replace block. Output ONLY blocks in the exact format."
        else:
            b = um[0]
            feedback = (f"Your SEARCH for file '{b['file']}' was NOT found in the source. This text "
                        f"does not exist there:\n{(b.get('search') or '')[:200]}\n"
                        "Copy the lines from the SOURCE above EXACTLY (same indentation/spacing).")
    return False, max_retries, blocks, history


def run(model_name, traces_p, nodes_p, adapter_ckpt, dataset, split, n_eval, max_new,
        repo_root, max_retries, dump="", emit_predictions="", no_graph=False, device_str=None):
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}  max_retries={max_retries}", flush=True)
    provider = FrozenQwenHInitProvider(model_name, device=device)
    model, tok = provider.model, provider.tok
    lm_dim = provider.hidden_size
    embedder = RealEmbedder(device)
    gnn = RGCNEncoder().to(device).eval()
    goal_enc = GoalEncoder().to(device).eval()
    adapter = V5AttentionAdapter(r_plan=3, r_evidence=4, lm_hidden_dim=lm_dim).to(device)
    adapter.load_state_dict(torch.load(adapter_ckpt, map_location=device))
    adapter.eval()
    injector = GraphAttentionInjector(adapter, gnn, goal_enc, device=device)

    traces = load_traces(traces_p)
    meta = load_symbol_meta(nodes_p)
    insts = {t["instance_id"]: t for t in load_instances(dataset, split, limit=0)}
    ids = [i for i in traces if i in insts][:n_eval]
    print(f"eval instances={len(ids)} | symbol meta={len(meta)}", flush=True)

    app1 = appk = scored = 0
    records = []
    preds = {}                              # instance_id -> git diff (swebench prediction)
    tf = {"task_family": "code_fix", "required_slots": []}
    for k, iid in enumerate(ids):
        t = traces[iid]; inst = insts[iid]
        support = [s for s in t["support_ids"] if s in meta]
        if not support:
            continue
        dest = Path(repo_root) / f"{inst['repo'].replace('/', '__')}__{inst['base_commit'][:8]}"
        ok, msg = checkout_repo(inst["repo"], inst["base_commit"], dest)
        if not ok:
            print(f"  [{k+1}] {iid} checkout FAILED"); continue
        src_parts = []
        if not no_graph:                     # cold baseline (--no-graph): no source, no injection
            for s in support[:6]:
                body = read_body(str(dest), meta[s]["file"], meta[s]["lineno"], max_lines=70)
                if body:
                    src_parts.append(f"# {meta[s]['file']}\n{body}")
        src_ctx = "\n\n".join(src_parts)
        node_ids = support[:24]
        text_emb = embedder.embed_nodes({s: meta[s]["text"] for s in node_ids})
        injector.prepare_session(_stub_graph(node_ids, {s: meta[s]["text"] for s in node_ids},
                                             {s: "fact" for s in node_ids}),
                                 node_ids, text_emb, tf, r_plan=3, r_evidence=4)

        applyable, attempts, blocks, hist = solve(model, tok, injector, t["issue"], src_ctx,
                                                  str(dest), max_retries, max_new, inject_on=not no_graph)
        scored += 1
        app1 += 1 if (hist and hist[0]["applyable"]) else 0
        appk += 1 if applyable else 0
        # turn the applyable SR blocks into a real git diff (the swebench prediction), then
        # restore the checkout so it's clean for the next instance.
        model_patch = ""
        if applyable:
            _, model_patch = apply_sr(str(dest), blocks)
            subprocess.run(["git", "-C", str(dest), "checkout", "--", "."], capture_output=True)
        preds[iid] = model_patch
        records.append({"id": iid, "applyable": applyable, "attempts": attempts, "history": hist,
                        "gold_patch": inst.get("patch", ""), "final_blocks": blocks})
        print(f"  [{k+1}/{len(ids)}] {iid:24} applyable={applyable} in {attempts} "
              f"(attempt1={'Y' if hist and hist[0]['applyable'] else 'N'})", flush=True)

    if dump:
        Path(dump).parent.mkdir(parents=True, exist_ok=True)
        with open(dump, "w", encoding="utf-8") as w:
            for r in records:
                w.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\ndumped {len(records)} -> {dump}", flush=True)

    if emit_predictions:
        n = write_predictions({i: p for i, p in preds.items() if p.strip()}, emit_predictions, "v5_retry")
        print(f"emitted {n} applyable predictions -> {emit_predictions}  (TIER-2: verify with\n"
              f"  python -m v5.graph_grower.swe_verify --predictions {emit_predictions} --dataset {dataset})", flush=True)

    print(f"\n=== RETRY LOOP (apply-feedback, no verifier) ===")
    print(f"  applyable@1   {app1}/{scored} ({100*app1/max(1,scored):.0f}%)")
    print(f"  applyable@{max_retries}   {appk}/{scored} ({100*appk/max(1,scored):.0f}%)")
    print("\napplyable@k > applyable@1 = iteration repairs unmatched SEARCH -> the loop lifts the"
          "\napplyable rate with zero infra. Tier-2 (run tests) plugs in when a verifier is available.")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Verifier-retry loop tier-1 (apply-feedback).")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--traces", nargs="+", default=["data/swe/grounded_traces.jsonl"])
    ap.add_argument("--nodes", nargs="+", default=["artifacts/graph_growth/swe_code_candidates.jsonl"])
    ap.add_argument("--adapter-ckpt", default="artifacts/stage_cache/adapter_code_s3.pt")
    ap.add_argument("--dataset", default="lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-eval", type=int, default=15)
    ap.add_argument("--max-new", type=int, default=500)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--repo-root", default="data/swe_repos")
    ap.add_argument("--dump", default="")
    ap.add_argument("--emit-predictions", default="",
                    help="write the loop's applyable patches as a swebench predictions jsonl -> tier-2 verify")
    ap.add_argument("--no-graph", action="store_true",
                    help="COLD baseline: no source-read, no injection -> ablation for the grounding lift")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    run(a.model, a.traces, a.nodes, a.adapter_ckpt, a.dataset, a.split, a.n_eval, a.max_new,
        a.repo_root, a.max_retries, dump=a.dump, emit_predictions=a.emit_predictions,
        no_graph=a.no_graph, device_str=a.device)


if __name__ == "__main__":
    raise SystemExit(main())
