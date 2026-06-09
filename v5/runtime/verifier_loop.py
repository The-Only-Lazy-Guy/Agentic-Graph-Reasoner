"""Tier-2 verifier loop — TEST-feedback retry (the inspection-driven lever).

Manual inspection showed the loop's failures are NOT wrong-location or no-fix: they're
close-but-not-test-passing (over-edit, alternative approach, rare mangling). Apply-feedback
can't see that — the patch *applies*, it's just not *correct*. Only the TESTS can. So:

  generate SR (source + injection + feedback) -> apply -> RUN THE TESTS
    -> pass: resolved, done
    -> fail: feed the failing-test output back -> regenerate (bounded)

This is "weak generator + strong checker", the design's core answer to weak synthesis. Needs
the Docker verifier in-loop, so it must run where the model (GPU) AND Docker coexist (a laptop
with WSL+Docker at 4-bit, or a privileged-Docker GPU box). Slow (a test run per attempt; the
image caches after the first), so use a SMALL --n-eval.

  V5_LM_TRUST_REMOTE_CODE=1 python -m v5.runtime.verifier_loop \
    --adapter-ckpt artifacts/stage_cache/adapter_code_sr.pt --n-eval 10 --max-test-retries 3 \
    --src-bodies 4 --src-lines 55
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import torch

from v5.adapter import GraphAttentionInjector
from v5.cross_attention import V5AttentionAdapter
from v5.gnn_encoder import RGCNEncoder
from v5.goal_encoder import GoalEncoder
from v5.training.providers import RealEmbedder, FrozenQwenHInitProvider
from v5.training.stage4_generate import _gen, _stub_graph
from v5.graph_grower.swe_load import load_instances, checkout_repo
from v5.graph_grower.swe_probe import load_traces
from v5.runtime.search_replace import SR_SYS, parse_sr, apply_sr
from v5.runtime.sr_withcode import load_symbol_meta, read_body, _file_text
from v5.graph_grower.swe_verify import write_predictions, run_swebench
import subprocess


def _user(issue, src, apply_fb="", test_fb=""):
    s = f"ISSUE:\n{issue[:1400]}\n\nRELEVANT SOURCE (the bug is in here):\n{src}\n\n"
    if test_fb:
        s += f"YOUR LAST PATCH APPLIED BUT FAILED THE TESTS:\n{test_fb[:900]}\nFix the logic.\n\n"
    if apply_fb:
        s += f"PREVIOUS ATTEMPT DID NOT MATCH THE FILE:\n{apply_fb[:300]}\n\n"
    return (s + "Output ONLY search/replace blocks: SEARCH must copy the source EXACTLY; "
            "REPLACE must DIFFER from SEARCH. Keep it minimal.")


def _gen_blocks(model, tok, injector, issue, src, dest, apply_fb, test_fb, max_new):
    """One generation -> parsed blocks + applyable flag + an apply-mismatch message."""
    msgs = [{"role": "system", "content": SR_SYS},
            {"role": "user", "content": _user(issue, src, apply_fb, test_fb)}]
    gen = _gen(model, tok, msgs, next(model.parameters()).device, injector, True, max_new)
    blocks = parse_sr(gen)
    um = [b for b in blocks if (b.get("search") or "").strip()
          and (b["search"]).strip() not in _file_text(dest, b.get("file"))]
    return blocks, (bool(blocks) and not um), (um[0]["search"][:200] if um else "no valid block")


def run_one_test(instance_id, patch, dataset, run_id, out_dir):
    """Apply the patch in Docker + run the instance's tests -> (resolved, failing-output tail)."""
    pred_path = str(Path(out_dir) / "loop_pred.jsonl")   # write_predictions returns the COUNT, not the path
    write_predictions({instance_id: patch}, pred_path, "v5_loop")
    res = run_swebench(pred_path, dataset, run_id, instance_ids=[instance_id], max_workers=1)
    resolved = bool(res.get(instance_id, False))
    err = ""
    logs = glob.glob(f"logs/run_evaluation/{run_id}/*/{instance_id}/test_output.txt")
    if logs:
        txt = Path(max(logs, key=lambda p: Path(p).stat().st_mtime)).read_text(errors="ignore")
        # keep the FAILED/Error lines (the useful signal), tail
        keep = [ln for ln in txt.splitlines() if any(k in ln for k in
                ("FAILED", "Error", "assert", "Exception", "PASSED"))]
        err = "\n".join(keep[-15:]) or txt[-900:]
    return resolved, err


def run(model_name, traces_p, nodes_p, adapter_ckpt, dataset, split, n_eval, max_new,
        repo_root, max_test_retries, src_bodies, src_lines, out_dir, device_str=None):
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}  max_test_retries={max_test_retries}", flush=True)
    provider = FrozenQwenHInitProvider(model_name, device=device)
    model, tok = provider.model, provider.tok
    embedder = RealEmbedder(device)
    gnn = RGCNEncoder().to(device).eval()
    goal_enc = GoalEncoder().to(device).eval()
    adapter = V5AttentionAdapter(r_plan=3, r_evidence=4, lm_hidden_dim=provider.hidden_size).to(device)
    adapter.load_state_dict(torch.load(adapter_ckpt, map_location=device)); adapter.eval()
    injector = GraphAttentionInjector(adapter, gnn, goal_enc, device=device)

    traces = load_traces(traces_p)
    meta = load_symbol_meta(nodes_p)
    insts = {t["instance_id"]: t for t in load_instances(dataset, split, limit=0)}
    ids = [i for i in traces if i in insts][:n_eval]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    print(f"tasks={len(ids)} | symbol meta={len(meta)}", flush=True)
    tf = {"task_family": "code_fix", "required_slots": []}
    r1 = rk = scored = 0

    for k, iid in enumerate(ids):
        t = traces[iid]; inst = insts[iid]
        support = [s for s in t["support_ids"] if s in meta]
        if not support:
            continue
        dest = Path(repo_root) / inst["repo"].replace("/", "__")
        ok, _ = checkout_repo(inst["repo"], inst["base_commit"], dest, timeout=1800)
        if not ok:
            print(f"  [{k+1}] {iid} checkout FAILED"); continue
        src = "\n\n".join(f"# {meta[s]['file']}\n{read_body(str(dest), meta[s]['file'], meta[s]['lineno'], src_lines)}"
                          for s in support[:src_bodies] if read_body(str(dest), meta[s]['file'], meta[s]['lineno'], src_lines))
        node_ids = support[:24]
        injector.prepare_session(_stub_graph(node_ids, {s: meta[s]["text"] for s in node_ids},
                                             {s: "fact" for s in node_ids}),
                                 node_ids, embedder.embed_nodes({s: meta[s]["text"] for s in node_ids}),
                                 tf, r_plan=3, r_evidence=4)
        scored += 1
        apply_fb = test_fb = ""
        passed = False
        for attempt in range(max_test_retries):
            blocks, applyable, mismatch = _gen_blocks(model, tok, injector, t["issue"], src,
                                                      str(dest), apply_fb, test_fb, max_new)
            if not applyable:
                apply_fb = mismatch; continue
            _, patch = apply_sr(str(dest), blocks)
            subprocess.run(["git", "-C", str(dest), "checkout", "--", "."], capture_output=True)
            if not patch.strip():
                apply_fb = "empty patch"; continue
            resolved, err = run_one_test(iid, patch, dataset, f"loop_{iid[:20]}", out_dir)
            print(f"  [{k+1}/{len(ids)}] {iid:24} attempt{attempt+1} applyable=Y resolved={resolved}", flush=True)
            if resolved:
                passed = True
                if attempt == 0:
                    r1 += 1
                break
            test_fb, apply_fb = err, ""        # next try: fix the LOGIC, given the test failure
        rk += 1 if passed else 0

    print(f"\n=== TIER-2 TEST-FEEDBACK LOOP ===")
    print(f"  resolve@1            {r1}/{scored}")
    print(f"  resolve@{max_test_retries} (test-retry)  {rk}/{scored}")
    print("\nresolve@k > resolve@1 = test-feedback recovers close-but-wrong patches (the inspection"
          "\nsymptom). weak generator + strong checker -> more resolves, no retrain.")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Tier-2 test-feedback verifier loop.")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--traces", nargs="+", default=["data/swe/grounded_traces.jsonl"])
    ap.add_argument("--nodes", nargs="+", default=["artifacts/graph_growth/swe_code_candidates.jsonl"])
    ap.add_argument("--adapter-ckpt", default="artifacts/stage_cache/adapter_code_sr.pt")
    ap.add_argument("--dataset", default="lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-eval", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=500)
    ap.add_argument("--max-test-retries", type=int, default=3)
    ap.add_argument("--repo-root", default="data/swe_repos")
    ap.add_argument("--src-bodies", type=int, default=4)
    ap.add_argument("--src-lines", type=int, default=55)
    ap.add_argument("--out-dir", default="artifacts/graph_growth/swe_verify")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    run(a.model, a.traces, a.nodes, a.adapter_ckpt, a.dataset, a.split, a.n_eval, a.max_new,
        a.repo_root, a.max_test_retries, a.src_bodies, a.src_lines, a.out_dir, device_str=a.device)


if __name__ == "__main__":
    raise SystemExit(main())
