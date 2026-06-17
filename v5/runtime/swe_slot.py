"""#9 (synthesis-focused, ENGINE-WIRED) — slot-graph DIAGNOSE->FIX vs one-shot, through SlotGraph.solve.

Localization held FIXED (gold support symbols) to isolate SYNTHESIS from the ~0.30 localization wall.
The SLOT path now runs the REAL engine (slot_coder.SlotGraph): DIAGNOSE (revise='rederive') -> FIX.
An unapplyable FIX -> empty value -> INSUFFICIENT -> dependency-directed BACKTRACK to DIAGNOSE ->
re-diagnose deeper -> re-FIX, to a fixpoint (or max_steps). Compared to ONE-SHOT (single SR emit).

`slot_solve()` is shared by the real 4B run AND a no-model `--selftest` that PROVES the engine wiring
(DIAGNOSE->FIX->INSUFFICIENT->backtrack->re-diagnose->applyable->fixpoint) without a GPU. This is the
fix for the earlier mistake where swe_slot bypassed the engine entirely (see memory verify-wiring-not-proxy).

  4B (A40): V5_LM_TRUST_REMOTE_CODE=1 python -m v5.runtime.swe_slot --n-eval 24
  wiring  : python -m v5.runtime.swe_slot --selftest      # no model, proves the slot-graph is invoked
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path

from v5.runtime.slot_coder import SlotGraph, SlotSpec, Pool


def _unmatched(blocks, dest):
    from v5.runtime.sr_withcode import _file_text
    return [b for b in blocks if (b.get("search") or "").strip()
            and (b.get("search") or "").strip() not in _file_text(dest, b.get("file"))]


def _patch(blocks, dest):                     # applyable -> git diff (the swebench prediction), then restore
    from v5.runtime.search_replace import apply_sr
    if not (bool(blocks) and not _unmatched(blocks, dest)):
        return ""
    _, p = apply_sr(dest, blocks)
    subprocess.run(["git", "-C", dest, "checkout", "--", "."], capture_output=True)
    return p


def fix_user(issue, src, diagnosis=""):
    s = f"ISSUE:\n{issue[:1400]}\n\nRELEVANT SOURCE (the bug is in here):\n{src}\n\n"
    if diagnosis:
        s += f"ROOT-CAUSE DIAGNOSIS (use it):\n{diagnosis}\n\n"
    return (s + "Fix the exact line(s) causing the bug. Output ONLY search/replace blocks: SEARCH "
            "must copy the source EXACTLY (character-for-character); REPLACE must DIFFER. Keep it minimal.")


def diag_user(issue, src, attempt):
    nudge = "" if attempt == 0 else (
        " NOTE: a previous diagnosis led to an UNAPPLYABLE or no-op fix. Be more specific — name the "
        "EXACT function and the EXACT line/token to change, copied verbatim from the source above.")
    return (f"ISSUE:\n{issue[:1400]}\n\nSOURCE:\n{src}\n\nIn 2-3 sentences, state the ROOT CAUSE of the "
            f"bug and exactly what must change (name the function/lines). Be specific.{nudge}")


def slot_solve(issue, src, diagnose_fn, fix_fn, max_steps=8, log=None):
    """The SHARED slot-graph: DIAGNOSE (rederive on failure) -> FIX (applyable = sufficient).
      diagnose_fn(issue, src, attempt) -> diagnosis text
      fix_fn(issue, src, diagnosis)    -> APPLYABLE patch text, or "" if unapplyable/no-op
    Returns (patch, trace, fixpoint, steps). Same engine for the 4B run and the selftest."""
    attempts = {"DIAGNOSE": 0}
    trace = {"diagnoses": [], "fix_attempts": []}

    def retr(q, kind):
        return [{"id": "src", "text": src}]                    # localization fixed -> evidence = the source

    def filler(slot, ev, pool):
        if slot.name == "DIAGNOSE":
            n = attempts["DIAGNOSE"]; attempts["DIAGNOSE"] = n + 1
            d = diagnose_fn(issue, src, n)
            trace["diagnoses"].append(d)
            return d
        diag = pool.get("DIAGNOSE")
        patch = fix_fn(issue, src, diag)                       # "" if unapplyable -> INSUFFICIENT -> backtrack
        trace["fix_attempts"].append({"applyable": bool(patch), "diag_used": diag[:160]})
        return patch

    specs = [
        SlotSpec("DIAGNOSE", [], "src", "ASSERT", query=lambda p: "root cause of the bug", revise="rederive"),
        SlotSpec("FIX", ["DIAGNOSE"], "src", "TRANSFORM", query=lambda p: "minimal applyable fix"),
    ]
    sg = SlotGraph(specs)
    pool = Pool(specs, context={"issue": issue, "src": src})
    ok, steps = sg.solve(pool, retr, filler, max_steps=max_steps, log=log)
    return pool.slots["FIX"].value, trace, ok, steps


# ── no-model wiring proof: the engine MUST drive DIAGNOSE->FIX->backtrack->re-diagnose->applyable ──
def _selftest():
    print("swe_slot --selftest: proving the SLOT path runs SlotGraph.solve (no model).\n")
    # stub model: shallow first diagnosis -> unapplyable fix; after backtrack, a deeper diagnosis -> applyable.
    def diagnose_fn(issue, src, attempt):
        return "vague: something is wrong" if attempt == 0 else "PRECISE: change line 42 token `<` to `<=`"
    def fix_fn(issue, src, diagnosis):
        return "diff --git a/x b/x\n+fixed\n" if diagnosis.startswith("PRECISE") else ""   # applyable only if precise
    log = []
    patch, trace, ok, steps = slot_solve("issue", "source", diagnose_fn, fix_fn, log=log)
    for row in log:
        print("   ", row)
    print(f"\n   diagnoses          : {trace['diagnoses']}")
    print(f"   fix attempts        : {[a['applyable'] for a in trace['fix_attempts']]}")
    print(f"   fixpoint={ok} steps={steps}  final patch applyable={bool(patch)}")
    backtracked = any(r[0] == "BACKTRACK" for r in log)
    rediagnosed = len(trace["diagnoses"]) >= 2
    ok_wired = bool(patch) and ok and backtracked and rediagnosed
    print(f"\n   WIRING PROOF: backtrack-fired={backtracked}  re-diagnosed={rediagnosed}  "
          f"recovered-applyable={bool(patch)}  -> {'PASS' if ok_wired else 'FAIL'}")
    print("   (attempt-1 fix unapplyable -> INSUFFICIENT -> backtrack to DIAGNOSE -> deeper diagnosis ->")
    print("    attempt-2 fix applyable -> fixpoint. The engine, not a hand-coded 2-call pipeline.)")
    return ok_wired


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="no-model proof the SLOT path uses the engine")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--traces", default="data/swe/grounded_traces.jsonl")
    ap.add_argument("--nodes", default="artifacts/graph_growth/swe_code_candidates.jsonl")
    ap.add_argument("--dataset", default="lite")
    ap.add_argument("--repo-root", default="data/swe_repos")
    ap.add_argument("--n-eval", type=int, default=10)
    ap.add_argument("--src-bodies", type=int, default=4)
    ap.add_argument("--src-lines", type=int, default=70)
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--dump", default="artifacts/swe_slot_dump.txt")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if _selftest() else 1)

    import torch
    from transformers import AutoTokenizer
    from v5.lm_loader import load_frozen_lm
    from v5.graph_grower.swe_load import load_instances, checkout_repo
    from v5.graph_grower.swe_probe import load_traces
    from v5.runtime.sr_withcode import load_symbol_meta, read_body
    from v5.runtime.search_replace import SR_SYS, parse_sr
    from v5.graph_grower.swe_verify import write_predictions

    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    model = load_frozen_lm(a.model); model.eval()
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=trust)
    dev = next(model.parameters()).device

    @torch.no_grad()
    def gen(system, user, ntok):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        kw = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
        try:
            enc = tok.apply_chat_template(msgs, enable_thinking=False, **kw).to(dev)
        except TypeError:
            enc = tok.apply_chat_template(msgs, **kw).to(dev)
        out = model.generate(**enc, max_new_tokens=ntok, do_sample=False, pad_token_id=tok.eos_token_id)
        t = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL).strip()

    traces = load_traces([a.traces])
    meta = load_symbol_meta([a.nodes])
    insts = {t["instance_id"]: t for t in load_instances(a.dataset, "test", limit=0)}
    ids = [i for i in traces if i in insts and all(s in meta for s in traces[i]["support_ids"])][:a.n_eval]
    print(f"instances={len(ids)} | symbol meta={len(meta)}", flush=True)

    dump = open(a.dump, "w", encoding="utf-8")
    oneshot_app = slot_app = scored = 0
    oneshot_preds, slot_preds = {}, {}
    for k, iid in enumerate(ids):
        t = traces[iid]; inst = insts[iid]
        support = [s for s in t["support_ids"] if s in meta]
        dest = Path(a.repo_root) / inst["repo"].replace("/", "__")
        ok, _ = checkout_repo(inst["repo"], inst["base_commit"], dest, timeout=1800)
        if not ok:
            print(f"  [{k+1}] {iid} checkout FAILED"); continue
        src = "\n\n".join(f"# {meta[s]['file']}\n{body}" for s in support[:a.src_bodies]
                          if (body := read_body(str(dest), meta[s]["file"], meta[s]["lineno"], a.src_lines)))
        if not src.strip():
            print(f"  [{k+1}] {iid} no source read"); continue
        scored += 1
        issue = t["issue"]

        # ONE-SHOT baseline (single SR emit)
        g1 = gen(SR_SYS, fix_user(issue, src), a.max_new)
        b1 = parse_sr(g1); app1 = bool(b1) and not _unmatched(b1, str(dest))
        oneshot_app += app1
        p1 = _patch(b1, str(dest))
        if p1.strip(): oneshot_preds[iid] = p1

        # SLOT path THROUGH THE ENGINE (DIAGNOSE -> FIX, backtrack/re-diagnose on unapplyable)
        def diagnose_fn(issue, src, attempt):
            return gen("You are a precise debugging assistant.", diag_user(issue, src, attempt), 160)
        def fix_fn(issue, src, diagnosis):
            g = gen(SR_SYS, fix_user(issue, src, diagnosis), a.max_new)
            return _patch(parse_sr(g), str(dest))           # "" unless applyable
        log = []
        p2, trace, fp, steps = slot_solve(issue, src, diagnose_fn, fix_fn, max_steps=a.max_steps, log=log)
        app2 = bool(p2.strip())
        slot_app += app2
        if app2: slot_preds[iid] = p2

        print(f"  [{k+1}/{len(ids)}] {iid:28} oneshot_app={app1} slot_app={app2} "
              f"slot_steps={steps} diag_attempts={len(trace['diagnoses'])} fixpoint={fp}", flush=True)
        dump.write(f"\n===== {iid} =====\nISSUE: {issue[:200]}\n\n"
                   f"SLOT log: {log}\n"
                   f"DIAGNOSES ({len(trace['diagnoses'])} attempts):\n" +
                   "\n".join(f"  [{i}] {d}" for i, d in enumerate(trace['diagnoses'])) +
                   f"\nFIX attempts applyable: {[x['applyable'] for x in trace['fix_attempts']]}\n"
                   f"ONESHOT applyable={app1}:\n{g1[:500]}\n")
    dump.close()
    n1 = write_predictions(oneshot_preds, "artifacts/swe_oneshot_preds.jsonl", "v5_oneshot")
    n2 = write_predictions(slot_preds, "artifacts/swe_slot_preds.jsonl", "v5_slot")
    print(f"\n=== #9 SYNTHESIS (engine-wired DIAGNOSE->FIX vs one-shot, given support) ===")
    print(f"  applyable@1:  ONE-SHOT {oneshot_app}/{scored}  |  SLOT(engine) {slot_app}/{scored}")
    print(f"  emitted predictions: oneshot {n1} | slot {n2}")
    print(f"  dump (MANUALLY INSPECT the diagnoses + retry behavior) -> {a.dump}")
    print(f"  VERIFY (Docker): gold-sanity FIRST, then both pred files with DISTINCT --run-id:")
    print(f"    python -m v5.graph_grower.swe_verify --gold-sanity --dataset {a.dataset} --limit 5")
    print(f"    python -m v5.graph_grower.swe_verify --predictions artifacts/swe_oneshot_preds.jsonl --dataset {a.dataset} --run-id oneshot")
    print(f"    python -m v5.graph_grower.swe_verify --predictions artifacts/swe_slot_preds.jsonl --dataset {a.dataset} --run-id slot")


if __name__ == "__main__":
    main()
