"""The ASSEMBLED LGGN loop (finally merged): traverse-rank -> realize (frozen LLM) -> verify ->
INVALIDATE failed operator + re-traverse, up to K iterations. Tests the HRM-transposed hypothesis:
does iterating against test-feedback beat one-shot?  -> resolved@1 vs resolved@K.

Loop logic is pluggable (rank_fn/realize_fn/verify_fn) so it selftests with mocks; main() wires the
REAL pieces (4B realize via swe_slot helpers + swe_verify Docker). Needs 4B + Docker in one process.

  V5_LM_TRUST_REMOTE_CODE=1 V5_LM_QUANT=4bit python -m v5.runtime.lggn_loop --model <4B> --n 8 --iters 4
  python -m v5.runtime.lggn_loop --selftest
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


# ── the loop (pure logic, pluggable) ──────────────────────────────────────────
def solve_instance(task, rank_fn, realize_fn, verify_fn, iters=4):
    """traverse -> realize -> verify -> INVALIDATE + re-traverse. Returns resolved + per-iter trace."""
    invalidated, feedback, trace = [], "", []
    for k in range(iters):
        plan = rank_fn(task, invalidated)                 # next-best operator not yet tried
        if plan is None:
            break
        patch = realize_fn(task, plan, feedback)          # frozen LLM realizes the plan (+ prior feedback)
        resolved, fb = verify_fn(task, patch) if patch else (False, "no patch produced")
        trace.append({"iter": k, "op": plan.get("name"), "applyable": bool(patch), "resolved": resolved})
        if resolved:
            return {"resolved": True, "iters": k + 1, "trace": trace}
        invalidated.append(plan.get("name")); feedback = fb   # INVALIDATE + carry the failure forward
    return {"resolved": False, "iters": len(trace), "trace": trace}


# ── real component wiring ─────────────────────────────────────────────────────
def _gold_file(patch):
    m = re.search(r"diff --git a/(\S+)", patch or "")
    return m.group(1) if m else ""


def _src_window(dest, patch, span=60):
    """Oracle-localized source window around the gold hunk (the loop tests REFINEMENT, not localization)."""
    f = _gold_file(patch)
    p = Path(dest) / f
    if not p.exists():
        return "", f
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    m = re.search(r"@@ -(\d+)", patch or "")
    lo = max(0, int(m.group(1)) - 1 - span // 2) if m else 0
    return "\n".join(lines[lo:lo + span]), f


def make_rank(embed_fn, ops):
    import numpy as np
    C = np.array([o["_centroid"] for o in ops if o.get("_centroid")], dtype=float)
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
    live = [o for o in ops if o.get("_centroid")]

    def rank(task, invalidated):
        g = np.asarray(embed_fn([task["_goal"]])[0]); g = g / (np.linalg.norm(g) + 1e-9)
        order = np.argsort(-(Cn @ g))
        for i in order:
            if live[i]["name"] not in invalidated:
                return live[i]
        return None
    return rank


DEBUG = [False]


def make_realize(gen_fn):
    from v5.runtime.swe_slot import fix_user, _repair_sr_to_src, _patch
    from v5.runtime.search_replace import parse_sr

    def realize(task, plan, feedback):
        dest = task["_dest"]
        src, f = _src_window(dest, task["patch"])
        prompt = fix_user(task.get("problem_statement", ""), src,
                          plan=f"{plan['name']}: {plan.get('realize_hint', '')}", test_failure=feedback)
        raw = gen_fn(prompt)
        blocks = parse_sr(raw)
        patch = ""
        if blocks:
            blocks = _repair_sr_to_src(blocks, dest)
            patch = _patch(blocks, dest)
        if DEBUG[0]:
            print(f"    [realize op={plan['name']}] file={f} src_len={len(src)} raw_len={len(raw)} "
                  f"blocks={len(blocks or [])} patch_len={len(patch)}")
            print(f"      RAW[:260]={raw[:260]!r}")
            if patch:
                print(f"      PATCH[:260]={patch[:260]!r}")
        return patch
    return realize


def make_proxy_verify():
    """No-Docker proxy: resolved := the emitted patch reproduces the gold added lines (recall>=0.7).
    Lets the LOOP MECHANISM (INVALIDATE + re-traverse) be tested on a GPU-only box (molab). Feedback is
    generic (like a failing test) so the loop must iterate via trying DIFFERENT operators, not oracle."""
    from v5.runtime.operator_discovery import extract_hunks
    from v5.runtime.solution_ladder import _fidelity

    def verify(task, patch):
        gold = "\n".join(l for _f, _r, a in extract_hunks(task["patch"]) for l in a)
        emit = "\n".join(l[1:] for l in (patch or "").splitlines() if l.startswith("+") and not l.startswith("+++"))
        rec, _ = _fidelity(emit, gold)
        if DEBUG[0]:
            print(f"    [proxy] recall={rec:.2f} emit[:120]={emit[:120]!r} gold[:120]={gold[:120]!r}")
        return (rec >= 0.7, "the previous patch did NOT pass the tests; try a different fix approach")
    return verify


def run(insts, ops, gen_fn, embed_fn, backend, dataset, iters, proxy=False, log=print):
    from v5.graph_grower.swe_load import checkout_repo
    rank_fn = make_rank(embed_fn, ops)
    realize_fn = make_realize(gen_fn)
    if proxy:
        _pv = make_proxy_verify()
    else:
        from v5.graph_grower.swe_verify import verify_one
    cache = Path("artifacts/lggn_loop_repos"); cache.mkdir(parents=True, exist_ok=True)
    at1 = solved = n = 0
    for task in insts:
        dest = cache / task["instance_id"]
        try:
            if not dest.exists():
                checkout_repo(task["repo"], task["base_commit"], dest)
        except Exception as e:                                # noqa: BLE001
            log(f"  [skip {task['instance_id']}] checkout failed: {e}"); continue
        task["_dest"] = str(dest)
        task["_goal"] = (task.get("problem_statement", "")[:900]) + "\nCODE:\n" + _src_window(str(dest), task["patch"])[0][:600]
        verify_fn = _pv if proxy else (lambda t, p: verify_one(t["instance_id"], p, dataset, f"lggnloop_{t['instance_id']}", backend))
        r = solve_instance(task, rank_fn, realize_fn, verify_fn, iters)
        n += 1
        solved += int(r["resolved"])
        at1 += int(bool(r["trace"]) and r["trace"][0]["resolved"])
        log(f"  {task['instance_id']}: resolved={r['resolved']} @iter={r['iters']} "
            f"ops={[t['op'] for t in r['trace']]}")
    print(f"\n=== LGGN LOOP (n={n}, iters<= {iters}) ===")
    print(f"  resolved@1 (one-shot)   : {at1}/{n} ({at1/max(1,n):.0%})")
    print(f"  resolved@K (iterated)   : {solved}/{n} ({solved/max(1,n):.0%})")
    print(f"  iteration lift          : +{solved-at1} ({(solved-at1)/max(1,n):+.0%}) -> "
          f"{'ITERATION-AGAINST-TESTS HELPS (build the trained refiner)' if solved > at1 else 'flat (frozen 4B does not refine -> refiner must be TRAINED)'}")


def _selftest() -> bool:
    print("lggn_loop --selftest: iteration must resolve what one-shot cannot (mock realize/verify)\n")
    ops = [{"name": "A", "realize_hint": "wrong"}, {"name": "B", "realize_hint": "right"}]
    task = {"instance_id": "t", "_goal": "g", "gold": "FIXED"}

    def rank(task, invalidated):
        for o in ops:
            if o["name"] not in invalidated:
                return o
        return None

    def realize(task, plan, feedback):                        # op B (2nd, after A invalidated) produces the fix
        return "PATCH_FIXED" if plan["name"] == "B" else "PATCH_WRONG"

    def verify(task, patch):
        return (patch == "PATCH_FIXED", "tests failed")
    r = solve_instance(task, rank, realize, verify, iters=4)
    print(f"  resolved={r['resolved']} @iter={r['iters']} trace={[t['op'] for t in r['trace']]}")
    assert r["resolved"] and r["iters"] == 2 and not r["trace"][0]["resolved"], "iteration (INVALIDATE A -> try B) must resolve"
    print("\n  LGGN LOOP SELFTEST -> PASS (INVALIDATE + re-traverse resolved what one-shot could not)")
    return True


def main():
    ap = argparse.ArgumentParser(description="Assembled LGGN loop: iterate against tests (resolved@1 vs @K).")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--dataset", default="lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--backend", default="docker")
    ap.add_argument("--ops", default="data/swe/discovered_ops_emb.jsonl")
    ap.add_argument("--debug", action="store_true", help="dump realize+verify chain for diagnosis")
    ap.add_argument("--proxy-verify", action="store_true", help="gold-match instead of Docker (GPU-only box, tests the mechanism)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if _selftest() else 1)
    DEBUG[0] = a.debug
    from v5.graph_grower.swe_load import load_instances
    from v5.runtime.solution_ladder import real_gen_fn
    import torch
    from v5.training.providers import RealEmbedder
    _e = RealEmbedder(torch.device("cpu"))

    def embed_fn(ts):
        d = _e.embed_nodes({str(j): t[:1000] for j, t in enumerate(ts)})
        return [d[str(j)] for j in range(len(ts))]
    ops = [json.loads(l) for l in Path(a.ops).read_text(encoding="utf-8").splitlines() if l.strip()]
    insts = list(load_instances(name=a.dataset, split=a.split, limit=a.n))
    print(f"[lggn-loop] {len(insts)} instances, model={a.model}, iters<= {a.iters}, "
          f"verify={'proxy(gold-match)' if a.proxy_verify else 'docker'}")
    run(insts, ops, real_gen_fn(a.model), embed_fn, a.backend, a.dataset, a.iters, proxy=a.proxy_verify)


if __name__ == "__main__":
    main()
