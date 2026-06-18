"""#9 (synthesis-focused, ENGINE-WIRED) — slot-graph DIAGNOSE->PLAN->FIX vs one-shot, through SlotGraph.solve.

Localization held FIXED (gold support symbols) to isolate SYNTHESIS from the ~0.30 localization wall.
The SLOT path now runs the REAL engine (slot_coder.SlotGraph): DIAGNOSE (revise='rederive') ->
PLAN (quote the exact source anchor + intended change) -> FIX. An ungrounded PLAN or an unapplyable /
misaligned FIX -> INSUFFICIENT -> dependency-directed BACKTRACK to the nearest upstream slot, then
re-plan / re-diagnose deeper, to a fixpoint (or max_steps). Compared to ONE-SHOT (single SR emit).

`slot_solve()` is shared by the real 4B run AND a no-model `--selftest` that PROVES the engine wiring
(DIAGNOSE->PLAN->FIX->INSUFFICIENT->backtrack->re-plan->anchored-fix->fixpoint) without a GPU. This is
the fix for the earlier mistake where swe_slot bypassed the engine entirely (see memory verify-wiring-not-proxy).

  4B (A40): V5_LM_TRUST_REMOTE_CODE=1 python -m v5.runtime.swe_slot --n-eval 24
  session : ... --session-out-dir artifacts/swe_slot_sessions --session-name lite_n24_run1
  exact   : ... --exact-verify --verify-backend docker    # gold-sanity + exact resolve on this box
  wiring  : python -m v5.runtime.swe_slot --selftest      # no model, proves the slot-graph is invoked
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

from v5.runtime.swe_exact_verify import SWEExactVerifier
from v5.runtime.slot_coder import SlotGraph, SlotSpec, Pool


def _canon_path(p: str) -> str:
    return (p or "").replace("\\", "/").strip()


def _same_file(a: str, b: str) -> bool:
    aa, bb = _canon_path(a), _canon_path(b)
    return bool(aa and bb) and (aa == bb or aa.endswith("/" + bb) or bb.endswith("/" + aa))


def _split_src_files(src: str) -> dict[str, str]:
    parts: dict[str, list[str]] = {}
    cur_file = ""
    cur_lines: list[str] = []
    for line in (src or "").splitlines():
        m = re.match(r"^# ([\w./\-]+\.\w+)\s*$", line)
        if m:
            if cur_file:
                parts[cur_file] = cur_lines[:]
            cur_file = _canon_path(m.group(1))
            cur_lines = []
            continue
        if cur_file:
            cur_lines.append(line)
    if cur_file:
        parts[cur_file] = cur_lines[:]
    return {k: "\n".join(v).rstrip("\n") for k, v in parts.items()}


def _format_plan(plan: dict[str, str]) -> str:
    return f"FILE: {plan['file']}\nSEARCH:\n{plan['search']}\nCHANGE:\n{plan['change']}\n"


def _parse_plan(text: str) -> dict[str, str]:
    file_match = re.search(r"(?mi)^FILE:\s*(.+?)\s*$", text or "")
    search_match = re.search(r"(?mis)^SEARCH:\s*\n(.*?)(?:\nCHANGE:\s*\n|\Z)", text or "")
    change_match = re.search(r"(?mis)^CHANGE:\s*\n(.*)$", text or "")
    return {
        "file": (file_match.group(1).strip() if file_match else ""),
        "search": (search_match.group(1).strip("\n") if search_match else ""),
        "change": (change_match.group(1).strip() if change_match else ""),
    }


def _best_search_anchor(file_body: str, search: str) -> str:
    file_lines = file_body.splitlines()
    groups: list[list[str]] = []
    cur: list[str] = []
    for line in (search or "").splitlines():
        if line.strip():
            cur.append(line.rstrip())
        elif cur:
            groups.append(cur[:])
            cur = []
    if cur:
        groups.append(cur[:])
    best_lines = 0
    best_chars = 0
    best_start = -1
    for group in groups:
        for start in range(len(file_lines)):
            matched = 0
            while matched < len(group) and start + matched < len(file_lines):
                if file_lines[start + matched].strip() != group[matched].strip():
                    break
                matched += 1
            if matched <= 0:
                continue
            chars = sum(len(file_lines[start + i].strip()) for i in range(matched))
            if matched > best_lines or (matched == best_lines and chars > best_chars):
                best_lines = matched
                best_chars = chars
                best_start = start
    if best_start < 0:
        return ""
    return "\n".join(file_lines[best_start: best_start + best_lines]).rstrip("\n")


def _repair_plan_to_src(plan_text: str, src: str) -> tuple[str, bool]:
    plan = _parse_plan(plan_text)
    fpath = _canon_path(plan["file"])
    if not (fpath and plan["change"].strip()):
        return plan_text, False
    file_body = _split_src_files(src).get(fpath, "")
    if not file_body:
        return plan_text, False
    search = plan["search"] or ""
    compact = [ln for ln in search.splitlines() if ln.strip()]
    if _plan_sufficient(plan_text, src) and len(compact) <= 8 and "\n\n" not in search:
        return plan_text, False
    repaired = _best_search_anchor(file_body, search)
    if not repaired:
        return plan_text, False
    plan["search"] = repaired
    new_text = _format_plan(plan)
    return new_text, new_text != plan_text


def _plan_sufficient(plan_text: str, src: str) -> bool:
    plan = _parse_plan(plan_text)
    fpath = _canon_path(plan["file"])
    search = (plan["search"] or "").strip()
    return bool(
        fpath
        and search
        and plan["change"].strip()
        and (f"# {fpath}" in src or fpath in src)
        and search in src
    )


def _blocks_match_plan(blocks: list[dict], plan_text: str) -> bool:
    plan = _parse_plan(plan_text)
    fpath = plan["file"]
    search = (plan["search"] or "").strip()
    if not (fpath and search):
        return False
    for b in blocks or []:
        if _same_file(b.get("file", ""), fpath):
            bsearch = (b.get("search") or "").strip()
            if search and (search in bsearch or bsearch in search):
                return True
    return False


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


def fix_user(issue, src, diagnosis="", plan=""):
    s = f"ISSUE:\n{issue[:1400]}\n\nRELEVANT SOURCE (the bug is in here):\n{src}\n\n"
    if diagnosis:
        s += f"ROOT-CAUSE DIAGNOSIS (use it):\n{diagnosis}\n\n"
    if plan:
        s += f"EDIT PLAN (follow it exactly):\n{plan}\n\n"
    return (s + "Fix the exact line(s) causing the bug. Output ONLY search/replace blocks: SEARCH "
            "must copy the source EXACTLY (character-for-character); REPLACE must DIFFER. Keep it minimal. "
            "If the diagnosis/change says the same edit must be made in multiple matching sites in the same "
            "file, update each matching site, but do not touch unrelated code.")


def plan_user(issue, src, diagnosis, attempt):
    nudge = "" if attempt == 0 else (
        " NOTE: the previous plan/fix missed the exact anchor. Copy the indentation EXACTLY and choose a "
        "smaller anchor from the shown source.")
    return (
        f"ISSUE:\n{issue[:1400]}\n\nRELEVANT SOURCE:\n{src}\n\nROOT-CAUSE DIAGNOSIS:\n{diagnosis}\n\n"
        "Plan the edit before patching. Output ONLY this format:\n"
        "FILE: path/from/source.py\nSEARCH:\n<exact existing code copied verbatim from the source>\n"
        "CHANGE:\n<one sentence saying what should change and why>\n"
        "The SEARCH block must be the SMALLEST exact source anchor that pins the buggy location (prefer 1-8 lines, "
        "keep the original indentation, do not quote a whole class/function when a smaller snippet will do). "
        "If the same edit repeats in one file, choose one representative exact anchor in source order and mention "
        "the repeated sites in CHANGE. FILE must match a shown source header."
        f"{nudge}"
    )


def diag_user(issue, src, attempt):
    nudge = "" if attempt == 0 else (
        " NOTE: a previous diagnosis led to an UNAPPLYABLE or no-op fix. Be more specific — name the "
        "EXACT function and the EXACT line/token to change, copied verbatim from the source above.")
    return (f"ISSUE:\n{issue[:1400]}\n\nSOURCE:\n{src}\n\nIn 2-3 sentences, state the ROOT CAUSE of the "
            f"bug and exactly what must change (name the function/lines). Be specific.{nudge}")


def slot_solve(issue, src, diagnose_fn, plan_fn, fix_fn, max_steps=8, log=None):
    """The SHARED slot-graph: DIAGNOSE -> PLAN -> FIX.
      diagnose_fn(issue, src, attempt)              -> diagnosis text
      plan_fn(issue, src, diagnosis, attempt)       -> FILE/SEARCH/CHANGE plan text
      fix_fn(issue, src, diagnosis, plan)           -> (APPLYABLE patch text, parsed SR blocks)
    Returns (patch, trace, fixpoint, steps). Same engine for the 4B run and the selftest."""
    attempts = {"DIAGNOSE": 0, "PLAN": 0}
    trace = {"diagnoses": [], "plans": [], "fix_attempts": []}
    fix_meta = {"blocks": [], "plan": ""}

    def retr(q, kind):
        return [{"id": "src", "text": src}]                    # localization fixed -> evidence = the source

    def filler(slot, ev, pool):
        if slot.name == "DIAGNOSE":
            n = attempts["DIAGNOSE"]; attempts["DIAGNOSE"] = n + 1
            d = diagnose_fn(issue, src, n)
            trace["diagnoses"].append(d)
            return d
        if slot.name == "PLAN":
            diag = pool.get("DIAGNOSE")
            n = attempts["PLAN"]; attempts["PLAN"] = n + 1
            plan = plan_fn(issue, src, diag, n)
            trace["plans"].append(plan)
            return plan
        diag = pool.get("DIAGNOSE")
        plan = pool.get("PLAN")
        patch, blocks = fix_fn(issue, src, diag, plan)        # "" if unapplyable/no-op -> INSUFFICIENT -> backtrack
        fix_meta["blocks"] = list(blocks or [])
        fix_meta["plan"] = plan
        trace["fix_attempts"].append({
            "applyable": bool(patch),
            "anchored": _blocks_match_plan(fix_meta["blocks"], plan),
            "diag_used": diag[:160],
        })
        return patch

    specs = [
        SlotSpec("DIAGNOSE", [], "src", "ASSERT", query=lambda p: "root cause of the bug", revise="rederive"),
        SlotSpec("PLAN", ["DIAGNOSE"], "src", "ASSERT",
                 query=lambda p: "exact quoted target lines and edit intent",
                 revise="rederive",
                 sufficient=lambda slot, pool: _plan_sufficient(slot.value, src)),
        SlotSpec("FIX", ["DIAGNOSE", "PLAN"], "src", "TRANSFORM",
                 query=lambda p: "minimal applyable fix aligned to the plan",
                 sufficient=lambda slot, pool: bool(slot.value)
                 and _blocks_match_plan(fix_meta["blocks"], fix_meta["plan"])),
    ]
    sg = SlotGraph(specs)
    pool = Pool(specs, context={"issue": issue, "src": src})
    ok, steps = sg.solve(pool, retr, filler, max_steps=max_steps, log=log)
    return pool.slots["FIX"].value, trace, ok, steps


# ── no-model wiring proof: the engine MUST reject an applyable-but-misaligned fix, then re-plan and recover ──
def _selftest():
    print("swe_slot --selftest: proving the SLOT path runs SlotGraph.solve (no model).\n")
    plan_src = (
        "# x.py\n"
        "def f():\n"
        "    if cond:\n"
        "        return bad()\n"
        "\n"
        "# y.py\n"
        "class Alpha:\n"
        "    value = 1\n"
        "\n"
        "class Beta:\n"
        "    value = 1\n"
    )
    raw_indent = (
        "FILE: x.py\nSEARCH:\n        if cond:\n            return bad()\nCHANGE:\n"
        "Use the fixed return.\n"
    )
    fixed_indent, repaired_indent = _repair_plan_to_src(raw_indent, plan_src)
    indent_ok = repaired_indent and _plan_sufficient(fixed_indent, plan_src)
    raw_order = (
        "FILE: y.py\nSEARCH:\nclass Beta:\n    value = 1\n\nclass Alpha:\n    value = 1\nCHANGE:\n"
        "Update both values.\n"
    )
    fixed_order, repaired_order = _repair_plan_to_src(raw_order, plan_src)
    order_ok = repaired_order and _plan_sufficient(fixed_order, plan_src)
    raw_trunc = (
        "FILE: x.py\nSEARCH:\ndef f():\n    if cond:\n        return bad(\nCHANGE:\n"
        "Close the call and fix the value.\n"
    )
    fixed_trunc, repaired_trunc = _repair_plan_to_src(raw_trunc, plan_src)
    trunc_ok = _plan_sufficient(fixed_trunc, plan_src)
    print(f"   plan repair (indent drift) : {'PASS' if indent_ok else 'FAIL'}")
    print(f"   plan repair (order drift)  : {'PASS' if order_ok else 'FAIL'}")
    print(f"   plan repair (truncation)   : {'PASS' if trunc_ok else 'FAIL'}"
          f"{' (snapped)' if repaired_trunc else ' (already sufficient)'}")

    src = "# x.py\ndef check(x, y):\n    if y < 5:\n        return y < 5\n    return x < 5\n"
    def diagnose_fn(issue, src, attempt):
        return "PRECISE: change `return x < 5` to `return x <= 5` in x.py"
    def plan_fn(issue, src, diagnosis, attempt):
        raw = "FILE: x.py\nSEARCH:\nreturn x < 5\nCHANGE:\nChange `<` to `<=` in the return line.\n"
        return _repair_plan_to_src(raw, src)[0]
    fix_attempts = {"n": 0}
    def fix_fn(issue, src, diagnosis, plan):
        fix_attempts["n"] += 1
        if fix_attempts["n"] == 1:
            blocks = [{"file": "x.py", "search": "return y < 5", "replace": "return y <= 5"}]
            return "diff --git a/x.py b/x.py\n+wrong-scope\n", blocks
        blocks = [{"file": "x.py", "search": "return x < 5", "replace": "return x <= 5"}]
        return "diff --git a/x.py b/x.py\n+fixed\n", blocks
    log = []
    patch, trace, ok, steps = slot_solve("issue", src, diagnose_fn, plan_fn, fix_fn, log=log)
    for row in log:
        print("   ", row)
    print(f"\n   diagnoses          : {trace['diagnoses']}")
    print(f"   plans              : {trace['plans']}")
    print(f"   fix attempts       : {[(a['applyable'], a['anchored']) for a in trace['fix_attempts']]}")
    print(f"   fixpoint={ok} steps={steps}  final patch applyable={bool(patch)}")
    backtracked = any(r[0] == "BACKTRACK" for r in log)
    replanned = len(trace["plans"]) >= 2
    anchored = trace["fix_attempts"][-1]["anchored"] if trace["fix_attempts"] else False
    rejected_wrong_scope = any(a["applyable"] and not a["anchored"] for a in trace["fix_attempts"][:-1])
    ok_wired = (
        indent_ok and order_ok and trunc_ok and bool(patch) and ok and backtracked and replanned
        and anchored and rejected_wrong_scope
    )
    print(f"\n   WIRING PROOF: backtrack-fired={backtracked}  re-planned={replanned}  "
          f"rejected-applyable-wrong-scope={rejected_wrong_scope}  recovered-anchored={anchored}"
          f"  -> {'PASS' if ok_wired else 'FAIL'}")
    print("   (attempt-1 fix was applyable but ignored the planned anchor -> INSUFFICIENT -> backtrack")
    print("    to PLAN -> attempt-2 follows the quoted source anchor -> fixpoint. The engine enforced scope.)")
    return ok_wired


def _exact_resolve_rate(verifier: SWEExactVerifier | None, name: str,
                        task_patches: list[tuple[dict, str]], scored: int):
    if verifier is None or scored <= 0:
        return None
    res = verifier.verify_task_batch_unique(task_patches, tag=name)
    resolved = sum(1 for task, _patch in task_patches if res.get(task["iid"], False))
    emitted = len(task_patches)
    return resolved / scored, emitted / scored, resolved, emitted


def _slug(text: str) -> str:
    keep = [c.lower() if c.isalnum() else "_" for c in (text or "session")]
    s = "".join(keep).strip("_")
    return s[:80] or "session"


def _prepare_outputs(args):
    if args.session_out_dir:
        tag = _slug(args.session_name or f"swe_slot_{args.dataset}_{args.split}_{time.strftime('%Y%m%d_%H%M%S')}")
        bundle = Path(args.session_out_dir) / tag
        bundle.mkdir(parents=True, exist_ok=True)
        return {
            "name": tag,
            "bundle": bundle,
            "dump": bundle / "dump.txt",
            "oneshot": bundle / "oneshot.jsonl",
            "slot": bundle / "slot.jsonl",
            "summary": bundle / "summary.json",
        }
    Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
    Path("artifacts").mkdir(parents=True, exist_ok=True)
    return {
        "name": "",
        "bundle": None,
        "dump": Path(args.dump),
        "oneshot": Path("artifacts/swe_oneshot_preds.jsonl"),
        "slot": Path("artifacts/swe_slot_preds.jsonl"),
        "summary": None,
    }


def _verify_run_ids(outputs, dataset: str, split: str) -> tuple[str, str]:
    if outputs["name"]:
        base = outputs["name"]
    else:
        base = _slug(f"swe_slot_{dataset}_{split}")
    return f"{base}_oneshot", f"{base}_slot"


def _apply_smoke_overrides(args):
    if not args.smoke:
        return args
    orig_n = args.n_eval
    orig_gold = args.verify_gold_sanity
    args.n_eval = min(args.n_eval, max(1, args.smoke_n_eval))
    if args.exact_verify:
        args.verify_gold_sanity = min(args.verify_gold_sanity, args.n_eval, max(1, args.smoke_gold_sanity))
    print(f"[SMOKE] preflight enabled: n_eval {orig_n} -> {args.n_eval}", flush=True)
    if args.exact_verify:
        print(f"[SMOKE] verifier preflight: gold_sanity {orig_gold} -> {args.verify_gold_sanity}", flush=True)
    return args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="no-model proof the SLOT path uses the engine")
    ap.add_argument("--smoke", action="store_true",
                    help="cheap preflight: run a tiny generation slice before the full expensive session")
    ap.add_argument("--smoke-n-eval", type=int, default=2,
                    help="when --smoke is set, cap n_eval to this many instances")
    ap.add_argument("--smoke-gold-sanity", type=int, default=2,
                    help="when --smoke and --exact-verify are both set, cap gold-sanity to this many instances")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--traces", default="data/swe/grounded_traces.jsonl")
    ap.add_argument("--nodes", default="artifacts/graph_growth/swe_code_candidates.jsonl")
    ap.add_argument("--dataset", default="lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--repo-root", default="data/swe_repos")
    ap.add_argument("--n-eval", type=int, default=10)
    ap.add_argument("--src-bodies", type=int, default=4)
    ap.add_argument("--src-lines", type=int, default=70)
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--exact-verify", action="store_true",
                    help="run exact SWE verification for one-shot and slot outputs after emission")
    ap.add_argument("--verify-backend", choices=["docker", "sbcli"], default="docker")
    ap.add_argument("--verify-out-dir", default="artifacts/graph_growth/swe_verify")
    ap.add_argument("--verify-max-workers", type=int, default=4)
    ap.add_argument("--verify-timeout", type=int, default=1800)
    ap.add_argument("--verify-poll-secs", type=int, default=20)
    ap.add_argument("--verify-gold-sanity", type=int, default=5,
                    help="when exact verify is active, require this many gold patches to resolve first")
    ap.add_argument("--session-out-dir", default="",
                    help="optional directory to write a per-run session bundle (predictions + dump + summary)")
    ap.add_argument("--session-name", default="",
                    help="optional session bundle name; default = swe_slot_<dataset>_<split>_<timestamp>")
    ap.add_argument("--dump", default="artifacts/swe_slot_dump.txt")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if _selftest() else 1)
    a = _apply_smoke_overrides(a)
    outputs = _prepare_outputs(a)
    oneshot_run_id, slot_run_id = _verify_run_ids(outputs, a.dataset, a.split)

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
    insts = {t["instance_id"]: t for t in load_instances(a.dataset, a.split, limit=0)}
    ids = [i for i in traces if i in insts and all(s in meta for s in traces[i]["support_ids"])][:a.n_eval]
    print(f"instances={len(ids)} | symbol meta={len(meta)}", flush=True)
    verifier = SWEExactVerifier(a.dataset, a.split, a.verify_backend, a.verify_out_dir,
                                max_workers=a.verify_max_workers, timeout=a.verify_timeout,
                                poll_secs=a.verify_poll_secs, model_name="swe_slot") if a.exact_verify else None

    dump = open(outputs["dump"], "w", encoding="utf-8")
    oneshot_app = slot_app = scored = 0
    oneshot_preds, slot_preds = {}, {}
    eval_tasks, oneshot_eval, slot_eval = [], [], []
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
        task = {"iid": iid, "gold": inst.get("patch", "")}
        eval_tasks.append(task)

        # ONE-SHOT baseline (single SR emit)
        g1 = gen(SR_SYS, fix_user(issue, src), a.max_new)
        b1 = parse_sr(g1); app1 = bool(b1) and not _unmatched(b1, str(dest))
        oneshot_app += app1
        p1 = _patch(b1, str(dest))
        if p1.strip():
            oneshot_preds[iid] = p1
            oneshot_eval.append((task, p1))

        # SLOT path THROUGH THE ENGINE (DIAGNOSE -> PLAN -> FIX, backtrack on ungrounded plans / wrong-scope fixes)
        def diagnose_fn(issue, src, attempt):
            return gen("You are a precise debugging assistant.", diag_user(issue, src, attempt), 160)
        def plan_fn(issue, src, diagnosis, attempt):
            raw = gen("You are a precise patch planner.", plan_user(issue, src, diagnosis, attempt), 220)
            return _repair_plan_to_src(raw, src)[0]
        def fix_fn(issue, src, diagnosis, plan):
            g = gen(SR_SYS, fix_user(issue, src, diagnosis, plan), a.max_new)
            blocks = parse_sr(g)
            return _patch(blocks, str(dest)), blocks        # "" unless applyable
        log = []
        p2, trace, fp, steps = slot_solve(issue, src, diagnose_fn, plan_fn, fix_fn,
                                          max_steps=a.max_steps, log=log)
        app2 = bool(p2.strip())
        slot_app += app2
        if app2:
            slot_preds[iid] = p2
            slot_eval.append((task, p2))

        print(f"  [{k+1}/{len(ids)}] {iid:28} oneshot_app={app1} slot_app={app2} "
              f"slot_steps={steps} diag_attempts={len(trace['diagnoses'])} "
              f"plan_attempts={len(trace['plans'])} fixpoint={fp}", flush=True)
        dump.write(f"\n===== {iid} =====\nISSUE: {issue[:200]}\n\n"
                   f"SLOT log: {log}\n"
                   f"DIAGNOSES ({len(trace['diagnoses'])} attempts):\n" +
                   "\n".join(f"  [{i}] {d}" for i, d in enumerate(trace['diagnoses'])) +
                   f"\nPLANS ({len(trace['plans'])} attempts):\n" +
                   "\n".join(f"  [{i}] {p}" for i, p in enumerate(trace['plans'])) +
                   f"\nFIX attempts (applyable, anchored): "
                   f"{[(x['applyable'], x['anchored']) for x in trace['fix_attempts']]}\n"
                   f"ONESHOT applyable={app1}:\n{g1[:500]}\n")
    dump.close()
    oneshot_path = str(outputs["oneshot"])
    slot_path = str(outputs["slot"])
    n1 = write_predictions(oneshot_preds, oneshot_path, "v5_oneshot")
    n2 = write_predictions(slot_preds, slot_path, "v5_slot")
    print(f"\n=== #9 SYNTHESIS (engine-wired DIAGNOSE->PLAN->FIX vs one-shot, given support) ===")
    print(f"  applyable@1:  ONE-SHOT {oneshot_app}/{scored}  |  SLOT(engine) {slot_app}/{scored}")
    print(f"  emitted predictions: oneshot {n1} -> {oneshot_path} | slot {n2} -> {slot_path}")
    print(f"  dump (MANUALLY INSPECT the diagnoses + retry behavior) -> {outputs['dump']}")
    exact1 = exact2 = None
    if verifier is not None:
        gold_n = min(a.verify_gold_sanity, len(eval_tasks))
        if gold_n > 0:
            ok, total = verifier.run_gold_sanity(eval_tasks, gold_n, tag="slot_gold_sanity")
            print(f"  gold-sanity: {ok}/{total} gold patches resolved", flush=True)
            if ok != total:
                raise SystemExit("gold-sanity failed; refusing to trust exact SWE verifier results")
        exact1 = _exact_resolve_rate(verifier, "slot_oneshot", oneshot_eval, scored)
        exact2 = _exact_resolve_rate(verifier, "slot_graph", slot_eval, scored)
        if exact1 is not None and exact2 is not None:
            r1, e1, ok1, emit1 = exact1
            r2, e2, ok2, emit2 = exact2
            print(f"  exact resolve: ONE-SHOT {ok1}/{scored} ({r1:.0%}) | SLOT(engine) {ok2}/{scored} ({r2:.0%})")
            print(f"  patch emission: ONE-SHOT {emit1}/{scored} ({e1:.0%}) | SLOT(engine) {emit2}/{scored} ({e2:.0%})")
    else:
        print("  exact verify not run here. To score these predictions on the verifier box:")
        smoke_gold = min(2, max(1, scored))
        smoke_oneshot = min(2, max(1, n1)) if n1 else 0
        smoke_slot = min(2, max(1, n2)) if n2 else 0
        print("  quick smoke first (cheap sanity before full Docker run):")
        print(f"    python -m v5.graph_grower.swe_verify --gold-sanity --dataset {a.dataset} --split {a.split} --limit {smoke_gold}")
        if smoke_oneshot:
            print(f"    python -m v5.graph_grower.swe_verify --predictions {oneshot_path} --dataset {a.dataset} --split {a.split} --run-id {oneshot_run_id}_smoke --predictions-limit {smoke_oneshot}")
        if smoke_slot:
            print(f"    python -m v5.graph_grower.swe_verify --predictions {slot_path} --dataset {a.dataset} --split {a.split} --run-id {slot_run_id}_smoke --predictions-limit {smoke_slot}")
        print("  then the full batch:")
        print(f"    python -m v5.graph_grower.swe_verify --gold-sanity --dataset {a.dataset} --split {a.split} --limit 5")
        print(f"    python -m v5.graph_grower.swe_verify --predictions {oneshot_path} --dataset {a.dataset} --split {a.split} --run-id {oneshot_run_id}")
        print(f"    python -m v5.graph_grower.swe_verify --predictions {slot_path} --dataset {a.dataset} --split {a.split} --run-id {slot_run_id}")
    if outputs["summary"] is not None:
        summary = {
            "session_name": outputs["name"],
            "dataset": a.dataset,
            "split": a.split,
            "model": a.model,
            "n_eval_requested": a.n_eval,
            "n_eval_scored": scored,
            "max_steps": a.max_steps,
            "max_new": a.max_new,
            "predictions": {
                "oneshot": oneshot_path,
                "slot": slot_path,
            },
            "dump_path": str(outputs["dump"]),
            "applyable": {
                "oneshot": {"count": oneshot_app, "total": scored},
                "slot": {"count": slot_app, "total": scored},
            },
            "verify_commands": {
                "gold_sanity_smoke": f"python -m v5.graph_grower.swe_verify --gold-sanity --dataset {a.dataset} --split {a.split} --limit {min(2, max(1, scored))}",
                "oneshot_smoke": (f"python -m v5.graph_grower.swe_verify --predictions {oneshot_path} --dataset {a.dataset} "
                                   f"--split {a.split} --run-id {oneshot_run_id}_smoke --predictions-limit {min(2, max(1, n1))}")
                                   if n1 else "",
                "slot_smoke": (f"python -m v5.graph_grower.swe_verify --predictions {slot_path} --dataset {a.dataset} "
                                f"--split {a.split} --run-id {slot_run_id}_smoke --predictions-limit {min(2, max(1, n2))}")
                                if n2 else "",
                "gold_sanity": f"python -m v5.graph_grower.swe_verify --gold-sanity --dataset {a.dataset} --split {a.split} --limit 5",
                "oneshot": f"python -m v5.graph_grower.swe_verify --predictions {oneshot_path} --dataset {a.dataset} --split {a.split} --run-id {oneshot_run_id}",
                "slot": f"python -m v5.graph_grower.swe_verify --predictions {slot_path} --dataset {a.dataset} --split {a.split} --run-id {slot_run_id}",
            },
        }
        if exact1 is not None and exact2 is not None:
            summary["exact_resolve"] = {
                "oneshot": {"resolved": exact1[2], "emitted": exact1[3], "total": scored},
                "slot": {"resolved": exact2[2], "emitted": exact2[3], "total": scored},
            }
        with open(outputs["summary"], "w", encoding="utf-8") as w:
            json.dump(summary, w, indent=2)
        print(f"  session bundle -> {outputs['bundle']}")
        print(f"  session summary -> {outputs['summary']}")


if __name__ == "__main__":
    main()
