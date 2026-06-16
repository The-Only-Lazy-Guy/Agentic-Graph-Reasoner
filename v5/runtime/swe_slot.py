"""#9 (synthesis-focused) — does slot-decomposed DIAGNOSE->FIX beat one-shot patch synthesis?

Holds localization FIXED (use the gold support symbols) to isolate the slot-graph's SYNTHESIS
contribution from the ~0.30 localization wall (which the slot-graph does NOT fix). For each instance:
read the support source, then compare:
  ONE-SHOT : emit a SEARCH/REPLACE patch directly from issue+source.
  SLOT     : DIAGNOSE (frozen 4B states the root cause from issue+source) -> FIX (emit SR from
             issue+source+DIAGNOSIS). The decomposition the slot-graph provides.
Metric = applyable@1 (SR SEARCH matches the file -- no verifier needed) + DUMP the diagnosis + patches
for MANUAL inspection (cheap scores lie). Verifier/resolve = the heavy optional last step.

  4B (A40): V5_LM_TRUST_REMOTE_CODE=1 python -m v5.runtime.swe_slot --n-eval 10
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import torch

from v5.lm_loader import load_frozen_lm
from v5.graph_grower.swe_load import load_instances, checkout_repo
from v5.graph_grower.swe_probe import load_traces
from v5.runtime.sr_withcode import load_symbol_meta, read_body, _file_text
from v5.runtime.search_replace import SR_SYS, parse_sr, apply_sr
from v5.graph_grower.swe_verify import write_predictions
import subprocess


def _unmatched(blocks, dest):
    return [b for b in blocks if (b.get("search") or "").strip()
            and (b.get("search") or "").strip() not in _file_text(dest, b.get("file"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--traces", default="data/swe/grounded_traces.jsonl")
    ap.add_argument("--nodes", default="artifacts/graph_growth/swe_code_candidates.jsonl")
    ap.add_argument("--dataset", default="lite")
    ap.add_argument("--repo-root", default="data/swe_repos")
    ap.add_argument("--n-eval", type=int, default=10)
    ap.add_argument("--src-bodies", type=int, default=4)
    ap.add_argument("--src-lines", type=int, default=70)
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--dump", default="artifacts/swe_slot_dump.txt")
    a = ap.parse_args()
    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    model = load_frozen_lm(a.model); model.eval()
    from transformers import AutoTokenizer
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

    def fix_user(issue, src, diagnosis=""):
        s = f"ISSUE:\n{issue[:1400]}\n\nRELEVANT SOURCE (the bug is in here):\n{src}\n\n"
        if diagnosis:
            s += f"ROOT-CAUSE DIAGNOSIS (use it):\n{diagnosis}\n\n"
        return (s + "Fix the exact line(s) causing the bug. Output ONLY search/replace blocks: SEARCH "
                "must copy the source EXACTLY (character-for-character); REPLACE must DIFFER. Keep it minimal.")

    traces = load_traces([a.traces])
    meta = load_symbol_meta([a.nodes])
    insts = {t["instance_id"]: t for t in load_instances(a.dataset, "test", limit=0)}
    ids = [i for i in traces if i in insts and all(s in meta for s in traces[i]["support_ids"])][:a.n_eval]
    print(f"instances={len(ids)} | symbol meta={len(meta)}", flush=True)

    dump = open(a.dump, "w", encoding="utf-8")
    oneshot_app = slot_app = scored = 0
    oneshot_preds, slot_preds = {}, {}
    def _patch(blocks, dest):                # applyable -> git diff (the swebench prediction), then restore
        if not (bool(blocks) and not _unmatched(blocks, dest)):
            return ""
        _, p = apply_sr(dest, blocks)
        subprocess.run(["git", "-C", dest, "checkout", "--", "."], capture_output=True)
        return p
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
        # ONE-SHOT
        g1 = gen(SR_SYS, fix_user(t["issue"], src), a.max_new)
        b1 = parse_sr(g1); app1 = bool(b1) and not _unmatched(b1, str(dest))
        oneshot_app += app1
        # SLOT: DIAGNOSE -> FIX
        diag = gen("You are a precise debugging assistant.",
                   f"ISSUE:\n{t['issue'][:1400]}\n\nSOURCE:\n{src}\n\nIn 2-3 sentences, state the ROOT "
                   f"CAUSE of the bug and exactly what must change (name the function/lines). Be specific.", 160)
        g2 = gen(SR_SYS, fix_user(t["issue"], src, diag), a.max_new)
        b2 = parse_sr(g2); app2 = bool(b2) and not _unmatched(b2, str(dest))
        slot_app += app2
        p1, p2 = _patch(b1, str(dest)), _patch(b2, str(dest))   # emit predictions for the verifier
        if p1.strip(): oneshot_preds[iid] = p1
        if p2.strip(): slot_preds[iid] = p2
        print(f"  [{k+1}/{len(ids)}] {iid:28} oneshot_app={app1} slot_app={app2}", flush=True)
        dump.write(f"\n===== {iid} =====\nISSUE: {t['issue'][:200]}\n\nDIAGNOSIS:\n{diag}\n\n"
                   f"ONESHOT blocks={len(b1)} applyable={app1}:\n{g1[:500]}\n\n"
                   f"SLOT blocks={len(b2)} applyable={app2}:\n{g2[:500]}\n")
    dump.close()
    n1 = write_predictions(oneshot_preds, "artifacts/swe_oneshot_preds.jsonl", "v5_oneshot")
    n2 = write_predictions(slot_preds, "artifacts/swe_slot_preds.jsonl", "v5_slot")
    print(f"\n=== #9 SYNTHESIS (DIAGNOSE->FIX vs one-shot, given support) ===")
    print(f"  applyable@1:  ONE-SHOT {oneshot_app}/{scored}  |  SLOT(diagnose->fix) {slot_app}/{scored}")
    print(f"  emitted predictions: oneshot {n1} -> artifacts/swe_oneshot_preds.jsonl | slot {n2} -> artifacts/swe_slot_preds.jsonl")
    print(f"  dump (MANUALLY INSPECT) -> {a.dump}")
    print(f"  VERIFY (Docker, CPU-only): gold-sanity FIRST, then both pred files -> the RESOLVE lift:")
    print(f"    python -m v5.graph_grower.swe_verify --gold-sanity --dataset {a.dataset} --limit 5")
    print(f"    python -m v5.graph_grower.swe_verify --predictions artifacts/swe_oneshot_preds.jsonl --dataset {a.dataset}")
    print(f"    python -m v5.graph_grower.swe_verify --predictions artifacts/swe_slot_preds.jsonl --dataset {a.dataset}")


if __name__ == "__main__":
    main()
