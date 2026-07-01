"""Solution ladder — isolate EMISSION vs DERIVATION vs TRAVERSAL-value on the frozen leaf.

Feed the leaf progressively more, measure how faithfully it emits the gold fix:
  a) issue + buggy code                      -> baseline (derive + emit)
  b) a + the operator PLAN                    -> does the plan help?           (b - a = plan value)
  c) a + the EXACT gold replacement code      -> pure EMISSION (given the answer, can it emit it?)

c low  -> the leaf can't emit a handed fix -> the wall is CODE-EMISSION (verbatim/format).
c high -> emission is fine -> the wall is DERIVATION/TRAVERSAL (finding the fix).
b > a  -> the operator plan lifts realization; b ~= a -> the plan doesn't help the leaf.

NO Docker (compares emitted REPLACE to the gold added lines). Needs the leaf (molab GPU or local 6GB).

  V5_LM_TRUST_REMOTE_CODE=1 V5_LM_QUANT=4bit python -m v5.runtime.solution_ladder --model Qwen/Qwen3.5-4B --n 30
  python -m v5.runtime.solution_ladder --selftest
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_SR = re.compile(r"<{5,}\s*SEARCH\s*(.*?)\s*={5,}\s*(.*?)\s*>{5,}\s*REPLACE", re.S)


def _norm_lines(s: str) -> list[str]:
    return [re.sub(r"\s+", " ", ln).strip() for ln in (s or "").splitlines() if ln.strip()]


def _emitted_replace(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)   # Qwen thinking models pollute the block
    m = _SR.search(text)
    return m.group(2) if m else ""


def _fidelity(emitted_replace: str, gold_added: str) -> tuple[float, bool]:
    """(line-recall of gold-added inside the emitted replace, exact-set-match)."""
    g = _norm_lines(gold_added)
    if not g:
        return 0.0, False
    e = set(_norm_lines(emitted_replace))
    recall = sum(1 for ln in g if ln in e) / len(g)
    return recall, set(g) == e


def _prompts(issue, removed, added, op_name, op_hint):
    base = (f"Issue:\n{issue[:800]}\n\nBuggy code:\n{removed[:800]}\n\n"
            "Output ONLY a search/replace block, no prose:\n"
            "<<<<<<< SEARCH\n<exact buggy code>\n=======\n<fixed code>\n>>>>>>> REPLACE")
    return {
        "a_issue": base + "\n\nFix the bug.",
        "b_plan": base + f"\n\nApply this repair operator: {op_name} -- {op_hint[:200]}",
        "c_gold": base + f"\n\nThe correct fixed code is EXACTLY:\n{added[:800]}\n\nEmit the block using it verbatim.",
    }


def run_ladder(instances, ops, gen_fn, log=print, dump=""):
    from v5.runtime.operator_discovery import extract_hunks, chunk_embedding, _fix_text
    import numpy as np
    embed = None
    if any(o.get("_centroid") for o in ops):                 # embed_fn for operator assignment
        import torch
        from v5.training.providers import RealEmbedder
        _e = RealEmbedder(torch.device("cpu"))

        def embed(ts):
            d = _e.embed_nodes({str(j): t[:1000] for j, t in enumerate(ts)})
            return [d[str(j)] for j in range(len(ts))]
    name2op = {o["name"]: o for o in ops}
    rungs = ["a_issue", "b_plan", "c_gold"]
    rec = {r: [] for r in rungs}; exact = {r: [] for r in rungs}; n = 0; samples = []
    for i in instances:
        hs = extract_hunks(i.get("patch", "") or "")
        if not hs:
            continue
        removed = "\n".join(l for _f, r, _a in hs for l in r)
        added = "\n".join(l for _f, _r, a in hs for l in a)
        if not added.strip():
            continue
        opname = (chunk_embedding(i["patch"], ops, embed) or ["<none>"])[0] if embed else "<none>"
        ophint = name2op.get(opname, {}).get("realize_hint", "apply the repair")
        pr = _prompts(i.get("problem_statement") or i.get("issue") or "", removed, added, opname, ophint)
        n += 1
        for r in rungs:
            g = gen_fn(pr[r])
            er = _emitted_replace(g)
            f, ex = _fidelity(er, added)
            rec[r].append(f); exact[r].append(int(ex))
            if dump and n <= 6:
                samples.append(f"===== {i['instance_id']} [{r}] recall={f:.0%} =====\n"
                               f"GOLD ADDED:\n{added[:250]}\n\nEMITTED REPLACE:\n{er[:250]}\n\nRAW GEN:\n{g[:400]}\n")
        if n % 5 == 0:
            log(f"  ...{n} instances")
    if dump and samples:
        Path(dump).write_text("\n".join(samples), encoding="utf-8"); log(f"  raw samples -> {dump}")
    out = {r: (float(np.mean(rec[r])) if rec[r] else 0.0, float(np.mean(exact[r])) if exact[r] else 0.0) for r in rungs}
    return out, n


def _report(out, n):
    print(f"\n=== SOLUTION LADDER (n={n}) — emission fidelity of the gold fix ===")
    lab = {"a_issue": "a) issue only        ", "b_plan": "b) + operator plan   ", "c_gold": "c) + EXACT gold code "}
    for r in ("a_issue", "b_plan", "c_gold"):
        rc, ex = out[r]
        print(f"  {lab[r]} line-recall {rc:.0%} | exact-match {ex:.0%}")
    plan = out["b_plan"][0] - out["a_issue"][0]
    print(f"\n  plan value (b - a)   = {plan:+.0%}  -> {'plan HELPS realization' if plan > 0.05 else 'plan does NOT help the leaf'}")
    print(f"  emission ceiling (c) = {out['c_gold'][0]:.0%}  -> "
          f"{'EMISSION is fine (wall = derivation/traversal)' if out['c_gold'][0] > 0.7 else 'EMISSION is the wall (leaf cannot emit a handed fix)'}")


def real_gen_fn(model_name):
    import torch
    from transformers import AutoTokenizer
    from v5.lm_loader import load_frozen_lm
    import os
    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    model = load_frozen_lm(model_name); model.eval()
    dev = next(model.parameters()).device

    @torch.no_grad()
    def gen(prompt):
        msgs = [{"role": "system", "content": "You are a precise code-fixing assistant. Output only a search/replace block, no reasoning."},
                {"role": "user", "content": prompt}]
        kw = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
        try:
            enc = tok.apply_chat_template(msgs, enable_thinking=False, **kw).to(dev)   # the real thinking switch
        except TypeError:
            enc = tok.apply_chat_template(msgs, **kw).to(dev)
        n_in = enc["input_ids"].shape[1]
        out = model.generate(**enc, max_new_tokens=512, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, n_in:], skip_special_tokens=True)
    return gen


def _selftest() -> bool:
    print("solution_ladder --selftest: mock leaf that echoes given code (no model)\n")
    ops = [{"name": "op", "realize_hint": "fix it", "_centroid": None}]
    inst = [{"instance_id": "t", "problem_statement": "bug", "patch":
             "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1,1 +1,1 @@\n-return x\n+return x + 1\n"}]

    def mock_gen(prompt):                                     # emits the gold only when handed it (rung c)
        repl = "return x + 1" if "correct fixed code is EXACTLY" in prompt else "return WRONG"
        return f"<<<<<<< SEARCH\nreturn x\n=======\n{repl}\n>>>>>>> REPLACE"
    out, n = run_ladder(inst, ops, mock_gen, log=lambda *a: None)
    print(f"  a={out['a_issue'][0]:.0%} b={out['b_plan'][0]:.0%} c={out['c_gold'][0]:.0%}")
    assert out["c_gold"][0] == 1.0 and out["a_issue"][0] < 1.0, "c (given gold) must be perfect, a must not"
    print("\n  SOLUTION LADDER SELFTEST -> PASS")
    return True


def main():
    ap = argparse.ArgumentParser(description="Solution ladder: emission vs derivation vs traversal-value.")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--dataset", default="lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--ops", default="data/swe/discovered_ops_emb.jsonl")
    ap.add_argument("--dump", default="artifacts/ladder_dump.txt", help="raw generations for the first 6 instances")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if _selftest() else 1)
    from v5.graph_grower.swe_load import load_instances
    ops = [json.loads(l) for l in Path(a.ops).read_text(encoding="utf-8").splitlines() if l.strip()]
    insts = list(load_instances(name=a.dataset, split=a.split, limit=a.n))
    print(f"[ladder] {len(insts)} instances, model={a.model}")
    _report(*run_ladder(insts, ops, real_gen_fn(a.model), dump=a.dump))


if __name__ == "__main__":
    main()
