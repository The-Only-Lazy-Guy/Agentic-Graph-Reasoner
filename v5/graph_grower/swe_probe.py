"""§12 brief-vs-cold probe, TIER 1 (adherence) — does the graph brief change what the
weak 4B actually DOES? No Docker / no test execution: the 4B generates a patch for each
issue COLD (issue only) vs WITH BRIEF (the gold support symbols' signatures), and we
measure how much the output overlaps the gold-touched files + support symbols.

Adherence != correctness, but it is the cheap FIRST gate: if the brief doesn't even move
which files/symbols the model touches, real test-pass won't help either -> v2 dies cheap,
no Docker needed. If adherence lifts -> stand up the Docker verifier (swe_verify) for the
real Tier-2 test-pass number.

This is the ORACLE-brief upper bound (gold support, not retrieved) — the best case for
grounding. If even a perfect brief doesn't steer the model, stop. Reuses grounded_traces
(issue + support_ids), code candidates (id->signature), and gold patches (touched files).
GPU box; Qwen3.5-4B 4-bit via lm_loader. See V5_V2_DESIGN §12 / READ_THIS.

  V5_LM_QUANT=4bit V5_LM_TRUST_REMOTE_CODE=1 python -m v5.graph_grower.swe_probe \
    --traces data/swe/grounded_traces.jsonl \
    --nodes artifacts/graph_growth/swe_code_candidates.jsonl \
    --dataset lite --limit 30 --model Qwen/Qwen3.5-4B
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence

from v5.graph_grower.swe_load import load_instances, patch_files


def load_node_texts(paths: Sequence[str]) -> Dict[str, str]:
    id2text: Dict[str, str] = {}
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            e = json.loads(line).get("raw_edit", json.loads(line))
            if e.get("op") == "add_node" and e.get("node_id"):
                id2text.setdefault(e["node_id"], e.get("text", "") or "")
    return id2text


def load_traces(paths: Sequence[str]) -> Dict[str, dict]:
    """instance_id -> {issue, support_ids} from grounded_traces."""
    out: Dict[str, dict] = {}
    for tp in paths:
        for line in Path(tp).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            v = r.get("v2_grounding") or {}
            iid = r.get("instance_id") or v.get("instance_id")
            if iid and (v.get("support_ids")):
                out[iid] = {"issue": v.get("task", ""), "support_ids": v.get("support_ids") or []}
    return out


def _symbol_name(sig: str) -> str:
    m = re.search(r"\b(?:def|class)\s+(\w+)", sig)
    if m:
        return m.group(1)
    m = re.match(r"\s*([A-Za-z_]\w*)", sig)
    return m.group(1) if m else ""


COLD_SYS = ("You are fixing a bug in a Python project. Produce a minimal unified diff "
            "patch (--- a/file +++ b/file with @@ hunks) that fixes the reported issue. "
            "Output ONLY the diff.")


def _user_prompt(issue: str, brief_sigs: List[str]) -> str:
    s = f"ISSUE:\n{issue[:2500]}\n\n"
    if brief_sigs:
        s += "RELEVANT CODE (the fix likely touches these symbols):\n"
        s += "\n".join(f"  - {x}" for x in brief_sigs[:12]) + "\n\n"
    return s + "Output ONLY a unified diff patch."


def _adherence(gen: str, gold_files: List[str], gold_symbols: List[str]) -> dict:
    g = gen or ""
    file_hits = sum(1 for f in gold_files if Path(f).name and Path(f).name in g)
    sym_hits = sum(1 for s in gold_symbols if s and re.search(rf"\b{re.escape(s)}\b", g))
    looks_diff = bool(re.search(r"^\+\+\+ |^@@ |^--- ", g, re.M))
    return {"file_cov": file_hits / max(1, len(gold_files)),
            "sym_cov": sym_hits / max(1, len(gold_symbols)),
            "is_diff": 1.0 if looks_diff else 0.0}


def _mean(rows: List[dict], key: str) -> float:
    return round(sum(r[key] for r in rows) / max(1, len(rows)), 4)


def main(argv=None) -> int:
    import torch
    from transformers import AutoTokenizer
    from v5.lm_loader import load_frozen_lm
    ap = argparse.ArgumentParser(description="§12 brief-vs-cold adherence probe (no Docker).")
    ap.add_argument("--traces", nargs="+", default=["data/swe/grounded_traces.jsonl"])
    ap.add_argument("--nodes", nargs="+", default=["artifacts/graph_growth/swe_code_candidates.jsonl"])
    ap.add_argument("--dataset", default="lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--out", default="artifacts/graph_growth/cloud_results/swe_probe.json")
    args = ap.parse_args(argv)

    traces = load_traces(args.traces)
    id2text = load_node_texts(args.nodes)
    insts = {t["instance_id"]: t for t in load_instances(args.dataset, args.split, limit=0)}
    ids = [i for i in traces if i in insts][: args.limit]
    print(f"probe instances={len(ids)} | model={args.model}", flush=True)
    if not ids:
        print("no instances overlap traces+dataset"); return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = load_frozen_lm(args.model, device=device)
    model.eval()

    def gen(issue: str, sigs: List[str]) -> str:
        msgs = [{"role": "system", "content": COLD_SYS},
                {"role": "user", "content": _user_prompt(issue, sigs)}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                                  pad_token_id=tok.pad_token_id or tok.eos_token_id)
        return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

    cold_rows, brief_rows = [], []
    for k, iid in enumerate(ids):
        t = traces[iid]; inst = insts[iid]
        gold_files = patch_files(inst.get("patch", ""))
        sigs = [id2text[s] for s in t["support_ids"] if s in id2text]
        gold_syms = [_symbol_name(s) for s in sigs if _symbol_name(s)]
        cold = gen(t["issue"], [])
        brief = gen(t["issue"], sigs)
        cold_rows.append(_adherence(cold, gold_files, gold_syms))
        brief_rows.append(_adherence(brief, gold_files, gold_syms))
        print(f"  [{k+1}/{len(ids)}] {iid:30} cold sym {cold_rows[-1]['sym_cov']:.2f} "
              f"-> brief {brief_rows[-1]['sym_cov']:.2f}", flush=True)

    res = {"instances": len(ids),
           "cold": {m: _mean(cold_rows, m) for m in ("file_cov", "sym_cov", "is_diff")},
           "brief": {m: _mean(brief_rows, m) for m in ("file_cov", "sym_cov", "is_diff")}}
    res["lift"] = {m: round(res["brief"][m] - res["cold"][m], 4)
                   for m in ("file_cov", "sym_cov", "is_diff")}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print("\n=== ADHERENCE (cold -> brief) ===")
    for m in ("file_cov", "sym_cov", "is_diff"):
        print(f"  {m:9} {res['cold'][m]:.3f} -> {res['brief'][m]:.3f}  (lift {res['lift'][m]:+.3f})")
    print("\nlift>0 on file_cov/sym_cov = the brief STEERS the 4B -> worth the Docker verifier.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
