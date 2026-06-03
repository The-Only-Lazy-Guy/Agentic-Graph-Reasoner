"""§12 brief-vs-cold probe, TIER 1 (adherence, ECHO-HARDENED) — does the graph brief
change what the weak 4B actually DOES, beyond parroting the symbols it was handed?

v1 was confounded: sym_cov jumped 0.23->0.91 but the brief literally CONTAINS those
symbols, so the lift was mostly echo. v2 fixes the measurement:
  - few-shot diff format in the prompt -> the 4B emits real patches (raise is_diff).
  - symbols only count if they appear on an actual `+`/`-` DIFF LINE (being edited),
    not anywhere in prose (echo-resistant).
  - HELD-OUT support: 1/3 of the gold support symbols are NEVER put in the brief.
    `held_cov` = did those hidden symbols show up in the edit? That CANNOT be echo
    (they weren't shown) -> the genuine "did the brief steer the model toward the right
    neighborhood" signal. `edit_cov` = all support on diff lines (echo-resistant-ish).

Still adherence != correctness (the truth is Tier-2 test-pass on a Docker box). This is
the free GPU-only gate before that spend. Oracle brief (gold support). See V5_V2_DESIGN §12.

  V5_LM_QUANT=4bit V5_LM_TRUST_REMOTE_CODE=1 python -m v5.graph_grower.swe_probe \
    --dataset lite --limit 30 --model Qwen/Qwen3.5-4B
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from v5.graph_grower.swe_load import load_instances, patch_files
from v5.graph_grower.swe_verify import write_predictions


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
    out: Dict[str, dict] = {}
    for tp in paths:
        for line in Path(tp).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            v = r.get("v2_grounding") or {}
            iid = r.get("instance_id") or v.get("instance_id")
            if iid and v.get("support_ids"):
                out[iid] = {"issue": v.get("task", ""), "support_ids": v.get("support_ids") or []}
    return out


def _symbol_name(sig: str) -> str:
    m = re.search(r"\b(?:def|class)\s+(\w+)", sig)
    if m:
        return m.group(1)
    m = re.match(r"\s*([A-Za-z_]\w*)", sig)
    return m.group(1) if m else ""


COLD_SYS = (
    "You are fixing a bug in a Python project. Produce a minimal unified diff patch that "
    "fixes the reported issue. Output ONLY the diff, in exactly this format:\n"
    "--- a/path/to/file.py\n+++ b/path/to/file.py\n@@ -10,7 +10,7 @@\n"
    "     unchanged line\n-    old line\n+    new line\n     unchanged line"
)


def _user_prompt(issue: str, brief_sigs: List[str]) -> str:
    s = f"ISSUE:\n{issue[:2500]}\n\n"
    if brief_sigs:
        s += "RELEVANT CODE (the fix likely touches these symbols):\n"
        s += "\n".join(f"  - {x}" for x in brief_sigs[:12]) + "\n\n"
    return s + "Output ONLY the unified diff patch."


def _extract_patch(gen: str) -> str:
    """Pull a clean unified diff out of the model's output (fenced block or first
    diff marker) so swebench's `git apply` has a chance. Weak outputs -> empty/garbage
    (counted unresolved, honest)."""
    g = gen or ""
    m = re.search(r"```(?:diff|patch)?\s*\n(.*?)```", g, re.S)
    if m:
        g = m.group(1)
    m = re.search(r"^(?:diff --git .*|--- a/.*)$", g, re.M)
    return (g[m.start():].rstrip() + "\n") if m else ""


def _diff_lines(gen: str) -> str:
    return "\n".join(l for l in (gen or "").splitlines()
                     if l[:1] in "+-" and not l.startswith(("+++", "---")))


def _adherence(gen: str, gold_files: List[str], all_syms: List[str],
               held_syms: List[str]) -> dict:
    g = gen or ""
    dl = _diff_lines(g)
    file_hits = sum(1 for f in gold_files if Path(f).name and Path(f).name in g)
    edit_hits = sum(1 for s in all_syms if s and re.search(rf"\b{re.escape(s)}\b", dl))
    held_hits = sum(1 for s in held_syms if s and re.search(rf"\b{re.escape(s)}\b", dl))
    return {"file_cov": file_hits / max(1, len(gold_files)),
            "edit_cov": edit_hits / max(1, len(all_syms)),                       # echo-resistant
            "held_cov": (held_hits / len(held_syms)) if held_syms else None,     # echo-FREE
            "is_diff": 1.0 if re.search(r"^@@ |^\+\+\+ ", g, re.M) else 0.0}


def _mean(rows: List[dict], key: str) -> Optional[float]:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def main(argv=None) -> int:
    import torch
    from transformers import AutoTokenizer
    from v5.lm_loader import load_frozen_lm
    ap = argparse.ArgumentParser(description="§12 brief-vs-cold adherence probe (echo-hardened).")
    ap.add_argument("--traces", nargs="+", default=["data/swe/grounded_traces.jsonl"])
    ap.add_argument("--nodes", nargs="+", default=["artifacts/graph_growth/swe_code_candidates.jsonl"])
    ap.add_argument("--dataset", default="lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--out", default="artifacts/graph_growth/cloud_results/swe_probe.json")
    ap.add_argument("--emit-predictions", default="",
                    help="dir to write cold_predictions.jsonl + brief_predictions.jsonl "
                         "(swebench format) -> move to a Docker box for Tier-2 swe_verify")
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
    model = load_frozen_lm(args.model, device=device); model.eval()
    rng = random.Random(0)

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
    cold_preds, brief_preds = {}, {}
    for k, iid in enumerate(ids):
        t = traces[iid]; inst = insts[iid]
        gold_files = patch_files(inst.get("patch", ""))
        pairs = [(s, id2text[s]) for s in t["support_ids"] if s in id2text]
        if not pairs:
            continue
        # hold out 1/3 of support from the brief (echo-free target)
        held_ids = set()
        if len(pairs) >= 2:
            held_ids = {s for s, _ in rng.sample(pairs, max(1, len(pairs) // 3))}
        brief_sigs = [txt for s, txt in pairs if s not in held_ids]
        all_syms = [n for _, txt in pairs if (n := _symbol_name(txt))]
        held_syms = [n for s, txt in pairs if s in held_ids and (n := _symbol_name(txt))]

        cold = gen(t["issue"], [])
        brief = gen(t["issue"], brief_sigs)
        cold_preds[iid] = _extract_patch(cold); brief_preds[iid] = _extract_patch(brief)
        cr = _adherence(cold, gold_files, all_syms, held_syms)
        br = _adherence(brief, gold_files, all_syms, held_syms)
        cold_rows.append(cr); brief_rows.append(br)
        hc = "n/a" if br["held_cov"] is None else f"{cr['held_cov']:.2f}->{br['held_cov']:.2f}"
        print(f"  [{k+1}/{len(ids)}] {iid:28} is_diff {cr['is_diff']:.0f}->{br['is_diff']:.0f} "
              f"edit {cr['edit_cov']:.2f}->{br['edit_cov']:.2f} held {hc}", flush=True)

    metrics = ("file_cov", "edit_cov", "held_cov", "is_diff")
    res = {"instances": len(cold_rows),
           "cold": {m: _mean(cold_rows, m) for m in metrics},
           "brief": {m: _mean(brief_rows, m) for m in metrics}}
    res["lift"] = {m: (round(res["brief"][m] - res["cold"][m], 4)
                       if res["brief"][m] is not None and res["cold"][m] is not None else None)
                   for m in metrics}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    if args.emit_predictions:
        d = Path(args.emit_predictions); d.mkdir(parents=True, exist_ok=True)
        nc = write_predictions(cold_preds, str(d / "cold_predictions.jsonl"), "v5_cold")
        nb = write_predictions(brief_preds, str(d / "brief_predictions.jsonl"), "v5_brief")
        print(f"\nemitted predictions -> {d}/  (cold {nc}, brief {nb}) — Tier-2: on a Docker box run\n"
              f"  swe_verify --gold-sanity --dataset {args.dataset} --limit 5\n"
              f"  swe_verify --predictions {d}/cold_predictions.jsonl --dataset {args.dataset} --run-id cold\n"
              f"  swe_verify --predictions {d}/brief_predictions.jsonl --dataset {args.dataset} --run-id brief")
    print("\n=== ADHERENCE (cold -> brief) ===")
    for m in metrics:
        c, b, l = res["cold"][m], res["brief"][m], res["lift"][m]
        tag = "  <- ECHO-FREE" if m == "held_cov" else ("  <- format gate" if m == "is_diff" else "")
        print(f"  {m:9} {c} -> {b}  (lift {l:+}{'' if l is None else ''}){tag}")
    print("\nKEY: held_cov lift = brief surfaces HIDDEN support symbols (echo-free steer);"
          "\n     is_diff = does the 4B even emit a patch. file_cov honest (paths not in brief).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
