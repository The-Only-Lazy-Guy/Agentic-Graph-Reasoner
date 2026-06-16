"""Operator-grounded coder — the OPERATOR injection in the real SWE loop (was: collapsed trained adapter).

grounded_coder/verifier_retry run the multi-step SWE loop (retrieve -> read source -> emit SR patch ->
apply-feedback retry) but inject via GraphAttentionInjector (the trained adapter that COLLAPSED to
generic). This swaps in the validated OperatorInjector: the SOURCE goes in context (RAG, for exact
code tokens), and a learned STRATEGY / failure_pattern node is injected as a typed operator at L26
(ASSERT an insight, INVALIDATE a pitfall) — grounded on the issue. Code-reasoning showed this beats
RAG of the same node (content-specific, +3.52 vs -5.27). Here we test it end-to-end on SWE.

  --strategy-mode operator : inject the retrieved strategy as a typed operator (the new path)
  --strategy-mode rag      : put the same strategy TEXT in the prompt (the contrast)
  --strategy-mode none     : source-only (no strategy) baseline
Metric = applyable@1 / applyable@k (does the SR patch match the file; verifier-pluggable). Frozen LM.

  V5_LM_TRUST_REMOTE_CODE=1 python -m v5.runtime.operator_coder --n-eval 15 --strategy-mode operator
"""
from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path

import numpy as np
import torch

from v5.lm_loader import load_frozen_lm
from v5.operator_injector import OperatorInjector
from v5.operator_schema import op_kind_for
from v5.training.providers import RealEmbedder
from v5.graph_grower.swe_load import load_instances, checkout_repo
from v5.graph_grower.swe_probe import load_traces, _symbol_name
from v5.runtime.search_replace import SR_SYS, parse_sr, apply_sr
from v5.runtime.sr_withcode import load_symbol_meta, read_body, _file_text
from v5.runtime.verifier_retry import _user, load_strategy_meta


def _unit(v):
    a = np.asarray(v, dtype=np.float32)
    return a / (np.linalg.norm(a) + 1e-9)


def _unmatched(blocks, dest):
    return [b for b in blocks if (b.get("search") or "").strip()
            and (b.get("search") or "").strip() not in _file_text(dest, b.get("file"))]


@torch.no_grad()
def _gen_op(model, tok, msgs, op_inj, v, max_new, dev):
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(dev)
    with (op_inj.inject(v) if v is not None else contextlib.nullcontext()):
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def solve_op(model, tok, op_inj, v, issue, src, dest, max_retries, max_new, dev, rag_extra=""):
    """SR retry loop (apply-feedback), generating through the operator vector v (None = no inject)."""
    feedback, blocks, hist = "", [], []
    for attempt in range(max_retries):
        user = _user(issue, (src + ("\n\n" + rag_extra if rag_extra else "")), feedback)
        gen = _gen_op(model, tok, [{"role": "system", "content": SR_SYS},
                                   {"role": "user", "content": user}], op_inj, v, max_new, dev)
        blocks = parse_sr(gen)
        um = _unmatched(blocks, dest)
        applyable = bool(blocks) and not um
        hist.append({"attempt": attempt + 1, "n_blocks": len(blocks), "applyable": applyable})
        if applyable:
            return True, attempt + 1, blocks, hist
        if not blocks:
            feedback = "You produced NO valid search/replace block. Output ONLY blocks in the exact format."
        else:
            b = um[0]
            feedback = (f"Your SEARCH for file '{b['file']}' was NOT found in the source. This text does "
                        f"not exist there:\n{(b.get('search') or '')[:200]}\nCopy from the SOURCE EXACTLY.")
    return False, max_retries, blocks, hist


def run(model_name, traces_p, nodes_p, strategy_nodes, dataset, split, n_eval, max_new, repo_root,
        max_retries, strategy_mode, layer, alpha, strat_topk, src_bodies, src_lines, device_str=None):
    dev = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={dev} | strategy-mode={strategy_mode} | layer={layer} alpha={alpha}", flush=True)
    model = load_frozen_lm(model_name); model.eval()
    from transformers import AutoTokenizer
    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    op_inj = OperatorInjector(model, tok, layer, alpha)
    embedder = RealEmbedder(dev)

    traces = load_traces(traces_p)
    meta = load_symbol_meta(nodes_p)
    insts = {t["instance_id"]: t for t in load_instances(dataset, split, limit=0)}
    ids = [i for i in traces if i in insts][:n_eval]
    strat_meta = load_strategy_meta(strategy_nodes) if strategy_mode != "none" else {}
    # nearest-other-task strategy retrieval (exclude self = no leakage)
    strat_vecs = {s: _unit(embedder.embed_nodes({"q": traces[s]["issue"]})["q"])
                  for s in strat_meta if s in traces}
    print(f"tasks={len(ids)} | symbol meta={len(meta)} | strategy instances={len(strat_meta)}", flush=True)

    def pick_strategy(iid, issue):
        if strategy_mode == "none" or not strat_vecs:
            return []
        qv = _unit(embedder.embed_nodes({"q": issue})["q"])
        top = sorted((s for s in strat_vecs if s != iid),
                     key=lambda s: float(np.dot(qv, strat_vecs[s])), reverse=True)[:strat_topk]
        pool = []
        for s in top:
            pool.extend(strat_meta[s])
        return pool[:4]                       # [(node_id, text, node_type)]

    app1 = appk = scored = 0
    for k, iid in enumerate(ids):
        t = traces[iid]; inst = insts[iid]
        support = [s for s in t["support_ids"] if s in meta]
        if not support:
            continue
        dest = Path(repo_root) / inst["repo"].replace("/", "__")
        ok, _ = checkout_repo(inst["repo"], inst["base_commit"], dest, timeout=1800)
        if not ok:
            print(f"  [{k+1}] {iid} checkout FAILED"); continue
        src = "\n\n".join(f"# {meta[s]['file']}\n{body}" for s in support[:src_bodies]
                          if (body := read_body(str(dest), meta[s]["file"], meta[s]["lineno"], src_lines)))

        strat = pick_strategy(iid, t["issue"])
        v, rag_extra = None, ""
        if strat and strategy_mode == "operator":
            # typed-operator inject the strategy/pitfall nodes, grounded on the issue
            nodes = [(stext, op_kind_for(stype)) for _, stext, stype in strat]
            v = op_inj.combine(nodes, t["issue"][:600], normalize=True)
        elif strat and strategy_mode == "rag":
            rag_extra = "STRATEGY HINTS:\n" + "\n".join(f"- {stext}" for _, stext, _ in strat)

        applyable, attempts, blocks, hist = solve_op(model, tok, op_inj, v, t["issue"], src, str(dest),
                                                     max_retries, max_new, dev, rag_extra=rag_extra)
        scored += 1
        app1 += 1 if (hist and hist[0]["applyable"]) else 0
        appk += 1 if applyable else 0
        import subprocess
        if applyable:
            subprocess.run(["git", "-C", str(dest), "checkout", "--", "."], capture_output=True)
        print(f"  [{k+1}/{len(ids)}] {iid:24} applyable={applyable} in {attempts} "
              f"(a1={'Y' if hist and hist[0]['applyable'] else 'N'}) strat={len(strat)}", flush=True)

    print(f"\n=== OPERATOR-GROUNDED CODER (strategy-mode={strategy_mode}) ===")
    print(f"  applyable@1   {app1}/{scored} ({100*app1/max(1,scored):.0f}%)")
    print(f"  applyable@{max_retries}   {appk}/{scored} ({100*appk/max(1,scored):.0f}%)")
    print(f"  COMPARE across --strategy-mode operator | rag | none: does typed-operator injection of a")
    print(f"  learned strategy beat RAG-of-the-same / no-strategy on applyable? (then add the verifier).")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Operator-grounded SWE coder (OperatorInjector in the loop).")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--traces", nargs="+", default=["data/swe/grounded_traces.jsonl"])
    ap.add_argument("--nodes", nargs="+", default=["artifacts/graph_growth/swe_code_candidates.jsonl"])
    ap.add_argument("--strategy-nodes", nargs="+",
                    default=["artifacts/graph_growth/swe_strategy_candidates_clean.jsonl"])
    ap.add_argument("--strategy-mode", choices=["operator", "rag", "none"], default="operator")
    ap.add_argument("--dataset", default="lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-eval", type=int, default=15)
    ap.add_argument("--max-new", type=int, default=500)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--repo-root", default="data/swe_repos")
    ap.add_argument("--layer", type=int, default=26)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--strat-topk", type=int, default=3)
    ap.add_argument("--src-bodies", type=int, default=6)
    ap.add_argument("--src-lines", type=int, default=70)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    run(a.model, a.traces, a.nodes, a.strategy_nodes, a.dataset, a.split, a.n_eval, a.max_new,
        a.repo_root, a.max_retries, a.strategy_mode, a.layer, a.alpha, a.strat_topk,
        a.src_bodies, a.src_lines, device_str=a.device)


if __name__ == "__main__":
    raise SystemExit(main())
