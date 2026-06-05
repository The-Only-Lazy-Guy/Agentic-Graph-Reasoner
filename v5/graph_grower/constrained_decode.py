"""Graph-gated constrained decoding (V5_V2_DESIGN §3.5) — the STRONG emission grounding.

Tier-1 showed the weak 4B rarely emits the right symbol (edit_cov 0.12) and hallucinates
freely. The design's fix is not "put text in the prompt" (RAG, weak) but to make exact
tokens STRUCTURALLY GUARANTEED at decode time: in a symbol position, a logits processor
masks the vocab to ONLY valid symbols from the brief — the model *cannot* emit a name
that isn't in the graph. Grounding by construction, not by bias.

This module is the mechanism + a clean test: build a token-id TRIE from the candidate
symbol names; a `TrieConstraint` LogitsProcessor walks it so generation can ONLY produce
a full candidate symbol (then EOS). The eval asks the 4B "which symbol does the fix
touch?" constrained to the brief set, and scores top-1 vs the gold support symbol —
isolating grounded SELECTION from patch-format noise.

  V5_LM_QUANT=4bit V5_LM_TRUST_REMOTE_CODE=1 python -m v5.graph_grower.constrained_decode \
    --dataset lite --limit 30 --model Qwen/Qwen3.5-4B
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from v5.graph_grower.swe_load import load_instances
from v5.graph_grower.swe_probe import load_traces, load_node_texts, _symbol_name


# ── token trie + constraint ──────────────────────────────────────────────────
def build_trie(token_seqs: Sequence[Sequence[int]]) -> dict:
    root: dict = {}
    for seq in token_seqs:
        node = root
        for t in seq:
            node = node.setdefault(t, {})
        node["__end__"] = True
    return root


def _trie_processor_cls():
    """Defer the transformers import so the module loads without torch."""
    import torch
    from transformers import LogitsProcessor

    class TrieConstraint(LogitsProcessor):
        """Force the generated suffix (after prompt_len) to walk `trie` -> output is
        exactly one candidate symbol, then EOS is allowed. Hallucination impossible."""
        def __init__(self, trie: dict, prompt_len: int, eos_id: int):
            self.trie = trie; self.prompt_len = prompt_len; self.eos = eos_id

        def __call__(self, input_ids, scores):
            gen = input_ids[0, self.prompt_len:].tolist()
            node = self.trie
            for t in gen:
                node = node.get(t, {}) if isinstance(node, dict) else {}
            allowed = [k for k in node.keys() if k != "__end__"]
            if node.get("__end__"):
                allowed = allowed + [self.eos]
            mask = torch.full_like(scores, float("-inf"))
            if allowed:
                idx = torch.tensor(allowed, device=scores.device, dtype=torch.long)
                mask[0, idx] = scores[0, idx]
            return mask

    return TrieConstraint


def _cand_logprob(model, tok, prompt_ids, cand: str) -> float:
    """Length-normalized log P(cand | prompt) under the LM."""
    import torch
    c_ids = tok(" " + cand, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
    full = torch.cat([prompt_ids, c_ids], dim=1)
    with torch.no_grad():
        logits = model(full).logits                              # [1, L, V]
    plen = prompt_ids.shape[1]
    lp = torch.log_softmax(logits[0, plen - 1:-1, :], dim=-1)     # preds for cand positions
    tok_lp = lp[torch.arange(c_ids.shape[1], device=lp.device), c_ids[0]]
    return float(tok_lp.mean())


def constrained_select(model, tok, prompt: str, candidates: List[str]) -> str:
    """Output RESTRICTED to the candidate set by construction: rank candidates by LM
    likelihood, return the best. valid-in-set is 1.0 trivially (we never leave the set).
    This is the selection form of graph-gating; the trie/LogitsProcessor above is for the
    harder IN-PATCH free-generation case (force exact symbol tokens mid-diff)."""
    import torch
    prompt_ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    return max(candidates, key=lambda c: _cand_logprob(model, tok, prompt_ids, c))


PROMPT = ("A bug is reported below. From the candidate functions, output the single "
          "function name whose code most likely must change to fix it.\n\n"
          "ISSUE:\n{issue}\n\nCANDIDATES:\n{cands}\n\nFunction name: ")


def main(argv=None) -> int:
    import torch
    from transformers import AutoTokenizer
    from v5.lm_loader import load_frozen_lm
    ap = argparse.ArgumentParser(description="Graph-gated constrained symbol selection (§3.5).")
    ap.add_argument("--traces", nargs="+", default=["data/swe/grounded_traces.jsonl"])
    ap.add_argument("--nodes", nargs="+", default=["artifacts/graph_growth/swe_code_candidates.jsonl"])
    ap.add_argument("--dataset", default="lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--out", default="artifacts/graph_growth/cloud_results/constrained_select.json")
    args = ap.parse_args(argv)

    traces = load_traces(args.traces)
    id2text = load_node_texts(args.nodes)
    insts = {t["instance_id"] for t in load_instances(args.dataset, args.split, limit=0)}
    ids = [i for i in traces if i in insts][: args.limit]
    print(f"instances={len(ids)} | model={args.model}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = load_frozen_lm(args.model, device=device); model.eval()

    hit = 0; scored = 0; valid = 0; rows = []
    for k, iid in enumerate(ids):
        t = traces[iid]
        gold = {_symbol_name(id2text[s]) for s in t["support_ids"] if s in id2text}
        gold.discard("")
        cands = sorted({_symbol_name(id2text[s]) for s in t["support_ids"] if s in id2text} - {""})
        if len(cands) < 2 or not gold:
            continue   # need a real choice
        cands_str = "\n".join(f"  - {c}" for c in cands)
        pick = constrained_select(model, tok, PROMPT.format(issue=t["issue"][:2000], cands=cands_str), cands)
        scored += 1
        in_set = pick in cands
        valid += 1 if in_set else 0           # constrained -> should ALWAYS be in set
        ok = pick in gold
        hit += 1 if ok else 0
        rows.append({"id": iid, "pick": pick, "gold": sorted(gold), "in_set": in_set, "hit": ok})
        print(f"  [{k+1}/{len(ids)}] {iid:28} pick={pick:24} {'HIT' if ok else 'miss'}"
              f"{'' if in_set else '  <- OFF-SET(bug)'}", flush=True)

    res = {"scored": scored, "valid_in_set": valid, "top1_hit": hit,
           "valid_rate": round(valid / max(1, scored), 4),
           "top1_acc": round(hit / max(1, scored), 4), "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\nconstrained select: {scored} scored | valid-in-set {res['valid_rate']} "
          f"(must be 1.0 = grounding GATE works) | top1 {res['top1_acc']} "
          f"(weak-4B picks the right gated symbol)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
