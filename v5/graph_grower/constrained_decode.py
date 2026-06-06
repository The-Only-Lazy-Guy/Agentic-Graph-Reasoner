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
from v5.graph_grower.swe_probe import load_node_texts, _symbol_name


def load_traces_rich(paths: Sequence[str]) -> Dict[str, dict]:
    """instance_id -> {issue, support_ids, retrieved_ids}. retrieved_ids (the brief's
    top-k, which mixes support + non-support) gives realistic DISTRACTORS for the
    selection test -- without them every candidate is gold and top1 is trivially 1.0."""
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
                out[iid] = {"issue": v.get("task", ""),
                            "support_ids": v.get("support_ids") or [],
                            "retrieved_ids": (v.get("brief") or {}).get("retrieved_ids") or []}
    return out


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


def make_inpatch_processor(tok, symbols: Sequence[str], prompt_len: int, bias: float = 4.0):
    """In-patch constrained decoding: a LogitsProcessor that (a) BIASES the first token of
    any brief symbol up (nudge the model to reach for a grounded symbol), and (b) once a
    symbol is started, FORCES the trie completion so it emits a VALID brief symbol (never a
    half-hallucinated one); on completion it exits -> free generation resumes. Combine with
    injection: injection picks the region/form, this guarantees the exact symbol tokens (§3.5)."""
    import torch
    from transformers import LogitsProcessor

    seqs = []
    for s in symbols:
        for variant in (" " + s, s):                       # BPE: leading-space + bare
            ids = tuple(tok.encode(variant, add_special_tokens=False))
            if ids:
                seqs.append(ids)
    trie = build_trie(seqs)
    first_ids = sorted({s[0] for s in seqs})

    class _InPatch(LogitsProcessor):
        def __init__(self):
            self.first = torch.tensor(first_ids, dtype=torch.long)

        def __call__(self, input_ids, scores):
            node = None                                     # re-derive symbol-walk from suffix
            for t in input_ids[0, prompt_len:].tolist():
                if node is None:
                    node = trie.get(t)
                    if node is not None and node.get("__end__") and len(node) == 1:
                        node = None                         # single-token symbol -> done
                else:
                    node = node.get(t)
                    if node is not None and node.get("__end__"):
                        node = None                         # reached a valid end -> exit
            if node is not None:                            # inside a symbol -> force valid child
                children = [k for k in node if k != "__end__"]
                m = torch.full_like(scores, float("-inf"))
                if children:
                    idx = torch.tensor(children, device=scores.device, dtype=torch.long)
                    m[0, idx] = scores[0, idx]
                return m
            scores[0, self.first.to(scores.device)] += bias   # free -> nudge a symbol start
            return scores

    return _InPatch()


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

    import random
    rng = random.Random(0)
    traces = load_traces_rich(args.traces)
    id2text = load_node_texts(args.nodes)
    insts = {t["instance_id"] for t in load_instances(args.dataset, args.split, limit=0)}
    ids = [i for i in traces if i in insts][: args.limit]
    # global distractor pool of symbol names (fallback when the brief has no extras)
    pool = sorted({n for txt in id2text.values() if (n := _symbol_name(txt))})
    print(f"instances={len(ids)} | distractor pool={len(pool)} | model={args.model}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = load_frozen_lm(args.model, device=device); model.eval()

    hit = 0; scored = 0; valid = 0; rows = []
    for k, iid in enumerate(ids):
        t = traces[iid]
        gold = {_symbol_name(id2text[s]) for s in t["support_ids"] if s in id2text} - {""}
        # candidates = retrieved brief (support + non-support) + sampled distractors
        cands = {_symbol_name(id2text[s]) for s in t.get("retrieved_ids", []) if s in id2text} - {""}
        cands |= gold
        if len(cands - gold) == 0 and pool:                 # brief had no distractors -> sample
            cands |= set(rng.sample(pool, min(8, len(pool))))
        non_gold = sorted((cands - gold) - {""})[:15]       # cap distractors (1 fwd each)
        cands = sorted(set(non_gold) | gold)
        if not gold or len(cands) < 2 or not (set(cands) - gold):
            continue   # need a real choice with at least one distractor
        cands_str = "\n".join(f"  - {c}" for c in cands)
        pick = constrained_select(model, tok, PROMPT.format(issue=t["issue"][:2000], cands=cands_str), cands)
        scored += 1
        in_set = pick in cands
        valid += 1 if in_set else 0           # constrained -> should ALWAYS be in set
        ok = pick in gold
        hit += 1 if ok else 0
        rows.append({"id": iid, "pick": pick, "gold": sorted(gold), "hit": ok,
                     "n_cand": len(cands), "n_gold": len(gold)})
        print(f"  [{k+1}/{len(ids)}] {iid:28} {len(cands):2}cand pick={pick:24} {'HIT' if ok else 'miss'}", flush=True)

    rand = sum(r["n_gold"] / r["n_cand"] for r in rows) / max(1, len(rows))
    res = {"scored": scored, "valid_in_set": valid, "top1_hit": hit,
           "valid_rate": round(valid / max(1, scored), 4),
           "top1_acc": round(hit / max(1, scored), 4),
           "random_baseline": round(rand, 4), "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\nconstrained select: {scored} scored | valid-in-set {res['valid_rate']} (=1.0 by construction) "
          f"| top1 {res['top1_acc']} vs random {res['random_baseline']} "
          f"-> {'LIFTS' if res['top1_acc'] > res['random_baseline'] else 'NO lift'} (gated weak-4B picks right symbol)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
