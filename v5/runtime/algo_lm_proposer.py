"""GRR-12: the LM as PROPOSER — language understanding where blind search hit its wall.

Measured motivation (READ_THIS): beam+epsilon saturates at 15/24 factory families REGARDLESS of budget
(4x budget, 2x epsilon, 14 rounds = same ceiling). The 9 resisters are deep programs whose 5-prefix must
survive 4 consecutive beam prunings (~5e-5/search) — but their TASK TEXTS ("the sum of the digit-reversal
of the square of the odd numbers...") are compositionally PARSEABLE. A language model reads the text and
EMITS candidate pipelines directly; every candidate passes the SAME _is_general gate (fuzz-general on two
disjoint input sets, MDL-first ordering). Same epistemics as search: the LM sees the task TEXT ONLY —
never the reference pipeline, never the oracle.

The escalation ladder (wired in algo_grr_loop):
    decode (1 verify)  ->  beam+epsilon (budget)  ->  LM PROPOSER (this module)  ->  give up this round
A proposer hit is consolidated + banked EXACTLY like a search find, with origin="lm" — the provenance ->
reuse table becomes three-way (beam / epsilon / lm). If the LM cracks a family once, the TRM amortizes it
and the LM is never needed again for that family: LM as expensive one-time teacher, graph+TRM as the
cheap permanent memory — the architecture thesis in one experiment.

The selftest uses a STUB "LM" that genuinely parses the task text back into a pipeline (template
inversion — the same job we claim a real LM does), so the ladder mechanics are proven end-to-end with no
GPU and no leak. The real-LM path (make_hf_gen) is the molab command.

  selftest (no LM):   python -m v5.runtime.algo_lm_proposer --selftest
  molab (real LM):    python -m v5.runtime.algo_grr_loop --loop --factory --lm Qwen/Qwen2.5-3B ...
"""
from __future__ import annotations

import argparse
import re
import sys

from v5.runtime.algo_dsl import Op

GEN_AGGS = ["sum", "max", "count", "len"]


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt + parse — the strict interchange format
# ═══════════════════════════════════════════════════════════════════════════════

def proposer_prompt(task_text: str, pred_names, map_names, sketch: dict | None = None) -> str:
    """Task TEXT + grammar + atom vocabulary -> ask for candidate pipelines. Contains NO reference
    pipeline and NO oracle — the LM must parse the language. `sketch` (GRR-16) is the TRM's
    confidence-gated THOUGHT rendered as text — atoms it points at + its (unverified) draft — phrased
    SOFT: the ads lesson says never hand the LM memory with authority it can't check."""
    hint = ""
    if sketch and (sketch.get("atoms") or sketch.get("draft")):
        lines = []
        if sketch.get("atoms"):
            lines.append("relevant atoms (with confidence): "
                         + ", ".join(f"{a} ({p:.2f})" for a, p in sketch["atoms"]))
        if sketch.get("draft"):
            lines.append("draft pipeline (UNVERIFIED, may be wrong): "
                         + " -> ".join(f"{k}({v})" for k, v in sketch["draft"]))
        hint = ("A small memory model that has solved related tasks suggests (use ONLY if it helps; "
                "it may be wrong):\n  " + "\n  ".join(lines) + "\n")
    return (
        "You translate a task description into a small program PIPELINE with this exact grammar:\n"
        "  PIPE: FILTER(<pred>) -> ... -> MAP(<fn>) -> ... -> REDUCE(<agg>)\n"
        "Rules: zero or more FILTER steps first (each keeps elements where <pred> is true), then one or\n"
        "more MAP steps (applied in the order written: the FIRST MAP is applied FIRST), then exactly one\n"
        "terminal REDUCE. 'the X of the Y' means apply Y first, then X.\n"
        f"Predicates: {', '.join(sorted(pred_names))}\n"
        f"Map functions: {', '.join(sorted(map_names))}\n"
        f"Aggregators: {', '.join(GEN_AGGS)} (count/len = how many elements survive the filters)\n"
        "Example task: 'the count of prime numbers in the list.'\n"
        "Example answer: PIPE: FILTER(is_prime) -> REDUCE(count)\n"
        + hint +
        f"Task: {task_text}\n"
        "Give up to 4 different candidate PIPE lines, most likely first. Answer with PIPE lines only."
    )


def parse_pipelines(text: str, pred_names, map_names) -> list:
    """Parse 'PIPE:' lines -> validated Op pipelines (unknown atoms / bad structure dropped; dedup,
    order kept). Robust to junk around the lines."""
    preds, maps = set(pred_names), set(map_names)
    out, seen = [], set()
    for line in text.splitlines():
        m = re.search(r"PIPE\s*:\s*(.+)", line)
        if not m:
            continue
        steps, ok = [], True
        for tok in m.group(1).split("->"):
            sm = re.match(r"\s*(FILTER|MAP|REDUCE)\s*\(\s*([A-Za-z_]\w*)\s*\)\s*$", tok)
            if not sm:
                ok = False; break
            kind, arg = sm.group(1), sm.group(2)
            if (kind == "FILTER" and arg not in preds) or (kind == "MAP" and arg not in maps) \
                    or (kind == "REDUCE" and arg not in GEN_AGGS):
                ok = False; break
            steps.append(Op(kind, arg))
        if not ok or not steps or steps[-1].kind != "REDUCE" \
                or any(s.kind == "REDUCE" for s in steps[:-1]):
            continue
        # canonical order guard: FILTERs must precede MAPs (realizer semantics)
        kinds = [s.kind for s in steps[:-1]]
        if "FILTER" in kinds and "MAP" in kinds and kinds.index("MAP") < len(kinds) - 1 - kinds[::-1].index("FILTER"):
            continue
        sig = tuple((s.kind, s.arg) for s in steps)
        if sig not in seen:
            seen.add(sig); out.append(steps)
    return out


def propose_and_verify(gen_fn, task_text: str, fam: str, is_general, pred_names, map_names,
                       k: int = 6, max_verify: int = 16, sketch: dict | None = None):
    """One ladder rung: prompt the LM (k samples), parse, MDL-order (shortest first), verify each through
    the SAME gate until a general hit. Returns (pipe|None, n_verified)."""
    prompt = proposer_prompt(task_text, pred_names, map_names, sketch=sketch)
    cands = []
    for out in gen_fn([prompt] * k):
        cands.extend(parse_pipelines(out, pred_names, map_names))
    # dedup across samples, MDL order: shortest pipelines first (ties: sample order)
    seen, uniq = set(), []
    for p in cands:
        sig = tuple((s.kind, s.arg) for s in p)
        if sig not in seen:
            seen.add(sig); uniq.append(p)
    uniq.sort(key=len)
    used = 0
    for pipe in uniq[:max_verify]:
        used += 1
        if is_general(pipe, fam):
            return pipe, used
    return None, used


# ═══════════════════════════════════════════════════════════════════════════════
# Real-LM generation (molab) — batched sampling on the GPU that finally gets used
# ═══════════════════════════════════════════════════════════════════════════════

def make_hf_gen(model_name: str, temperature: float = 0.8, max_new_tokens: int = 160):
    """gen_fn(prompts)->texts backed by a real HF causal LM (lazy-loaded once). Batches all prompts in
    one padded generate call."""
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=trust)
    model.eval()

    def gen(prompts):
        msgs = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                        add_generation_prompt=True) for p in prompts]
        enc = tok(msgs, return_tensors="pt", padding=True, padding_side="left").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, do_sample=True, temperature=temperature, top_p=0.95,
                                 max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
        return [tok.decode(out[i, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                for i in range(len(prompts))]

    return gen


# ═══════════════════════════════════════════════════════════════════════════════
# Stub "LM" — genuinely PARSES the task text (template inversion). Proves the ladder without a GPU and
# without leaks: it reads only the prompt, exactly like the real LM.
# ═══════════════════════════════════════════════════════════════════════════════

def make_stub_gen():
    from v5.runtime.algo_dsl_gen import _PHRASE, _SYN_AGG, _SYN_PHRASE, GEN_ATOMS
    # phrase -> (role, atom); longest-first so "perfect-square" wins over "square"
    vocab = {}
    for a, (_t, _c, _f, role) in GEN_ATOMS.items():
        for ph in [_PHRASE[a]] + _SYN_PHRASE.get(a, []):
            vocab[ph.lower()] = (role, a)
    phrases = sorted(vocab, key=len, reverse=True)
    aggw = {}
    for agg in ("sum", "max"):
        for w in _SYN_AGG[agg] + ([_PHRASE.get(agg)] if agg in _PHRASE else []):
            if w:
                aggw[w.lower()] = agg
    aggw.update({"the sum": "sum", "the largest value": "max"})
    for w in set(_SYN_AGG["count"] + _SYN_AGG["len"] + ["the count", "the number"]):
        aggw[w.lower()] = "count"

    def gen(prompts):
        outs = []
        for p in prompts:
            m = re.search(r"^Task:\s*(.+)$", p, re.M)
            if not m:
                outs.append(""); continue
            # the 'needs a, b.' hint lists raw ATOM NAMES ("is_odd" contains "odd") -> strip before
            # scanning, and match phrases only at word boundaries ("the numbers" must not hit the
            # count-word "the number"; "perfect-square" must not yield a bare "square")
            text = re.sub(r"\bneeds [^.]*\.", "", m.group(1).lower())
            found, taken = [], []
            for ph in phrases:
                for mm in re.finditer(rf"(?<![a-z_-]){re.escape(ph)}(?![a-z_-])", text):
                    i, j = mm.span()
                    if not any(i < e and s < j for s, e in taken):
                        taken.append((i, j)); found.append((i, vocab[ph]))
            found.sort()
            preds = sorted({a for _i, (r, a) in found if r == "pred"})
            maps_in_text = [a for _i, (r, a) in found if r == "map"]
            # 'the X of the Y' templates read outermost-first -> reverse; 'apply X then Y' is forward
            chain = maps_in_text if "apply" in text else list(reversed(maps_in_text))
            agg = "sum"
            for w in sorted(aggw, key=len, reverse=True):
                if re.search(rf"(?<![a-z_-]){re.escape(w)}(?![a-z_-])", text):
                    agg = aggw[w]; break
            if agg == "count":
                chain = []
            pipe = " -> ".join([f"FILTER({a})" for a in preds] + [f"MAP({a})" for a in chain]
                               + [f"REDUCE({agg})"])
            outs.append(f"PIPE: {pipe}")
        return outs

    return gen


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST — parse robustness, stub-parses-text, and the FULL ladder in the loop: families the beam
# cannot crack get discovered via the proposer, banked with origin="lm", and rebuilt from the graph.
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    import numpy as np
    from v5.runtime.algo_dsl_gen import GEN_ATOMS, gen_families, pipe_is_general, pipe_text_variants
    print("algo_lm_proposer --selftest: parse + stub-LM text-parsing + the full ladder in the loop\n")
    preds = [a for a, v in GEN_ATOMS.items() if v[3] == "pred"]
    maps = [a for a, v in GEN_ATOMS.items() if v[3] == "map"]

    # [1] parser: valid lines in, junk/invalid dropped, canonical order enforced, dedup
    txt = ("noise\nPIPE: FILTER(is_odd) -> MAP(square) -> REDUCE(sum)\n"
           "PIPE: MAP(bogus_atom) -> REDUCE(sum)\nPIPE: MAP(square) -> FILTER(is_odd) -> REDUCE(max)\n"
           "PIPE: FILTER(is_odd) -> MAP(square) -> REDUCE(sum)\nPIPE: REDUCE(count)\n")
    ps = parse_pipelines(txt, preds, maps)
    assert len(ps) == 2 and len(ps[0]) == 3 and len(ps[1]) == 1, ps
    print("  [1] parse: 2 valid kept (dup + unknown atom + MAP-before-FILTER dropped) -> PASS")

    # [1b] GRR-16 sketch hint: present iff given, SOFT phrasing, absent when confidence-gated empty
    sk = dict(atoms=[("is_odd", 0.62), ("square", 0.31)], draft=[("FILTER", "is_odd"), ("REDUCE", "sum")])
    p_h = proposer_prompt("t", preds, maps, sketch=sk)
    assert "may be wrong" in p_h and "is_odd (0.62)" in p_h and "FILTER(is_odd) -> REDUCE(sum)" in p_h
    assert proposer_prompt("t", preds, maps, sketch=dict(atoms=[], draft=None)) == \
        proposer_prompt("t", preds, maps)
    print("  [1b] sketch hint: rendered SOFT with confidences; empty sketch -> identical prompt -> PASS")

    # [2] the stub LM parses REAL factory texts back to verifying pipelines (text only, no reference)
    fams = gen_families(16, seed=3, max_chain=4)
    stub = make_stub_gen()
    chk = lambda p, f: pipe_is_general(p, fams, f, n=24)
    solved = 0
    for f, pipe in fams.items():
        for text in pipe_text_variants(pipe, 3):
            got, _n = propose_and_verify(stub, text, f, chk, preds, maps, k=1)
            if got:
                solved += 1
                break
    assert solved >= 0.75 * len(fams), (solved, len(fams))
    print(f"  [2] stub LM parses task TEXT -> verified pipeline for {solved}/{len(fams)} families "
          f"(incl. depths beam struggles with) -> PASS")

    # [3] THE LADDER: loop with a beam too weak for deep fams + the proposer as the next rung ->
    #     lm-origin discoveries land in the graph, provenance shows the third row, rebuild recalls them
    from v5.runtime.algo_grr_loop import factory_domain, rebuild_net, wake_sleep_loop
    import json
    import tempfile
    from pathlib import Path
    from graph_core import MemoryGraph
    rng = np.random.default_rng(0)
    d_in = 64
    dom = factory_domain(n_families=12, fam_seed=3, para_train=2, para_eval=1, beam=6, max_chain=4,
                         explore=2)
    pipes = gen_families(12, seed=3, max_chain=4)
    fam_base = {f: rng.standard_normal(d_in).astype("float32") for f in pipes}
    text2fam = {t: f for f, p in pipes.items() for t in pipe_text_variants(p, 6)}

    def embed(d):
        out = {}
        for k, text in d.items():
            f = text2fam.get(text)
            out[k] = ((fam_base[f] if f in fam_base else 0.05 * rng.standard_normal(d_in))
                      + 0.15 * rng.standard_normal(d_in)).astype("float32")
        return out

    with tempfile.TemporaryDirectory() as td:
        gp = str(Path(td) / "g.json")
        dom["seed_graph"](gp)
        _m, hist = wake_sleep_loop(gp, embed, rounds=3, budget=120, sft_steps=600, n_wake=2, seed=0,
                                   domain=dom, lm_gen=stub, lm_k=2)
        g = MemoryGraph.load_json(gp)
        origins = {n.metadata.get("family"): n.metadata.get("origin")
                   for n in g.nodes.values() if n.metadata.get("kind") == "program"}
        n_lm = sum(1 for o in origins.values() if o == "lm")
        assert n_lm >= 3, origins                      # the proposer cracked fams the weak beam couldn't
        assert hist[-1][3] >= 10, hist[-1]             # near-full zero-shot coverage with the ladder
        fz, nf = rebuild_net(gp, embed, seed=7, domain=dom, log=False)
        assert fz >= 10, (fz, nf)
        print(f"  [3] ladder in the loop: {n_lm} families banked with origin=lm (beam budget 120 alone "
              f"couldn't), zero-shot {hist[-1][3]}/{hist[-1][4]} fams, rebuild {fz}/{nf} -> PASS")

    print("\n  ALGO_LM_PROPOSER SELFTEST -> PASS  (LM proposes from language, the gate verifies, the "
          "graph remembers — the LM is a one-time teacher)")
    return True


def main():
    ap = argparse.ArgumentParser(description="GRR-12: LM proposer (text -> candidate pipelines, gated).")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
