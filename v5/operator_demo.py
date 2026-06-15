"""LEARNING EVALUATION SUITE — does a frozen model LEARN from one graph edit? (the INTFAIR keystone)

Different from every prior test (which used logit-belief forced-choice): this grades ACTUAL greedy
GENERATION by verifiable string-match. The model must literally OUTPUT a different, correct answer
after we add one graph node — weights never change.

Trustworthiness = the control matrix per item. A flip only counts as LEARNING if ALL hold:
  COLD          : model is WRONG before            (precheck: target NOT generated cold -> no contrived before)
  LEARN         : add the node (operator) -> target generated
  RANDOM        : an IRRELEVANT node (same op) -> target NOT generated   (kills "any injection works")
  SPECIFICITY   : an unrelated question + the node -> unrelated answer UNCHANGED  (kills collateral / "says target everywhere")
We also report RAG (node text in the prompt) for contrast (operator vs plain context).
Injection is per-query, so "revert" == COLD by construction (no persistent state) -> COLD is the revert proof.

Three learn-types map the BOUNDARY (the diagnosis "what is the model missing"):
  A correction (INVALIDATE) : avoid a wrong reasoning path
  B value      (ASSERT/SLOT): recall an injected value
  C new fact   (ASSERT)     : a specific fact the model lacks  <- the content-injection stress test

  4B (L26, A40): V5_LM_TRUST_REMOTE_CODE=1 python -m v5.operator_demo --layer 26 --alpha 1.0
"""
from __future__ import annotations

import argparse
import contextlib
import os
import re
import torch

from v5.lm_loader import load_frozen_lm
from v5.operator_injector import OperatorInjector

RANDOM_NODE = "Bananas are a good source of dietary potassium."   # irrelevant control node
SPEC_Q = "The capital city of France is"                         # unrelated specificity probe
SPEC_TARGET = "paris"

# each item: question (ends where the answer begins), target (correct, graded), node (the graph edit),
# op (operator kind), wrong (the cold mistake, for display). target chosen to be checkable + short.
ITEMS = [
    # --- A: correction (INVALIDATE a tempting wrong answer). Precheck filters the ones the 4B gets cold. ---
    dict(t="A-correction", q="A lily patch doubles in size every day and covers the whole lake in 48 "
         "days. It covered exactly half the lake on day number", target="47",
         node="It is half covered on day 24.", op="INVALIDATE"),
    dict(t="A-correction", q="A farmer has 15 sheep and all but 8 of them run away. The number of sheep "
         "the farmer still has is", target="8",
         node="The farmer has 7 sheep left.", op="INVALIDATE"),
    dict(t="A-correction", q="In a foot race you just overtook the runner in second place. The place "
         "you are now in is", target="second",
         node="You are now in first place.", op="INVALIDATE"),

    # --- B: value (ASSERT a value -> recall it). Values chosen so the model can't guess them cold. ---
    dict(t="B-value", q="The secret access number is", target="58",
         node="The secret access number is 58.", op="ASSERT"),
    dict(t="B-value", q="The meeting is scheduled for room number", target="42",
         node="The meeting is in room 42.", op="ASSERT"),
    dict(t="B-value", q="The project codename is the word", target="falcon",
         node="The project codename is Falcon.", op="ASSERT"),

    # --- C: new fact (ASSERT a fact the model cannot know) ---
    dict(t="C-fact", q="The capital city of the country Zorbia is", target="quint",
         node="The capital of Zorbia is Quint.", op="ASSERT"),
    dict(t="C-fact", q="The chemical element zentium has the symbol", target="zx",
         node="Zentium has the chemical symbol Zx.", op="ASSERT"),
    dict(t="C-fact", q="The novel Whispering Tides was written by an author named", target="voss",
         node="Whispering Tides was written by Mara Voss.", op="ASSERT"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--layer", type=int, default=26)
    ap.add_argument("--alpha", type=float, default=0.4)   # low: injected every gen step, compounds -> over-steers high
    ap.add_argument("--ntok", type=int, default=6)
    ap.add_argument("--layers", default=None, help="comma list -> multi-layer injection (adapter fix)")
    a = ap.parse_args()
    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    from transformers import AutoTokenizer
    model = load_frozen_lm(a.model); model.eval()
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=trust)
    if a.layers:
        from v5.operator_injector_ml import OperatorInjectorML
        inj = OperatorInjectorML(model, tok, [int(x) for x in a.layers.split(",")], a.alpha)
    else:
        inj = OperatorInjector(model, tok, a.layer, a.alpha)
    dev = next(model.parameters()).device
    print(f"loaded | layer {a.layer} | alpha {a.alpha} | {len(ITEMS)} items | gen {a.ntok} tok greedy\n", flush=True)

    @torch.no_grad()
    def gen(prompt, v=None):
        with (inj.inject(v) if v is not None else contextlib.nullcontext()):
            enc = tok(prompt, return_tensors="pt").to(dev)
            out = model.generate(**enc, max_new_tokens=a.ntok, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()

    def has(text, target):
        n = re.sub(r"[^a-z0-9.]", "", text.lower())
        return re.sub(r"[^a-z0-9.]", "", target.lower()) in n

    results = {}
    for it in ITEMS:
        q, tgt, op = it["q"], it["target"], it["op"]
        v_learn = inj.combine([(it["node"], op)], q, normalize=True)
        v_rand = inj.combine([(RANDOM_NODE, op)], q, normalize=True)
        g_cold = gen(q)
        g_learn = gen(q, v_learn)
        g_rand = gen(q, v_rand)
        g_rag = gen(f"{it['node']}\n{q}")
        # specificity: same node-operator injected on an unrelated question
        v_spec = inj.combine([(it["node"], op)], SPEC_Q, normalize=True)
        g_spec_cold = gen(SPEC_Q)
        g_spec = gen(SPEC_Q, v_spec)

        cold_wrong = not has(g_cold, tgt)                 # precheck: genuinely wrong before
        learned = has(g_learn, tgt)
        rand_clean = not has(g_rand, tgt)                 # control: random node doesn't cause it
        # specificity: injecting this node must NOT break an unrelated answer the model knows cold.
        spec_cold_ok = has(g_spec_cold, SPEC_TARGET)
        spec_ok = (not spec_cold_ok) or has(g_spec, SPEC_TARGET)   # N/A if model can't answer it cold
        rag = has(g_rag, tgt)
        passed = cold_wrong and learned and rand_clean and spec_ok
        results.setdefault(it["t"], []).append((passed, cold_wrong))
        flags = f"{'PASS' if passed else 'fail'} [cold_wrong={int(cold_wrong)} learned={int(learned)} " \
                f"rand_clean={int(rand_clean)} spec_ok={int(spec_ok)} | rag={int(rag)}]"
        print(f"[{it['t']}] target='{tgt}'  {flags}")
        print(f"    COLD  : {g_cold[:50]!r}")
        print(f"    LEARN : {g_learn[:50]!r}")
        print(f"    RANDOM: {g_rand[:50]!r}   RAG: {g_rag[:40]!r}")
        print(f"    SPEC  : cold {g_spec_cold[:30]!r} -> +node {g_spec[:30]!r} (want '{SPEC_TARGET}' preserved)\n", flush=True)

    print("=== learning by type (PASS = cold-wrong & learned & random-clean & specificity-ok) ===")
    for t, rs in results.items():
        valid = [r for r in rs if r[1]]   # only items that were genuinely wrong cold
        p = sum(1 for r in rs if r[0])
        print(f"  {t:13} {p}/{len(rs)} learned   ({len(valid)}/{len(rs)} valid cold-wrong prechecks)")
    print("\n  DIAGNOSE: which types learn (A correction / B value) vs not (C fact) -> what the model is missing.")


if __name__ == "__main__":
    main()
