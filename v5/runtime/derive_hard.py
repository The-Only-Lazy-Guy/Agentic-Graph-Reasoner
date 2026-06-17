"""Hard DERIVE family + BASELINE = the GAP measurement (before any LoRA/RL).

Multi-step formulas (sum/diff/product/weighted) the model must INFER from natural language, with
DISTRACTOR noise it must ignore — where the frozen 4B SOMETIMES fails. Each derive is scored by the
validated derive_reward (grounded x solves x unique; grounding = recompute the declared formula over
the named upstream, so a fill that picks a distractor number or mis-composes is PUNISHED).

The baseline reward distribution = 'what's missing' (the capability gap a LoRA must close). RL needs
headroom: if the frozen 4B already aces this, there is no training signal — that's the point of measuring.

  baseline (4B, A40):           V5_LM_TRUST_REMOTE_CODE=1 python -m v5.runtime.derive_hard
  harness selftest (no model):  python -m v5.runtime.derive_hard --selftest
"""
from __future__ import annotations

import argparse
import os
import re

from v5.runtime.slot_coder import SlotGraph, SlotSpec, Pool
from v5.runtime.derive_reward import derive_reward, _nums

# (name, {slot: fact}, formula(nums)->int, gold, [distractor noise], natural-language instruction)
HARD = [
    ("total+ship", {"A": "Widget Fizzbolt costs 7 credits", "B": "Widget Glimstone costs 5 credits",
                    "SHIP": "Shipping costs 3 credits"}, lambda n: n["A"] + n["B"] + n["SHIP"], 15,
     ["The store opened in 2019.", "It is located in aisle 12."], "the total cost including shipping"),
    ("subtotal-discount", {"A": "Item Quorblade costs 20 credits", "B": "Item Snarvane costs 14 credits",
                    "DISC": "A discount of 4 credits applies"}, lambda n: n["A"] + n["B"] - n["DISC"], 30,
     ["This is order number 5567.", "The warehouse holds 200 units."], "the subtotal after the discount"),
    ("bulk-product", {"UNIT": "Each Vexrod unit costs 6 credits", "QTY": "The order is for 4 units"},
     lambda n: n["UNIT"] * n["QTY"], 24,
     ["See catalog page 88.", "Rated 5 stars by 30 reviewers."], "the total cost for the full order quantity"),
    ("weighted", {"A": "Part Mirethorn costs 10 credits", "MUL": "You need 2 of part Mirethorn",
                    "B": "Part Drelune costs 5 credits"}, lambda n: n["A"] * n["MUL"] + n["B"], 25,
     ["The SKU is 4471.", "In stock since week 12."], "the total cost for all the parts needed"),
    ("triple-sum-noisy", {"A": "Service Alpha costs 8 credits", "B": "Service Beta costs 9 credits",
                    "C": "Service Gamma costs 6 credits"}, lambda n: n["A"] + n["B"] + n["C"], 23,
     ["Contract year is 2024.", "There are 100 active users.", "This is the tier 3 plan."],
     "the combined cost of all three services"),
]


def _build(task, derive_fn):
    name, facts, formula, gold, noise, instr = task
    graph = [{"id": k, "kind": "fact", "text": v} for k, v in facts.items()]

    def retr(q, kind):
        for g in graph:                                  # upstream slot query is f"[{k}]" -> unambiguous
            if f"[{g['id']}]".lower() in q.lower():
                return [g]
        return []                                        # DERIVE query -> miss -> derive fires

    def fill(slot, ev, pool):
        if slot.name == "DERIVE":
            return ""                                    # derive handles it
        return str(_nums(ev[0]["text"])[0]) if (ev and _nums(ev[0]["text"])) else ""

    specs = [SlotSpec(k, [], "fact", "ASSERT", query=(lambda p, kk=k: f"value [{kk}]")) for k in facts]
    specs.append(SlotSpec("DERIVE", list(facts.keys()), "fact", "ASSERT",
                          query=lambda p: "final computed total", derive=derive_fn))
    return SlotGraph(specs), Pool(specs), retr, fill


def _derive_fn(task, gen):
    name, facts, formula, gold, noise, instr = task

    def derive(slot, pool):                              # TRANSFORM: compute from upstream + ignore noise
        givens = "; ".join(f"{k} = {pool.get(k)}" for k in facts)
        prompt = (f"Given values: {givens}. (Notes, irrelevant: {' '.join(noise)}) "
                  f"Compute {instr}. Use ONLY the given values; ignore the notes. Answer with only the number:")
        n = _nums(gen(prompt))
        return str(n[0]) if n else ""
    return derive


def run_baseline(gen):
    print("HARD DERIVE baseline (the GAP) — frozen model, scored by derive_reward (grounded x solves x unique).\n")
    rows = []
    for task in HARD:
        name, facts, formula, gold, noise, instr = task
        sg, pool, retr, fill = _build(task, _derive_fn(task, gen))
        sg.solve(pool, retr, fill, log=None)
        val = pool.get("DERIVE")
        named = {k: pool.get(k) for k in facts}
        r, b = derive_reward(val, [], gold=gold, upstream_named=named, formula=formula)
        rows.append((name, val, gold, r, b))
        print(f"  [{name:17}] upstream={named} derived={val!r} gold={gold}  reward={r:+.2f}  {b['verdict']}")
    n = len(rows)
    mean = sum(r for _, _, _, r, _ in rows) / n
    halluc = sum(1 for _, _, _, r, _ in rows if r < 0) / n
    solved = sum(1 for _, _, _, _, b in rows if b.get("solved")) / n
    print(f"\n=== BASELINE GAP (frozen model) ===")
    print(f"  mean reward = {mean:+.2f} | hallucination-rate = {halluc:.0%} | solve-rate = {solved:.0%}")
    print("  this distribution = 'what's missing'. Headroom here (mean < +1.5, any hallucination/unsolved)")
    print("  is the signal a LoRA+RL can close. If the frozen model already aces it, pick a harder family.")
    return rows


def _selftest():
    # stub 'model': solves the first 3 tasks, hallucinates (9999) on the last 2 -> verify the harness +
    # reward score them correctly end-to-end (slot-graph routes DERIVE -> reward grades), no GPU.
    def stub_gen(prompt):
        for i, task in enumerate(HARD):
            if task[5] in prompt:                        # match by the instruction
                return str(task[3]) if i < 3 else "9999"
        return "0"
    rows = run_baseline(stub_gen)
    good = all(rows[i][3] > 1.0 for i in range(3)) and all(rows[i][3] < 0 for i in (3, 4))
    print(f"\n  HARNESS SELFTEST -> {'PASS' if good else 'FAIL'}  (first 3 solved -> rewarded; last 2 hallucinated -> punished)")
    return good


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="no-model harness proof (stub model)")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--max-new", type=int, default=16)
    a = ap.parse_args()
    if a.selftest:
        import sys
        sys.exit(0 if _selftest() else 1)

    from v5.lm_loader import load_frozen_lm
    from transformers import AutoTokenizer
    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    model = load_frozen_lm(a.model); model.eval()
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=trust)
    dev = next(model.parameters()).device

    def gen(prompt):
        import torch
        msgs = [{"role": "user", "content": prompt}]
        kw = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
        try:
            enc = tok.apply_chat_template(msgs, enable_thinking=False, **kw).to(dev)
        except TypeError:
            enc = tok.apply_chat_template(msgs, **kw).to(dev)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        return re.sub(r"<think>.*?</think>", "", tok.decode(out[0, enc["input_ids"].shape[1]:],
                      skip_special_tokens=True), flags=re.DOTALL).strip()

    run_baseline(gen)


if __name__ == "__main__":
    main()
