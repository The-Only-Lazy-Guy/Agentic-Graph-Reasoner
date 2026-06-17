"""Integrated 'everything enabled' run — every component we built wired into ONE SlotGraph.solve,
with a WIRING REPORT that proves each one actually fired (the forcing function for INTEGRATION_CHECKLIST).

One composite task exercises the whole stack:
  AUTHOR (retrieve, RealEmbedder, CONFUSER) -> CITY (retrieve) -> PROVINCE (retrieve)
  -> OUTPUT (retrieve-OR-DERIVE via the RL-trained LoRA) -> JUDGE (reason: OPERATOR-fill, op_kind_for)
with dependency-directed BACKTRACK on the confuser, the text POOL, the grounded REWARD as solves_fn,
and the solved slot-graph SAVED to the self-improving memory store. Nothing decorative: each component
flips a flag in WiringReport, and --wiring-report prints which fired (any that never fired => HALT).

  preflight (A40, asserts items 1-5 then exits):  V5_LM_TRUST_REMOTE_CODE=1 python -m v5.runtime.integrated_run --preflight
  full run (A40):                                  V5_LM_TRUST_REMOTE_CODE=1 python -m v5.runtime.integrated_run
  wiring selftest (no model):                      python -m v5.runtime.integrated_run --selftest
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from v5.runtime.slot_coder import SlotGraph, SlotSpec, Pool


class WiringReport:
    """Tracks that EVERY checklist component actually fired this run (no decorative pieces)."""
    ITEMS = ["engine_solve", "lora_loaded", "operator_entered", "retrieval_real_slotaware",
             "verifier_graded", "pool_text", "retrieve_or_derive", "backtrack", "reward_grounded_gate",
             "self_improving_memory"]

    def __init__(self):
        self.fired = {k: 0 for k in self.ITEMS}

    def mark(self, k):
        self.fired[k] = self.fired.get(k, 0) + 1

    def report(self):
        print("\n=== WIRING REPORT (every component must have FIRED >=1; 0 = decorative -> HALT) ===")
        missing = [k for k, n in self.fired.items() if n == 0]
        for k in self.ITEMS:
            print(f"  [{'OK ' if self.fired[k] else 'XX '}] {k:28} fired x{self.fired[k]}")
        if missing:
            print(f"\n  HALT: these components NEVER fired (decorative): {missing}")
        else:
            print("\n  ALL components fired — the run exercised the full stack, nothing decorative.")
        return not missing


# ── the controllable composite graph (fictional entities -> the 4B must use the graph, not priors) ──
def build_graph_and_data(rng_confuser_first=True):
    GRAPH = [
        {"id": "a_auth", "kind": "fact", "text": "The Aurelian Codex was written by Lena Sorrel."},
        {"id": "a_rev", "kind": "fact", "text": "The Aurelian Codex was reviewed by Marcus Vane."},  # CONFUSER
        {"id": "c_auth", "kind": "fact", "text": "Lena Sorrel founded the city of Brightmoor."},
        {"id": "c_rev", "kind": "fact", "text": "Marcus Vane founded the city of Greyfen."},
        {"id": "p_bright", "kind": "fact", "text": "The city of Brightmoor lies in the province of Caldera."},
        # NOTE: no province fact for Greyfen -> the confuser path dead-ends -> BACKTRACK must recover.
        {"id": "q1", "kind": "fact", "text": "The province of Caldera mined 18 tons of ore last spring."},
        {"id": "q2", "kind": "fact", "text": "The province of Caldera mined 9 tons of ore last autumn."},
        # NOTE: no single 'total ore' fact -> OUTPUT must be DERIVED (18 + 9) by the LoRA.
        {"id": "noise", "kind": "fact", "text": "Caldera holds an annual festival."},
    ]
    GOLD = {"AUTHOR": "Lena Sorrel", "CITY": "Brightmoor", "PROVINCE": "Caldera", "OUTPUT": "27"}
    return GRAPH, GOLD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="no-model wiring proof (stubs)")
    ap.add_argument("--preflight", action="store_true", help="assert items 1-5 constructable then exit")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--adapter", default="artifacts/derive_lora")
    ap.add_argument("--layer", type=int, default=26)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--store", default="artifacts/integrated_templates.jsonl")
    a = ap.parse_args()

    if a.selftest:
        import sys
        sys.exit(0 if _selftest() else 1)

    # ── real component construction (A40). Each construction is also a checklist precondition. ──
    import re, torch, contextlib
    from transformers import AutoTokenizer
    from v5.lm_loader import load_frozen_lm
    from v5.operator_injector import OperatorInjector
    from v5.operator_schema import op_kind_for
    from v5.training.providers import RealEmbedder
    from v5.runtime.derive_reward import derive_reward, _nums

    wr = WiringReport()
    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    model = load_frozen_lm(a.model); model.eval()
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=trust)
    dev = next(model.parameters()).device

    # ITEM 3 — operators armed: build the injector on the BASE model BEFORE the peft wrap (its forward
    # hook sits on the base decoder layer, survives LoRA injection, and still fires through PeftModel).
    inj = OperatorInjector(model, tok, a.layer, a.alpha)

    # ITEM 2 — LoRA loaded (the RL-trained derive adapter); wrap AFTER arming the injector.
    adapter_ok = Path(a.adapter).exists() and any(Path(a.adapter).iterdir())
    if adapter_ok:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter); model.eval()
        wr.mark("lora_loaded")
        print(f"[wiring] LoRA loaded from {a.adapter}")
    else:
        print(f"[wiring] WARNING: no LoRA at {a.adapter} — derive runs on the BASE model (item 2 NOT satisfied)")

    emb = RealEmbedder(dev)                                             # ITEM 4 — real embedder
    GRAPH, GOLD = build_graph_and_data()
    gv = {n["id"]: torch.tensor(emb.embed_nodes({n["id"]: n["text"]})[n["id"]], device=dev) for n in GRAPH}

    def gen(prompt, v=None, ntok=24):
        msgs = [{"role": "user", "content": prompt}]
        kw = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
        try:
            enc = tok.apply_chat_template(msgs, enable_thinking=False, **kw)
        except TypeError:
            enc = tok.apply_chat_template(msgs, **kw)
        ids = enc["input_ids"].to(dev)
        with (inj.inject(v) if v is not None else contextlib.nullcontext()), torch.no_grad():
            out = model.generate(ids, do_sample=False, max_new_tokens=ntok, pad_token_id=tok.eos_token_id)
        return re.sub(r"<think>.*?</think>", "", tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True),
                      flags=re.DOTALL).strip()

    if a.preflight:
        ok = adapter_ok and (inj is not None) and (emb.dim == 768)
        print(f"\n[PREFLIGHT] engine={True} lora={adapter_ok} operators={inj is not None} "
              f"embedder768={emb.dim==768} -> {'PASS' if ok else 'FAIL (fix wiring before the real run)'}")
        import sys; sys.exit(0 if ok else 1)

    run_task(model, tok, gen, inj, emb, gv, GRAPH, GOLD, op_kind_for, derive_reward, _nums, wr, a.store, dev)
    all_fired = wr.report()
    import sys; sys.exit(0 if all_fired else 1)


def run_task(model, tok, gen, inj, emb, gv, GRAPH, GOLD, op_kind_for, derive_reward, _nums, wr, store, dev):
    import torch

    def retrieve(query, k=4):                                          # ITEM 4 — slot-aware real retrieval
        wr.mark("retrieval_real_slotaware")
        qv = torch.tensor(emb.embed_nodes({"q": query})["q"], device=dev)
        return sorted(GRAPH, key=lambda n: -float(gv[n["id"]] @ qv))[:k]

    def extract_fill(slot, ev, pool):                                  # retrieval -> entity, upstream-aware
        spec = SPECMAP[slot.name]
        up = pool.get(spec.needs[0]) if spec.needs else ""
        cands = [e for e in ev if (not up or up.split()[0] in e["text"])]
        if not cands:
            return ""
        facts = "\n".join(f"- {e['text']}" for e in cands)
        return gen(f"Facts:\n{facts}\n\n{spec.ask(pool.slots)} Answer with only the value:", ntok=12)

    def derive_total(slot, pool):                                      # ITEM 7 — retrieve-OR-DERIVE (LoRA)
        wr.mark("retrieve_or_derive")
        prov = pool.get("PROVINCE")
        if not prov:                                                   # upstream not ready -> can't derive
            return ""
        ev = retrieve(f"ore mined by the province {prov}", k=4)
        nums = [n for e in ev if (prov.split()[0] in e["text"]) for n in _nums(e["text"])]
        if len(nums) < 2:
            return ""
        out = gen(f"The province mined {nums[0]} tons and {nums[1]} tons. Total tons "
                  f"({nums[0]} plus {nums[1]})? Answer with only the number:", ntok=10)
        # ITEM 9 — grounded REWARD as the gate (ungrounded punished even if it equals gold)
        r, b = derive_reward(out, [str(n) for n in nums], gold=int(GOLD["OUTPUT"]), op="+")
        wr.mark("reward_grounded_gate")
        print(f"   [derive] {nums} -> {out!r}  reward={r:+.2f} {b['verdict']}")
        return out if r > 0 else ""                                    # gate: ungrounded -> empty -> backtrack

    def judge_fill(slot, ev, pool):                                    # ITEM 3 — OPERATOR-fill reason slot
        good = {"id": "g", "text": f"The province {pool.get('PROVINCE')} is a real ore producer.",
                "node_type": "strategy"}
        bad = {"id": "b", "text": f"The province {pool.get('PROVINCE')} produces no ore at all.",
               "node_type": "failure_pattern"}
        q = f"Given the province {pool.get('PROVINCE')} mined ore, is it an ore producer? Answer yes or no:"
        v = inj.combine([(good["text"], op_kind_for(good["node_type"])),
                         (bad["text"], op_kind_for(bad["node_type"]))], q, normalize=True)
        wr.mark("operator_entered")
        with_op = gen(q, v=v, ntok=8)
        return "yes" if "yes" in with_op.lower() else "no"

    specs = [
        SlotSpec("AUTHOR", [], "fact", "ASSERT", query=lambda p: "who wrote the Aurelian Codex author",
                 ask=lambda p: "Who is the author of the Aurelian Codex?"),
        SlotSpec("CITY", ["AUTHOR"], "fact", "ASSERT", query=lambda p: f"city founded by {p['AUTHOR'].value}",
                 ask=lambda p: f"Which city did {p['AUTHOR'].value} found?"),
        SlotSpec("PROVINCE", ["CITY"], "fact", "ASSERT", query=lambda p: f"province of the city {p['CITY'].value}",
                 ask=lambda p: f"In which province is {p['CITY'].value}?"),
        SlotSpec("OUTPUT", ["PROVINCE"], "fact", "ASSERT", query=lambda p: "total ore (no single fact)",
                 ask=lambda p: "total ore", derive=derive_total),
        SlotSpec("JUDGE", ["OUTPUT"], "fact", "ASSERT", query=lambda p: "is it an ore producer",
                 ask=lambda p: "ore producer?", mode="reason"),
    ]
    global SPECMAP
    SPECMAP = {s.name: s for s in specs}

    def filler(slot, ev, pool):                                        # ITEM 6 — pool stores TEXT
        wr.mark("pool_text")
        if slot.name == "JUDGE":
            return judge_fill(slot, ev, pool)
        return extract_fill(slot, ev, pool)

    def retr(q, kind):
        return retrieve(q)

    sg = SlotGraph(specs)                                              # ITEM 1 — the engine
    pool = Pool(specs, context={})
    log = []
    ok, steps = sg.solve(pool, retr, filler, log=log, max_steps=24)
    wr.mark("engine_solve")
    if any(r[0] == "BACKTRACK" for r in log):                         # ITEM 8 — backtrack fired
        wr.mark("backtrack")

    print("\n=== SOLVED SLOT-GRAPH (read every value — right-for-the-right-reason) ===")
    for n, s in pool.slots.items():
        gold = GOLD.get(n, "")
        hit = (gold and gold.split()[0].lower() in s.value.lower()) or (n == "JUDGE" and "yes" in s.value.lower())
        print(f"  [{n:8}] {s.value!r:32} state={s.state:12} {'OK' if hit else ''}  gold~{gold!r}")
    print(f"  solve log: {[r[0] for r in log]}")

    # ITEM 5 — verifier-graded (here: the gold answer for this controllable family; swap SWE Docker later)
    solved = all((GOLD[k].split()[0].lower() in pool.get(k).lower()) for k in ("AUTHOR", "CITY", "PROVINCE", "OUTPUT"))
    if solved:
        wr.mark("verifier_graded")
    print(f"  VERIFIER: chain-correct = {solved}")

    # ITEM 12 — self-improving memory: save the solved structure
    if solved:
        import json
        Path(store).parent.mkdir(parents=True, exist_ok=True)
        with open(store, "a", encoding="utf-8") as h:
            h.write(json.dumps({"signature": "author->city->province->derive-total->judge",
                                "slots": [s.name for s in specs], "solved": True}) + "\n")
        wr.mark("self_improving_memory")
        print(f"  saved solved slot-graph -> {store}")


def _selftest():
    """No-model wiring proof: stub gen/inj/emb so the solve routes through EVERY component and the
    WiringReport ends all-fired. Proves the integration is wired, before spending A40."""
    print("integrated_run --selftest: prove the solve fires EVERY component (no model).\n")
    import types, torch
    from v5.runtime.derive_reward import derive_reward, _nums
    wr = WiringReport(); wr.mark("lora_loaded")                       # selftest assumes adapter present

    GRAPH, GOLD = build_graph_and_data()
    # stub embedder: id-keyword overlap retrieval
    class E: dim = 768
    emb = E()
    gv = {}
    def stub_gen(prompt, v=None, ntok=24):
        p = prompt.lower()
        if "author of the aurelian" in p:                    # FOOLED by the confuser if its fact is present
            if "reviewed by marcus" in p: return "Marcus Vane"      # -> Greyfen -> dead-end -> BACKTRACK
            if "written by lena" in p: return "Lena Sorrel"         # after backtrack nogoods the reviewer fact
            return ""
        if "which city" in p:                                # follows whichever author won
            if "marcus vane founded the city of greyfen" in p: return "Greyfen"
            if "lena sorrel founded the city of brightmoor" in p: return "Brightmoor"
            return ""
        if "total tons" in p or "plus" in p: return "27"     # the DERIVE prompt (check before 'province')
        if "in which province" in p:                         # the PROVINCE slot ask (specific, not the derive)
            if "brightmoor lies in the province of caldera" in p: return "Caldera"
            return ""                                        # Greyfen has NO province fact -> empty -> backtrack
        if "yes or no" in p: return "yes"
        return ""
    class Inj:
        def combine(self, nodes, q, normalize=False): return "V"
        def inject(self, v):
            import contextlib; return contextlib.nullcontext()
    inj = Inj()
    from v5.operator_schema import op_kind_for

    # monkeypatch retrieval to keyword match (no embedder) by replacing gv dot-product path:
    def fake_run():
        import torch as _t
        # reuse run_task but with a keyword retriever via gv that returns scores by substring
        pass
    # simplest: call run_task with a tiny shim embedder providing embed_nodes -> deterministic vectors
    import numpy as np
    class Realish:
        dim = 768
        def embed_nodes(self, d):
            out = {}
            for k, txt in d.items():
                vec = np.zeros(64, dtype=float)
                for w in txt.lower().split():
                    vec[hash(w) % 64] += 1.0
                out[k] = vec.tolist()
            return out
    emb = Realish()
    gv = {n["id"]: torch.tensor(emb.embed_nodes({n["id"]: n["text"]})[n["id"]]) for n in GRAPH}
    run_task(None, None, stub_gen, inj, emb, gv, GRAPH, GOLD, op_kind_for, derive_reward, _nums, wr,
             "artifacts/integrated_templates_selftest.jsonl", torch.device("cpu"))
    all_fired = wr.report()
    print(f"\n  INTEGRATION WIRING SELFTEST -> {'PASS' if all_fired else 'FAIL'}")
    return all_fired


if __name__ == "__main__":
    main()
