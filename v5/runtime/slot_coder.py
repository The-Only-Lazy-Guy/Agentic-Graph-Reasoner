"""Slot-graph reasoning substrate — the collapse-safe core (DESIGN_SLOT_GRAPH.md).

A task = a Task Slot-Graph: slots + DATAFLOW deps + 5 filledness states, solved to a FIXPOINT.
Slot values are stored as TEXT in the task POOL (no latent passing -> no generic-collapse). Each slot
is filled by SLOT-AWARE retrieval (the query is built from the slot's need = role + upstream slot
values, so the retriever knows what the LM needs) -> operator-inject the evidence -> the frozen LM
fills -> write back to the pool. Graph edits / new info re-enter via STALE propagation.

Pieces are mostly reused (retrieval, operators, graph); the new bits are: the Pool, the SlotSpec
(slot-aware query), and the fixpoint solve. Filler + retriever are pluggable (stub for the loop test,
OperatorInjector + a real ranker for the model run).

  demo (loop mechanics, no model):  python -m v5.runtime.slot_coder --demo
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Callable, Dict, List

EMPTY, TENTATIVE, VALID, STALE, INSUFFICIENT = "empty", "tentative", "valid", "stale", "insufficient"


@dataclass
class SlotSpec:
    name: str
    needs: List[str]                       # upstream slot names (DATAFLOW deps)
    evidence_kind: str                     # which node type / pool to retrieve from the big graph
    op: str                                # operator on the retrieved evidence (ASSERT/INVALIDATE/...)
    query: Callable[[Dict[str, "Slot"]], str]   # SLOT-AWARE: build the retrieval query from the pool
    ask: Callable[[Dict[str, "Slot"]], str] = None   # the LM sub-question to fill the slot (real run)
    mode: str = "extract"                  # 'extract' = exact content -> RAG; 'reason' = decision -> OPERATORS


@dataclass
class Slot:
    name: str
    value: str = ""
    state: str = EMPTY
    justification: dict = field(default_factory=dict)   # {upstream:[...], evidence:[...]}


class Pool:
    """The task pool = the working memory. Stores slot values as text (faithful, no collapse)."""
    def __init__(self, specs: List[SlotSpec], context: Dict[str, str] | None = None):
        self.slots: Dict[str, Slot] = {s.name: Slot(s.name) for s in specs}
        self.context = context or {}       # task-level givens (issue, source, ...)

    def get(self, name: str) -> str:
        if name in self.context:
            return self.context[name]
        s = self.slots.get(name)
        return s.value if s and s.state in (VALID, TENTATIVE) else ""

    def store(self, name: str, value: str, just: dict, state: str):
        s = self.slots[name]; s.value, s.justification, s.state = value, just, state


class SlotGraph:
    def __init__(self, specs: List[SlotSpec]):
        self.specs = {s.name: s for s in specs}
        self.order = self._toposort(specs)            # DATAFLOW DAG order (upstream first)

    def _toposort(self, specs):
        done, order = set(), []
        def visit(n):
            if n in done or n not in self.specs:
                return
            for u in self.specs[n].needs:
                visit(u)
            done.add(n); order.append(n)
        for s in specs:
            visit(s.name)
        return order

    def _gate_sufficient(self, pool: Pool, name: str) -> bool:
        # sufficiency = all of this slot's needs are filled valid (the basis is complete)
        return all(pool.slots.get(u, Slot(u)).state == VALID or u in pool.context
                   for u in self.specs[name].needs)

    def solve(self, pool: Pool, retriever: Callable[[str, str], list],
              filler: Callable[[Slot, list, Pool], str], max_steps: int = 24, log: list | None = None):
        """Fixpoint: fill empty/stale/insufficient slots (upstream-first) until all valid+sufficient."""
        for step in range(max_steps):
            target = None
            for name in self.order:                    # topological: fill upstream first
                s = pool.slots[name]
                if s.state in (EMPTY, STALE, INSUFFICIENT):
                    target = name; break
            if target is None:
                if log is not None: log.append(("FIXPOINT", step))
                return True, step
            spec = self.specs[target]
            q = spec.query(pool.slots | {})            # SLOT-AWARE query (knows the need)
            evidence = retriever(q, spec.evidence_kind) # targeted retrieval for THIS slot
            value = filler(pool.slots[target], evidence, pool)   # operator-inject + LM fill
            complete = bool(value) and self._gate_sufficient(pool, target)
            pool.store(target, value, {"q": q, "evidence": [e.get("id") for e in evidence]},
                       VALID if complete else TENTATIVE)
            # PROPAGATE forward: dependents go STALE (their basis changed)
            for name, spec2 in self.specs.items():
                if target in spec2.needs and pool.slots[name].state == VALID:
                    pool.slots[name].state = STALE
            if log is not None:
                log.append((target, value[:40], pool.slots[target].state, q[:40]))
        return False, max_steps

    def invalidate(self, pool: Pool, node_id: str):
        """Graph edit / new info touching node_id -> any slot justified by it goes STALE (+ propagate)."""
        for name, s in pool.slots.items():
            if node_id in (s.justification.get("evidence") or []):
                s.state = STALE


# ── toy task-family to prove the loop mechanics (no model): 2-hop fact reasoning ──
def _toy_demo():
    GRAPH = [   # the "big graph"
        {"id": "f_capital", "kind": "fact", "text": "The capital of Aterra is Vionne."},
        {"id": "f_river", "kind": "fact", "text": "Vionne sits on the river Sel."},
        {"id": "f_len", "kind": "fact", "text": "The river Sel is 900 km long."},
        {"id": "noise1", "kind": "fact", "text": "Bananas contain potassium."},
    ]
    def _ent(text):                          # last proper-noun (Capitalized) word — the key entity
        ws = [w.strip(".,") for w in text.split() if w[:1].isupper()]
        return ws[-1] if ws else ""
    def retriever(query, kind):             # slot-AWARE: best word-overlap with the slot's query
        qw = {w for w in query.lower().split() if len(w) > 3}
        scored = [(len(qw & {w.strip(".,") for w in n["text"].lower().split() if len(w) > 3}), n)
                  for n in GRAPH if n["kind"] == kind]
        return [n for sc, n in sorted(scored, key=lambda x: -x[0]) if sc > 0][:1]
    def filler(slot, evidence, pool):       # stub LM: "reads" the retrieved evidence (operators would inject)
        return evidence[0]["text"] if evidence else ""

    specs = [
        SlotSpec("CAPITAL", [], "fact", "ASSERT", lambda p: "capital of Aterra"),
        SlotSpec("RIVER", ["CAPITAL"], "fact", "ASSERT",
                 lambda p: f"river {_ent(p['CAPITAL'].value)} sits on"),     # query built from CAPITAL
        SlotSpec("LENGTH", ["RIVER"], "fact", "ASSERT",
                 lambda p: f"river {_ent(p['RIVER'].value)} long kilometers"),  # query built from RIVER
    ]
    sg = SlotGraph(specs); pool = Pool(specs, context={"issue": "How long is Aterra's capital's river?"})
    log = []
    ok, steps = sg.solve(pool, retriever, filler, log=log)
    print(f"toy solve: fixpoint={ok} in {steps} steps")
    for row in log:
        print("  ", row)
    print("  POOL:", {n: s.value[:30] for n, s in pool.slots.items()})
    # demonstrate INVALIDATION: edit the graph fact -> the slot that used it goes STALE
    sg.invalidate(pool, "f_len")
    print("  after editing f_len -> LENGTH state:", pool.slots["LENGTH"].state, "(stale = needs re-fill)")


# ── V3: save a solved slot-graph as a GENERALIZED template, reuse it on a 2nd task -> easier? ──
def _ent(text):
    ws = [w.strip(".,") for w in text.split() if w[:1].isupper()]
    return ws[-1] if ws else ""


def _toks(s):                               # punctuation-fair tokens (strip 's and trailing marks)
    out = set()
    for w in s.lower().split():
        w = w.split("'")[0].strip(".,?!;:()")
        if len(w) > 3:
            out.add(w)
    return out


def _mk_retriever(graph):
    def retriever(query, kind):
        qw = _toks(query)
        scored = [(len(qw & _toks(n["text"])), n) for n in graph if n["kind"] == kind]
        return [n for sc, n in sorted(scored, key=lambda x: -x[0]) if sc > 0][:1]
    return retriever


def _filler(slot, evidence, pool):
    return evidence[0]["text"] if evidence else ""


# the GENERALIZED template = entity-agnostic slot structure (parameterised by the task's subject).
def _template_specs(country):
    return [
        SlotSpec("CAPITAL", [], "fact", "ASSERT", lambda p: f"capital of {country}"),
        SlotSpec("RIVER", ["CAPITAL"], "fact", "ASSERT", lambda p: f"river {_ent(p['CAPITAL'].value)} sits on"),
        # query intentionally avoids "long"/"river" (the question's words) -> the answer fact is only
        # reachable by chaining through RIVER's entity, so a single undecomposed retrieval can't shortcut.
        SlotSpec("LENGTH", ["RIVER"], "fact", "ASSERT", lambda p: f"{_ent(p['RIVER'].value)} stretches kilometers distance"),
    ]


def _cold_specs(country, question):                 # NO template: one undecomposed retrieval
    return [SlotSpec("ANSWER", [], "fact", "ASSERT", lambda p: question)]


def _v3_test():
    GRAPH_A = [
        {"id": "a1", "kind": "fact", "text": "The capital of Aterra is Vionne."},
        {"id": "a2", "kind": "fact", "text": "Vionne sits on the river Sel."},
        {"id": "a3", "kind": "fact", "text": "Sel stretches 900 kilometers."},
        {"id": "an", "kind": "fact", "text": "Bananas contain potassium."},
    ]
    GRAPH_B = [   # a DIFFERENT task: new entities, SAME structure
        {"id": "b1", "kind": "fact", "text": "The capital of Bravos is Mira."},
        {"id": "b2", "kind": "fact", "text": "Mira sits on the river Tol."},
        {"id": "b3", "kind": "fact", "text": "Tol stretches 500 kilometers."},
        {"id": "bn", "kind": "fact", "text": "Helium is lighter than air."},
    ]
    TEMPLATE_STORE = {}

    def solve(specs, graph, country, question):
        sg = SlotGraph(specs)
        pool = Pool(specs, context={"issue": question})
        ok, steps = sg.solve(pool, _mk_retriever(graph), _filler, log=None)
        answer = pool.slots[sg.order[-1]].value           # terminal slot
        retrievals = sum(1 for s in pool.slots.values() if s.value)
        return answer, steps, retrievals

    def correct(answer, km):
        return km in answer

    # TASK 1 (Aterra): solve from the (hand-built) structure, then SAVE it generalized.
    ans1, st1, _ = solve(_template_specs("Aterra"), GRAPH_A, "Aterra", "How long is Aterra's capital's river?")
    TEMPLATE_STORE["capital-river-length"] = _template_specs   # save the GENERALIZED template (factory)
    print(f"TASK 1 (Aterra): answer={ans1[:34]!r} correct={correct(ans1,'900')}  -> template SAVED\n")

    Q2 = "How long is Bravos's capital's river?"
    # TASK 2 COLD (no template): one undecomposed retrieval
    a_cold, st_cold, r_cold = solve(_cold_specs("Bravos", Q2), GRAPH_B, "Bravos", Q2)
    # TASK 2 TEMPLATE: retrieve the saved structure, instantiate for Bravos
    factory = TEMPLATE_STORE.get("capital-river-length")
    a_tmpl, st_tmpl, r_tmpl = solve(factory("Bravos"), GRAPH_B, "Bravos", Q2)

    print(f"TASK 2 (Bravos):  '{Q2}'")
    print(f"  COLD     (no template, 1 retrieval): answer={a_cold[:34]!r}  correct={correct(a_cold,'500')}  retrievals={r_cold}")
    print(f"  TEMPLATE (reused slot-graph)       : answer={a_tmpl[:34]!r}  correct={correct(a_tmpl,'500')}  retrievals={r_tmpl} steps={st_tmpl}")
    print(f"\n=== V3 (2nd-task-easier) ===")
    won = correct(a_tmpl, "500") and not correct(a_cold, "500")
    print(f"  template solves the 2nd task: {correct(a_tmpl,'500')} | cold solves it: {correct(a_cold,'500')}")
    print(f"  RESULT: {'PASS — the SAVED reasoning structure makes the 2nd (different) task solvable where cold (no decomposition) fails' if won else 'see numbers'}")
    print("  (toy: structure-transfer mechanism; the real lift = a frozen 4B doing the multi-hop, next phase)")


# ── #8: REAL run — frozen 4B fills each slot via OperatorInjector + real-embedder retrieval ──
# Controlled 4-hop chain over FICTIONAL entities (the 4B has no prior -> must use the graph), with
# VERBOSE per-slot dumps so the result is MANUALLY INSPECTED (no cheap pass): read the retrieved
# evidence + the 4B's actual fill for every slot, verify right-for-the-right-reason.
def _real_run(model_name, layer, alpha, ntok, distractors=0):
    import contextlib, os, random, torch
    from v5.lm_loader import load_frozen_lm
    from v5.operator_injector import OperatorInjector
    from v5.training.providers import RealEmbedder
    from transformers import AutoTokenizer

    GRAPH = [
        {"id": "g1", "kind": "fact", "text": "The Zarnvolt Protocol was authored by Helena Voss."},
        {"id": "g2", "kind": "fact", "text": "Helena Voss founded the city of Kesmir."},
        {"id": "g3", "kind": "fact", "text": "The city of Kesmir lies in the province of Talgrid."},
        {"id": "g4", "kind": "fact", "text": "The province of Talgrid produces a metal called quintsteel."},
        {"id": "n1", "kind": "fact", "text": "The Zarnvolt Protocol has fourteen articles."},
        {"id": "n2", "kind": "fact", "text": "Helena Voss was born in winter."},
        {"id": "n3", "kind": "fact", "text": "Kesmir has a population of two million."},
        {"id": "n4", "kind": "fact", "text": "Talgrid borders the sea."},
    ]
    if distractors:
        # CONFUSER chain: the Zarnvolt Protocol's REVIEWER (not author) leads to a WRONG answer.
        GRAPH += [
            {"id": "c1", "kind": "fact", "text": "The Zarnvolt Protocol was reviewed by Tomas Reln."},
            {"id": "c2", "kind": "fact", "text": "Tomas Reln founded the city of Dremont."},
            {"id": "c3", "kind": "fact", "text": "The city of Dremont lies in the province of Veska."},
            {"id": "c4", "kind": "fact", "text": "The province of Veska produces a metal called ironwood."},
        ]
        # VOLUME distractors: many same-structure facts about unrelated entities (overload the context).
        rng = random.Random(0)
        syl = ["Vor", "Kel", "Tarn", "Mire", "Dol", "Bre", "Quen", "Zad", "Lor", "Fyn", "Grim", "Vex", "Sel", "Oru"]
        nm = lambda: rng.choice(syl) + rng.choice(syl).lower()
        METALS = ["bronzine", "ferrolite", "steelix", "corandium", "palebrass", "ironclad", "greysteel",
                  "duralume", "coppernite", "tinveil", "zincara", "leadolite", "nickelan", "cobalith"]
        for i in range(distractors):
            pr, pe, ci, pv, mt = f"{nm()} Pact", f"{nm()} {nm()}", nm(), nm(), rng.choice(METALS)
            GRAPH += [
                {"id": f"d{i}a", "kind": "fact", "text": f"The {pr} was authored by {pe}."},
                {"id": f"d{i}b", "kind": "fact", "text": f"{pe} founded the city of {ci}."},
                {"id": f"d{i}c", "kind": "fact", "text": f"The city of {ci} lies in the province of {pv}."},
                {"id": f"d{i}d", "kind": "fact", "text": f"The province of {pv} produces a metal called {mt}."},
            ]
    print(f"GRAPH size = {len(GRAPH)} facts (distractors={distractors})\n")
    QUESTION = ("What metal is produced by the home province of the city founded by the author of the "
                "Zarnvolt Protocol?")
    ANSWER = "quintsteel"

    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    model = load_frozen_lm(model_name); model.eval()
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    inj = OperatorInjector(model, tok, layer, alpha)
    dev = next(model.parameters()).device
    emb = RealEmbedder(dev)
    gvecs = emb.embed_nodes({n["id"]: n["text"] for n in GRAPH})
    gtens = {k: torch.tensor(v, device=dev) for k, v in gvecs.items()}

    import re
    def gen(prompt, v=None):
        msgs = [{"role": "user", "content": prompt}]
        kw = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
        try:                                  # Qwen3.5 reasoning model: disable <think> so it answers directly
            enc = tok.apply_chat_template(msgs, enable_thinking=False, **kw).to(dev)
        except TypeError:                     # older template (e.g. Qwen2.5) has no enable_thinking
            enc = tok.apply_chat_template(msgs, **kw).to(dev)
        with (inj.inject(v) if v is not None else contextlib.nullcontext()):
            out = model.generate(**enc, max_new_tokens=ntok, do_sample=False, pad_token_id=tok.eos_token_id)
        txt = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL).strip()   # backup strip
        lines = [ln.strip(" .*\"'`:") for ln in txt.splitlines() if ln.strip(" .*\"'`:")]
        return lines[0] if lines else ""                                       # the entity (first real line)

    def retrieve(query, kind, k=4):          # top-K (not top-1): give the slot a small focused set,
        qv = torch.tensor(emb.embed_nodes({"q": query})["q"], device=dev)   # the LM disambiguates within it
        ranked = sorted(GRAPH, key=lambda n: -float(gtens[n["id"]] @ qv))
        return ranked[:k]

    specs = [
        SlotSpec("AUTHOR", [], "fact", "ASSERT",
                 query=lambda p: "author of the Zarnvolt Protocol",
                 ask=lambda p: "Who is the author of the Zarnvolt Protocol? Answer with only the name:"),
        SlotSpec("CITY", ["AUTHOR"], "fact", "ASSERT",
                 query=lambda p: f"city founded by {p['AUTHOR'].value}",
                 ask=lambda p: f"Which city did {p['AUTHOR'].value} found? Answer with only the city name:"),
        SlotSpec("PROVINCE", ["CITY"], "fact", "ASSERT",
                 query=lambda p: f"province that contains the city {p['CITY'].value}",
                 ask=lambda p: f"In which province does the city {p['CITY'].value} lie? Answer with only the province:"),
        SlotSpec("PRODUCT", ["PROVINCE"], "fact", "ASSERT",
                 query=lambda p: f"metal produced by the province {p['PROVINCE'].value}",
                 ask=lambda p: f"What metal does the province {p['PROVINCE'].value} produce? Answer with only the metal:"),
    ]
    sg = SlotGraph(specs)
    specmap = {s.name: s for s in specs}
    dump = []

    def op_filler(slot, evidence, pool):
        # EXTRACTION slot = exact entity -> the top-K facts go IN CONTEXT (RAG); the LM disambiguates
        # the right relation within the slot's small set (top-1 retrieval propagates confuser errors).
        spec = specmap[slot.name]
        facts = "\n".join(f"- {e['text']}" for e in evidence)
        q = f"Facts:\n{facts}\n\n{spec.ask(pool.slots)}"
        val = gen(q)
        dump.append((slot.name, spec.query(pool.slots), [e["text"] for e in evidence], spec.ask(pool.slots), val))
        return val

    pool = Pool(specs, context={"issue": QUESTION})
    ok, steps = sg.solve(pool, retrieve, op_filler, log=None)

    EXPECT = {"AUTHOR": "voss", "CITY": "kesmir", "PROVINCE": "talgrid", "PRODUCT": "quintsteel"}
    print("="*78)
    print("REAL 4B slot-graph run — 4-hop fictional chain. MANUALLY INSPECT every slot below.\n")
    print(f"QUESTION: {QUESTION}\nGOLD: {ANSWER}\n")
    per_slot_ok = {}
    for name, rq, ev, ask, val in dump:
        good = EXPECT[name] in val.lower()
        per_slot_ok[name] = good
        print(f"[{name}]  per-slot correct={good}  (expect '{EXPECT[name]}')")
        print(f"   retrieval query : {rq}")
        print(f"   retrieved fact  : {ev[0] if ev else '(none)'}")
        print(f"   LM sub-question : {ask}")
        print(f"   4B FILLED       : {val!r}")
        print()
    final = pool.slots["PRODUCT"].value
    chain_ok = all(per_slot_ok.values())
    print(f"PER-SLOT chain: {per_slot_ok}")
    print(f"CHAIN COMPLETE (every hop right) = {chain_ok}   <- the HONEST metric (not just the final slot,")
    print(f"     which the last retrieved fact can fake)")
    print(f"SLOT-CHAIN final (PRODUCT) = {final!r}   final-correct={ANSWER in final.lower()}  (fixpoint={ok} in {steps})")

    # COLD baseline: 4B given the question + ALL facts in context, one shot (RAG, no decomposition).
    cold_prompt = (f"Facts:\n" + "\n".join(f"- {n['text']}" for n in GRAPH) +
                   f"\n\n{QUESTION}\nAnswer with only the metal:")
    cold = gen(cold_prompt, None)
    print(f"\nCOLD-FULL (ALL {len(GRAPH)} facts in context — only possible on a SMALL graph) = {cold!r}   "
          f"correct={ANSWER in cold.lower()}")
    # COLD-CAPPED: the REALISTIC baseline at scale — a big graph can't fit, so ONE retrieval for the
    # whole question (no decomposition). It gets the endpoints but misses the INTERMEDIATE hops.
    cap_ev = retrieve(QUESTION, "fact", k=8)
    cap_prompt = ("Facts:\n" + "\n".join(f"- {e['text']}" for e in cap_ev) +
                  f"\n\n{QUESTION}\nAnswer with only the metal:")
    capped = gen(cap_prompt)
    print(f"\nCOLD-CAPPED (top-8 retrieval, the realistic big-graph baseline) = {capped!r}   "
          f"correct={ANSWER in capped.lower()}")
    print(f"   one-shot retrieved for the question (note the MISSING intermediate hops):")
    for e in cap_ev:
        print(f"      - {e['text']}")
    print(f"\n  VALUE: slot-graph (per-hop retrieval) vs COLD-CAPPED (one-shot retrieval) — both are what")
    print(f"  you can actually run on a graph too big to fit. If the chain holds and cold-capped misses a")
    print(f"  hop -> the DECOMPOSITION earns it. cold-FULL is the unrealistic 'everything fits' ceiling.")
    print("\nINSPECT: did the 4B fill EACH slot from the retrieved fact correctly (right-for-right-reason),")
    print("and did the chain reach the gold? Compare to cold one-shot. Read the actual fills above.")


# ── #14: operators in REASONING slots — operator-fill (INVALIDATE pitfalls / ASSERT insight) vs RAG ──
# Reuses op_kind_for (operator_schema) + OperatorInjector + the proven bare-misconception content.
# A reasoning slot JUDGES an approach; the operator-fill subtracts the wrong path (INVALIDATE), where
# RAG of the same misconception POISONS (optest_shape). This is operator_loop_v2's logic, as a slot-fill.
def _reason_demo(model_name, layer, alpha, ntok):
    import contextlib, os, re, torch
    from v5.lm_loader import load_frozen_lm
    from v5.operator_injector import OperatorInjector
    from v5.operator_schema import op_kind_for
    from transformers import AutoTokenizer

    # (question -> yes/no validity judgement, correct, evidence node, node_type). bare misconception
    # nodes -> op_kind_for=INVALIDATE; a correct-approach node -> ASSERT (the valid control).
    ITEMS = [
        ("A student proves  sum 1/(a(1+b)) >= 3/(1+abc)  by writing 1+b >= 2 sqrt(b), hence "
         "1/(a(1+b)) <= 1/(2a sqrt(b)), then summing. Is this proof VALID? Answer yes or no:", "no",
         "By AM-GM 1+b >= 2 sqrt(b), so 1/(a(1+b)) <= 1/(2a sqrt(b)), and summing proves the lower bound.",
         "failure_pattern"),
        ("A student argues there are infinitely many primes because N = (product of all primes so far) "
         "+ 1 is ITSELF always prime. Is this argument VALID? Answer yes or no:", "no",
         "N = (product of all primes up to p) + 1 is itself a prime not in the list.", "failure_pattern"),
        ("To prove  sum 1/(a(1+b)) >= 3/(1+abc),  a student clears denominators and reduces it to a "
         "polynomial inequality, then applies AM-GM to the polynomial form. Is this approach VALID? "
         "Answer yes or no:", "yes",
         "Clearing denominators to a polynomial inequality and bounding it with AM-GM is a standard, "
         "valid route for this inequality.", "strategy"),
    ]
    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    model = load_frozen_lm(model_name); model.eval()
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    inj = OperatorInjector(model, tok, layer, alpha); dev = next(model.parameters()).device

    def gen(prompt, v=None):
        kw = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
        try:
            enc = tok.apply_chat_template([{"role": "user", "content": prompt}], enable_thinking=False, **kw).to(dev)
        except TypeError:
            enc = tok.apply_chat_template([{"role": "user", "content": prompt}], **kw).to(dev)
        with (inj.inject(v) if v is not None else contextlib.nullcontext()):
            out = model.generate(**enc, max_new_tokens=ntok, do_sample=False, pad_token_id=tok.eos_token_id)
        t = re.sub(r"<think>.*?</think>", "", tok.decode(out[0, enc["input_ids"].shape[1]:],
                   skip_special_tokens=True), flags=re.DOTALL).strip().lower()
        m = re.search(r"\b(yes|no)\b", t)     # first clear yes/no anywhere (model may explain then answer)
        return m.group(1) if m else (t[:18] or "(empty)")

    print("#14 REASONING SLOTS: operator-fill vs RAG (per-slot). MANUALLY INSPECT.\n")
    op_ok = rag_ok = cold_ok = 0
    for q, correct, node, ntype in ITEMS:
        op = op_kind_for(ntype)
        v = inj.combine([(node, op)], q, normalize=True)
        cold, opr, rag = gen(q), gen(q, v), gen(f"Note: {node}\n\n{q}")
        op_ok += (opr == correct); rag_ok += (rag == correct); cold_ok += (cold == correct)
        print(f"[{op:10} expect={correct}]  cold={cold!r}  OPERATOR={opr!r}  RAG={rag!r}  "
              f"{'OP✓' if opr==correct else 'OP✗'}{' RAGpoison' if rag!=correct and op=='INVALIDATE' else ''}")
        print(f"   Q: {q[:70]}")
        print(f"   node({ntype}->{op}): {node[:70]}\n")
    n = len(ITEMS)
    print(f"=== operator-fill {op_ok}/{n} | RAG-fill {rag_ok}/{n} | cold {cold_ok}/{n} ===")
    print("  INSPECT: on the INVALIDATE (misconception) items, does OPERATOR judge 'no' while RAG of the")
    print("  same misconception poisons toward 'yes'? On the ASSERT (valid) item, does OPERATOR keep 'yes'?")
    print("  (operator-fill = operator_loop_v2's edge-gated combine, now as the reasoning-slot fill.)")


# ── #13: REAL self-improving memory — serializable templates, save -> retrieve-by-signature -> instantiate ──
# V3 was a toy dict of lambda-factories. This serializes the SOLVED slot-graph STRUCTURE as string
# templates (placeholders {SUBJECT}, {SLOT}, {SLOT.ent}), saves to a store, retrieves the nearest by
# signature-embedding for a new task, and instantiates it -> the reasoning structure is REUSED.
def _fmt(s, slots, subject):
    out = s.replace("{SUBJECT}", subject)
    for name, slot in slots.items():
        val = getattr(slot, "value", "") or ""
        out = out.replace("{" + name + ".ent}", _ent2(val)).replace("{" + name + "}", val)
    return out


def _ent2(text):
    ws = [w.strip(".,") for w in text.split() if w[:1].isupper()]
    return ws[-1] if ws else text


def instantiate(template, subject):
    """A serialized template + a task subject -> live SlotSpecs (query/ask format the placeholders)."""
    specs = []
    for s in template["slots"]:
        specs.append(SlotSpec(
            s["name"], s["needs"], "fact", s.get("op", "ASSERT"),
            query=(lambda slots, t=s["query"]: _fmt(t, slots, subject)),
            ask=(lambda slots, t=s["ask"]: _fmt(t, slots, subject)),
            mode=s.get("mode", "extract")))
    return specs


def _selfimprove_demo(model_name, layer, alpha, ntok):
    import contextlib, json, os, re, torch
    from pathlib import Path
    from v5.lm_loader import load_frozen_lm
    from v5.operator_injector import OperatorInjector
    from v5.training.providers import RealEmbedder
    from transformers import AutoTokenizer

    # the GENERALIZED reasoning structure (entity-agnostic; {SUBJECT} = the task's document).
    TEMPLATE = {
        "signature": "the metal produced by the home province of the city founded by the author of a document",
        "slots": [
            {"name": "AUTHOR", "needs": [], "query": "author of the {SUBJECT}",
             "ask": "Who is the author of the {SUBJECT}? Answer with only the name:"},
            {"name": "CITY", "needs": ["AUTHOR"], "query": "city founded by {AUTHOR}",
             "ask": "Which city did {AUTHOR} found? Answer with only the city name:"},
            {"name": "PROVINCE", "needs": ["CITY"], "query": "province that contains the city {CITY}",
             "ask": "In which province does the city {CITY} lie? Answer with only the province:"},
            {"name": "PRODUCT", "needs": ["PROVINCE"], "query": "metal produced by the province {PROVINCE}",
             "ask": "What metal does the province {PROVINCE} produce? Answer with only the metal:"},
        ],
    }
    GRAPH_A = [
        {"id": "a1", "text": "The Zarnvolt Protocol was authored by Helena Voss."},
        {"id": "a2", "text": "Helena Voss founded the city of Kesmir."},
        {"id": "a3", "text": "The city of Kesmir lies in the province of Talgrid."},
        {"id": "a4", "text": "The province of Talgrid produces a metal called quintsteel."},
        {"id": "an", "text": "Helena Voss was born in winter."},
    ]
    GRAPH_B = [   # a DIFFERENT task: new document + entities, SAME structure
        {"id": "b1", "text": "The Brunel Accord was authored by Sera Mond."},
        {"id": "b2", "text": "Sera Mond founded the city of Othal."},
        {"id": "b3", "text": "The city of Othal lies in the province of Wessen."},
        {"id": "b4", "text": "The province of Wessen produces a metal called brightsteel."},
        {"id": "bn", "text": "Othal is famous for its bridges."},
    ]
    STORE = Path("artifacts/slot_templates.jsonl")

    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    model = load_frozen_lm(model_name); model.eval()
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    OperatorInjector(model, tok, layer, alpha)               # (hooks installed; extract slots use RAG)
    dev = next(model.parameters()).device
    emb = RealEmbedder(dev)

    def gen(prompt):
        kw = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
        try:
            enc = tok.apply_chat_template([{"role": "user", "content": prompt}], enable_thinking=False, **kw).to(dev)
        except TypeError:
            enc = tok.apply_chat_template([{"role": "user", "content": prompt}], **kw).to(dev)
        out = model.generate(**enc, max_new_tokens=ntok, do_sample=False, pad_token_id=tok.eos_token_id)
        t = re.sub(r"<think>.*?</think>", "", tok.decode(out[0, enc["input_ids"].shape[1]:],
                   skip_special_tokens=True), flags=re.DOTALL).strip()
        ls = [x.strip(" .*\"'`:") for x in t.splitlines() if x.strip(" .*\"'`:")]
        return ls[0] if ls else ""

    def solve_over(graph, specs):
        gv = {n["id"]: torch.tensor(emb.embed_nodes({n["id"]: n["text"]})[n["id"]], device=dev) for n in graph}
        def retr(query, kind, k=4):
            qv = torch.tensor(emb.embed_nodes({"q": query})["q"], device=dev)
            return sorted(graph, key=lambda n: -float(gv[n["id"]] @ qv))[:k]
        sm = {s.name: s for s in specs}
        def fill(slot, ev, pool):
            facts = "\n".join(f"- {e['text']}" for e in ev)
            return gen(f"Facts:\n{facts}\n\n{sm[slot.name].ask(pool.slots)}")
        sg = SlotGraph(specs); pool = Pool(specs, context={})
        sg.solve(pool, retr, fill)
        return pool.slots[sg.order[-1]].value

    def save_template(t, sig_emb):
        STORE.parent.mkdir(parents=True, exist_ok=True)
        with STORE.open("a", encoding="utf-8") as h:
            h.write(json.dumps({"template": t, "sig_emb": sig_emb}, ensure_ascii=False) + "\n")

    def retrieve_template(task_text):
        if not STORE.exists():
            return None
        qv = torch.tensor(emb.embed_nodes({"q": task_text})["q"], device=dev)
        best, bd = None, -1e9
        for line in STORE.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            sc = float(qv @ torch.tensor(r["sig_emb"], device=dev))
            if sc > bd:
                bd, best = sc, r["template"]
        return best

    print("#13 SELF-IMPROVING MEMORY — save a solved slot-graph, reuse it on a NEW task. INSPECT.\n")
    # TASK A: solve via the template, then SAVE it (with its signature embedding).
    specs_a = instantiate(TEMPLATE, "Zarnvolt Protocol")
    ans_a = solve_over(GRAPH_A, specs_a)
    sig_emb = emb.embed_nodes({"s": TEMPLATE["signature"]})["s"]
    save_template(TEMPLATE, sig_emb)
    print(f"TASK A (Zarnvolt Protocol): solved -> {ans_a!r}  correct={'quintsteel' in ans_a.lower()}  -> template SAVED to {STORE}\n")

    # TASK B (NEW): retrieve the template by the question's signature, instantiate for Brunel, solve.
    QB = "What metal is produced by the home province of the city founded by the author of the Brunel Accord?"
    tmpl = retrieve_template(QB)
    print(f"TASK B retrieved a template: {tmpl is not None and tmpl['signature'][:50]!r}")
    specs_b = instantiate(tmpl, "Brunel Accord")
    ans_b = solve_over(GRAPH_B, specs_b)
    print(f"TASK B (Brunel Accord): solved via the REUSED template -> {ans_b!r}  correct={'brightsteel' in ans_b.lower()}")
    print(f"\n=== #13: a SOLVED reasoning structure was saved + retrieved + reused on a DIFFERENT task ===")
    print(f"  INSPECT: did B reuse A's saved slot-graph (signature match) and solve a NEW document correctly?")
    print(f"  This is grow-reasoning-not-params: the graph accumulates reasoning STRUCTURES.")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Slot-graph reasoning substrate.")
    ap.add_argument("--demo", action="store_true", help="run the toy loop-mechanics demo (no model)")
    ap.add_argument("--v3", action="store_true", help="V3: save a template, reuse on a 2nd task (2nd-task-easier)")
    ap.add_argument("--real", action="store_true", help="#8: REAL frozen-4B fill + embedder retrieval, verbose dumps")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--layer", type=int, default=26)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--ntok", type=int, default=64)   # reasoning model thinks first; strip + take the answer line
    ap.add_argument("--distractors", type=int, default=0,
                    help=">0 = big-graph value-demo: add a confuser (reviewer) chain + N*4 volume facts; "
                         "tests whether cold (all facts in context) fails while slot-graph retrieve-per-hop holds")
    ap.add_argument("--reason", action="store_true", help="#14: operators in REASONING slots (op-fill vs RAG)")
    ap.add_argument("--selfimprove", action="store_true", help="#13: save a solved slot-graph, reuse on a new task")
    a = ap.parse_args(argv)
    if a.selfimprove:
        _selfimprove_demo(a.model, a.layer, a.alpha, a.ntok)
    elif a.reason:
        _reason_demo(a.model, a.layer, a.alpha, a.ntok)
    elif a.real:
        _real_run(a.model, a.layer, a.alpha, a.ntok, a.distractors)
    elif a.v3:
        _v3_test()
    elif a.demo:
        _toy_demo()
    else:
        print("use --demo (mechanics) / --v3 (template reuse) / --real (frozen-4B fill, A40)")


if __name__ == "__main__":
    main()
