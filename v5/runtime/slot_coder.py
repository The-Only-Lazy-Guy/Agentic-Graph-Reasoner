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
    # retrieve-OR-DERIVE: when retrieval returns nothing usable, derive the value from upstream slots
    # (the TRANSFORM operator). Returns "" if it cannot derive -> slot goes INSUFFICIENT -> backtrack.
    derive: Callable[["Slot", "Pool"], str] = None
    # own-evidence GATE: is THIS slot's basis sufficient (not just "are my deps valid")? Default = the
    # slot produced a non-empty value. SWE FIX overrides this with applyable/verify.
    sufficient: Callable[["Slot", "Pool"], bool] = None
    # how a downstream failure REVISES this slot when backtracking to it:
    #   'evidence' = rule out the supporting evidence + re-retrieve (retrieval ambiguity, e.g. a confuser)
    #   'rederive' = keep the evidence, re-fill with more effort/feedback (synthesis, e.g. re-DIAGNOSE)
    revise: str = "evidence"


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
        done, order, visiting = set(), [], set()
        def visit(n):
            if n in done or n not in self.specs:
                return
            if n in visiting:                          # back-edge -> cyclic DATAFLOW (not a DAG)
                raise ValueError(f"cyclic DATAFLOW dependency at slot {n!r}; slot-graph must be a DAG")
            visiting.add(n)
            for u in self.specs[n].needs:
                visit(u)
            visiting.discard(n); done.add(n); order.append(n)
        for s in specs:
            visit(s.name)
        return order

    def _deps_complete(self, pool: Pool, name: str) -> bool:
        # all of this slot's DATAFLOW needs are filled valid (the upstream basis is complete)
        return all(pool.slots.get(u, Slot(u)).state == VALID or u in pool.context
                   for u in self.specs[name].needs)

    def _sufficient(self, pool: Pool, name: str) -> bool:
        """GATE = own-evidence sufficiency AND upstream complete. The slot's own check (default:
        produced a non-empty value) must pass too -- not just 'are my deps valid'."""
        slot = pool.slots[name]; spec = self.specs[name]
        own = spec.sufficient(slot, pool) if spec.sufficient else bool(slot.value)
        return bool(own) and self._deps_complete(pool, name)

    def solve(self, pool: Pool, retriever: Callable[[str, str], list],
              filler: Callable[[Slot, list, Pool], str], max_steps: int = 24, log: list | None = None,
              derive: Callable[[Slot, Pool], str] | None = None, enable_backtrack: bool = True):
        """Fixpoint over the TSG. Per non-(valid+sufficient) slot, upstream-first:
          FILL   : slot-aware retrieve (minus nogoods) -> filler; if empty -> DERIVE from upstream.
          GATE   : own-evidence sufficiency + deps complete -> VALID, else INSUFFICIENT.
          BACKTRACK (dependency-directed): an INSUFFICIENT slot revises its weakest VALID upstream --
                    records the upstream's evidence as a NOGOOD and marks it STALE so re-fill picks
                    differently -- then retries. No upstream to revise -> the slot is unsolvable (parked).
        Returns (all_valid, steps)."""
        from collections import defaultdict
        nogood: Dict[str, set] = defaultdict(set)      # slot -> evidence ids ruled out (de Kleer nogoods)
        failed: set = set()                            # slots proven unsolvable -> parked (picker skips)
        for step in range(max_steps):
            target = None
            for name in self.order:                    # topological: upstream first
                s = pool.slots[name]
                if name not in failed and s.state in (EMPTY, STALE, INSUFFICIENT):
                    target = name; break
            if target is None:
                if log is not None: log.append(("FIXPOINT", step))
                return all(pool.slots[n].state == VALID for n in self.order), step
            spec = self.specs[target]; slot = pool.slots[target]

            # ── INSUFFICIENT -> BACKTRACK: revise the nearest VALID upstream (dependency-directed) ──
            if slot.state == INSUFFICIENT and enable_backtrack:
                ups = [u for u in spec.needs if pool.slots.get(u) and pool.slots[u].state == VALID]
                if ups:
                    weak = max(ups, key=lambda u: self.order.index(u))   # nearest upstream = closest cause
                    if self.specs[weak].revise == "rederive":            # synthesis: keep evidence, re-fill harder
                        pool.slots[weak].state = STALE
                    else:                                                # retrieval: rule out the SUPPORTING
                        j = pool.slots[weak].justification               # evidence (not the whole bag) + re-retrieve
                        for eid in (j.get("used") or j.get("evidence") or []):
                            nogood[weak].add(eid)
                        pool.slots[weak].state = STALE
                    slot.state = EMPTY                                   # retry once the upstream changes
                    if log is not None: log.append(("BACKTRACK", target, "revise", weak, self.specs[weak].revise))
                    continue
                failed.add(target)                                       # nothing to revise -> give up
                if log is not None: log.append(("UNSOLVABLE", target))
                continue
            if slot.state == INSUFFICIENT:               # backtrack disabled -> park (old behavior)
                failed.add(target)
                if log is not None: log.append(("STUCK", target))
                continue

            # ── FILL: slot-aware retrieve (minus nogoods) -> filler; retrieve-OR-DERIVE ──
            q = spec.query(pool.slots | {})
            ev = [e for e in retriever(q, spec.evidence_kind) if e.get("id") not in nogood[target]]
            value = filler(slot, ev, pool) if ev else ""
            ev_ids = [e.get("id") for e in ev]                           # full provenance (for invalidate)
            used = [e.get("id") for e in ev if value and value in (e.get("text") or "")]  # what SUPPORTS it
            dfn = spec.derive or derive
            if not value and dfn is not None:                            # retrieval missed -> DERIVE
                value = dfn(slot, pool)
                if value:
                    ev_ids = used = list(spec.needs)                     # justified by upstream slots
                    if log is not None: log.append((target, "DERIVE " + value[:32], "derived", q[:34]))
            if not value:                                                # neither retrieve nor derive
                pool.store(target, "", {"q": q, "evidence": ev_ids, "used": used}, INSUFFICIENT)
                if log is not None: log.append((target, "(unfilled)", INSUFFICIENT, q[:40]))
                continue
            pool.store(target, value, {"q": q, "evidence": ev_ids, "used": used or ev_ids[:1]}, TENTATIVE)
            # ── GATE: own-evidence sufficiency + deps complete ──
            slot.state = VALID if self._sufficient(pool, target) else INSUFFICIENT
            # ── PROPAGATE forward: dependents go STALE (their basis changed) ──
            for name, spec2 in self.specs.items():
                if target in spec2.needs and pool.slots[name].state == VALID:
                    pool.slots[name].state = STALE
            if log is not None and slot.state == VALID:
                log.append((target, value[:40], slot.state, q[:40]))
        return all(pool.slots[n].state == VALID for n in self.order), max_steps

    def invalidate(self, pool: Pool, node_id: str):
        """Graph edit / new info touching node_id -> slots justified by it go STALE, and STALE
        PROPAGATES transitively to their dependents (their basis changed). Belief revision (design §3)."""
        dirty = {name for name, s in pool.slots.items()
                 if node_id in (s.justification.get("evidence") or [])}
        for name in dirty:
            pool.slots[name].state = STALE
        changed = True
        while changed:                                 # propagate downstream along DATAFLOW
            changed = False
            for name, spec in self.specs.items():
                if pool.slots[name].state == VALID and any(u in dirty for u in spec.needs):
                    pool.slots[name].state = STALE; dirty.add(name); changed = True


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


# ── #14 (REAL): operators COMPOSED INTO solve() — a reason slot's fill IS an operator inject ──
# 2-slot graph EVIDENCE (retrieve the relevant nodes) -> JUDGE (mode='reason'): the JUDGE fill runs
# OperatorInjector.combine+inject INSIDE the fixpoint loop (op_kind_for: strategy->ASSERT, failure
# _pattern->INVALIDATE). Shared `_reason_build` is used by BOTH the no-model `--reason-selftest` (proves
# the reason slot enters the operator path during solve) AND the 4B `--reason` run (proves op>RAG value
# on the code items where op>RAG is already proven, code_reasoning_suite). No more decorative injector.
def _reason_build(q, ev_nodes, judge_fill):
    """EVIDENCE -> JUDGE(reason) through SlotGraph.solve. judge_fill(q, ev) -> verdict; ev = the nodes.
    Returns (judge_value, solve_log)."""
    def retr(query, kind):
        return ev_nodes                                   # the slot's evidence pool (a ranker fetches these)

    def filler(slot, ev, pool):
        if slot.name == "EVIDENCE":
            return ",".join(e["id"] for e in ev)          # register what was retrieved (extract slot)
        return judge_fill(pool.get("q"), ev)              # JUDGE = reason slot: operator/RAG fill, IN-LOOP

    specs = [SlotSpec("EVIDENCE", [], "node", "ASSERT", query=lambda p: q, mode="extract"),
             SlotSpec("JUDGE", ["EVIDENCE"], "node", "ASSERT", query=lambda p: q, mode="reason")]
    sg = SlotGraph(specs); pool = Pool(specs, context={"q": q})
    log = []
    sg.solve(pool, retr, filler, log=log)
    return pool.slots["JUDGE"].value, log


def _reason_selftest():
    """No-model proof: the JUDGE reason slot ENTERS the operator inject path during solve() (not a
    decorative injector). A fake injector flags when inject() is entered; the op judge reads that flag."""
    import contextlib
    from v5.operator_schema import op_kind_for
    print("slot_coder --reason-selftest: prove the reason slot enters the OPERATOR path inside solve().\n")
    ev = [{"id": "good", "text": "correct insight", "node_type": "strategy"},
          {"id": "bad", "text": "the misconception", "node_type": "failure_pattern"}]

    class _FakeInj:
        def __init__(self): self.entered = False; self.ops = None
        def combine(self, nodes, q, normalize=False): self.ops = [op for _, op in nodes]; return "OPVEC"
        @contextlib.contextmanager
        def inject(self, v):
            self.entered = True
            try: yield
            finally: pass
    fake = _FakeInj()

    def op_judge(q, evn):                                  # mirrors the real operator fill
        nodes = [(e["text"], op_kind_for(e["node_type"])) for e in evn]
        v = fake.combine(nodes, q, normalize=True)
        with fake.inject(v):
            return "A" if fake.entered else "B"            # the gen would see the steered state -> correct
    def rag_judge(q, evn):
        return "B"                                         # raw misconception in context -> poisoned -> wrong

    from v5.operator_injector import SIGN
    op_v, oplog = _reason_build("Is X valid?", ev, op_judge)
    rag_v, _ = _reason_build("Is X valid?", ev, rag_judge)
    routed = any(r[0] == "EVIDENCE" for r in oplog) and any(r[0] == "JUDGE" for r in oplog)
    # the SIGN semantics (not the exact op string): the valid insight grounds POSITIVE, the
    # misconception SUBTRACTS (INVALIDATE). strategy->TRANSFORM(+1), failure_pattern->INVALIDATE(-1).
    signs_ok = bool(fake.ops) and SIGN.get(fake.ops[0], 0) > 0 and SIGN.get(fake.ops[-1], 0) < 0
    ok = op_v == "A" and rag_v == "B" and fake.entered and routed and signs_ok
    print("   solve log:", oplog)
    print(f"   JUDGE via operator={op_v!r}  via RAG={rag_v!r}  inject-entered={fake.entered}  "
          f"op_kinds={fake.ops} signs={[SIGN.get(o,0) for o in (fake.ops or [])]}  routed={routed}")
    print(f"\n   WIRING PROOF -> {'PASS' if ok else 'FAIL'}  (the reason slot's fill RAN inj.combine+inject")
    print("    INSIDE solve, signed by op_kind_for; the injector is no longer decorative).")
    return ok


def _reason_demo(model_name, layer, alpha, ntok):
    import contextlib, os, re, torch
    from v5.lm_loader import load_frozen_lm
    from v5.operator_injector import OperatorInjector
    from v5.operator_schema import op_kind_for
    from transformers import AutoTokenizer

    # the PROVEN op>RAG code items (inlined; code_reasoning_suite.py removed in the repo migration).
    # (kind, question ending "Answer: (", correct, wrong, CORRECT insight -> ASSERT/+, MISCONCEPTION -> INVALIDATE/-)
    ITEMS = [
        ("bug", "def add(x, lst=[]):\n    lst.append(x)\n    return lst\n\nadd(1) returned [1]. The next "
         "call add(2) returns (A) [2] (B) [1, 2]. Answer: (", "B", "A",
         "A default list argument is created once at definition and shared across all calls, so it accumulates.",
         "Each function call creates a fresh empty list for a default list argument."),
        ("bug", "fns = [lambda: i for i in range(3)]\n\nfns[0]() returns (A) 0 (B) 2. Answer: (", "B", "A",
         "Closures capture the loop variable by reference, so after the loop they all see its final value.",
         "Each lambda captures the current value of the loop variable at the moment it is created."),
        ("bug", "In CPython:  a = 257; b = 257;  the expression (a is b) evaluates to (A) True (B) False. "
         "Answer: (", "B", "A",
         "CPython caches small integers from -5 to 256; 257 is outside that range, so the two are distinct "
         "objects and 'is' is False.",
         "Equal integers are always the same object in Python, so 'is' returns True."),
        ("bug", "import time\ndef f(t=time.time()):\n    return t\n\nCalling f() at different times returns "
         "(A) a different value each call (B) the same value every call. Answer: (", "B", "A",
         "Default argument values are evaluated once when the function is defined, not on each call.",
         "Default arguments are re-evaluated on every call, so time.time() gives a fresh value each time."),
        ("fix", "Issue: a KeyError is raised when an expected config key is missing. The more targeted, "
         "robust fix is (A) use dict.get(key, default) (B) wrap a large block in try/except Exception. "
         "Answer: (", "A", "B",
         "Use dict.get with a default for an expected-missing key; reserve try/except for genuinely "
         "exceptional cases, since broad except hides other bugs.",
         "Wrapping the whole block in try/except Exception is the safest, most robust way to handle a "
         "missing key."),
        ("fix", "Issue: O(n^2) slowness caused by repeated list.index() calls inside a loop. The fix that "
         "addresses the root cause is (A) build a dict/set for O(1) lookups (B) add an lru_cache decorator "
         "to the function. Answer: (", "A", "B",
         "Repeated linear scans (list.index in a loop) are fixed by a dict/set giving O(1) lookup; caching "
         "does not help when the inputs are all distinct.",
         "Adding an lru_cache decorator removes the O(n^2) cost of the repeated list.index calls."),
    ]

    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    model = load_frozen_lm(model_name); model.eval()
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    inj = OperatorInjector(model, tok, layer, alpha); dev = next(model.parameters()).device

    def tid(s): return tok(s, add_special_tokens=False).input_ids[-1]
    A_id, B_id = tid("A"), tid("B")

    def score(prompt, R, W, v=None):
        """answer-position logit MARGIN logit(R)-logit(W) + the discrete pick. The prompt ends
        'Answer: (' so the next token is A/B (raw prompt, matches the proven code_reasoning_suite setup;
        the margin is the SENSITIVE reasoning signal that a coarse A/B pick hides)."""
        lg = inj.answer_logits(prompt, v)
        mr = float(lg[tid(R)] - lg[tid(W)])
        pick = "A" if lg[A_id] >= lg[B_id] else "B"
        return pick, mr

    margins = {"cold": [], "op": [], "rag": []}
    cur = {}

    def mk_judge(mode):
        def judge(q, ev):                                  # runs INSIDE solve() as the JUDGE reason-slot fill
            R, W = cur["R"], cur["W"]
            if mode == "op":                               # OPERATOR: op-signed combine -> inject
                nodes = [(e["text"], op_kind_for(e["node_type"])) for e in ev]
                pick, mr = score(q, R, W, inj.combine(nodes, q, normalize=True))
            elif mode == "rag":                            # RAG: the SAME node texts in context
                notes = "\n".join(f"Note: {e['text']}" for e in ev)
                pick, mr = score(f"{notes}\n\n{q}", R, W, None)
            else:                                          # COLD: no grounding
                pick, mr = score(q, R, W, None)
            margins[mode].append(mr)
            return pick
        return judge
    op_judge, rag_judge, cold_judge = mk_judge("op"), mk_judge("rag"), mk_judge("cold")

    print(f"#14 operators COMPOSED INTO solve() — reason-slot fill = operator inject. {len(ITEMS)} code items.")
    print("metric: discrete pick THROUGH solve() + belief MARGIN logit(correct)-logit(wrong) (the reasoning signal).\n")
    op_ok = rag_ok = cold_ok = 0
    for kind, q, R, W, good, bad in ITEMS:
        cur["R"], cur["W"] = R, W
        ev = [{"id": "good", "text": good, "node_type": "strategy"},        # -> ASSERT/+ (positive grounding)
              {"id": "bad", "text": bad, "node_type": "failure_pattern"}]   # -> INVALIDATE/- (subtract)
        op_v, log = _reason_build(q, ev, op_judge);   op_m = margins["op"][-1]
        rag_v, _ = _reason_build(q, ev, rag_judge);   rag_m = margins["rag"][-1]
        cold_v, _ = _reason_build(q, ev, cold_judge); cold_m = margins["cold"][-1]
        op_ok += (op_v == R); rag_ok += (rag_v == R); cold_ok += (cold_v == R)
        print(f"[{kind:3} expect={R}]  pick cold={cold_v} OP={op_v} RAG={rag_v}   "
              f"margin(corr-wrong) cold={cold_m:+.2f} OP={op_m:+.2f} RAG={rag_m:+.2f}  "
              f"{'OP>RAG' if op_m > rag_m else 'op<=rag'}")
        print(f"   Q: {q[:64]!r}   routed: {[r[0] for r in log]}")
    n = len(ITEMS)
    import statistics as st
    mm = lambda k: st.mean(margins[k])
    opw = sum(1 for o, r in zip(margins['op'], margins['rag']) if o > r)
    print(f"\n=== THROUGH solve() (N={n}) ===")
    print(f"  discrete pick:  operator {op_ok}/{n} | RAG {rag_ok}/{n} | cold {cold_ok}/{n}")
    print(f"  MEAN MARGIN  :  operator {mm('op'):+.2f} | RAG {mm('rag'):+.2f} | cold {mm('cold'):+.2f}   "
          f"(higher = stronger pull to the CORRECT option)")
    print(f"  operator margin > RAG margin on {opw}/{n} items")
    print("  READING: discrete A/B is coarse + ties easily. The MARGIN answers 'does the operator help the")
    print("  model PICK the better option?' — op_margin>rag_margin means the typed subtract (INVALIDATE the")
    print("  misconception) pulls preference toward the correct fix MORE than dumping the same text via RAG.")


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


# ── #10: dependency-directed BACKTRACKING + INSUFFICIENT + retrieve-OR-DERIVE (no model, deterministic) ──
# Proves the engine behaviors that were design-only before: a CONFUSER (reviewer ranked above author)
# poisons the chain so a downstream slot can't fill -> INSUFFICIENT -> backtrack rules out the bad
# upstream value (nogood) and walks back until the chain resolves; a terminal slot with NO retrievable
# fact is filled by DERIVE from upstream. A/B (backtrack off vs on) = the right-for-right-reason proof.
def _backtrack_demo():
    GRAPH = [
        {"id": "r1", "kind": "fact", "text": "The Codex was reviewed by Bob Krell."},   # CONFUSER (reviewer)
        {"id": "a1", "kind": "fact", "text": "The Codex was written by Ana Stahl."},     # the real author
        {"id": "bk", "kind": "fact", "text": "Bob Krell founded the town of Redhollow."},
        {"id": "as", "kind": "fact", "text": "Ana Stahl founded the town of Greenford."},
        {"id": "gf", "kind": "fact", "text": "The town of Greenford mines a metal called auralite."},
        # NOTE: no "metal mined by Redhollow" fact -> the confuser path dead-ends -> backtrack must recover.
        {"id": "n1", "kind": "fact", "text": "The Codex has nine chapters."},
    ]
    GOLD = "auralite"

    def _author(t):
        i = t.lower().rfind(" by "); tail = t[i + 4:].strip(" .") if i >= 0 else ""
        out = []
        for w in tail.split():
            cw = w.strip(".,")
            if cw[:1].isupper(): out.append(cw)
            else: break
        return " ".join(out)

    def _after(t, kw):
        i = t.lower().find(kw); tail = t[i + len(kw):].strip(" .") if i >= 0 else ""
        return tail.split()[0].strip(".,") if tail else ""

    def retriever(query, kind):                 # deterministic keyword-routed ranker (imperfect on AUTHOR)
        ql = query.lower()
        if "metal" in ql or "mines" in ql:
            return [n for n in GRAPH if "mines a metal" in n["text"]]
        if "town" in ql or "founded" in ql:
            return [n for n in GRAPH if "founded the town" in n["text"]]
        if "author" in ql or "who" in ql or "wrote" in ql:
            return [next(n for n in GRAPH if n["id"] == "r1"),     # CONFUSER (reviewer) ranked FIRST
                    next(n for n in GRAPH if n["id"] == "a1")]
        return []                                                  # SUMMARY query -> nothing -> DERIVE

    specs = [
        SlotSpec("AUTHOR", [], "fact", "ASSERT", query=lambda p: "author of the Codex"),
        SlotSpec("CITY", ["AUTHOR"], "fact", "ASSERT", query=lambda p: f"town founded by {p['AUTHOR'].value}"),
        SlotSpec("METAL", ["CITY"], "fact", "ASSERT", query=lambda p: f"metal from the town {p['CITY'].value}"),
        SlotSpec("SUMMARY", ["METAL"], "fact", "ASSERT", query=lambda p: "one-line recap for the Codex",
                 derive=lambda slot, pool: (f"The Codex's metal is {pool.slots['METAL'].value}."
                                            if pool.slots["METAL"].value else "")),
    ]
    sbyn = {s.name: s for s in specs}

    def filler(slot, ev, pool):                 # LM stub: answer ONLY if evidence mentions the upstream entity
        need = sbyn[slot.name].needs
        up = pool.slots[need[0]].value if need else ""
        cands = [e for e in ev if (not up or up in e["text"])]
        if not cands:
            return ""
        t = cands[0]["text"]
        if slot.name == "AUTHOR": return _author(t)
        if slot.name == "CITY":   return _after(t, "town of ")
        if slot.name == "METAL":  return _after(t, "called ")
        return ""

    def run(enable):
        sg = SlotGraph(specs); pool = Pool(specs, context={"issue": "What metal traces to the Codex's author?"})
        log = []
        ok, steps = sg.solve(pool, retriever, filler, log=log, enable_backtrack=enable)
        return ok, steps, pool, log

    print("#10 BACKTRACKING + INSUFFICIENT + retrieve-OR-DERIVE — deterministic, MANUALLY INSPECT.\n")
    print("Chain AUTHOR->CITY->METAL->SUMMARY. The ranker puts the REVIEWER (Bob Krell) above the author")
    print("(Ana Stahl) -> METAL dead-ends on Redhollow. SUMMARY has no fact -> must DERIVE.\n")
    for label, enable in [("BACKTRACK OFF (old engine)", False), ("BACKTRACK ON (fixed engine)", True)]:
        ok, steps, pool, log = run(enable)
        final = pool.slots["METAL"].value
        print(f"=== {label} ===")
        for row in log:
            print("   ", row)
        print(f"   POOL: {{ {', '.join(f'{n}={s.value!r}/{s.state}' for n, s in pool.slots.items())} }}")
        print(f"   fixpoint={ok} steps={steps}  METAL={final!r}  SUMMARY={pool.slots['SUMMARY'].value!r}  "
              f"correct={GOLD in final}\n")
    print("EXPECT: OFF -> METAL unfilled (INSUFFICIENT, never recovered) -> WRONG. "
          "ON -> backtrack rules out the\nreviewer, walks back AUTHOR, chain resolves to 'auralite', "
          "SUMMARY derived. INSPECT the BACKTRACK/DERIVE log rows.")


# ── BACKEND TEST SUITE: deterministic asserts over the engine (no model). Every state + transition. ──
def _engine_tests():
    def node(i, t): return {"id": i, "kind": "fact", "text": t}
    def first_word_ret(nodes):
        def r(q, kind):
            key = q.split()[0].lower() if q.split() else ""
            return [n for n in nodes if key and key in n["text"].lower()]
        return r
    def take_first(slot, ev, pool):
        return ev[0]["text"] if ev else ""

    results = []
    def check(name, cond, detail=""):
        results.append((name, bool(cond), detail))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail and not cond else ''}")

    # T1 — linear happy path: both slots fill -> fixpoint, all VALID.
    s = [SlotSpec("A", [], "fact", "ASSERT", query=lambda p: "alpha x"),
         SlotSpec("B", ["A"], "fact", "ASSERT", query=lambda p: "beta y")]
    nodes = [node("a", "alpha thing"), node("b", "beta thing")]
    pool = Pool(s); ok, steps = SlotGraph(s).solve(pool, first_word_ret(nodes), take_first)
    check("T1 linear->fixpoint all VALID", ok and all(pool.slots[n].state == VALID for n in "AB"),
          f"ok={ok} states={[pool.slots[n].state for n in 'AB']}")

    # T2 — retrieve-OR-DERIVE: B has no retrievable fact (query 'none') -> derive from A.
    s = [SlotSpec("A", [], "fact", "ASSERT", query=lambda p: "alpha x"),
         SlotSpec("B", ["A"], "fact", "ASSERT", query=lambda p: "none",
                  derive=lambda slot, pool: f"derived[{pool.get('A')[:5]}]")]
    pool = Pool(s); ok, _ = SlotGraph(s).solve(pool, first_word_ret([node("a", "alpha thing")]), take_first)
    check("T2 retrieve-OR-derive fills B", ok and pool.slots["B"].value.startswith("derived")
          and pool.slots["B"].state == VALID, f"B={pool.slots['B'].value!r}/{pool.slots['B'].state}")

    # T3 — unsolvable: no evidence, no derive, no upstream -> INSUFFICIENT, parked, terminates (not hang).
    s = [SlotSpec("A", [], "fact", "ASSERT", query=lambda p: "none")]
    pool = Pool(s); ok, steps = SlotGraph(s).solve(pool, lambda q, k: [], take_first, max_steps=12)
    check("T3 unsolvable -> False, INSUFFICIENT, terminates", (not ok)
          and pool.slots["A"].state == INSUFFICIENT and steps < 12, f"ok={ok} steps={steps}")

    # T4/T5/T8 — CONFUSER chain: ranker puts reviewer above author; METAL dead-ends -> backtrack must
    # rule out ONLY the confuser (surgical nogood) and recover the right author (not collapse to empty).
    conf = [node("rev", "Doc reviewed by Bob"), node("aut", "Doc written by Ana"),
            node("bt", "Bob founded Redtown"), node("at", "Ana founded Greentown"),
            node("gm", "Greentown makes gold")]                      # NOTE: no Redtown metal
    def conf_ret(q, kind):
        ql = q.lower()
        if "metal" in ql: return [n for n in conf if "makes" in n["text"]]
        if "town" in ql:  return [n for n in conf if "founded" in n["text"]]
        return [conf[0], conf[1]]                                    # AUTHOR: reviewer FIRST (the mistake)
    def conf_fill(slot, ev, pool):
        need = {"AUTHOR": None, "CITY": "AUTHOR", "METAL": "CITY"}[slot.name]
        up = pool.get(need) if need else ""
        cands = [e for e in ev if (not up or up in e["text"])]
        if not cands: return ""
        t = cands[0]["text"]
        if slot.name == "AUTHOR": return t.split(" by ")[-1].strip()           # Bob / Ana
        if slot.name == "CITY":   return t.split("founded ")[-1].strip()        # Redtown / Greentown
        return t.split("makes ")[-1].strip()                                    # gold
    cs = [SlotSpec("AUTHOR", [], "fact", "ASSERT", query=lambda p: "author Doc"),
          SlotSpec("CITY", ["AUTHOR"], "fact", "ASSERT", query=lambda p: f"town founded {pool_get(p,'AUTHOR')}"),
          SlotSpec("METAL", ["CITY"], "fact", "ASSERT", query=lambda p: f"metal {pool_get(p,'CITY')}")]
    t4log = []
    pool = Pool(cs); ok_on, _ = SlotGraph(cs).solve(pool, conf_ret, conf_fill, log=t4log)
    bt_fired = any(r[0] == "BACKTRACK" for r in t4log)
    picked_confuser = any(r[0] == "AUTHOR" and "Bob" in str(r[1]) for r in t4log)  # confuser WAS chosen first
    check("T4 backtrack recovers chain (confuser picked, BACKTRACK fired, METAL=gold)", ok_on
          and pool.slots["METAL"].value == "gold" and pool.slots["AUTHOR"].value == "Ana"
          and bt_fired and picked_confuser,
          f"AUTHOR={pool.slots['AUTHOR'].value!r} METAL={pool.slots['METAL'].value!r} "
          f"bt_fired={bt_fired} confuser_first={picked_confuser}")
    check("T8 surgical nogood: AUTHOR recovered to Ana (not empty-collapse)",
          pool.slots["AUTHOR"].value == "Ana" and pool.slots["AUTHOR"].state == VALID)
    pool2 = Pool(cs); ok_off, _ = SlotGraph(cs).solve(pool2, conf_ret, conf_fill, enable_backtrack=False)
    check("T5 backtrack OFF control -> METAL unfilled, False", (not ok_off)
          and pool2.slots["METAL"].value == "", f"ok={ok_off} METAL={pool2.slots['METAL'].value!r}")

    # T6 — belief revision: solve, edit the graph + invalidate -> dependent re-fills new value.
    s = [SlotSpec("A", [], "fact", "ASSERT", query=lambda p: "alpha x")]
    g1 = [node("a", "alpha OLD")]; sg = SlotGraph(s); pool = Pool(s)
    sg.solve(pool, first_word_ret(g1), take_first)
    v_old = pool.slots["A"].value
    g2 = [node("a", "alpha NEW")]
    sg.invalidate(pool, "a")                                          # graph edit touches node 'a'
    stale = pool.slots["A"].state == STALE
    sg.solve(pool, first_word_ret(g2), take_first)                    # re-solve over the edited graph
    check("T6 invalidate->STALE->re-fill new value", stale and v_old == "alpha OLD"
          and pool.slots["A"].value == "alpha NEW", f"stale={stale} now={pool.slots['A'].value!r}")

    # T7 — own-evidence GATE (sufficient()=False with a NON-empty value) + rederive recovery.
    cnt = {"D": 0}
    def t7_fill(slot, ev, pool):
        if slot.name == "D":
            n = cnt["D"]; cnt["D"] += 1
            return "shallow" if n == 0 else "deep"
        return "GOOD-fix" if "deep" in pool.get("D") else "BAD-fix"     # F applyable only if D deep
    s = [SlotSpec("D", [], "fact", "ASSERT", query=lambda p: "x", revise="rederive"),
         SlotSpec("F", ["D"], "fact", "ASSERT", query=lambda p: "y",
                  sufficient=lambda slot, pool: slot.value.startswith("GOOD"))]
    pool = Pool(s); ok, _ = SlotGraph(s).solve(pool, lambda q, k: [node("e", "ev")], t7_fill)
    check("T7 own-GATE rejects BAD value -> rederive -> GOOD", ok
          and pool.slots["F"].value == "GOOD-fix" and cnt["D"] >= 2,
          f"F={pool.slots['F'].value!r} D_attempts={cnt['D']}")

    # T9 — toposort: every dep precedes its dependent.
    s = [SlotSpec("Z", ["Y"], "f", "ASSERT", query=lambda p: ""),
         SlotSpec("Y", ["X"], "f", "ASSERT", query=lambda p: ""),
         SlotSpec("X", [], "f", "ASSERT", query=lambda p: "")]
    order = SlotGraph(s).order
    check("T9 toposort deps-before-dependents", order.index("X") < order.index("Y") < order.index("Z"),
          f"order={order}")

    # T10 — cyclic DATAFLOW is REJECTED (was an unguarded RecursionError crash).
    raised = False
    try:
        SlotGraph([SlotSpec("A", ["B"], "f", "ASSERT", query=lambda p: ""),
                   SlotSpec("B", ["A"], "f", "ASSERT", query=lambda p: "")])
    except ValueError:
        raised = True
    check("T10 cyclic deps rejected (no crash)", raised)

    # T11 — diamond DAG: D needs [B,C], B,C need A -> solve to fixpoint (multi-parent FILL).
    sd = [SlotSpec("D", ["B", "C"], "f", "ASSERT", query=lambda p: "d"),
          SlotSpec("B", ["A"], "f", "ASSERT", query=lambda p: "b"),
          SlotSpec("C", ["A"], "f", "ASSERT", query=lambda p: "c"),
          SlotSpec("A", [], "f", "ASSERT", query=lambda p: "a")]
    pool = Pool(sd); ok, _ = SlotGraph(sd).solve(pool, lambda q, k: [node("x", q.split()[0])], take_first)
    check("T11 diamond DAG -> fixpoint all VALID", ok and all(pool.slots[n].state == VALID for n in "ABCD"))

    # T12 — max_steps guard: a rederive slot that NEVER improves must terminate (not hang), return False.
    s = [SlotSpec("D", [], "f", "ASSERT", query=lambda p: "x", revise="rederive"),
         SlotSpec("F", ["D"], "f", "ASSERT", query=lambda p: "y",
                  sufficient=lambda slot, pool: False)]                 # F never sufficient -> endless retry
    pool = Pool(s); ok, steps = SlotGraph(s).solve(
        pool, lambda q, k: [node("e", "ev")], lambda slot, ev, pool: "same", max_steps=10)
    check("T12 non-converging rederive terminates at max_steps", (not ok) and steps == 10, f"steps={steps}")

    # T13 — multi-slot belief revision: edit upstream -> dependent goes STALE (propagated) -> re-fill.
    s = [SlotSpec("A", [], "fact", "ASSERT", query=lambda p: "a x"),
         SlotSpec("B", ["A"], "fact", "ASSERT", query=lambda p: "b y")]
    sg = SlotGraph(s); pool = Pool(s)
    sg.solve(pool, first_word_ret([node("a", "a v1"), node("b", "b v1")]), take_first)
    before = (pool.slots["A"].state, pool.slots["B"].state)
    sg.invalidate(pool, "a")                                           # edit A's evidence
    propagated = pool.slots["A"].state == STALE and pool.slots["B"].state == STALE   # B propagated too
    sg.solve(pool, first_word_ret([node("a", "a v2"), node("b", "b v2")]), take_first)
    check("T13 belief revision propagates STALE to dependent + re-fills", before == (VALID, VALID)
          and propagated and pool.slots["A"].value == "a v2" and pool.slots["B"].value == "b v2"
          and pool.slots["B"].state == VALID, f"propagated={propagated} B={pool.slots['B'].value!r}")

    # T14 — derive is a FALLBACK: when retrieval HITS, the filler is used and derive is NOT called.
    called = {"d": False}
    def _d(slot, pool): called["d"] = True; return "DERIVED"
    s = [SlotSpec("A", [], "fact", "ASSERT", query=lambda p: "a x", derive=_d)]
    pool = Pool(s); SlotGraph(s).solve(pool, first_word_ret([node("a", "a v1")]), take_first)
    check("T14 derive is fallback (not called when retrieval hits)",
          pool.slots["A"].value == "a v1" and not called["d"], f"value={pool.slots['A'].value!r} d_called={called['d']}")

    n_pass = sum(1 for _, c, _ in results if c)
    print(f"\n=== ENGINE BACKEND: {n_pass}/{len(results)} PASS ===")
    return n_pass == len(results)


def pool_get(slots, name):                # helper for query lambdas inside _engine_tests (slots = pool.slots)
    s = slots.get(name)
    return s.value if s and s.state in (VALID, TENTATIVE) else ""


# ── retrieve-OR-DERIVE with a REAL 4B: when the graph has NO fact for a slot, the frozen LM DERIVES
# the value from upstream slots (TRANSFORM). Tests the value (computable -> correct), the contrast
# (no derive -> unfillable), and the BOUNDARY (non-computable arbitrary fact -> hallucinated). ──
def _derive_demo(model_name, layer, alpha, ntok):
    import os, re
    from v5.lm_loader import load_frozen_lm
    from transformers import AutoTokenizer
    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    model = load_frozen_lm(model_name); model.eval()
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    dev = next(model.parameters()).device

    def gen(prompt, nt=16):
        msgs = [{"role": "user", "content": prompt}]
        kw = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
        try:
            enc = tok.apply_chat_template(msgs, enable_thinking=False, **kw).to(dev)
        except TypeError:
            enc = tok.apply_chat_template(msgs, **kw).to(dev)
        out = model.generate(**enc, max_new_tokens=nt, do_sample=False, pad_token_id=tok.eos_token_id)
        return re.sub(r"<think>.*?</think>", "", tok.decode(out[0, enc["input_ids"].shape[1]:],
                      skip_special_tokens=True), flags=re.DOTALL).strip()
    def num(s):
        m = re.search(r"-?\d+", s); return m.group(0) if m else ""

    # DERIVABLE: TOTAL = costA + costB, with NO "total" fact in the graph -> must be derived from upstream.
    # fictional widget names so it is genuinely a computation, not recall.
    ITEMS = [("Fizzbolt", 7, "Glimstone", 5, 12), ("Quorblade", 13, "Snarvane", 8, 21),
             ("Vexrod", 20, "Mirethorn", 14, 34)]

    def build(na, ca, nb, cb, with_derive):
        graph = [{"id": "fa", "kind": "fact", "text": f"Widget {na} costs {ca} credits."},
                 {"id": "fb", "kind": "fact", "text": f"Widget {nb} costs {cb} credits."}]
        def retr(q, kind):
            ql = q.lower()
            if na.lower() in ql: return [graph[0]]
            if nb.lower() in ql: return [graph[1]]
            return []                                             # TOTAL query -> NO fact -> retrieval MISSES
        def fill(slot, ev, pool):
            if slot.name == "TOTAL":
                return ""                                         # no evidence -> derive must handle it
            return num(gen(f"{ev[0]['text']} How many credits does it cost? Answer with only the number:")) if ev else ""
        def derive(slot, pool):                                   # TRANSFORM: compute from UPSTREAM slot values
            a, b = pool.get("COST_A"), pool.get("COST_B")
            return num(gen(f"Widget {na} costs {a} credits and widget {nb} costs {b} credits. What is the "
                           f"TOTAL cost of buying one of each? Answer with only the number:"))
        specs = [SlotSpec("COST_A", [], "fact", "ASSERT", query=lambda p: f"cost of widget {na}"),
                 SlotSpec("COST_B", [], "fact", "ASSERT", query=lambda p: f"cost of widget {nb}"),
                 SlotSpec("TOTAL", ["COST_A", "COST_B"], "fact", "ASSERT", query=lambda p: "total combined cost",
                          derive=(derive if with_derive else None))]
        return SlotGraph(specs), Pool(specs), retr, fill

    print("retrieve-OR-DERIVE (4B): the graph has NO 'total' fact -> the LM must DERIVE it from upstream.\n")
    print("DERIVABLE (TOTAL = costA + costB):")
    correct = 0
    for na, ca, nb, cb, gold in ITEMS:
        sg, pool, retr, fill = build(na, ca, nb, cb, True)
        sg.solve(pool, retr, fill, log=None)
        got = pool.slots["TOTAL"].value
        sg0, pool0, retr0, fill0 = build(na, ca, nb, cb, False)
        sg0.solve(pool0, retr0, fill0, enable_backtrack=False, log=None)   # no derive -> show unfillable cleanly
        ok = got == str(gold); correct += ok
        print(f"  [{na} {ca} + {nb} {cb} = {gold}]  WITH-derive TOTAL={got!r} correct={ok} (COST_A={pool.get('COST_A')!r} "
              f"COST_B={pool.get('COST_B')!r}) | NO-derive TOTAL={pool0.slots['TOTAL'].value!r}/{pool0.slots['TOTAL'].state}")
    print(f"  derive-correct: {correct}/{len(ITEMS)}  (and NO-derive leaves TOTAL unfilled/insufficient = derive is load-bearing)\n")

    # BOUNDARY control: a value that is NOT computable from upstream -> derive HALLUCINATES (ungrounded).
    print("BOUNDARY (non-derivable arbitrary fact — derive should NOT be trusted here):")
    na, ca = ITEMS[0][0], ITEMS[0][1]
    graph = [{"id": "fa", "kind": "fact", "text": f"Widget {na} costs {ca} credits."}]
    def retr_c(q, kind):
        return [graph[0]] if na.lower() in q.lower() else []
    def fill_c(slot, ev, pool):
        return num(gen(f"{ev[0]['text']} How many credits? Answer with only the number:")) if (ev and slot.name == "COST") else ""
    def derive_secret(slot, pool):
        return gen(f"Widget {na} costs {pool.get('COST')} credits. What is the manufacturer's secret "
                   f"4-digit discount code for it? Answer with only the code:", nt=12)
    specs = [SlotSpec("COST", [], "fact", "ASSERT", query=lambda p: f"cost of widget {na}"),
             SlotSpec("SECRET", ["COST"], "fact", "ASSERT", query=lambda p: "secret discount code",
                      derive=derive_secret)]
    sg, pool = SlotGraph(specs), Pool(specs)
    sg.solve(pool, retr_c, fill_c, log=None)
    print(f"  [secret code]  derive produced={pool.get('SECRET')!r}  -> HALLUCINATED (not in the graph, ungrounded).")
    print("  BOUNDARY: derive only computes values DERIVABLE from upstream (arithmetic/composition); it is")
    print("  NOT a fact-gap filler. For arbitrary missing facts it invents -> must stay INSUFFICIENT, not derive.")
    print("\n  INSPECT: did WITH-derive compute the right totals from the retrieved costs, did NO-derive leave")
    print("  TOTAL unfilled, and did the boundary case hallucinate? That is the honest scope of derive.")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Slot-graph reasoning substrate.")
    ap.add_argument("--demo", action="store_true", help="run the toy loop-mechanics demo (no model)")
    ap.add_argument("--derive", action="store_true", help="retrieve-OR-DERIVE with the 4B (compute-from-upstream + boundary)")
    ap.add_argument("--test", action="store_true", help="run the no-model engine backend test suite (asserts)")
    ap.add_argument("--backtrack", action="store_true", help="#10: backtracking + INSUFFICIENT + derive (no model)")
    ap.add_argument("--v3", action="store_true", help="V3: save a template, reuse on a 2nd task (2nd-task-easier)")
    ap.add_argument("--real", action="store_true", help="#8: REAL frozen-4B fill + embedder retrieval, verbose dumps")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--layer", type=int, default=26)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--ntok", type=int, default=64)   # reasoning model thinks first; strip + take the answer line
    ap.add_argument("--distractors", type=int, default=0,
                    help=">0 = big-graph value-demo: add a confuser (reviewer) chain + N*4 volume facts; "
                         "tests whether cold (all facts in context) fails while slot-graph retrieve-per-hop holds")
    ap.add_argument("--reason", action="store_true", help="#14: operators COMPOSED INTO solve() (op vs RAG, 4B)")
    ap.add_argument("--reason-selftest", action="store_true", help="#14: no-model proof reason slot enters operator path")
    ap.add_argument("--selfimprove", action="store_true", help="#13: save a solved slot-graph, reuse on a new task")
    a = ap.parse_args(argv)
    if a.test:
        import sys
        sys.exit(0 if _engine_tests() else 1)
    elif a.reason_selftest:
        import sys
        sys.exit(0 if _reason_selftest() else 1)
    elif a.derive:
        _derive_demo(a.model, a.layer, a.alpha, a.ntok)
    elif a.backtrack:
        _backtrack_demo()
    elif a.selfimprove:
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
