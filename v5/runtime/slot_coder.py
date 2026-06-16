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


def main(argv=None):
    ap = argparse.ArgumentParser(description="Slot-graph reasoning substrate.")
    ap.add_argument("--demo", action="store_true", help="run the toy loop-mechanics demo (no model)")
    ap.add_argument("--v3", action="store_true", help="V3: save a template, reuse on a 2nd task (2nd-task-easier)")
    a = ap.parse_args(argv)
    if a.v3:
        _v3_test()
    elif a.demo:
        _toy_demo()
    else:
        print("use --demo (loop mechanics) or --v3 (save/reuse template); model run + SWE = next phase")


if __name__ == "__main__":
    main()
