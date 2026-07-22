"""membrane_edits.py — reflexive graph editing + spreading activation, bound to membrane.py's REAL AtomGraph.

Ports proven FORMULAS from a separate, unwired stack (algo_graph_edits.ReflexiveEditor, algo_grr_health.
PruningMonitor, algo_grr_retrieval.SpreadingActivationRetriever — all real, tested, but bound to a different
Node/Edge type in graph_core.py and never actually wired into any live pipeline) onto membrane.py's own
AtomGraph, rather than switching to that parallel pipeline.

DESIGN PRINCIPLE: node `kind` and edge `relation` stay FREE natural-language text throughout. Nothing here
introduces a closed vocabulary or a string-keyed weight table (a first draft did — corrected). The only
things that become numeric are per-node/per-edge METRICS (confidence, access_count, importance, edge
strength), and every one of those numbers moves ONLY from real verified outcomes (record_success/
record_failure below) — never from pattern-matching the relation/kind text. We don't know in advance what
relation phrasing or node categories the model will actually need, so nothing here presumes it.

Free functions taking `graph` explicitly, matching membrane.py's own convention (learn_any(g, retr, ...)),
not methods bolted onto AtomGraph — keeps membrane.py's blast radius small.

    python -m v5.runtime.membrane_edits --selftest
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from embedder import encode_batch


def _find_edge(g, s: str, d: str):
    """An existing edge between s and d in EITHER direction, or None. Returns the (src,dst,relation) tuple
    as it actually appears in g.edges (direction/relation text preserved, whatever it is)."""
    for es, ed, er in g.edges:
        if (es, ed) == (s, d) or (es, ed) == (d, s):
            return (es, ed, er)
    return None


def record_success(g, pathway: list[str], task_text: str = "", boost: float = 0.15) -> None:
    """A REAL verified success used these atoms together (e.g. Membrane._credit's `used`). Ports
    algo_graph_edits.py:227-244's formula onto AtomGraph.

    FIX vs a literal port: the source only strengthens an edge that ALREADY exists between consecutive
    pathway atoms — in practice two co-retrieved atoms usually have no edge yet, so a literal port would
    silently no-op most of the time. Here: if missing, g.link(a, b, <free-text relation>) first, THEN
    bump_strength() toward 1.0. The POSITIVE direction comes from this being the success path, not from
    matching the relation string — any relation text works, including one nothing has used before."""
    for name in pathway:
        a = g.get(name)
        if a is None:
            continue
        a.confidence = min(1.0, a.confidence + boost * (1.0 - a.confidence))
        a.access_count += 1
        a.importance = min(1.0, a.importance + 0.05)

    for i in range(len(pathway) - 1):
        s, d = pathway[i], pathway[i + 1]
        if s not in g or d not in g or s == d:
            continue
        edge = _find_edge(g, s, d)
        if edge is None:
            g.link(s, d, "used-together")
            edge = (s, d, "used-together")
        g.bump_strength(*edge, +0.1)


def record_failure(g, task_text: str, failed_code: str = "", target_description: str = "") -> str:
    """A REAL verified failure. Ports algo_graph_edits.py:246-330's shape (creates a trap node, links it to
    related nodes) onto AtomGraph — REPLACES learn_any's inline trap block (a single trap-creation path, not
    two divergent ones). Upgrade over the source: links via real cosine similarity (AtomGraph already has
    real embeddings), not the source's token-overlap heuristic. Linked edges get a LOW strength (toward
    0.1) set directly by this function — the NEGATIVE/suppressive meaning comes from being created on the
    failure path, not from a reserved relation keyword; spreading_activate naturally propagates less
    through a low-strength edge, achieving the same practical effect as a signed 'avoid' weight without one."""
    from v5.runtime.membrane import Atom          # deferred: avoid a hard import-cycle at module load

    tnm = f"trap_{abs(hash(task_text)) % 100000}"
    if tnm in g:
        return tnm
    desc = f"FAILED: {task_text}"
    if target_description:
        desc += f" (wanted: {target_description})"
    trap = Atom(name=tnm, code=failed_code, description=desc, kind="trap", provenance="learned",
               confidence=1.0)                     # confidence HERE means "certain this failed", not trustworthiness
    g.add(trap)

    M, order = g.matrix()
    if len(order) > 1:
        sims = M @ trap.emb
        ranked = sorted(((float(sims[i]), order[i]) for i in range(len(order)) if order[i] != tnm),
                        reverse=True)
        for _, name in ranked[:2]:
            g.link(tnm, name, "context-for-failure")
            g.bump_strength(tnm, name, "context-for-failure", -0.4)   # 0.5 default -> 0.1
    return tnm


def utility(g, name: str, w_conf: float = 0.35, w_access: float = 0.35, w_importance: float = 0.30) -> float:
    """Ports algo_grr_health.py:136-147's formula, over confidence/access_count/importance — uniformly for
    every node, regardless of `kind`. The source special-cases kind in ('hub','concept') to always return
    1.0 (protect structurally-important nodes from pruning); deliberately DROPPED here, per the open-
    vocabulary principle — a node's importance should emerge from real usage (confidence/access_count),
    not from a hardcoded string carve-out. Nothing in this plan actually prunes yet; this is a metric only."""
    a = g.get(name)
    if a is None:
        return 0.0
    max_access = max((x.access_count for x in g.atoms.values()), default=0) or 1
    norm_access = a.access_count / max_access
    return w_conf * a.confidence + w_access * norm_access + w_importance * a.importance


def spreading_activate(g, seed_energies: dict, steps: int = 5) -> dict:
    """Real edge-weighted diffusion. Ports algo_grr_retrieval.py:230-253's formula:
        a_next[dst] = tanh(sum(a[src] * strength(src,dst,rel) for edges src->dst) + 0.5*a[dst])
    Propagation weight per edge is g.strength(src,dst,rel) — the LEARNED per-edge scalar from
    record_success/record_failure — NOT a relation-string-keyed table (the source's EDGE_WEIGHTS dict);
    this is the direct fix for 'don't fix the edge type'. Gated by the destination node's confidence."""
    a = {n: 0.0 for n in g.atoms}
    for n, v in seed_energies.items():
        if n in a:
            a[n] = v
    for _ in range(max(1, steps)):
        nxt = {n: 0.5 * a[n] for n in a}
        for s, d, r in g.edges:
            if s in a and d in a:
                nxt[d] += a[s] * g.strength(s, d, r) * g.atoms[d].confidence
        a = {n: math.tanh(v) for n, v in nxt.items()}
    return a


def glowing_subgraph(g, query_text: str, steps: int = 5, threshold: float = 0.1, n_seeds: int = 5) -> list:
    """Seeds via g.cosine_rank() (already on AtomGraph — no new dependency, unlike the source which needs an
    injected base retriever), runs spreading_activate, returns every node above threshold: 'a coherent
    context block' focused on the query — the session-focus building block Phase 2 wraps."""
    seeds = set(g.cosine_rank(query_text, k=n_seeds))
    if not seeds:
        return []
    qv = encode_batch([query_text])[0]
    M, order = g.matrix()
    sims = M @ qv
    seed_energies = {order[i]: max(0.0, float(sims[i])) for i in range(len(order)) if order[i] in seeds}
    activation = spreading_activate(g, seed_energies, steps=steps)
    return [n for n, v in activation.items() if v >= threshold]


def _selftest():
    from v5.runtime.membrane import AtomGraph, Atom
    print("membrane_edits.py --selftest\n")
    g = AtomGraph()
    g.add(Atom(name="a", code="", description="alpha concept about numbers and counting"))
    g.add(Atom(name="b", code="", description="beta concept about numbers and arithmetic"))
    g.add(Atom(name="c", code="", description="gamma unrelated concept about weather and rain"))

    assert g.atoms["a"].confidence == 0.5 and g.atoms["a"].access_count == 0
    record_success(g, ["a", "b"], "task")
    boosted = g.atoms["a"].confidence > 0.5 and g.atoms["a"].access_count == 1 and g.atoms["b"].confidence > 0.5
    edge_created = any((s, d) in (("a", "b"), ("b", "a")) for s, d, _ in g.edges)
    print(f"  [1] record_success   confidence boosted={boosted}  edge created={edge_created}   "
          f"{'PASS' if boosted and edge_created else 'FAIL'}")

    n0 = len(g)
    tnm = record_failure(g, "solve xyz", failed_code="return None", target_description="the xyz thing")
    trap_created = tnm in g and g.atoms[tnm].kind == "trap"
    grew = len(g) == n0 + 1
    trap_edges = [(s, d, r) for s, d, r in g.edges if s == tnm]
    low_strength = bool(trap_edges) and all(g.strength(s, d, r) < 0.5 for s, d, r in trap_edges)
    print(f"  [2] record_failure   trap created={trap_created}  grew={grew}  linked low-strength={low_strength}   "
          f"{'PASS' if trap_created and grew and low_strength else 'FAIL'}")

    act = spreading_activate(g, {"a": 1.0}, steps=3)
    reaches_b_via_strong_edge = act.get("b", 0.0) > 0.0
    print(f"  [3] spreading_activate   b (reached via boosted a->b edge) activation={act.get('b', 0.0):.3f}   "
          f"{'PASS' if reaches_b_via_strong_edge else 'FAIL'}")

    sub = glowing_subgraph(g, "tell me about numbers and arithmetic", steps=3, threshold=0.05)
    relevant_surfaced = "a" in sub or "b" in sub
    print(f"  [4] glowing_subgraph   {sub}   relevant surfaced={relevant_surfaced}   "
          f"{'PASS' if relevant_surfaced else 'FAIL'}")

    print("\n  selftest done")


if __name__ == "__main__":
    _selftest()
