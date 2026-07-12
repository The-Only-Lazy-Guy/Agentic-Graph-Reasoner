"""Reasoning-over-graph, SYNERGIZED — "graph reasons, LM realizes", made strong (propose-verify, not
predict) with every piece feeding the next through ONE shared QUERY.

The bottleneck was: compose is the LM's RARE spontaneous decision -> can't be trained. Fix: make
composition a guided SEARCH + VERIFY (the 18%->100% tool_memory pattern / DreamCoder wake), and let
the LM only write GLUE (call named atoms) — a trivial job that sidesteps the 0.21 realizer wall.

The synergy — one Query threads the whole loop (the user's ask):

    make_query(task)  ->  Query{text, vec}          (what the task NEEDS)
         |  vec  ---------------------------------->  RETRIEVE ranked atoms (MGRetriever / ranker)
         |  text ---------------------------------->  a FEATURE in the realizer prompt (LM knows what
         |                                             it's composing) -> glue-realize (best-of-N)
         v
      VERIFY (execution) -> keep the composition that passes
         |
      GROW edges: task-concept `uses` atoms, atom `depend` atom  (graph_grower, health-gated)
         |                                             = the composition MEMORY that biases the NEXT
         `-------------------------------------------> search, and the trainable target for the query
                                                        ranker (verified task->atoms pairs).

Reuses: MGRetriever + MemoryGraph (v5.runtime.algo_graph_mg), graph_grower bridge (algo_graph_edits),
tool_compose.verify_fn, derive_reward, algo_graph_run (tasks/verify/extract). The query ranker
(traversal_ranker / memory_refiner) is the trainable upgrade of make_query — same interface.

  selftest (no model):  python -m v5.runtime.algo_graph_reason --selftest
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass

import numpy as np

from graph_core import MemoryGraph
from v5.runtime.algo_graph_edits import grow, propose_edits
from v5.runtime.algo_graph_mg import MGRetriever, _fn_name, seed_graph
from v5.runtime.algo_graph_run import _sig, _task_verify, _top_defs
from v5.runtime.derive_reward import _def_names
from v5.runtime.tool_memory import _extract_code


@dataclass
class Query:
    """What the task NEEDS — the shared thread: drives retrieval (vec), features the realizer (text),
    is the trainable target for the ranker (task -> the atoms that verified)."""
    text: str
    vec: np.ndarray


def make_query(task, embed_fn) -> Query:
    """v0: query = the task's own need statement. The trainable upgrade is traversal_ranker/
    memory_refiner (LM-hidden -> query vec), same (text, vec) interface — swap in without changing
    the loop. Kept a seam so the ranker's query can replace this verbatim."""
    text = task.text
    vec = np.asarray(list(embed_fn({"q": text}).values())[0], dtype=np.float32)
    return Query(text=text, vec=vec)


def _realize_prompt(task, query_text: str, candidates: list[tuple[str, str]]) -> str:
    """Glue-realizer prompt: the QUERY is a FEATURE (the LM is TOLD what it needs), the atoms are the
    building blocks. The LM only writes glue (call the named atoms), never derives content."""
    parts = [task.text, f"\nWhat this needs: {query_text}"]
    if candidates:
        parts.append("Building blocks already available (CALL them by name — do NOT re-implement):")
        for name, code in candidates:
            parts.append(f"  {_sig(code) or name}")
        parts.append("COMPOSE these to solve it (wire their outputs together).")
    parts.append(f"Write `{task.name}(...)` in ONE Python code block.")
    return "\n\n".join(parts)


def search_compose(task, retriever: MGRetriever, gen_fn, query: Query, k: int = 6, samples: int = 8,
                   min_cos: float = 0.2):
    """Guided SEARCH + VERIFY: retrieve ranked atoms (by the query vec), glue-realize best-of-N
    (query text as a feature), verify each by execution, keep a VERIFIED one — preferring one that
    actually composed a retrieved atom. Returns (code, used_atoms, verified)."""
    ranked = retriever.retrieve_vec(query.vec, k=k, min_cos=min_cos)   # [(fn_name, code)]
    names = [n for n, _ in ranked]
    prompt = _realize_prompt(task, query.text, ranked)
    pick, fallback = None, ("", [], False)
    for gen in gen_fn([prompt] * samples):
        code = _extract_code(gen)
        called = [n for n in names if n not in _def_names(code)
                  and re.search(rf"\b{re.escape(n)}\s*\(", code)]
        deps = "\n\n".join(c for n, c in ranked if n in called)
        if _task_verify(task, code, deps):                   # VERIFY grounds the search
            if called and (pick is None):                    # prefer a composing solution
                pick = (code, called, True)
            elif fallback[0] == "" or not fallback[2]:
                fallback = (code, called, True)
        elif fallback[0] == "":
            fallback = (code, called, False)
    return pick or fallback


def reason_grow(task, graph_path: str, out_path: str, retriever: MGRetriever, gen_fn,
                embed_fn, concept: str = "concept_algorithms", k: int = 6, samples: int = 8,
                min_cos: float = 0.2):
    """One reasoning step: query -> search+verify -> if it composed, GROW edges (the composition
    memory) so the next search is biased by what worked."""
    query = make_query(task, embed_fn)
    code, used, verified = search_compose(task, retriever, gen_fn, query, k=k, samples=samples,
                                          min_cos=min_cos)
    grown = None
    if verified and used:                                    # verified COMPOSITION -> edges
        sol_id = f"impl_{task.name}"
        stores = [(sol_id, code, f"solves: {task.text[:60]}")]
        edges = [(sol_id, concept, "part_of")] + [(sol_id, f"impl_{a}", "depend") for a in used]
        grown = grow(graph_path, out_path, propose_edits(f"reason_{task.name}", stores, edges))
    return dict(name=task.name, verified=verified, used=used, code=code, grown=grown)


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST (no model) — the SYNERGY: one query -> retrieve + realizer-feature -> search+verify ->
# a COMPOSE-NECESSARY task solved by composing TWO atoms -> edges grow (composition memory)
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    import json
    import tempfile
    from pathlib import Path
    from v5.memory.store import make_fake_embedder
    from v5.runtime.algo_graph_run import MBPPTask
    print("algo_graph_reason --selftest: query -> retrieve+feature -> search+verify(compose 2 atoms) "
          "-> grow edges\n")
    embed = make_fake_embedder()

    IS_PRIME = ("def is_prime(n):\n    if n < 2:\n        return False\n    i = 2\n"
                "    while i * i <= n:\n        if n % i == 0:\n            return False\n"
                "        i += 1\n    return True")
    DIGIT_SUM = "def digit_sum(n):\n    return sum(int(c) for c in str(abs(n)))"

    with tempfile.TemporaryDirectory() as td:
        # seed a graph that ALREADY has the two atoms (the search needs a candidate set)
        gp = str(Path(td) / "g.json")
        nodes = [{"id": "concept_number_theory", "text": "number theory", "node_type": "concept"},
                 {"id": "impl_is_prime", "text": "prime test — is n prime", "node_type": "implementation",
                  "metadata": {"code": IS_PRIME}},
                 {"id": "impl_digit_sum", "text": "digit sum of an integer", "node_type": "implementation",
                  "metadata": {"code": DIGIT_SUM}}]
        Path(gp).write_text(json.dumps({"metadata": {}, "nodes": nodes, "edges": []}))
        g = MemoryGraph.load_json(gp)
        retr = MGRetriever(g, embed)

        # a COMPOSE-NECESSARY task: sum of digit-sums of the primes in a list (needs BOTH atoms)
        task = MBPPTask(
            "sum_digitsum_primes",
            "sum of digit-sums of the prime numbers in a list. needs: prime test and digit sum",
            ["assert sum_digitsum_primes([11, 4, 23]) == (1+1) + (2+3)",   # primes 11,23 -> 2 + 5 = 7
             "assert sum_digitsum_primes([8, 9, 10]) == 0"])               # no primes -> 0

        # the QUERY threads the loop: it retrieves the atoms AND features the realizer prompt
        q = make_query(task, embed)
        ranked = retr.retrieve_vec(q.vec, k=6, min_cos=-1.0)   # fake embedder: top-k regardless of score
        assert {n for n, _ in ranked} >= {"is_prime", "digit_sum"}, f"query retrieved atoms: {ranked}"
        p = _realize_prompt(task, q.text, ranked)
        assert q.text in p and "is_prime" in p and "digit_sum" in p, "query is a realizer FEATURE"
        print("  [1] one query -> retrieves BOTH atoms + is a realizer feature -> PASS")

        # glue-realizer stub: composes the two named atoms (the LM only writes glue)
        def _glue(prompts):
            return ["```python\ndef sum_digitsum_primes(lst):\n"
                    "    return sum(digit_sum(x) for x in lst if is_prime(x))\n```"] * len(prompts)

        code, used, verified = search_compose(task, retr, _glue, q, min_cos=-1.0)
        assert verified and set(used) == {"is_prime", "digit_sum"}, f"composed BOTH atoms: {used}"
        print(f"  [2] search+verify: compose-necessary task solved by GLUING 2 atoms {sorted(used)} -> PASS")

        # verified composition -> GROW edges (the composition memory): sol depend is_prime + digit_sum
        out = str(Path(td) / "g2.json")
        res = reason_grow(task, gp, out, retr, _glue, embed, concept="concept_number_theory", min_cos=-1.0)
        assert res["verified"] and res["grown"]["gate_passed"], res
        g2 = MemoryGraph.load_json(out)
        assert g2.edge_between("impl_sum_digitsum_primes", "impl_is_prime") is not None
        assert g2.edge_between("impl_sum_digitsum_primes", "impl_digit_sum") is not None
        print(f"  [3] verified composition -> {res['grown']['edit_stats']['edge_edits']} depend/part_of "
              f"EDGES grown (composition memory) -> PASS")

    print("\n  ALGO_GRAPH_REASON SELFTEST -> PASS")
    return True


def main():
    ap = argparse.ArgumentParser(description="Reasoning-over-graph: guided search + verify + glue-realizer.")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
