"""Meta-learning nodes (#54) — the graph stores not just CODE atoms but POLICIES: natural-language
know-how about HOW to approach a kind of task ("this input type -> decompose thus", "when you see X,
do Y"). Q4's novel idea. Representation-open: a policy node is form='nl' (node_type='policy'), never
executed — it's RETRIEVED and INJECTED as a strategy hint into the realizer prompt, and CREDITED by the
downstream verified outcome (representation-agnostic: it doesn't have to be executable to be graded).

  policy node  = {node_type: policy, text: the policy, metadata:{form: nl}}
  retrieve     = MetaRetriever (cosine of policy text vs the task) — separate lane from impl atoms
  use          = inject_policies() prepends the hints to the realize prompt
  credit       = meta_reward(used, verified): used+solved -> reward, used+failed -> penalty (STRENGTHEN
                 / WEAKEN via graph_edits, same lifecycle as code atoms)

This is the meta layer: a policy can encode a DECOMPOSITION ("solve A then B"), a reaction to a kind of
information, or a remembered approach for a task seen before — knowledge the model can't regenerate but
can be handed.

  selftest (no model):  python -m v5.runtime.algo_meta --selftest
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

POLICY_TYPE = "policy"


def policy_node(pid: str, text: str, tags: str = "") -> dict:
    """A model-chosen POLICY -> a graph node (NL form, not code)."""
    return {"id": pid, "text": text, "node_type": POLICY_TYPE,
            "metadata": {"form": "nl", "tags": tags}}


class MetaRetriever:
    """Retrieve POLICY nodes by cosine of their text vs the task — a separate lane from the impl-atom
    retriever (MGRetriever). Same embed_fn, different node type."""

    def __init__(self, graph, embed_fn):
        self.graph, self.embed_fn = graph, embed_fn
        pol = [(nid, n) for nid, n in graph.nodes.items() if n.node_type == POLICY_TYPE]
        self.ids = [nid for nid, _ in pol]
        if self.ids:
            vecs = embed_fn({nid: n.text for nid, n in pol})
            self.mat = np.asarray([vecs[i] for i in self.ids], dtype=np.float32)
            self.mat /= (np.linalg.norm(self.mat, axis=1, keepdims=True) + 1e-9)
        else:
            self.mat = np.zeros((0, 1), dtype=np.float32)

    def retrieve(self, query: str, k: int = 2, min_cos: float = 0.25):
        """Top-k policy TEXTS for a task query (the hints to inject)."""
        if not self.ids:
            return []
        q = np.asarray(list(self.embed_fn({"q": query}).values())[0], dtype=np.float32)
        q /= (np.linalg.norm(q) + 1e-9)
        scores = self.mat @ q
        out = []
        for j in np.argsort(-scores)[:k]:
            if scores[j] >= min_cos:
                out.append((self.ids[j], self.graph.nodes[self.ids[j]].text))
        return out


def inject_policies(prompt: str, policies: list[str]) -> str:
    """Prepend retrieved policy hints to the realizer prompt (the model is TOLD how to approach it)."""
    if not policies:
        return prompt
    hints = "\n".join(f"  - {p}" for p in policies)
    return f"Strategy hints from memory (how to approach this):\n{hints}\n\n{prompt}"


def meta_reward(used: bool, verified: bool) -> float:
    """Policy credit is representation-AGNOSTIC — graded by the DOWNSTREAM verified outcome, not by
    executability. Used + solved -> strengthen; used + failed -> weaken; unused -> neutral."""
    if not used:
        return 0.0
    return 0.30 if verified else -0.20


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST (no model) — a policy is retrieved + injected, CHANGES the outcome (solve with / fail
# without), and its credit follows the downstream verified result
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    import json
    import tempfile
    from pathlib import Path
    from graph_core import MemoryGraph
    from v5.memory.store import make_fake_embedder
    from v5.runtime.algo_graph_run import verify_asserts
    print("algo_meta --selftest: policy retrieved + injected -> changes outcome + downstream-graded\n")
    embed = make_fake_embedder()

    with tempfile.TemporaryDirectory() as td:
        # a graph carrying ONE policy node: the know-how for a 'mystery' task the model can't guess
        gp = str(Path(td) / "g.json")
        nodes = [{"id": "concept_algorithms", "text": "algorithms", "node_type": "concept"},
                 policy_node("policy_mystery",
                             "for a mystery(x) task: it means return x doubled then plus one",
                             tags="mystery")]
        Path(gp).write_text(json.dumps({"metadata": {}, "nodes": nodes, "edges": []}))
        g = MemoryGraph.load_json(gp)
        mr = MetaRetriever(g, embed)

        got = mr.retrieve("solve the mystery task", k=2, min_cos=-1.0)
        assert got and got[0][0] == "policy_mystery", f"policy retrieved: {got}"
        policies = [t for _, t in got]
        print(f"  [1] policy retrieved for the task: '{policies[0][:46]}...' -> PASS")

        # base realizer prompt (no spec the model could guess) + the injected policy
        base_prompt = "Write `mystery(x)` in one code block."
        injected = inject_policies(base_prompt, policies)
        assert "Strategy hints" in injected and "doubled then plus one" in injected
        assert base_prompt in injected
        print("  [2] policy INJECTED as a strategy hint into the realizer prompt -> PASS")

        # the policy CHANGES the outcome: a stub that only knows the rule FROM the hint solves it
        tests = ["assert mystery(3) == 7"]                     # 3*2+1 = 7 — unguessable without the policy

        def _stub(prompt):
            if "doubled then plus one" in prompt:              # used the policy -> correct
                return "def mystery(x):\n    return x * 2 + 1"
            return "def mystery(x):\n    return x        # guessed, wrong"

        with_policy = verify_asserts(_stub(injected), tests)
        without = verify_asserts(_stub(base_prompt), tests)
        assert with_policy and not without, f"policy must flip the outcome (with={with_policy} without={without})"
        print("  [3] policy CHANGES the outcome: solved WITH the hint, failed WITHOUT -> PASS")

        # credit is downstream-graded (representation-agnostic): used+solved rewards, used+failed punishes
        assert meta_reward(True, True) > 0 and meta_reward(True, False) < 0 and meta_reward(False, True) == 0
        print(f"  [4] credit: used+solved={meta_reward(True,True):+.2f}, used+failed={meta_reward(True,False):+.2f}, "
              f"unused={meta_reward(False,True):+.2f} (downstream-graded, not executability) -> PASS")

    print("\n  ALGO_META SELFTEST -> PASS")
    return True


def main():
    ap = argparse.ArgumentParser(description="Meta-learning policy nodes (NL know-how, injected + credited).")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
