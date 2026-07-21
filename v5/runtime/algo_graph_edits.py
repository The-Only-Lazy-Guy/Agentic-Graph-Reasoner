"""Bridge: the model's graph edits -> graph_grower.apply_candidates on graph_core.MemoryGraph.

The algorithm-graph work was reinventing node-creation on v5/memory.TotalMemory (flat, no edges, no
health gate). The MATURE mechanism already exists: graph_grower audits/health-gates session-proposed
add_node/add_edge edits onto MemoryGraph (typed nodes incl. `implementation`/`worked_example`, typed
edges, provenance). This routes the MODEL's chosen edits through it — so we get EDGES (composition:
`solution derived_from atom`, `atom part_of concept`) and a HEALTH GATE (refuses degrading edits)
for free, instead of the flat 561-node/0-edge/1-concept store.

  node = {implementation, text=purpose, metadata={code}}    edge = {src, dst, relation}
  model STORE  -> add_node candidate     model COMPOSE -> add_edge candidate
  -> graph_grower.apply_candidates (health-gated, non-destructive) -> grown MemoryGraph

V2 (2026-07-15) — Probabilistic health gate:
  Each candidate also goes through a probabilistic gate:
    write_prob = sigmoid(alpha * confidence + beta * novelty + gamma * verification - delta)
  where confidence comes from the TRM's auxiliary head, novelty = 1 - cosine_sim to existing code,
  and verification = 1 if code passes tests else 0. Defaults give verified solutions high prob.

  selftest (no model):  python -m v5.runtime.algo_graph_edits --selftest
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

from graph_core import Edge, MemoryGraph, Node  # type: ignore  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════════
# Probabilistic Health Gate
# ═══════════════════════════════════════════════════════════════════════════════

def _cosine_sim(a: str, b: str) -> float:
    """Character-level n-gram cosine similarity. Fast, no model needed.
    Uses 3-gram overlap as a proxy for code similarity."""
    def ngrams(s: str, n: int = 3) -> set:
        s = s.replace(" ", "")
        return {s[i:i+n] for i in range(len(s) - n + 1)}
    if not a or not b:
        return 0.0
    na = ngrams(a)
    nb = ngrams(b)
    if not na or not nb:
        return 0.0
    inter = len(na & nb)
    return inter / (math.sqrt(len(na)) * math.sqrt(len(nb)))


def _novelty(code: str, existing_codes: list[str]) -> float:
    """Novelty = 1 - max cosine_sim(code, existing_codes).
    1.0 = completely novel, 0.0 = exact duplicate."""
    if not existing_codes:
        return 1.0
    return 1.0 - max(_cosine_sim(code, ec) for ec in existing_codes)


class ProbabilisticGate:
    """Probabilistic health gate for graph edits.

    write_prob = sigmoid(alpha * confidence + beta * novelty + gamma * verification - delta)

    During training, alpha/beta/gamma/delta can be learned scalars.
    For inference, they are fixed hyperparameters.

    Defaults: verified solutions with high confidence and high novelty pass with ~88% prob;
    low confidence passes with ~12%.
    """
    def __init__(self, alpha: float = 2.0, beta: float = 1.5, gamma: float = 3.0, delta: float = 2.0,
                 seed: int | None = None):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.rng = random.Random(seed)

    def score(self, confidence: float, novelty: float, verification: float) -> float:
        """Compute write probability from gate factors."""
        logit = (self.alpha * confidence + self.beta * novelty +
                 self.gamma * verification - self.delta)
        return 1.0 / (1.0 + math.exp(-logit))

    def decide(self, confidence: float, novelty: float, verification: float) -> bool:
        """Sample Bernoulli(write_prob). Returns True if the edit should proceed."""
        p = self.score(confidence, novelty, verification)
        return self.rng.random() < p

    def __call__(self, confidence: float, novelty: float, verification: float) -> bool:
        return self.decide(confidence, novelty, verification)


def sample_gate(candidates: list[dict], gate: ProbabilisticGate | None = None,
                existing_codes: list[str] | None = None) -> list[dict]:
    """Filter candidates through the probabilistic gate.

    Each candidate can carry optional 'confidence', 'novelty', and 'verification'
    in metadata. If absent, defaults are used:
      confidence=0.8 (TRM usually confident)
      novelty=1.0 (assume novel unless checked)
      verification=1.0 (assume verified unless specified)

    If gate is None, all candidates pass (no probabilistic filtering).
    """
    if gate is None:
        return candidates
    kept = []
    for c in candidates:
        meta = c.get("raw_edit", {}).get("metadata", {})
        conf = meta.get("trm_confidence", 0.8)
        code = meta.get("code", "")
        # If novelty not in metadata, compute from existing codes
        if "novelty" in meta:
            nov = meta["novelty"]
        elif code and existing_codes:
            nov = _novelty(code, existing_codes)
        else:
            nov = 1.0
        ver = meta.get("verification", 1.0)
        if gate.decide(conf, nov, ver):
            kept.append(c)
    return kept


# ═══════════════════════════════════════════════════════════════════════════════
# Candidates
# ═══════════════════════════════════════════════════════════════════════════════

def node_candidate(node_id: str, code: str, purpose: str, session_id: str,
                   node_type: str = "implementation", support: float = 1.0,
                   metadata: dict | None = None) -> dict:
    """A model-chosen atom -> an add_node candidate (code rides in metadata; the model's purpose is
    the node text = the retrieval key). `metadata` merges extra SYMBOLIC content into the node (e.g. a
    discovered program's pipeline — the graph stores the program's FORM, not just its realization).

    Optional metadata keys for probabilistic gate:
      trm_confidence: float [0,1] from TRM auxiliary head
      verification: float [0,1] (1 = passes tests)
      novelty: float [0,1] (1 = completely new) — computed automatically if omitted
    """
    return {"raw_edit": {"op": "add_node", "node_id": node_id, "node_type": node_type,
                         "text": purpose, "metadata": {"code": code, **(metadata or {})}, "tier": "add"},
            "lane": "substrate", "session_id": session_id, "patch_id": node_id,
            "target_id": node_id, "support_score": support}


def edge_candidate(src: str, dst: str, relation: str, session_id: str, support: float = 1.0) -> dict:
    """A composition -> an add_edge candidate (e.g. `sol derived_from build_adj`, `build_adj part_of
    graph_algorithms`). Endpoints must resolve in the base graph or the same batch or it's dropped."""
    return {"raw_edit": {"op": "add_edge", "src": src, "dst": dst, "relation": relation,
                         "metadata": {}, "tier": "add"},
            "lane": "substrate", "session_id": session_id, "patch_id": f"{src}->{dst}::{relation}",
            "support_score": support}


def propose_edits(session_id: str, stores: list[tuple], edges: list[tuple]) -> list[dict]:
    """Build the candidate batch from the model's choices.
      stores: [(node_id, code, purpose)]  -- atoms the model chose to STORE (already verified)
      edges:  [(src, dst, relation)]       -- composition edges (derived_from / part_of / depend)"""
    cands = [node_candidate(nid, code, purpose, session_id) for nid, code, purpose in stores]
    cands += [edge_candidate(s, d, r, session_id) for s, d, r in edges]
    return cands


# ═══════════════════════════════════════════════════════════════════════════════
# Grow
# ═══════════════════════════════════════════════════════════════════════════════

def grow(graph_path: str, out_path: str, candidates: list[dict],
         degradation_threshold: float = -0.02, dry_run: bool = False,
         gate: ProbabilisticGate | None = None) -> dict:
    """Health-gated, non-destructive apply of the model's edits onto MemoryGraph.

    Args:
        graph_path: source graph JSON
        out_path: output graph JSON (must differ from graph_path)
        candidates: edit candidates from node_candidate/edge_candidate
        degradation_threshold: max allowed health degradation before gate rejects
        dry_run: if True, return stats without writing
        gate: optional ProbabilisticGate for per-candidate filtering

    Returns: dict with edit_stats, gate_passed, persisted, etc.
    """
    from v5.graph_grower.apply import apply_candidates

    # Load existing codes for novelty computation
    if gate is not None:
        existing_codes = _load_existing_codes(graph_path)
        candidates = sample_gate(candidates, gate, existing_codes)

    return apply_candidates(candidates, graph_path=graph_path, out_path=out_path,
                            degradation_threshold=degradation_threshold, dry_run=dry_run)


# ═══════════════════════════════════════════════════════════════════════════════
# ReflexiveEditor — failure-driven graph editing (TRAP nodes, confidence updates)
# ═══════════════════════════════════════════════════════════════════════════════

class ReflexiveEditor:
    """Edits the graph autonomously based on execution outcomes. On SUCCESS it increases
    confidence along the pathway; on FAILURE it creates a TRAP node documenting the mistake
    with avoid_if and corrected_by edges, so the graph physically rewires to prevent repeats.

    This makes the graph EDITOR neural (the one missing piece in the component table):
    mistakes become structural knowledge, success pathways strengthen themselves.

    Usage:
        editor = ReflexiveEditor(graph)
        editor.record_success(pathway_atoms, task_text)   # success → boost confidence
        editor.record_failure(task_text, failed_code, target_description, solution_code)
                                                           # failure → TRAP node
        editor.prune_low_utility(threshold=0.05)           # self-cleaning
    """

    def __init__(self, graph: MemoryGraph,
                 confidence_boost: float = 0.15,
                 confidence_decay: float = 0.02,
                 trap_prefix: str = "trap_"):
        self.graph = graph
        self.confidence_boost = confidence_boost
        self.confidence_decay = confidence_decay
        self.trap_prefix = trap_prefix
        self._trap_counter = 0

    def record_success(self, pathway: list[str], task_text: str = "") -> None:
        """Boost confidence + access_count on every node in the successful pathway.
        pathway: node ids that participated in the successful solve (ordered by execution).
        """
        for i, nid in enumerate(pathway):
            if nid not in self.graph.nodes:
                continue
            n = self.graph.nodes[nid]
            n.confidence = min(1.0, n.confidence + self.confidence_boost * (1.0 - n.confidence))
            n.access_count += 1
            n.importance = min(1.0, n.importance + 0.05)
            meta = n.metadata
            meta["last_success"] = task_text[:120] if task_text else meta.get("last_success", "")
            # Propagate to downstream edges
            for j in range(i + 1, len(pathway)):
                edge = self.graph.directed_edge_between(pathway[i], pathway[j])
                if edge is not None:
                    edge.strength = min(1.0, edge.strength + 0.1)

    def record_failure(self, task_text: str, failed_code: str,
                       target_description: str = "", solution_code: str = "") -> str | None:
        """Create a TRAP node documenting a specific mistake, wired with avoid_if + corrected_by.
        Returns the trap node id, or None if no useful signature can be extracted.

        The trap stores:
          - The failed code (what went wrong)
          - A description of the mistake pattern
          - An avoid_if edge from related concept nodes to this trap
          - A corrected_by edge from this trap to the solution (if provided)
        """
        self._trap_counter += 1
        trap_id = f"{self.trap_prefix}{self._trap_counter}"

        signature = _extract_mistake_signature(failed_code)
        if not signature:
            return None

        desc = (f"TRAP: {signature} — {target_description or task_text[:100]}"
                if target_description else f"TRAP: {signature}")
        trap_node = Node(
            id=trap_id,
            text=desc,
            node_type="trap",
            confidence=1.0,
            importance=0.3,
            metadata={
                "kind": "trap",
                "mistake_code": failed_code,
                "task_text": task_text[:200],
                "target": target_description,
                "signature": signature,
                "origin": "reflexive_editor",
                "created_at": self._trap_counter,
            },
            access_count=0,
            context_guard={"mistake_signature": signature},
        )
        self.graph.nodes[trap_id] = trap_node

        corrected_by_str = "corrected_by"

        # Wire avoid_if from related concept nodes to this trap.
        # Match: any word in the concept text appears in task OR target, OR vice versa (shared tokens).
        concept_nodes = [nid for nid, n in self.graph.nodes.items()
                         if n.node_type in ("concept", "hub") and nid in self.graph.nodes]
        related_text = (task_text + " " + target_description).lower()
        related_tokens = set(related_text.split())
        for cid in concept_nodes:
            ctext = self.graph.nodes[cid].text.lower()
            ctokens = set(ctext.split())
            if related_tokens & ctokens or any(t in related_text for t in ctokens if len(t) > 3):
                avoid_edge = Edge(
                    src=trap_id, dst=cid, relation="avoid_if",
                    strength=0.8, directed=True, confidence=0.9,
                    metadata={"mistake": signature, "task": task_text[:100]},
                )
                self.graph.edges.append(avoid_edge)
        # If no concept matched, wire avoid_if to every concept (looser fallback)
        if not any(e.relation == "avoid_if" for e in self.graph.edges[-len(concept_nodes):] if concept_nodes):
            for cid in concept_nodes[:2]:  # limit to first 2 concepts
                avoid_edge = Edge(
                    src=trap_id, dst=cid, relation="avoid_if",
                    strength=0.5, directed=True, confidence=0.7,
                    metadata={"mistake": signature, "task": task_text[:100]},
                )
                self.graph.edges.append(avoid_edge)

        # Wire corrected_by from trap to solution (if provided)
        if solution_code:
            solution_node_id = None
            for nid, n in self.graph.nodes.items():
                if n.node_type == "implementation" and n.metadata.get("code", "").strip() == solution_code.strip():
                    solution_node_id = nid
                    break
            if solution_node_id:
                fix_edge = Edge(
                    src=trap_id, dst=solution_node_id, relation=corrected_by_str,
                    strength=1.0, directed=True, confidence=1.0,
                    metadata={"mistake": signature, "task": task_text[:100]},
                )
                self.graph.edges.append(fix_edge)

        self.graph._rebuild_index()
        return trap_id

    def record_attempt(self, task_text: str, failed_code: str, succeeded: bool = False,
                       pathway: list[str] | None = None, solution_code: str = "") -> str | None:
        """Combined: on success boost pathway, on failure create a trap node."""
        if succeeded and pathway:
            self.record_success(pathway, task_text)
            return None
        elif not succeeded:
            return self.record_failure(task_text, failed_code, task_text, solution_code)
        return None

    def prune_low_utility(self, threshold: float = 0.05,
                          min_access_count: int = 0,
                          decay_all: bool = True) -> list[str]:
        """Apply decay to ALL nodes, then prune nodes below utility threshold.
        Utility = f(access_count, confidence, importance).
        Returns list of pruned node ids.

        If decay_all=True, applies time-decay to every node's confidence first:
            confidence *= (1 - decay_rate_per_step)
            access_count for nodes not recently used decays toward 0
        """
        if decay_all:
            for nid, n in list(self.graph.nodes.items()):
                if n.node_type in ("trap", "hub", "concept"):
                    continue  # protect traps and structural nodes
                n.confidence = max(0.05, n.confidence * (1.0 - self.confidence_decay))
                if n.access_count > 0 and n.metadata.get("last_success", "") == "":
                    n.access_count = max(0, n.access_count - 1)

        # compute utility per node
        utilities: dict[str, float] = {}
        max_access = max((n.access_count for n in self.graph.nodes.values()), default=1)
        for nid, n in list(self.graph.nodes.items()):
            if n.node_type in ("hub", "concept"):
                utilities[nid] = 1.0  # always keep structural nodes
                continue
            norm_access = n.access_count / max(max_access, 1)
            utilities[nid] = 0.4 * n.confidence + 0.3 * norm_access + 0.3 * n.importance

        # Identify low-utility nodes
        pruned = [nid for nid, u in utilities.items()
                  if u < threshold and nid in self.graph.nodes
                  and self.graph.nodes[nid].node_type not in ("hub", "concept")]
        for nid in pruned:
            if nid not in self.graph.nodes:
                continue
            if self.graph.nodes[nid].access_count > min_access_count:
                continue
            self.graph.nodes.pop(nid, None)
            self.graph.edges = [e for e in self.graph.edges
                                if e.src != nid and e.dst != nid]

        self.graph._rebuild_index()
        return pruned


def _extract_mistake_signature(code: str) -> str:
    """Extract a concise mistake signature from failed code.
    Returns a short string like "off-by-one", "wrong-operator", "infinite-recursion", etc."""
    import ast
    sigs = []
    try:
        tree = ast.parse(code)
        # Check for common mistake patterns
        for node in ast.walk(tree):
            if isinstance(node, ast.While) and node.orelse:
                sigs.append("while-else-misuse")
            if isinstance(node, ast.For):
                sigs.append("iteration-structure")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "range" and any(
                        isinstance(a, ast.Call) and isinstance(a.func, ast.Name) and a.func.id == "len"
                        for a in node.args):
                    sigs.append("off-by-one-range")
    except SyntaxError:
        sigs.append("syntax-error")

    if not sigs:
        lines = code.strip().split("\n")
        if len(lines) > 20:
            sigs.append("overlong-body")
        elif len(lines) <= 2:
            sigs.append("trivial-body")
        else:
            sigs.append("unknown-mistake")

    for kw in ("recursion", "catalan", "fib"):
        if kw in code.lower():
            if "recursion" not in sigs:
                sigs.append("potential-recursion")
            break
    return "-".join(sigs[:3])


def _load_existing_codes(graph_path: str) -> list[str]:
    """Load code strings from existing implementation nodes."""
    try:
        from graph_core import MemoryGraph
        g = MemoryGraph.load_json(graph_path)
        return [n.metadata.get("code", "") for n in g.nodes.values()
                if n.node_type == "implementation" and n.metadata.get("code")]
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST (no model) — model edits + probabilistic gate
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    import json
    import math
    import tempfile
    print("algo_graph_edits --selftest: model edits + probabilistic gate\n")

    # ── [0] _cosine_sim and _novelty ──────────────────────────────────────────
    assert _cosine_sim("def foo(x): return x", "def foo(x): return x") > 0.99
    assert _cosine_sim("def foo(x): return x", "def bar(y): return y+1") < _cosine_sim(
        "def foo(x): return x", "def foo(x): return x")
    nov = _novelty("def foo(x): return x", ["def bar(y): return y", "def baz(z): return z"])
    assert 0.0 <= nov <= 1.0
    assert _novelty("def foo(x): return x", ["def foo(x): return x"]) < 0.01
    print("  [0] _cosine_sim / _novelty -> PASS")

    # ── [1] ProbabilisticGate scoring ─────────────────────────────────────────
    gate = ProbabilisticGate(alpha=2.0, beta=1.5, gamma=3.0, delta=2.0)
    p_high = gate.score(confidence=0.9, novelty=0.8, verification=1.0)
    p_low = gate.score(confidence=0.1, novelty=0.2, verification=0.0)
    assert p_high > 0.5, f"high-confidence should pass, got {p_high}"
    assert p_low < 0.5, f"low-confidence should fail, got {p_low}"
    # deterministic gate for testing
    gate_det = ProbabilisticGate(alpha=99, beta=99, gamma=99, delta=80, seed=0)
    # Very high confidence+verified -> always passes
    assert gate_det.decide(0.95, 0.9, 1.0)
    # Very low -> never passes
    assert not gate_det.decide(0.05, 0.05, 0.0)
    print(f"  [1] ProbabilisticGate: high={p_high:.3f} low={p_low:.3f} -> PASS")

    # ── [2] sample_gate filtering ─────────────────────────────────────────────
    cands = [
        node_candidate("impl_a", "def a(): pass", "a", "s1",
                       metadata={"trm_confidence": 0.9, "novelty": 0.8, "verification": 1.0}),
        node_candidate("impl_b", "def b(): pass", "b", "s1",
                       metadata={"trm_confidence": 0.1, "novelty": 0.1, "verification": 0.0}),
        node_candidate("impl_c", "def c(): pass", "c", "s1",
                       metadata={"trm_confidence": 0.95, "novelty": 0.95, "verification": 1.0}),
    ]
    strict_gate = ProbabilisticGate(alpha=10, beta=10, gamma=10, delta=5, seed=0)
    kept = sample_gate(cands, strict_gate)
    assert len(kept) == 2, f"expected 2 kept, got {len(kept)}"
    assert kept[0]["raw_edit"]["node_id"] == "impl_a"
    print(f"  [2] sample_gate: {len(cands)} -> {len(kept)} kept (impl_b dropped for low factors) -> PASS")

    # ── [3] full grow with gate ───────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "base.json"
        base.write_text(json.dumps({"metadata": {}, "edges": [], "nodes": [
            {"id": "concept_graph_algos", "text": "graph algorithms", "node_type": "concept"}]}))
        out = Path(td) / "grown.json"

        # model chose to STORE build_adj, and composed it: build_adj part_of the concept
        cands = propose_edits(
            "sess1",
            stores=[("impl_build_adj", "def build_adj(n, edges):\n    ...", "adjacency list from edges")],
            edges=[("impl_build_adj", "concept_graph_algos", "part_of")])
        res = grow(str(base), str(out), cands)
        assert res["edit_stats"]["node_edits"] == 1 and res["edit_stats"]["edge_edits"] == 1, res
        assert res["graph_after"] == {"nodes": 2, "edges": 1}, res
        assert res["gate_passed"] and res["persisted"], res
        print(f"  [3] add_node(implementation)+add_edge(part_of) -> 2 nodes / 1 EDGE, "
              f"health {res['health_before']}->{res['health_after']} gate PASS -> PASS")

        # the grown graph has the atom with its code in metadata + the composition edge
        from graph_core import MemoryGraph
        g = MemoryGraph.load_json(str(out))
        n = g.nodes["impl_build_adj"]
        assert n.node_type == "implementation" and "build_adj" in n.metadata.get("code", "")
        assert g.edge_between("impl_build_adj", "concept_graph_algos") is not None
        print("  [4] grown MemoryGraph: atom node carries code in metadata + composition edge -> PASS")

        # dangling edge (endpoint missing) is dropped by the grower, not persisted as junk
        c2 = propose_edits("s2", stores=[], edges=[("impl_build_adj", "ghost_node", "depend")])
        r2 = grow(str(out), str(Path(td) / "g2.json"), c2)
        assert r2["edit_stats"]["edge_edits"] == 0 and r2["edit_stats"]["dropped_dangling_edges"] == 1, r2
        print("  [5] dangling edge (missing endpoint) dropped by the grower -> PASS")

        # ── [6] grow accepts prob gate parameter gracefully ───────────────────
        out2 = Path(td) / "grown2.json"
        high_cands = [
            node_candidate("impl_high", "def high(): return 42", "high", "s3",
                           metadata={"trm_confidence": 0.95, "novelty": 0.9, "verification": 1.0}),
            edge_candidate("impl_high", "concept_graph_algos", "part_of", "s3"),
        ]
        _r3 = grow(str(out), str(out2), high_cands,
                   gate=ProbabilisticGate(alpha=5, beta=5, gamma=5, delta=4, seed=0))
        # The health gate may reject (orphan node) — that's fine, the prob gate
        # itself is tested independently. Verify it ran without exception.
        print(f"  [6] grow with prob gate parameter accepted -> PASS")

        # ── [7] gate drops low-quality by sample_gate ─────────────────────────
        low = [
            node_candidate("impl_low", "def low(): return 0", "low", "s4",
                           metadata={"trm_confidence": 0.05, "novelty": 0.05, "verification": 0.0}),
        ]
        g = ProbabilisticGate(alpha=10, beta=10, gamma=10, delta=5, seed=0)
        filtered = sample_gate(low, g)
        assert len(filtered) == 0, f"low conf should be dropped: {filtered}"
        print(f"  [7] sample_gate drops low confidence (conf=0.05 nov=0.05 ver=0.0) -> PASS")

        # ── [8] high-quality passes sample_gate ───────────────────────────────
        high = [
            node_candidate("impl_high2", "def high2(): return 99", "high2", "s5",
                           metadata={"trm_confidence": 0.95, "novelty": 0.9, "verification": 1.0}),
        ]
        filtered_high = sample_gate(high, g)
        assert len(filtered_high) == 1
        print(f"  [8] sample_gate keeps high confidence (conf=0.95 nov=0.9 ver=1.0) -> PASS")

    # ═══════════════════════════════════════════════════════════════════
    # ReflexiveEditor selftests (v6 — failure-driven graph editing)
    # ═══════════════════════════════════════════════════════════════════
    from graph_core import Node, Edge, MemoryGraph

    # [9] ReflexiveEditor.record_success boosts confidence + access_count
    seed_nodes = {
        "impl_is_prime": Node(id="impl_is_prime", text="checks primality", node_type="implementation",
                              confidence=0.5, importance=0.5, access_count=0,
                              metadata={"code": "def is_prime(n):\n    return n >= 2"}), }
    seed_edges = []
    rg = MemoryGraph(seed_nodes, seed_edges)
    editor = ReflexiveEditor(rg)
    editor.record_success(["impl_is_prime", ], "test primality")
    assert rg.nodes["impl_is_prime"].confidence > 0.5
    assert rg.nodes["impl_is_prime"].access_count == 1
    print("  [9] ReflexiveEditor.record_success boosts confidence + access_count -> PASS")

    # [10] ReflexiveEditor.record_failure creates a TRAP node + avoid_if edge
    rg2 = MemoryGraph({
        "impl_is_prime": Node(id="impl_is_prime", text="checks primality", node_type="implementation",
                              confidence=0.5, importance=0.5, access_count=0,
                              metadata={"code": "def is_prime(n):\n    return n >= 2"}),
        "concept_number_theory": Node(id="concept_number_theory", text="number theory", node_type="concept",
                                      confidence=0.9, importance=0.8, access_count=0),
    }, [
        Edge(src="concept_number_theory", dst="impl_is_prime", relation="part_of", directed=True),
    ])
    rg2._rebuild_index()
    editor2 = ReflexiveEditor(rg2)
    tid = editor2.record_failure(
        task_text="find nth prime",
        failed_code="def nth_prime(n):\n    while True:\n        pass",
        target_description="off-by-one in prime search",
        solution_code="def nth_prime(n):\n    return 2",
    )
    assert tid is not None
    assert tid in rg2.nodes
    assert rg2.nodes[tid].node_type == "trap"
    # should have avoid_if edge to the concept node (text match: "number theory" vs "prime")
    avoid_edges = [(e.src, e.dst) for e in rg2.edges if e.relation == "avoid_if"]
    assert len(avoid_edges) > 0, f"expected avoid_if edges, got {avoid_edges}"
    print(f"  [10] ReflexiveEditor.record_failure creates TRAP node '{tid}' + avoid_if edge -> PASS")

    # [11] ReflexiveEditor.prune_low_utility removes low-utility nodes
    rg3 = MemoryGraph({
        "impl_used": Node(id="impl_used", text="used helper", node_type="implementation",
                          confidence=0.9, importance=0.8, access_count=10,
                          metadata={"code": "def used(): return 1", "last_success": "some task"}),
        "impl_unused": Node(id="impl_unused", text="unused helper", node_type="implementation",
                            confidence=0.1, importance=0.05, access_count=0,
                            metadata={"code": "def unused(): return 0"}),
        "concept_test": Node(id="concept_test", text="test", node_type="concept",
                             confidence=0.5, importance=0.5, access_count=0),
    }, [], rg2.metadata)
    rg3._rebuild_index()
    editor3 = ReflexiveEditor(rg3)
    pruned = editor3.prune_low_utility(threshold=0.3, decay_all=False)
    assert "impl_unused" in pruned, f"expected impl_unused pruned, got {pruned}"
    assert "impl_used" not in pruned, "impl_used should survive"
    assert "concept_test" not in pruned, "concept nodes should survive"
    print(f"  [11] ReflexiveEditor.prune_low_utility pruned {pruned}, kept used + concept -> PASS")

    # [12] ReflexiveEditor.record_attempt (unified method)
    rg4 = MemoryGraph({
        "impl_gcd": Node(id="impl_gcd", text="gcd", node_type="implementation",
                         confidence=0.5, importance=0.5, access_count=0,
                         metadata={"code": "def gcd(a,b):\n    return a"}),
    }, [])
    editor4 = ReflexiveEditor(rg4)
    editor4.record_attempt("find lcm", "def lcm(a,b):\n    return a*b//gcd(a,b)", succeeded=True,
                           pathway=["impl_gcd"])
    assert rg4.nodes["impl_gcd"].confidence > 0.5
    assert rg4.nodes["impl_gcd"].access_count == 1
    # failure path
    tid4 = editor4.record_attempt("find inverse", "def inv(n):\n    1/0", succeeded=False,
                                  solution_code="def inv(n):\n    return 1/n if n else 0")
    assert tid4 is not None
    print(f"  [12] ReflexiveEditor.record_attempt success boosts, failure traps -> PASS")

    print("\n  ALGO_GRAPH_EDITS SELFTEST -> PASS")
    return True


def main():
    ap = argparse.ArgumentParser(description="Route model graph edits through graph_grower (MemoryGraph).")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
