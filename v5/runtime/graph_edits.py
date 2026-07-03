"""Graph edits engine — the WRITE path of LGGN (behavior-space structural learning).

The read path is validated (refiner -> FiLM -> decode). This module is the amortization
mechanism: how the graph structurally improves from outcomes, not just grows. See
LGGN_DESIGN.md "GRAPH STRUCTURAL EDITS".

10 edit types over a behavior-space graph:
  per-outcome (observe):   MINT, STRENGTHEN, WEAKEN, CONNECT,
                           REFINE_EMBEDDING, REFINE_PRECONDITION
  maintenance (maintain):  MERGE, SPLIT, RETIRE, COMPOSE

Design invariants:
  - Poison gate preserved: minted nodes start below the reuse confidence floor;
    only outcome-confirmed STRENGTHEN makes them retrievable.
  - Every edit is an append-only EditRecord (observability: if the edits can't be
    audited, the graph isn't learning — it's mutating).
  - Two embeddings per node: behavior_emb (FiLM-config space, what the renderer
    reads) and precondition centroids (ctx space, what retrieval matches).
  - Failure is signal: WEAKEN + negative-precondition learning ("where NOT to apply").

Pure numpy, no model needed:

    python -m v5.runtime.graph_edits --selftest
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ── gates (mirrors lggn.py semantics) ────────────────────────────────────────────
CONF_FLOOR = 0.40      # below this: not retrievable for reuse (re-validation only)
MINT_CONF = 0.35       # minted nodes start UNPROVEN (< floor) — the poison gate
SEED_CONF = 0.50


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _kmeans2(X: np.ndarray, iters: int = 20, seed: int = 0):
    """Tiny 2-means (numpy only). Returns (labels, centroids)."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), 2, replace=False)
    C = X[idx].copy()
    labels = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        d0 = np.linalg.norm(X - C[0], axis=1)
        d1 = np.linalg.norm(X - C[1], axis=1)
        new = (d1 < d0).astype(int)
        if np.array_equal(new, labels) and _ > 0:
            break
        labels = new
        for k in (0, 1):
            if (labels == k).any():
                C[k] = X[labels == k].mean(axis=0)
    return labels, C


# ── node / graph ─────────────────────────────────────────────────────────────────

@dataclass
class BehaviorNode:
    """One operator in behavior space. behavior_emb = HOW it modulates the model
    (compressed FiLM configuration); precond_pos/neg = WHEN it applies (ctx space)."""
    op_id: str
    name: str
    behavior_emb: list                    # behavior space (bottleneck dim)
    precond_pos: Optional[list] = None    # ctx-space centroid of successful contexts
    precond_neg: Optional[list] = None    # ctx-space centroid of failed contexts
    confidence: float = MINT_CONF
    validation_count: int = 0
    failure_count: int = 0
    age: int = 0                          # outcomes seen since created
    source: str = "mint"
    retrievable: bool = True
    n_pos: int = 0                        # running-mean counters for centroids
    n_neg: int = 0
    parents: Optional[list] = None        # provenance for COMPOSE/MERGE


@dataclass
class Outcome:
    """One decode outcome — the observation that drives edits.

    trajectory: op_ids used in order ([] = DERIVE branch, decoder invented).
    behavior_emb: the behavior vector that actually ran (from the FiLM state,
                  compressed through BehaviorEncoder).
    ctx_emb: input features (issue+code context embedding).
    verified: external oracle outcome (Docker / tests)."""
    iid: str
    ctx_emb: np.ndarray
    behavior_emb: np.ndarray
    trajectory: List[str]
    verified: bool


@dataclass
class EditRecord:
    ts: float
    kind: str            # MINT|STRENGTHEN|WEAKEN|CONNECT|REFINE_EMBEDDING|...
    op_ids: list
    info: dict = field(default_factory=dict)


@dataclass
class EngineConfig:
    ema_alpha: float = 0.2         # REFINE_EMBEDDING step toward the state that worked
    neg_penalty: float = 0.5       # retrieval penalty weight for neg-precondition match
    conf_up: float = 0.3           # strengthen: conf += (1-conf)*conf_up
    conf_down: float = 0.6         # weaken: conf *= conf_down
    edge_gain: float = 1.0         # CONNECT weight per verified co-occurrence
    edge_decay: float = 0.995      # per-observe multiplicative edge decay
    merge_behavior_cos: float = 0.95
    merge_precond_cos: float = 0.90
    merge_min_val: int = 2         # both nodes must be validated to merge
    compose_min_count: int = 3     # bigram successes before COMPOSE mints
    split_min_usage: int = 8       # usage records before SPLIT considered
    split_gap: float = 0.4         # success-rate gap between ctx clusters to split
    retire_age: int = 50           # outcomes since creation
    retire_max_val: int = 0        # retire only if never validated...
    retire_conf: float = CONF_FLOOR  # ...and still below reuse floor
    usage_cap: int = 64            # per-node usage history bound


class BehaviorGraph:
    """Nodes + weighted directed edges + bounded usage history + edit log."""

    def __init__(self):
        self.nodes: Dict[str, BehaviorNode] = {}
        self.edges: Dict[str, Dict[str, float]] = {}     # src -> {dst: weight}
        self.usage: Dict[str, list] = {}                 # op_id -> [(ctx list, success), ...]
        self.bigrams: Dict[str, int] = {}                # "a->b" -> verified count
        self.log: List[EditRecord] = []

    # ── persistence ──────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "nodes": [asdict(n) for n in self.nodes.values()],
            "edges": self.edges,
            "bigrams": self.bigrams,
            "log": [asdict(r) for r in self.log],
        }
        p.write_text(json.dumps(blob), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "BehaviorGraph":
        g = cls()
        p = Path(path)
        if not p.is_file():
            return g
        blob = json.loads(p.read_text(encoding="utf-8"))
        for nd in blob.get("nodes", []):
            n = BehaviorNode(**nd)
            g.nodes[n.op_id] = n
        g.edges = blob.get("edges", {})
        g.bigrams = blob.get("bigrams", {})
        g.log = [EditRecord(**r) for r in blob.get("log", [])]
        return g

    def snapshot(self) -> "BehaviorGraph":
        """Deep copy for the V6 kill-test (fresh vs after-A comparison)."""
        import copy
        return copy.deepcopy(self)


class GraphEditEngine:
    """Applies the 10 structural edits. observe() per decode outcome; maintain() periodically."""

    def __init__(self, graph: BehaviorGraph, cfg: EngineConfig | None = None):
        self.g = graph
        self.cfg = cfg or EngineConfig()

    # ── helpers ──────────────────────────────────────────────────────────────
    def _rec(self, kind: str, op_ids: list, **info) -> EditRecord:
        r = EditRecord(ts=time.time(), kind=kind, op_ids=op_ids, info=info)
        self.g.log.append(r)
        return r

    def _push_usage(self, op_id: str, ctx: np.ndarray, success: bool) -> None:
        u = self.g.usage.setdefault(op_id, [])
        u.append((ctx.tolist(), bool(success)))
        if len(u) > self.cfg.usage_cap:
            del u[0]

    def _update_centroid(self, old: Optional[list], n: int, x: np.ndarray):
        """Running mean centroid update. Returns (new_centroid_list, new_n)."""
        if old is None or n == 0:
            return x.tolist(), 1
        c = np.asarray(old, dtype=np.float32)
        c = c + (x - c) / (n + 1)
        return c.tolist(), n + 1

    # ── retrieval (conf-gated, precondition-scored) ──────────────────────────
    def retrieve(self, ctx_emb: np.ndarray, k: int = 3,
                 conf_floor: float = CONF_FLOOR) -> List[BehaviorNode]:
        scored = []
        for n in self.g.nodes.values():
            if not n.retrievable or n.confidence < conf_floor:
                continue
            if n.precond_pos is None:
                continue
            s = _cos(ctx_emb, np.asarray(n.precond_pos, dtype=np.float32))
            if n.precond_neg is not None:
                s -= self.cfg.neg_penalty * max(
                    0.0, _cos(ctx_emb, np.asarray(n.precond_neg, dtype=np.float32)))
            scored.append((s, n))
        scored.sort(key=lambda x: -x[0])
        return [n for s, n in scored[:k] if s > 0]

    # ── per-outcome edits ─────────────────────────────────────────────────────
    def observe(self, out: Outcome) -> List[EditRecord]:
        """Derive and apply all per-outcome edits from one decode outcome."""
        recs: List[EditRecord] = []
        cfg = self.cfg

        # age tick + edge decay
        for n in self.g.nodes.values():
            n.age += 1
        for src in self.g.edges:
            for dst in self.g.edges[src]:
                self.g.edges[src][dst] *= cfg.edge_decay

        used = [self.g.nodes[i] for i in out.trajectory if i in self.g.nodes]

        if out.verified:
            if used:
                # STRENGTHEN each used op
                for n in used:
                    n.confidence += (1.0 - n.confidence) * cfg.conf_up
                    n.validation_count += 1
                    recs.append(self._rec("STRENGTHEN", [n.op_id],
                                          conf=round(n.confidence, 3)))
                    # REFINE_PRECONDITION (positive): success context into pos centroid
                    n.precond_pos, n.n_pos = self._update_centroid(
                        n.precond_pos, n.n_pos, out.ctx_emb)
                    self._push_usage(n.op_id, out.ctx_emb, True)
                    recs.append(self._rec("REFINE_PRECONDITION", [n.op_id], polarity="pos"))
                # REFINE_EMBEDDING: EMA toward the behavior state that actually worked
                for n in used:
                    e = np.asarray(n.behavior_emb, dtype=np.float32)
                    e = _unit((1 - cfg.ema_alpha) * e + cfg.ema_alpha * _unit(out.behavior_emb))
                    n.behavior_emb = e.tolist()
                    recs.append(self._rec("REFINE_EMBEDDING", [n.op_id]))
                # CONNECT: consecutive pairs in a verified trajectory
                for a, b in zip(out.trajectory, out.trajectory[1:]):
                    if a in self.g.nodes and b in self.g.nodes and a != b:
                        self.g.edges.setdefault(a, {})
                        self.g.edges[a][b] = self.g.edges[a].get(b, 0.0) + cfg.edge_gain
                        key = f"{a}->{b}"
                        self.g.bigrams[key] = self.g.bigrams.get(key, 0) + 1
                        recs.append(self._rec("CONNECT", [a, b],
                                              w=round(self.g.edges[a][b], 3)))
            else:
                # DERIVE succeeded -> MINT (starts below reuse floor: poison gate)
                op_id = f"mint_{out.iid}_{len(self.g.nodes)}"
                node = BehaviorNode(
                    op_id=op_id, name=f"Mint_{out.iid}",
                    behavior_emb=_unit(out.behavior_emb).tolist(),
                    precond_pos=out.ctx_emb.tolist(),
                    confidence=MINT_CONF, validation_count=1,
                    source=f"mint:{out.iid}", n_pos=1)
                self.g.nodes[op_id] = node
                self._push_usage(op_id, out.ctx_emb, True)
                recs.append(self._rec("MINT", [op_id], iid=out.iid))
        else:
            # failure: WEAKEN used ops + negative precondition
            for n in used:
                n.confidence *= cfg.conf_down
                n.failure_count += 1
                recs.append(self._rec("WEAKEN", [n.op_id], conf=round(n.confidence, 3)))
                n.precond_neg, n.n_neg = self._update_centroid(
                    n.precond_neg, n.n_neg, out.ctx_emb)
                self._push_usage(n.op_id, out.ctx_emb, False)
                recs.append(self._rec("REFINE_PRECONDITION", [n.op_id], polarity="neg"))
        return recs

    # ── maintenance edits ─────────────────────────────────────────────────────
    def maintain(self) -> List[EditRecord]:
        recs: List[EditRecord] = []
        recs += self._merge_pass()
        recs += self._split_pass()
        recs += self._compose_pass()
        recs += self._retire_pass()
        return recs

    def _merge_pass(self) -> List[EditRecord]:
        """MERGE near-duplicate validated nodes (keeps library compact)."""
        recs = []
        cfg = self.cfg
        ids = [i for i, n in self.g.nodes.items()
               if n.retrievable and n.validation_count >= cfg.merge_min_val]
        merged = set()
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                if a in merged or b in merged:
                    continue
                na, nb = self.g.nodes[a], self.g.nodes[b]
                bc = _cos(np.asarray(na.behavior_emb), np.asarray(nb.behavior_emb))
                if bc < cfg.merge_behavior_cos:
                    continue
                if na.precond_pos is not None and nb.precond_pos is not None:
                    pc = _cos(np.asarray(na.precond_pos), np.asarray(nb.precond_pos))
                    if pc < cfg.merge_precond_cos:
                        continue
                # keep the more-validated node; fold the other in
                keep, drop = (na, nb) if na.validation_count >= nb.validation_count else (nb, na)
                wk, wd = keep.validation_count, drop.validation_count
                e = (wk * np.asarray(keep.behavior_emb) + wd * np.asarray(drop.behavior_emb)) / (wk + wd)
                keep.behavior_emb = _unit(e).tolist()
                keep.validation_count += drop.validation_count
                keep.failure_count += drop.failure_count
                keep.parents = (keep.parents or []) + [drop.op_id]
                # redirect edges
                for src in list(self.g.edges):
                    if drop.op_id in self.g.edges.get(src, {}):
                        w = self.g.edges[src].pop(drop.op_id)
                        if src != keep.op_id:
                            self.g.edges[src][keep.op_id] = self.g.edges[src].get(keep.op_id, 0.0) + w
                if drop.op_id in self.g.edges:
                    for dst, w in self.g.edges.pop(drop.op_id).items():
                        if dst != keep.op_id:
                            self.g.edges.setdefault(keep.op_id, {})
                            self.g.edges[keep.op_id][dst] = self.g.edges[keep.op_id].get(dst, 0.0) + w
                self.g.usage.setdefault(keep.op_id, []).extend(self.g.usage.pop(drop.op_id, []))
                del self.g.nodes[drop.op_id]
                merged.add(drop.op_id)
                recs.append(self._rec("MERGE", [keep.op_id, drop.op_id], behavior_cos=round(bc, 3)))
        return recs

    def _split_pass(self) -> List[EditRecord]:
        """SPLIT a mixed-success node whose contexts cluster by outcome.
        Child specializes to the success cluster; parent learns the failure
        cluster as a negative precondition."""
        recs = []
        cfg = self.cfg
        for op_id in list(self.g.nodes):
            n = self.g.nodes[op_id]
            u = self.g.usage.get(op_id, [])
            if len(u) < cfg.split_min_usage:
                continue
            succ = np.array([s for _, s in u])
            rate = succ.mean()
            if not (0.2 <= rate <= 0.8):     # only split genuinely mixed operators
                continue
            X = np.asarray([c for c, _ in u], dtype=np.float32)
            labels, C = _kmeans2(X)
            r0 = succ[labels == 0].mean() if (labels == 0).any() else 0.0
            r1 = succ[labels == 1].mean() if (labels == 1).any() else 0.0
            if abs(r0 - r1) < cfg.split_gap:
                continue
            good, bad = (0, 1) if r0 > r1 else (1, 0)
            child_id = f"{op_id}_s{len(self.g.nodes)}"
            child = BehaviorNode(
                op_id=child_id, name=f"{n.name}_specialized",
                behavior_emb=list(n.behavior_emb),
                precond_pos=C[good].tolist(),
                precond_neg=C[bad].tolist(),
                confidence=max(n.confidence, CONF_FLOOR),
                validation_count=int(succ[labels == good].sum()),
                source=f"split:{op_id}", parents=[op_id], n_pos=int((labels == good).sum()),
                n_neg=int((labels == bad).sum()))
            self.g.nodes[child_id] = child
            # parent keeps its identity but learns the failure region
            n.precond_neg = C[bad].tolist()
            n.n_neg = int((labels == bad).sum())
            # reset usage so the same evidence doesn't re-trigger the split
            self.g.usage[op_id] = []
            recs.append(self._rec("SPLIT", [op_id, child_id],
                                  rate_good=round(max(r0, r1), 2), rate_bad=round(min(r0, r1), 2)))
        return recs

    def _compose_pass(self) -> List[EditRecord]:
        """COMPOSE a frequent verified bigram A->B into a composite node.
        behavior_emb here is the parents' mean — an approximation; true composition
        is the renderer stacking both parents' modulations (parents kept for that)."""
        recs = []
        cfg = self.cfg
        for key in list(self.g.bigrams):
            if self.g.bigrams[key] < cfg.compose_min_count:
                continue
            a, b = key.split("->")
            if a not in self.g.nodes or b not in self.g.nodes:
                continue
            comp_id = f"comp_{a}_{b}"
            if comp_id in self.g.nodes:
                continue
            na, nb = self.g.nodes[a], self.g.nodes[b]
            e = _unit((np.asarray(na.behavior_emb) + np.asarray(nb.behavior_emb)) / 2)
            pp = None
            if na.precond_pos is not None and nb.precond_pos is not None:
                pp = ((np.asarray(na.precond_pos) + np.asarray(nb.precond_pos)) / 2).tolist()
            comp = BehaviorNode(
                op_id=comp_id, name=f"{na.name}+{nb.name}",
                behavior_emb=e.tolist(), precond_pos=pp,
                confidence=MINT_CONF,               # composite must earn reuse too
                validation_count=self.g.bigrams[key],
                source=f"compose:{key}", parents=[a, b])
            self.g.nodes[comp_id] = comp
            recs.append(self._rec("COMPOSE", [comp_id, a, b], count=self.g.bigrams[key]))
        return recs

    def _retire_pass(self) -> List[EditRecord]:
        """RETIRE stale never-validated low-confidence nodes (soft delete)."""
        recs = []
        cfg = self.cfg
        for n in self.g.nodes.values():
            if (n.retrievable and n.age > cfg.retire_age
                    and n.validation_count <= cfg.retire_max_val
                    and n.confidence < cfg.retire_conf):
                n.retrievable = False
                recs.append(self._rec("RETIRE", [n.op_id], age=n.age))
        return recs


# ── bridge: seed the behavior graph from discovered operator centroids ────────────

def seed_from_centroids(centroids: np.ndarray, names: Optional[List[str]] = None,
                        d_behavior: Optional[int] = None) -> BehaviorGraph:
    """Bootstrap nodes from discovery-time operator centroids (e.g. KMeans over
    fix displacements). behavior_emb starts as the centroid (or its first
    d_behavior dims) — REFINE_EMBEDDING migrates it toward FiLM states that work."""
    g = BehaviorGraph()
    for i, c in enumerate(centroids):
        e = c[:d_behavior] if d_behavior else c
        name = names[i] if names else f"op{i}"
        g.nodes[f"seed_{i}"] = BehaviorNode(
            op_id=f"seed_{i}", name=name,
            behavior_emb=_unit(np.asarray(e, dtype=np.float32)).tolist(),
            confidence=SEED_CONF, source="seed")
    return g


# ── selftest (no model) ───────────────────────────────────────────────────────────

def _selftest() -> bool:
    print("graph_edits --selftest: 10 structural edits + poison gate + V6 mini\n")
    rng = np.random.RandomState(0)
    d_ctx, d_beh = 16, 8

    def vec(seed_v, noise=0.05):
        return _unit(seed_v + noise * rng.randn(len(seed_v)).astype(np.float32))

    ctx_A = _unit(rng.randn(d_ctx).astype(np.float32))     # problem family A
    ctx_B = _unit(rng.randn(d_ctx).astype(np.float32))     # unrelated family B
    beh_A = _unit(rng.randn(d_beh).astype(np.float32))

    g = BehaviorGraph()
    eng = GraphEditEngine(g)

    # 1) MINT: derive-success creates an UNPROVEN node (poison gate)
    eng.observe(Outcome("t1", vec(ctx_A), vec(beh_A), [], True))
    assert len(g.nodes) == 1
    node = next(iter(g.nodes.values()))
    assert node.confidence < CONF_FLOOR, "minted node must start below reuse floor"
    assert not eng.retrieve(ctx_A), "unproven mint is NOT retrievable (poison gate)"
    print(f"  MINT: node created conf={node.confidence:.2f} < floor {CONF_FLOOR} -> gated (poison gate OK)")

    # 2) STRENGTHEN: two confirmed reuses -> retrievable
    eng.observe(Outcome("t2", vec(ctx_A), vec(beh_A), [node.op_id], True))
    eng.observe(Outcome("t3", vec(ctx_A), vec(beh_A), [node.op_id], True))
    assert node.confidence >= CONF_FLOOR
    got = eng.retrieve(ctx_A)
    assert got and got[0].op_id == node.op_id, "validated node becomes retrievable"
    print(f"  STRENGTHEN x2: conf={node.confidence:.2f} val={node.validation_count} -> retrievable")

    # 3) REFINE_EMBEDDING: behavior emb migrated toward the states that worked
    beh_drift = _unit(beh_A + 0.5 * rng.randn(d_beh).astype(np.float32))
    c_before = _cos(np.asarray(node.behavior_emb), beh_drift)
    for k in range(4):
        eng.observe(Outcome(f"t4{k}", vec(ctx_A), beh_drift, [node.op_id], True))
    c_after = _cos(np.asarray(node.behavior_emb), beh_drift)
    assert c_after > c_before, "EMA must move behavior_emb toward the working state"
    print(f"  REFINE_EMBEDDING: cos(emb, working_state) {c_before:.3f} -> {c_after:.3f}")

    # 4) WEAKEN + negative precondition: failures on ctx_B teach avoidance
    conf0 = node.confidence
    for k in range(2):
        eng.observe(Outcome(f"t5{k}", vec(ctx_B), vec(beh_A), [node.op_id], False))
    assert node.confidence < conf0 and node.precond_neg is not None
    r_B = eng.retrieve(ctx_B)
    print(f"  WEAKEN x2: conf {conf0:.2f} -> {node.confidence:.2f}; neg precondition set; "
          f"retrieve(ctx_B) -> {[n.op_id for n in r_B]}")

    # 5) CONNECT: verified trajectory forms edges
    eng2 = GraphEditEngine(BehaviorGraph())
    ga = eng2.g
    for oid in ("opA", "opB"):
        ga.nodes[oid] = BehaviorNode(op_id=oid, name=oid,
                                     behavior_emb=_unit(rng.randn(d_beh)).tolist(),
                                     precond_pos=ctx_A.tolist(),
                                     confidence=SEED_CONF, validation_count=2)
    for k in range(3):
        eng2.observe(Outcome(f"t6{k}", vec(ctx_A), vec(beh_A), ["opA", "opB"], True))
    assert ga.edges.get("opA", {}).get("opB", 0) > 0, "CONNECT must form A->B edge"
    print(f"  CONNECT: edge opA->opB w={ga.edges['opA']['opB']:.2f} after 3 verified trajectories")

    # 6) COMPOSE: frequent bigram -> composite node (starts unproven)
    recs = eng2.maintain()
    comp = [r for r in recs if r.kind == "COMPOSE"]
    assert comp, "bigram count >= 3 must mint a composite"
    cnode = ga.nodes[comp[0].op_ids[0]]
    assert cnode.parents == ["opA", "opB"] and cnode.confidence < CONF_FLOOR
    print(f"  COMPOSE: {cnode.op_id} minted from bigram (parents kept, conf gated)")

    # 7) MERGE: near-duplicate validated nodes fold together
    gm = BehaviorGraph(); em = GraphEditEngine(gm)
    e_dup = _unit(rng.randn(d_beh).astype(np.float32))
    for i in range(2):
        gm.nodes[f"dup{i}"] = BehaviorNode(
            op_id=f"dup{i}", name=f"dup{i}",
            behavior_emb=vec(e_dup, 0.01).tolist(), precond_pos=ctx_A.tolist(),
            confidence=SEED_CONF, validation_count=3)
    recs = em.maintain()
    assert any(r.kind == "MERGE" for r in recs) and len(gm.nodes) == 1
    print(f"  MERGE: 2 near-duplicates -> 1 node (val={next(iter(gm.nodes.values())).validation_count})")

    # 8) SPLIT: mixed-success node with bimodal contexts specializes
    gs = BehaviorGraph(); es = GraphEditEngine(gs)
    gs.nodes["mx"] = BehaviorNode(op_id="mx", name="mixed",
                                  behavior_emb=_unit(rng.randn(d_beh)).tolist(),
                                  precond_pos=ctx_A.tolist(), confidence=SEED_CONF,
                                  validation_count=2)
    for k in range(6):
        es.observe(Outcome(f"s{k}", vec(ctx_A), vec(beh_A), ["mx"], True))
    for k in range(6):
        es.observe(Outcome(f"f{k}", vec(ctx_B), vec(beh_A), ["mx"], False))
    recs = es.maintain()
    splits = [r for r in recs if r.kind == "SPLIT"]
    assert splits, "bimodal mixed-success node must split"
    child = gs.nodes[splits[0].op_ids[1]]
    assert _cos(ctx_A, np.asarray(child.precond_pos)) > _cos(ctx_B, np.asarray(child.precond_pos)), \
        "child must specialize to the SUCCESS context cluster"
    print(f"  SPLIT: child {child.op_id} specialized to success cluster "
          f"(rate_good={splits[0].info['rate_good']}, rate_bad={splits[0].info['rate_bad']})")

    # 9) RETIRE: stale never-validated node becomes non-retrievable
    gr = BehaviorGraph(); er = GraphEditEngine(gr)
    gr.nodes["stale"] = BehaviorNode(op_id="stale", name="stale",
                                     behavior_emb=_unit(rng.randn(d_beh)).tolist(),
                                     precond_pos=ctx_A.tolist(),
                                     confidence=MINT_CONF, validation_count=0, age=60)
    recs = er.maintain()
    assert any(r.kind == "RETIRE" for r in recs) and not gr.nodes["stale"].retrievable
    print(f"  RETIRE: stale unproven node soft-deleted at age 60")

    # 10) V6 mini: does solving A help a similar A'? (fresh vs after-A retrieval)
    fresh = BehaviorGraph()
    after_A = BehaviorGraph(); eA = GraphEditEngine(after_A)
    eA.observe(Outcome("a1", vec(ctx_A), vec(beh_A), [], True))          # derive -> MINT
    m_id = next(iter(after_A.nodes))
    eA.observe(Outcome("a2", vec(ctx_A), vec(beh_A), [m_id], True))      # confirm x2
    eA.observe(Outcome("a3", vec(ctx_A), vec(beh_A), [m_id], True))
    ctx_A2 = vec(ctx_A, 0.1)                                             # similar task A'
    hit_fresh = GraphEditEngine(fresh).retrieve(ctx_A2)
    hit_after = eA.retrieve(ctx_A2)
    assert not hit_fresh and hit_after, "library-after-A must help the similar task A'"
    print(f"  V6 mini: fresh retrieval=0 ops, after-A retrieval={len(hit_after)} op(s) -> amortization mechanism works")

    # persistence roundtrip
    import tempfile, os
    tmp = os.path.join(tempfile.gettempdir(), "graph_edits_selftest.json")
    after_A.save(tmp)
    g2 = BehaviorGraph.load(tmp)
    assert len(g2.nodes) == len(after_A.nodes) and len(g2.log) == len(after_A.log)
    print(f"  persistence: {len(g2.nodes)} nodes, {len(g2.log)} edit records roundtripped")

    kinds = sorted(set(r.kind for g_ in (g, ga, gm, gs, gr, after_A) for r in g_.log))
    print(f"\n  edit kinds exercised: {kinds}")
    print("  GRAPH-EDITS SELFTEST -> PASS (10 edits + poison gate + V6 mini)")
    return True


def main():
    ap = argparse.ArgumentParser(description="LGGN graph edits engine (the write path).")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if _selftest() else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
