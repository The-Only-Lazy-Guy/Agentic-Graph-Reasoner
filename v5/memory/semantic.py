"""L2 SEMANTIC memory — concepts: memory of SOMETHING, organizing memory of
IMPLEMENTATIONS. A concept is a BehaviorNode (v5.runtime.graph_edits — reused untouched):

  behavior_emb = centroid in SKELETON space (what strategy this is — identifiers masked)
  precond_pos/neg = centroids in CTX space (when it applies / when it failed)
  confidence + lifecycle: MINT / STRENGTHEN / WEAKEN / REFINE_* / MERGE / SPLIT / RETIRE
  (poison gate: minted concepts start below the reuse floor until validated)

plus `members`: concept_id -> [impl_ids] (kept OUTSIDE BehaviorNode so graph_edits stays
untouched). Read path: ctx -> retrieve concepts (conf-gated) -> their member impls.
Write path: impl outcome -> assign to nearest concept in skeleton space (else MINT via the
engine's DERIVE branch) -> engine.observe drives all edits.

  python -m v5.memory.semantic --selftest
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from v5.runtime.graph_edits import (BehaviorGraph, BehaviorNode, EngineConfig,
                                    GraphEditEngine, Outcome, SEED_CONF)

ASSIGN_THETA = 0.45          # min skeleton-cos to join an existing concept; else MINT


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


class ConceptStore:
    """GraphEditEngine-backed concept layer with an impl membership map."""

    def __init__(self, root: str | Path = "data/memory", cfg: EngineConfig | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.graph_path = self.root / "l2_graph.json"
        self.members_path = self.root / "l2_members.json"
        self.graph = BehaviorGraph.load(str(self.graph_path))
        self.engine = GraphEditEngine(self.graph, cfg)
        self.members: dict[str, list[str]] = {}
        if self.members_path.exists():
            self.members = json.loads(self.members_path.read_text(encoding="utf-8"))
        self._observes_since_maintain = 0

    def __len__(self) -> int:
        return len(self.graph.nodes)

    # ── persistence ──────────────────────────────────────────────────────────
    def save(self) -> None:
        self.graph.save(str(self.graph_path))
        tmp = self.members_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.members), encoding="utf-8")
        tmp.replace(self.members_path)

    # ── bootstrap from seeded episodic memory ────────────────────────────────
    def bootstrap(self, impl_store, k: int = 32, seed: int = 0, log=print) -> int:
        """KMeans over impl SKELETON embeddings -> seed concepts. Idempotent-ish: skips
        if concepts already exist."""
        if self.graph.nodes:
            log(f"  [semantic] {len(self.graph.nodes)} concepts already exist — skip bootstrap")
            return 0
        ids = impl_store.all_ids()
        if len(ids) < k:
            k = max(1, len(ids) // 4) or 1
        skel = impl_store.emb_skel.get(ids)
        ctx = impl_store.emb_ctx.get(ids)
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(skel)
        for j in range(k):
            mask = km.labels_ == j
            if not mask.any():
                continue
            cid = f"con_{j:04d}"
            self.graph.nodes[cid] = BehaviorNode(
                op_id=cid, name=f"Concept_{j}",
                behavior_emb=_unit(skel[mask].mean(0)).tolist(),
                precond_pos=_unit(ctx[mask].mean(0)).tolist(),
                confidence=SEED_CONF, source="seed", n_pos=int(mask.sum()))
            self.members[cid] = [ids[i] for i in np.where(mask)[0]]
        self.save()
        log(f"  [semantic] bootstrapped {len(self.graph.nodes)} concepts over {len(ids)} impls")
        return len(self.graph.nodes)

    # ── read path ────────────────────────────────────────────────────────────
    def retrieve(self, ctx_vec, k: int = 3) -> list[dict]:
        """Conf-gated precondition-scored concepts + their member impl ids."""
        nodes = self.engine.retrieve(np.asarray(ctx_vec, dtype=np.float32), k=k)
        return [{"concept_id": n.op_id, "name": n.name, "confidence": n.confidence,
                 "impl_ids": list(self.members.get(n.op_id, []))} for n in nodes]

    # ── write path ───────────────────────────────────────────────────────────
    def assign(self, skel_vec) -> str | None:
        """Nearest concept in skeleton space above ASSIGN_THETA (retired excluded)."""
        best, best_s = None, ASSIGN_THETA
        q = _unit(skel_vec)
        for n in self.graph.nodes.values():
            if not n.retrievable:
                continue
            s = float(np.dot(q, _unit(n.behavior_emb)))
            if s > best_s:
                best, best_s = n.op_id, s
        return best

    def observe_impl(self, impl_id: str, ctx_vec, skel_vec, verified: bool,
                     maintain_every: int = 25) -> str | None:
        """Lifecycle update for one loop outcome. Returns the concept id the impl landed in
        (existing, freshly minted on verified-unassigned, or None on unassigned failure)."""
        cid = self.assign(skel_vec)
        out = Outcome(iid=impl_id, ctx_emb=np.asarray(ctx_vec, dtype=np.float32),
                      behavior_emb=np.asarray(skel_vec, dtype=np.float32),
                      trajectory=[cid] if cid else [], verified=verified)
        recs = self.engine.observe(out)
        if cid is None:
            minted = [r for r in recs if r.kind == "MINT"]
            cid = minted[0].op_ids[0] if minted else None
        if cid is not None and verified:
            self.members.setdefault(cid, [])
            if impl_id not in self.members[cid]:
                self.members[cid].append(impl_id)
        self._observes_since_maintain += 1
        if self._observes_since_maintain >= maintain_every:
            self.engine.maintain()
            self._observes_since_maintain = 0
            self._sync_members()
        self.save()
        return cid

    def _sync_members(self) -> None:
        """After MERGE/RETIRE, fold members of dead nodes into their surviving parents."""
        alive = set(self.graph.nodes)
        for cid in list(self.members):
            if cid in alive:
                continue
            orphans = self.members.pop(cid)
            heir = next((n.op_id for n in self.graph.nodes.values()
                         if n.parents and cid in n.parents), None)
            if heir:
                dst = self.members.setdefault(heir, [])
                dst.extend(i for i in orphans if i not in dst)

    def stats(self) -> dict:
        ns = self.graph.nodes.values()
        return {"concepts": len(self.graph.nodes),
                "retrievable": sum(1 for n in ns if n.retrievable),
                "member_impls": sum(len(v) for v in self.members.values()),
                "edits": len(self.graph.log)}


# ── selftest ────────────────────────────────────────────────────────────────────

def _selftest() -> bool:
    import tempfile
    print("memory.semantic --selftest: bootstrap, retrieve, assign/MINT, lifecycle, persist\n")

    class _FakeImpls:                                       # controlled geometry
        def __init__(self):
            rng = np.random.RandomState(0)
            a, b = _unit(rng.randn(768)), _unit(rng.randn(768))
            self.ids = [f"impl_a{i}" for i in range(6)] + [f"impl_b{i}" for i in range(6)]
            self._skel = {}
            self._ctx = {}
            for i in range(6):
                self._skel[f"impl_a{i}"] = _unit(a + 0.05 * rng.randn(768))
                self._ctx[f"impl_a{i}"] = _unit(a + 0.05 * rng.randn(768))
                self._skel[f"impl_b{i}"] = _unit(b + 0.05 * rng.randn(768))
                self._ctx[f"impl_b{i}"] = _unit(b + 0.05 * rng.randn(768))
            self.emb_skel = self
            self.emb_ctx = self

        def all_ids(self):
            return self.ids

        def get(self, ids):                                 # serves both skel/ctx views
            src = self._skel if self is self.emb_skel else self._ctx
            return np.stack([src[i] for i in ids])

    with tempfile.TemporaryDirectory() as td:
        fake = _FakeImpls()
        cs = ConceptStore(td)
        n = cs.bootstrap(fake, k=2, seed=0, log=lambda *a: None)
        assert n == 2 and len(cs.members) == 2
        sizes = sorted(len(v) for v in cs.members.values())
        assert sizes == [6, 6], f"clusters should split a/b evenly: {sizes}"
        print("  [1] bootstrap KMeans -> PASS")

        qa = fake._ctx["impl_a0"]
        got = cs.retrieve(qa, k=1)
        assert got and set(got[0]["impl_ids"]) <= {f"impl_a{i}" for i in range(6)}, \
            "ctx query lands in the A concept"
        print("  [2] retrieve concept->members -> PASS")

        cid = cs.assign(fake._skel["impl_a1"])
        assert cid is not None and cid == got[0]["concept_id"], "assign matches retrieve"
        far = _unit(np.random.RandomState(9).randn(768))
        assert cs.assign(far) is None, "far skeleton unassigned"
        print("  [3] assign threshold -> PASS")

        pre_conf = cs.graph.nodes[cid].confidence
        cs.observe_impl("impl_new1", qa, fake._skel["impl_a1"], verified=True)
        assert cs.graph.nodes[cid].confidence > pre_conf, "STRENGTHEN raised confidence"
        assert "impl_new1" in cs.members[cid], "verified impl joined members"
        minted_cid = cs.observe_impl("impl_new2", far, far, verified=True)
        assert minted_cid and minted_cid.startswith("mint_"), "unassigned verified -> MINT"
        assert cs.graph.nodes[minted_cid].confidence < 0.4, "poison gate: below reuse floor"
        none_cid = cs.observe_impl("impl_new3", far, _unit(np.random.RandomState(11).randn(768)),
                                   verified=False)
        assert none_cid is None, "unassigned failure mints nothing"
        print("  [4] observe: STRENGTHEN / MINT+poison-gate / failure -> PASS")

        cs2 = ConceptStore(td)
        assert len(cs2) == len(cs) and cs2.members[cid] == cs.members[cid], "reload"
        assert cs2.stats()["edits"] > 0
        print("  [5] persistence + stats -> PASS")
    print("\n  MEMORY.SEMANTIC SELFTEST -> PASS")
    return True


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        raise SystemExit(0 if _selftest() else 1)
    ap.print_help()
