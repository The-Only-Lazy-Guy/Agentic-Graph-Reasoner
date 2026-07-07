"""L1 EPISODIC memory — implementation records: how something was CONCRETELY done.

One record per verified (or attempted) edit/build: the context it happened in, the code
before/after, the reasoning trace, and the outcome. Two embeddings per record:
  emb_ctx  — mpnet(ctx_text): WHEN this applies (retrieval key, precondition space)
  emb_skel — mpnet(trace_skeleton(trace)): WHAT STRATEGY it is (concept space; identifiers
             masked so implementations of the same strategy cluster — v5.runtime.lggn_tracer)

Seeds: Fable-5 realizer triples (verified="weak" — accepted by the original agent but not
test-verified here). The loop appends verified="strong" (tests passed) records.

  python -m v5.memory.episodic --selftest
  python -m v5.memory.episodic --seed          # fable5 triples -> data/memory (needs mpnet)
"""
from __future__ import annotations

import time
from pathlib import Path

from v5.memory.store import EmbStore, JsonlWal, stable_id


def _skel(trace: str) -> str:
    from v5.runtime.lggn_tracer import trace_skeleton
    return trace_skeleton(trace or "")


class ImplStore:
    """Append-only implementation records + ctx/skeleton embedding shards."""

    def __init__(self, root: str | Path = "data/memory", embed_fn=None):
        self.root = Path(root)
        self.wal = JsonlWal(self.root / "l1_impl.jsonl")
        self.emb_ctx = EmbStore(self.root, "l1_ctx")
        self.emb_skel = EmbStore(self.root, "l1_skel")
        self.embed_fn = embed_fn
        self.records: dict[str, dict] = {r["impl_id"]: r for r in self.wal.read_all()}

    def __len__(self) -> int:
        return len(self.records)

    def add(self, ctx_text: str, old: str, new: str, trace: str, file_path: str = "",
            outcome: str = "unknown", verified: str = "weak", task_id: str = "",
            concept_id: str = "", kind: str = "") -> str | None:
        """Returns impl_id, or None if a duplicate (content-addressed on old+new+trace).
        kind (create/cross/debug/extend, project_gen.py's session kind): a structural signal
        cosine similarity has no access to -- stored so a ranker can use it as a feature, not
        just the record's semantic content."""
        impl_id = stable_id("impl", (old or "") + "\x00" + (new or "") + "\x00" + (trace or ""))
        if impl_id in self.records:
            return None
        rec = {"impl_id": impl_id, "ctx_text": (ctx_text or "")[:600], "old": old, "new": new,
               "trace": (trace or "")[:400], "file_path": file_path, "outcome": outcome,
               "verified": verified, "task_id": task_id, "concept_id": concept_id,
               "kind": kind, "ts": time.time()}
        if self.embed_fn is not None:
            v_ctx = self.embed_fn({impl_id: rec["ctx_text"] or rec["trace"] or old})[impl_id]
            v_skel = self.embed_fn({impl_id: _skel(rec["trace"]) or rec["ctx_text"]})[impl_id]
            self.emb_ctx.add([impl_id], [v_ctx])
            self.emb_skel.add([impl_id], [v_skel])
        self.wal.append(rec)
        self.records[impl_id] = rec
        return impl_id

    def get(self, impl_id: str) -> dict | None:
        return self.records.get(impl_id)

    def search_ctx(self, query_vec, k: int = 8, within: list[str] | None = None) -> list[tuple[str, float]]:
        return self.emb_ctx.search(query_vec, k, within=within)

    def all_ids(self) -> list[str]:
        return list(self.records)

    def seed_from_fable5(self, triples_path: str = "", limit: int = 0, log=print) -> int:
        """936-triple funnel -> weak-verified implementation seeds."""
        from v5.runtime.lggn_realizer import TRIPLES, load_triples
        triples = load_triples(triples_path or TRIPLES, limit=limit, log=log)
        n = 0
        for t in triples:
            got = self.add(ctx_text=t["goal"][:400], old=t["old"], new=t["new"],
                           trace=t["trace"], outcome="accepted", verified="weak",
                           task_id=f"fable5:{t.get('session_id', '')}")
            n += int(got is not None)
        log(f"  [episodic] seeded {n} weak-verified impls from fable5 ({len(self)} total)")
        return n


# ── selftest ────────────────────────────────────────────────────────────────────

def _selftest() -> bool:
    import tempfile
    import numpy as np
    from v5.memory.store import make_fake_embedder
    print("memory.episodic --selftest: add/dedup, dual embeddings, search, persist (no model)\n")
    with tempfile.TemporaryDirectory() as td:
        st = ImplStore(td, embed_fn=make_fake_embedder())
        i1 = st.add("add retry to http client", "return _raw(url)", "return retry(3)(_raw)(url)",
                    "wrap `_raw` in a retry decorator", outcome="pass", verified="strong",
                    kind="extend")
        i2 = st.add("cap the stream rows", "rows = all()", "rows = all()[:cap]",
                    "slice the row list to `cap`")
        assert i1 and i2 and i1 != i2
        assert st.add("x", "return _raw(url)", "return retry(3)(_raw)(url)",
                      "wrap `_raw` in a retry decorator") is None, "content dedup"
        assert len(st) == 2 and len(st.emb_ctx) == 2 and len(st.emb_skel) == 2
        assert st.get(i1)["kind"] == "extend" and st.get(i2)["kind"] == "", \
            "kind stored (structural feature cosine can't see) -- defaults to empty string"
        print("  [1] add + dedup + dual embeddings + kind -> PASS")

        q = st.emb_ctx.get([i1])[0]
        top = st.search_ctx(q, k=1)
        assert top[0][0] == i1 and top[0][1] > 0.99
        sub = st.search_ctx(q, k=5, within=[i2])
        assert sub[0][0] == i2, "within restriction"
        print("  [2] ctx search + within -> PASS")

        st2 = ImplStore(td, embed_fn=make_fake_embedder())
        assert len(st2) == 2 and st2.get(i1)["verified"] == "strong", "reload"
        assert np.allclose(st2.emb_skel.get([i1]), st.emb_skel.get([i1]), atol=1e-3)
        print("  [3] persistence -> PASS")

        # skeleton embedding groups strategy: identical strategy w/ different identifiers
        # -> same skeleton text -> IDENTICAL fake vector (hash of masked text)
        i3 = st.add("other ctx", "return _fetch(uri)", "return retry(3)(_fetch)(uri)",
                    "wrap `_fetch` in a retry decorator")
        s1, s3 = st.emb_skel.get([i1])[0], st.emb_skel.get([i3])[0]
        assert float(np.dot(s1, s3)) > 0.99, "masked skeletons collapse to same strategy"
        print("  [4] skeleton strategy-space -> PASS")
    print("\n  MEMORY.EPISODIC SELFTEST -> PASS")
    return True


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seed", action="store_true", help="seed data/memory from fable5 triples (mpnet)")
    ap.add_argument("--root", default="data/memory")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if _selftest() else 1)
    if a.seed:
        from v5.memory.store import make_mpnet_embedder
        st = ImplStore(a.root, embed_fn=make_mpnet_embedder())
        st.seed_from_fable5(limit=a.limit)
    else:
        ap.print_help()
