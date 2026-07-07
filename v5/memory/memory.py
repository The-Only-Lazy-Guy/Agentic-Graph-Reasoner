"""TotalMemory — the single facade the agent loop talks to. Clear LM<->graph contract:

  read(ctx)  -> MemoryHit: ONE short implementation payload (trace + capped old->new) chosen
                by the two-hop path  ctx -> L2 concepts (conf-gated) -> member impls ->
                LOCAL-FIT re-rank (0.6·ctx-cos + 0.4·ident_overlap with the code in front of
                the model). Payload is DATA for the prompt's trace slot (~<=300 tokens);
                steering/constraint channels come later (P5) and cost zero tokens.
  write(...) -> L1 append (verified strong on pass) + L2 lifecycle observe (+ caller feeds
                touched files to L0 via syntax.scan_files).

Modes (the GM1 ablation): off (empty hit) | flat (ANN over ALL impls, no concepts) |
concept (two-hop). Same interface, so the loop's arms differ ONLY in memory content.

  python -m v5.memory.memory --selftest
  python -m v5.memory.memory --seed       # fable5 -> L1, KMeans -> L2 (mpnet, local)
  python -m v5.memory.memory --stats
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from v5.memory.episodic import ImplStore, _skel
from v5.memory.semantic import ConceptStore
from v5.memory.store import stable_id
from v5.memory.syntax import SyntaxStore, ident_overlap

W_CTX, W_IDENT = 0.6, 0.4          # local-fit blend
FLAT_POOL = 16                     # candidates pulled before local-fit re-rank
MIN_FIT = 0.35                     # relevance gate: below this, deliver NOTHING (GM1 lesson:
                                   # irrelevant payload craters an empty-slot-trained proposer)
SAME_FILE_BOOST = 0.25             # GB1 lesson: small shared identifier vocab across sibling
                                   # files (ts/level/msg in a logparse repo) lets ident_overlap
                                   # pick the WRONG file's history; prefer this file's own past


@dataclass
class MemoryHit:
    impls: list = field(default_factory=list)          # full L1 records, best first
    concepts: list = field(default_factory=list)       # [{concept_id, confidence, ...}]
    trace_text: str = ""                               # the deliverable payload (may be "")
    tokens_est: int = 0                                # ~chars/4, the GS speed budget number


class TotalMemory:
    def __init__(self, root: str | Path = "data/memory", mode: str = "concept",
                 embed_fn=None):
        assert mode in ("off", "flat", "concept"), mode
        self.root = Path(root)
        self.mode = mode
        self.embed_fn = embed_fn
        self.impls = ImplStore(self.root, embed_fn=embed_fn)
        self.concepts = ConceptStore(self.root)
        self.syntax = SyntaxStore(self.root, embed_fn=embed_fn)

    # ── read ─────────────────────────────────────────────────────────────────
    def _embed_one(self, text: str) -> np.ndarray:
        key = stable_id("q", text)
        return np.asarray(self.embed_fn({key: text})[key], dtype=np.float32)

    def read(self, goal: str, span: str = "", obs: str = "", file_path: str = "",
             k_impl: int = 1, min_fit: float = MIN_FIT) -> MemoryHit:
        if self.mode == "off" or len(self.impls) == 0 or self.embed_fn is None:
            return MemoryHit()
        q = self._embed_one((goal or "")[:400] + ("\n" + obs[-200:] if obs else ""))
        concepts_meta: list[dict] = []
        if self.mode == "concept":
            concepts_meta = self.concepts.retrieve(q, k=3)
            pool = [i for c in concepts_meta for i in c["impl_ids"]]
            cand = self.impls.search_ctx(q, k=FLAT_POOL, within=pool) if pool else []
            if not cand:                                   # cold concepts -> flat fallback
                cand = self.impls.search_ctx(q, k=FLAT_POOL)
        else:                                              # flat
            cand = self.impls.search_ctx(q, k=FLAT_POOL)
        # Same-file boost is a fallback, not a blanket bonus: a flat additive term can't tell
        # apart "the file has no explicit cross-file cue -> trust its own history" (DEBUG:
        # "fix parse_line" names nothing else) from "the goal explicitly names a DIFFERENT
        # file's convention" (EXTEND: "the SAME tax rate that pricing uses") -- boosting the
        # target file's own (irrelevant) history in the second case actively buries the
        # needed cross-file record. Gate the boost on whether the goal names another file.
        gl = (goal or "").lower()
        other_named = any(
            (rec.get("file_path") or "") != file_path and Path(rec["file_path"]).stem.lower() in gl
            for rec in (self.impls.get(iid) for iid, _ in cand) if rec and rec.get("file_path")
        )
        scored = []
        for impl_id, ctx_cos in cand:
            rec = self.impls.get(impl_id)
            if rec is None:
                continue
            fit = W_CTX * ctx_cos + W_IDENT * ident_overlap(
                span or goal, (rec["old"] or "") + "\n" + (rec["new"] or ""))
            if file_path and rec.get("file_path") == file_path and not other_named:
                fit += SAME_FILE_BOOST
            scored.append((fit, rec))
        scored.sort(key=lambda x: -x[0])
        scored = [(f, r) for f, r in scored if f >= min_fit]   # relevance gate
        top = [r for _, r in scored[:k_impl]]
        payload = ""
        if top:
            r = top[0]
            payload = (r["trace"] or "").strip()[:400]
            delta = (r["old"] or "")[:250] + "\n=>\n" + (r["new"] or "")[:350]
            payload = (payload + "\n" + delta).strip()
        sym_block = self._read_symbols(q, goal, span, file_path)
        if sym_block:
            payload = (sym_block + "\n" + payload).strip()
        return MemoryHit(impls=top, concepts=concepts_meta, trace_text=payload,
                         tokens_est=len(payload) // 4)

    def _read_symbols(self, q, goal: str, span: str, file_path: str = "",
                      k_syms: int = 2) -> str:
        """L0 delivery: own-repo symbols relevant to the task. A file whose NAME is mentioned
        in the goal ('the same format as inventory receipts') gets a strong boost — that
        explicit mention wins regardless of which file it names. The CURRENT target file
        additionally gets the same boost when the goal names no OTHER file (own-file
        continuity is the reliable default), but that fallback is suppressed when the goal
        explicitly points elsewhere — else the target file's own symbols can bury the
        correctly-named cross-file ones (the EXTEND regression: boosting orders.py's own
        symbols over pricing.py's, despite the goal saying 'the SAME tax rate that pricing
        uses'). Same relevance-gate philosophy throughout: deliver nothing rather than junk."""
        if len(self.syntax) == 0:
            return ""
        cands = self.syntax.search(q, k=12)
        gl = (goal or "").lower()
        other_named = any(
            Path(s.get("file", "")).stem.lower() in gl
            for s in cands if s.get("file") and s["file"] != file_path
        )
        scored = []
        for s in cands:
            file = s.get("file", "")
            stem = Path(file).stem.lower()
            named = bool(stem) and stem in gl
            own_file = file_path and file == file_path and not other_named
            boost = 0.35 if named or own_file else 0.0
            fit = 0.5 * s["score"] + 0.5 * ident_overlap(span or goal, s["text"]) + boost
            if fit >= MIN_FIT:
                scored.append((fit, s))
        scored.sort(key=lambda x: -x[0])
        return "\n".join(s["text"][:300] for _, s in scored[:k_syms])

    # ── write ────────────────────────────────────────────────────────────────
    def write(self, goal: str, old: str, new: str, trace: str, verified: bool,
              file_path: str = "", task_id: str = "") -> str | None:
        """One loop outcome -> L1 record + L2 lifecycle. Returns impl_id (None = duplicate)."""
        impl_id = self.impls.add(
            ctx_text=goal[:400], old=old, new=new, trace=trace, file_path=file_path,
            outcome="pass" if verified else "fail",
            verified="strong" if verified else "fail", task_id=task_id)
        if impl_id is None or self.embed_fn is None:
            return impl_id
        ctx_vec = self.impls.emb_ctx.get([impl_id])[0]
        skel_vec = self.impls.emb_skel.get([impl_id])[0]
        cid = self.concepts.observe_impl(impl_id, ctx_vec, skel_vec, verified)
        if cid:
            self.impls.records[impl_id]["concept_id"] = cid
        return impl_id

    def seed(self, k_concepts: int = 32, limit: int = 0, log=print) -> None:
        self.impls.seed_from_fable5(limit=limit, log=log)
        self.concepts.bootstrap(self.impls, k=k_concepts, log=log)

    def stats(self) -> dict:
        return {"mode": self.mode, "impls": len(self.impls),
                "syntax": len(self.syntax), **self.concepts.stats()}


# ── selftest ────────────────────────────────────────────────────────────────────

def _selftest() -> bool:
    import tempfile
    from v5.memory.store import make_fake_embedder
    print("memory.memory --selftest: modes, two-hop read, local fit, write lifecycle\n")
    fe = make_fake_embedder()
    with tempfile.TemporaryDirectory() as td:
        tm = TotalMemory(td, mode="concept", embed_fn=fe)
        assert tm.read("anything").trace_text == "", "empty memory -> empty hit"

        # seed impls: two strategy families with distinct identifiers
        for i in range(5):
            tm.impls.add(f"retry http call variant {i}", f"return _raw{i}(url)",
                         f"return retry(3)(_raw{i})(url)", "wrap the call in a retry decorator")
        for i in range(5):
            tm.impls.add(f"cap rows in stream {i}", f"rows{i} = fetch_all()",
                         f"rows{i} = fetch_all()[:cap]", "slice the fetched list to the cap")
        tm.concepts.bootstrap(tm.impls, k=2, log=lambda *a: None)
        assert tm.stats()["concepts"] == 2 and tm.stats()["impls"] == 10
        print("  [1] seed + bootstrap -> PASS")

        # same-ctx query: fake embedder is hash-exact, so reuse a seeded ctx text
        hit = tm.read("retry http call variant 2", span="x = _raw2(url)")
        assert hit.impls and "retry" in hit.trace_text, "concept read returns retry impl"
        assert hit.tokens_est <= 300, f"payload budget {hit.tokens_est}"
        # local fit: identifier overlap must prefer the _raw2 variant among equals
        assert "_raw2" in hit.impls[0]["old"], f"ident fit picked {hit.impls[0]['old']}"
        print("  [2] two-hop read + local ident fit -> PASS")

        tm_flat = TotalMemory(td, mode="flat", embed_fn=fe)

        # same-file boost: two impls with IDENTICAL identifiers/text shape (indistinguishable
        # by ctx-cos or ident_overlap alone) in different files — file_path must break the tie
        # toward the file actually being edited (the logparse debug regression: shared
        # ts/level/msg vocab let a SIBLING file's history outrank the target file's own past).
        # flat mode: no concept-membership gate, isolates the scoring mechanism itself.
        tm_flat.impls.add("touch shared vars", "x = a + b  # p", "x = a + b + 1  # p",
                          "bump the total", file_path="parser.py")
        tm_flat.impls.add("touch shared vars", "x = a + b  # w", "x = a + b + 1  # w",
                          "bump the total", file_path="writer.py")
        hit_pp = tm_flat.read("touch shared vars", span="x = a + b", file_path="parser.py")
        assert hit_pp.impls and hit_pp.impls[0].get("file_path") == "parser.py", \
            f"same-file boost picked {hit_pp.impls[0].get('file_path')!r}"
        print("  [2b] same-file continuity boost breaks ties -> PASS")

        # EXTEND-shaped case: goal explicitly names a DIFFERENT file ('pricing') than the
        # target being edited ('orders.py') -> the target's same-file boost must be
        # SUPPRESSED so the correctly cross-file-named record still wins (the real
        # regression this reproduces: orders.py's own irrelevant history outranking
        # pricing.py's needed one). Same ctx_text on both -> IDENTICAL ctx embedding under
        # the fake embedder, neutralizing ctx_cos into a clean tie; pricing's content shares
        # the query's 'tax' vocabulary (higher ident_overlap) so it wins on merit alone —
        # proving the target-file boost did NOT override a correctly-named cross-file match.
        Q = "apply the SAME tax rate that pricing uses"
        tm_flat.impls.add(Q, "x = a + b + tax", "x = a + b + tax + 1",
                          "reuse pricing's tax", file_path="pricing.py")
        tm_flat.impls.add(Q, "x = a + b", "x = a + b + 1",
                          "bump the total", file_path="orders.py")
        hit_cross = tm_flat.read(Q, span="x = a + b + tax", file_path="orders.py")
        assert hit_cross.impls and hit_cross.impls[0].get("file_path") == "pricing.py", \
            f"explicit cross-file mention should win over the target's same-file boost, " \
            f"got {hit_cross.impls[0].get('file_path')!r}"
        print("  [2c] explicit cross-file mention suppresses same-file boost -> PASS")
        hf = tm_flat.read("retry http call variant 2", span="x = _raw2(url)")
        assert hf.impls and hf.concepts == [], "flat mode skips concepts"
        tm_off = TotalMemory(td, mode="off", embed_fn=fe)
        assert tm_off.read("retry http call variant 2").trace_text == ""
        print("  [3] flat + off modes -> PASS")

        n_before = tm.stats()["member_impls"]
        iid = tm.write("retry http call variant 9", "return _raw9(url)",
                       "return retry(3)(_raw9)(url)", "wrap the call in a retry decorator",
                       verified=True, task_id="t9")
        assert iid is not None
        s = tm.stats()
        assert s["impls"] == 11 and s["member_impls"] == n_before + 1, "write joined a concept"
        assert tm.impls.get(iid)["concept_id"], "impl remembers its concept"
        dup = tm.write("retry http call variant 9", "return _raw9(url)",
                       "return retry(3)(_raw9)(url)", "wrap the call in a retry decorator",
                       verified=True)
        assert dup is None, "duplicate write ignored"
        print("  [4] write -> lifecycle + dedup -> PASS")
    print("\n  MEMORY.MEMORY SELFTEST -> PASS")
    return True


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seed", action="store_true", help="fable5 -> L1 + KMeans -> L2 (mpnet)")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--root", default="data/memory")
    ap.add_argument("--k-concepts", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if _selftest() else 1)
    if a.seed:
        from v5.memory.store import make_mpnet_embedder
        tm = TotalMemory(a.root, embed_fn=make_mpnet_embedder())
        tm.seed(k_concepts=a.k_concepts, limit=a.limit)
        print(f"  stats: {tm.stats()}")
    elif a.stats:
        tm = TotalMemory(a.root)
        print(tm.stats())
    else:
        ap.print_help()
