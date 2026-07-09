"""v4 — Iterative Traversal Ranker (latent-only multi-hop).

Wraps the proven v3 Refiner.Net + feat_proj in a multi-hop loop. Each hop:
  1. Retrieves candidate pool using current h (in mpnet-768) as query
  2. Builds ctx from candidate embeddings + feature fusion
  3. Runs Refiner.Net at K=1 for one refinement step → new h
  4. Reads top-k records from pool using h's cosine similarity
  5. h IS the query for the next hop — no LM generation between hops

Key difference from v3 (one-shot, single pool, K>=1):
  v4 does SEQUENTIAL hops with CHANGING pools. Each hop excludes records already
  found, so h naturally migrates toward the next needed record via cosine similarity
  in mpnet space.

The conditional dependency works because:
  - Hop 1 converges toward record A (e.g., config.py with PREFERRED_STRATEGY="nash")
  - h_1 carries A's content signature (keyword "nash" is in A's embedding)
  - search_ctx(h_1) at hop 2 retrieves records sharing that signature
  - Refiner.Net at hop 2 picks the correct conditional target from the new pool

  python -m v5.runtime.traversal_ranker --selftest
"""
from __future__ import annotations

import numpy as np

from v5.memory.memory import _merge_cand, FLAT_POOL
from v5.runtime.memory_refiner import _candidate_features, N_FEAT, make_query_fn


class TraversalResult:
    """Output of a traversal retrieval."""
    def __init__(self, records: list, hop_count: int, final_h: np.ndarray,
                 hop_records: list[list] | None = None,
                 hop_hs: list[np.ndarray] | None = None):
        self.records = records
        self.hops = hop_count
        self.final_h = final_h
        self.hop_records = hop_records or [records]
        self.hop_hs = hop_hs or []


class TraversalRanker:
    """Iterative latent multi-hop retrieval.

    Wraps Refiner.Net + feat_proj in a loop where h_K at hop k IS the query for
    hop k+1. No LM text generation between hops — the traversal is pure mpnet space.

    Args:
        impl_store: ImplStore from TotalMemory
        concepts: ConceptStore from TotalMemory
        embed_fn: mpnet embedder function
        refiner_net: trained Refiner.Net (from memory_refiner)
        feat_proj: trained Linear adapter for feature fusion
        ops: operator basis vectors
        pool_k: max candidates per hop
        k_impl: records to keep per hop
        max_hops: max traversal depth
        gap_detector: optional GapDetector instance
        K_steps: refinement steps per hop (1=v4 default, K>1 possible)
    """

    def __init__(self, impl_store, concepts, embed_fn,
                 refiner_net=None, feat_proj=None, ops=None,
                 pool_k: int = FLAT_POOL, k_impl: int = 2, max_hops: int = 3,
                 gap_detector=None, K_steps: int = 1):
        self.impl_store = impl_store
        self.concepts = concepts
        self.embed_fn = embed_fn
        self.refiner_net = refiner_net
        self.feat_proj = feat_proj
        self.ops = ops
        self.pool_k = pool_k
        self.k_impl = k_impl
        self.max_hops = max_hops
        self.gap_detector = gap_detector
        self.K_steps = K_steps

    def _embed(self, text: str) -> np.ndarray:
        """Single-text mpnet embedding."""
        from v5.memory.store import stable_id
        key = stable_id("q", text)
        return np.asarray(self.embed_fn({key: text})[key], dtype=np.float32)

    def _search_ctx(self, h: np.ndarray, exclude: set[str]) -> list[tuple[str, float]]:
        """Search ImplStore by cosine, excluding already-found IDs."""
        cand = self.impl_store.search_ctx(h, k=self.pool_k)
        cand = [(iid, s) for iid, s in cand if iid not in exclude]
        return cand[:self.pool_k]

    def _build_ctx_and_feats(self, cand: list[tuple[str, float]], goal: str,
                              span: str, file_path: str):
        """Build ctx matrix and feature matrix from candidate list."""
        ids = [iid for iid, _ in cand]
        if not ids:
            return np.zeros((0, 768), dtype=np.float32), np.zeros((0, N_FEAT), dtype=np.float32)
        ctx = self.impl_store.emb_ctx.get(ids)
        feats = np.stack([_candidate_features(self.impl_store.get(iid), self.concepts,
                                              goal, span, file_path) for iid in ids])
        return ctx, feats

    def _pad(self, ctx, feats, T):
        """Pad to pool_k with zero vectors."""
        pad = self.pool_k - T
        if pad > 0:
            ctx = np.concatenate([ctx, np.zeros((pad, ctx.shape[1]), ctx.dtype)], 0)
            feats = np.concatenate([feats, np.zeros((pad, feats.shape[1]), feats.dtype)], 0)
        return ctx, feats, [True] * T + [False] * pad

    def _refine(self, h: np.ndarray, ctx: np.ndarray, feats: np.ndarray,
                 cmask: list[bool]) -> np.ndarray:
        """Run Refiner.Net over one pool, K_steps refinement."""
        if self.refiner_net is None or len(ctx) == 0 or not any(cmask):
            return h
        import torch
        with torch.no_grad():
            ctx_t = torch.as_tensor(ctx[None], dtype=torch.float32)
            ct_in = ctx_t
            if self.feat_proj is not None and feats is not None:
                ct_in = ctx_t + self.feat_proj(torch.as_tensor(feats[None], dtype=torch.float32))
            h_out = self.refiner_net(
                torch.as_tensor(h[None], dtype=torch.float32),
                ct_in,
                torch.as_tensor([cmask], dtype=torch.bool),
                torch.as_tensor(self.ops, dtype=torch.float32) if self.ops is not None else None,
                self.K_steps, True, True
            )
        return h_out[0].numpy()

    def _top_k(self, h: np.ndarray, cand: list[tuple[str, float]],
                ctx: np.ndarray) -> list[dict]:
        """Select top-k records from candidates by cosine(h, ctx).

        NOTE: do NOT filter by sims[i] > 0. mpnet cosine between distinct-but-relevant
        texts is routinely small or slightly negative; after hop 0 refines h toward the
        first source, the *next* needed source often has cosine ~0 or negative and would
        be wrongly discarded — collapsing multi-hop traversal to a single record (this was
        the root cause of DEP=0.000 on `compose`). `search_ctx` already returns the top
        pool_k most-similar candidates, so returning the top-k_impl of those by similarity
        is the correct readout.
        """
        ids = [iid for iid, _ in cand]
        if not ids or len(ctx) == 0:
            return []
        sims = np.dot(h, ctx.T) / (np.linalg.norm(h) * np.linalg.norm(ctx, axis=1) + 1e-9)
        top_indices = np.argsort(-sims)[:self.k_impl]
        return [self.impl_store.get(ids[i]) for i in top_indices if i < len(ids)]

    def retrieve(self, goal: str, span: str = "", file_path: str = "",
                 initial_h: np.ndarray | None = None) -> TraversalResult:
        """Run traversal retrieval. Returns accumulated records across all hops.

        Args:
            goal: query text (spec + current context)
            span: additional context text
            file_path: target file path
            initial_h: optional starting query vector (default: embed(goal))
        """
        h = initial_h if initial_h is not None else self._embed((goal or "")[:400])
        seen_ids: set[str] = set()
        all_records: list[dict] = []
        hop_records: list[list[dict]] = []
        hop_hs: list[np.ndarray] = []
        n_hops = 0

        for hop in range(self.max_hops):
            n_hops += 1
            cand = self._search_ctx(h, exclude=seen_ids)
            if not cand:
                break

            ctx, feats = self._build_ctx_and_feats(cand, goal, span, file_path)
            if len(ctx) == 0:
                break

            T = len([iid for iid, _ in cand])
            ctx_pad, feats_pad, cmask = self._pad(ctx, feats, T)
            h = self._refine(h, ctx_pad, feats_pad, cmask)
            hop_hs.append(h.copy())

            records = self._top_k(h, cand, ctx)
            hop_records.append(records)
            for rec in records:
                rid = rec.get("impl_id", "") or id(rec)
                if rid not in seen_ids:
                    seen_ids.add(rid)
                    all_records.append(rec)

            if self.gap_detector is not None:
                if self.gap_detector.should_stop(h, hop, self.max_hops):
                    break

        return TraversalResult(
            records=all_records, hop_count=n_hops, final_h=h,
            hop_records=hop_records, hop_hs=hop_hs
        )


# ── training data collection: multi-hop traversal reprs ────────────────────────────

def build_traversal_reprs(lm, insts: list[dict], embed_fn,
                           chains_root: str = "data/memory_chains_traversal",
                           pool_k: int = FLAT_POOL, gen_batch: int = 16,
                           log=print) -> dict:
    """Build training examples for multi-hop traversal ranker.

    For each dependency session with source_session_idxs (a LIST), this records
    the sequence of hops needed:
      - hop 1: pool = all records, positive = source_session_idxs[0]
      - hop 2: pool = (all records except source_session_idxs[0]),
               positive = source_session_idxs[1]
      - ... until all sources found

    Each hop is a TRAINING EXAMPLE for the ranker: (h_in, ctx, feats, pos_idx).
    The labels for consecutive hops are known from the gold chain structure.

    For non-list source_session_idx (single-source), only 1 hop is recorded.
    """
    import shutil
    from pathlib import Path

    from v5.memory.memory import TotalMemory
    from v5.runtime.lggn_realizer import why_prompt
    from v5.runtime.project_loop import WHY_MAX_NEW, _gen_chunked, session_data

    states = []
    for inst in insts:
        chain_dir = Path(chains_root) / inst["instance_id"]
        if chain_dir.exists():
            shutil.rmtree(chain_dir)
        memory = TotalMemory(chain_dir / "mem", mode="concept", embed_fn=embed_fn)
        states.append({"inst": inst, "memory": memory, "repo": {}, "sid_to_impl": {}})

    H_IN, CTX, FEATS, CMASK, POS = [], [], [], [], []
    n_dep = n_kept = n_injected = n_skipped_no_source = 0

    max_depth = max((len(st["inst"]["sessions"]) for st in states), default=0)
    for depth in range(max_depth):
        active = [st for st in states if depth < len(st["inst"]["sessions"])]
        if not active:
            continue
        for st in active:
            s = st["inst"]["sessions"][depth]
            target = s["target_file"]
            if s.get("buggy"):
                st["repo"][target] = s["buggy"][target]
            st["s"], st["target"] = s, target
            st["current"] = st["repo"].get(target, "")

        wps = [why_prompt(st["s"]["spec"], st["current"]) for st in active]
        outs = _gen_chunked(lm, wps, WHY_MAX_NEW, gen_batch)
        for st, out in zip(active, outs):
            st["why_text"] = (out or "").strip() or st["s"]["spec"]

        for st in active:
            s, target, memory = st["s"], st["target"], st["memory"]
            src_idxs = s.get("source_session_idxs")
            if not src_idxs:
                idx = s.get("source_session_idx")
                src_idxs = [idx] if idx is not None else []

            if src_idxs:
                n_dep += 1
                correct_sids = [st["inst"]["sessions"][j]["sid"] for j in src_idxs]
                correct_impls = [ci for ci in
                                 (st["sid_to_impl"].get(sid) for sid in correct_sids)
                                 if ci is not None]
                if not correct_impls:
                    n_skipped_no_source += 1
                    continue

                # Traversal mode does NOT run Call A at inference (goal_for_query == spec),
                # so training must embed the SAME spec the retrieve() loop will embed.
                q = memory._embed_one((s["spec"] or "")[:400])
                span = session_data(s["spec"], st["current"])
                q_span = memory._embed_one(span[:400]) if span else None

                # Build base pool (same as v3 build_ranker_reprs)
                pool = []
                if memory.mode == "concept":
                    meta = memory.concepts.retrieve(q, k=3)
                    p = [i for c in meta for i in c["impl_ids"]]
                    pool = memory.impls.search_ctx(q, k=pool_k, within=p) if p else []
                pool = _merge_cand(pool, memory.impls.search_ctx(q, k=pool_k))
                if q_span is not None:
                    pool = _merge_cand(pool, memory.impls.search_ctx(q_span, k=pool_k))
                pool_ids = [iid for iid, _ in pool][:pool_k]

                # Generate one training example PER HOP. First hop: pool includes ALL
                # sources; subsequent hops exclude already-found sources.
                found = set()
                for hop_i, ci in enumerate(correct_impls):
                    cand_ids = list(pool_ids)
                    if hop_i > 0:
                        cand_ids = [iid for iid in cand_ids if iid not in found]

                    # Force-inject this hop's target if missing
                    if ci not in cand_ids:
                        n_injected += 1
                        if len(cand_ids) < pool_k:
                            cand_ids.append(ci)
                        else:
                            for r in range(len(cand_ids) - 1, -1, -1):
                                if cand_ids[r] not in correct_impls:
                                    cand_ids[r] = ci
                                    break
                            else:
                                cand_ids[-1] = ci

                    ctx_vecs = memory.impls.emb_ctx.get(cand_ids)
                    feat_vecs = np.stack([
                        _candidate_features(memory.impls.get(iid), memory.concepts,
                                            s["spec"], span, target)
                        for iid in cand_ids])
                    T = len(cand_ids)
                    pad = pool_k - T
                    if pad > 0:
                        ctx_vecs = np.concatenate(
                            [ctx_vecs, np.zeros((pad, ctx_vecs.shape[1]), ctx_vecs.dtype)], 0)
                        feat_vecs = np.concatenate(
                            [feat_vecs, np.zeros((pad, feat_vecs.shape[1]), feat_vecs.dtype)], 0)
                    cmask_row = [True] * T + [False] * pad
                    if ci in cand_ids:
                        H_IN.append(q)
                        CTX.append(ctx_vecs)
                        FEATS.append(feat_vecs)
                        CMASK.append(cmask_row)
                        POS.append(cand_ids.index(ci))
                        n_kept += 1
                    found.add(ci)

            gold_body = s["gold"][target]
            impl_id = memory.write(goal=s["spec"], old=st["current"], new=gold_body,
                                   trace=s["spec"][:400], verified=True,
                                   file_path=target, task_id=s["sid"], kind=s["kind"])
            if impl_id:
                st["sid_to_impl"][s["sid"]] = impl_id
            st["repo"][target] = gold_body
        log(f"  [traversal-data] depth {depth+1}/{max_depth}: dep_seen={n_dep}")

    log(f"  [traversal-data] {n_kept}/{n_dep} dependency sessions -> {n_kept} hop-examples "
        f"({n_skipped_no_source} skipped, {n_injected} force-injections)")
    return {"h_in": np.asarray(H_IN, dtype=np.float32),
            "ctx": np.asarray(CTX, dtype=np.float32),
            "feats": np.asarray(FEATS, dtype=np.float32),
            "cmask": np.asarray(CMASK, dtype=bool),
            "pos_idx": np.asarray(POS, dtype=np.int64)}


def _train_traversal_ranker(h_in, ctx, cmask, pos_idx, ops, feats=None,
                            K: int = 1, r: int = 128, n_op: int | None = None,
                            epochs: int = 400, seed: int = 0,
                            contrastive: float = 0.3, margin: float = 0.2,
                            k_warmup: float = 0.5, log=print):
    """Train Refiner.Net for traversal (K=1 per hop, designed for multi-hop stacking).

    Identical loss to v3 _train_ranker: fidelity(h_K → pos_emb) + contrastive margin
    against hardest negative. The only difference: K is 1 by default (one refinement
    step per hop, since the traversal loop itself provides the multi-step iteration).

    Returns (net, feat_proj) for use in TraversalRanker.
    """
    import torch
    import torch.nn as nn
    from v5.runtime.lggn_refine import Refiner
    from v5.runtime.memory_refiner import N_FEAT

    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, T, d = ctx.shape
    if feats is None:
        feats = np.zeros((N, T, N_FEAT), dtype=np.float32)
    F = feats.shape[-1]
    net = Refiner.Net(d, r=r, n_op=ops.shape[0], max_K=max(K, 8)).to(dev)
    feat_proj = nn.Linear(F, d, bias=False).to(dev)
    T_ = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt).to(dev)
    gt, ct_raw, ot, ft = T_(h_in), T_(ctx), T_(ops), T_(feats)
    cm = torch.as_tensor(cmask).to(dev)
    pos = torch.as_tensor(pos_idx, dtype=torch.long).to(dev)
    n_real = cm.sum(1)
    has_neg = n_real > 1
    opt = torch.optim.Adam(list(net.parameters()) + list(feat_proj.parameters()),
                           lr=1e-3, weight_decay=1e-4)
    Fn = torch.nn.functional.cosine_similarity
    for ep in range(epochs):
        K_cur = 1 + int((K - 1) * min(1.0, ep / max(1, epochs * k_warmup - 1)))
        net.train(); feat_proj.train(); opt.zero_grad()
        ct_in = ct_raw + feat_proj(ft)
        h = net(gt, ct_in, cm, ot, K_cur, True, True)
        f_pos = ct_raw[torch.arange(N, device=dev), pos]
        loss = (1 - Fn(h, f_pos)).mean()
        if contrastive > 0 and has_neg.any():
            sims = Fn(h.unsqueeze(1), ct_raw, dim=-1)
            sims = sims.masked_fill(~cm, -1e9)
            sims = sims.scatter(1, pos.unsqueeze(1), -1e9)
            hard_neg_sim, _ = sims.max(1)
            neg_term = torch.clamp(hard_neg_sim - margin, min=0)[has_neg]
            loss = loss + contrastive * neg_term.mean()
        loss.backward(); opt.step()
        if (ep + 1) % 100 == 0 or ep == epochs - 1:
            log(f"      ep {ep+1}/{epochs} loss={loss.item():.4f}")
    net.eval(); feat_proj.eval()
    net.cpu(); feat_proj.cpu()
    return net, feat_proj


def _hit_rate(net, feat_proj, h_in, ctx, cmask, pos_idx, feats, ops, K=1) -> float:
    """Fraction of examples where argmax cos(h_K, ctx) == pos_idx."""
    import torch
    T_ = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt)
    with torch.no_grad():
        ctx_t = T_(ctx)
        ct_in = ctx_t + feat_proj(T_(feats))
        h = net(T_(h_in), ct_in, torch.as_tensor(cmask), T_(ops), K, True, True)
        sims = torch.nn.functional.cosine_similarity(h.unsqueeze(1), ctx_t, dim=-1)
        sims = sims.masked_fill(~torch.as_tensor(cmask), -1e9)
        pred = sims.argmax(1).numpy()
    return float((pred == pos_idx).mean())


# ── selftest ─────────────────────────────────────────────────────────────────────

def _synth_traversal(N=200, T=6, d=32, n_op=8, n_hops=2, seed=0):
    """Synthetic multi-hop ranking: 2 hops with conditional dependency.

    Hop 1: pool contains a DECOY + the TRIGGER (mimics config.py). g has tag + noise.
    Hop 2: pool contains TARGET A and TARGET B (mimics nash.py and maxmin.py).
           The trigger record from hop 1's tag determines which target is correct at hop 2.
           A correctly refined h_1 carries the trigger's tag; search with h_1 naturally
           favors the matching target over the non-matching one.
    """
    rng = np.random.RandomState(seed)
    ops = rng.randn(n_op, d).astype("float32")
    ops /= np.linalg.norm(ops, axis=1, keepdims=True)

    tag_a = rng.randn(d).astype("float32"); tag_a /= np.linalg.norm(tag_a)
    tag_b = rng.randn(d).astype("float32"); tag_b /= np.linalg.norm(tag_b)
    # tag_a and tag_b are orthogonal — this simulates "nash" vs "maxmin" as distinct keywords

    H_IN, CTX, FEATS, CMASK, POS = [], [], [], [], []
    for _ in range(N):
        use_a = rng.random() < 0.5
        correct_tag = tag_a if use_a else tag_b
        wrong_tag = tag_b if use_a else tag_a

        # Hop 1 pool: trigger (has correct_tag) + decoy (random)
        trigger = 0.3 * rng.randn(d).astype("float32") + 0.7 * correct_tag
        decoy = 0.3 * rng.randn(d).astype("float32")

        # Initial query: has both tags weakly + noise (mimics "strategy preference")
        g = 0.3 * rng.randn(d).astype("float32") + 0.3 * tag_a + 0.3 * tag_b

        # Hop 2 pool: target A (tag_a) and target B (tag_b), with distractors
        target_a = 0.2 * rng.randn(d).astype("float32") + 0.6 * tag_a
        target_b = 0.2 * rng.randn(d).astype("float32") + 0.6 * tag_b
        distractor = 0.5 * rng.randn(d).astype("float32")

        # Build 2-hop training example
        # Hop 1: pool = [trigger, decoy, distractor], pos = trigger
        h1_pool = np.stack([trigger, decoy, distractor]).astype("float32")
        h1_perm = rng.permutation(len(h1_pool))
        h1_arr = h1_pool[h1_perm]
        h1_pos = int(np.where(h1_perm == 0)[0][0])

        # Hop 2: pool = [target_a, target_b, decoy], pos = 0 (target_a) if use_a else 1 (target_b)
        h2_pool = np.stack([target_a, target_b, decoy]).astype("float32")
        h2_perm = rng.permutation(len(h2_pool))
        h2_arr = h2_pool[h2_perm]
        h2_pos = int(np.where(h2_perm == (0 if use_a else 1))[0][0])

        for pool_arr, pos_i in [(h1_arr, h1_pos), (h2_arr, h2_pos)]:
            T_real = len(pool_arr)
            pad = T - T_real
            if pad:
                pool_arr = np.concatenate([pool_arr, np.zeros((pad, d), "float32")], 0)
            H_IN.append(g)
            CTX.append(pool_arr)
            FEATS.append(np.zeros((T, N_FEAT), dtype=np.float32))
            CMASK.append([True] * T_real + [False] * pad)
            POS.append(pos_i)

    return (np.asarray(H_IN, "float32"), np.asarray(CTX, "float32"),
            np.asarray(FEATS, "float32"), np.asarray(CMASK, bool),
            np.asarray(POS, "int64"), ops)


def _selftest() -> bool:
    print("traversal_ranker --selftest: synthetic multi-hop, save/load, "
          "TraversalRanker.retrieve loop (no GPU/network)\n")

    # 1. Training + hit-rate on synthetic multi-hop data
    h_in, ctx, feats, cmask, pos_idx, ops = _synth_traversal(seed=0)
    n_tr = 300
    net, feat_proj = _train_traversal_ranker(
        h_in[:n_tr], ctx[:n_tr], cmask[:n_tr], pos_idx[:n_tr], ops,
        feats=feats[:n_tr], K=1, r=64, epochs=150, log=lambda *a: None)

    hit = _hit_rate(net, feat_proj, h_in[n_tr:], ctx[n_tr:], cmask[n_tr:],
                    pos_idx[n_tr:], feats[n_tr:], ops, K=1)
    import torch
    Fn = torch.nn.functional.cosine_similarity
    gt, ct = torch.as_tensor(h_in[n_tr:]), torch.as_tensor(ctx[n_tr:])
    sims0 = Fn(gt.unsqueeze(1), ct, dim=-1).masked_fill(
        ~torch.as_tensor(cmask[n_tr:]), -1e9)
    base_hit = float((sims0.argmax(1).numpy() == pos_idx[n_tr:]).mean())
    assert hit > base_hit + 0.10, \
        f"trained ranker ({hit:.2f}) should beat raw NN ({base_hit:.2f})"
    print(f"  [1] traversal ranker hit-rate {hit:.2f} > raw-NN {base_hit:.2f} -> PASS")

    # 2. Save/load roundtrip
    import tempfile
    from v5.runtime.memory_refiner import save_ranker, load_ranker
    with tempfile.TemporaryDirectory() as td:
        save_ranker(net, feat_proj, ops, K=1, r=64, out_dir=td)
        net2, feat_proj2, ops2, K2 = load_ranker(td)
        assert K2 == 1 and np.allclose(ops2, ops)
        hit2 = _hit_rate(net2, feat_proj2, h_in[n_tr:], ctx[n_tr:], cmask[n_tr:],
                         pos_idx[n_tr:], feats[n_tr:], ops2, K=1)
        assert abs(hit2 - hit) < 1e-6, f"save/load changed hit: {hit} vs {hit2}"
    print("  [2] save/load roundtrip -> PASS")

    # 3. TraversalRanker.retrieve() loop with synthetic data
    class _FakeImplStore:
        def __init__(self, vecs, n_feat=N_FEAT):
            self._vecs = vecs
            self._n_feat = n_feat
            self._call_count = 0
        def search_ctx(self, q, k=8):
            self._call_count += 1
            idxs = list(range(min(k, len(self._vecs))))
            # If exclude was passed, simulate by returning different slices per call
            # (the actual exclude filtering is in TraversalRanker._search_ctx)
            return [(str(i), 0.5) for i in idxs]
        def get(self, iid):
            return {"file_path": "f.py", "kind": "create", "concept_id": "",
                    "impl_id": iid}
        class _Emb:
            def __init__(self, outer):
                self._outer = outer
            def get(self, ids):
                return np.stack([self._outer._vecs[int(i)] for i in ids])
        @property
        def emb_ctx(self):
            return self._Emb(self)

    N_VECS = 10
    vecs = np.random.RandomState(42).randn(N_VECS, 32).astype("float32")
    store = _FakeImplStore(vecs)

    def fake_embed(texts: dict) -> dict:
        return {k: np.random.RandomState(abs(hash(v)) % (2**31)).randn(32).astype("float32")
                for k, v in texts.items()}

    ranker = TraversalRanker(store, None, fake_embed, net, feat_proj, ops,
                              pool_k=6, k_impl=2, max_hops=3, K_steps=1)
    result = ranker.retrieve("test goal", "test span", "f.py")
    assert result.hops >= 1, f"traversal should complete at least 1 hop, got {result.hops}"
    assert result.final_h.shape == (32,), f"final_h shape {result.final_h.shape}"
    print(f"  [3] TraversalRanker.retrieve: {result.hops} hops, {len(result.records)} records -> PASS")

    # 4. Cold store (no impls) — falls through gracefully
    empty_store = _FakeImplStore(np.zeros((0, 32), "float32"))
    empty_store.search_ctx = lambda q, k=8: []
    ranker2 = TraversalRanker(empty_store, None, fake_embed, net, feat_proj, ops,
                               pool_k=6, k_impl=2, max_hops=3, K_steps=1)
    result2 = ranker2.retrieve("test goal")
    assert result2.hops == 1 and len(result2.records) == 0
    print("  [4] cold store fallback -> PASS")

    print("\n  TRAVERSAL RANKER SELFTEST -> PASS")
    return True


def main():
    import argparse
    ap = argparse.ArgumentParser(description="v4 — Traversal Ranker (latent multi-hop).")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--build-reprs", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--lora", default="artifacts/project_lora")
    ap.add_argument("--reprs", default="artifacts/traversal_reprs.npz")
    ap.add_argument("--out", default="artifacts/traversal_ranker")
    ap.add_argument("--archetypes", default="preference",
                    help="comma-separated archetypes for --build-reprs")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--r", type=int, default=128)
    ap.add_argument("--n-op", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-seeds", type=int, default=30)
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if _selftest() else 1)
    if a.build_reprs:
        from v5.memory.store import make_mpnet_embedder
        from v5.runtime.lggn_realizer import RawLM
        from v5.runtime.project_gen import make_split
        lm = RawLM(a.model)   # base model only (LoRA not needed for why_text gen)
        archetypes = tuple(x.strip() for x in a.archetypes.split(",") if x.strip())
        insts = make_split(archetypes=archetypes, seeds=range(100, 100 + a.n_seeds))
        reprs = build_traversal_reprs(lm, insts, make_mpnet_embedder(), log=print)
        from pathlib import Path
        Path(a.reprs).parent.mkdir(parents=True, exist_ok=True)
        np.savez(a.reprs, **reprs)
        print(f"  [build-reprs] {len(reprs['h_in'])} hop-examples -> {a.reprs}")
        return
    if a.train:
        from v5.runtime.lggn_refine import _random_ops
        d = np.load(a.reprs)
        h_in, ctx, cmask, pos_idx = d["h_in"], d["ctx"], d["cmask"], d["pos_idx"]
        feats = d["feats"] if "feats" in d.files else None
        disp = ctx[np.arange(len(h_in)), pos_idx] - h_in
        ops = _random_ops(disp, a.n_op, a.seed)
        net, feat_proj = _train_traversal_ranker(
            h_in, ctx, cmask, pos_idx, ops, feats=feats,
            K=a.k, r=a.r, epochs=a.epochs, seed=a.seed, log=print)
        hit = _hit_rate(net, feat_proj, h_in, ctx, cmask, pos_idx,
                        feats if feats is not None
                        else np.zeros((*ctx.shape[:2], N_FEAT), "float32"),
                        ops, K=a.k)
        print(f"  [train] train-set hit-rate {hit:.3f} (n={len(h_in)})")
        from v5.runtime.memory_refiner import save_ranker
        save_ranker(net, feat_proj, ops, K=a.k, r=a.r, out_dir=a.out)
        print(f"  [train] checkpoint -> {a.out}")
        return
    ap.print_help()


if __name__ == "__main__":
    main()
