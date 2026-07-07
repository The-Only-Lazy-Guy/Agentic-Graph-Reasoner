"""v3 Stage 2 — REFINER-AS-RANKER (gated on Stage 1's boundary checklist, cleared 2026-07-08).

Repurposes `Refiner.Net` (v5/runtime/lggn_refine.py, reused UNCHANGED — no forward() changes)
for a DIFFERENT job than v1 validated it on. v1: converge h_K toward a GENERATED fix
representation in Qwen space (proven: cos 0.70-0.74, all 4 pillars load-bearing). Here:
converge h_K toward the CORRECT MEMORY RECORD in mpnet-768 space, so h_K lands directly in
`ImplStore.emb_ctx`'s space — zero new shard, zero Qwen coupling in the inference-time memory
path. Ranking, not generation — the exact distinction v2's tracer death (embedding inversion)
and this session's k_impl=2 fix both point at: Refiner.Net's convergence mechanism works,
decoding NOVEL CONTENT out of a compressed latent does not.

  g   = query embedding (why_text, mpnet-768)
  ctx = the CANDIDATE POOL itself (up to FLAT_POOL impls' own ctx embeddings, padded) — the
        refiner cross-attends over the actual records it's choosing between, not "code
        tokens" (v1's framing) or "current file" (the original Stage 2 sketch, written before
        this session's retrieval redesign made the candidate pool the natural `ctx`)
  f   = the CORRECT candidate's own embedding (ground truth: source_session_idx, task #18)
  neg = the OTHER real candidates in that SAME pool (hard-mined: the highest-scoring wrong
        one per example) — not lggn_decode.py's torch.roll shifted-batch pseudo-negative;
        genuinely harder, and instances with < 2 real candidates skip the margin term rather
        than fabricate one

query_fn's chicken-and-egg problem: `query_fn(goal, span, file_path)` (memory.py task #19)
runs BEFORE any candidate pool exists — that pool is normally the OUTPUT of a search using
the query. Resolved as a two-stage retrieve-then-rerank: `make_query_fn`'s closure does its
OWN cheap flat search (same `search_ctx` primitive read() uses) to gather a pool, THEN runs
the refiner over it. This means `make_query_fn` needs the chain's `ImplStore`, which doesn't
exist until `TotalMemory` is constructed — callers must attach query_fn AFTER construction
(`memory.query_fn = make_query_fn(...)`), not pass it into `__init__`. Wiring is task #21.

  python -m v5.runtime.memory_refiner --selftest                  # synthetic data, no GPU
  python -m v5.runtime.memory_refiner --build-reprs --model ...    # molab: real why_text
  python -m v5.runtime.memory_refiner --train --reprs artifacts/ranker_reprs.npz
"""
from __future__ import annotations

import numpy as np

from v5.memory.memory import FLAT_POOL, _merge_cand

# ── training ───────────────────────────────────────────────────────────────────────

def _train_ranker(g: np.ndarray, ctx: np.ndarray, cmask: np.ndarray, pos_idx: np.ndarray,
                  ops: np.ndarray, K: int = 3, r: int = 128, n_op: int | None = None,
                  epochs: int = 400, seed: int = 0, contrastive: float = 0.3,
                  margin: float = 0.2, k_warmup: float = 0.5, log=print):
    """g:[N,d] query embeddings, ctx:[N,T,d] candidate pool per example, cmask:[N,T] bool,
    pos_idx:[N] int index into ctx of the correct candidate, ops:[n_op,d] operator basis.
    Returns the trained net (eval mode, on CPU).

    Loss = fidelity (h_K -> the correct candidate's own embedding) + a contrastive margin
    against the HARDEST real negative in the same pool (highest cos(h_K, wrong_candidate)),
    masked out for examples with < 2 real candidates (nothing to contrast against)."""
    import time

    import torch

    from v5.runtime.lggn_refine import Refiner
    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, T, d = ctx.shape
    net = Refiner.Net(d, r=r, n_op=ops.shape[0], max_K=max(K, 8)).to(dev)
    T_ = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt).to(dev)
    gt, ct, ot = T_(g), T_(ctx), T_(ops)
    cm = torch.as_tensor(cmask).to(dev)
    pos = torch.as_tensor(pos_idx, dtype=torch.long).to(dev)
    n_real = cm.sum(1)
    has_neg = n_real > 1
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Fn = torch.nn.functional.cosine_similarity
    t0 = time.time()
    for ep in range(epochs):
        K_cur = 1 + int((K - 1) * min(1.0, ep / max(1, epochs * k_warmup - 1)))
        net.train(); opt.zero_grad()
        h = net(gt, ct, cm, ot, K_cur, True, True)              # [N, d]
        f_pos = ct[torch.arange(N, device=dev), pos]             # [N, d]
        loss = (1 - Fn(h, f_pos)).mean()
        if contrastive > 0 and has_neg.any():
            sims = Fn(h.unsqueeze(1), ct, dim=-1)                 # [N, T]
            sims = sims.masked_fill(~cm, -1e9)
            sims = sims.scatter(1, pos.unsqueeze(1), -1e9)        # exclude the positive itself
            hard_neg_sim, _ = sims.max(1)                          # [N] hardest wrong candidate
            neg_term = torch.clamp(hard_neg_sim - margin, min=0)[has_neg]
            loss = loss + contrastive * neg_term.mean()
        loss.backward(); opt.step()
        if (ep + 1) % 100 == 0 or ep == epochs - 1:
            log(f"      ep {ep+1}/{epochs} loss={loss.item():.4f} ({time.time()-t0:.0f}s)")
    net.eval()
    net.cpu()
    return net


def _hit_rate(net, g, ctx, cmask, pos_idx, ops, K=3) -> float:
    """Fraction of examples where argmax cos(h_K, ctx) over REAL slots == pos_idx."""
    import torch
    T_ = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt)
    with torch.no_grad():
        h = net(T_(g), T_(ctx), torch.as_tensor(cmask), T_(ops), K, True, True)
        sims = torch.nn.functional.cosine_similarity(h.unsqueeze(1), T_(ctx), dim=-1)
        sims = sims.masked_fill(~torch.as_tensor(cmask), -1e9)
        pred = sims.argmax(1).numpy()
    return float((pred == pos_idx).mean())


# ── training data collection (molab: needs a Stage-1-trained RawLM for real why_text) ────

def build_ranker_reprs(lm, insts: list[dict], embed_fn, chains_root: str = "data/memory_chains_ranker",
                       pool_k: int = FLAT_POOL, gen_batch: int = 16, log=print) -> dict:
    """Replays chains using GOLD code (no generation/sandbox needed — gold always 'passes' by
    construction) but REAL why_text from a Stage-1-trained RawLM, mirroring project_loop.
    run_arm's per-chain memory bookkeeping and cross-chain batching. At every dependency
    session (source_session_idx is not None, task #18), builds the SAME candidate pool
    read() would (same search_ctx/concepts.retrieve primitives) and records which candidate
    is correct.

    The correct candidate is force-injected into the pool if retrieval didn't naturally
    surface it (replacing the weakest slot) — Stage 2 trains RANKING given a pool, not pool
    construction (that's the already-validated flat-supplement fix); an example where the
    answer isn't even reachable teaches the ranker nothing about ranking. How often this
    happens is logged — a high rate would mean the pools being trained on aren't realistic.

    Returns {g, ctx, cmask, pos_idx} ready for _train_ranker."""
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

    G, CTX, CMASK, POS = [], [], [], []
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
            if s.get("source_session_idx") is not None:
                n_dep += 1
                correct_sid = st["inst"]["sessions"][s["source_session_idx"]]["sid"]
                correct_impl = st["sid_to_impl"].get(correct_sid)
                if correct_impl is None:
                    n_skipped_no_source += 1
                else:
                    q = memory._embed_one(st["why_text"][:400])
                    span = session_data(s["spec"], st["current"])
                    q_span = memory._embed_one(span[:400]) if span else None
                    pool = []
                    if memory.mode == "concept":
                        meta = memory.concepts.retrieve(q, k=3)
                        p = [i for c in meta for i in c["impl_ids"]]
                        pool = memory.impls.search_ctx(q, k=pool_k, within=p) if p else []
                    pool = _merge_cand(pool, memory.impls.search_ctx(q, k=pool_k))
                    if q_span is not None:
                        pool = _merge_cand(pool, memory.impls.search_ctx(q_span, k=pool_k))
                    cand_ids = [iid for iid, _ in pool][:pool_k]
                    if correct_impl not in cand_ids:
                        n_injected += 1
                        if len(cand_ids) >= pool_k:
                            cand_ids[-1] = correct_impl
                        else:
                            cand_ids.append(correct_impl)
                    pos_i = cand_ids.index(correct_impl)
                    ctx_vecs = memory.impls.emb_ctx.get(cand_ids)
                    T = len(cand_ids)
                    pad = pool_k - T
                    if pad > 0:
                        ctx_vecs = np.concatenate(
                            [ctx_vecs, np.zeros((pad, ctx_vecs.shape[1]), ctx_vecs.dtype)], 0)
                    G.append(q); CTX.append(ctx_vecs)
                    CMASK.append([True] * T + [False] * pad); POS.append(pos_i)
                    n_kept += 1

            gold_body = s["gold"][target]
            impl_id = memory.write(goal=s["spec"], old=st["current"], new=gold_body,
                                   trace=s["spec"][:400], verified=True,
                                   file_path=target, task_id=s["sid"], kind=s["kind"])
            if impl_id:
                st["sid_to_impl"][s["sid"]] = impl_id
            st["repo"][target] = gold_body
        log(f"  [ranker-data] depth {depth+1}/{max_depth}: dep_seen={n_dep} kept={n_kept}")

    log(f"  [ranker-data] {n_kept}/{n_dep} dependency sessions -> training examples "
        f"({n_skipped_no_source} skipped: source impl never written, {n_injected} needed "
        f"force-injection into the pool)")
    return {"g": np.asarray(G, dtype=np.float32), "ctx": np.asarray(CTX, dtype=np.float32),
            "cmask": np.asarray(CMASK, dtype=bool), "pos_idx": np.asarray(POS, dtype=np.int64)}


# ── inference-time closure + checkpoint I/O ───────────────────────────────────────────

def make_query_fn(net, ops, embed_fn, impl_store, K: int = 3, pool_k: int = FLAT_POOL):
    """(goal, span, file_path) -> refined 768-d vector, for TotalMemory.query_fn.

    impl_store must be the SAME chain's ImplStore the caller's TotalMemory uses — bind this
    AFTER TotalMemory construction (memory.query_fn = make_query_fn(..., memory.impls, ...)),
    not passed into TotalMemory.__init__ (task #21 wiring). file_path is unused (the refiner
    ranks by content, not by which file is being edited -- SAME_FILE_BOOST is a Stage-1-only
    heuristic, deliberately not reproduced here).

    span DOES feed pool construction (merged with goal via the same _merge_cand technique
    Stage 1's read() uses), for consistency with how build_ranker_reprs builds TRAINING
    pools -- a query_fn that gathers candidates differently at inference than the ranker was
    trained on is a real distribution-shift risk. NOTE: at this benchmark's per-chain scale
    (~3-5 records total by a chain's later sessions, well under pool_k), goal-alone vs
    goal+span likely return the SAME candidate set regardless (search_ctx returns everything
    when the store is smaller than k) -- so this fix is about correctness/consistency and
    matters more as real repos scale past pool_k, not necessarily the explanation for any
    specific small-benchmark result. Don't over-attribute a single eval regression to it
    without checking pool sizes first."""
    import torch

    from v5.memory.store import stable_id
    net.eval()

    def _embed(text: str) -> np.ndarray:
        key = stable_id("q", text)
        return np.asarray(embed_fn({key: text})[key], dtype=np.float32)

    def query_fn(goal: str, span: str, file_path: str) -> np.ndarray:
        g = _embed((goal or "")[:400])
        cand = impl_store.search_ctx(g, k=pool_k)
        if span:
            cand = _merge_cand(cand, impl_store.search_ctx(_embed(span[:400]), k=pool_k))
        if not cand:
            return g                                          # cold store -- nothing to refine over
        ids = [iid for iid, _ in cand]
        ctx = impl_store.emb_ctx.get(ids)
        T = len(ids)
        pad = pool_k - T
        if pad > 0:
            ctx = np.concatenate([ctx, np.zeros((pad, ctx.shape[1]), ctx.dtype)], 0)
        cmask = np.array([True] * T + [False] * pad)
        with torch.no_grad():
            h = net(torch.as_tensor(g[None], dtype=torch.float32),
                    torch.as_tensor(ctx[None], dtype=torch.float32),
                    torch.as_tensor(cmask[None]),
                    torch.as_tensor(ops, dtype=torch.float32), K, True, True)
        return h[0].numpy()

    return query_fn


def save_ranker(net, ops: np.ndarray, K: int, r: int, out_dir: str) -> None:
    from pathlib import Path

    import torch
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), str(d / "ranker.pt"))
    np.savez(str(d / "ranker_meta.npz"), ops=ops, K=K, r=r, d=ops.shape[1], n_op=ops.shape[0])


def load_ranker(path: str):
    """-> (net, ops, K), matching project_loop.py's `net, ops, K_r = ranker` unpack."""
    from pathlib import Path

    import torch

    from v5.runtime.lggn_refine import Refiner
    d = Path(path)
    meta = np.load(str(d / "ranker_meta.npz"))
    ops, K, r = meta["ops"], int(meta["K"]), int(meta["r"])
    net = Refiner.Net(int(meta["d"]), r=r, n_op=int(meta["n_op"]), max_K=max(K, 8))
    net.load_state_dict(torch.load(str(d / "ranker.pt"), map_location="cpu"))
    net.eval()
    return net, ops, K


# ── selftest (synthetic data, no GPU/network) ─────────────────────────────────────────

def _synth_pool(N=400, T=6, d=32, n_op=8, seed=0):
    """Synthetic ranking task where raw cosine nearest-neighbor of g is SYSTEMATICALLY WRONG,
    so beating it proves the refiner's cross-attention mechanism (not just memorized cosine)
    is doing the work -- same spirit as lggn_refine.py's _synth() (a fixed CUE only a learned
    projection can pick up, not raw similarity).

    Every example gets a DECOY planted at g + tiny noise (wins raw cosine trivially, cos~0.98)
    and the CORRECT candidate shares only a fixed `tag` direction with g (unrelated base
    vector otherwise, cos~-0.1..0.3) -- a bias-free linear attention query CAN learn to
    project onto the tag direction (g always contains a strong, consistent tag component),
    but raw full-vector cosine cannot separate "shares the tag" from "is just a close copy"."""
    rng = np.random.RandomState(seed)
    ops = rng.randn(n_op, d).astype("float32")
    ops /= np.linalg.norm(ops, axis=1, keepdims=True)
    tag = rng.randn(d).astype("float32"); tag /= np.linalg.norm(tag)
    G, CTX, CMASK, POS = [], [], [], []
    for _ in range(N):
        n_real = rng.randint(3, T + 1)
        g = 0.3 * rng.randn(d).astype("float32") + 0.8 * tag
        correct = 0.3 * rng.randn(d).astype("float32") + 0.8 * tag
        decoy = g + 0.05 * rng.randn(d).astype("float32")
        others = [0.3 * rng.randn(d).astype("float32") for _ in range(n_real - 2)]
        pool_list = [correct, decoy] + others
        perm = rng.permutation(len(pool_list))
        pool_arr = np.stack(pool_list)[perm].astype("float32")
        pos_i = int(np.where(perm == 0)[0][0])
        pad = T - len(pool_list)
        ctx = np.concatenate([pool_arr, np.zeros((pad, d), "float32")], 0) if pad else pool_arr
        G.append(g); CTX.append(ctx)
        CMASK.append([True] * len(pool_list) + [False] * pad); POS.append(pos_i)
    return (np.asarray(G, "float32"), np.asarray(CTX, "float32"),
            np.asarray(CMASK, bool), np.asarray(POS, "int64"), ops)


def _selftest() -> bool:
    print("memory_refiner --selftest: synthetic ranking task, contrastive loss, save/load, "
          "query_fn wiring (no GPU/network)\n")
    g, ctx, cmask, pos_idx, ops = _synth_pool(seed=0)
    n_tr = 320
    net = _train_ranker(g[:n_tr], ctx[:n_tr], cmask[:n_tr], pos_idx[:n_tr], ops,
                        K=3, r=64, epochs=150, log=lambda *a: None)
    hit = _hit_rate(net, g[n_tr:], ctx[n_tr:], cmask[n_tr:], pos_idx[n_tr:], ops, K=3)
    # baseline: nearest-neighbor of g itself (no refinement) -- must be WORSE, since g is
    # constructed as a displacement AWAY from the correct candidate
    import torch
    Fn = torch.nn.functional.cosine_similarity
    gt, ct = torch.as_tensor(g[n_tr:]), torch.as_tensor(ctx[n_tr:])
    sims0 = Fn(gt.unsqueeze(1), ct, dim=-1).masked_fill(~torch.as_tensor(cmask[n_tr:]), -1e9)
    base_hit = float((sims0.argmax(1).numpy() == pos_idx[n_tr:]).mean())
    assert hit > base_hit + 0.15, f"trained ranker ({hit:.2f}) should beat raw NN ({base_hit:.2f})"
    print(f"  [1] trained hit-rate {hit:.2f} > raw-NN baseline {base_hit:.2f} -> PASS")

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        save_ranker(net, ops, K=3, r=64, out_dir=td)
        net2, ops2, K2 = load_ranker(td)
        assert K2 == 3 and np.allclose(ops2, ops)
        hit2 = _hit_rate(net2, g[n_tr:], ctx[n_tr:], cmask[n_tr:], pos_idx[n_tr:], ops2, K=K2)
        assert abs(hit2 - hit) < 1e-6, f"save/load roundtrip changed hit-rate: {hit} vs {hit2}"
    print("  [2] save/load roundtrip -> PASS")

    # query_fn wiring: a fake ImplStore-like object exposing search_ctx/emb_ctx.get, proving
    # make_query_fn's closure reaches the pool and returns a same-shape vector.
    class _FakeImplStore:
        def __init__(self, vecs):
            self._vecs = vecs
        def search_ctx(self, q, k=8):
            return [(str(i), 0.5) for i in range(min(k, len(self._vecs)))]
        class _Emb:
            def __init__(self, outer):
                self._outer = outer
            def get(self, ids):
                return np.stack([self._outer._vecs[int(i)] for i in ids])
        @property
        def emb_ctx(self):
            return self._Emb(self)

    def fake_embed(texts: dict) -> dict:
        return {k: np.random.RandomState(abs(hash(v)) % (2**31)).randn(32).astype("float32")
               for k, v in texts.items()}

    store = _FakeImplStore(ctx[0])
    qfn = make_query_fn(net, ops, fake_embed, store, K=3, pool_k=6)
    out = qfn("some goal", "some span", "f.py")
    assert out.shape == (32,), f"query_fn output shape {out.shape}"
    empty_store = _FakeImplStore(np.zeros((0, 32), "float32"))
    empty_store.search_ctx = lambda q, k=8: []
    out2 = make_query_fn(net, ops, fake_embed, empty_store, K=3)("g", "s", "f.py")
    assert out2.shape == (32,), "cold store falls back to plain embed, still a valid vector"
    print("  [3] make_query_fn: pool search -> refine -> vector, cold-store fallback -> PASS")
    print("\n  MEMORY_REFINER SELFTEST -> PASS")
    return True


def main():
    import argparse
    ap = argparse.ArgumentParser(description="v3 Stage 2 — refiner-as-ranker.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--build-reprs", action="store_true", help="molab: real why_text + gold chains")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--lora", default="artifacts/project_lora")
    ap.add_argument("--reprs", default="artifacts/ranker_reprs.npz")
    ap.add_argument("--out", default="artifacts/memory_ranker")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--r", type=int, default=128)
    ap.add_argument("--n-op", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-seeds", type=int, default=30,
                    help="TRAIN seed chains per archetype for --build-reprs (default 30, "
                         "matching project_loop.TRAIN_SEEDS' width -- widen this first if "
                         "the ranker's train hit-rate is low; build_ranker_reprs needs no "
                         "generation/sandbox so more seeds is cheap. Always starts at 100, "
                         "disjoint from EVAL_SEEDS=range(0,20) -- no train/eval leakage "
                         "regardless of how wide this gets.")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if _selftest() else 1)
    if a.build_reprs:
        from v5.memory.store import make_mpnet_embedder
        from v5.runtime.lggn_realizer import RawLM
        from v5.runtime.project_gen import make_split
        lm = RawLM.load_checkpoint(a.model, a.lora)
        insts = make_split(seeds=range(100, 100 + a.n_seeds))
        reprs = build_ranker_reprs(lm, insts, make_mpnet_embedder(), log=print)
        from pathlib import Path
        Path(a.reprs).parent.mkdir(parents=True, exist_ok=True)
        np.savez(a.reprs, **reprs)
        print(f"  [build-reprs] {len(reprs['g'])} examples -> {a.reprs}")
        return
    if a.train:
        from v5.runtime.lggn_refine import _random_ops
        d = np.load(a.reprs)
        g, ctx, cmask, pos_idx = d["g"], d["ctx"], d["cmask"], d["pos_idx"]
        disp = ctx[np.arange(len(g)), pos_idx] - g
        ops = _random_ops(disp, a.n_op, a.seed)
        net = _train_ranker(g, ctx, cmask, pos_idx, ops, K=a.k, r=a.r,
                            epochs=a.epochs, seed=a.seed, log=print)
        hit = _hit_rate(net, g, ctx, cmask, pos_idx, ops, K=a.k)
        print(f"  [train] train-set hit-rate {hit:.3f} (n={len(g)})")
        save_ranker(net, ops, K=a.k, r=a.r, out_dir=a.out)
        print(f"  [train] checkpoint -> {a.out}")
        return
    ap.print_help()


if __name__ == "__main__":
    main()
