"""algo_grr_sessionscale — the session A/B at scale, on harder problems, over REAL KV-eviction spans.

WHY. At 40 GSM8K problems the recall baseline is 37/37 = 1.00 with a 12-pick budget, so a graph edit
has no headroom and the A/B measures nothing. Three axes are opened here, all of them real:

  SCALE      200+ problems instead of 40. Spans outnumber picks by ~50x, so retrieval must choose.
  DIFFICULTY Hendrycks MATH (cached locally) with a `level` filter, not just GSM8K word problems.
             MATH carries LaTeX and [asy] drawing blocks and its gold digits are harder to reach.
  SUBSTRATE  Real KV-cache eviction spans. `_run_stream` is called with an EMPTY query -- its own
             "absorb-only pass: no question was asked" branch -- so the document is pushed through
             the real bounded cache and every evicted span is decoded and written to a real
             SessionGraph, with NO generation anywhere. Span boundaries are set by the cache window
             rather than a word count, which is the point: gold digits straddle eviction boundaries
             and a single span can pool several problems.

[asy] IS STRIPPED FROM GOLD. A MATH [asy]...[/asy] block is drawing coordinates; counting those
digits as gold invents targets no semantic mechanism can reach (this project already measured one
problem whose entire gold was [asy] coordinates). 7 of the first 200 MATH problems contain one.

Metric unchanged and still the honest one: query = first 6 words of a problem, gold = the problem's
digits minus the cue's digits, recall = gold is a subset of the UNION of digits over retrieved
spans. No trie, no number boosting, no full-text query.

    selftest : python -m v5.runtime.algo_grr_sessionscale --selftest
    words    : python -m v5.runtime.algo_grr_sessionscale --run --dataset math --n 200
    eviction : python -m v5.runtime.algo_grr_sessionscale --run --source eviction --n 200
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("HF_HOME", r"E:\cache\hf")

import torch                                                              # noqa: E402

from v5.runtime.algo_grr_editcur import (                                 # noqa: E402
    EditGraph, EditPolicy, dynamics_stats, pair_matrix)
from v5.runtime.algo_grr_sessionwire import (                             # noqa: E402
    _digits, _make_controller, _tensors, _window_embs)

_ASY = re.compile(r"\[asy\].*?\[/asy\]", re.S)
_CACHE = {"gsm8k": r"E:\cache\hf\datasets\openai___gsm8k",
          "math": r"E:\cache\hf\datasets\qwedsacf___competition_math"}


def problem_nums(text: str) -> set:
    """Digits of a problem with [asy] drawing blocks removed -- those coordinates are not recallable
    content, and scoring them creates an unreachable gold."""
    return _digits(_ASY.sub(" ", text or ""))


def load_problems(dataset: str = "gsm8k", n: int = 200, offset: int = 0,
                  min_level: int = 0) -> list:
    """Read the local arrow cache with pyarrow (`import datasets` deadlocks after torch here). No
    model is involved in reading text."""
    import pyarrow as pa
    root = _CACHE[dataset]
    fs = sorted(glob.glob(str(Path(root) / "**" / "*.arrow"), recursive=True))
    if not fs:
        raise FileNotFoundError(f"{dataset} arrow cache not found under {root}")
    tbl = pa.ipc.open_stream(pa.memory_map(fs[0], "rb")).read_all()
    col = "question" if dataset == "gsm8k" else "problem"
    out = []
    for i in range(tbl.num_rows):
        if dataset == "math" and min_level:
            lv = str(tbl.column("level")[i].as_py() or "")
            m = re.search(r"(\d)", lv)
            if not m or int(m.group(1)) < min_level:
                continue
        t = tbl.column(col)[i].as_py()
        if not t or not problem_nums(t):
            continue                                          # no gold -> nothing to recall
        out.append(t)
        if len(out) >= n + offset:
            break
    return out[offset:]


# ================================================================================================
# two span substrates
# ================================================================================================
def graph_from_words(problems: list, span_words: int = 25) -> EditGraph:
    """Word-chunked spans chained by `follows` -- the cheap stand-in used so far."""
    g = EditGraph()
    prev = None
    for p in problems:
        w = p.split()
        for a in range(0, len(w), span_words):
            nid = f"span_{len(g.text):04d}"
            g.add(nid, " ".join(w[a:a + span_words]))
            if prev is not None:
                g.link(prev, nid, "follows")
            prev = nid
    return g


def graph_from_eviction(problems: list, lm_name: str, window: int = 512, sinks: int = 8,
                        chunk: int = 128, verbose: bool = True) -> EditGraph:
    """REAL KV-cache eviction. The document stream is pushed through the model with a bounded cache
    and an EMPTY question, so `_run_stream` takes its absorb-only branch: forward passes and
    evictions happen for real, nothing is generated. Each evicted span is decoded verbatim and
    written to a real SessionGraph, whose `follows` chain is mirrored into an EditGraph."""
    from v5.runtime.dcpd_latent import WhiteBox
    from v5.runtime.membrane import _run_stream
    from v5.runtime.membrane_session import SessionGraph
    wb = WhiteBox(lm_name, quant="4bit")
    sess = SessionGraph()
    doc = "\n\n".join(problems)
    doc_ids = wb.tok(doc, return_tensors="pt").input_ids.to(wb.device)
    q_ids = torch.empty(1, 0, dtype=torch.long, device=wb.device)
    _txt, _len, n_evict = _run_stream(wb, doc_ids, q_ids, window, sinks, chunk, sess=sess)
    if verbose:
        vram = (torch.cuda.max_memory_allocated() / 2 ** 30) if torch.cuda.is_available() else 0.0
        print(f"    stream {doc_ids.shape[1]} tokens -> {n_evict} evictions, "
              f"{len(sess.order)} spans written, peak VRAM {vram:.2f} GB", flush=True)
    g = EditGraph()
    for nm in sess.order:
        g.add(nm, sess.g.atoms[nm].code)
    for a, b, r in getattr(sess.g, "edges", []):
        if a in g.text and b in g.text and r == "follows":
            g.link(a, b, "follows")
    return g


# ================================================================================================
# the A/B
# ================================================================================================
def recall_ab(g0: EditGraph, problems: list, pol=None, top_k: int = 12, cycles: int = 4,
              picks: int = 3, cand_cap: int = 60, boost_by_strength: bool = False,
              verbose: bool = True) -> dict:
    """Identical mechanism in both arms (real select_nodes, frozen Hopfield prior, depth-1 walk);
    the only difference is whether the policy's `related` edges are present."""
    from embedder import encode_batch
    ids = sorted(g0.ids())
    if len(ids) < 8:
        return {"n": 0, "note": "too few spans"}
    trm = _make_controller(384)
    Ew, Wp = _window_embs([g0.text[n] for n in ids])
    cues = [" ".join(p.split()[:6]) for p in problems]
    golds = [problem_nums(p) - _digits(c) for p, c in zip(problems, cues)]
    Q = torch.as_tensor(encode_batch(cues), dtype=torch.float32)

    def run(g, tag):
        E, ei, et, es = _tensors(g, ids)
        tools = torch.zeros(0, E.shape[1])
        ok = tot = 0
        for k, gold in enumerate(golds):
            if not gold:
                continue
            n_idx, _e, _t = trm.select_nodes(
                Q[k], E, ei, et, es, tools, top_k=top_k, top_tools=0, cycles=cycles,
                picks_per_cycle=picks, neighbor_boost=3.0, follows_type=None,
                win_embs_lm=Ew, win_parent=Wp, boost_by_strength=boost_by_strength)
            got: set = set()
            for i in n_idx:
                got |= _digits(g0.text[ids[i]])
            tot += 1
            ok += int(gold <= got)
        if verbose:
            print(f"    {tag:<34} recall {ok}/{tot} = {ok / max(1, tot):.3f}", flush=True)
        return ok / max(1, tot), tot

    base, n = run(g0, "follows edges only")
    if pol is None:
        return {"n": n, "base": base, "edited": None, "added": 0, "spans": len(ids)}
    g1 = g0.copy()
    cofire, theta = dynamics_stats(g0)
    E0 = torch.as_tensor(encode_batch([g0.text[i] for i in ids]), dtype=torch.float32)
    E0 = E0 / (E0.norm(dim=1, keepdim=True) + 1e-9)
    sim = E0 @ E0.t()
    added = 0
    for a, s in enumerate(ids):
        # Shortlist candidates by embedding similarity. Scoring all N per source is O(N^2) feature
        # builds, and a policy that must consider every span of a 600-span session is not the
        # deployable object anyway -- a retriever always shortlists first.
        cap = min(cand_cap, len(ids) - 1)
        near = sim[a].topk(cap + 1).indices.tolist()
        cands = [ids[j] for j in near if ids[j] != s and not g0.has(s, ids[j])
                 and not g0.has(ids[j], s)]
        if len(cands) < 2:
            continue
        with torch.no_grad():
            sc = pol(pair_matrix(g0, s, cands, cofire, theta))
        # The edge carries the policy's CONFIDENCE, not just its choice: softmax-max over the
        # shortlist, which is high only when one candidate clearly wins. Under
        # boost_by_strength the recall loop then weighs a hesitant edge less than the
        # structural `follows` edges, which stay at 1.0.
        conf = float(torch.softmax(sc, dim=0).max())
        g1.link(s, cands[int(sc.argmax())], "related", weight=conf)
        added += 1
    edited, _ = run(g1, f"+ {added} policy `related` edges")
    return {"n": n, "base": base, "edited": edited, "added": added, "spans": len(ids)}


# ================================================================================================
# entry points
# ================================================================================================
def _load_policy():
    ck = Path(_ROOT) / "artifacts" / "editpolicy_multidomain.pt"
    pol = EditPolicy()
    if ck.exists():
        pol.load_state_dict(torch.load(str(ck)))
        return pol, True
    return pol, False


def _selftest() -> bool:
    print("algo_grr_sessionscale --selftest: scale / difficulty / eviction plumbing\n")
    ok = True

    def chk(t, c, d=""):
        nonlocal ok
        ok &= bool(c)
        print(f"  [{'PASS' if c else 'FAIL'}] {t}{('  - ' + d) if d else ''}")

    gs = load_problems("gsm8k", 50)
    ms = load_problems("math", 50)
    chk("[1] both corpora load from the local arrow cache",
        len(gs) == 50 and len(ms) == 50, f"gsm8k={len(gs)} math={len(ms)}")

    hard = load_problems("math", 20, min_level=4)
    chk("[2] MATH level filter selects a different, harder slice",
        len(hard) == 20 and hard != ms[:20], "level>=4 differs from unfiltered")

    asy = 'Find $x$. [asy] draw((0,0)--(7,9)); label("12"); [/asy] The side is 5.'
    chk("[3] [asy] blocks are stripped from gold",
        problem_nums(asy) == {"5"}, f"{sorted(problem_nums(asy))} (not 0/7/9/12)")

    g = graph_from_words(ms, span_words=25)
    chk("[4] MATH spans build with a follows chain",
        len(g.ids()) > 50 and sum(1 for e in g.edges if e[2] == "follows") == len(g.ids()) - 1,
        f"{len(g.ids())} spans")

    big = graph_from_words(load_problems("gsm8k", 200), span_words=25)
    chk("[5] at 200 problems spans FAR outnumber the pick budget (headroom exists)",
        len(big.ids()) > 12 * 10, f"{len(big.ids())} spans vs top_k=12")

    cf, th = dynamics_stats(big, max_probes=20)
    chk("[6] dynamics_stats survives session scale (probe subsampling)",
        len(th) == len(big.ids()) and len(cf) > 0,
        f"{len(cf)} co-fire pairs over {len(big.ids())} spans")

    print(f"\n  ALGO_GRR_SESSIONSCALE SELFTEST -> {'PASS' if ok else 'FAIL'}")
    return ok


def _run(dataset: str, n: int, source: str, lm: str, min_level: int, top_k: int,
         span_words: int, window: int = 512, chunk: int = 128,
         boost_by_strength: bool = False, drop_follows: float = 0.0) -> bool:
    pol, loaded = _load_policy()
    print(f"algo_grr_sessionscale --run: dataset={dataset} n={n} source={source} "
          f"min_level={min_level} top_k={top_k}")
    print(f"  policy: {'artifacts/editpolicy_multidomain.pt' if loaded else 'UNTRAINED (no ckpt)'}\n")
    problems = load_problems(dataset, n, min_level=min_level)
    print(f"  {len(problems)} problems with non-empty gold")
    g = (graph_from_eviction(problems, lm, window=window, chunk=chunk)
         if source == "eviction" else graph_from_words(problems, span_words=span_words))
    if drop_follows > 0:
        # FRAGMENT THE CHAIN. A real session graph is not a perfect line -- this project measured
        # 84% isolated nodes / 74 components on its own long-term graph before repair. With the
        # chain intact every span already has a follows neighbour boosted at full strength, and
        # because the boost is taken as a max, a strength-weighted learned edge (median confidence
        # 0.18) can never outrank it. Repair is the only regime where a learned edge can contribute.
        # SORTED, not set order. g.edges is a set of string tuples, and Python randomises string
        # hashing per process, so a seeded RNG walking the set drops a DIFFERENT subset in every
        # run -- measured: two arms that should have shared an identical fragmented graph got
        # bases of 0.447 and 0.467. Sorting makes the corruption reproducible across processes.
        import random as _r
        _rng = _r.Random(0)
        for _a, _b, _r2 in sorted(e for e in g.edges if e[2] == "follows"):
            if _rng.random() < drop_follows:
                g.unlink(_a, _b, _r2)
    print(f"  graph: {len(g.ids())} spans, {len(g.edges)} follows edges\n")
    r = recall_ab(g, problems, pol=pol, top_k=top_k,
                  boost_by_strength=boost_by_strength)
    if not r.get("n"):
        print(f"  {r.get('note')}")
        return False
    d = r["edited"] - r["base"]
    print(f"\n  base {r['base']:.3f} -> edited {r['edited']:.3f}  (delta {d:+.3f}, "
          f"{r['added']} edges, {r['spans']} spans, n={r['n']})")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Session A/B at scale, harder problems, real eviction.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "math"])
    ap.add_argument("--source", default="words", choices=["words", "eviction"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--min-level", type=int, default=0, dest="min_level",
                    help="MATH only: keep problems at or above this level (1-5)")
    ap.add_argument("--top-k", type=int, default=12, dest="top_k")
    ap.add_argument("--span-words", type=int, default=25, dest="span_words")
    ap.add_argument("--lm", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--window", type=int, default=512, help="KV cache cap (eviction)")
    ap.add_argument("--chunk", type=int, default=128, help="feed chunk (eviction)")
    ap.add_argument("--strength-boost", action="store_true", dest="sb",
                    help="scale neighbor_boost by edge strength (learned edges carry confidence)")
    ap.add_argument("--drop-follows", type=float, default=0.0, dest="df",
                    help="fragment the temporal chain: drop this fraction of follows edges")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    if a.run:
        sys.exit(0 if _run(a.dataset, a.n, a.source, a.lm, a.min_level, a.top_k,
                           a.span_words, a.window, a.chunk, a.sb, a.df) else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
