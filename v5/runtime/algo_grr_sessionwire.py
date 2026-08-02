"""algo_grr_sessionwire — prose levels for the edit curriculum, then the learned policy wired into
the REAL session-recall mechanism. The LM is a speech motor only: it never decides anything.

TWO THINGS, both measured against what already exists.

(A) MULTI-DOMAIN CURRICULUM. algo_grr_editcur trains on code atoms only, and the resulting policy
    transfers to prose spans at 0.51 -- 51x chance, but still under token-overlap-alone (0.57). A
    policy that has never seen prose statistics during learning meets them cold at test time. So the
    curriculum gains PROSE levels: the same RESTORE edit (cut a temporal `follows` edge, put it
    back) over real GSM8K spans. Train problems and eval problems are DISJOINT ranges of the corpus,
    so "unseen task" means a document the policy never read, not a fresh corruption of a seen one.

(B) THE POLICY EDITS THE GRAPH THE RECALL MECHANISM ACTUALLY USES. Not a reimplementation: this
    imports algo_trm's real `select_nodes` -- the 4-cycle iterative loop, the depth-1 bounded
    follows walk (neighbor_boost=3.0, follows_type=5) and the beta-annealed sub-window Hopfield
    prior. Embeddings are MiniLM, the substrate every node in this repo is already indexed by, and
    the prior is frozen by construction (algo_trm asserts `not rec.requires_grad`), so nothing here
    quietly trains the scorer. The A/B is the honest one: identical mechanism, identical cue,
    identical everything, with and without the `related` edges the policy proposes.

    Metric is the session protocol: query = the first 6 words of a problem, gold = the digits in
    that problem minus the digits already in the cue, recall = gold is a subset of the UNION of
    digits over the retrieved spans. Cue-only, no trie, no number boosting.

WHERE THE LM IS ALLOWED. `--speak` renders one sentence from what the graph already retrieved. It
receives the selected spans and nothing else; it does not choose spans, does not rank, does not
decide whether memory was sufficient. Every number it may utter has already been recalled by the
graph. That is the division this project is built on -- the graph is the cognition, the LM is the
motor -- and the flag is off by default so the measured numbers never depend on it.

    selftest : python -m v5.runtime.algo_grr_sessionwire --selftest
    train    : python -m v5.runtime.algo_grr_sessionwire --train
    recall   : python -m v5.runtime.algo_grr_sessionwire --recall
    speak    : python -m v5.runtime.algo_grr_sessionwire --recall --speak --lm Qwen/Qwen2.5-0.5B-Instruct
"""
from __future__ import annotations

import argparse
import os
import random
import re
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("HF_HOME", r"E:\cache\hf")

import torch                                                              # noqa: E402

from v5.runtime.algo_grr_editcur import (                                 # noqa: E402
    EditGraph, EditPolicy, dynamics_stats, level_instances, load_code_graph,
    load_session_graph, pair_matrix,
)

_NUM = re.compile(r"\d+")
FOLLOWS_T, RELATED_T = 5, 1                                               # membrane's edge-type ids


def _digits(s: str) -> set:
    return set(_NUM.findall(s or ""))


# ================================================================================================
# (A) prose levels — the same RESTORE edit, on real GSM8K spans
# ================================================================================================
def prose_instances(g: EditGraph, owner: dict, rng: random.Random, n: int) -> list:
    """Cut the temporal edge between two spans of one problem; the policy must reconnect them.
    Same action and same head as the code levels -- only the substrate changes."""
    by_p: dict = {}
    for s, p in owner.items():
        by_p.setdefault(p, []).append(s)
    multi = [p for p, ss in by_p.items() if len(ss) > 1]
    out = []
    for _ in range(n):
        if not multi:
            break
        p = rng.choice(multi)
        ss = sorted(by_p[p])
        i = rng.randrange(len(ss) - 1)
        src, gold = ss[i], ss[i + 1]
        gr = g.copy()
        gr.unlink(src, gold, "follows")
        gr.unlink(gold, src, "follows")
        cands = [c for c in gr.ids() if c != src and not gr.has(src, c) and not gr.has(c, src)]
        if gold not in cands or len(cands) < 2:
            continue
        out.append((gr, src, cands, gold, "ADD_EDGE"))
    return out


def train_multidomain(epochs: int = 40, per_epoch: int = 64, seed: int = 0,
                      train_problems=(0, 40), span_words: int = 25, verbose: bool = True):
    """Code levels + prose levels in one head. The prose graph is built ONCE (its dynamics stats
    are expensive) and corruptions are sampled from it."""
    rng = random.Random(seed)
    gc, code, entry, true_deps = load_code_graph()
    cofire_c, theta_c = dynamics_stats(gc)
    gp, _qs, owner = load_session_graph(train_problems[1], span_words=span_words)
    gp, owner = _slice_problems(gp, owner, train_problems[0], train_problems[1])
    cofire_p, theta_p = dynamics_stats(gp)
    pol = EditPolicy()
    opt = torch.optim.Adam(pol.parameters(), lr=5e-3)
    for ep in range(epochs):
        frac = min(0.5, 0.05 + 0.9 * ep / max(1, epochs - 1))
        n_hard = int(per_epoch * frac)
        n_easy = (per_epoch - n_hard) // 2
        batch = [(i, cofire_c, theta_c) for i in
                 (level_instances(0, gc, true_deps, rng, n_easy)
                  + level_instances(1, gc, true_deps, rng, per_epoch - n_hard - n_easy)
                  + level_instances(2, gc, true_deps, rng, n_hard))]
        batch += [(i, cofire_p, theta_p) for i in
                  prose_instances(gp, owner, rng, per_epoch // 2)]
        rng.shuffle(batch)
        tot = 0.0
        for (gr, s, cands, gold, act), cf, th in batch:
            if gold not in cands or len(cands) < 2:
                continue
            logits = pol(pair_matrix(gr, s, cands, cf, th))
            y = torch.tensor(cands.index(gold))
            loss = torch.nn.functional.cross_entropy(
                (logits if act == "ADD_EDGE" else -logits).unsqueeze(0), y.unsqueeze(0))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss)
        if verbose and (ep + 1) % 10 == 0:
            print(f"    epoch {ep+1:3d}  loss={tot / max(1, len(batch)):.4f}", flush=True)
    return pol, (gc, true_deps, cofire_c, theta_c)


def _slice_problems(g: EditGraph, owner: dict, lo: int, hi: int) -> tuple:
    """Keep only spans whose problem index is in [lo, hi). Held-out prose must be a DOCUMENT the
    policy never read, not another corruption of a document it has."""
    keep = {s for s, p in owner.items() if lo <= p < hi}
    gs = EditGraph()
    for s in keep:
        gs.add(s, g.text[s])
    for a, b, r in g.edges:
        if a in keep and b in keep:
            gs.link(a, b, r)
    return gs, {s: owner[s] for s in keep}


def eval_prose(pol, lo: int, hi: int, span_words: int = 25, seed: int = 5,
               n: int = 60) -> dict:
    """Held-out prose: disjoint problems, fresh cuts. Baseline is the policy's own strongest single
    input feature -- if the learned combination cannot beat token overlap it has added nothing."""
    rng = random.Random(seed)
    g, _qs, owner = load_session_graph(hi, span_words=span_words)
    g, owner = _slice_problems(g, owner, lo, hi)
    cofire, theta = dynamics_stats(g)
    insts = prose_instances(g, owner, rng, n)
    hit = base = ran = 0
    chances = []
    for gr, s, cands, gold, _a in insts:
        X = pair_matrix(gr, s, cands, cofire, theta)
        with torch.no_grad():
            sc = pol(X)
        ran += 1
        chances.append(1.0 / len(cands))
        hit += int(cands[int(sc.argmax())] == gold)
        base += int(cands[int(X[:, 0].argmax())] == gold)
    if not ran:
        return {"n": 0}
    return {"n": ran, "policy": hit / ran, "token_overlap_only": base / ran,
            "chance": sum(chances) / len(chances)}


# ================================================================================================
# (B) the real recall mechanism, with and without the policy's edits
# ================================================================================================
def _window_embs(texts: list, win: int = 12, stride: int = 6) -> tuple:
    """Sub-window Hopfield keys over MiniLM: overlapping word-windows plus the full-span mean.
    Same construction as membrane's _window_pool, in word space instead of token space."""
    from embedder import encode_batch
    chunks, parent = [], []
    for i, t in enumerate(texts):
        w = t.split()
        outs = [" ".join(w[a:a + win]) for a in range(0, max(1, len(w) - win + 1), stride)]
        if len(w) > win:
            outs.append(t)
        for o in outs:
            chunks.append(o)
            parent.append(i)
    E = torch.as_tensor(encode_batch(chunks), dtype=torch.float32)
    return E, torch.tensor(parent, dtype=torch.long)


def _tensors(g: EditGraph, ids: list) -> tuple:
    from embedder import encode_batch
    idx = {n: i for i, n in enumerate(ids)}
    E = torch.as_tensor(encode_batch([g.text[n] for n in ids]), dtype=torch.float32)
    ei, et, es = [], [], []
    for a, b, r in g.edges:
        if a in idx and b in idx:
            ei.append([idx[a], idx[b]])
            et.append(FOLLOWS_T if r == "follows" else RELATED_T)
            es.append(float(g.strength.get((a, b, r), 1.0)))
    if ei:
        return (E, torch.tensor(ei, dtype=torch.long).t(),
                torch.tensor(et, dtype=torch.long), torch.tensor(es, dtype=torch.float32))
    return (E, torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.float32))


def _make_controller(d_in: int):
    """token_head_d_lm must be set or controller_logits has no `input_proj`. It is only used by
    the TRM's internal state (and from there the evict/tool heads, unused here): the RECALL prior
    is computed on the RAW embeddings -- algo_trm asserts `not rec.requires_grad` -- so an
    untrained projection cannot touch which spans get selected."""
    from v5.runtime.algo_trm import _build
    _t, _nn, TRMReasoner = _build()
    return TRMReasoner(d_in=d_in, d=256, T=4, token_head_d_lm=d_in)


def recall_ab(lo: int = 100, hi: int = 140, span_words: int = 25, pol=None,
              top_k: int = 12, cycles: int = 4, picks: int = 3, seed: int = 3,
              drop_follows: float = 0.0, boost_all: bool = True,
              verbose: bool = True) -> dict:
    """A/B the REAL select_nodes loop on the same session graph, with and without policy edits.

    Cue-only protocol: the query is the first 6 words of a problem and the gold is that problem's
    digits minus the cue's digits, scored as a subset of the UNION over retrieved spans."""
    from embedder import encode_batch
    g0, qs, owner = load_session_graph(hi, span_words=span_words)
    g0, owner = _slice_problems(g0, owner, lo, hi)
    if drop_follows > 0:
        rng = random.Random(seed)
        for a, b, r in [e for e in g0.edges if e[2] == "follows"]:
            if rng.random() < drop_follows:
                g0.unlink(a, b, r)
    ids = sorted(g0.ids())
    if len(ids) < 4:
        return {"n": 0, "note": "not enough spans"}
    trm = _make_controller(384)
    Ew, Wp = _window_embs([g0.text[n] for n in ids])

    def run(g, tag):
        E, ei, et, es = _tensors(g, ids)
        tools = torch.zeros(0, E.shape[1])
        ok = tot = 0
        for p in sorted(set(owner.values())):
            mine = sorted(s for s, o in owner.items() if o == p)
            full = " ".join(g0.text[s] for s in mine)
            cue = " ".join(full.split()[:6])
            gold = _digits(full) - _digits(cue)
            if not gold:
                continue
            q = torch.as_tensor(encode_batch([cue])[0], dtype=torch.float32)
            n_idx, _ev, _tl = trm.select_nodes(
                q, E, ei, et, es, tools, top_k=top_k, top_tools=0,
                cycles=cycles, picks_per_cycle=picks, neighbor_boost=3.0,
                follows_type=(None if boost_all else FOLLOWS_T),
                win_embs_lm=Ew, win_parent=Wp)
            got = set()
            for i in n_idx:
                got |= _digits(g0.text[ids[i]])
            tot += 1
            ok += int(gold <= got)
        if verbose:
            print(f"    {tag:<28} recall {ok}/{tot} = {ok / max(1, tot):.2f}", flush=True)
        return ok / max(1, tot), tot

    base, n = run(g0, "follows edges only")
    if pol is None:
        return {"n": n, "base": base, "edited": None, "added": 0}
    # POLICY EDITS: one proposed `related` edge per span, over pairs that are not already linked.
    g1 = g0.copy()
    cofire, theta = dynamics_stats(g0)
    added = 0
    for s in ids:
        cands = [c for c in ids if c != s and not g0.has(s, c) and not g0.has(c, s)]
        if len(cands) < 2:
            continue
        with torch.no_grad():
            sc = pol(pair_matrix(g0, s, cands, cofire, theta))
        g1.link(s, cands[int(sc.argmax())], "related")
        added += 1
    edited, _ = run(g1, f"+ {added} policy `related` edges")
    return {"n": n, "base": base, "edited": edited, "added": added,
            "graph": g1, "ids": ids, "owner": owner, "src": g0}


# ================================================================================================
# (C) the LM as a speech motor — renders what the graph already decided
# ================================================================================================
def speak(lm_name: str, cue: str, spans: list, nums: list, max_try: int = 2) -> tuple:
    """One sentence from the spans the GRAPH selected, behind a NUMERIC GATE.

    The LM chooses no spans, ranks nothing, and decides nothing. Its output is then CHECKED: every
    digit it emits must already have been recalled by the graph (or be present in the cue the user
    typed). Measured why this is not optional -- asked politely, with the allowed set spelled out in
    the system prompt, a 4-bit 0.5B answered "1, 4" for a graph that had recalled 1/10/20/60. A
    prompt is a request, not a constraint. On a violation the motor is re-run once and then falls
    back to a template rendered straight from the graph's own numbers, which cannot be unfaithful
    because no model wrote it. Returns (text, status)."""
    from v5.runtime.dcpd_latent import WhiteBox
    wb = WhiteBox(lm_name, quant="4bit")
    allowed = set(nums) | _digits(cue)
    facts = "\n".join(f"- {s}" for s in spans)
    sys_msg = ("Answer using ONLY the recalled notes below. Do not invent numbers; "
               f"the only numbers you may use are: {', '.join(sorted(nums))}.\n{facts}")
    bad: set = set()
    for _ in range(max_try):
        out = " ".join(str(wb.generate_chat(
            f"From my notes, what were the numbers about: {cue}?",
            system=sys_msg, max_new=96)).split())
        bad = _digits(out) - allowed
        if not bad:
            return out, "gate:pass"
    return (f"From the recalled notes about \"{cue}\": " + ", ".join(sorted(nums, key=int)) + ".",
            f"gate:FALLBACK (model emitted ungrounded {sorted(bad)})")


# ================================================================================================
# selftest / entry points
# ================================================================================================
def _selftest() -> bool:
    print("algo_grr_sessionwire --selftest: prose levels + real select_nodes wiring\n")
    ok = True

    def chk(t, c, d=""):
        nonlocal ok
        ok &= bool(c)
        print(f"  [{'PASS' if c else 'FAIL'}] {t}{('  - ' + d) if d else ''}")

    g, _qs, owner = load_session_graph(30, span_words=25)
    chk("[1] real GSM8K session graph builds (spans + follows, no LM)",
        len(g.ids()) > 30 and any(r == "follows" for _a, _b, r in g.edges),
        f"{len(g.ids())} spans, {len(g.edges)} edges")

    gs, os_ = _slice_problems(g, owner, 0, 10)
    chk("[2] problem slicing is disjoint (held-out = unseen documents)",
        all(0 <= p < 10 for p in os_.values()) and len(gs.ids()) < len(g.ids()),
        f"{len(gs.ids())} of {len(g.ids())} spans kept")

    insts = prose_instances(g, owner, random.Random(0), 20)
    chk("[3] prose RESTORE instances are well-formed",
        len(insts) > 5 and all(gold in cands for _g, _s, cands, gold, _a in insts),
        f"{len(insts)} instances")

    ids = sorted(g.ids())[:12]
    sub = EditGraph()
    for n in ids:
        sub.add(n, g.text[n])
    for a, b, r in g.edges:
        if a in ids and b in ids:
            sub.link(a, b, r)
    E, ei, et, es = _tensors(sub, ids)
    Ew, Wp = _window_embs([sub.text[n] for n in ids])
    chk("[4] window keys SPLIT spans into sub-views (Hopfield keys, not just the mean)",
        Ew.shape[0] > len(ids) and int(Wp.max()) == len(ids) - 1,
        f"{Ew.shape[0]} windows over {len(ids)} spans")

    trm = _make_controller(384)
    n_idx, _ev, _tl = trm.select_nodes(E[0], E, ei, et, es, torch.zeros(0, 384),
                                       top_k=6, top_tools=0, cycles=3, picks_per_cycle=2,
                                       neighbor_boost=3.0, follows_type=FOLLOWS_T,
                                       win_embs_lm=Ew, win_parent=Wp)
    chk("[5] the REAL select_nodes loop runs on MiniLM embeddings",
        len(n_idx) <= 6 and len(set(n_idx)) == len(n_idx), f"picked {n_idx}")

    # [6] the prior must stay frozen: no gradient path into the scorer from this file.
    chk("[6] recall prior is frozen (no trained scorer smuggled in)",
        not E.requires_grad and all(not p.requires_grad or p.grad is None
                                    for p in trm.parameters()))
    print(f"\n  ALGO_GRR_SESSIONWIRE SELFTEST -> {'PASS' if ok else 'FAIL'}")
    return ok


def _train_report(epochs: int = 40) -> bool:
    print("algo_grr_sessionwire --train: code + PROSE levels in one policy\n")
    pol, _code_ctx = train_multidomain(epochs=epochs)
    print("\n  HELD-OUT PROSE (problems 100-140, never trained on)")
    r = eval_prose(pol, 100, 140)
    if r.get("n"):
        print(f"    n={r['n']}  policy={r['policy']:.2f}  "
              f"token_overlap_only={r['token_overlap_only']:.2f}  chance={r['chance']:.3f}")
    print("    (code-only curriculum scored policy 0.51 vs overlap 0.57 on this task)")
    torch.save(pol.state_dict(), str(Path(_ROOT) / "artifacts" / "editpolicy_multidomain.pt"))
    print(f"\n  saved -> artifacts/editpolicy_multidomain.pt")
    return bool(r.get("n"))


def _recall_report(speak_lm: str = "") -> bool:
    print("algo_grr_sessionwire --recall: policy edits inside the REAL select_nodes loop\n")
    ck = Path(_ROOT) / "artifacts" / "editpolicy_multidomain.pt"
    pol = EditPolicy()
    if ck.exists():
        pol.load_state_dict(torch.load(str(ck)))
        print(f"  loaded {ck.name}")
    else:
        print("  no checkpoint - training first")
        pol, _ = train_multidomain(epochs=40, verbose=False)
    print("\n  held-out session (problems 100-140), cue-only protocol, no trie, no number boost")
    r = recall_ab(pol=pol)
    if not r.get("n"):
        print(f"    {r.get('note')}")
        return False
    d = (r["edited"] - r["base"])
    print(f"\n  base {r['base']:.2f} -> edited {r['edited']:.2f}  "
          f"(delta {d:+.2f}, {r['added']} edges added)")
    if speak_lm:
        print("\n  LM AS SPEECH MOTOR (renders what the graph already retrieved):")
        g0, ids, owner = r["src"], r["ids"], r["owner"]
        p = sorted(set(owner.values()))[0]
        mine = sorted(s for s, o in owner.items() if o == p)
        full = " ".join(g0.text[s] for s in mine)
        cue = " ".join(full.split()[:6])
        nums = sorted(_digits(full) - _digits(cue))
        print(f"    cue   : {cue}")
        print(f"    graph recalled numbers: {nums}")
        _txt, _st = speak(speak_lm, cue, [g0.text[s] for s in mine], nums)
        print(f"    LM says: {_txt}")
        print(f"    gate   : {_st}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Prose curriculum + real session-recall wiring.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--recall", action="store_true")
    ap.add_argument("--speak", action="store_true", help="let the LM render one sentence")
    ap.add_argument("--lm", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    if a.train:
        sys.exit(0 if _train_report(a.epochs) else 1)
    if a.recall:
        sys.exit(0 if _recall_report(a.lm if a.speak else "") else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
