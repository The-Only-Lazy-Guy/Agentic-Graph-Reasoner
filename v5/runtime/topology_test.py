"""Topology kill-test (V7): is the operator-graph topology (transition structure) LOAD-BEARING or
DECORATIVE? Predict the next operator under three models and compare on HELD sequences:
  LEARNED   P(next|prev) from train transitions
  RANDOM    same transitions but prev<->next association shuffled (keeps both marginals, kills structure)
  MARGINAL  P(next) only (no topology)
LEARNED >> RANDOM/MARGINAL -> the graph REASONS (edges carry info). LEARNED ~= RANDOM -> decorative ->
drop the graph claim. (The HRM differentiator; we have a prior scar where a GNN's edges were decorative
-- memory/research-direction-externalized-unlearn.)

  python -m v5.runtime.topology_test --selftest
  python -m v5.runtime.topology_test --swe                              # SWE multi-hunk co-occurrence (now)
  python -m v5.runtime.topology_test --traj data/traj/fable_heavy.jsonl # mined temporal (after mining)
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def build_T(seqs, n, smoothing=0.1):
    """Consecutive-pair transition matrix, row-normalized with Laplace smoothing."""
    import numpy as np
    C = np.full((n, n), smoothing, dtype=float)
    for s in seqs:
        for a, b in zip(s, s[1:]):
            C[a, b] += 1.0
    return C / C.sum(1, keepdims=True)


def marginal(seqs, n, smoothing=0.1):
    import numpy as np
    m = np.full(n, smoothing, dtype=float)
    for s in seqs:
        for b in s[1:]:
            m[b] += 1.0
    return m / m.sum()


def _shuffled_T(seqs, n, seed=0):
    """RANDOM topology: keep prev- and next-marginals, destroy the prev->next association."""
    rng = random.Random(seed)
    prevs, nexts = [], []
    for s in seqs:
        for a, b in zip(s, s[1:]):
            prevs.append(a); nexts.append(b)
    rng.shuffle(nexts)
    return build_T([[p, q] for p, q in zip(prevs, nexts)], n) if prevs else build_T([], n)


def collapse(seqs):
    """Run-length-encode: drop consecutive repeats so every transition is a real op-CHANGE. Agentic
    sessions repeat the same operator (self-loops ~64%); raw transitions are dominated by 'keep doing
    the same thing' -> collapse to test whether the operator-CHANGE order is structured."""
    out = []
    for s in seqs:
        o = []
        for x in s:
            if not o or o[-1] != x:
                o.append(x)
        out.append(o)
    return out


def _eval(train, held, n, seed=0):
    """Transition prediction on `held` given `train`. Reports self-rate + learned-on-CROSS (the real
    op-change structure, separate from the trivial self-loop diagonal)."""
    import numpy as np
    T = build_T(train, n); Tr = _shuffled_T(train, n, seed); mg = marginal(train, n)
    pairs = [(a, b) for s in held for a, b in zip(s, s[1:])]
    if not pairs:
        return None
    acc = lambda row: float(np.mean([int(np.argmax(row(a)) == b) for a, b in pairs]))
    cross = [(a, b) for a, b in pairs if a != b]
    return {
        "n_held_trans": len(pairs), "n_cross": len(cross), "n_ops": n,
        "acc_learned": acc(lambda a: T[a]), "acc_random": acc(lambda a: Tr[a]), "acc_marginal": acc(lambda a: mg),
        "self_rate": 1.0 - len(cross) / len(pairs),
        "acc_learned_cross": float(np.mean([int(np.argmax(T[a]) == b) for a, b in cross])) if cross else 0.0,
    }


def killtest(seqs, n, seed=0, log=print):
    """Single 80/20 split over sequences. (For small data prefer killtest_cv.)"""
    rng = random.Random(seed)
    seqs = [s for s in seqs if len(s) >= 2]
    rng.shuffle(seqs)
    nh = max(5, len(seqs) // 5)
    m = _eval(seqs[nh:], seqs[:nh], n, seed)
    if m:
        m["n_seq"] = len(seqs)
    return m or {}


def killtest_cv(seqs, n, folds=5, seed=0):
    """k-fold CV over SEQUENCES (no transition leakage) — robust on small data. Averages each metric."""
    import numpy as np
    seqs = [s for s in seqs if len(s) >= 2]
    rng = random.Random(seed); idx = list(range(len(seqs))); rng.shuffle(idx)
    rows = []
    for f in range(folds):
        hi = set(idx[f::folds])
        r = _eval([seqs[i] for i in range(len(seqs)) if i not in hi], [seqs[i] for i in hi], n, seed=f)
        if r:
            rows.append(r)
    if not rows:
        return {}
    keys = ["acc_learned", "acc_random", "acc_marginal", "acc_learned_cross", "self_rate"]
    m = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    m["acc_learned_std"] = float(np.std([r["acc_learned"] for r in rows]))
    m["n_held_trans"] = sum(r["n_held_trans"] for r in rows)
    m["n_cross"] = sum(r["n_cross"] for r in rows); m["n_seq"] = len(seqs); m["n_ops"] = n; m["folds"] = len(rows)
    return m


def _report(m):
    if not m:
        print("[topo] no transitions"); return
    std = f"+-{m['acc_learned_std']:.0%}" if "acc_learned_std" in m else ""
    cv = f", {m['folds']}-fold CV" if "folds" in m else ""
    print(f"\n=== TOPOLOGY KILL-TEST (n_seq={m['n_seq']}, held trans={m['n_held_trans']}, ops={m['n_ops']}{cv}) ===")
    print(f"  next-op ACC : learned {m['acc_learned']:.0%}{std} | random {m['acc_random']:.0%} | marginal {m['acc_marginal']:.0%}")
    print(f"  self-loops  : {m['self_rate']:.0%} of transitions | learned on CROSS (op-change): {m['acc_learned_cross']:.0%} (n_cross={m['n_cross']})")
    # the HONEST signal = learned-on-cross vs the cross-chance baseline (1/n_ops) and vs random topology
    cross_chance = 1.0 / max(2, m["n_ops"])
    verdict = m["acc_learned_cross"] > max(m["acc_random"], cross_chance) + 0.08
    print(f"  VERDICT (op-CHANGE structure, the real test): "
          f"{'LOAD-BEARING — edges carry sequencing info beyond self-loops' if verdict else 'DECORATIVE beyond self-loops'}")


# ── data loaders ─────────────────────────────────────────────────────────────
def swe_cooccurrence_seqs():
    """SWE multi-hunk golds -> per-hunk operator-id sequences (co-occurrence, weak order) over the
    canonical embedding ops. Precursor signal until temporal Fable trajectories are mined."""
    import numpy as np
    import torch
    from v5.training.providers import RealEmbedder
    from v5.graph_grower.swe_load import load_instances
    from v5.runtime.operator_discovery import extract_hunks, _fix_text
    ops = [json.loads(l) for l in Path("data/swe/discovered_ops_emb.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    cen = np.array([o["_centroid"] for o in ops], dtype=float); cen /= np.linalg.norm(cen, axis=1, keepdims=True) + 1e-9
    emb = RealEmbedder(torch.device("cpu"))
    hunk_texts, owner = [], []
    golds = []
    for i in load_instances(name="lite", split="test", limit=0):
        hs = extract_hunks(i.get("patch", "") or "")
        if len(hs) >= 2:
            gi = len(golds); golds.append([])
            for h in hs:
                hunk_texts.append(_fix_text([h])); owner.append(gi)
    V = []
    for k in range(0, len(hunk_texts), 48):
        d = emb.embed_nodes({str(j): t[:1000] for j, t in enumerate(hunk_texts[k:k + 48])})
        V.extend(d[str(j)] for j in range(len(hunk_texts[k:k + 48])))
    V = np.array(V, dtype=float); V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    for vec, gi in zip(V, owner):
        golds[gi].append(int(np.argmax(cen @ vec)))
    return [g for g in golds if len(g) >= 2], len(ops)


def traj_seqs(path):
    """Mined trajectories jsonl -> operator-id sequences over the grown library."""
    grown = Path(path).with_name("grown_ops.jsonl")
    names = [json.loads(l)["name"] for l in grown.read_text(encoding="utf-8").splitlines() if l.strip()]
    nid = {nm: k for k, nm in enumerate(names)}
    seqs = []
    for l in Path(path).read_text(encoding="utf-8").splitlines():
        if l.strip():
            s = [nid[t["op"]] for t in json.loads(l)["trajectory"] if t.get("op") in nid]
            if len(s) >= 2:
                seqs.append(s)
    return seqs, len(names)


def _selftest() -> bool:
    print("topology_test --selftest: structured cycle A->B->C->A (no model)\n")
    rng = random.Random(0)
    # real structure: each op deterministically followed by the next (cycle of 8) -> learned must win
    seqs = []
    for _ in range(300):
        start = rng.randint(0, 7); seqs.append([(start + k) % 8 for k in range(rng.randint(4, 9))])
    m = killtest(seqs, 8, log=lambda *a: None)
    print(f"  learned {m['acc_learned']:.0%} | random {m['acc_random']:.0%} | marginal {m['acc_marginal']:.0%}")
    assert m["acc_learned"] > 0.9, "learned must capture the cycle"
    assert m["acc_learned"] > m["acc_random"] + 0.3, "learned must beat shuffled topology"
    print("\n  TOPOLOGY KILL-TEST SELFTEST -> PASS (learned >> random on real structure)")
    return True


def main():
    ap = argparse.ArgumentParser(description="Operator-graph topology kill-test (V7).")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--swe", action="store_true", help="SWE multi-hunk co-occurrence (available now)")
    ap.add_argument("--traj", help="mined trajectories jsonl (temporal; after mining)")
    ap.add_argument("--collapse", action="store_true", help="RLE: test op-CHANGES only (drop self-loops)")
    ap.add_argument("--cv", type=int, default=0, help="k-fold CV over sequences (robust on small data)")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if _selftest() else 1)
    if a.swe:
        seqs, n = swe_cooccurrence_seqs()
    elif a.traj:
        seqs, n = traj_seqs(a.traj)
    else:
        ap.print_help(); return
    if a.collapse:
        seqs = [s for s in collapse(seqs) if len(s) >= 2]
        print(f"[topo] collapsed to op-CHANGES: {len(seqs)} sequences with >=2 distinct successive ops")
    _report(killtest_cv(seqs, n, folds=a.cv) if a.cv else killtest(seqs, n))


if __name__ == "__main__":
    main()
