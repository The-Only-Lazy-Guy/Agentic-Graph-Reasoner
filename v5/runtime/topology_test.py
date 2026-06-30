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


def killtest(seqs, n, seed=0, log=print):
    """seqs: list of operator-id sequences (len>=2). Returns metrics dict."""
    import numpy as np
    rng = random.Random(seed)
    seqs = [s for s in seqs if len(s) >= 2]
    rng.shuffle(seqs)
    nh = max(5, len(seqs) // 5)
    held, train = seqs[:nh], seqs[nh:]
    T = build_T(train, n); Tr = _shuffled_T(train, n, seed); mg = marginal(train, n)
    pairs = [(a, b) for s in held for a, b in zip(s, s[1:])]
    if not pairs:
        log("[topo] no held transitions"); return {}
    def acc(pred_row): return np.mean([int(np.argmax(pred_row(a)) == b) for a, b in pairs])
    def ppl(pred_row): return float(np.exp(-np.mean([np.log(pred_row(a)[b] + 1e-12) for a, b in pairs])))
    m = {
        "n_seq": len(seqs), "n_held_trans": len(pairs), "n_ops": n,
        "acc_learned": acc(lambda a: T[a]), "acc_random": acc(lambda a: Tr[a]), "acc_marginal": acc(lambda a: mg),
        "ppl_learned": ppl(lambda a: T[a]), "ppl_random": ppl(lambda a: Tr[a]), "ppl_marginal": ppl(lambda a: mg),
    }
    return m


def _report(m):
    if not m:
        return
    print(f"\n=== TOPOLOGY KILL-TEST (n_seq={m['n_seq']}, held transitions={m['n_held_trans']}, ops={m['n_ops']}) ===")
    print(f"  next-op ACC : learned {m['acc_learned']:.0%} | random {m['acc_random']:.0%} | marginal {m['acc_marginal']:.0%}")
    print(f"  next-op PPL : learned {m['ppl_learned']:.2f} | random {m['ppl_random']:.2f} | marginal {m['ppl_marginal']:.2f}  (lower=better)")
    gap = m["acc_learned"] - max(m["acc_random"], m["acc_marginal"])
    print(f"  gap (learned - best baseline) = {gap:+.0%}")
    print(f"  VERDICT: {'TOPOLOGY LOAD-BEARING (edges carry info -> the graph reasons)' if gap > 0.08 else 'topology DECORATIVE (learned ~= random/marginal -> drop the graph claim)'}")


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
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if _selftest() else 1)
    if a.swe:
        seqs, n = swe_cooccurrence_seqs()
        _report(killtest(seqs, n))
    elif a.traj:
        seqs, n = traj_seqs(a.traj)
        _report(killtest(seqs, n))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
