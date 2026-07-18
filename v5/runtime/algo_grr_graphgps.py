"""algo_grr_graphgps — structure-aware routing at scale (#3): the real fix for routing degradation.

Earlier `--scale` showed the flat router collapsing 0.98->0.22, BUT that test was CONFOUNDED (atoms on a
fixed circle -> more atoms = denser packing = harder discrimination, not pure scale — cosine also dropped).
This module (a) de-confounds the test (separated clusters that DON'T densify as N grows) and (b) builds the
GraphGPS router = content + LapPE/RWSE (global position) + one-hop MESSAGE-PASSING (the specific-edge
signal positional encodings miss). Shows: content-only is STUCK (structurally blind), GraphGPS HOLDS at
scale, both stable with N (no confounded degradation).

    python -m v5.runtime.algo_grr_graphgps --selftest   # no-GPU: content-only vs GraphGPS Recall vs N
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from v5.runtime.algo_grr_struct import struct_features  # noqa: E402


def gen_deconfounded(N, d=24, seed=0):
    """N atoms in SEPARATED clusters (added, not densified, as N grows). Each atom has a DEPEND partner in
    a DIFFERENT cluster (content-dissimilar structural edge). Task q needs {q, dep[q]}: content finds q,
    only STRUCTURE finds dep[q]. Cluster separation is preserved as N grows -> no confound."""
    rng = np.random.default_rng(seed)
    C = max(3, N // 25)
    centers = rng.standard_normal((C, d)); centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-9
    comm = rng.integers(0, C, N)
    content = centers[comm] + 0.12 * rng.standard_normal((N, d))
    content /= np.linalg.norm(content, axis=1, keepdims=True) + 1e-9
    dep = np.array([int(rng.choice(np.where(comm != comm[i])[0])) for i in range(N)])
    A = np.zeros((N, N), dtype="float32")
    for i in range(N):
        A[i, dep[i]] = A[dep[i], i] = 1.0
    return content.astype("float32"), A, dep


def gps_features(content: np.ndarray, A: np.ndarray) -> np.ndarray:
    """GraphGPS atom features: [content ; LapPE+RWSE ; one-hop message-passed content]. The message-passing
    term (row-normalised A @ content) carries each atom's NEIGHBOUR content -> the specific-edge signal."""
    S = struct_features(A, k_lap=6, k_rw=6)
    d = A.sum(1, keepdims=True)
    P = A / np.maximum(d, 1e-9)
    agg = (P @ content).astype("float32")          # neighbour content (the dep partner's content)
    F = np.concatenate([content, S, agg], axis=1)
    return ((F - F.mean(0)) / (F.std(0) + 1e-9)).astype("float32")


def _split_tasks(N, dep, rng):
    nodes = rng.permutation(N)
    tr, te = nodes[:int(0.6 * N)], nodes[int(0.6 * N):]
    mk = lambda pool: [(int(q), {int(q), int(dep[q])}) for q in pool]  # noqa: E731
    return mk(tr), mk(te)


def _eval(feat, train, test, epochs=250):
    """Return (Recall@2, Recall@10). The pipeline searches+verifies over a candidate SET, so top-K (K~10)
    is the operative metric — the router must keep the needed atoms in a small candidate set, not rank #1-2."""
    from v5.runtime.algo_grr_router import _build, train_router, _recall_at_k
    import torch
    torch, nn, NeuralRouter = _build()
    A = torch.as_tensor(feat)
    router = NeuralRouter(feat.shape[1])
    train_router(router, [(feat[q], nd) for q, nd in train], feat, epochs=epochs, lr=3e-3)
    with torch.no_grad():
        sc = [router(torch.as_tensor(feat[q]), A).tolist() for q, _ in test]
    r2 = float(np.mean([_recall_at_k(s, nd, 2) for s, (_, nd) in zip(sc, test)]))
    r10 = float(np.mean([_recall_at_k(s, nd, 10) for s, (_, nd) in zip(sc, test)]))
    return r2, r10


def _selftest(sizes=(60, 150, 300, 500)) -> bool:
    print("algo_grr_graphgps --selftest: routing at scale, de-confounded (content-only vs GraphGPS)\n")
    print(f"  {'#atoms':>7} | {'content@2':>9} {'content@10':>10} | {'GPS@2':>6} {'GPS@10':>7}   "
          f"(operative metric = @10: needed atoms in the candidate SET)")
    ok = True
    for N in sizes:
        content, A, dep = gen_deconfounded(N, seed=0)
        train, test = _split_tasks(N, dep, np.random.default_rng(1))
        c2, c10 = _eval(content, train, test)
        g2, g10 = _eval(gps_features(content, A), train, test)
        held = g10 > 0.85 and g10 - c10 > 0.2
        ok &= held
        print(f"  {N:>7} | {c2:>9.2f} {c10:>10.2f} | {g2:>6.2f} {g10:>7.2f}   {'holds' if held else 'weak'}",
              flush=True)
    print(f"\n  -> {'PASS' if ok else 'FAIL'}: at the operative metric (@10, the candidate set the planner")
    print(f"     searches+verifies over) GraphGPS keeps the cross-cluster dep atom in-set at scale where")
    print(f"     content-only is blind. (@2-of-N is a needle problem for any features; not the router's job.)")
    print(f"\n  ALGO_GRR_GRAPHGPS SELFTEST -> {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
