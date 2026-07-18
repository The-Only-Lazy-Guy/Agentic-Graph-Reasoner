"""algo_grr_struct — STRUCTURE as a representation dimension (the encoder-question answer, made real).

Plain message-passing GNNs (our RGCN) are 1-WL-limited: they look edge-by-edge and are blind to motifs /
global position. To route by SUBCOMPONENT (not per-atom) we add STRUCTURAL features computed from the graph
itself — the GraphGPS recipe:
  - Laplacian positional encoding (LapPE): low eigenvectors of the normalized Laplacian = "graph Fourier
    modes"; adjacent nodes get SIMILAR values -> encodes global position + community structure.
  - Random-walk structural encoding (RWSE): return probabilities at steps 1..k = local loop/structure.

Demonstration: a graph where each atom has a DEPEND-child whose CONTENT is unrelated (random). A content-only
router can find the queried atom but is BLIND to its depend-child. With structural features appended, the
router finds the child too — structure supplies the dimension content can't. This is the principled version
of the topology retriever + the fix for routing-at-scale (route by structural neighbourhood).

    python -m v5.runtime.algo_grr_struct --selftest   # no-GPU: content-only vs +structure routing
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def lap_pe(A: np.ndarray, k: int = 8) -> np.ndarray:
    """Laplacian positional encoding: k lowest non-trivial eigenvectors of the normalized Laplacian."""
    d = A.sum(1)
    dinv = 1.0 / np.sqrt(np.maximum(d, 1e-9))
    L = np.eye(len(A)) - (dinv[:, None] * A * dinv[None, :])
    w, v = np.linalg.eigh(L)                       # ascending eigenvalues
    pe = v[:, 1:k + 1]                             # skip the trivial (~0) eigenvector
    if pe.shape[1] < k:                            # pad tiny graphs
        pe = np.pad(pe, ((0, 0), (0, k - pe.shape[1])))
    return pe.astype("float32")


def rwse(A: np.ndarray, k: int = 8) -> np.ndarray:
    """Random-walk structural encoding: return probability diag(P^s) for s=1..k."""
    d = A.sum(1, keepdims=True)
    P = A / np.maximum(d, 1e-9)
    feats, M = [], np.eye(len(A))
    for _ in range(k):
        M = M @ P
        feats.append(np.diag(M).copy())
    return np.stack(feats, 1).astype("float32")    # [N, k]


def struct_features(A: np.ndarray, k_lap: int = 8, k_rw: int = 8) -> np.ndarray:
    """Per-node structural embedding = [LapPE ; RWSE], column-standardised. This is the 'extra dimension'
    that captures subcomponents/motifs a per-edge message-passing encoder misses."""
    S = np.concatenate([lap_pe(A, k_lap), rwse(A, k_rw)], axis=1)
    return ((S - S.mean(0)) / (S.std(0) + 1e-9)).astype("float32")


def _gen_struct_corpus(N=80, d_content=16, seed=0):
    """A graph where each atom i has a DEPEND-child child[i] whose CONTENT is unrelated. Tasks query q and
    need {q, child[q]} — the child is reachable only via STRUCTURE, not content similarity."""
    rng = np.random.default_rng(seed)
    child = rng.permutation(N)
    A = np.zeros((N, N), dtype="float32")
    for i in range(N):
        A[i, child[i]] = A[child[i], i] = 1.0      # the depend edge (undirected for the Laplacian)
    C = max(2, N // 10)                            # + community edges for richer structure
    comm = rng.integers(0, C, N)
    for i in range(N):
        for j in range(i + 1, N):
            if comm[i] == comm[j] and rng.random() < 0.12:
                A[i, j] = A[j, i] = 1.0
    content = rng.standard_normal((N, d_content)).astype("float32")
    content /= np.linalg.norm(content, axis=1, keepdims=True) + 1e-9
    S = struct_features(A)

    # HELD-OUT node split: test queries are atoms NEVER used as a training query -> content-only cannot
    # memorise their child (no signal); the structural rule ("fetch the struct-adjacent atom") generalises.
    nodes = rng.permutation(N)
    train_nodes, test_nodes = nodes[:int(0.6 * N)], nodes[int(0.6 * N):]

    def gen(pool, n, s):
        r = np.random.default_rng(s)
        qs = pool[r.integers(0, len(pool), n)]
        return [(int(q), {int(q), int(child[q])}) for q in qs]

    return content, S, child, gen(train_nodes, 600, seed + 1), gen(test_nodes, 200, seed + 2)


def _selftest() -> bool:
    print("algo_grr_struct --selftest: structure as a representation dimension (no GPU)\n")
    from v5.runtime.algo_grr_router import _build, train_router, _recall_at_k
    torch, nn, NeuralRouter = _build()

    content, S, child, train, test = _gen_struct_corpus()
    N = content.shape[0]
    cat = np.concatenate([content, S], axis=1)     # content + structure

    def run(atom_feat, q_feat):
        d = atom_feat.shape[1]
        A = torch.as_tensor(atom_feat)
        tr = [(q_feat[q], nd) for q, nd in train]
        router = NeuralRouter(d)
        train_router(router, tr, atom_feat, epochs=300, lr=3e-3)
        with torch.no_grad():
            return float(np.mean([_recall_at_k(router(torch.as_tensor(q_feat[q]), A).tolist(), nd, 2)
                                  for q, nd in test]))

    print("  routing Recall@2 for {query, its depend-child} — child has UNRELATED content:\n")
    r_content = run(content, content)              # content-only: blind to the depend edge
    r_struct = run(cat, cat)                        # + structural features
    print(f"    content-only router     : {r_content:.2f}   (HELD-OUT nodes: can't memorise the child)")
    print(f"    content + STRUCTURE     : {r_struct:.2f}   (LapPE/RWSE add a structural signal)")
    lift = r_struct - r_content
    ok = lift > 0.08                                # honest bar: structure gives a real lift on held-out nodes
    print(f"    structure lift          : {lift:+.2f}  ->  {'PASS (structure helps)' if ok else 'FAIL'}")
    print("\n  HONEST READ: LapPE/RWSE encode GLOBAL position/community -> a real but MODEST lift; they don't\n"
          "  uniquely pin a SPECIFIC depend-edge (many nodes share a neighbourhood). Recovering a specific\n"
          "  edge wants MESSAGE-PASSING (query-conditioned adjacency propagation) = the topology retriever we\n"
          "  ALREADY have. The full recipe is GraphGPS: MPNN (topology boost) + positional encodings (here).\n"
          "  Build-phase roadmap: fuse both into the router's atom features.")
    print(f"\n  ALGO_GRR_STRUCT SELFTEST -> {'PASS' if ok else 'FAIL'}")
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
