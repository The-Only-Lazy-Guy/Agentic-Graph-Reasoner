"""Targeted scaling benchmark: isolates matrix rebuild cost and TRM pre-filter speedup."""

import time
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import numpy as np
from embedder import encode_batch, EMBED_DIM
from v5.runtime.membrane import AtomGraph, Atom, TRMRetriever, seed_graph, build_examples


def bench_matrix_rebuild():
    """Measure matrix rebuild cost: number of matrix() calls after bulk insert."""
    print("=== Matrix rebuild cost (1 matrix() call after N inserts) ===")
    for n_ops in [10, 100, 500, 1000]:
        g = seed_graph()
        for i in range(n_ops):
            g.add(Atom(name="synth_%d" % i, code="def f(n): return n",
                       description="test", provenance="test"))
        t0 = time.perf_counter()
        M, order = g.matrix()
        t_rebuild = time.perf_counter() - t0
        # Subsequent matrix() calls are cached
        t0 = time.perf_counter()
        for _ in range(100):
            g.matrix()
        t_cached = (time.perf_counter() - t0) / 100
        print("  %5d atoms: rebuild=%.3fms  cached=%.3fms" % (
            n_ops, t_rebuild * 1000, t_cached * 1000))


def bench_trm_prefilter():
    """Compare TRM train speed with and without pre-filter at various atom counts."""
    print()
    print("=== TRM pre-filter speed (10 epochs train) ===")
    for n_atoms in [10, 100, 1000, 5000, 10000]:
        g = seed_graph()
        added = 0
        for i in range(n_atoms - len(g)):
            g.add(Atom(name="synth_%d" % i,
                       code="def x(n):\n    return n",
                       description="synthetic atom %d for scaling tests" % i,
                       kind="concept",
                       provenance="test"))
            added += 1
        train_ex = build_examples("train")
        n_actual = len(g)
        if n_actual < 10:
            print("  %5d atoms  (skipping - too few)" % n_actual)
            continue

        # With pre-filter (top-256)
        r1 = TRMRetriever(g, trm_top_k=256)
        t0 = time.perf_counter()
        r1.train(train_ex, epochs=10, verbose=False)
        t1 = time.perf_counter()
        acc1 = r1.top1_accuracy(train_ex)

        # Without pre-filter (0 = all atoms)
        torch.manual_seed(0)
        g2 = seed_graph()
        for i in range(n_atoms - len(g2)):
            g2.add(Atom(name="synth_%d" % i,
                        code="def x(n):\n    return n",
                        description="synthetic atom %d for scaling tests" % i,
                        kind="concept",
                        provenance="test"))
        r2 = TRMRetriever(g2, trm_top_k=0)
        t2 = time.perf_counter()
        r2.train(train_ex, epochs=10, verbose=False)
        t3 = time.perf_counter()
        acc2 = r2.top1_accuracy(train_ex)

        speedup = (t3 - t2) / max(t1 - t0, 1e-9)
        print("  %5d atoms  pre-filter(256): %.3fs (acc=%.2f)" % (n_actual, t1 - t0, acc1))
        print("  %5d atoms  no pre-filter:   %.3fs (acc=%.2f)" % (n_actual, t3 - t2, acc2))
        print("  %5d        speedup: %.1fx" % (n_actual, speedup))


if __name__ == "__main__":
    print("TARGETED SCALING BENCHMARK")
    print("=" * 50)
    torch.manual_seed(0)
    bench_matrix_rebuild()
    bench_trm_prefilter()
