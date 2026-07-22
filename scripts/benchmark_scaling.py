"""Benchmark for scaling improvements in membrane.py / trm_wm.py.
Measures correctness (demo parity) and performance (retrieval speed, TRM speed, memory)."""

import time
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch

from embedder import encode_batch, EMBED_DIM
from v5.runtime.membrane import (
    AtomGraph, Atom, TRMRetriever, Membrane, seed_graph,
    build_examples, demo,
)


def benchmark_retrieval_speed(g, retriever, queries, label="", n_runs=5):
    """Measure retrieval speed for N queries."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        for q in queries:
            _ = retriever.rank(q, k=3)
        times.append(time.perf_counter() - t0)
    avg = sum(times) / len(times)
    print(f"  {label}  {avg*1000/len(queries):.2f} ms/query  (total {avg:.3f}s for {len(queries)} runs)")
    return avg


def benchmark_cosine_speed(g, queries, label="", n_runs=5):
    """Measure cosine retrieval speed."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        for q in queries:
            _ = g.cosine_rank(q, k=3)
        times.append(time.perf_counter() - t0)
    avg = sum(times) / len(times)
    print(f"  {label}  {avg*1000/len(queries):.2f} ms/query  (total {avg:.3f}s for {len(queries)} runs)")
    return avg


def benchmark_train_speed(retriever, train_ex, label="", n_runs=3):
    """Measure TRM training speed for one epoch."""
    # warmup
    retriever.train(train_ex, epochs=5)
    times = []
    for _ in range(n_runs):
        retriever.trm = retriever.__class__.__module__  # reset won't work perfectly; just measure 1 epoch
        # Actually let's just measure train for a fixed number of epochs
        t0 = time.perf_counter()
        retriever.train(train_ex, epochs=10)
        times.append(time.perf_counter() - t0)
    avg = sum(times) / len(times)
    print(f"  {label}  {avg/10:.3f}s/epoch  (total {avg:.3f}s for 10 epochs)")
    return avg


def benchmark_insert_speed(g, n_atoms=100):
    """Measure atom insert speed."""
    atoms = []
    for i in range(n_atoms):
        atoms.append(Atom(
            name=f"bench_atom_{i}",
            code=f"def bench_{i}(n):\n    return n + {i}",
            description=f"benchmark atom number {i} that adds {i} to n",
            provenance="bench",
        ))
    t0 = time.perf_counter()
    for a in atoms:
        g.add(a)
    elapsed = time.perf_counter() - t0
    print(f"  insert {n_atoms} atoms  {elapsed*1000/n_atoms:.2f} ms/insert  (total {elapsed:.3f}s)")
    return elapsed


def benchmark_add_or_merge_speed(g, n_atoms=50):
    """Measure add_or_merge insert speed with dedup."""
    atoms = []
    for i in range(n_atoms):
        atoms.append(Atom(
            name=f"merge_atom_{i}",
            code=f"def merge_{i}(n):\n    return n * {i}",
            description=f"merge test atom {i} that multiplies by {i}",
            provenance="bench",
        ))
    t0 = time.perf_counter()
    for a in atoms:
        g.add_or_merge(a)
    elapsed = time.perf_counter() - t0
    print(f"  add_or_merge {n_atoms} atoms  {elapsed*1000/n_atoms:.2f} ms/insert  (total {elapsed:.3f}s)")
    return elapsed


def benchmark_full_demo():
    """Run the full demo and return all metrics for correctness comparison."""
    print("\n  -- Running full demo (correctness check) --")
    import io
    old_stdout = sys.stdout
    sys.stdout = captured = io.StringIO()
    try:
        result = demo()
    finally:
        sys.stdout = old_stdout
    output = captured.getvalue()
    sys.stdout = old_stdout  # restore
    # Parse key numbers from output
    lines = output.splitlines()
    print("  (demo output suppressed for cleanliness)")
    return result, output


def run():
    print("=" * 60)
    print("SCALING BENCHMARK — membrane.py / trm_wm.py")
    print("=" * 60)

    torch.manual_seed(0)

    # === 1. Small graph benchmarks (baseline: 10 atoms) ===
    print("\n--- 1. Small graph (10 atoms) ---")
    g = seed_graph()
    retr = TRMRetriever(g)
    train_ex, test_ex = build_examples("train"), build_examples("test")

    queries = [t for t, _ in test_ex]
    print("\n  [retrieval speed]")
    cs = benchmark_cosine_speed(g, queries, "cosine_rank   (10 atoms)")
    tr = benchmark_retrieval_speed(g, retr, queries, "TRM rank      (10 atoms)")

    print("\n  [TRM train speed]")
    t0 = time.perf_counter()
    stats = retr.train(train_ex, epochs=80, verbose=False)
    train_time = time.perf_counter() - t0
    print(f"  TRM train 80 epochs  {train_time:.3f}s total  ({train_time/80:.3f}s/epoch)")
    acc = retr.top1_accuracy(test_ex)
    print(f"  TRM held-out accuracy: {acc:.2f}")

    print("\n  [insert speed]")
    benchmark_insert_speed(g, 100)
    benchmark_add_or_merge_speed(g, 50)

    # === 2. Scaled graph (1000 atoms with synthetic data) ===
    print("\n--- 2. Scaled graph (1000 atoms) ---")
    g2 = seed_graph()
    print(f"  starting with {len(g2)} seed atoms")
    for i in range(990):
        g2.add(Atom(
            name=f"synth_{i}",
            code=f"def synth_{i}(n):\n    return n * {i}",
            description=f"synthetic atom number {i} that multiplies input by {i} for scaling tests",
            kind="concept" if i % 3 == 0 else "atom",
            provenance="bench",
        ))
    print(f"  graph now has {len(g2)} atoms")
    retr2 = TRMRetriever(g2)

    scaled_queries = queries + [f"synthetic test query number {i}" for i in range(20)]
    print("\n  [retrieval speed at scale]")
    cs2 = benchmark_cosine_speed(g2, scaled_queries, "cosine_rank (1000 atoms)", n_runs=3)
    tr2 = benchmark_retrieval_speed(g2, retr2, scaled_queries, "TRM rank    (1000 atoms)", n_runs=3)

    print("\n  [TRM train speed at scale]")
    train_ex2 = train_ex + [("synth query " + str(i), "synth_" + str(i % 990)) for i in range(20)]
    t0 = time.perf_counter()
    stats2 = retr2.train(train_ex2, epochs=10, verbose=False)
    train_time2 = time.perf_counter() - t0
    print(f"  TRM train 10 epochs  {train_time2:.3f}s total  ({train_time2/10:.3f}s/epoch)")

    # === 3. Full demo correctness ===
    print("\n--- 3. Full demo (correctness check) ---")
    result, output = benchmark_full_demo()

    print("\n  Benchmark results summary:")
    print(f"  {'Metric':<40} {'10 atoms':<15} {'1000 atoms':<15}")
    print(f"  {'cosine_rank (ms/query)':<40} {cs*1000/len(queries):<15.2f} {cs2*1000/len(scaled_queries):<15.2f}")
    print(f"  {'TRM rank (ms/query)':<40} {tr*1000/len(queries):<15.2f} {tr2*1000/len(scaled_queries):<15.2f}")
    print(f"  {'TRM train (s/epoch)':<40} {train_time/80:<15.3f} {train_time2/10:<15.3f}")
    print(f"  {'TRM held-out accuracy':<40} {acc:<15.2f} {'N/A (mixed)':<15}")

    print("\n  Demo correctness:")
    print(f"    cosine: {result['cos']:.2f}  TRM before: {result['trm_before']:.2f}  "
          f"TRM after: {result['trm_after']:.2f}")
    print(f"    solved: {result['solved']}  composed: {result['composed']}  "
          f"reuse: {result['reuse']}  cot: {result['cot']}  rebuild: {result['rebuild']:.2f}")

    return {
        "cosine_small_ms": cs * 1000 / len(queries),
        "trm_small_ms": tr * 1000 / len(queries),
        "trm_train_small_spoch": train_time / 80,
        "trm_accuracy": acc,
        "cosine_large_ms": cs2 * 1000 / len(scaled_queries),
        "trm_large_ms": tr2 * 1000 / len(scaled_queries),
        "trm_train_large_spoch": train_time2 / 10,
        "demo": result,
    }


if __name__ == "__main__":
    run()
