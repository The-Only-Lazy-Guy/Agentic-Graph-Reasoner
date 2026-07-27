"""One-shot prep for `--task-domain swe-action`: fetch real Open-SWE-Traces trajectories and build the
step-concept graph they get grounded against.

MUST be run as its own process, before any training run. Two hard reasons, both confirmed by real crashes
in this codebase, not style preferences:
  1. HF `datasets` streaming's first real fetch segfaults if torch is already loaded in the process, and
     `trm_wm.py` imports torch at module scope -- so the fetch cannot happen inside a --run invocation.
     This is the same conflict that forced --grow-cot-docs-path and --math-cot-docs-path.
  2. Graph growth is O(N^2)-ish (every insert re-checks similarity against the whole graph), so it is
     worth doing once and reusing across every arm of an A/B rather than repeating per run.

    python -m scripts.prep_swe_domain --n-trajectories 200 \
        --out-trajs artifacts/swe_trajs.jsonl --out-graph graphs/swe_graph.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Prep real SWE trajectories + step-concept graph.")
    ap.add_argument("--n-trajectories", type=int, default=200)
    ap.add_argument("--out-trajs", type=str, default="artifacts/swe_trajs.jsonl")
    ap.add_argument("--out-graph", type=str, default="graphs/swe_graph.json")
    ap.add_argument("--min-step-chars", type=int, default=40)
    ap.add_argument("--config", type=str, default="openhands")
    ap.add_argument("--split", type=str, default="minimax_m25")
    a = ap.parse_args(argv)

    # STEP 1 -- fetch FIRST, before torch is touched anywhere (see module docstring).
    from v5.graph_grower.fetch_swe_traces import stream_swe_trajectories
    trajs = list(stream_swe_trajectories(config=a.config, split=a.split,
                                         resolved_only=True, limit=a.n_trajectories))
    Path(a.out_trajs).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out_trajs, "w", encoding="utf-8") as f:
        for t in trajs:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    steps = [len(t.get("steps") or []) for t in trajs]
    avg = sum(steps) / max(1, len(steps))
    print(f"[1/2] wrote {len(trajs)} resolved trajectories -> {a.out_trajs}")
    print(f"      steps per trajectory: avg {avg:.1f}, min {min(steps) if steps else 0}, "
          f"max {max(steps) if steps else 0}  (long-horizon: this is why eviction can actually fire here)")

    # STEP 2 -- only NOW import torch-backed modules, and feed them the already-materialized rows.
    import v5.graph_grower.fetch_swe_traces as fst
    fst.stream_swe_trajectories = lambda **kw: iter(trajs[:kw.get("limit") or len(trajs)])
    from v5.runtime.membrane import AtomGraph
    from v5.runtime.trm_wm import _grow_swe_step_concepts

    g = AtomGraph()
    stats = _grow_swe_step_concepts(g, n_trajectories=len(trajs), min_step_chars=a.min_step_chars)
    Path(a.out_graph).parent.mkdir(parents=True, exist_ok=True)
    g.save(a.out_graph)
    print(f"[2/2] graph -> {a.out_graph}: {len(g)} nodes, {len(g.edges)} edges, census={g.census()}")
    print(f"      growth: {stats}")
    print(f"\nnext, e.g.:\n"
          f"  python -m v5.runtime.trm_wm --run --lm Qwen/Qwen3-4B-Instruct-2507 --quant 4bit \\\n"
          f"    --graph-path {a.out_graph} --task-domain swe-action --swe-docs-path {a.out_trajs} \\\n"
          f"    --epochs 40 --n-train 48 --n-held 16 --cotrain-samples 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
