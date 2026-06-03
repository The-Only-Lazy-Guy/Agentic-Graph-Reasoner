"""SWE verifier harness (stage 5) — execute a candidate patch against an instance's
FAIL_TO_PASS / PASS_TO_PASS tests and return resolved / not. This is the deferred
"expensive rung": it lets the §12 brief-vs-cold probe SCORE the 4B's patches by real
test execution, not heuristics.

Architecture decision: WRAP the official `swebench` harness (Docker, per-instance env)
rather than reinvent 300+ repo environments. We only add the glue: build predictions,
shell out to `swebench.harness.run_evaluation`, parse the report -> {instance_id: resolved}.

Requirements (on the runner box): `pip install swebench` + a running Docker daemon.
First, ALWAYS gold-sanity: feed the GOLD patches as predictions -> they must (nearly all)
resolve. If gold doesn't resolve, the harness/env is broken, NOT the model. Only trust
model-patch numbers once gold-sanity passes.

  # 1) prove the harness works (gold patches must resolve)
  python -m v5.graph_grower.swe_verify --gold-sanity --dataset lite --limit 5

  # 2) score model patches (predictions jsonl: {instance_id, model_patch})
  python -m v5.graph_grower.swe_verify --predictions preds.jsonl --dataset lite --run-id probe1
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from v5.graph_grower.swe_load import load_instances

DATASET_MAP = {
    "lite": "princeton-nlp/SWE-bench_Lite",
    "verified": "princeton-nlp/SWE-bench_Verified",
    "full": "princeton-nlp/SWE-bench",
}
MODEL_NAME = "v5_probe"   # used by swebench in the report/output filename


def write_predictions(id2patch: Dict[str, str], path: str, model_name: str = MODEL_NAME) -> int:
    """{instance_id: patch} -> swebench predictions jsonl."""
    n = 0
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as w:
        for iid, patch in id2patch.items():
            if not (patch or "").strip():
                continue
            w.write(json.dumps({"instance_id": iid, "model_name_or_path": model_name,
                                "model_patch": patch}, ensure_ascii=False) + "\n")
            n += 1
    return n


def run_swebench(preds_path: str, dataset: str, run_id: str,
                 instance_ids: Optional[Sequence[str]] = None, max_workers: int = 4,
                 model_name: str = MODEL_NAME, timeout: int = 1800) -> Dict[str, bool]:
    """Shell out to the official harness; return {instance_id: resolved bool}."""
    ds = DATASET_MAP.get(dataset, dataset)
    cmd = [sys.executable, "-m", "swebench.harness.run_evaluation",
           "--dataset_name", ds, "--predictions_path", preds_path,
           "--run_id", run_id, "--max_workers", str(max_workers),
           "--cache_level", "env"]
    if instance_ids:
        cmd += ["--instance_ids", *instance_ids]
    print("  $ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    sys.stdout.write(proc.stdout[-4000:]); sys.stderr.write(proc.stderr[-2000:])
    return parse_report(model_name, run_id)


def parse_report(model_name: str, run_id: str) -> Dict[str, bool]:
    """swebench writes <model_name>.<run_id>.json with resolved/unresolved id lists."""
    cands = glob.glob(f"{model_name}.{run_id}.json") + glob.glob(f"*{run_id}*.json")
    for p in cands:
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:   # noqa: BLE001
            continue
        if "resolved_ids" in d or "resolved_instances" in d:
            resolved = set(d.get("resolved_ids") or [])
            submitted = set(d.get("submitted_ids") or d.get("completed_ids") or [])
            ids = submitted or (resolved | set(d.get("unresolved_ids") or []))
            return {i: (i in resolved) for i in ids}
    print("  WARN: no swebench report json found; check the harness output above.")
    return {}


def gold_sanity(dataset: str, split: str, limit: int, instance_ids: Optional[List[str]],
                run_id: str, max_workers: int, out_dir: str) -> int:
    insts = load_instances(dataset, split, limit=0)
    by_id = {t["instance_id"]: t for t in insts}
    if instance_ids:
        chosen = [by_id[i] for i in instance_ids if i in by_id]
    else:
        chosen = insts[:limit] if limit else insts
    id2patch = {t["instance_id"]: t.get("patch", "") for t in chosen}
    preds = os.path.join(out_dir, "gold_predictions.jsonl")
    n = write_predictions(id2patch, preds)
    print(f"gold-sanity: {n} instances -> {preds}  (gold patches MUST resolve)", flush=True)
    res = run_swebench(preds, dataset, run_id, instance_ids=list(id2patch), max_workers=max_workers)
    if res:
        ok = sum(1 for v in res.values() if v)
        print(f"\nGOLD resolved {ok}/{len(res)}  "
              f"({'HARNESS OK' if ok == len(res) else 'CHECK: gold should ~all resolve'})")
        for i, v in res.items():
            print(f"  {'PASS' if v else 'FAIL'}  {i}")
    return 0 if res else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SWE verifier (wraps official swebench harness).")
    ap.add_argument("--gold-sanity", action="store_true",
                    help="feed GOLD patches as predictions -> must resolve (proves the harness)")
    ap.add_argument("--predictions", default="", help="predictions jsonl {instance_id, model_patch}")
    ap.add_argument("--dataset", default="lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--instance-ids", nargs="*", default=None)
    ap.add_argument("--run-id", default="v5verify")
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--out-dir", default="artifacts/graph_growth/swe_verify")
    args = ap.parse_args(argv)

    # preflight: swebench + docker present?
    try:
        import swebench  # noqa: F401
    except ImportError:
        print("FATAL: swebench not installed. Run: pip install swebench  (+ a running Docker daemon)")
        return 2
    if subprocess.run(["docker", "info"], capture_output=True, text=True).returncode != 0:
        print("FATAL: Docker daemon not reachable (`docker info` failed). The harness needs Docker.")
        return 2

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if args.gold_sanity:
        return gold_sanity(args.dataset, args.split, args.limit, args.instance_ids,
                           args.run_id, args.max_workers, args.out_dir)
    if args.predictions:
        res = run_swebench(args.predictions, args.dataset, args.run_id,
                           instance_ids=args.instance_ids, max_workers=args.max_workers)
        ok = sum(1 for v in res.values() if v)
        report = os.path.join(args.out_dir, f"verify_{args.run_id}.json")
        json.dump({"resolved": ok, "total": len(res),
                   "results": res}, open(report, "w"), indent=2)
        print(f"\nresolved {ok}/{len(res)} -> {report}")
        return 0
    print("nothing to do: pass --gold-sanity or --predictions <jsonl>")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
