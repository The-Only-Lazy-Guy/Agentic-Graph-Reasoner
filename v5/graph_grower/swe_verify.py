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
import time
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


def read_prediction_instance_ids(path: str, limit: int) -> List[str]:
    """Read the first N prediction instance_ids from a jsonl file for cheap smoke verifies."""
    if limit <= 0 or not os.path.exists(path):
        return []
    ids: List[str] = []
    seen = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            iid = str(row.get("instance_id", "")).strip()
            if not iid or iid in seen:
                continue
            ids.append(iid)
            seen.add(iid)
            if len(ids) >= limit:
                break
    return ids


def run_swebench(preds_path: str, dataset: str, run_id: str,
                 instance_ids: Optional[Sequence[str]] = None, max_workers: int = 4,
                 model_name: str = MODEL_NAME, timeout: int = 1800) -> Dict[str, bool]:
    """Shell out to the official harness; return {instance_id: resolved bool}."""
    if not os.path.exists(preds_path):
        print(f"  ERROR: predictions file does not exist: {preds_path}", flush=True)
        return {}
    ds = DATASET_MAP.get(dataset, dataset)
    cmd = [sys.executable, "-m", "swebench.harness.run_evaluation",
           "--dataset_name", ds, "--predictions_path", preds_path,
           "--run_id", run_id, "--max_workers", str(max_workers),
           "--cache_level", "env"]
    if instance_ids:
        cmd += ["--instance_ids", *instance_ids]
    print("  $ " + " ".join(cmd), flush=True)
    started = time.time()
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    sys.stdout.write(proc.stdout[-4000:]); sys.stderr.write(proc.stderr[-2000:])
    if proc.returncode != 0:
        print(f"  ERROR: swebench harness exited {proc.returncode}; ignoring any stale prior reports for run_id={run_id}.", flush=True)
        return {}
    return parse_report(model_name, run_id, min_mtime=started - 1.0)


def parse_report(model_name: str, run_id: str, min_mtime: float | None = None) -> Dict[str, bool]:
    """swebench writes <model_name>.<run_id>.json with resolved/unresolved id lists."""
    cands = glob.glob(f"{model_name}.{run_id}.json") + glob.glob(f"*{run_id}*.json")
    cands = [p for p in dict.fromkeys(cands) if os.path.isfile(p)]
    if min_mtime is not None:
        cands = [p for p in cands if os.path.getmtime(p) >= min_mtime]
    cands = sorted(cands, key=os.path.getmtime, reverse=True)
    for p in cands:
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
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
                run_id: str, max_workers: int, out_dir: str, backend: str = "docker") -> int:
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
    res = _verify(preds, dataset, run_id, backend, list(id2patch), max_workers, out_dir, split)
    if res:
        ok = sum(1 for v in res.values() if v)
        print(f"\nGOLD resolved {ok}/{len(res)}  "
              f"({'HARNESS OK' if ok == len(res) else 'CHECK: gold should ~all resolve'})")
        for i, v in res.items():
            print(f"  {'PASS' if v else 'FAIL'}  {i}")
    return 0 if res else 1


DATASET_SBCLI = {"lite": "swe-bench_lite", "verified": "swe-bench_verified", "full": "swe-bench"}


def run_sbcli(preds_path: str, dataset: str, run_id: str, split: str = "test",
              out_dir: str = "artifacts/graph_growth/swe_verify", submit: bool = True,
              poll_secs: int = 20, max_wait: int = 1800) -> Dict[str, bool]:
    """sb-cli backend (hosted, NO Docker): submit predictions, then poll `get-report` until the
    remote eval finishes (pending=0) -> {instance_id: resolved}. Needs sb-cli + SWEBENCH_API_KEY.
    submit=False just polls an existing run_id."""
    import glob
    import time
    sub = DATASET_SBCLI.get(dataset, dataset)
    if submit:
        cmd = ["sb-cli", "submit", sub, split, "--predictions_path", preds_path,
               "--run_id", run_id, "--output_dir", out_dir]
        print("  $ " + " ".join(cmd), flush=True)
        subprocess.run(cmd, text=True, capture_output=True, timeout=600)
    deadline = time.time() + max_wait
    d: dict = {}
    while True:
        subprocess.run(["sb-cli", "get-report", sub, split, run_id, "-o", out_dir, "--overwrite", "1"],
                       text=True, capture_output=True, timeout=300)
        cands = [p for p in glob.glob(os.path.join(out_dir, "*.json")) if run_id in os.path.basename(p)]
        if cands:
            try:
                with open(max(cands, key=os.path.getmtime), encoding="utf-8") as f:
                    d = json.load(f)
            except Exception:   # noqa: BLE001
                d = {}
        pend, subm = d.get("pending_instances", 1), d.get("submitted_instances", 0)
        print(f"  poll: submitted={subm} pending={pend} resolved={d.get('resolved_instances', 0)} "
              f"failed={d.get('failed_instances', 0)}", flush=True)
        if d and pend == 0 and subm > 0:
            break
        if time.time() > deadline:
            print("  TIMEOUT waiting for the remote eval (results may still finish; re-poll later)."); break
        time.sleep(poll_secs)
    resolved = set(d.get("resolved_ids") or [])
    ids = set(d.get("submitted_ids") or [])
    return {i: (i in resolved) for i in ids}


def _verify(preds_path, dataset, run_id, backend, instance_ids, max_workers, out_dir, split):
    if backend == "sbcli":
        return run_sbcli(preds_path, dataset, run_id, split=split, out_dir=out_dir)
    return run_swebench(preds_path, dataset, run_id, instance_ids=instance_ids, max_workers=max_workers)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SWE verifier (Docker swebench harness OR hosted sb-cli).")
    ap.add_argument("--gold-sanity", action="store_true",
                    help="feed GOLD patches as predictions -> must resolve (proves the harness)")
    ap.add_argument("--predictions", default="", help="predictions jsonl {instance_id, model_patch}")
    ap.add_argument("--dataset", default="lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--instance-ids", nargs="*", default=None)
    ap.add_argument("--predictions-limit", type=int, default=0,
                    help="verify only the first N instance_ids found in the predictions file (cheap smoke run)")
    ap.add_argument("--run-id", default="v5verify")
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--out-dir", default="artifacts/graph_growth/swe_verify")
    ap.add_argument("--backend", choices=["docker", "sbcli"], default="docker",
                    help="docker = official swebench harness (needs Docker); sbcli = hosted (no Docker)")
    args = ap.parse_args(argv)

    # preflight per backend
    if args.backend == "docker":
        try:
            import swebench  # noqa: F401
        except ImportError:
            print("FATAL: swebench not installed. Run: pip install swebench  (+ a running Docker daemon)")
            return 2
        if subprocess.run(["docker", "info"], capture_output=True, text=True).returncode != 0:
            print("FATAL: Docker daemon not reachable (`docker info`). Use --backend sbcli (hosted, no Docker).")
            return 2
    else:  # sbcli
        if subprocess.run(["sb-cli", "--help"], capture_output=True, text=True).returncode != 0:
            print("FATAL: sb-cli not installed. Run: pip install sb-cli  (+ export SWEBENCH_API_KEY=...)")
            return 2
        if not os.environ.get("SWEBENCH_API_KEY"):
            print("WARN: SWEBENCH_API_KEY unset -> sb-cli submit will fail. `sb-cli gen-api-key your@email`.")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if args.gold_sanity:
        return gold_sanity(args.dataset, args.split, args.limit, args.instance_ids,
                           args.run_id, args.max_workers, args.out_dir, backend=args.backend)
    if args.predictions:
        if not args.instance_ids and args.predictions_limit > 0:
            args.instance_ids = read_prediction_instance_ids(args.predictions, args.predictions_limit)
            if not args.instance_ids:
                print(f"FATAL: no prediction ids found in {args.predictions} for --predictions-limit {args.predictions_limit}")
                return 2
            print(f"prediction smoke: verifying first {len(args.instance_ids)} ids from {args.predictions}")
        res = _verify(args.predictions, args.dataset, args.run_id, args.backend,
                      args.instance_ids, args.max_workers, args.out_dir, args.split)
        if not res:
            print("\nverification failed or produced no results; not writing a resolved count from stale files.")
            return 1
        ok = sum(1 for v in res.values() if v)
        report = os.path.join(args.out_dir, f"verify_{args.run_id}.json")
        with open(report, "w", encoding="utf-8") as f:
            json.dump({"resolved": ok, "total": len(res), "results": res}, f, indent=2)
        print(f"\nresolved {ok}/{len(res)} -> {report}")
        return 0
    print("nothing to do: pass --gold-sanity or --predictions <jsonl>")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
