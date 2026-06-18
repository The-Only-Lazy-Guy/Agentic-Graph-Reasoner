"""Shared exact SWE verifier adapter for runtime experiments.

Wraps the existing `v5.graph_grower.swe_verify` harness helpers so runtime
experiments can do exact pass/fail evaluation without each file reimplementing
Docker / sb-cli preflight, prediction writing, caching, and gold sanity.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import time
from pathlib import Path

from v5.graph_grower.swe_verify import run_sbcli, run_swebench, write_predictions


class SWEExactVerifier:
    """Small adapter around the existing swe_verify harness, with preflight + caching.

    `verify_patch()` is for one patch on one instance.
    `verify_task_batch_unique()` is for one patch per unique instance.
    """

    def __init__(self, dataset: str, split: str, backend: str, out_dir: str,
                 max_workers: int = 4, timeout: int = 1800, poll_secs: int = 20,
                 model_name: str = "swe_runtime") -> None:
        self.dataset = dataset
        self.split = split
        self.backend = backend
        self.out_dir = Path(out_dir)
        self.max_workers = max_workers
        self.timeout = timeout
        self.poll_secs = poll_secs
        self.model_name = model_name
        self.cache: dict[tuple[str, str], bool] = {}
        self._counter = 0
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._preflight()

    def _preflight(self) -> None:
        if self.backend == "docker":
            try:
                import swebench  # noqa: F401
            except ImportError as exc:  # pragma: no cover - environment-dependent
                raise SystemExit("FATAL: swebench not installed; exact verifier mode needs `pip install swebench`.") from exc
            if subprocess.run(["docker", "info"], capture_output=True, text=True).returncode != 0:
                raise SystemExit("FATAL: Docker daemon not reachable; use --verify-backend sbcli or stay in proxy mode.")
        else:
            if subprocess.run(["sb-cli", "--help"], capture_output=True, text=True).returncode != 0:
                raise SystemExit("FATAL: sb-cli not installed; exact verifier mode needs `pip install sb-cli`.")
            if not os.environ.get("SWEBENCH_API_KEY"):
                print("WARN: SWEBENCH_API_KEY unset; hosted sb-cli submit may fail.", flush=True)

    def _slug(self, text: str) -> str:
        keep = [c.lower() if c.isalnum() else "_" for c in (text or "run")]
        s = "".join(keep).strip("_")
        return s[:48] or "run"

    def _next_run_id(self, tag: str) -> str:
        self._counter += 1
        return f"{self._slug(tag)}_{int(time.time())}_{self._counter:05d}"

    def _pred_path(self, run_id: str) -> Path:
        return self.out_dir / f"{run_id}.predictions.jsonl"

    def _run_docker(self, preds_path: str, run_id: str, instance_ids: list[str]) -> dict[str, bool]:
        return run_swebench(preds_path, self.dataset, run_id, instance_ids=instance_ids,
                            max_workers=self.max_workers, model_name=self.model_name,
                            timeout=self.timeout)

    def _run(self, preds_path: str, run_id: str, instance_ids: list[str]) -> dict[str, bool]:
        if self.backend == "sbcli":
            return run_sbcli(preds_path, self.dataset, run_id, split=self.split,
                             out_dir=str(self.out_dir), submit=True,
                             poll_secs=self.poll_secs, max_wait=self.timeout)
        return self._run_docker(preds_path, run_id, instance_ids)

    def verify_patch(self, task: dict, patch: str, tag: str = "reward") -> bool:
        key = (task["iid"], hashlib.sha1(patch.encode("utf-8")).hexdigest())
        if key in self.cache:
            return self.cache[key]
        run_id = self._next_run_id(f"{tag}_{task['iid']}")
        preds_path = self._pred_path(run_id)
        n = write_predictions({task["iid"]: patch}, str(preds_path), model_name=self.model_name)
        if n <= 0:
            self.cache[key] = False
            return False
        res = self._run(str(preds_path), run_id, [task["iid"]])
        ok = bool(res.get(task["iid"], False))
        self.cache[key] = ok
        return ok

    def verify_task_batch_unique(self, task_patches: list[tuple[dict, str]], tag: str = "eval") -> dict[str, bool]:
        results: dict[str, bool] = {}
        id2patch: dict[str, str] = {}
        key_by_iid: dict[str, tuple[str, str]] = {}
        for task, patch in task_patches:
            key = (task["iid"], hashlib.sha1(patch.encode("utf-8")).hexdigest())
            if key in self.cache:
                results[task["iid"]] = self.cache[key]
                continue
            if task["iid"] in id2patch:
                raise ValueError("verify_task_batch_unique requires at most one patch per instance_id")
            id2patch[task["iid"]] = patch
            key_by_iid[task["iid"]] = key
        if id2patch:
            run_id = self._next_run_id(tag)
            preds_path = self._pred_path(run_id)
            n = write_predictions(id2patch, str(preds_path), model_name=self.model_name)
            fresh = self._run(str(preds_path), run_id, list(id2patch)) if n > 0 else {}
            for iid, key in key_by_iid.items():
                ok = bool(fresh.get(iid, False))
                self.cache[key] = ok
                results[iid] = ok
        return results

    def run_gold_sanity(self, tasks: list[dict], n: int, tag: str = "gold_sanity") -> tuple[int, int]:
        gold = [(t, t["gold"]) for t in tasks[:n] if (t.get("gold") or "").strip()]
        if not gold:
            raise SystemExit("exact verifier requested, but no gold patches available for sanity check")
        res = self.verify_task_batch_unique(gold, tag=tag)
        ok = sum(1 for task, _patch in gold if res.get(task["iid"], False))
        return ok, len(gold)
