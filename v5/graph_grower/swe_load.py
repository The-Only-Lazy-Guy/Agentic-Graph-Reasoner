"""Load SWE-bench instances + checkout repos for the v2 code-data pipeline.

DATA only (no training). See SWE_DATA_PIPELINE.md. Loads SWE-bench(_Lite/_Verified)
or SWE-gym via `datasets`, exposes instances, and shallow-checks-out a repo at its
base_commit (partial clone -> fetch the commit -> checkout) so `code_extract` can
parse the touched files.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

DATASETS = {
    "lite": "princeton-nlp/SWE-bench_Lite",
    "verified": "princeton-nlp/SWE-bench_Verified",
    "full": "princeton-nlp/SWE-bench",
    "gym": "SWE-Gym/SWE-Gym",
}
FIELDS = ["instance_id", "repo", "base_commit", "problem_statement", "patch",
          "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS"]
# small repos -> fast smoke
SMALL_REPOS = ["psf/requests", "pallets/flask"]


def load_instances(name: str = "lite", split: str = "test", limit: int = 0,
                   repos: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    from datasets import load_dataset
    ds = load_dataset(DATASETS.get(name, name), split=split)
    out: List[Dict[str, Any]] = []
    for row in ds:
        if repos and row["repo"] not in repos:
            continue
        out.append({k: row.get(k) for k in FIELDS})
        if limit and len(out) >= limit:
            break
    return out


def patch_files(patch: str) -> List[str]:
    """Files a unified diff touches (from `diff --git a/X b/X` headers)."""
    files: List[str] = []
    for m in re.finditer(r"^diff --git a/(.+?) b/(.+?)$", patch or "", flags=re.M):
        f = m.group(2)
        if f not in files:
            files.append(f)
    return files


def _run(cmd: List[str], cwd: Optional[str] = None, timeout: int = 900):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)


def checkout_repo(repo: str, base_commit: str, dest: str | Path, timeout: int = 900):
    """Shallow-checkout <repo>@<base_commit> into dest. Partial (blobless) clone for
    speed on big repos. Returns (ok, message)."""
    dest = Path(dest)
    url = f"https://github.com/{repo}.git"
    if not (dest / ".git").exists():
        dest.mkdir(parents=True, exist_ok=True)
        r = _run(["git", "clone", "--filter=blob:none", "--no-checkout", url, str(dest)],
                 timeout=timeout)
        if r.returncode != 0:
            return False, f"clone failed: {r.stderr[-300:]}"
    f = _run(["git", "fetch", "--depth", "1", "origin", base_commit], cwd=str(dest), timeout=timeout)
    if f.returncode != 0:
        return False, f"fetch failed: {f.stderr[-300:]}"
    c = _run(["git", "checkout", "-f", base_commit], cwd=str(dest), timeout=timeout)
    if c.returncode != 0:
        return False, f"checkout failed: {c.stderr[-300:]}"
    return True, "ok"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Load SWE-bench instances + checkout repos.")
    ap.add_argument("--dataset", default="lite", help="lite|verified|full|gym or a HF repo id")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--small", action="store_true", help="only small repos (fast smoke)")
    ap.add_argument("--checkout", default="", help="instance_id to checkout")
    ap.add_argument("--repo-root", default="data/swe_repos", help="where to checkout repos")
    ap.add_argument("--smoke", action="store_true",
                    help="load 1 small-repo instance, checkout, list its patch files")
    args = ap.parse_args(argv)

    repos = SMALL_REPOS if (args.small or args.smoke) else None
    insts = load_instances(args.dataset, args.split,
                           limit=(1 if args.smoke else args.limit), repos=repos)
    print(f"loaded {len(insts)} instances from {DATASETS.get(args.dataset, args.dataset)}")
    for i in insts:
        print(f"  {i['instance_id']:34} {i['repo']:22} files={patch_files(i['patch'])}")

    target = None
    if args.smoke and insts:
        target = insts[0]
    elif args.checkout:
        target = next((i for i in insts if i["instance_id"] == args.checkout), None)
        if target is None:
            insts = load_instances(args.dataset, args.split, repos=repos)
            target = next((i for i in insts if i["instance_id"] == args.checkout), None)

    if target:
        dest = Path(args.repo_root) / target["repo"].replace("/", "__")
        print(f"\ncheckout {target['instance_id']} ({target['repo']}@{target['base_commit'][:10]}) -> {dest}")
        ok, msg = checkout_repo(target["repo"], target["base_commit"], dest)
        print(f"  {'OK' if ok else 'FAILED'}: {msg}")
        if ok:
            files = patch_files(target["patch"])
            print(f"  patch touches {len(files)} files:")
            for f in files:
                p = dest / f
                print(f"    {'[exists]' if p.exists() else '[MISSING]'} {f}")
            print(f"\n  next: python -m v5.graph_grower.code_extract --repo-dir {dest} "
                  f"--repo {target['repo']} --files {' '.join(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
