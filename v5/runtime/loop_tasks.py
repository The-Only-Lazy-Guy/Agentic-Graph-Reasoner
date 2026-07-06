"""Task pool for the v3 agent loop: MBPP build tasks + synthesized mutation-debug tasks.

BUILD: MBPP spec ("Write a function to ...") + its assert tests. DEBUG: a gold solution
with ONE surviving mutation applied (mutant must COMPILE and FAIL >=1 test in the sandbox,
else discarded) — the agent must repair it. Both verified locally in ~100ms.

Split hygiene (deterministic, disjoint MBPP task ids; debug tasks derive ONLY from their
own split's golds — no cross-split gold leakage):
  lora_train 300 | pool_a 300 (memory learns) | pool_b 200 (compounding eval) | dev 100

  python -m v5.runtime.loop_tasks --fetch          # HF -> data/tasks/mbpp.jsonl (once)
  python -m v5.runtime.loop_tasks --build-pools    # splits + mutants -> data/tasks/pools.json
  python -m v5.runtime.loop_tasks --selftest       # no network
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

MBPP_CACHE = "data/tasks/mbpp.jsonl"
POOLS_PATH = "data/tasks/pools.json"
SPLITS = (("lora_train", 300), ("pool_a", 300), ("pool_b", 200), ("dev", 100))

# mutation operators: (name, pattern, replacement) — applied at ONE random match site
_MUTATIONS = [
    ("lt_le", r"(?<![<>=!])<(?![<=])", "<="),
    ("le_lt", r"<=", "<"),
    ("gt_ge", r"(?<![<>=!])>(?![>=])", ">="),
    ("ge_gt", r">=", ">"),
    ("eq_ne", r"==", "!="),
    ("plus1_drop", r"\+\s*1\b", "+ 0"),
    ("minus1_drop", r"-\s*1\b", "- 0"),
    ("range_off", r"\brange\(", "range(1, "),
    ("and_or", r"\band\b", "or"),
    ("or_and", r"\bor\b", "and"),
    ("true_false", r"\bTrue\b", "False"),
    ("zero_one", r"(?<![\w.])0(?![\w.])", "1"),
]


def fetch_mbpp(cache: str = MBPP_CACHE, log=print) -> int:
    """HF -> normalized jsonl cache (one-time, network)."""
    from datasets import load_dataset
    Path(cache).parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("google-research-datasets/mbpp", "full")
    n = 0
    with open(cache, "w", encoding="utf-8") as w:
        for split in ("train", "test", "validation", "prompt"):
            if split not in ds:
                continue
            for r in ds[split]:
                w.write(json.dumps({
                    "task_id": f"mbpp_{r['task_id']}",
                    "spec": r["text"], "code": r["code"],
                    "tests": list(r["test_list"]),
                    "setup": r.get("test_setup_code") or "",
                }, ensure_ascii=False) + "\n")
                n += 1
    log(f"  [tasks] cached {n} MBPP tasks -> {cache}")
    return n


def load_mbpp(cache: str = MBPP_CACHE) -> list[dict]:
    p = Path(cache)
    if not p.exists():
        raise SystemExit(f"{cache} missing — run: python -m v5.runtime.loop_tasks --fetch")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def mutate(code: str, rng: random.Random, sandbox_run, tests: list[str], setup: str = "",
           max_checks: int = 10) -> tuple[str, str] | None:
    """One SURVIVING mutant: compiles, and fails >=1 test (but not everything-crashes).
    Enumerates APPLICABLE ops in shuffled order — equivalent/green mutants rejected by
    actually running the tests. Returns (mutant, op_name) or None."""
    candidates = []
    for name, pat, rep in _MUTATIONS:
        for m in re.finditer(pat, code):
            candidates.append((name, pat, rep, m.start(), m.end()))
    rng.shuffle(candidates)
    checks = 0
    for name, pat, rep, s, e in candidates:
        if checks >= max_checks:
            break
        mutant = code[:s] + re.sub(pat, rep, code[s:e]) + code[e:]
        if mutant == code:
            continue
        try:
            compile(mutant, "<mutant>", "exec")
        except SyntaxError:
            continue
        checks += 1
        res = sandbox_run(mutant, tests, setup=setup, timeout=5)
        if res["first_fail"] == "timeout" or res["passed"]:
            continue                                   # still green (equivalent) or hangs
        if res["n_pass"] == 0 and "crash" in (res["first_fail"] or ""):
            continue                                   # too broken — trivial to spot
        return mutant, name
    return None


def build_pools(cache: str = MBPP_CACHE, out: str = POOLS_PATH, seed: int = 0,
                debug_frac: float = 0.5, log=print) -> dict:
    """Deterministic splits + per-split mutation-debug tasks (sandbox-verified)."""
    from v5.runtime.sandbox import run as sbx_run
    tasks = load_mbpp(cache)
    # keep only tasks whose GOLD passes locally (drops env-dependent flakes)
    good = [t for t in tasks if sbx_run(t["code"], t["tests"], setup=t["setup"])["passed"]]
    log(f"  [tasks] {len(good)}/{len(tasks)} golds pass locally")
    rng = random.Random(seed)
    rng.shuffle(good)
    pools: dict[str, list[dict]] = {}
    pos = 0
    for name, size in SPLITS:
        chunk = good[pos:pos + size]
        pos += size
        split_tasks = []
        for t in chunk:
            split_tasks.append({**t, "kind": "build"})
        n_debug = int(len(chunk) * debug_frac)
        for t in chunk[:n_debug]:
            got = mutate(t["code"], rng, sbx_run, t["tests"], t["setup"])
            if got is None:
                continue
            mutant, op = got
            split_tasks.append({"task_id": t["task_id"] + f"_dbg_{op}", "kind": "debug",
                                "spec": t["spec"], "code": mutant, "tests": t["tests"],
                                "setup": t["setup"], "gold": t["code"], "mutation": op})
        pools[name] = split_tasks
        log(f"  [tasks] {name}: {sum(1 for x in split_tasks if x['kind']=='build')} build "
            f"+ {sum(1 for x in split_tasks if x['kind']=='debug')} debug")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(pools), encoding="utf-8")
    ids = [t["task_id"].split("_dbg_")[0] for p in pools.values() for t in p]
    base_by_pool = [{t["task_id"].split("_dbg_")[0] for t in p} for p in pools.values()]
    for i in range(len(base_by_pool)):
        for j in range(i + 1, len(base_by_pool)):
            assert not (base_by_pool[i] & base_by_pool[j]), "split leak"
    log(f"  [tasks] pools -> {out} ({len(ids)} tasks total, splits disjoint)")
    return pools


def load_pools(path: str = POOLS_PATH) -> dict:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"{path} missing — run: python -m v5.runtime.loop_tasks --build-pools")
    return json.loads(p.read_text(encoding="utf-8"))


# ── selftest (no network) ───────────────────────────────────────────────────────

def _selftest() -> bool:
    from v5.runtime.sandbox import run as sbx_run
    print("loop_tasks --selftest: mutation generator + split hygiene (no network)\n")
    gold = ("def count_up(n):\n"
            "    out = []\n"
            "    for i in range(n):\n"
            "        out.append(i)\n"
            "    return out\n")
    tests = ["assert count_up(3) == [0, 1, 2]", "assert count_up(1) == [0]",
             "assert count_up(0) == []"]
    assert sbx_run(gold, tests)["passed"]
    rng = random.Random(0)
    got = mutate(gold, rng, sbx_run, tests)
    assert got is not None, "no surviving mutant found"
    mutant, op = got
    res = sbx_run(mutant, tests)
    assert not res["passed"] and res["n_pass"] < len(tests), (op, res)
    compile(mutant, "<m>", "exec")
    print(f"  [1] surviving mutant via '{op}' fails {len(tests)-res['n_pass']}/{len(tests)} -> PASS")

    # equivalent-mutant rejection: code with no matchable sites for most ops
    got2 = mutate("def id_(x):\n    return x\n", random.Random(1), sbx_run, ["assert id_(5) == 5"])
    assert got2 is None or not sbx_run(got2[0], ["assert id_(5) == 5"])["passed"]
    print("  [2] equivalent/green mutants rejected -> PASS")

    # split disjointness on a synthetic pool set
    fake = [{"task_id": f"mbpp_{i}", "spec": "s", "code": gold, "tests": tests, "setup": ""}
            for i in range(20)]
    rng = random.Random(0)
    rng.shuffle(fake)
    cuts = [("a", 8), ("b", 6), ("c", 6)]
    pos, seen = 0, set()
    for _n, sz in cuts:
        ids = {t["task_id"] for t in fake[pos:pos + sz]}
        assert not (ids & seen)
        seen |= ids
        pos += sz
    print("  [3] split disjointness -> PASS")
    print("\n  LOOP_TASKS SELFTEST -> PASS")
    return True


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--fetch", action="store_true", help="download MBPP -> jsonl cache")
    ap.add_argument("--build-pools", action="store_true", help="splits + verified mutants")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if _selftest() else 1)
    if a.fetch:
        fetch_mbpp()
    elif a.build_pools:
        build_pools(seed=a.seed)
    else:
        ap.print_help()
