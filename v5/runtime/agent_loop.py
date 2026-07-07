"""v3 AGENT LOOP — propose -> sandbox-verify -> feedback retry, with TotalMemory in the
trace slot. The falsifiable core of "graph elevates the small model":

  arms (same proposer LoRA, same tasks — ONLY the trace-slot content differs):
    off      empty slot
    flat     nearest implementation over ALL of L1 (no concepts)
    concept  two-hop L2->L1 + local-fit (the designed memory)

  gates:
    GM0  arm=off dev solve-rate (loop sanity)
    GM1  concept > flat > off on pool_a (concept - flat >= +3pp)
    GM2  compounding: pool_a with --write-back into a copied memory root, then pool_b
         with experienced vs fresh root (>= +3pp or -0.3 attempts/solve)
    GS   memory wall-clock overhead <= 15%; payload tokens/step reported

Prompt = DATA only (zero instructions), extending the validated SEP format:
  BUILD: spec + first-assert + SEP_T + [memory payload | obs] + SEP_N -> function code
  DEBUG: spec + buggy code   + SEP_T + [memory payload | obs] + SEP_N -> fixed code
Retry feedback = sandbox obs_text (failing assert, counts) appended in the slot — data.

  python -m v5.runtime.agent_loop --selftest                  # no model
  python -m v5.runtime.agent_loop --smoke                     # 0.5B local end-to-end
  python -m v5.runtime.agent_loop --train-lora                # proposer (molab)
  python -m v5.runtime.agent_loop --run --arm off --pool dev  # GM0 (molab)
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from v5.runtime.lggn_realizer import SEP_N, SEP_T, RawLM
from v5.runtime.loop_tasks import load_pools
from v5.runtime.sandbox import obs_text, run as sbx_run

LORA_DIR = "artifacts/loop_lora"
RESULTS_PATH = "artifacts/loop_results.json"
PAYLOAD_CAP = 1200          # chars of memory+obs in the slot (~300 tok GS budget)


def task_data(task: dict) -> str:
    if task["kind"] == "build":
        return task["spec"] + "\n" + (task["tests"][0] if task["tests"] else "")
    return task["spec"] + "\n" + task["code"]                  # buggy code is data


def build_prompt(task: dict, payload: str) -> str:
    return task_data(task) + SEP_T + (payload or "")[:PAYLOAD_CAP] + SEP_N


def gold_of(task: dict) -> str:
    return task["gold"] if task["kind"] == "debug" else task["code"]


def _save_results(update: dict, path: str = RESULTS_PATH) -> dict:
    p = Path(path)
    merged = {}
    if p.exists():
        try:
            merged = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            merged = {}
    merged.update(update)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return merged


# ── proposer training ───────────────────────────────────────────────────────────

def _neighbor_payload(task: dict) -> str:
    """A neighbor task rendered exactly like a TotalMemory payload (trace + old=>new),
    so train-time slots match eval-time retrieved slots."""
    return (task["spec"] or "").strip()[:400] + "\n\n=>\n" + gold_of(task)[:350]


def train_lora(model_name: str, pools: dict, out_dir: str = LORA_DIR, epochs: int = 2,
               batch_size: int = 8, max_tokens: int = 1024, augment: bool = True,
               log=print) -> None:
    """One shared task-format LoRA on the lora_train split. Pair mixture:
      empty slot            (1x/task)   — the off-arm distribution
      obs-conditioned       (debug)     — the retry distribution
      RELEVANT neighbor slot(1x/task)   — nearest OTHER train task's solution as payload:
                                          learn to USE retrieved memory
      JUNK slot             (~0.5x/task)— a far task's payload, target unchanged:
                                          learn to IGNORE irrelevant memory
    GM1 first run proved the empty-slot-only proposer craters when the slot is filled
    (0.617 -> 0.18): slot content must be IN the training distribution."""
    train = pools["lora_train"]
    pairs = []
    for t in train:
        pairs.append((build_prompt(t, ""), gold_of(t)))
        if t["kind"] == "debug":
            res = sbx_run(t["code"], t["tests"], setup=t.get("setup", ""))
            o = obs_text(res)
            if o:
                pairs.append((build_prompt(t, o), gold_of(t)))
    if augment:
        import numpy as np
        import random as _rng
        from v5.memory.store import make_mpnet_embedder
        embed = make_mpnet_embedder()
        vecs = embed({str(i): t["spec"] for i, t in enumerate(train)})
        M = np.asarray([vecs[str(i)] for i in range(len(train))], dtype=np.float32)
        sims = M @ M.T
        np.fill_diagonal(sims, -np.inf)
        rng = _rng.Random(7)
        for i, t in enumerate(train):
            nbr = int(np.argmax(sims[i]))
            pairs.append((build_prompt(t, _neighbor_payload(train[nbr])), gold_of(t)))
            if rng.random() < 0.5:
                far = int(np.argsort(sims[i])[len(train) // 10])   # bottom-decile similarity
                pairs.append((build_prompt(t, _neighbor_payload(train[far])), gold_of(t)))
    log(f"  [lora] {len(pairs)} pairs from {len(train)} tasks (augment={augment})")
    lm = RawLM(model_name)
    lm.train_on(pairs, epochs=epochs, batch_size=batch_size, max_tokens=max_tokens, log=log)
    lm.save_checkpoint(out_dir)
    log(f"  [lora] checkpoint -> {out_dir}")
    lm.cleanup()


# ── the loop (wave-batched: attempt k for all still-failing tasks in one batch) ──

def run_arm(lm, tasks: list[dict], memory, budget: int = 3, max_new: int = 384,
            eval_batch: int = 8, write_back: bool = False, log=print) -> dict:
    t_wall = time.time()
    state = []
    for t in tasks:
        hit = memory.read(goal=t["spec"], span=task_data(t)) if memory is not None \
            else None
        state.append({"task": t, "payload": (hit.trace_text if hit else ""), "obs": "",
                      "attempts": 0, "passed": False, "code": "", "tokens": 0,
                      "mem_tokens": (hit.tokens_est if hit else 0)})
    pending = list(range(len(state)))
    for att in range(budget):
        if not pending:
            break
        for b0 in range(0, len(pending), eval_batch):
            chunk = pending[b0:b0 + eval_batch]
            prompts = []
            for j in chunk:
                s = state[j]
                slot = (s["payload"] + ("\n" + s["obs"] if s["obs"] else "")).strip()
                prompts.append(build_prompt(s["task"], slot))
            outs = lm.generate_raw_batch(prompts, max_new_tokens=max_new)
            for j, gen, pr in zip(chunk, outs, prompts):
                s = state[j]
                s["attempts"] += 1
                s["tokens"] += len(pr) // 4
                s["code"] = gen
                res = sbx_run(gen, s["task"]["tests"], setup=s["task"].get("setup", ""))
                s["passed"] = res["passed"]
                s["obs"] = obs_text(res)
        solved_now = [j for j in pending if state[j]["passed"]]
        pending = [j for j in pending if not state[j]["passed"]]
        log(f"      attempt {att+1}: solved {len(solved_now)}, pending {len(pending)}")
    if write_back and memory is not None:
        for s in state:
            t = s["task"]
            memory.write(goal=t["spec"], old=(t["code"] if t["kind"] == "debug" else ""),
                         new=s["code"], trace=(s["payload"] or t["spec"])[:400],
                         verified=s["passed"], task_id=t["task_id"])
    n = len(state)
    solved = sum(1 for s in state if s["passed"])
    out = {
        "n": n, "solved": solved, "solve_rate": solved / max(1, n),
        "attempts_per_solve": (sum(s["attempts"] for s in state if s["passed"]) /
                               max(1, solved)),
        "mean_prompt_tokens": sum(s["tokens"] for s in state) / max(1, n),
        "mean_mem_tokens": sum(s["mem_tokens"] for s in state) / max(1, n),
        "wall_s": round(time.time() - t_wall, 1),
        "by_kind": {k: [sum(1 for s in state if s["task"]["kind"] == k and s["passed"]),
                        sum(1 for s in state if s["task"]["kind"] == k)]
                    for k in ("build", "debug")},
    }
    return out


def _memory_for(arm: str, root: str, embed_fn):
    if arm == "off":
        return None
    from v5.memory.memory import TotalMemory
    return TotalMemory(root, mode=("flat" if arm == "flat" else "concept"),
                       embed_fn=embed_fn)


def _report(results: dict, log=print) -> None:
    log("\n=== AGENT LOOP (solve-rate; same proposer, slot content is the only variable) ===")
    for key in sorted(results):
        r = results[key]
        if not isinstance(r, dict) or "solve_rate" not in r:
            continue
        bk = r.get("by_kind", {})
        log(f"  {key:34} solve {r['solved']}/{r['n']} = {r['solve_rate']:.3f}  "
            f"att/solve {r['attempts_per_solve']:.2f}  tok {r['mean_prompt_tokens']:.0f} "
            f"(mem {r['mean_mem_tokens']:.0f})  wall {r['wall_s']}s  "
            f"build {bk.get('build', '?')} debug {bk.get('debug', '?')}")

    def sr(key):
        return results[key]["solve_rate"] if key in results else None

    for pool in ("dev", "pool_a", "pool_b"):
        off, flat, con = (sr(f"{pool}:off"), sr(f"{pool}:flat"), sr(f"{pool}:concept"))
        if off is not None and con is not None:
            line = f"  [{pool}] concept-off {con-off:+.3f}"
            if flat is not None:
                line += f" | concept-flat {con-flat:+.3f} | flat-off {flat-off:+.3f}"
                verdict = "PASS" if (con - flat) >= 0.03 else "FAIL"
                line += f"  -> GM1 {verdict}"
            log(line)
        fresh, exp = sr(f"{pool}:concept"), sr(f"{pool}:concept_exp")
        if fresh is not None and exp is not None:
            log(f"  [{pool}] experienced-fresh {exp-fresh:+.3f}  -> GM2 "
                f"{'PASS' if (exp - fresh) >= 0.03 else 'FAIL'}")
        if off is not None and con is not None and f"{pool}:off" in results:
            w_off, w_con = results[f"{pool}:off"]["wall_s"], results[f"{pool}:concept"]["wall_s"]
            if w_off > 0:
                over = (w_con - w_off) / w_off
                log(f"  [{pool}] GS overhead {over:+.1%} "
                    f"-> {'PASS' if over <= 0.15 else 'FAIL'}")


# ── selftest (no model) ─────────────────────────────────────────────────────────

class _StubLM:
    """Echoes gold for tasks whose spec contains 'easy'; garbage otherwise; on retry
    (obs present in prompt) solves 'medium' too — exercises the wave bookkeeping."""

    def __init__(self, golds: dict):
        self.golds = golds

    def generate_raw_batch(self, prompts, max_new_tokens=0, **kw):
        outs = []
        for p in prompts:
            spec = p.split(SEP_T)[0]
            key = next((k for k in self.golds if k in spec), None)
            has_obs = "tests pass" in p or "test" in p.split(SEP_T)[1]
            if key and ("easy" in spec or (has_obs and "medium" in spec)):
                outs.append(self.golds[key])
            else:
                outs.append("def broken(:\n")

        return outs


def _selftest() -> bool:
    print("agent_loop --selftest: prompt assembly, wave retries, write-back plumbing (no model)\n")
    gold = "def add(a, b):\n    return a + b\n"
    tasks = [
        {"task_id": "t_easy", "kind": "build", "spec": "easy: add two numbers KEY_A",
         "code": gold, "tests": ["assert add(1, 2) == 3"], "setup": ""},
        {"task_id": "t_med", "kind": "debug", "spec": "medium: add two numbers KEY_A",
         "code": "def add(a, b):\n    return a - b\n", "tests": ["assert add(1, 2) == 3"],
         "setup": "", "gold": gold},
        {"task_id": "t_hard", "kind": "build", "spec": "hard: unsolvable KEY_B",
         "code": gold, "tests": ["assert add(9, 9) == 0"], "setup": ""},
    ]
    p = build_prompt(tasks[1], "some payload")
    assert p.startswith(tasks[1]["spec"]) and SEP_T in p and p.endswith(SEP_N)
    assert tasks[1]["code"] in p, "debug prompt carries buggy code as data"
    print("  [1] prompt assembly -> PASS")

    lm = _StubLM({"KEY_A": gold})
    res = run_arm(lm, tasks, memory=None, budget=2, eval_batch=2, log=lambda *a: None)
    assert res["n"] == 3 and res["solved"] == 2, res
    assert res["by_kind"]["build"] == [1, 2] and res["by_kind"]["debug"] == [1, 1], res
    assert res["attempts_per_solve"] > 1.0, "medium needed the obs retry"
    print("  [2] wave retries: easy@1, medium@2-with-obs, hard never -> PASS")

    import tempfile
    from v5.memory.memory import TotalMemory
    from v5.memory.store import make_fake_embedder
    with tempfile.TemporaryDirectory() as td:
        tm = TotalMemory(td, mode="flat", embed_fn=make_fake_embedder())
        res2 = run_arm(_StubLM({"KEY_A": gold}), tasks, memory=tm, budget=2,
                       eval_batch=2, write_back=True, log=lambda *a: None)
        assert res2["solved"] == 2
        assert len(tm.impls) >= 2, "write-back stored outcomes"
        vs = {r["verified"] for r in tm.impls.records.values()}
        assert "strong" in vs and "fail" in vs, "both outcome polarities recorded"
    print("  [3] write-back plumbing -> PASS")

    rep = {}
    rep["dev:off"] = dict(n=10, solved=4, solve_rate=0.4, attempts_per_solve=1.5,
                          mean_prompt_tokens=100, mean_mem_tokens=0, wall_s=10,
                          by_kind={"build": [3, 6], "debug": [1, 4]})
    rep["dev:flat"] = dict(rep["dev:off"], solved=5, solve_rate=0.5)
    rep["dev:concept"] = dict(rep["dev:off"], solved=6, solve_rate=0.6, wall_s=11)
    _report(rep, log=lambda *a: None)
    print("  [4] report smoke -> PASS")
    print("\n  AGENT_LOOP SELFTEST -> PASS")
    return True


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main():
    import sys
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="v3 agent loop — propose/verify/retry + memory arms.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="0.5B, 8 dev tasks, arms off+concept")
    ap.add_argument("--train-lora", action="store_true")
    ap.add_argument("--no-augment", action="store_true", help="ablation: empty-slot-only LoRA")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--lora", default=LORA_DIR)
    ap.add_argument("--arm", choices=["off", "flat", "concept"], default="off")
    ap.add_argument("--pool", choices=["dev", "pool_a", "pool_b", "lora_train"], default="dev")
    ap.add_argument("--n", type=int, default=0, help="cap tasks (0=all)")
    ap.add_argument("--budget", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--eval-batch", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--memory-root", default="data/memory")
    ap.add_argument("--write-back", action="store_true")
    ap.add_argument("--copy-memory-to", default="",
                    help="GM2: copy memory root here first and write into the copy")
    ap.add_argument("--result-key", default="", help="override results key (e.g. pool_b:concept_exp)")
    a = ap.parse_args()

    if a.selftest:
        raise SystemExit(0 if _selftest() else 1)

    pools = load_pools()
    if a.train_lora:
        train_lora(a.model, pools, out_dir=a.lora, epochs=a.epochs, batch_size=a.batch_size,
                   augment=not a.no_augment)
        return

    if a.smoke:
        a.model = "Qwen/Qwen2.5-0.5B" if a.model == "Qwen/Qwen2.5-3B" else a.model
        tasks = pools["dev"][:8]
        print(f"[SMOKE] model={a.model} tasks={len(tasks)} arms off+concept (tiny inline LoRA)")
        lm = RawLM(a.model)
        pairs = [(build_prompt(t, ""), gold_of(t)) for t in pools["lora_train"][:24]]
        lm.train_on(pairs, epochs=1, batch_size=2, log=print)
        from v5.memory.store import make_mpnet_embedder
        for arm in ("off", "concept"):
            mem = _memory_for(arm, a.memory_root, make_mpnet_embedder())
            r = run_arm(lm, tasks, mem, budget=2, max_new=256, eval_batch=2, log=print)
            print(f"  [smoke:{arm}] {r['solved']}/{r['n']} wall={r['wall_s']}s "
                  f"mem_tok={r['mean_mem_tokens']:.0f}")
        lm.cleanup()
        return

    if a.run:
        root = a.memory_root
        if a.copy_memory_to:
            dst = Path(a.copy_memory_to)
            if not dst.exists():
                shutil.copytree(a.memory_root, dst)
                print(f"  [gm2] memory copied {a.memory_root} -> {dst}")
            root = str(dst)
        tasks = pools[a.pool]
        if a.n:
            tasks = tasks[:a.n]
        print(f"[loop] model={a.model} lora={a.lora} arm={a.arm} pool={a.pool} "
              f"n={len(tasks)} budget={a.budget} write_back={a.write_back} root={root}")
        lm = RawLM.load_checkpoint(a.model, a.lora)
        embed_fn = None
        if a.arm != "off":
            from v5.memory.store import make_mpnet_embedder
            embed_fn = make_mpnet_embedder()
        mem = _memory_for(a.arm, root, embed_fn)
        res = run_arm(lm, tasks, mem, budget=a.budget, max_new=a.max_new,
                      eval_batch=a.eval_batch, write_back=a.write_back, log=print)
        key = a.result_key or f"{a.pool}:{a.arm}"
        merged = _save_results({key: res})
        _report(merged, log=print)
        lm.cleanup()
        return
    ap.print_help()


if __name__ == "__main__":
    main()
