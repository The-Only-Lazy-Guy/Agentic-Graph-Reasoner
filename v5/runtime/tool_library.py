"""Self-Extending Algorithm Library: the model solves a SEQUENCE of tasks by composing tools it
has already built, and INDUCES a new primitive only when the store lacks one. The library
compounds — a primitive induced for one task is REUSED for later tasks.

This is the "memory helps" thesis in the form that actually works (verified reusable ALGORITHMS,
not few-shot examples). Loop per task:

  1. give the model the task + the CURRENT store's tool signatures
  2. it writes a composite `solve(...)` that CALLS store tools (does not re-implement)
  3. scan its code for referenced registry tools; INDUCE any not yet in the store (tool_memory
     loop), add them; REUSE the ones already there
  4. VERIFY the composite by executing it with the referenced tools in scope; refine if wrong
  5. on success, the composite itself becomes a reusable tool in the store

Metric = distinct primitives INDUCED vs total REUSED across the sequence. If tasks share
primitives, induction happens once per primitive and reuse dominates -> the library compounds
(the graph memory of algorithms grows and pays off). Builds on tool_compose (induce + verify_fn).

  selftest (no model):  python -m v5.runtime.tool_library --selftest
  run (GPU):            V5_LM_TRUST_REMOTE_CODE=1 python -m v5.runtime.tool_library --size 3
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from v5.runtime.reason_rl import make_game, pure_ne, EVAL_SEEDS, TRAIN_SEEDS, batch_generate
from v5.runtime.tool_memory import _extract_code, _log, _ne_explanation
from v5.runtime.tool_compose import verify_fn, induce, _generic_diag, _nash_diag


# ═══════════════════════════════════════════════════════════════════════════════
# ORACLES  (ground truth for building verification cases)
# ═══════════════════════════════════════════════════════════════════════════════

def _o_nash(g):
    return list(g["ne"])


def _o_payoff(g, cell):
    i, j = cell
    return [g["row_pay"][i][j], g["col_pay"][i][j]]


def _o_pareto(g, cell):
    """Is `cell` Pareto-efficient — no other cell gives >= to BOTH players with one strict?"""
    R, C = g["row_pay"], g["col_pay"]
    m, n = g["size"], g["size"]
    ci, cj = cell
    for i in range(m):
        for j in range(n):
            if (i, j) == (ci, cj):
                continue
            if R[i][j] >= R[ci][cj] and C[i][j] >= C[ci][cj] and \
               (R[i][j] > R[ci][cj] or C[i][j] > C[ci][cj]):
                return False
    return True


def _cell_for(g):
    return (g["seed"] % g["size"], (g["seed"] // 2) % g["size"])


# ═══════════════════════════════════════════════════════════════════════════════
# PRIMITIVE REGISTRY  — specs the loop can induce on demand
# ═══════════════════════════════════════════════════════════════════════════════

PRIMS = {
    "nash_solver": dict(
        sig="nash_solver(R, C)", desc="the pure Nash equilibrium cell (i, j)",
        task=("Write `nash_solver(R, C)` returning the pure-strategy Nash equilibrium cell (i, j). "
              "R[i][j] is the ROW player's payoff, C[i][j] the COL player's. A cell is a NE iff "
              "R[i][j] is the max DOWN its COLUMN j AND C[i][j] is the max ACROSS its ROW i."),
        cases=lambda gs: [([g["row_pay"], g["col_pay"]], _o_nash(g)) for g in gs],
        diag=_nash_diag),
    "payoff": dict(
        sig="payoff(R, C, cell)", desc="the (row_payoff, col_payoff) at cell=(i, j)",
        task=("Write `payoff(R, C, cell)` where cell=(i, j). Return the tuple "
              "(R[i][j], C[i][j])."),
        cases=lambda gs: [([g["row_pay"], g["col_pay"], list(_cell_for(g))],
                           _o_payoff(g, _cell_for(g))) for g in gs],
        diag=_generic_diag),
    "pareto_efficient": dict(
        sig="pareto_efficient(R, C, cell)", desc="True iff cell is Pareto-efficient",
        task=("Write `pareto_efficient(R, C, cell)` where cell=(i, j). Return True iff NO other "
              "cell gives BOTH players a payoff >= the payoffs at cell with at least one strictly "
              "greater (i.e. cell is Pareto-optimal), else False."),
        cases=lambda gs: [([g["row_pay"], g["col_pay"], list(_cell_for(g))],
                           _o_pareto(g, _cell_for(g))) for g in gs],
        diag=_generic_diag),
}


# ═══════════════════════════════════════════════════════════════════════════════
# TASK SEQUENCE  — composites that SHARE primitives (so reuse can compound)
# ═══════════════════════════════════════════════════════════════════════════════

def _o_row_at_eq(g):
    i, j = g["ne"]; return g["row_pay"][i][j]


def _o_col_at_eq(g):
    i, j = g["ne"]; return g["col_pay"][i][j]


def _o_gap_at_eq(g):
    i, j = g["ne"]; return g["row_pay"][i][j] - g["col_pay"][i][j]


def _o_eq_efficient(g):
    return _o_pareto(g, g["ne"])


TASKS = [
    dict(name="row_payoff_at_eq",
         task="Write `row_payoff_at_eq(R, C)`: the ROW player's payoff AT the Nash equilibrium.",
         cases=lambda gs: [([g["row_pay"], g["col_pay"]], _o_row_at_eq(g)) for g in gs]),
    dict(name="col_payoff_at_eq",
         task="Write `col_payoff_at_eq(R, C)`: the COL player's payoff AT the Nash equilibrium.",
         cases=lambda gs: [([g["row_pay"], g["col_pay"]], _o_col_at_eq(g)) for g in gs]),
    dict(name="payoff_gap_at_eq",
         task="Write `payoff_gap_at_eq(R, C)`: (row payoff MINUS col payoff) AT the Nash equilibrium.",
         cases=lambda gs: [([g["row_pay"], g["col_pay"]], _o_gap_at_eq(g)) for g in gs]),
    dict(name="eq_is_efficient",
         task="Write `eq_is_efficient(R, C)`: True iff the Nash equilibrium cell is Pareto-efficient.",
         cases=lambda gs: [([g["row_pay"], g["col_pay"]], _o_eq_efficient(g)) for g in gs]),
]


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL STORE  (the graph memory of verified algorithms)
# ═══════════════════════════════════════════════════════════════════════════════

class ToolStore:
    def __init__(self):
        self.tools: dict[str, dict] = {}     # name -> {code, acc, sig, desc, calls}

    REUSE_THRESH = 0.8      # reuse a primitive that's good-enough rather than re-induce from scratch

    def has(self, name):
        return name in self.tools and self.tools[name]["acc"] >= self.REUSE_THRESH

    def add(self, name, code, acc, sig, desc, calls=None):
        self.tools[name] = dict(code=code, acc=acc, sig=sig, desc=desc, calls=calls or [])

    def deps_for(self, names):
        return "\n\n".join(self.tools[n]["code"] for n in names if n in self.tools)

    def signatures(self):
        return [(t["sig"], t["desc"]) for t in self.tools.values()]

    def save(self, path):
        """Persist verified tools to disk — the library is a MEMORY: a tool verified once (even
        in a prior run) is loaded and reused, never re-induced (dodges induction variance)."""
        import json
        from pathlib import Path
        d = Path(path); d.mkdir(parents=True, exist_ok=True)
        for name, t in self.tools.items():
            if t["acc"] >= self.REUSE_THRESH:      # only persist tools worth reusing (not junk)
                (d / f"{name}.json").write_text(json.dumps(t), encoding="utf-8")

    def load(self, path):
        import json
        from pathlib import Path
        d = Path(path)
        if not d.exists():
            return
        for f in d.glob("*.json"):
            try:
                self.tools[f.stem] = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                pass


def _compose_prompt(task_text, fn_name, catalog, built):
    """Advertise the CATALOG of tools the library can provide (the registry) — not just what's
    already built — so the model calls them from the first task (they get induced on first use,
    reused after). Without this the empty first-task store leaves nothing to call and the model
    just re-implements, and the library never bootstraps."""
    parts = [task_text]
    parts.append("\nYou can CALL these helper tools by name — they are (or will be) DEFINED for "
                 "you, so just call them, do NOT re-implement their logic:")
    for name, sig, desc in catalog:
        parts.append(f"  {sig}  -> {desc}" + ("  [already verified]" if name in built else ""))
    parts.append(f"\nWrite `{fn_name}(R, C)` by CALLING the helpers above wherever useful "
                 f"(build the answer FROM them, do not reinvent them). Output ONLY a Python code block.")
    return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# THE LIBRARY LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def run_library(model_name, size, rounds, samples, chunk, verify_n, eval_n,
                store_dir="artifacts/tool_store", fresh=False):
    from transformers import AutoTokenizer
    from v5.lm_loader import load_frozen_lm

    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    _log("  [lib] loading model...")
    model = load_frozen_lm(model_name); model.eval()
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    dev = next(model.parameters()).device
    vg = [make_game(s, size=size) for s in list(TRAIN_SEEDS)[:verify_n]]
    eg = [make_game(s, size=size) for s in list(EVAL_SEEDS)[:eval_n]]

    store = ToolStore()
    if not fresh:
        store.load(store_dir)
        if store.tools:
            _log(f"  [lib] loaded {len(store.tools)} tool(s) from {store_dir}: "
                 + ", ".join(f"{n}@{t['acc']:.0%}" for n, t in store.tools.items()))
    _log(f"SELF-EXTENDING LIBRARY on {size}x{size} | model={model_name} | {len(TASKS)} tasks "
         f"| store={store_dir}\n")

    catalog = [(n, PRIMS[n]["sig"], PRIMS[n]["desc"]) for n in PRIMS]   # tools the library CAN provide
    n_induced, n_reused, results = 0, 0, []

    for t in TASKS:
        name, task_text = t["name"], t["task"]
        _log(f"\n{'='*60}\n### TASK: {name}")
        built = {n for n in store.tools if n in PRIMS and store.has(n)}
        # 1. DRAFT: model proposes a composite calling catalog tools (advertised even if not built)
        draft = batch_generate(model, tok, [_compose_prompt(task_text, name, catalog, built)],
                               dev, max_new=360, sample=True, temperature=0.9, chunk=1)[0]
        draft_code = _extract_code(draft)
        refs = [p for p in PRIMS if (p + "(") in draft_code]
        _log(f"  drafted; references library tools: {refs or '(none — re-implemented inline)'}")

        # 2. FULFILL: induce any referenced primitive not yet in the store; reuse the rest
        for r in refs:
            if store.has(r):
                n_reused += 1
                _log(f"  REUSE  {r} (already verified {store.tools[r]['acc']:.0%})")
            else:
                spec = PRIMS[r]
                _log(f"  INDUCE {r} (new — not in library)")
                code, acc = induce(model, tok, dev, r, spec["task"], spec["cases"](vg),
                                   spec["diag"], rounds=rounds, samples=samples, chunk=chunk,
                                   eval_cases=spec["cases"](eg))
                store.add(r, code, acc, spec["sig"], spec["desc"])
                store.save(store_dir)             # persist immediately — build once, reuse forever
                n_induced += 1

        # 3. COMPOSE + VERIFY: induce the composite, SEEDED FROM THE DRAFT (which already calls the
        # tools) so the loop refines a tool-calling version instead of re-deriving one that doesn't.
        deps = store.deps_for([r for r in refs if store.has(r)])
        avail = [(PRIMS[r]["sig"], PRIMS[r]["desc"]) for r in refs if store.has(r)]
        code, acc = induce(model, tok, dev, name, task_text, t["cases"](vg), _generic_diag,
                           avail=avail, deps_code=deps, rounds=rounds, samples=samples,
                           chunk=chunk, eval_cases=t["cases"](eg), seed_code=draft_code)
        calls = [r for r in PRIMS if code and (r + "(") in code]
        ev, _, _ = verify_fn(code, name, t["cases"](eg), store.deps_for(calls)) if code else (0.0, [], "")
        if acc >= 0.999:
            store.add(name, code, acc, f"{name}(R, C)", task_text[:50], calls=calls)
            store.save(store_dir)
        results.append((name, ev, calls))
        _log(f"  => {name}: held-out {ev:.0%}, calls {calls or 'NOTHING (re-implemented)'}")

    # ── report: did the library COMPOUND? ───────────────────────────────────────
    _log("\n" + "=" * 60 + "\n=== SELF-EXTENDING LIBRARY RESULT ===")
    prims_in_store = [n for n in store.tools if n in PRIMS]
    n_distinct = len(prims_in_store)
    _log(f"  tasks solved (100%): {sum(1 for _, ev, _ in results if ev >= 0.999)}/{len(TASKS)}")
    _log(f"  primitives INDUCED (distinct): {n_distinct}  -> {prims_in_store}  "
         f"({n_induced} induction events)")
    _log(f"  primitive REUSES across tasks: {n_reused}")
    _log(f"  library grew to {len(store.tools)} tools ({n_distinct} primitives + "
         f"{len(store.tools)-n_distinct} composites)")
    _log(f"  COMPOUNDING: {n_reused} reuses on {n_distinct} distinct primitives "
         f"(=> {'the library pays off — build once, reuse many' if n_reused > n_distinct else 'little reuse'})")
    for name, ev, calls in results:
        _log(f"    {name:22} {ev:.0%}  calls={calls}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST (no model) — oracles + composition-by-calling across the task family
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    print("tool_library --selftest: oracles + store + reuse plumbing (no model)\n")
    games = [make_game(s, size=3) for s in range(20)]

    # correct primitives
    nash = ("def nash_solver(R, C):\n    m,n=len(R),len(R[0])\n"
            "    for i in range(m):\n        for j in range(n):\n"
            "            if R[i][j]==max(R[r][j] for r in range(m)) and C[i][j]==max(C[i][c] for c in range(n)):\n"
            "                return (i,j)\n")
    pay = "def payoff(R, C, cell):\n    i,j=cell\n    return (R[i][j], C[i][j])\n"
    par = ("def pareto_efficient(R, C, cell):\n    ci,cj=cell\n    m,n=len(R),len(R[0])\n"
           "    for i in range(m):\n        for j in range(n):\n"
           "            if (i,j)==(ci,cj): continue\n"
           "            if R[i][j]>=R[ci][cj] and C[i][j]>=C[ci][cj] and (R[i][j]>R[ci][cj] or C[i][j]>C[ci][cj]):\n"
           "                return False\n    return True\n")
    for nm, code in [("nash_solver", nash), ("payoff", pay), ("pareto_efficient", par)]:
        acc, _, err = verify_fn(code, nm, PRIMS[nm]["cases"](games))
        assert acc == 1.0, f"{nm} oracle mismatch: {acc:.0%} {err}"
    print("  [1] all 3 primitive oracles verify 100% -> PASS")

    # store: add primitives, reuse across tasks
    store = ToolStore()
    store.add("nash_solver", nash, 1.0, PRIMS["nash_solver"]["sig"], "")
    store.add("payoff", pay, 1.0, PRIMS["payoff"]["sig"], "")
    assert store.has("nash_solver") and store.has("payoff")
    print("  [2] ToolStore add/has/deps -> PASS")

    # composites over the SHARED primitives verify with store deps
    comps = {
        "row_payoff_at_eq": "def row_payoff_at_eq(R, C):\n    return payoff(R, C, nash_solver(R, C))[0]\n",
        "col_payoff_at_eq": "def col_payoff_at_eq(R, C):\n    return payoff(R, C, nash_solver(R, C))[1]\n",
        "payoff_gap_at_eq": ("def payoff_gap_at_eq(R, C):\n    a,b=payoff(R,C,nash_solver(R,C))\n"
                             "    return a-b\n"),
    }
    deps = store.deps_for(["nash_solver", "payoff"])
    for t in TASKS:
        if t["name"] in comps:
            acc, _, err = verify_fn(comps[t["name"]], t["name"], t["cases"](games), deps)
            assert acc == 1.0, f"{t['name']} composite mismatch: {acc:.0%} {err}"
    print("  [3] 3 composites REUSE {nash_solver, payoff} and verify 100% -> PASS")

    # eq_is_efficient composes nash + pareto
    store.add("pareto_efficient", par, 1.0, PRIMS["pareto_efficient"]["sig"], "")
    eff = "def eq_is_efficient(R, C):\n    return pareto_efficient(R, C, nash_solver(R, C))\n"
    acc, _, err = verify_fn(eff, "eq_is_efficient", TASKS[3]["cases"](games),
                            store.deps_for(["nash_solver", "pareto_efficient"]))
    assert acc == 1.0, f"eq_is_efficient mismatch: {acc:.0%} {err}"
    print("  [4] eq_is_efficient composes {nash_solver, pareto_efficient} -> 100% -> PASS")

    print("\n  TOOL_LIBRARY SELFTEST -> PASS")
    return True


def main():
    ap = argparse.ArgumentParser(description="Self-extending algorithm library (induce + reuse + compose).")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--size", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--verify-n", type=int, default=40)
    ap.add_argument("--eval-n", type=int, default=40)
    ap.add_argument("--store-dir", default="artifacts/tool_store",
                    help="persist verified tools here; a later run loads + reuses them (the memory)")
    ap.add_argument("--fresh", action="store_true", help="ignore any persisted tools, start empty")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    run_library(a.model, a.size, a.rounds, a.samples, a.chunk, a.verify_n, a.eval_n,
                a.store_dir, a.fresh)


if __name__ == "__main__":
    main()
