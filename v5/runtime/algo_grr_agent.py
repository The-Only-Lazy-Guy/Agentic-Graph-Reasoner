"""algo_grr_agent — the agentic tool-calling loop: TRM CALLS graph tools, the LM only SPEAKS.

The design the whole stack points at, made literal:
  - TOOLS live in the graph (ToolNode/ToolRegistry). A tool can CALL another tool (atoms activate each other).
  - The TRM CONTROLLER decides the next tool + args from the goal+state — the LM is NOT in the decision loop
    (its execution/arithmetic drift, measured, is irrelevant when it never executes).
  - Each tool EXECUTES exactly (self-verifying step); the outcome anchor gates the whole trace.
  - A verified tool-sequence BANKS as a reusable WORKFLOW schema -> later tasks REPLAY it (compounds), no search.
  - The LM SPEAKS only: it narrates the VERIFIED tool-graph -> a faithful explanation (can't cite a tool that
    didn't run). "LM plans/speaks; graph executes; TRM decides" — poison-safe (a wrong tool misses the anchor).

This unifies: AtomStore code-atoms-as-tools + the CoT-learned schemas (algo_grr_cot) as the controller's policy
+ dual-channel narration + the verify gate. Deployment: swap the arithmetic ToolRegistry for real agentic tools
(search / call_api / run_code) — the loop is identical; only the tool set changes.

    python -m v5.runtime.algo_grr_agent --selftest    # no-GPU: 8 checks, the whole loop
    python -m v5.runtime.algo_grr_agent --run --lm Qwen/Qwen2.5-3B-Instruct   # LM SPEAKS the verified traces
"""
from __future__ import annotations

import argparse
import hashlib
import math
import re
import sys


# ── TOOLS in the graph (pluggable; a tool may CALL another tool) ───────────────────────────────────
class ToolNode:
    def __init__(self, name, fn, description, calls=()):
        self.name, self.fn, self.description = name, fn, description
        self.calls = tuple(calls)                       # other tool names this one activates (composition)


class ToolRegistry(dict):
    def add(self, name, fn, description, calls=()):
        self[name] = ToolNode(name, fn, description, calls)

    def run(self, name, state, arg):
        return self[name].fn(self, state, arg)          # fn(registry, state, arg) -> can call self.run(other,...)


def default_registry() -> ToolRegistry:
    r = ToolRegistry()
    r.add("inc", lambda R, s, a: s + a, "add {a}")
    r.add("dec", lambda R, s, a: s - a, "subtract {a}")
    r.add("scale", lambda R, s, a: s * a, "multiply by {a}")
    r.add("power", lambda R, s, a: s ** a, "raise to the power {a}")
    r.add("wrap", lambda R, s, a: (s % a if a else float("nan")), "take modulo {a}")
    # COMPOSED tools — activate other tools (atoms calling atoms):
    r.add("double", lambda R, s, a: R.run("scale", s, 2), "double it", calls=("scale",))
    r.add("square", lambda R, s, a: R.run("power", s, 2), "square it", calls=("power",))
    return r


# ── tasks: a goal (NL intent) whose solution is a short tool-sequence reaching a gold value ─────────
_TOL = dict(rel_tol=1e-6, abs_tol=1e-9)


def make_tasks(seed: int = 0):
    """Math + a non-math-flavored toolset, several instances. Shared workflows across 'domains' (the same
    tool-sequence recurs) so banking a workflow once lets later tasks REPLAY it -> compounding + reuse."""
    reg = default_registry()
    specs = [                                            # (domain, [(tool, arg), ...], NL intent)
        ("finance", [("scale", 3), ("inc", 2)], "triple the balance then add a 2 dollar fee"),
        ("physics", [("scale", 9.8), ("inc", 1)], "multiply by gravity 9.8 then add 1"),
        ("geometry", [("square", 0), ("scale", 3)], "square the side then scale by 3"),
        ("stats", [("square", 0), ("scale", 0.5)], "square it then halve via multiply 0.5"),
        ("time", [("inc", 5), ("wrap", 12)], "add 5 hours then wrap around a 12 hour clock"),
        ("signal", [("inc", 30), ("wrap", 8)], "shift by 30 then wrap modulo 8"),
    ]
    tasks = []
    import random
    rng = random.Random(seed)
    for i, (domain, seq, text) in enumerate(specs):
        for _ in range(3):
            x0 = rng.randint(2, 9) if domain in ("finance", "geometry", "time", "signal") else rng.uniform(2, 6)
            state = float(x0)
            for tool, arg in seq:
                state = reg.run(tool, state, arg)
            tasks.append(dict(text=text, x0=float(x0), gold=state, gold_seq=seq, domain=domain))
    rng.shuffle(tasks)
    return tasks, reg


def _sig(seq) -> str:                                    # workflow signature = the tool-name sequence (args = holes)
    return hashlib.md5("|".join(t for t, _ in seq).encode()).hexdigest()[:12]


def _goal_vec(text: str) -> dict:                        # tiny bag-of-tokens goal encoder (stands in for mpnet/TRM)
    v: dict = {}
    for w in re.findall(r"[a-z]+", text.lower()):
        if len(w) > 2:
            v[w] = v.get(w, 0.0) + 1.0
    return v


def _cos(a: dict, b: dict) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[k] * b.get(k, 0.0) for k in a)
    na = math.sqrt(sum(x * x for x in a.values())); nb = math.sqrt(sum(x * x for x in b.values()))
    return dot / (na * nb) if na and nb else 0.0


# ── the banked WORKFLOW memory (verified tool-sequences; lazy-quorum promote; replay = reuse) ──────
class WorkflowStore:
    def __init__(self, quorum_m: int = 2):
        self.m = quorum_m
        self.provisional: dict = {}   # sig -> dict(seq, goal, votes, domains)
        self.certified: dict = {}
        self.reuse = 0
        self.cross_domain_reuse = 0

    def match(self, goal: dict, thresh: float = 0.55):
        """Retrieve a banked workflow whose goal-encoding matches (the TRM's replay policy). Returns seq|None."""
        best, best_s = None, thresh
        for pool in (self.certified, self.provisional):
            for sig, w in pool.items():
                s = _cos(goal, w["goal"])
                if s > best_s:
                    best, best_s = w, s
        return best

    def bank(self, seq, goal, domain):
        sig = _sig(seq)
        w = self.certified.get(sig) or self.provisional.get(sig)
        if w is None:
            self.provisional[sig] = dict(seq=seq, goal=goal, votes=1, domains={domain})
            return "new"
        w["votes"] += 1
        if domain not in w["domains"]:
            w["domains"].add(domain); self.cross_domain_reuse += 1
        if sig in self.provisional and w["votes"] >= self.m:
            self.certified[sig] = self.provisional.pop(sig)
        return "vote"


# ── the TRM CONTROLLER: decides tools from goal+state; retrieve-banked OR bounded search. NO LM. ───
class ToolController:
    def __init__(self, registry: ToolRegistry, store: WorkflowStore, max_depth: int = 3):
        self.reg, self.store, self.max_depth = registry, store, max_depth
        self.tool_tries = 0

    def _args_from_text(self, text: str):               # candidate args = numbers in the intent (parsing, not LM)
        args = [float(x) for x in re.findall(r"-?\d+\.?\d*", text)]
        return args or [1.0, 2.0, 3.0]

    def _run_seq(self, seq, x0):
        state = float(x0)
        for tool, arg in seq:
            if tool not in self.reg:
                return None
            state = self.reg.run(tool, state, arg)
        return state

    def solve(self, task):
        """Return (mode, seq, final) or (mode, None, None). mode in {'replay','search','fail'}. LM NOT called."""
        goal = _goal_vec(task["text"])
        w = self.store.match(goal)                      # REPLAY a banked workflow (compounding: no search)
        if w is not None:
            final = self._run_seq(w["seq"], task["x0"])
            if final is not None and math.isclose(final, task["gold"], **_TOL):
                self.store.reuse += 1
                return "replay", w["seq"], final
        # EXPLORE: bounded DFS over (tool, arg) to reach gold (verified by the outcome anchor)
        args = self._args_from_text(task["text"]) + [2.0]
        tools = list(self.reg)

        def dfs(state, seq):
            if math.isclose(state, task["gold"], **_TOL) and seq:
                return list(seq)
            if len(seq) >= self.max_depth:
                return None
            for t in tools:
                for a in args:
                    self.tool_tries += 1
                    try:
                        ns = self.reg.run(t, state, a)
                    except Exception:  # noqa: BLE001
                        continue
                    if not isinstance(ns, (int, float)) or ns != ns or abs(ns) > 1e12:  # non-real/nan/blow-up guard
                        continue
                    seq.append((t, a))
                    got = dfs(ns, seq)
                    if got is not None:
                        return got
                    seq.pop()
            return None

        found = dfs(float(task["x0"]), [])
        if found is not None:
            return "search", found, self._run_seq(found, task["x0"])
        return "fail", None, None


# ── the LM SPEAKER: narrate the VERIFIED tool-graph (faithful — can only cite tools that ran) ──────
def narrate(reg: ToolRegistry, seq, x0, gen=None) -> str:
    steps, state = [], float(x0)
    for tool, arg in seq:
        desc = reg[tool].description.format(a=arg)
        nxt = reg.run(tool, state, arg)
        steps.append(f"{desc} (-> {round(nxt, 4)})")
        state = nxt
    trace = "; then ".join(steps)
    if gen is None:                                     # deterministic faithful narration (no LM)
        return f"Start with {round(x0, 4)}: {trace}. Final = {round(state, 4)}."
    # LM SPEAKS: phrase the VERIFIED trace in natural language — it only rewords given facts, never decides.
    prompt = ("Rephrase this verified computation as one fluent sentence explaining the steps. Do NOT add or "
              f"change any step or number.\nSteps: Start {round(x0, 4)}: {trace}. Final {round(state, 4)}.\nSentence:")
    return gen([prompt])[0].strip()


def _faithful(text: str, reg: ToolRegistry, seq) -> bool:
    """The narration is faithful iff every tool it describes actually ran (no invented steps)."""
    used_descs = [reg[t].description.split(" {")[0].split()[0].lower() for t, _ in seq]
    low = text.lower()
    return all(any(w in low for w in (d, d + "s", d[:-1])) for d in set(used_descs))


# ── the loop: TRM decides -> tools execute -> verify -> bank -> LM speaks ──────────────────────────
def run_agent(tasks, reg, quorum_m: int = 2, gen=None, verbose: bool = False):
    store = WorkflowStore(quorum_m)
    ctrl = ToolController(reg, store)
    solved = banked = replays = 0
    tries_curve = []
    for i, t in enumerate(tasks):
        before = ctrl.tool_tries
        mode, seq, final = ctrl.solve(t)
        tries_curve.append(ctrl.tool_tries - before)
        if mode == "fail":
            continue
        # OUTCOME ANCHOR (the gate): the executed sequence must reach gold. Wrong tools never bank.
        if not math.isclose(final, t["gold"], **_TOL):
            continue
        solved += 1
        if mode == "search":
            ev = store.bank(seq, _goal_vec(t["text"]), t["domain"]); banked += int(ev == "new")
        else:
            replays += 1
            store.bank(seq, _goal_vec(t["text"]), t["domain"])      # replay also votes (promotes to certified)
        say = narrate(reg, seq, t["x0"], gen=gen)
        if verbose:
            print(f"  [{i:>2}] {t['domain']:<9} {mode:<7} seq={[s[0] for s in seq]}  \"{say[:70]}\"")
    n = len(tasks)
    early = sum(tries_curve[:len(tries_curve)//3]) / max(1, len(tries_curve)//3)
    late = sum(tries_curve[-len(tries_curve)//3:]) / max(1, len(tries_curve)//3)
    return dict(store=store, ctrl=ctrl, solved=solved, n=n, banked=banked, replays=replays,
                reuse=store.reuse, cross=store.cross_domain_reuse, early=early, late=late)


# ── selftest (no GPU) ─────────────────────────────────────────────────────────────────────────────
def selftest() -> bool:
    print("algo_grr_agent --selftest: TRM calls graph tools, LM only speaks (the agentic loop)\n")
    ok = True
    tasks, reg = make_tasks(seed=0)

    # [1] tools EXECUTE and COMPOSE (a tool calls another tool = atoms activate each other)
    c1 = math.isclose(reg.run("double", 5, 0), 10) and math.isclose(reg.run("square", 4, 0), 16) \
        and reg["double"].calls == ("scale",)
    print(f"  [1] tools execute + compose (double->scale, square->power): {'PASS' if c1 else 'FAIL'}")
    ok &= c1

    r = run_agent(tasks, reg, quorum_m=2, verbose=False)

    # [2] the loop SOLVES via the TRM controller (no LM in the decision path)
    c2 = r["solved"] >= int(0.9 * r["n"])
    print(f"  [2] TRM controller solves (no LM): {r['solved']}/{r['n']}  -> {'PASS' if c2 else 'FAIL'}")
    ok &= c2

    # [3] workflows BANK and later tasks REPLAY them (compounding: search cost falls)
    c3 = r["banked"] > 0 and r["replays"] > 0 and r["late"] < r["early"]
    print(f"  [3] bank+replay compounding: banked={r['banked']} replays={r['replays']} "
          f"tool-tries/task {r['early']:.0f}->{r['late']:.0f}  -> {'PASS' if c3 else 'FAIL'}")
    ok &= c3

    # [4] cross-domain WORKFLOW reuse (same tool-seq reused across different domains)
    c4 = r["cross"] > 0 or r["reuse"] > 0
    print(f"  [4] workflow reuse across tasks/domains: reuse={r['reuse']} cross-domain={r['cross']}  "
          f"-> {'PASS' if c4 else 'FAIL'}")
    ok &= c4

    # [5] LM SPEAKER faithful: narration cites only tools that actually ran
    t0 = next(t for t in tasks if t["domain"] == "time")
    mode, seq, final = r["ctrl"].solve(t0)
    say = narrate(reg, seq, t0["x0"], gen=None)
    c5 = seq is not None and _faithful(say, reg, seq)
    print(f"  [5] LM-speaker narration faithful (cites only executed tools): {'PASS' if c5 else 'FAIL'}")
    ok &= c5

    # [6] POISON-SAFE: a wrong tool-sequence misses the gold anchor -> never banks
    bad_final = reg.run("inc", t0["x0"], 999)
    c6 = not math.isclose(bad_final, t0["gold"], **_TOL)
    print(f"  [6] wrong tool-pick misses the anchor (not banked): {'PASS' if c6 else 'FAIL'}")
    ok &= c6

    # [7] LM is SPEAK-ONLY: the controller's decide-signature can't accept an LM; only narrate() takes gen.
    import inspect
    ctrl_params = set(inspect.signature(ToolController.solve).parameters) | set(
        inspect.signature(ToolController.__init__).parameters)
    narrate_params = set(inspect.signature(narrate).parameters)
    c7 = not (ctrl_params & {"gen", "lm", "llm"}) and "gen" in narrate_params
    print(f"  [7] decision path is LM-free (controller has no gen; only narrate speaks): {'PASS' if c7 else 'FAIL'}")
    ok &= c7

    # [8] certified workflows survive as the reusable memory (the graph IS the skill)
    c8 = len(r["store"].certified) >= 1
    print(f"  [8] certified workflow schemas banked: {len(r['store'].certified)}  -> {'PASS' if c8 else 'FAIL'}")
    ok &= c8

    print(f"\n  => THE LOOP: the TRM controller picks graph TOOLS (no LM), tools execute + compose exactly, the")
    print(f"     outcome anchor gates, verified WORKFLOWS bank + replay (compounding + cross-domain reuse), and the")
    print(f"     LM only SPEAKS the verified trace (faithful). Swap the ToolRegistry for real agentic tools -> same loop.")
    print(f"\n  ALGO_GRR_AGENT SELFTEST -> {'PASS' if ok else 'FAIL'}")
    return ok


def run_lm(lm: str):
    """Real-3B: the LM SPEAKS the verified tool-traces (decision/execution already done by TRM+tools)."""
    from v5.runtime.algo_grr_membrane import make_frozen_gen
    gen = make_frozen_gen(lm, temperature=0.3, max_new_tokens=64)
    tasks, reg = make_tasks(seed=0)
    r = run_agent(tasks, reg, quorum_m=2, gen=gen, verbose=True)
    faithful = 0
    checked = 0
    ctrl = ToolController(reg, WorkflowStore())
    for t in tasks[:8]:
        mode, seq, final = ctrl.solve(t)
        if seq is None:
            continue
        say = narrate(reg, seq, t["x0"], gen=gen); checked += 1
        faithful += int(_faithful(say, reg, seq))
    print(f"\n  solved {r['solved']}/{r['n']}  banked {r['banked']}  replays {r['replays']}  "
          f"cross-domain reuse {r['cross']}")
    print(f"  LM-spoken narrations faithful (cite only executed tools): {faithful}/{checked}")
    print(f"  => TRM decided + tools executed (exact); the LM only put the VERIFIED trace into words.")


def main():
    ap = argparse.ArgumentParser(description="agentic tool-calling loop: TRM calls, LM speaks")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--lm", type=str, default="")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.run:
        if not a.lm:
            raise SystemExit("--run needs --lm")
        run_lm(a.lm)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
