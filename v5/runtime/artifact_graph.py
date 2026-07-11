"""Self-Growing Algorithm Graph: the system grows its OWN node vocabulary by CAUSAL outcome-credit,
instead of us pre-declaring node types.

Everything before this (tool_memory/tool_compose/tool_library) hard-coded the taxonomy: a fixed
PRIMS registry, a fixed TASKS list, "induce nash_solver then payoff then compose". That is us
deciding what nodes exist. But there are MILLIONS of algorithms; we cannot enumerate the node
types. So here there is NO type enum and NO primitive registry. A node is just an ARTIFACT — a
piece of code the model wrote. Its "type" is emergent (its embedding / where it lands in the
graph). The graph decides which artifacts to KEEP by one law:

    an artifact earns its place iff REUSING it raises VERIFIED downstream success.

Concretely (this is the physics, not a taxonomy):

  wake — for each task in a stream (specified ONLY by I/O + an oracle, never by "which
         primitive to build"):
    1. retrieve candidate artifacts from the graph (embedding rank) and advertise them
    2. the model authors a solution: it may CALL advertised artifacts and/or DEFINE new
       helper functions for sub-computations it invents (we never tell it what to factor)
    3. VERIFY the solution end-to-end by execution against the oracle (grounded)
    4. CAUSAL credit: an advertised artifact is credited only if it was CALLED *and* removing
       it BREAKS verification (counterfactual — kills inert "present but useless" credit)
    5. register every function the model defined (target + helpers) as new candidate artifacts
  sleep — periodically:
    * PRUNE artifacts that were never causally reused by a *later* task (one-offs decay out)
    * MERGE behavioral duplicates (same fingerprint on a probe set) into one node

What emerges: the survivors after pruning are the recurring sub-algorithms the stream needed —
which appear in NO task spec (we only ever asked for count_primes, digital_root, ...). The graph
INVENTED its own primitive vocabulary (is_prime, digits, digit_sum, ...) and kept exactly the ones
that paid off. This is the DreamCoder wake/sleep shape, on a verified substrate; it bottoms out —
honestly — on the leaves being verifiable (the credit signal needs a grounded check somewhere).

Builds on: tool_compose.verify_fn (execute-with-deps), tool_memory._extract_code/_log,
reason_rl.batch_generate. The retrieve-ranking policy over a large graph is the noted frontier
(HRM-latent / ranker over the algorithm graph) — v1 advertises the top-k by a cheap embedding.

  selftest (no model):  python -m v5.runtime.artifact_graph --selftest
  run (GPU):            V5_LM_TRUST_REMOTE_CODE=1 python -m v5.runtime.artifact_graph --model Qwen/Qwen2.5-3B
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path

from v5.runtime.tool_compose import verify_fn
from v5.runtime.tool_memory import _extract_code, _log


# ═══════════════════════════════════════════════════════════════════════════════
# CHEAP EMBEDDING  (emergent "type" address; real runs may inject mpnet)
# ═══════════════════════════════════════════════════════════════════════════════

def _hash_embed(text: str, dim: int = 128) -> list[float]:
    """Deterministic bag-of-(token+trigram) hashing embedding — no network. Good enough to RANK
    a small graph; the real retrieval policy over a big graph is the noted frontier."""
    v = [0.0] * dim
    t = (text or "").lower()
    grams = re.findall(r"[a-z0-9_]+", t) + [t[i:i + 3] for i in range(max(0, len(t) - 2))]
    for g in grams:
        h = int(hashlib.md5(g.encode()).hexdigest(), 16) % dim
        v[h] += 1.0
    n = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / n for x in v]


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# ═══════════════════════════════════════════════════════════════════════════════
# BEHAVIORAL FINGERPRINT  (for dedup/merge — same behavior => same node)
# ═══════════════════════════════════════════════════════════════════════════════

# probe ARGUMENT TUPLES of varied arity/type, so single-arg fns (is_prime/digit_sum) AND multi-arg
# fns (build_adj(n,edges), dijkstra(n,edges,s)) both get a DISTINCT output signature. Large ints
# (19,199,9999,12345) separate a multi-iteration fn (digital_root) from a single-pass one; the graph
# tuples give 2-/3-/4-arg fns a non-ERR signature so equivalent impls MERGE (fixes the earlier
# multi-arg fp=None -> never-merges gap noted in the design doc).
_FP_ARGS = [(0,), (1,), (2,), (3,), (4,), (5,), (7,), (9,), (10,), (17,), (19,), (100,), (121,),
            (199,), (888,), (9999,), (12345,),
            ([],), ([2],), ([1, 2, 3, 4],), ([4, 6, 8, 9],), ([2, 3, 5, 7, 11],), ("aba",), ("abc",),
            (3, [(0, 1, 2), (1, 2, 3)]),                                    # (n, edges)
            (4, [(0, 1, 2), (0, 2, 5), (1, 2, 1), (2, 3, 3)], 0),           # (n, edges, s)
            (4, [(0, 1, 2), (0, 2, 5), (1, 2, 1), (2, 3, 3)], 0, 3),        # (n, edges, s, t)
            # graphs where BFS order != DFS order != topo, so order-sensitive traversals do NOT
            # fingerprint-collide (fixes MERGE dfs_order==bfs_order — distinct algos, distinct fp):
            (4, [(0, 1, 1), (1, 2, 1), (0, 3, 1)], 0),                      # bfs [0,1,3,2] dfs [0,1,2,3]
            (5, [(0, 1, 1), (0, 2, 1), (1, 3, 1), (2, 4, 1)], 0),           # bfs≠dfs on a wider tree
            (6, [(0, 1, 1), (0, 2, 1), (1, 3, 1), (2, 4, 1), (3, 5, 1)], 0)]


def _fingerprint(full_code: str, fn_name: str, timeout: float = 6.0) -> str | None:
    """Run fn_name over the probe arg-tuples (with deps prepended); hash the output vector. None if it
    can't even be evaluated (treat as unique — don't merge crashers together)."""
    harness = "\n".join([
        full_code, "",
        "if True:",
        f"    _probes = {_FP_ARGS!r}",
        "    _out = []",
        "    for _p in _probes:",
        "        try:",
        f"            _out.append(repr({fn_name}(*_p)))",
        "        except Exception as _e:",
        "            _out.append('ERR:' + type(_e).__name__)",
        "    print('FPRINT', '\\t'.join(_out))",
    ])
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "fp.py"
        p.write_text(harness, encoding="utf-8")
        try:
            proc = subprocess.run([sys.executable, "-I", str(p)], cwd=td, capture_output=True,
                                  text=True, encoding="utf-8", errors="replace", timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
    line = next((ln for ln in reversed((proc.stdout or "").splitlines())
                 if ln.startswith("FPRINT")), "")
    if not line:
        return None
    sig = line[len("FPRINT "):]
    if all(tok.startswith("ERR:") for tok in sig.split("\t")):
        return None                          # never evaluated on any probe -> unique, don't merge
    return hashlib.md5(sig.encode()).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════════
# CODE UTILITIES  (split top-level defs, signatures, call edges)
# ═══════════════════════════════════════════════════════════════════════════════

def _defs_in(code: str) -> dict[str, str]:
    """Map top-level function name -> its source segment."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            seg = ast.get_source_segment(code, node)
            if seg:
                out[node.name] = seg
    return out


def _sig_of(src: str, name: str) -> str:
    m = re.search(rf"def\s+{re.escape(name)}\s*\(([^)]*)\)", src)
    return f"{name}({m.group(1).strip()})" if m else f"{name}(...)"


def _defnames(code: str) -> set[str]:
    """Every function name DEFINED anywhere in `code` (top-level or nested). Used to distinguish a
    genuine library CALL from a locally re-implemented shadow of the same name."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    return {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}


def _calls_in(src: str, universe: set[str], self_name: str) -> list[str]:
    """Names from `universe` that `src` genuinely CALLS — excluding itself AND any name `src`
    DEFINES locally (nested/inline), so a re-implemented shadow isn't recorded as a library edge."""
    local = _defnames(src)
    return sorted(n for n in universe if n != self_name and n not in local
                  and re.search(rf"\b{re.escape(n)}\s*\(", src))


# ═══════════════════════════════════════════════════════════════════════════════
# THE ARTIFACT + THE GRAPH  (no type enum — a node is just code + its stats)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Artifact:
    name: str
    code: str                       # the single def's source
    sig: str
    doc: str
    calls: list[str] = field(default_factory=list)   # graph edges (artifacts it calls)
    fp: str | None = None           # behavioral fingerprint (dedup key)
    emb: list[float] = field(default_factory=list)   # retrieval address ("emergent type")
    born: int = 0                   # task index created
    last_used: object = -1          # task key last causally reused (int idx or task fingerprint)
    wins: int = 0                   # causal-credit reuse count (raw)
    uses: int = 0                   # times called at all (incl. inert)
    reused_by: list = field(default_factory=list)  # DISTINCT task keys that causally reused it
                                                    # -> amort(a) = len(set(reused_by)) = reuse BREADTH


class ArtifactGraph:
    """Type-blind store of code artifacts. No enum, no registry — the vocabulary is whatever the
    model authors and outcome-credit keeps."""

    def __init__(self, embed_fn=None):
        self.arts: dict[str, Artifact] = {}
        self.embed_fn = embed_fn or _hash_embed
        self.events: list[str] = []          # audit log (for reporting / selftest)

    # ── dependency closure (transitive) ─────────────────────────────────────────
    def closure(self, names) -> list[str]:
        seen, stack = [], list(names)
        while stack:
            n = stack.pop()
            if n in seen or n not in self.arts:
                continue
            seen.append(n)
            stack.extend(self.arts[n].calls)
        return seen

    def deps_code(self, names) -> str:
        # order among defs is irrelevant: all are defined before the target is CALLED at runtime.
        return "\n\n".join(self.arts[n].code for n in self.closure(names))

    # ── retrieval (advertise top-k by embedding) ────────────────────────────────
    def retrieve(self, task_text: str, k: int = 8) -> list[Artifact]:
        q = self.embed_fn(task_text)
        ranked = sorted(self.arts.values(), key=lambda a: _cos(q, a.emb), reverse=True)
        return ranked[:k]

    # ── register a new artifact, with behavioral dedup/merge ────────────────────
    def register(self, name: str, code: str, doc: str, born: int, calls: list[str]) -> str:
        fp = _fingerprint(self.deps_code(calls) + "\n\n" + code, name)
        if fp is not None:
            for a in self.arts.values():                 # behavioral duplicate already present?
                if a.fp == fp:
                    self.events.append(f"MERGE {name} == {a.name} (same behavior)")
                    return a.name                        # reuse the existing node, don't add a copy
        key = name
        if key in self.arts and self.arts[key].fp != fp:  # same name, different behavior -> version
            key = f"{name}__{born}"
        self.arts[key] = Artifact(name=key, code=code, sig=_sig_of(code, name), doc=doc,
                                  calls=[c for c in calls if c in self.arts], fp=fp,
                                  emb=self.embed_fn(f"{name} {doc} {code}"), born=born)
        self.events.append(f"ADD {key}")
        return key

    def credit(self, name: str, task_key):
        """Record a causal reuse. task_key is an int task index (artifact_graph stream) OR a task
        behavioral fingerprint (curriculum) — DISTINCT keys measure reuse BREADTH (amortization),
        so re-solving the same task on many seeds does NOT inflate it (anti-memorization)."""
        if name in self.arts:
            a = self.arts[name]
            a.wins += 1
            a.last_used = task_key
            if task_key not in a.reused_by:
                a.reused_by.append(task_key)
            self.events.append(f"CREDIT {name} (win {a.wins}, amort {self.amort(name)}, key {task_key})")

    def amort(self, name: str) -> int:
        """Amortization = number of BEHAVIORALLY-DISTINCT tasks that causally reused this artifact.
        High = broadly reusable; 1 = single-context (over-specific / memorized)."""
        return len(set(self.arts[name].reused_by)) if name in self.arts else 0

    def touch(self, name: str):
        if name in self.arts:
            self.arts[name].uses += 1

    # ── sleep: prune never-causally-reused one-offs ─────────────────────────────
    def prune(self, now: int, grace: int = 2) -> list[str]:
        dead = [k for k, a in self.arts.items()
                if a.wins == 0 and (now - a.born) >= grace]
        for k in dead:
            # keep an artifact that something SURVIVING still depends on (transitive dep)
            if any(k in self.closure([o]) for o in self.arts if o not in dead):
                continue
            self.events.append(f"PRUNE {k} (never reused, born task {self.arts[k].born})")
            del self.arts[k]
        return [k for k in dead if k not in self.arts]

    # ── persistence (the graph is a memory: survives box resets / runs) ─────────
    def save(self, path: str):
        d = Path(path); d.mkdir(parents=True, exist_ok=True)
        for k, a in self.arts.items():
            (d / f"{k}.json").write_text(json.dumps(asdict(a)), encoding="utf-8")

    def load(self, path: str):
        d = Path(path)
        if not d.exists():
            return
        for f in d.glob("*.json"):
            try:
                self.arts[f.stem] = Artifact(**json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                pass

    def survivors(self) -> list[str]:
        return sorted(self.arts.keys())


# ═══════════════════════════════════════════════════════════════════════════════
# TASK STREAM  — specified ONLY by (target fn, oracle, I/O). The shared sub-algorithms
# (is_prime, digits, digit_sum) are LATENT — they appear in NO task text. If they
# emerge as graph nodes, the graph invented its own vocabulary.
# ═══════════════════════════════════════════════════════════════════════════════

def _is_prime(n):
    if not isinstance(n, int) or n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def _digits(n):
    return [int(c) for c in str(abs(int(n)))]


def _digit_sum(n):
    return sum(_digits(n))


def _digital_root(n):
    n = abs(int(n))
    while n >= 10:
        n = _digit_sum(n)
    return n


def _is_pal(seq):
    s = list(str(seq)) if isinstance(seq, int) else list(seq)
    return s == s[::-1]


@dataclass
class Task:
    name: str          # target function the model must write
    text: str          # natural-language spec (NO mention of latent primitives)
    oracle: object     # reference impl for verification
    gen: object        # seed -> a single input args list


def _rlist(seed, lo=1, hi=30):
    r = random.Random(seed)
    return [r.randint(lo, hi) for _ in range(r.randint(4, 8))]


def _rint(seed, lo=1, hi=999):
    return [random.Random(seed).randint(lo, hi)]


STREAM = [
    Task("count_primes", "Write `count_primes(lst)`: how many elements of the list are prime numbers.",
         lambda a: sum(_is_prime(x) for x in a[0]), lambda s: [_rlist(s)]),
    Task("sum_primes", "Write `sum_primes(lst)`: the sum of the prime numbers in the list.",
         lambda a: sum(x for x in a[0] if _is_prime(x)), lambda s: [_rlist(s)]),
    Task("largest_prime", "Write `largest_prime(lst)`: the largest prime in the list, or -1 if none.",
         lambda a: max([x for x in a[0] if _is_prime(x)] or [-1]), lambda s: [_rlist(s)]),
    Task("digit_sum", "Write `digit_sum(n)`: the sum of the decimal digits of a non-negative integer n.",
         lambda a: _digit_sum(a[0]), lambda s: _rint(s)),
    Task("digital_root", "Write `digital_root(n)`: repeatedly sum the digits of n until one digit remains.",
         lambda a: _digital_root(a[0]), lambda s: _rint(s, hi=99999)),
    Task("is_palindrome_num", "Write `is_palindrome_num(n)`: True iff the decimal digits of n read the same forwards and backwards.",
         lambda a: _is_pal(_digits(a[0])), lambda s: _rint(s, hi=99999)),
    Task("count_prime_digits", "Write `count_prime_digits(n)`: how many decimal digits of n are prime (2,3,5,7).",
         lambda a: sum(_is_prime(d) for d in _digits(a[0])), lambda s: _rint(s, hi=99999)),
    Task("sum_digitsums_of_primes", "Write `sum_digitsums_of_primes(lst)`: for each prime in the list, take the sum of its digits; return the total.",
         lambda a: sum(_digit_sum(x) for x in a[0] if _is_prime(x)), lambda s: [_rlist(s, hi=99)]),
    Task("celsius_to_f", "Write `celsius_to_f(lst)`: convert each Celsius temperature to Fahrenheit (c*9/5+32), return the list.",
         lambda a: [c * 9 / 5 + 32 for c in a[0]], lambda s: [_rlist(s, lo=-20, hi=40)]),
]


def _cases(task: Task, seeds) -> list:
    return [(task.gen(s), task.oracle(task.gen(s))) for s in seeds]


# ═══════════════════════════════════════════════════════════════════════════════
# AUTHOR one task  (gen -> extract -> verify -> refine), decoupled via gen_fn so the
# SAME loop runs under the StubLM (no GPU, proves mechanism) or the real LM.
# ═══════════════════════════════════════════════════════════════════════════════

def _author_prompt(task: Task, advertised: list[Artifact], prev, fails) -> str:
    parts = [task.text]
    if advertised:
        parts.append("\nReusable functions already in your library (already DEFINED — CALL them by "
                     "name, do NOT re-implement their logic):")
        for a in advertised:
            parts.append(f"  {a.sig}  -> {a.doc}")
    parts.append("\nIf you need a sub-computation that isn't listed, DEFINE it as its own small "
                 "helper function (so it can be reused later), then call it.")
    if prev:
        parts.append(f"\nYour previous attempt was wrong:\n```python\n{prev}\n```")
        for args, exp, got in fails[:4]:
            parts.append(f"  on input {args} it returned {got}, correct is {exp}")
    parts.append(f"\nWrite `{task.name}(...)`. Output ONLY one Python code block.")
    return "\n\n".join(parts)


def solve_task(graph: ArtifactGraph, gen_fn, task: Task, now: int, verify_seeds, eval_seeds,
               k=8, rounds=3, samples=4):
    """Returns (eval_acc, called_reuse, causal, new_defs, code)."""
    vcases, ecases = _cases(task, verify_seeds), _cases(task, eval_seeds)
    advertised = graph.retrieve(task.text, k=k)
    prev, fails, best = None, [], (0.0, "", [])
    for _ in range(rounds):
        prompts = [_author_prompt(task, advertised, prev, fails)] * samples
        cands = gen_fn(prompts)
        for gen in cands:
            code = _extract_code(gen)
            sol_defs = _defs_in(code)
            # reuse = advertised artifacts CALLED but NOT redefined in this solution
            called = [a.name for a in advertised
                      if a.name not in sol_defs and re.search(rf"\b{re.escape(a.name)}\s*\(", code)]
            acc, f, _ = verify_fn(code, task.name, vcases, graph.deps_code(called))
            if acc > best[0]:
                best = (acc, code, called)
        acc, code, called = best
        if acc >= 0.999:
            break
        # refine from the current best's failures
        _, f, _ = verify_fn(best[1], task.name, vcases, graph.deps_code(best[2])) if best[1] else (0, [], "")
        prev, fails = best[1], f
    acc, code, called = best
    eval_acc, _, _ = verify_fn(code, task.name, ecases, graph.deps_code(called)) if code else (0.0, [], "")

    causal, new_defs = [], []
    if eval_acc >= 0.999:
        causal = _causal_credit(graph, code, task.name, ecases, called)
        for nm in causal:
            graph.credit(nm, now)
        for nm in called:
            graph.touch(nm)
        new_defs = _register_solution(graph, code, task, now, existing_called=called)
    return eval_acc, called, causal, new_defs, code


def _causal_credit(graph: ArtifactGraph, code: str, fn: str, cases, called) -> list[str]:
    """An advertised artifact is credited only if REMOVING it breaks verification — kills credit
    for calls that are present but inert."""
    full, _, _ = verify_fn(code, fn, cases, graph.deps_code(called))
    causal = []
    for nm in called:
        rest = [c for c in called if c != nm]
        acc, _, _ = verify_fn(code, fn, cases, graph.deps_code(rest))
        if acc < full - 1e-9:
            causal.append(nm)
    return causal


def _register_solution(graph: ArtifactGraph, code: str, task: Task, now: int,
                       existing_called) -> list[str]:
    """Register every function the model DEFINED (target + invented helpers) as a candidate node.
    Artifacts it merely reused (existing_called) are already in the graph."""
    defs = _defs_in(code)
    universe = set(defs) | set(graph.arts)
    added = []
    for name, src in defs.items():
        calls = _calls_in(src, universe, name)
        doc = task.text.split(":", 1)[-1].strip()[:70] if name == task.name \
            else f"helper invented while solving {task.name}"
        key = graph.register(name, src, doc, now, calls)
        added.append(key)
    return added


# ═══════════════════════════════════════════════════════════════════════════════
# THE RUN  (wake over the stream, sleep to prune/merge, report emergence+compounding)
# ═══════════════════════════════════════════════════════════════════════════════

def run_graph(gen_fn, stream=STREAM, verify_n=24, eval_n=24, k=8, rounds=3, samples=4,
              grace=2, store_dir=None, quiet=False):
    graph = ArtifactGraph()
    if store_dir:
        graph.load(store_dir)
        if graph.arts and not quiet:
            _log(f"  [graph] loaded {len(graph.arts)} artifact(s): {graph.survivors()}")
    vseeds = list(range(1000, 1000 + verify_n))
    eseeds = list(range(5000, 5000 + eval_n))

    log = []
    n_authored, n_reuse = 0, 0
    for i, task in enumerate(stream):
        before = set(graph.arts)                         # keys present BEFORE this task
        ev, called, causal, new_defs, _ = solve_task(graph, gen_fn, task, i, vseeds, eseeds,
                                                      k=k, rounds=rounds, samples=samples)
        # authored-NEW = genuinely new keys (a behavioral merge returns an EXISTING key -> not fresh)
        fresh = [d for d in new_defs if d not in before]
        n_authored += len(fresh)
        n_reuse += len(causal)
        log.append(dict(task=task.name, ok=ev >= 0.999, reused=causal, authored=fresh))
        if not quiet:
            _log(f"[t{i}] {task.name:24} {'OK ' if ev>=0.999 else 'FAIL'} "
                 f"reuse={causal or '-'}  new={fresh or '-'}")
        graph.prune(i, grace=grace)                      # sleep: decay one-offs each step

    graph.prune(len(stream) + grace, grace=grace)        # final sweep
    if store_dir:
        graph.save(store_dir)

    solved = sum(1 for r in log if r["ok"])
    reused_vocab = [k for k, a in graph.arts.items() if a.wins >= 1]   # primitives that paid off
    if not quiet:
        _log("\n" + "=" * 62 + "\n=== SELF-GROWN GRAPH ===")
        _log(f"  tasks solved: {solved}/{len(stream)}")
        _log(f"  authored {n_authored} node(s) total | {n_reuse} causal REUSES on "
             f"{len(reused_vocab)} distinct primitive(s)  "
             f"({'COMPOUNDS — build once, reuse many' if n_reuse > len(reused_vocab) else 'little reuse'})")
        _log(f"  EMERGENT VOCABULARY (survivors — none were named in a task spec):")
        for k_ in graph.survivors():
            a = graph.arts[k_]
            _log(f"     {a.sig:26} wins={a.wins} born=t{a.born} calls={a.calls or '-'}")
    return graph, log


# ═══════════════════════════════════════════════════════════════════════════════
# STUB LM  — deterministic author (no GPU). Respects the advertised store: CALLS a
# helper if it's advertised, else DEFINES it inline. Exercises reuse/credit/register.
# ═══════════════════════════════════════════════════════════════════════════════

_HELPER_SRC = {
    "is_prime": ("def is_prime(n):\n    if n < 2:\n        return False\n"
                 "    i = 2\n    while i * i <= n:\n        if n % i == 0:\n"
                 "            return False\n        i += 1\n    return True"),
    "digits": "def digits(n):\n    return [int(c) for c in str(abs(n))]",
    "digit_sum": "def digit_sum(n):\n    return sum(int(c) for c in str(abs(n)))",
}

# each task: the target body, and which helpers it needs (call if advertised, else inline)
_STUB = {
    "count_primes": (["is_prime"], "def count_primes(lst):\n    return sum(1 for x in lst if is_prime(x))"),
    "sum_primes": (["is_prime"], "def sum_primes(lst):\n    return sum(x for x in lst if is_prime(x))"),
    "largest_prime": (["is_prime"], "def largest_prime(lst):\n    ps = [x for x in lst if is_prime(x)]\n    return max(ps) if ps else -1"),
    "digit_sum": ([], "def digit_sum(n):\n    return sum(int(c) for c in str(abs(n)))"),
    "digital_root": (["digit_sum"], "def digital_root(n):\n    while n >= 10:\n        n = digit_sum(n)\n    return n"),
    "is_palindrome_num": (["digits"], "def is_palindrome_num(n):\n    d = digits(n)\n    return d == d[::-1]"),
    "count_prime_digits": (["digits", "is_prime"], "def count_prime_digits(n):\n    return sum(1 for d in digits(n) if is_prime(d))"),
    "sum_digitsums_of_primes": (["is_prime", "digit_sum"], "def sum_digitsums_of_primes(lst):\n    return sum(digit_sum(x) for x in lst if is_prime(x))"),
    "celsius_to_f": (["c2f"], "def celsius_to_f(lst):\n    return [c2f(c) for c in lst]"),
}
_HELPER_SRC["c2f"] = "def c2f(c):\n    return c * 9 / 5 + 32"


def _stub_gen(prompts: list[str]) -> list[str]:
    out = []
    for prompt in prompts:
        # the target fn is named in the trailing "Write `NAME(...)`" instruction (last occurrence —
        # the task text ALSO opens with "Write `NAME(args)`")
        name = re.findall(r"Write `([a-z_][a-z0-9_]*)\(", prompt)[-1]
        needs, body = _STUB.get(name, ([], f"def {name}(x):\n    return x"))
        # read only the advertised "Reusable functions" block to decide reuse-vs-define
        block = ""
        if "already DEFINED" in prompt:
            block = prompt.split("already DEFINED", 1)[1].split("If you need a sub-computation", 1)[0]
        pieces = []
        for h in needs:
            advertised = re.search(rf"\b{re.escape(h)}\s*\(", block)
            if not advertised and h in _HELPER_SRC:
                pieces.append(_HELPER_SRC[h])           # define inline (first use — not yet in library)
        pieces.append(body)
        out.append("```python\n" + "\n\n".join(pieces) + "\n```")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST (no model) — the mechanism: dedup/merge, transitive deps, causal credit,
# and a full stub run proving emergence + compounding + prune + persistence.
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    print("artifact_graph --selftest: self-grown graph mechanism (no model)\n")

    # [1] behavioral fingerprint + merge: is_prime and a renamed copy => ONE node
    g = ArtifactGraph()
    g.register("is_prime", _HELPER_SRC["is_prime"], "prime?", 0, [])
    dup = _HELPER_SRC["is_prime"].replace("is_prime", "prime_ok")
    key = g.register("prime_ok", dup, "prime?", 1, [])
    assert key == "is_prime" and len(g.arts) == 1, f"behavioral dup must merge, got {g.survivors()}"
    print("  [1] behavioral duplicate (renamed is_prime) MERGES into one node -> PASS")

    # [2] transitive dependency closure: target -> dsum2 -> digits (behaviorally distinct nodes)
    g2 = ArtifactGraph()
    g2.register("digits", _HELPER_SRC["digits"], "digits", 0, [])
    g2.register("dsum2", "def dsum2(n):\n    return sum(digits(n))", "digit sum via digits", 0, ["digits"])
    g2.register("times3", "def times3(n):\n    return dsum2(n) * 3", "3x digit sum", 0, ["dsum2"])
    clo = set(g2.closure(["times3"]))
    assert clo == {"times3", "dsum2", "digits"}, f"closure wrong: {clo}"
    acc, _, err = verify_fn("def t(n):\n    return times3(n)", "t",
                            [([123], 18), ([9999], 108)], g2.deps_code(["times3"]))
    assert acc == 1.0, f"transitive deps must resolve, got {acc:.0%} {err}"
    print("  [2] transitive closure times3->dsum2->digits resolves + verifies -> PASS")

    # [3] causal credit: a genuinely-needed call is credited; an inert call is NOT
    g3 = ArtifactGraph()
    g3.register("is_prime", _HELPER_SRC["is_prime"], "prime?", 0, [])
    g3.register("digits", _HELPER_SRC["digits"], "digits", 0, [])
    cp = "def count_primes(lst):\n    return sum(1 for x in lst if is_prime(x))"
    cases = _cases(STREAM[0], range(1000, 1010))
    causal = _causal_credit(g3, cp, "count_primes", cases, ["is_prime", "digits"])
    assert causal == ["is_prime"], f"only is_prime is causal, got {causal}"
    print("  [3] causal credit: is_prime credited, inert `digits` call NOT -> PASS")

    # [4] full stub run — emergence + compounding + prune + no-taxonomy
    graph, log = run_graph(_stub_gen, grace=2, quiet=True)
    assert all(r["ok"] for r in log), f"stub should solve all: {[r['task'] for r in log if not r['ok']]}"
    surv = set(graph.survivors())
    # the vocabulary the graph INVENTED — none of these are a task target except digit_sum,
    # and is_prime/digits were NEVER asked for (they are helpers the model factored)
    assert {"is_prime", "digits", "digit_sum"} <= surv, f"emergent primitives missing: {surv}"
    assert "c2f" not in surv, "one-off helper c2f must be pruned"
    # composites that nothing reused must be pruned (only reusable atoms crystallize)
    assert "count_primes" not in surv and "celsius_to_f" not in surv, f"dead composites survived: {surv}"
    for prim in ("is_prime", "digits", "digit_sum"):
        assert graph.arts[prim].wins >= 1, f"{prim} should have reuse wins"
    n_reuse = sum(len(r["reused"]) for r in log)
    reused_vocab = [k for k, a in graph.arts.items() if a.wins >= 1]
    # compounding = reuses exceed the number of distinct primitives reused (build once, reuse many)
    assert n_reuse > len(reused_vocab), f"must compound: {n_reuse} reuses !> {len(reused_vocab)} primitives"
    print(f"  [4] stub run: 9/9 solved | emergent vocab {sorted(surv)}")
    print(f"      compounding: {n_reuse} causal reuses on {len(reused_vocab)} distinct primitives;")
    print(f"      one-off c2f pruned, dead composites pruned -> PASS")

    # [5] persistence: save survivors, reload, rerun -> loaded PRIMITIVES are reused, never
    # re-authored (targets are still written — that's the work, not the library)
    import tempfile as _tf
    with _tf.TemporaryDirectory() as td:
        graph.save(td)
        g5 = ArtifactGraph(); g5.load(td)
        assert set(g5.survivors()) == surv, "save/load must roundtrip the vocabulary"
        _, log2 = run_graph(_stub_gen, store_dir=td, grace=99, quiet=True)  # grace=99 => no pruning
        authored_names = set().union(*[set(r["authored"]) for r in log2]) if log2 else set()
        assert not ({"is_prime", "digits", "digit_sum"} & authored_names), \
            f"loaded primitives must be reused, not re-authored — but re-authored {authored_names}"
        reuses2 = sum(len(r["reused"]) for r in log2)
        assert reuses2 >= n_reuse, f"loaded run should reuse at least as much ({reuses2} vs {n_reuse})"
    print(f"  [5] persistence: reload vocabulary -> {reuses2} reuses, 0 primitives re-authored -> PASS")

    print("\n  ARTIFACT_GRAPH SELFTEST -> PASS")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# REAL RUN  (wraps batch_generate as gen_fn)
# ═══════════════════════════════════════════════════════════════════════════════

def _real_gen_fn(model_name: str, chunk: int):
    import os
    from transformers import AutoTokenizer
    from v5.lm_loader import load_frozen_lm
    from v5.runtime.reason_rl import batch_generate

    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    _log(f"  [graph] loading {model_name}...")
    model = load_frozen_lm(model_name); model.eval()
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    dev = next(model.parameters()).device

    def gen_fn(prompts):
        return batch_generate(model, tok, prompts, dev, max_new=420, sample=True,
                              temperature=1.0, chunk=chunk)
    return gen_fn


def main():
    ap = argparse.ArgumentParser(description="Self-growing algorithm graph (causal outcome-credit).")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--verify-n", type=int, default=24)
    ap.add_argument("--eval-n", type=int, default=24)
    ap.add_argument("--k", type=int, default=8, help="artifacts advertised per task")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--grace", type=int, default=2, help="tasks an unused node survives before prune")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="run only the first N tasks (0 = all; smoke)")
    ap.add_argument("--store-dir", default=None, help="persist/reload the graph here (the memory)")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    stream = STREAM[:a.limit] if a.limit else STREAM
    gen_fn = _real_gen_fn(a.model, a.chunk)
    run_graph(gen_fn, stream=stream, verify_n=a.verify_n, eval_n=a.eval_n, k=a.k, rounds=a.rounds,
              samples=a.samples, grace=a.grace, store_dir=a.store_dir)


if __name__ == "__main__":
    main()
