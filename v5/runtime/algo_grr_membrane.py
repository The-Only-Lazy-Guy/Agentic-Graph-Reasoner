"""algo_grr_membrane — the frozen-compiler + TRM-membrane closed-loop solver (GRR-Tool, build A).

Design (READ_THIS "POISON DIAGNOSIS + frozen-compiler resolution", 2026-07-16)
-----------------------------------------------------------------------------
LM = FROZEN COMPILER. It is a pure `compile(spec) -> code`; its weights never change, so there is
NO gradient path graph->LM (kills weight-poison). The TRM is the MEMBRANE between graph and LM:
the graph NEVER reaches the LM directly. The membrane retrieves, tentatively composes, and hands
the LM ONLY a curated spec {task, selected atoms (+dep closure), holes} — never a raw top-k dump
(kills context-poison). A bad atom only costs if it is selected AND survives the hard verify gate;
compiling a bad spec -> verify fails -> not banked. The loop self-cleans.

This module is the ORCHESTRATOR. Retrieval policy is injectable:
  - default = iterative, VERIFIER-GATED cosine (the honest baseline: one-shot cosine made multi-hop
    and program-conditioned by re-scoring each hop against realized coverage);
  - a trained TRM policy (build B) drops in via `policy_fn` with the same interface.

The LM compile_fn is injectable too:
  - `make_stub_compiler(recipes)` = deterministic no-GPU stand-in for the selftest;
  - a frozen 3B (algo_lm_author.make_hf_gen wrapper) for molab.

    selftest (no GPU/LM):  python -m v5.runtime.algo_grr_membrane --selftest
"""
from __future__ import annotations

import argparse
import ast
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from graph_core import MemoryGraph, Node, Edge  # type: ignore  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════════
# Dependency-closure realizer — assemble an atom + its transitive `depend` closure
# ═══════════════════════════════════════════════════════════════════════════════

def _depend_map(graph: MemoryGraph) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {nid: [] for nid in graph.nodes}
    for e in graph.edges:
        if e.relation == "depend" and e.src in graph.nodes and e.dst in graph.nodes:
            out[e.src].append(e.dst)
    return out


def resolve_closure(graph: MemoryGraph, atom_ids: list[str]) -> list[str]:
    """Transitive `depend` closure of atom_ids, ordered deps-FIRST (realizer order)."""
    dep = _depend_map(graph)
    order: list[str] = []
    seen: set[str] = set()

    def visit(nid: str) -> None:
        if nid in seen or nid not in graph.nodes:
            return
        seen.add(nid)
        for d in dep.get(nid, []):
            visit(d)
        order.append(nid)

    for a in atom_ids:
        visit(a)
    return order


def realize_closure_code(graph: MemoryGraph, atom_ids: list[str]) -> str:
    """Concatenated source of atom_ids + their dep closure, deps first."""
    parts = []
    for nid in resolve_closure(graph, atom_ids):
        code = graph.nodes[nid].metadata.get("code", "")
        if code:
            parts.append(code.rstrip("\n"))
    return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# VERIFY — the hard gate. exec code, run I/O tests, return (fraction_pass, detail)
# ═══════════════════════════════════════════════════════════════════════════════

def _called(code: str, name: str) -> bool:
    """True iff `name` is CALLED in code (a `name(` occurrence that is not its own `def name(`)."""
    calls = len(re.findall(r"(?<![\w.])" + re.escape(name) + r"\s*\(", code))
    defs = len(re.findall(r"\bdef\s+" + re.escape(name) + r"\s*\(", code))
    return calls - defs > 0


def reuse_set(code: str, entry: str, atom_entries: set[str]) -> list[str]:
    """Atom entry-names REACHABLE from `entry` through the call graph (BFS over FunctionDef bodies).

    This is the true reuse unit: an atom that merely appears in the concatenated closure (e.g. a
    composite the retriever tried whose body calls `sum_divisors`) but is NOT reachable from the
    entry does not count -> no spurious depend edge is banked. Falls back to a flat call scan if the
    code does not parse."""
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return [a for a in atom_entries if _called(code, a)]
    calls: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            named = {n.func.id for n in ast.walk(node)
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
            calls[node.name] = named
    seen: set[str] = set()
    stack = [entry]
    reached: list[str] = []
    while stack:
        fn = stack.pop()
        if fn in seen:
            continue
        seen.add(fn)
        for c in calls.get(fn, ()):
            if c in atom_entries and c not in reached:
                reached.append(c)
            if c in calls and c not in seen:
                stack.append(c)
    return reached


def verify_code(code: str, entry: str, tests: list[tuple]) -> tuple[float, str]:
    """Returns (fraction of tests passing, detail). Any exception -> that test fails."""
    if not tests:
        return 0.0, "no tests"
    ns: dict = {}
    try:
        exec(compile(code, f"<compile:{entry}>", "exec"), ns)
    except Exception as e:  # noqa: BLE001
        return 0.0, f"compile error: {e!r}"
    fn = ns.get(entry)
    if not callable(fn):
        return 0.0, f"entry '{entry}' not defined"
    n_ok = 0
    first_err = ""
    for args, expected in tests:
        try:
            got = fn(*args)
        except Exception as e:  # noqa: BLE001
            if not first_err:
                first_err = f"{entry}{args!r} raised {e!r}"
            continue
        if got == expected:
            n_ok += 1
        elif not first_err:
            first_err = f"{entry}{args!r} -> {got!r} != {expected!r}"
    return n_ok / len(tests), (first_err or "all pass")


# ═══════════════════════════════════════════════════════════════════════════════
# Retrieval — no-GPU token-overlap embedder + cosine (mpnet injected on molab)
# ═══════════════════════════════════════════════════════════════════════════════

_TOK = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> list[str]:
    # light plural-fold so "anagram"=="anagrams", "string"=="strings" match in the cosine baseline
    # (a real embedder is stemming-invariant; the TRM policy makes this moot).
    out = []
    for t in _TOK.findall(s.lower()):
        if len(t) >= 5 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return out


class TokenRetriever:
    """Dependency-free cosine over bag-of-token vectors of node purpose text.

    Stand-in for mpnet in the no-GPU selftest. Same interface a real embed retriever exposes:
    `rank(query_text, exclude) -> [(node_id, score), ...]` over implementation nodes only.
    """

    def __init__(self, graph: MemoryGraph):
        self.graph = graph
        self.impl_ids = [nid for nid, n in graph.nodes.items() if n.node_type == "implementation"]
        self._vecs = {nid: Counter(_tokens(graph.nodes[nid].text)) for nid in self.impl_ids}
        # idf over the impl corpus so generic words ("the", "of") don't dominate
        df: Counter = Counter()
        for v in self._vecs.values():
            df.update(v.keys())
        n = max(1, len(self.impl_ids))
        self._idf = {t: math.log(1.0 + n / (1 + c)) for t, c in df.items()}

    def _w(self, counter: Counter) -> dict[str, float]:
        return {t: c * self._idf.get(t, math.log(1.0 + len(self.impl_ids))) for t, c in counter.items()}

    @staticmethod
    def _cos(a: dict[str, float], b: dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(a[t] * b.get(t, 0.0) for t in a)
        na = math.sqrt(sum(x * x for x in a.values()))
        nb = math.sqrt(sum(x * x for x in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    def rank(self, query_text: str, exclude: set[str] | None = None) -> list[tuple[str, float]]:
        exclude = exclude or set()
        q = self._w(Counter(_tokens(query_text)))
        scored = [(nid, self._cos(q, self._w(self._vecs[nid])))
                  for nid in self.impl_ids if nid not in exclude]
        scored.sort(key=lambda x: -x[1])
        return scored


# ═══════════════════════════════════════════════════════════════════════════════
# Stub compiler — deterministic no-GPU stand-in for the FROZEN LM
# ═══════════════════════════════════════════════════════════════════════════════

def make_stub_compiler(recipes: dict[str, str]) -> Callable[[dict], str]:
    """A fake 'frozen compiler'. Given a spec, prepend the CURATED atoms' closure code, then emit the
    entry body from a per-entry recipe (stands in for what a real LM would infer from the spec).

    recipes: entry_name -> body source (a `def <entry>(...)` block that CALLS the selected atoms).
    The stub can only produce correct code when the spec.atoms contain what the recipe needs -> the
    membrane's coverage signal is real (missing atom -> NameError -> verify fails).
    """
    def compile_fn(spec: dict) -> str:
        closure = "\n\n".join(a["code"].rstrip("\n") for a in spec.get("atoms", []))
        body = recipes.get(spec["entry"], "")
        return (closure + "\n\n" + body) if closure else body
    return compile_fn


# ═══════════════════════════════════════════════════════════════════════════════
# Frozen-LM compiler — the REAL frozen 3B as `compile(spec)->code` (molab)
# ═══════════════════════════════════════════════════════════════════════════════

def _sig(code: str) -> str:
    """Extract the `name(params)` signature from an atom's source (first def line)."""
    m = re.search(r"def\s+([A-Za-z_]\w*)\s*\(([^)]*)\)", code or "")
    return f"{m.group(1)}({m.group(2)})" if m else ""


def render_compile_prompt(spec: dict) -> str:
    """Render the CURATED spec into the frozen compiler's prompt. Only spec['atoms'] appear —
    the graph never reaches the LM (the membrane). Atoms are framed as already-defined helpers the
    model composes; it writes ONLY the entry glue. Verified atom code is prepended by the caller, so
    the LM cannot corrupt the atoms."""
    lines = [
        "You are a code compiler. Solve the task by COMPOSING small functions. Rules:",
        "- CALL a provided helper wherever it applies; never reimplement one.",
        "- For any sub-computation NOT covered by a provided helper, define it as a SEPARATE, "
        "TOP-LEVEL, GENERAL function with a descriptive name (e.g. `sum_of_squares`, "
        "`nth_fibonacci`) — do NOT inline it, and do NOT nest it inside the entry function.",
        f"- Finally define `{spec['entry']}` to CALL those functions.",
        "- Output ONLY function definitions. No test code, no print(), no calls at module level.",
        "",
        "Required shape (helpers at top level, NOT nested inside the entry):",
        "```python",
        "def sum_of_squares(k):        # general top-level helper",
        "    return sum(i * i for i in range(1, k + 1))",
        "def solve_it(n):              # entry calls the helper",
        "    return sum_of_squares(n)",
        "```",
        "",
        f"Task: {spec['task_text']}",
        f"Entry function name: {spec['entry']}",
    ]
    atoms = spec.get("atoms", [])
    if atoms:
        lines += ["", "Already-defined helper functions you may call (do NOT redefine them):"]
        for a in atoms:
            lines.append(f"  # {a['purpose']}")
            lines.append(f"  {_sig(a['code']) or a['name']}")
    else:
        lines += ["", "No helper functions are available — write the full solution yourself."]
    tests = spec.get("tests", [])[:4]
    if tests:
        lines += ["", "It must satisfy:"]
        for args, expected in tests:
            inner = ", ".join(repr(x) for x in args)
            lines.append(f"  {spec['entry']}({inner}) == {expected!r}")
    fail = spec.get("failure")
    if fail:
        lines += ["", "Your previous attempt FAILED. Code:", "```python", str(fail.get("code", "")),
                  "```", f"Error: {fail.get('error', '')}", "Fix it."]
    if spec.get("derive"):
        lines += [
            "",
            "None of the helpers covers the core operation. Factor it out: first define a SMALL, "
            "GENERAL, reusable helper function named for WHAT IT COMPUTES (not for this task, e.g. "
            "`sum_of_squares`/`nth_fibonacci`, never `solve`/the task name), then define "
            f"`{spec['entry']}` to CALL that helper. The helper must be reusable by other tasks — "
            "no task-specific constants baked in."]
    lines += ["", "Return ALL the functions (top-level helpers first, then the entry) in one "
              "```python code block."]
    return "\n".join(lines)


def strip_module_exec(code: str) -> str:
    """Keep only top-level defs/imports; drop stray executable statements (print/check()/__main__
    blocks) the LM appends. Hygiene (our verify calls the entry) + anti-cheat: the LM must never run
    its OWN test harness inside our sandbox. Preserves original source of each kept node."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                             ast.Import, ast.ImportFrom)):
            seg = ast.get_source_segment(code, node)
            if seg:
                keep.append(seg)
    return "\n\n".join(keep) if keep else code


def _extract_code(text: str) -> str:
    """Pull code from a fenced ```python block, else from the first top-level def onward."""
    m = re.search(r"```(?:python)?\s*(.+?)```", text, re.S)
    if m:
        return m.group(1).strip()
    i = text.find("def ")
    return text[i:].strip() if i >= 0 else text.strip()


def _strip_redefs(glue: str, atom_names: set[str]) -> str:
    """Remove any top-level `def <atom>` blocks the LM redefined — the graph's verified atoms are
    authoritative. A block runs from `def name(` to the next top-level `def`/EOF."""
    if not atom_names:
        return glue
    lines = glue.splitlines()
    out: list[str] = []
    skip = False
    for ln in lines:
        m = re.match(r"def\s+([A-Za-z_]\w*)\s*\(", ln)
        if m:
            skip = m.group(1) in atom_names
        elif skip and ln and not ln[0].isspace() and not re.match(r"def\s", ln):
            skip = False
        if not skip:
            out.append(ln)
    return "\n".join(out)


def make_frozen_gen(model_name: str, temperature: float = 0.6, max_new_tokens: int = 220):
    """gen_fn(prompts)->texts on a FROZEN LM loaded via v5.lm_loader.load_frozen_lm.

    Use this over algo_lm_proposer.make_hf_gen on molab: make_hf_gen loads with device_map="auto",
    which SEGFAULTS on the box mid-weight-load (accelerate auto-placement, commit 0a1223a). load_frozen_lm
    loads then does an explicit .to(device) — the proven path. Model stays frozen (no training)."""
    import os
    import torch
    from transformers import AutoTokenizer
    from v5.lm_loader import load_frozen_lm
    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_frozen_lm(model_name)

    def gen(prompts: list[str]) -> list[str]:
        msgs = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                        add_generation_prompt=True) for p in prompts]
        enc = tok(msgs, return_tensors="pt", padding=True, padding_side="left").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, do_sample=True, temperature=temperature, top_p=0.95,
                                 max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
        return [tok.decode(out[i, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                for i in range(len(prompts))]
    return gen


def make_lm_compiler(gen_fn: Callable[[list[str]], list[str]]) -> Callable[[dict], str]:
    """Wrap a FROZEN generation fn (e.g. algo_lm_proposer.make_hf_gen) as `compile(spec)->code`.

    The LM writes only the entry glue over the curated helpers; the verified atom closure is prepended
    from the graph so the atoms cannot be corrupted. The LM is NEVER trained (frozen compiler)."""
    from v5.runtime.algo_lm_author import repair_code  # light import

    def compile_fn(spec: dict) -> str:
        prompt = render_compile_prompt(spec)
        text = gen_fn([prompt])[0]
        glue = repair_code(_extract_code(text), spec["entry"])
        atom_names = {a["name"] for a in spec.get("atoms", [])}
        glue = _strip_redefs(glue, atom_names)
        glue = strip_module_exec(glue)          # drop the LM's own print()/check() calls
        closure = "\n\n".join(a["code"].rstrip("\n") for a in spec.get("atoms", []))
        return (closure + "\n\n" + glue) if closure else glue
    return compile_fn


# ═══════════════════════════════════════════════════════════════════════════════
# The membrane solver
# ═══════════════════════════════════════════════════════════════════════════════

class MembraneSolver:
    """Frozen-compiler + membrane closed loop.

    The LM (compile_fn) receives ONLY the curated spec.atoms (the membrane). Retrieval is iterative
    and verifier-gated: each hop keeps the candidate that most raises realized test coverage; stop
    when coverage == 1.0 or no candidate helps (or budget). On final failure, re-reason with failure
    context; on exhausted budget, DERIVE (hand the LM a hole) and bank the verified result.
    """

    def __init__(self, graph: MemoryGraph, compile_fn: Callable[[dict], str],
                 retriever: TokenRetriever | None = None,
                 policy_fn: Callable | None = None,
                 k: int = 3, max_hops: int = 6, max_retries: int = 2):
        self.graph = graph
        self.compile_fn = compile_fn
        self.retriever = retriever or TokenRetriever(graph)
        # policy_fn(task, selected, graph, retriever) -> [(node_id, score), ...].
        # Default = the untrained iterative-cosine baseline. The trained TRM policy (build B) drops
        # in here with the SAME signature and retrieves the COMPLEMENT of the partial program.
        self.policy_fn = policy_fn or (lambda task, selected, graph, retr:
                                       retr.rank(task["text"], exclude=set(selected)))
        self.k = k
        self.max_hops = max_hops
        self.max_retries = max_retries
        self.compile_inputs: list[dict] = []  # audit: everything the LM ever saw (membrane check)

    # ── curate: build the spec the LM is allowed to see (selected atoms only) ──────
    def _curate(self, task: dict, selected: list[str], failure: dict | None = None) -> dict:
        closure = resolve_closure(self.graph, selected)
        atoms = [{"name": self.graph.nodes[nid].metadata.get("entry", nid),
                  "purpose": self.graph.nodes[nid].text,
                  "code": self.graph.nodes[nid].metadata.get("code", "")}
                 for nid in closure]
        spec = {"task_text": task["text"], "entry": task["entry"],
                "tests": task["tests"], "atoms": atoms}
        if failure:
            spec["failure"] = failure
        return spec

    def _coverage(self, task: dict, selected: list[str], failure: dict | None = None) -> tuple[float, str, str]:
        """Compile the curated spec and verify -> (fraction_pass, code, detail)."""
        spec = self._curate(task, selected, failure)
        self.compile_inputs.append(spec)              # audit every LM input
        code = self.compile_fn(spec)
        frac, detail = verify_code(code, task["entry"], task["tests"])
        return frac, code, detail

    # ── the loop ───────────────────────────────────────────────────────────────
    def solve(self, task: dict, min_score: float = 1e-3) -> dict:
        trace: list[dict] = []
        selected: list[str] = []
        cur_cov, cur_code, cur_detail = self._coverage(task, [])  # recipe alone (usually 0)

        # iterative, program-conditioned retrieval. Candidates are added SPECULATIVELY by rank
        # (a composition where neither atom alone yields partial credit still climbs); realized
        # coverage is the STOP signal, and un-called atoms are pruned at the end. Adding a
        # definition never lowers coverage, so speculative add is monotone-safe.
        for hop in range(self.max_hops):
            ranked = self.policy_fn(task, selected, self.graph, self.retriever)
            cand = next((c for c, s in ranked if s > min_score and c not in selected), None)
            if cand is None:
                break                                  # ret_stop: nothing else relevant to try
            selected.append(cand)
            cur_cov, cur_code, cur_detail = self._coverage(task, selected)
            trace.append({"hop": hop, "picked": cand, "coverage": cur_cov})
            # NOTE: the cosine baseline keeps a STABLE query (task text) + exclude-selected. Folding
            # selected purposes back into the query reinforces the already-covered concept and buries
            # the missing complement -> that program-conditioning is the TRM policy's job (build B),
            # which retrieves the COMPLEMENT; the untrained baseline must not fake it.
            if cur_cov >= 1.0:
                break

        # retries with failure context (LM re-compiles the SAME curated atoms, sees the error)
        retries = 0
        while cur_cov < 1.0 and retries < self.max_retries and selected:
            failure = {"code": cur_code, "error": cur_detail}
            cov, code, detail = self._coverage(task, selected, failure)
            retries += 1
            trace.append({"retry": retries, "coverage": cov})
            if cov > cur_cov:
                cur_cov, cur_code, cur_detail = cov, code, detail

        derived = False
        if cur_cov < 1.0:
            # DERIVE-on-gap: hand the LM a hole (frozen capability), verify, bank if it passes
            spec = self._curate(task, selected)
            spec["derive"] = True
            self.compile_inputs.append(spec)
            code = self.compile_fn(spec)
            frac, detail = verify_code(code, task["entry"], task["tests"])
            trace.append({"derive": True, "coverage": frac})
            if frac >= 1.0:
                cur_cov, cur_code, cur_detail, derived = frac, code, detail, True

        # PRUNE to the true reuse unit: over the full dependency CLOSURE of the selected atoms
        # (a selected atom can pull others transitively), keep exactly those whose entry is CALLED
        # in the verified code. Speculatively-tried-but-unused atoms — including a high-similarity
        # poison the gate never let compile — drop out here and are never banked.
        if cur_cov >= 1.0 and selected:
            closure = resolve_closure(self.graph, selected)
            ent = {nid: self.graph.nodes[nid].metadata.get("entry", nid) for nid in closure}
            reached = set(reuse_set(cur_code, task["entry"], set(ent.values())))
            selected = [nid for nid in closure if ent[nid] in reached]

        return {"solved": cur_cov >= 1.0, "coverage": cur_cov, "code": cur_code,
                "detail": cur_detail, "selected": selected, "derived": derived, "trace": trace}


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST — no GPU, no LM. Proves the membrane plumbing + anti-poison mechanism.
# ═══════════════════════════════════════════════════════════════════════════════

def _seed_graph() -> MemoryGraph:
    from v5.runtime.algo_grr_seed import build_graph
    g = build_graph()
    nodes = {n["id"]: Node.from_dict(n) for n in g["nodes"]}
    edges = [Edge.from_dict(e) for e in g["edges"]]
    return MemoryGraph(nodes, edges, metadata=g["metadata"])


def _selftest() -> bool:
    print("algo_grr_membrane --selftest: frozen-compiler + membrane closed loop\n")
    graph = _seed_graph()
    ok_all = True

    # Tasks: NL purpose + entry + I/O tests. Solutions require finding (and composing) seed atoms.
    #   single-atom : retrieve one atom, wrap it
    #   two-atom    : requires a SECOND hop (compose two atoms) -> tests iterative retrieval
    tasks = [
        dict(text="check whether a number is prime", entry="checkprime",
             tests=[((7,), True), ((8,), False), ((2,), True)]),
        dict(text="reverse the digits of an integer", entry="revnum",
             tests=[((123,), 321), ((100,), 1)]),
        dict(text="largest sum of a contiguous subarray", entry="maxsub",
             tests=[(([-2, 1, -3, 4, -1, 2, 1, -5, 4],), 6), (([1, 2, 3],), 6)]),
        # two-atom composition: count how many divisors of n are prime
        dict(text="count the divisors of n that are prime numbers", entry="nprimediv",
             tests=[((12,), 2), ((30,), 3), ((7,), 1)]),
    ]
    # Recipes stand in for what the FROZEN LM would infer from the curated spec.
    recipes = {
        "checkprime": "def checkprime(n):\n    return is_prime(n)\n",
        "revnum": "def revnum(n):\n    return reverse_digits(n)\n",
        "maxsub": "def maxsub(xs):\n    return max_subarray_sum(xs)\n",
        "nprimediv": ("def nprimediv(n):\n"
                      "    return sum(1 for d in divisors(n) if is_prime(d))\n"),
    }
    compiler = make_stub_compiler(recipes)

    # ── [1] each task solves; membrane selects the right atom(s) ──────────────────
    solver = MembraneSolver(graph, compiler)
    n_solved = 0
    for t in tasks:
        r = solver.solve(t)
        picked = [graph.nodes[s].metadata.get("entry", s) for s in r["selected"]]
        status = "OK" if r["solved"] else "FAIL"
        print(f"  [task] {t['entry']:12s} solved={r['solved']!s:5s} "
              f"cov={r['coverage']:.2f} atoms={picked}")
        if r["solved"]:
            n_solved += 1
    print(f"  [1] tasks solved: {n_solved}/{len(tasks)} -> {'PASS' if n_solved == len(tasks) else 'FAIL'}")
    ok_all &= (n_solved == len(tasks))

    # ── [2] iterative retrieval really fired on the 2-atom task ───────────────────
    r = MembraneSolver(graph, compiler).solve(tasks[-1])
    got = {graph.nodes[s].metadata.get("entry", s) for s in r["selected"]}
    multi = {"divisors", "is_prime"}.issubset(got)
    print(f"  [2] 2-atom task selected {sorted(got)} (needs divisors+is_prime) -> "
          f"{'PASS' if multi else 'FAIL'}")
    ok_all &= multi

    # ── [3] MEMBRANE HELD: the LM only ever saw CURATED atoms, never the full graph ─
    all_impl = {n.metadata.get("entry", nid) for nid, n in graph.nodes.items()
                if n.node_type == "implementation"}
    leaked = False
    for spec in solver.compile_inputs:
        seen = {a["name"] for a in spec["atoms"]}
        # every atom the LM saw must be within the selected closure (<= a few), never ~all 21
        if len(seen) > 6 or (seen and not seen.issubset(all_impl)):
            leaked = True
    print(f"  [3] membrane: {len(solver.compile_inputs)} LM specs, max atoms/spec "
          f"{max((len(s['atoms']) for s in solver.compile_inputs), default=0)}, "
          f"leak={leaked} -> {'PASS' if not leaked else 'FAIL'}")
    ok_all &= (not leaked)

    # ── [4] ANTI-POISON: inject a high-similarity WRONG atom; gate must reject it ──
    poison = Node(id="impl_prime_poison",
                  text="check whether a number is prime prime primality test number",
                  node_type="implementation", confidence=0.9, importance=0.9,
                  metadata={"code": "def prime_poison(n):\n    return True\n",  # WRONG: always True
                            "entry": "prime_poison", "kind": "atom", "origin": "poison"})
    pgraph = MemoryGraph(dict(graph.nodes, **{poison.id: poison}), list(graph.edges), graph.metadata)
    pcompiler = make_stub_compiler(dict(recipes, **{
        # if the membrane were fooled into using the poison, this is the code it'd compile:
        "checkprime_p": "def checkprime_p(n):\n    return prime_poison(n)\n"}))
    ptask = dict(text="check whether a number is prime", entry="checkprime",
                 tests=[((7,), True), ((8,), False), ((2,), True)])
    presolver = MembraneSolver(pgraph, pcompiler)
    pr = presolver.solve(ptask)
    picked = {pgraph.nodes[s].metadata.get("entry", s) for s in pr["selected"]}
    # poison ranks high on text, but coverage(poison) fails (8 is not prime -> True is wrong);
    # the real is_prime raises coverage to 1.0 -> membrane keeps the verified atom, drops poison.
    poison_rejected = "prime_poison" not in picked and pr["solved"]
    # confirm the poison WAS a top candidate (so rejection is by the gate, not by ranking luck)
    top = [c for c, _ in presolver.retriever.rank(ptask["text"])[:3]]
    poison_was_tempting = "impl_prime_poison" in top
    print(f"  [4] anti-poison: poison in top-3={poison_was_tempting}, final atoms={sorted(picked)}, "
          f"solved={pr['solved']} -> {'PASS' if poison_rejected and poison_was_tempting else 'FAIL'}")
    ok_all &= (poison_rejected and poison_was_tempting)

    # ── [5] frozen-LM compiler wrapper: membrane prompt + strip_redefs + verify (fake gen) ────
    is_prime_atom = {"name": "is_prime", "purpose": graph.nodes["impl_is_prime"].text,
                     "code": graph.nodes["impl_is_prime"].metadata["code"]}
    spec = {"task_text": "check whether a number is prime", "entry": "checkprime",
            "tests": [((7,), True), ((8,), False)], "atoms": [is_prime_atom]}
    prompt = render_compile_prompt(spec)
    membrane_ok = ("is_prime" in prompt and "reverse_digits" not in prompt
                   and "divisors" not in prompt)          # only the curated atom reaches the LM
    # fake frozen LM: redefines is_prime WRONGLY + writes the entry -> strip_redefs must drop the redef
    fake_gen = lambda prompts: ["```python\ndef is_prime(n):\n    return False\n"
                                "def checkprime(n):\n    return is_prime(n)\n```"]
    code = make_lm_compiler(fake_gen)(spec)
    frac, _ = verify_code(code, "checkprime", spec["tests"])
    authoritative = code.count("def is_prime(") == 1 and frac >= 1.0   # graph's atom won, LM redef gone
    print(f"  [5] frozen-LM compiler: membrane={membrane_ok}, atom-authoritative={authoritative}, "
          f"verify={frac:.2f} -> {'PASS' if membrane_ok and authoritative else 'FAIL'}")
    ok_all &= (membrane_ok and authoritative)

    print(f"\n  ALGO_GRR_MEMBRANE SELFTEST -> {'PASS' if ok_all else 'FAIL'}")
    return ok_all


# ═══════════════════════════════════════════════════════════════════════════════
# RUN — drive the membrane on a task set with the FROZEN 3B (molab)
# ═══════════════════════════════════════════════════════════════════════════════

def run_tasks(graph: MemoryGraph, tasks: list[dict], compile_fn: Callable[[dict], str],
              verbose: bool = True) -> dict:
    """Solve each task through the membrane; report solve-rate, reuse, and per-task LM calls."""
    solver = MembraneSolver(graph, compile_fn)
    solved = 0
    reuse_events = 0
    lm_calls_before = 0
    per = []
    for t in tasks:
        n0 = len(solver.compile_inputs)
        r = solver.solve(t)
        lm_calls = len(solver.compile_inputs) - n0
        picked = [graph.nodes[s].metadata.get("entry", s) for s in r["selected"]]
        reuse_events += len(picked)
        solved += int(r["solved"])
        per.append({"entry": t["entry"], "solved": r["solved"], "atoms": picked,
                    "lm_calls": lm_calls, "derived": r["derived"]})
        if verbose:
            print(f"  {t['entry']:16s} solved={r['solved']!s:5s} atoms={picked} "
                  f"lm_calls={lm_calls} derived={r['derived']}")
    out = {"n": len(tasks), "solved": solved, "reuse_events": reuse_events,
           "avg_lm_calls": sum(p["lm_calls"] for p in per) / max(1, len(per)), "per": per}
    if verbose:
        print(f"\n  solved {solved}/{len(tasks)}  reuse_events={reuse_events}  "
              f"avg_lm_calls={out['avg_lm_calls']:.2f}")
    return out


# Demo task set for --run: NL purpose + entry + I/O tests, solvable by composing seed atoms.
DEMO_TASKS = [
    dict(text="check whether a number is prime", entry="is_it_prime",
         tests=[((7,), True), ((8,), False), ((2,), True), ((1,), False)]),
    dict(text="reverse the digits of an integer", entry="rev_int",
         tests=[((123,), 321), ((100,), 1), ((5,), 5)]),
    dict(text="largest sum of a contiguous subarray (kadane)", entry="best_run",
         tests=[(([-2, 1, -3, 4, -1, 2, 1, -5, 4],), 6), (([1, 2, 3],), 6), (([-1, -2],), -1)]),
    dict(text="count how many divisors of n are prime numbers", entry="count_prime_divisors",
         tests=[((12,), 2), ((30,), 3), ((7,), 1), ((1,), 0)]),
    dict(text="least common multiple of two integers", entry="lcm_of",
         tests=[((4, 6), 12), ((3, 5), 15), ((7, 7), 7)]),
    dict(text="check if a string is an anagram of another", entry="are_anagrams",
         tests=[(("listen", "silent"), True), (("abc", "abd"), False)]),
]


def _load_seed(path: str) -> MemoryGraph:
    return MemoryGraph.load_json(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true", help="drive the membrane on a task set (frozen 3B)")
    ap.add_argument("--graph", default="graphs/grr_seed_clean.json")
    ap.add_argument("--lm", default="Qwen/Qwen2.5-3B-Instruct", help="frozen compiler model")
    ap.add_argument("--stub", action="store_true", help="use the no-GPU stub compiler instead of the LM")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(0 if _selftest() else 1)

    if a.run:
        graph = _load_seed(a.graph)
        if a.stub:
            # deterministic stub: recipes for the demo tasks (no GPU) — smoke the loop end to end
            compile_fn = make_stub_compiler({
                "is_it_prime": "def is_it_prime(n):\n    return is_prime(n)\n",
                "rev_int": "def rev_int(n):\n    return reverse_digits(n)\n",
                "best_run": "def best_run(xs):\n    return max_subarray_sum(xs)\n",
                "count_prime_divisors": "def count_prime_divisors(n):\n"
                                        "    return sum(1 for d in divisors(n) if is_prime(d))\n",
                "lcm_of": "def lcm_of(a, b):\n    return lcm(a, b)\n",
                "are_anagrams": "def are_anagrams(a, b):\n    return is_anagram(a, b)\n",
            })
        else:
            gen = make_frozen_gen(a.lm, temperature=0.6, max_new_tokens=220)
            compile_fn = make_lm_compiler(gen)
        print(f"membrane --run: graph={a.graph} lm={'stub' if a.stub else a.lm} "
              f"tasks={len(DEMO_TASKS)}\n")
        run_tasks(graph, DEMO_TASKS, compile_fn)
        return

    ap.print_help()


if __name__ == "__main__":
    main()
