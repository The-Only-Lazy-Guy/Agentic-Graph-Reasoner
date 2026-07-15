"""GRR-14: the LM AUTHORS NEW ATOMS — the invention rung, on real open-source tasks.

Measured motivation: MBPP+ pipeline-shaped human tasks through the current ladder = 1/40 (2%). Outside
its atom vocabulary the system is dead — composition cannot invent PRIMITIVES. This module adds the
final rung: when no pipeline over existing atoms can solve a task, the LM WRITES code.

  retrieve   MGRetriever advertises the graph's existing atoms for the task (reuse pressure: the prompt
             SHOWS what already exists, so the LM composes rather than reinvents)
  author     the LM writes the solution function (algo_graph_mg.solve_mg — the validated loop)
  GATE       the task's OWN dense tests (MBPP+ original asserts + the full EvalPlus script, subprocess)
             — GRR-1 epistemics on real data: solutions that only fit one assert die here
  BANK       the verified solution becomes an implementation node (origin="lm_author", text = the human
             task text = the retrieval key) + depend edges to every atom it CALLS + part_of the concept
             — health-gated through graph_grower like every other write
  reuse      later tasks retrieve earlier AUTHORED atoms; when a new solution calls one, that's
             CROSS-TASK REUSE — the compounding question, now on real data. Counted and reported.

Same contract as everywhere: the LM sees the task TEXT + advertised atom signatures — never reference
solutions. The gate, not the model, decides what enters memory.

  selftest (no GPU):  python -m v5.runtime.algo_lm_author --selftest
  molab (real LM):    python -m v5.runtime.algo_lm_author --run --model Qwen/Qwen2.5-3B-Instruct \
                          --limit 60 --graph graphs/algo_mbpp_author.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from graph_core import MemoryGraph
from v5.runtime.algo_graph_edits import edge_candidate, grow, node_candidate
from v5.runtime.algo_graph_mg import (
    MGRetriever,
    _edits_from_solve,
    _fn_name,
    _is_safe_atom_name,
    seed_graph,
)


def _author_prompt_purpose(task, advertised, purposes: dict) -> str:
    """Full-code advertisement with import framing and purpose annotations."""
    parts = [task.text]
    if advertised:
        parts.append("# ── Already imported — call these by name, do NOT redefine ──")
        for name, code in advertised:
            purpose = (purposes.get(name) or "").strip()
            if purpose:
                parts.append(f"# {purpose}\n```python\n{code}\n```")
            else:
                parts.append(f"```python\n{code}\n```")
    parts.append(f"\nWrite `{task.name}(...)` in ONE Python code block. You may use the imported "
                 f"functions above — they are already defined and ready to call.")
    return "\n\n".join(parts)


def _solve_author(retr: MGRetriever, gen_fn, task, k: int, samples: int, ad_style: str):
    """The author solve with pluggable advertisement (off | sig | purpose) and the raw-dump-diagnosed
    REPAIRS applied uniformly (repair_code: prose-in-fence trim + entry-name alias; the gate still
    decides). Mirrors solve_mg's verify/reward flow."""
    import re as _re
    from v5.runtime.algo_graph_run import _author_prompt, _task_verify, _top_defs
    from v5.runtime.derive_reward import _def_names, code_reward, grounded_code
    from v5.runtime.tool_memory import _extract_code
    advertised = [] if ad_style == "off" else retr.retrieve(task.text, k=k)
    if ad_style == "purpose" and advertised:
        purposes = {}
        for nid in retr.ids:
            node = retr.graph.nodes[nid]
            fn = _fn_name(node.metadata.get("code", "")) or nid
            purposes[fn] = (node.text or "").splitlines()[0][:120]
        prompt = _author_prompt_purpose(task, advertised, purposes)
    else:
        prompt = _author_prompt(task, advertised)
    adv_names = [n for n, _ in advertised]
    best = ("", [], False, "")
    attempts = []                                       # the DISCOVERY trajectory (search trace):
    for gen in gen_fn([prompt] * samples):              # every attempt + its verifier verdict/error
        code = repair_code(_extract_code(gen), task.name)
        defined = _def_names(code)
        called = [n for n in adv_names if n not in defined and _is_safe_atom_name(n)
                  and _re.search(rf"(?<![\w.]){_re.escape(n)}\s*\(", code)]
        deps = "\n\n".join(c for n, c in advertised if n in called)
        if _task_verify(task, code, deps):
            attempts.append({"gen": gen, "verified": True, "error": ""})
            best = (code, called, True, gen); break
        from v5.runtime.algo_graph_run import verify_asserts_detail
        _ok, err = verify_asserts_detail((deps + "\n" + code) if deps else code, task.tests,
                                         getattr(task, "setup", ""))
        attempts.append({"gen": gen, "verified": False, "error": err})
        if not best[0]:
            best = (code, called, False, gen)
    code, reused, verified, raw = best
    _, used = grounded_code(code, adv_names)
    new_helpers = [n for n in _top_defs(code) if n != task.name and n not in adv_names]
    R, _ = code_reward(verified, composed_used=used,
                       authored_new_verified=len(new_helpers) if verified else 0)
    return dict(name=task.name, verified=verified, reward=round(R, 3), reused=used, code=code, raw=raw,
                prompt=prompt, attempts=attempts)


def repair_code(code: str, entry: str) -> str:
    """Two raw-dump-diagnosed repairs, model untouched:
    1) prose-in-codeblock: the model writes explanation lines INSIDE the fence (sort_matrix ended with
       'Returns a sorted matrix based on...') -> trim trailing lines until the block compiles.
    2) name mismatch: the model defines find_volume for `find_Volume(...)` (snake_case instinct) ->
       if the entry is missing but exactly one def matches case-insensitively (or there is exactly one
       top-level def), append an alias binding. The GATE still decides — a wrong-logic alias fails."""
    if not code:
        return code
    lines = code.rstrip().splitlines()
    for _ in range(min(15, len(lines) - 1)):
        try:
            compile("\n".join(lines), "<gen>", "exec")
            break
        except SyntaxError:
            lines = lines[:-1]
    else:
        try:
            compile("\n".join(lines), "<gen>", "exec")
        except SyntaxError:
            return code                                  # unrecoverable — leave for the taxonomy
    fixed = "\n".join(lines)
    defs = re.findall(r"^def\s+([A-Za-z_]\w*)\s*\(", fixed, re.M)
    if entry and entry not in defs and defs:
        ci = [d for d in defs if d.lower() == entry.lower()]
        target = ci[0] if len(ci) == 1 else (defs[-1] if len(defs) == 1 else None)
        if target:
            fixed += f"\n\n{entry} = {target}"
    return fixed


def _called_atoms(code: str, atom_names) -> list:
    """Atoms the solution CALLS (not re-defines) — the depend edges + the reuse signal.
    (?<![\\w.]) guards method calls: `lst.count(x)` must NOT count as calling a banked atom named
    `count` (\\b matches after '.', which produced false-positive reuse events). Builtin-colliding
    names such as `sum` are also ignored: ordinary builtin calls are not graph reuse."""
    defined = set(re.findall(r"def\s+([A-Za-z_]\w*)\s*\(", code or ""))
    return [a for a in atom_names if a not in defined and _is_safe_atom_name(a)
            and re.search(rf"(?<![\w.]){re.escape(a)}\s*\(", code or "")]


# ── concept auto-classifier ─────────────────────────────────────────────────
_CONCEPT_KEYWORDS: list[tuple[str, str]] = [
    ("concept_number_theory",  r"prime|factor|divisib|digit|fibonacci|modulo|gcd|lcm|parity|odd|even"),
    ("concept_strings",        r"string|char|palindrome|anagram|substring|uppercase|lowercase|trim|regex|word|sentence"),
    ("concept_lists",          r"list|array|tuple|sequence|element|index|subarray|flatten|nested|matrix"),
    ("concept_math",           r"area|volume|perimeter|angle|triangle|circle|square|cube|root|power|log|ceil|floor"),
    ("concept_search_sort",    r"sort|search|binary.search|merge|heap|queue|stack|topolog|permut|combin"),
    ("concept_algorithms",     r""),            # default / catch-all
]


def _classify_concept(task_text: str) -> str:
    """Map a task's natural-language description to the best-fit concept node id.
    Simple ordered keyword match — first match wins, 'concept_algorithms' is the fallback."""
    t = task_text.lower()
    for concept_id, pattern in _CONCEPT_KEYWORDS:
        if pattern and re.search(pattern, t):
            return concept_id
    return "concept_algorithms"


# ── fn_name -> node_id reverse index ────────────────────────────────────────

def _fn_to_nid_map(graph) -> dict[str, str]:
    """Build {fn_name: node_id} from the live MemoryGraph so depend-edge targets use the real
    node ID (not the assumed impl_{fn_name} which diverges whenever the LM renamed its entry
    point — 23 such mismatches confirmed in the 68% run graph)."""
    mapping: dict[str, str] = {}
    for nid, node in graph.nodes.items():
        if node.node_type != "implementation":
            continue
        code = node.metadata.get("code", "")
        fn = _fn_name(code)
        if _is_safe_atom_name(fn):
            mapping[fn] = nid
    return mapping


def _bank_solution(graph_path: str, retr: MGRetriever, task, res_solve: dict, called: list,
                   session: str, concept: str | None = None):
    """A VERIFIED solve -> (a) the solution as an implementation node (origin=lm_author, task text =
    retrieval key) + depend edges to called atoms using the CORRECT target node IDs (via fn->nid
    reverse index, not assumed impl_{fn}); (b) the model's STORE-action HELPERS as their OWN atoms;
    (c) the task is auto-classified to one of 6 concept nodes instead of always concept_algorithms.
    Health-gated. Returns (banked_solution, helper_names)."""
    code = res_solve["code"]
    cands, helper_names = [], []
    nid = f"impl_{task.name}"

    # auto-classify to the right concept (the 68% run had 371/371 atoms under concept_algorithms)
    concept = concept or _classify_concept(task.text)
    # ensure the concept node exists (it may be absent in graphs seeded with only one concept)
    if concept not in retr.graph.nodes:
        concept = "concept_algorithms"    # safe fallback — always present in seed

    # build fn_name -> real node_id map so depend edges land on actual nodes, not guessed impl_{fn}
    fn_to_nid = _fn_to_nid_map(retr.graph)
    atom_names = list(fn_to_nid.keys())   # fn names that actually exist in the graph

    if nid not in retr.graph.nodes:
        # dedup: skip if a node with the same entry fn name already exists
        entry_fn = _fn_name(code)
        if not entry_fn or entry_fn not in fn_to_nid:
            cands.append(node_candidate(nid, code, task.text.splitlines()[0][:200], session,
                                        metadata={"kind": "authored", "origin": "lm_author"}))
            cands.append(edge_candidate(nid, concept, "part_of", session))
            for a in called:
                target_nid = fn_to_nid.get(a)          # use the REAL node id, not f"impl_{a}"
                if target_nid and target_nid != nid:
                    cands.append(edge_candidate(nid, target_nid, "depend", session))

    stores, _legacy_edges = _edits_from_solve(res_solve, task, concept)  # model-chosen helpers (Fix D)
    for hid, src, purpose in stores:
        if hid not in retr.graph.nodes:
            h_name = hid[len("impl_"):]
            # dedup: skip if a node with the same helper fn_name already exists
            if h_name in fn_to_nid:
                continue
            h_concept = _classify_concept(purpose + " " + task.text)
            if h_concept not in retr.graph.nodes:
                h_concept = concept
            cands.append(node_candidate(hid, src, purpose, session,
                                        metadata={"kind": "authored", "origin": "lm_author_helper"}))
            cands.append(edge_candidate(hid, h_concept, "part_of", session))
            helper_names.append(h_name)
            h_called = _called_atoms(src, atom_names)
            for a in h_called:
                target_nid = fn_to_nid.get(a)
                if target_nid and target_nid not in (nid, hid):
                    cands.append(edge_candidate(hid, target_nid, "depend", session))
            if nid in retr.graph.nodes or (entry_fn and entry_fn not in fn_to_nid):
                cands.append(edge_candidate(nid, hid, "depend", session))

    if not cands:
        return False, []
    newp = graph_path + ".grown"
    r = grow(graph_path, newp, cands)
    if r.get("persisted"):
        Path(newp).replace(graph_path)
        return True, helper_names
    return False, []


def _failure_class(t, code: str, deps: str) -> str:
    """Taxonomy for a miss: no_code | syntax | assert_fail (fails the original asserts) |
    plus_only_fail (passes the original asserts, fails the DENSE EvalPlus script = the gate catching a
    benchmark-overfit — plain-MBPP leaderboards would have counted this one as SOLVED)."""
    from v5.runtime.algo_graph_run import verify_asserts
    if not code or "def " not in code:
        return "no_code"
    try:
        compile(code, "<gen>", "exec")
    except SyntaxError:
        return "syntax"
    originals = [x for x in t.tests if x.lstrip().startswith("assert")]
    plus = [x for x in t.tests if not x.lstrip().startswith("assert")]
    full = (deps + "\n" + code) if deps else code
    if originals and verify_asserts(full, originals, getattr(t, "setup", "")):
        return "plus_only_fail" if plus else "assert_fail"
    return "assert_fail"


def dump_raw(graph_path: str, embed_fn, gen_fn, tasks, out: str = "artifacts/grr14_raw.jsonl",
             k_retrieve: int = 6, samples: int = 2, ad_style: str = "off"):
    """THE RAW PIPELINE, per task, nothing aggregated: exact prompt -> every generation verbatim ->
    extracted code -> per-sample verify verdict + failing line (stderr). Written to jsonl for
    inspection. This is the what-exactly-happened view; run it BEFORE arguing about causes."""
    import re as _re
    from v5.runtime.algo_graph_run import (_author_prompt, verify_asserts_detail)
    from v5.runtime.derive_reward import _def_names
    from v5.runtime.tool_memory import _extract_code
    if not Path(graph_path).exists():
        seed_graph(graph_path, ("concept_algorithms",))
    retr = MGRetriever(MemoryGraph.load_json(graph_path), embed_fn)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for i, t in enumerate(tasks):
            advertised = [] if ad_style == "off" else retr.retrieve(t.text, k=k_retrieve)
            prompt = _author_prompt(t, advertised)
            gens = gen_fn([prompt] * samples)
            rec = {"i": i, "task": t.name, "text": t.text, "prompt": prompt, "samples": []}
            for g in gens:
                raw_code = _extract_code(g)
                code = repair_code(raw_code, t.name)
                defined = _def_names(code)
                called = [n for n, _c in advertised if n not in defined and _is_safe_atom_name(n)
                          and _re.search(rf"(?<![\w.]){_re.escape(n)}\s*\(", code)]
                deps = "\n\n".join(c for n, c in advertised if n in called)
                full = (deps + "\n" + code) if deps else code
                ok, err = verify_asserts_detail(full, t.tests, getattr(t, "setup", ""))
                rec["samples"].append({"generation": g, "extracted_code": raw_code,
                                       "repaired_code": code if code != raw_code else None,
                                       "called": called, "verified": ok, "error": err})
            f.write(json.dumps(rec) + "\n")
            print(f"  [{i+1}/{len(tasks)}] {t.name}: "
                  f"{sum(s['verified'] for s in rec['samples'])}/{len(gens)} samples verified", flush=True)
    print(f"  raw dump -> {out}", flush=True)


def run_author_loop(graph_path: str, embed_fn, gen_fn, tasks, k_retrieve: int = 6, samples: int = 4,
                    reindex_every: int = 5, log: bool = True,
                    failure_log: str = "artifacts/grr14_failures.jsonl", ad_style: str = "sig",
                    shuffle_seed: int | None = None, progress_log: str = "",
                    traces_log: str = "", resume: bool = False, checkpoint_every: int = 0):
    """The loop over real tasks. Returns the report dict. The graph GROWS as it runs — atoms authored
    for early tasks are advertised (and reused) by later ones. Failures are classified + logged; the
    report prints the solve rate per 20-task bucket.

    LONG-RUN infrastructure (D3, molab resets every ~4h are a HARD design input):
      progress_log      per-task outcome jsonl (append) — the resume ledger
      resume=True       skip tasks already attempted per the ledger (a lost box costs minutes, not runs)
      traces_log        (prompt -> verified code) jsonl for SOLVED tasks — the STaR training set
      checkpoint_every  every N attempted tasks: flush logs + git-commit graph+logs locally (NO push —
                        pushing needs the user's credentials)"""
    import subprocess
    if not Path(graph_path).exists():
        seed_graph(graph_path, ("concept_algorithms",))
    if shuffle_seed is not None:                        # the corpus-ORDERING control
        import random
        tasks = list(tasks)
        random.Random(shuffle_seed).shuffle(tasks)
    attempted_before: set = set()
    if resume and progress_log and Path(progress_log).exists():
        with open(progress_log, encoding="utf-8") as f:
            attempted_before = {json.loads(line)["task"] for line in f if line.strip()}
        tasks = [t for t in tasks if t.name not in attempted_before]
    retr = MGRetriever(MemoryGraph.load_json(graph_path), embed_fn)
    solved = banked = 0
    authored_this_run: set = set()
    reuse_events = []                                   # (task, called earlier-authored atoms)
    since_reindex = 0
    buckets: list = []                                  # per-20-task solve counts (the decline curve)
    fail_counts: dict = {}
    Path(failure_log).parent.mkdir(parents=True, exist_ok=True)
    flog = open(failure_log, "w", encoding="utf-8")
    plog = open(progress_log, "a", encoding="utf-8") if progress_log else None
    tlog = open(traces_log, "a", encoding="utf-8") if traces_log else None

    def _checkpoint(i):
        for h in (flog, plog, tlog):
            if h:
                h.flush()
        try:
            paths = [graph_path, failure_log] + [p for p in (progress_log, traces_log) if p]
            subprocess.run(["git", "add", "-f", *paths], capture_output=True, timeout=60)
            subprocess.run(["git", "commit", "-m", f"checkpoint: author loop @{i} "
                            f"(solved {solved}, banked {banked})"], capture_output=True, timeout=60)
        except Exception:
            pass                                        # checkpointing must never kill the run

    for i, t in enumerate(tasks):
        if i % 20 == 0:
            buckets.append(0)
        res = _solve_author(retr, gen_fn, t, k=k_retrieve, samples=samples, ad_style=ad_style)
        if plog:
            plog.write(json.dumps({"task": t.name, "solved": bool(res["verified"])}) + "\n")
        if tlog and res["verified"]:
            # the STaR record carries the WHOLE discovery, not just the answer (targeting the answer
            # alone measurably hurt: holdout 70 -> 62.5): raw = winning generation with its reasoning
            # text; attempts = the search trace (failed tries + verifier errors) ending in the success
            tlog.write(json.dumps({"task": t.name, "prompt": res["prompt"], "code": res["code"],
                                   "raw": res.get("raw", ""),
                                   "attempts": res.get("attempts", [])}) + "\n")
        if checkpoint_every and (i + 1) % checkpoint_every == 0:
            _checkpoint(i + 1)
        if not res["verified"]:
            cls = _failure_class(t, res.get("code", ""), "")
            fail_counts[cls] = fail_counts.get(cls, 0) + 1
            flog.write(json.dumps({"i": i, "task": t.name, "class": cls,
                                   "graph_nodes": len(retr.graph.nodes),
                                   "code": (res.get("code") or "")[:1500]}) + "\n")
        if res["verified"]:
            solved += 1
            buckets[-1] += 1
            atom_names = [_fn_name(retr.graph.nodes[nid].metadata.get("code", "")) or nid
                          for nid in retr.ids]
            called = _called_atoms(res["code"], atom_names)
            reused_authored = [a for a in called if a in authored_this_run]
            if reused_authored:
                reuse_events.append((t.name, reused_authored))
            ok, helper_names = _bank_solution(graph_path, retr, t, res, called, f"grr14_{i}")
            if ok:
                banked += 1
                authored_this_run.add(t.name)
                authored_this_run.update(helper_names)  # helpers = the reuse-granular atoms
                since_reindex += 1
                if since_reindex >= reindex_every:      # new atoms become retrievable for later tasks
                    retr = MGRetriever(MemoryGraph.load_json(graph_path), embed_fn)
                    since_reindex = 0
        if log and (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(tasks)}] solved {solved} | banked {banked} | "
                  f"cross-task reuse events {len(reuse_events)}", flush=True)
    if since_reindex:
        retr = MGRetriever(MemoryGraph.load_json(graph_path), embed_fn)
    if checkpoint_every:
        _checkpoint(len(tasks))
    flog.close()
    for h in (plog, tlog):
        if h:
            h.close()
    bucket_sizes = [min(20, len(tasks) - 20 * b) for b in range(len(buckets))]
    report = dict(tasks=len(tasks), solved=solved, solve_rate=round(solved / max(1, len(tasks)), 3),
                  banked=banked, reuse_events=len(reuse_events),
                  reuse_detail=reuse_events[:10], graph_nodes=len(retr.graph.nodes),
                  curve=[round(s / max(1, n), 2) for s, n in zip(buckets, bucket_sizes)],
                  failures=fail_counts, k_retrieve=k_retrieve, ad_style=ad_style,
                  shuffle_seed=shuffle_seed)
    if log:
        print(f"\n  === GRR-14 report (ad_style={ad_style}, k={k_retrieve}, "
              f"shuffle={shuffle_seed}) ===", flush=True)
        print(f"  solved {solved}/{len(tasks)} ({report['solve_rate']:.0%}) — the no-authoring ladder "
              f"baseline was 2% (1/40)", flush=True)
        print(f"  SOLVE CURVE per 20 tasks (the over-time question): {report['curve']}", flush=True)
        print(f"  FAILURE TAXONOMY: {fail_counts}  (plus_only_fail = passed the original asserts, "
              f"killed by the DENSE gate — a plain-MBPP leaderboard counts those as solved)", flush=True)
        print(f"  atoms banked: {banked} (origin=lm_author, health-gated, depend edges to called atoms)",
              flush=True)
        print(f"  CROSS-TASK REUSE: {len(reuse_events)} events "
              f"{('e.g. ' + '; '.join(f'{n} called {c}' for n, c in reuse_events[:3])) if reuse_events else ''}",
              flush=True)
        print(f"  graph: {report['graph_nodes']} nodes | failure detail -> {failure_log}", flush=True)
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST (no GPU) — a "perfect author" stub proves the loop mechanics: verified solutions bank with
# provenance + depend edges; a later task REUSES an earlier authored atom (the compounding event);
# an unsolvable task banks NOTHING (the gate holds).
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    import tempfile
    import numpy as np
    from v5.runtime.algo_graph_run import MBPPTask
    print("algo_lm_author --selftest: author -> dense gate -> bank(origin) -> cross-task reuse\n")

    # [0] repair_code — the raw-dump-diagnosed fixes: prose trimmed, case alias appended, junk untouched
    r1 = repair_code("def f(x):\n    return x + 1\nThis returns x plus one.", "f")
    assert r1.rstrip().endswith("return x + 1"), r1
    r2 = repair_code("def find_volume(a, b):\n    return a * b", "find_Volume")
    assert r2.endswith("find_Volume = find_volume"), r2
    ns: dict = {}
    exec(r2, ns); assert ns["find_Volume"](3, 4) == 12
    assert repair_code("not code at all ][", "f") == "not code at all ]["
    print("  [0] repair_code: prose-in-fence trimmed, snake_case aliased to the required entry name, "
          "unrecoverable input left for the taxonomy -> PASS")

    assert _called_atoms("def f(xs):\n    return sum(xs)", ["sum"]) == []
    assert _called_atoms("def f(xs):\n    return digit_total(xs[0])", ["digit_total"]) == ["digit_total"]
    assert _called_atoms("def f(lst):\n    return lst.count(1)", ["count"]) == []
    print("  [0b] _called_atoms ignores builtins + method calls, keeps real graph calls -> PASS")

    # keyword-keyed stub embed (a fake random embedder retrieves nothing — t2 must FIND t1's atom)
    rng = np.random.default_rng(0)
    base = rng.standard_normal(64).astype("float32")

    def embed(d):
        out = {}
        for k, t in d.items():
            v = base if "decimal digit" in t.lower() else rng.standard_normal(64).astype("float32")
            out[k] = (v + 0.05 * rng.standard_normal(64)).astype("float32")
        return out

    t1 = MBPPTask("digit_total", "Compute the sum of decimal digits of n.\nWrite `digit_total(n)`.",
                  ["assert digit_total(123) == 6", "assert digit_total(9) == 9",
                   "assert digit_total(4051) == 10"])
    t2 = MBPPTask("digit_total_list",
                  "Sum of decimal digit sums across a list. ctx: sum of decimal digits of n",
                  ["assert digit_total_list([12, 34]) == 10", "assert digit_total_list([5]) == 5"])
    t3 = MBPPTask("impossible", "Return the 10th busy beaver number.\nWrite `impossible()`.",
                  ["assert impossible() == -1"])

    t4 = MBPPTask("count_odds", "Count the odd numbers in a list.\nWrite `count_odds(xs)`.",
                  ["assert count_odds([1, 2, 3]) == 2", "assert count_odds([2, 4]) == 0"])

    CODE1 = "def digit_total(n):\n    return sum(int(c) for c in str(abs(n)))"
    CODE2 = ("def digit_total_list(xs):\n    return sum(digit_total(x) for x in xs)")  # REUSES t1's atom
    CODE4 = ("def is_odd_h(n):\n    return n % 2 == 1\n\n"
             "def count_odds(xs):\n    return sum(1 for x in xs if is_odd_h(x))")

    def stub_gen(prompts):
        outs = []
        for p in prompts:
            if "digit_total_list" in p:
                outs.append(f"```python\n{CODE2}\n```")     # calls the ADVERTISED authored atom
            elif "digit_total" in p:
                outs.append(f"```python\n{CODE1}\n```")
            elif "count_odds" in p:
                # the author CHOOSES to store a helper (Fix D STORE action) -> reuse-granular atom
                outs.append(f"```python\n{CODE4}\n```\nSTORE is_odd_h: odd-number predicate helper")
            else:
                outs.append("```python\ndef impossible():\n    return 42\n```")   # fails its assert
        return outs

    with tempfile.TemporaryDirectory() as td:
        gp = str(Path(td) / "g.json")
        report = run_author_loop(gp, embed, stub_gen, [t1, t4, t2, t3], samples=1, reindex_every=1,
                                 log=False)
        g = MemoryGraph.load_json(gp)

        # [1] verified solutions banked with provenance; the failed one is NOT; the model's STORE
        #     helper became its OWN atom (the reuse-granular unit)
        assert report["solved"] == 3 and report["banked"] == 3, report
        assert "impl_digit_total" in g.nodes and "impl_impossible" not in g.nodes
        assert g.nodes["impl_digit_total"].metadata.get("origin") == "lm_author"
        assert g.nodes["impl_is_odd_h"].metadata.get("origin") == "lm_author_helper"
        print(f"  [1] 3/4 solved -> banked with origin=lm_author; STORE helper is_odd_h banked as its "
              f"OWN atom (origin=lm_author_helper); the gate-failing task banked NOTHING -> PASS")

        # [2] cross-task REUSE: t2's solution CALLS t1's authored atom; depend edge landed
        assert report["reuse_events"] == 1 and report["reuse_detail"][0][1] == ["digit_total"], report
        assert g.edge_between("impl_digit_total_list", "impl_digit_total") is not None
        print(f"  [2] cross-task reuse: digit_total_list CALLS the authored digit_total "
              f"(+ depend edge in the graph) — compounding on authored knowledge -> PASS")

        # [3] the banked code is the runnable knowledge: resolve deps through the graph and execute
        retr = MGRetriever(g, embed)
        deps = retr.resolve_deps(["digit_total_list"])
        ns: dict = {}
        exec(deps, ns)
        assert ns["digit_total_list"]([12, 34]) == 10
        print(f"  [3] graph walk resolves the authored dependency chain; code executes -> PASS")

    print("\n  ALGO_LM_AUTHOR SELFTEST -> PASS  (the LM invents primitives; the gate decides; the "
          "graph compounds)")
    return True


def main():
    ap = argparse.ArgumentParser(description="GRR-14: LM authors new atoms on MBPP+ (gated, banked, reused).")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true", help="real LM over the prepped MBPP+ corpus (molab)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--corpus", default="artifacts/mbpp_plus_prepped.jsonl")
    ap.add_argument("--graph", default="graphs/algo_mbpp_author.json")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--pipeline-only", action="store_true", help="restrict to pipeline-shaped tasks")
    ap.add_argument("--ad-style", default="off", choices=["off", "sig", "purpose"],
                    help="advertisement arm: off=DEFAULT (measured: sig ads cost ~16pp by task 90 and "
                         "cause the over-time decline) | sig=bare-sig status quo | purpose=repaired ads")
    ap.add_argument("--shuffle", type=int, default=-1, help="shuffle tasks with this seed (-1 = corpus order)")
    ap.add_argument("--dump-raw", type=int, default=0, metavar="N",
                    help="raw pipeline dump for the first N tasks (prompt/generations/verify verbatim)")
    ap.add_argument("--progress-log", default="artifacts/grr14_progress.jsonl")
    ap.add_argument("--traces-log", default="artifacts/grr14_traces.jsonl")
    ap.add_argument("--resume", action="store_true", help="skip tasks already in the progress ledger")
    ap.add_argument("--checkpoint-every", type=int, default=50,
                    help="git-commit graph+logs locally every N tasks (0 = off; never pushes)")
    ap.add_argument("--lora", default="", help="trained PEFT adapter dir (loaded on top of --model)")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    if a.dump_raw:
        from v5.memory.store import make_mpnet_embedder
        from v5.runtime.algo_lm_proposer import make_hf_gen
        from v5.runtime.algo_mbpp_prep import load_prepped
        tasks = load_prepped(a.corpus, limit=a.dump_raw, pipeline_only=a.pipeline_only)
        dump_raw(a.graph, make_mpnet_embedder(), make_hf_gen(a.model, max_new_tokens=400, lora_path=a.lora),
                 tasks, ad_style=a.ad_style, samples=a.samples)
        return
    if a.run:
        from v5.memory.store import make_mpnet_embedder
        from v5.runtime.algo_lm_proposer import make_hf_gen
        from v5.runtime.algo_mbpp_prep import load_prepped
        tasks = load_prepped(a.corpus, limit=a.limit, pipeline_only=a.pipeline_only)
        print(f"GRR-14 author loop (real LM {a.model}): {len(tasks)} MBPP+ tasks | graph {a.graph} | "
              f"ad_style={a.ad_style} shuffle={a.shuffle} lora={'yes' if a.lora else 'no'}", flush=True)
        run_author_loop(a.graph, make_mpnet_embedder(), make_hf_gen(a.model, max_new_tokens=400, lora_path=a.lora),
                        tasks, samples=a.samples, ad_style=a.ad_style,
                        shuffle_seed=None if a.shuffle < 0 else a.shuffle,
                        progress_log=a.progress_log, traces_log=a.traces_log, resume=a.resume,
                        checkpoint_every=a.checkpoint_every)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
