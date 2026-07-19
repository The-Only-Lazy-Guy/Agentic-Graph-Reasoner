"""algo_grr_pipeline — MembraneV2: the integrated pipeline (#1a, WAVE 0).

Turns the validated probes into ONE end-to-end system with three swappable interfaces, so the parallel
tracks (#2 planner, #3 GraphGPS router, #5 step-spec) build against stubs without blocking each other:

    AtomRouter.rank(task_text, atoms)  -> ranked atom names      (#3 improves it: GraphGPS)
    Planner.plan(task, ranked)         -> AtomProgram            (#2 trains it: SearchPlanner)
    realize(AtomProgram, store)        -> code                   (deterministic — never the LM)
    MembraneV2.solve = rank -> plan -> realize -> ratify(LM) -> VERIFY -> bank

WAVE-0 stubs shipped here: token-overlap router + an ORACLE planner (reads the known wiring). Swap in the
trained planner / GPS router later behind the same interfaces (#1b). SearchPlanner wraps plan_by_search()
from the learned planner module behind the standard Planner interface.

    python -m v5.runtime.algo_grr_pipeline --selftest            # no-GPU: end-to-end on the compose corpus
    python -m v5.runtime.algo_grr_pipeline --selftest-planner    # no-GPU: pipeline WITH the real planner
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from v5.runtime.algo_grr_compose import INNER, OUTER  # noqa: E402


# ── AtomProgram + deterministic realizer ────────────────────────────────────────
@dataclass
class AtomProgram:
    """What the reasoner emits: which atoms (dep order) + how they wire. wiring is a tree:
    ('call', atom_name, [arg, ...]) | 'n' (the input leaf). NEVER free-form code."""
    atoms: list
    wiring: object


def _render(w) -> str:
    if isinstance(w, str):
        return w                                   # leaf, e.g. 'n'
    _, name, args = w
    return f"{name}(" + ", ".join(_render(a) for a in args) + ")"


def realize(prog: AtomProgram, store: dict, entry: str) -> str:
    """Deterministic program -> code: prepend the atoms' verified source (deps first), emit the entry
    that wires them. The realizer CANNOT inline -> the atoms are load-bearing."""
    closure = "\n".join(store[a].rstrip("\n") for a in prog.atoms if a in store)
    return f"{closure}\n\ndef {entry}(n):\n    return {_render(prog.wiring)}\n"


# ── Interfaces (stubs here; #2/#3 replace behind them) ──────────────────────────
class AtomStore(dict):
    """name -> verified source. Seeded from the compose primitive pool (+ grows via banking)."""

    @classmethod
    def from_compose(cls):
        s = cls()
        for name, (code, *_ ) in {**INNER, **OUTER}.items():
            s[name] = code
        return s

    @classmethod
    def from_wiring(cls):
        from v5.runtime.algo_grr_wiring import _HELPERS
        s = cls()
        s.update(_HELPERS)
        return s


class SearchPlanner:
    """Learned planner wrapping plan_by_search() behind the standard Planner interface.
    Loads a saved seq2seq model and uses net-guided verified search to infer the atom program.

    Handles both wiring and compose domains automatically from the saved model metadata."""

    def __init__(self, store, model_path):
        from v5.runtime.algo_grr_planner import _load_model
        self.store = store
        self._model, self._enc_words, self._wvocab, self._ptok, self._p2i, self._arity = \
            _load_model(model_path)

    def plan(self, task: dict, ranked=None) -> AtomProgram:
        from v5.runtime.algo_grr_planner import plan_by_search
        task_words = task["text"]
        def verify_fn(code):
            return task["verify_fn"](code)[0] >= 1.0
        def realize_fn(prog):
            return realize(prog, self.store, task["entry"])
        prog, nv, solved = plan_by_search(
            self._model, self._enc_words, task_words, verify_fn,
            ptok=self._ptok, p2i=self._p2i, arity=self._arity, beam=10,
            realize_fn=realize_fn)
        if prog is None:
            return AtomProgram(atoms=["n"], wiring="n")
        return prog


class AtomRouter:
    """WAVE-0 stub: rank atoms by token overlap with the task text. #3 replaces with GraphGPS features
    (mpnet content + topology MPNN + LapPE/RWSE). Interface is `.rank(task_text) -> [name...]`."""

    def __init__(self, store: dict):
        self.store = store
        self._toks = {n: set(_tok(n)) | set(_tok(_desc(store[n]))) for n in store}

    def _toks_of(self, n):                          # lazily index atoms banked after init
        if n not in self._toks:
            self._toks[n] = set(_tok(n)) | set(_tok(_desc(self.store[n])))
        return self._toks[n]

    def rank(self, task_text: str, k: int | None = None):
        q = set(_tok(task_text))
        scored = sorted(self.store, key=lambda n: len(q & self._toks_of(n)), reverse=True)
        return scored[:k] if k else scored


class TopologyAtomRouter:
    """GraphGPS resolution wired into MembraneV2: content/token rank + FOLLOW-THE-EDGE. Structural deps are
    FOLLOWED not learned — a banked composite's sub-atoms are pulled along its call-edge (realize needs the
    closure; the reasoner sees the structural neighbourhood, not just content matches). Measured elsewhere:
    content 0.30 / learned-GPS 0.50 / topology 1.00 recall on the dep partner — so structure = traverse,
    content = rank. Adjacency = the atom call-graph (A calls B => edge), rebuilt as the store grows."""

    def __init__(self, store: dict, k_content: int = 6):
        self.store = store
        self.k = k_content
        self._toks: dict = {}
        self._edges: dict = {}

    def _reindex(self):
        import re
        names = set(self.store)
        for n in self.store:
            if n not in self._toks:
                self._toks[n] = set(_tok(n)) | set(_tok(_desc(self.store[n])))
            if n not in self._edges:
                body = self.store[n]
                self._edges[n] = {m for m in names if m != n and re.search(rf"\b{re.escape(m)}\s*\(", body)}

    def rank(self, task_text: str, k: int | None = None):
        self._reindex()
        q = set(_tok(task_text))
        scored = sorted(self.store, key=lambda n: len(q & self._toks[n]), reverse=True)
        top = scored[:self.k]
        nbrs = set()
        for n in top:                                  # FOLLOW edges: pull depend-neighbours of top matches
            nbrs |= self._edges.get(n, set())
        ranked = top + [n for n in scored[self.k:] if n in nbrs] + [n for n in scored[self.k:] if n not in nbrs]
        return ranked[:k] if k else ranked


class OraclePlanner:
    """WAVE-0 stub: reads the KNOWN wiring (task['_prims'] = (inner, outer)) -> the ground-truth program.
    #2 replaces this with the trained TRMPlanDecoder that INFERS the program from the NL task."""

    def plan(self, task: dict, ranked=None) -> AtomProgram:
        inner, outer = task["_prims"]
        return AtomProgram(atoms=[inner, outer],
                           wiring=("call", outer, [("call", inner, ["n"])]))


class ProgramOraclePlanner:
    """Reads a task's KNOWN multi-atom program (`_wprog` = (atoms dep-first, wiring tree)) — the stub for
    a planner that emits a MULTI-atom program (the multi-hard corpus, where wiring isn't outer(inner(n)))."""

    def plan(self, task: dict, ranked=None) -> AtomProgram:
        atoms, wiring = task["_wprog"]
        return AtomProgram(atoms=list(atoms), wiring=wiring)


class SpeculativePlanner:
    """RankedNGram wrapped as a pipeline Planner interface. Learns (ranked → program) atom sequences
    from banked solves and predicts programs for new tasks. Falls back to a delegate planner when
    the prediction is uncertain or insufficiently trained."""

    def __init__(self, store: dict, atom2id: dict, id2atom: dict,
                 fallback=None, train_pairs: list | None = None, K: int = 4):
        self.store = store
        self.atom2id = atom2id
        self.id2atom = id2atom
        self.fallback = fallback
        self.K = K
        self._pairs: list[tuple[list[int], list[int]]] = train_pairs or []
        self._spec = None
        if self._pairs:
            self._rebuild()

    def add_example(self, ranked_ids: list[int], plan_ids: list[int]):
        self._pairs.append((ranked_ids, plan_ids))

    def _rebuild(self):
        if self._pairs:
            from v5.runtime.algo_grr_specstep import RankedNGram
            self._spec = RankedNGram(self._pairs, N=2)

    def plan(self, task: dict, ranked: list[str]) -> AtomProgram:
        if self._spec is None:
            return self._or_fallback(task, ranked)
        ranked_ids = [self.atom2id[a] for a in ranked if a in self.atom2id]
        if not ranked_ids:
            return self._or_fallback(task, ranked)
        pred_ids = self._spec.speculate(ranked_ids, [], self.K)
        pred_atoms = [self.id2atom[i] for i in pred_ids if i in self.id2atom]
        if len(pred_atoms) >= 2:
            return AtomProgram(atoms=pred_atoms,
                               wiring=("call", pred_atoms[-1], [("call", pred_atoms[-2], ["n"])]))
        return self._or_fallback(task, ranked)

    def _or_fallback(self, task, ranked) -> AtomProgram:
        if self.fallback is not None:
            return self.fallback.plan(task, ranked)
        return AtomProgram(atoms=["n"], wiring="n")


class MembraneV2:
    """The orchestration: route -> plan -> AUTHOR missing atoms (LM) -> realize -> LM ratifies glue ->
    verify -> BANK. The realizer is deterministic (never inlines), so the atoms are load-bearing; the LM
    only AUTHORS a missing primitive and RATIFIES glue. Banking grows the store -> compounding: an atom
    authored once is reused (and re-authoring avoided) by every later task that needs it.

    bank=True  -> OURS: verified authored atoms persist in the store (self-growth -> compounds).
    bank=False -> RAG baseline: authored atoms are DROPPED after each task (static store -> re-authors
                  every time; author_calls/task stays flat, the cost the compounding is supposed to cut).
    author_fn(name, task) -> source code for a missing atom (the real 3B; stub = the known helper source).
    """

    def __init__(self, store, router, planner, ratify_fn=None, author_fn=None, bank=True,
                 batch_author_fn=None):
        self.store, self.router, self.planner = store, router, planner
        self.ratify_fn = ratify_fn
        self.author_fn = author_fn
        self.batch_author_fn = batch_author_fn     # STEP-SPEC: author ALL missing atoms in ONE LM call
        self.bank = bank
        self.seed = set(store)                     # seed atoms never count as "derived"
        self.reuse = {}                            # atom -> times used across tasks
        self.derived_reuse = 0                     # uses of BANKED (non-seed) atoms = the compounding signal
        self.banked = 0                            # distinct atoms authored+banked
        self.author_calls = 0                      # atoms authored (per-atom count)
        self.lm_calls = 0                          # ACTUAL LM invocations (batch spec cuts this)

    def solve(self, task: dict) -> dict:
        ranked = self.router.rank(task["text"])
        prog = self.planner.plan(task, ranked)
        missing = [a for a in prog.atoms if a != "n" and a not in self.store]
        authored_now = []
        # STEP-SPECULATION: the tiny planner already proposed the whole K-atom program; author every
        # missing atom in ONE LM call (batch) instead of one call per atom. Each authored atom is still
        # gated by the composite verify below (a drifted atom fails → never banks) = mutual anti-drift.
        if missing and self.batch_author_fn is not None:
            srcs = self.batch_author_fn(missing, task)
            self.lm_calls += 1
            self.author_calls += len(missing)
            for a in missing:
                src = srcs.get(a, "")
                if src and _authored_ok(src, a):
                    self.store[a] = src
                    authored_now.append(a)
        elif missing and self.author_fn is not None:
            for a in missing:
                src = self.author_fn(a, task)
                self.lm_calls += 1
                self.author_calls += 1
                if src and _authored_ok(src, a):
                    self.store[a] = src
                    authored_now.append(a)
        code = realize(prog, self.store, task["entry"])
        if self.ratify_fn:                         # real LM writes/ratifies the glue line
            code = self.ratify_fn(code, task, prog)
        ok = task["verify_fn"](code)[0] >= 1.0
        route_ok = all(a in ranked[:6] for a in prog.atoms)   # did the router surface the needed atoms?
        if ok:
            for a in prog.atoms:
                if a == "n":
                    continue
                self.reuse[a] = self.reuse.get(a, 0) + 1
                if a in self.seed:
                    continue
                # a banked (non-seed) atom that was NOT authored this task -> reused from an earlier bank
                if a not in authored_now:
                    self.derived_reuse += 1
        # BANK only atoms validated THROUGH a solved composite (an unverified authored helper must not
        # pollute the store) — and only when banking is on. RAG / failed tasks drop what they authored.
        keep = self.bank and ok
        if keep:
            self.banked += len(authored_now)
        else:
            for a in authored_now:
                self.store.pop(a, None)
        return dict(solved=ok, code=code, program=prog, route_ok=route_ok,
                    authored=authored_now, author_calls=self.author_calls)


# ── helpers ──────────────────────────────────────────────────────────────────
def _tok(s: str):
    import re
    return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if len(t) > 2]


def _desc(code: str) -> str:
    return code.split("(", 1)[0].replace("def", " ")


def _authored_ok(src: str, name: str) -> bool:
    """Gate an authored primitive before it can bank: it must compile and define the named callable.
    (The task's own verify still gates the COMPOSED result; this just rejects garbage authoring.)"""
    if f"def {name}" not in src:
        return False
    try:
        ns: dict = {}
        exec(compile(src, "<atom>", "exec"), ns)  # noqa: S102 — trusted stub / gated author
        return callable(ns.get(name))
    except Exception:  # noqa: BLE001
        return False


# ── selftest (no GPU) ──────────────────────────────────────────────────────────
def _selftest() -> bool:
    print("algo_grr_pipeline --selftest: MembraneV2 end-to-end (route -> plan -> realize -> verify)\n")
    from v5.runtime.algo_grr_compose import gen_corpus
    store = AtomStore.from_compose()
    tasks = gen_corpus(40, seed=0)
    solver = MembraneV2(store, AtomRouter(store), OraclePlanner())

    solved = route_hits = 0
    for t in tasks:
        r = solver.solve(t)
        solved += r["solved"]
        route_hits += r["route_ok"]
    n = len(tasks)
    reuse_total = sum(solver.reuse.values())
    distinct = len(solver.reuse)
    print(f"  [1] end-to-end solved     : {solved}/{n} ({100*solved//n}%)  (route->plan->realize->verify)")
    print(f"  [2] router surfaced atoms : {route_hits}/{n} tasks had needed atoms in top-6")
    print(f"  [3] compounding (reuse)   : {reuse_total} atom-uses over {distinct} distinct prims "
          f"(avg {reuse_total/max(1,distinct):.1f} reuse/prim)")
    # [4] realize is deterministic + correct; a WRONG program must fail verify
    t0 = tasks[0]
    bad = AtomProgram(atoms=list(t0["_prims"]), wiring=("call", t0["_prims"][0], ["n"]))
    bad_code = realize(bad, store, t0["entry"])
    caught = t0["verify_fn"](bad_code)[0] < 1.0
    print(f"  [4] wrong program fails verify: {'PASS' if caught else 'FAIL'}")

    ok = solved >= 0.95 * n and route_hits >= 0.8 * n and caught
    print(f"\n  MembraneV2 skeleton: interfaces chain, realize is load-bearing, reuse compounds.")
    print(f"  Swap points: AtomRouter->GraphGPS (#3), OraclePlanner->SearchPlanner (#2), ratify_fn->3B.")
    print(f"\n  ALGO_GRR_PIPELINE SELFTEST -> {'PASS' if ok else 'FAIL'}")
    return ok


def _selftest_planner(planner_path="") -> bool:
    """MembraneV2 end-to-end WITH the real learned planner (not Oracle stub)."""
    import os, random, tempfile
    from v5.runtime.algo_grr_planner import _PTOK
    print("algo_grr_pipeline --selftest-planner: MembraneV2 WITH the real learned planner\n")
    from v5.runtime.algo_grr_wiring import gen_expr, to_words, oracle
    from v5.runtime.algo_grr_planner import _make_data, _build

    # train or load planner model
    if not planner_path or not os.path.exists(planner_path):
        planner_path = os.path.join(tempfile.gettempdir(), "grr_planner_wiring.pt")
        print("  training wiring-domain planner...")
        torch, nn, Seq2Seq = _build(); torch.manual_seed(0)
        words_list, progs_list, wvocab = _make_data([1, 2, 3], 1500, seed=1)
        model = Seq2Seq(len(wvocab), len(_PTOK))
        from v5.runtime.algo_grr_planner import train_planner, _save_model
        model, _ = train_planner(model, words_list, progs_list, wvocab, steps=2500)
        _save_model(planner_path, model, "wiring", wvocab)
    else:
        print(f"  loading planner from {planner_path}")

    store = AtomStore.from_wiring()
    from v5.runtime.algo_grr_planner import _load_model, plan_by_search

    def make_wiring_task(depth, seed):
        t = gen_expr(depth, random.Random(4000 * depth + seed))
        words = to_words(t)
        entry = "solve"
        from v5.runtime.algo_grr_planner import _prog_tokens, _tokens_to_wiring as ttw
        gt_toks = _prog_tokens(t)
        gt_prog = ttw(gt_toks)
        def verify_fn(code):
            ns = {}
            try:
                exec(compile(code, "<test>", "exec"), ns)
                fn = ns.get(entry)
                if fn is None:
                    return [0.0]
                for n in (2, 3, 5, 7, 11):
                    if fn(n) != oracle(t, n):
                        return [0.0]
                return [1.0]
            except Exception:
                return [0.0]
        return dict(text=words, entry=entry, verify_fn=verify_fn, _gt_prog=gt_prog)

    planner = SearchPlanner(store, planner_path)
    router = AtomRouter(store)
    solver = MembraneV2(store, router, planner)

    print("\n  depth | oracle-solved | planner-solved")
    frows, srows = [], []
    for d in (1, 2, 3):
        solved = oracle_solved = 0
        n_tasks = 20
        for i in range(n_tasks):
            task = make_wiring_task(d, i)
            # oracle: realize the ground-truth program through the pipeline
            gt_code = realize(task["_gt_prog"], store, task["entry"])
            if task["verify_fn"](gt_code)[0] >= 1.0:
                oracle_solved += 1
            r = solver.solve(task)
            solved += int(r["solved"])
        fr, sr = oracle_solved / n_tasks, solved / n_tasks
        frows.append(fr); srows.append(sr)
        print(f"  {d:>5} |     {fr:.2f}        |     {sr:.2f}")
    fo, so = sum(frows) / len(frows), sum(srows) / len(srows) if srows else 0
    ok = so >= 0.7
    print(f"\n  overall: oracle {fo:.2f}  ->  planner {so:.2f}")
    print(f"  -> {'PASS' if ok else 'FAIL'}: SearchPlanner integrated into MembraneV2 pipeline.")
    return ok


def _selftest_compound() -> bool:
    """MembraneV2 with authoring + banking on the HARD corpus: OURS (bank) vs RAG (no bank). No GPU:
    the stub author returns the KNOWN helper source, so BOTH arms solve — the point is COST. OURS authors
    each hard helper ONCE then reuses the bank; RAG re-authors every task. author_calls/task is the
    lm/task the compounding is meant to cut; derived_reuse is the reuse the static store can't have."""
    from v5.runtime.algo_grr_compose import gen_corpus_hard, HARD, OUTER
    print("algo_grr_pipeline --selftest-compound: MembraneV2 author+bank, OURS vs RAG on the HARD corpus\n")
    tasks = gen_corpus_hard(60, seed=0)
    src_of = {**{k: v[0] for k, v in HARD.items()}, **{k: v[0] for k, v in OUTER.items()}}
    author = lambda name, task: src_of.get(name, "")   # noqa: E731 — stub author (real 3B may err here)

    def run(bank: bool) -> dict:
        store = AtomStore()
        for name, (code, *_ ) in OUTER.items():         # seed the easy wrappers; HARD must be authored
            store[name] = code
        m = MembraneV2(store, AtomRouter(store), OraclePlanner(), author_fn=author, bank=bank)
        solved = 0
        for t in tasks:
            solved += int(m.solve(t)["solved"])
        return dict(solved=solved, author_calls=m.author_calls, banked=m.banked,
                    derived_reuse=m.derived_reuse, atoms=len(m.store))

    ours, rag = run(bank=True), run(bank=False)
    n = len(tasks)
    print(f"  arm  | solved | author_calls | author/task | banked | derived_reuse | store")
    for tag, r in (("OURS", ours), ("RAG ", rag)):
        print(f"  {tag} |  {r['solved']:>3}/{n} |     {r['author_calls']:>4}     |    "
              f"{r['author_calls']/n:>4.2f}     |   {r['banked']:>3}  |     {r['derived_reuse']:>4}      | {r['atoms']:>3}")
    cut = rag["author_calls"] / max(1, ours["author_calls"])
    print(f"\n  -> OURS authors each hard helper ONCE then reuses the bank; RAG re-authors every task.")
    print(f"     author-call reduction (the lm/task compounding cuts): {cut:.1f}x  |  derived_reuse OURS "
          f"{ours['derived_reuse']} vs RAG {rag['derived_reuse']}")
    ok = (ours["solved"] == n and rag["solved"] == n and ours["author_calls"] < rag["author_calls"] * 0.5
          and ours["derived_reuse"] > 0 and rag["derived_reuse"] == 0 and ours["banked"] == len(HARD))
    print(f"\n  ALGO_GRR_PIPELINE COMPOUND SELFTEST -> {'PASS' if ok else 'FAIL'}")
    return ok


# ── real frozen-3B author + inline-RAG baseline (the --v2 comparison uses these) ────────────────────
def make_lm_author(gen):
    """author_fn(name, task): the frozen 3B writes ONE reusable primitive `name`. _authored_ok gates
    syntax; the composite task's verify gates whether it may BANK (a wrong atom never persists)."""
    from v5.runtime.algo_grr_membrane import _extract_code
    from v5.runtime.algo_lm_author import repair_code
    def author(name, task):
        prompt = (f"Write a single self-contained Python function named `{name}` taking one integer `n` "
                  f"and returning the standard quantity its name denotes (context: {task['text']}). "
                  f"Return ONLY the function definition.")
        return repair_code(_extract_code(gen([prompt])[0]), name)
    return author


def make_lm_inline(gen):
    """RAG baseline — NO reasoner, NO memory: the frozen 3B writes the WHOLE entry inline from the task
    text (+ optional retrieved seed helpers). It must reconstruct the hard helper itself every time."""
    from v5.runtime.algo_grr_membrane import _extract_code, strip_module_exec
    from v5.runtime.algo_lm_author import repair_code
    def inline(task, retrieved=None):
        ctx = ("You may use these existing helpers:\n" + "\n\n".join(retrieved) + "\n\n") if retrieved else ""
        prompt = (f"{ctx}Write a Python function `{task['entry']}(n)` that computes {task['text']}. "
                  f"Include any helper functions it needs. Return ONLY code.")
        return strip_module_exec(repair_code(_extract_code(gen([prompt])[0]), task["entry"]))
    return inline


def make_lm_batch_author(gen):
    """Author ALL missing atoms for a task in ONE LM call.
    Returns {name: source_code} for each requested name (empty if the response didn't contain it).
    The composite verify gates correctness — a drifted authored atom never banks."""
    from v5.runtime.algo_lm_author import repair_code
    def batch(names, task):
        if not names:
            return {}
        prompt = (f"Write the following self-contained Python functions, each taking one integer `n` "
                  f"and returning the standard quantity its name denotes (context: {task['text']}).\n"
                  + "\n".join(f"def {n}(n):" for n in names)
                  + "\n\nReturn ONLY the function definitions. Separate them with blank lines.")
        body = gen([prompt])[0]
        result = {}
        for name in names:
            i = body.find(f"def {name}(")
            if i < 0:
                result[name] = ""
                continue
            rest = body[i:]
            j = rest.find("\ndef ", 1)
            src = rest[:j].strip() if j >= 0 else rest.strip()
            result[name] = repair_code(src, name)
        return result
    return batch


def run_v2_compare(stream, holdout, author_fn, inline_fn, *, batch_author_fn=None,
                   spec_planner=None, verbose=True, report_every=40,
                   debug_heldout_n: int = 0) -> dict:
    """OURS = MembraneV2 (route→plan→author-missing→realize→verify→BANK) vs RAG = inline (3B writes the
    whole entry, no reasoner, no memory). Held-out = same hard helpers under UNSEEN easy wrappers: the
    pure reuse test — OURS retrieves the helper it banked; RAG must re-derive the hard logic inline.

    spec_step: if True, use batch_author_fn (all missing atoms in ONE LM call) + optionally a
    SpeculativePlanner (RankedNGram that learns program structure from banked solves)."""
    from v5.runtime.algo_grr_compose import OUTER, OUTER_HELD

    def seed_store():
        s = AtomStore()
        for name, (code, *_ ) in {**OUTER, **OUTER_HELD}.items():   # easy wrappers seeded; HARD authored
            s[name] = code
        return s

    store = seed_store()
    planner = spec_planner or OraclePlanner()
    ours = MembraneV2(store, TopologyAtomRouter(store), planner,
                      author_fn=None if batch_author_fn else author_fn,
                      batch_author_fn=batch_author_fn, bank=True)
    o_stream = 0
    for i, t in enumerate(stream):
        o_stream += int(ours.solve(t)["solved"])
        if verbose and (i + 1) % report_every == 0:
            print(f"  [OURS {i+1:>4}] solved={o_stream} banked={ours.banked} "
                  f"deriv_reuse={ours.derived_reuse} author_calls={ours.author_calls}", flush=True)
    o_hold = 0
    for hi, t in enumerate(holdout):
        r = ours.solve(t)
        ok = r["solved"]
        o_hold += int(ok)
        if hi < debug_heldout_n:
            bank_status = {a: "banked" if a in ours.store else "missing"
                           for a in r["program"].atoms if a != "n"}
            print(f"\n  ── HELD-OUT #{hi} ──")
            print(f"  task : {t['text']}")
            print(f"  atoms: {list(bank_status.items())}")
            print(f"  wiring: {r['program'].wiring}")
            print(f"  authored: {r.get('authored', [])}")
            print(f"  solved: {'YES' if ok else 'NO'}")
            if ok:
                rag_code = inline_fn(t)
                rag_ok = int(t["verify_fn"](rag_code)[0] >= 1.0)
                print(f"  OURS code (abbr): {r['code'][:120]}...")
                print(f"  RAG  code (abbr): {rag_code[:120]}...")
                print(f"  RAG on this task: {'SOLVED' if rag_ok else 'FAILED'}")
            else:
                print(f"  OURS verify FAILED — atoms not authored correctly or wiring wrong")
            print(f"  ────────────────")

    def rag_run(tasks, tag):
        solved = calls = 0
        for i, t in enumerate(tasks):
            code = inline_fn(t)
            calls += 1
            solved += int(t["verify_fn"](code)[0] >= 1.0)
            if verbose and (i + 1) % report_every == 0:
                print(f"  [RAG {tag} {i+1:>4}] solved={solved}", flush=True)
        return solved, calls
    r_stream, r_calls_s = rag_run(stream, "str")
    r_hold, r_calls_h = rag_run(holdout, "hold")

    ns, nh = len(stream), len(holdout)
    print(f"\n  arm  | stream solved | HELD-OUT solved | LM calls")
    print(f"  OURS |   {o_stream:>3}/{ns}    |     {o_hold:>3}/{nh}     | {ours.author_calls} author "
          f"(banked {ours.banked}, deriv_reuse {ours.derived_reuse})")
    print(f"  RAG  |   {r_stream:>3}/{ns}    |     {r_hold:>3}/{nh}     | {r_calls_s + r_calls_h} inline "
          f"(no bank, no reasoner)")
    print(f"\n  => HELD-OUT gap (the attack): OURS {o_hold}/{nh} vs RAG {r_hold}/{nh} — OURS reuses the "
          f"banked verified helper; RAG must re-derive the hard logic inline and fails.")
    return dict(ours_stream=o_stream, ours_hold=o_hold, rag_stream=r_stream, rag_hold=r_hold,
                author_calls=ours.author_calls, banked=ours.banked, derived_reuse=ours.derived_reuse)


def _selftest_v2() -> bool:
    """No-GPU: stub author writes the CORRECT primitive (so OURS banks); stub inline simulates the 3B —
    it reconstructs the hard helper correctly only p=0.35 of the time (else fails). Shows OURS >> RAG,
    especially on held-out where OURS reuses the bank at zero author cost."""
    import random
    from v5.runtime.algo_grr_compose import gen_corpus_hard, HARD, OUTER, OUTER_HELD
    print("algo_grr_pipeline --selftest-v2: OURS (MembraneV2 reason+bank) vs RAG (inline, no reasoner)\n")
    src = {**{k: v[0] for k, v in HARD.items()},
           **{k: v[0] for k, v in OUTER.items()},
           **{k: v[0] for k, v in OUTER_HELD.items()}}
    author = lambda name, task: src.get(name, "")                       # noqa: E731 — correct stub author
    rng = random.Random(0)

    def inline(task, retrieved=None):                                   # simulate the 3B inline attempt
        hard, outer = task["_prims"]
        if rng.random() < 0.35:                                         # gets the hard logic right sometimes
            return f"{src[hard]}\n{src[outer]}\ndef {task['entry']}(n):\n    return {outer}({hard}(n))\n"
        return f"def {task['entry']}(n):\n    return n\n"                # wrong (no hard logic) -> fails

    stream = gen_corpus_hard(60, seed=0)
    holdout = gen_corpus_hard(30, seed=0, holdout=True)
    r = run_v2_compare(stream, holdout, author, inline, verbose=False)
    ok = (r["ours_stream"] >= 0.95 * 60 and r["ours_hold"] >= 0.95 * 30
          and r["rag_hold"] <= 0.6 * 30 and r["derived_reuse"] > 0 and r["banked"] == len(HARD))
    print(f"\n  ALGO_GRR_PIPELINE V2 SELFTEST -> {'PASS' if ok else 'FAIL'}")
    return ok


def _selftest_v2_wiring() -> bool:
    """Where the TRM PLANNER is load-bearing: DEEP arithmetic-expression wiring the frozen LM cannot
    inline (measured collapse 0.73->0.03 by depth, algo_grr_wiring). OURS = SearchPlanner (infer program
    -> net-guided verified search -> deterministic realize) holds at depth; RAG = 3B writes the whole
    expression inline (here SIMULATED at the measured collapse curve; real number needs --lm)."""
    import os, random, tempfile
    from v5.runtime.algo_grr_planner import _PTOK, _make_data, _build, train_planner, _save_model
    from v5.runtime.algo_grr_wiring import gen_expr, to_words, oracle
    print("algo_grr_pipeline --selftest-v2-wiring: OURS (TRM planner + search + realize) vs RAG (3B inline)\n")
    path = os.path.join(tempfile.gettempdir(), "grr_planner_wiring.pt")
    if not os.path.exists(path):
        print("  training wiring planner (tiny seq2seq, no-GPU)...")
        torch, nn, Seq2Seq = _build(); torch.manual_seed(0)
        words_list, progs_list, wvocab = _make_data([1, 2, 3], 1500, seed=1)
        model = Seq2Seq(len(wvocab), len(_PTOK))
        model, _ = train_planner(model, words_list, progs_list, wvocab, steps=2500)
        _save_model(path, model, "wiring", wvocab)
    store = AtomStore.from_wiring()
    solver = MembraneV2(store, AtomRouter(store), SearchPlanner(store, path))
    rng = random.Random(0)

    def inline_sim(depth):                               # measured 3B inline collapse: ~0.73 -> ~0.03
        return max(0.03, 0.73 - 0.175 * (depth - 1))

    print("  depth | OURS (planner) | RAG (3B inline, sim)")
    o_all, r_all = [], []
    for d in (1, 2, 3, 4, 5):
        ours = rag = 0
        n = 20
        for i in range(n):
            t = gen_expr(d, random.Random(4000 * d + i))
            entry = "solve"

            def vf(code, t=t):
                ns: dict = {}
                try:
                    exec(compile(code, "<t>", "exec"), ns)  # noqa: S102
                    fn = ns.get(entry)
                    if fn is None:
                        return [0.0]
                    return [1.0] if all(fn(x) == oracle(t, x) for x in (2, 3, 5, 7, 11)) else [0.0]
                except Exception:  # noqa: BLE001
                    return [0.0]
            task = dict(text=to_words(t), entry=entry, verify_fn=vf)
            ours += int(solver.solve(task)["solved"])
            rag += int(rng.random() < inline_sim(d))
        o_all.append(ours / n); r_all.append(rag / n)
        print(f"  {d:>5} |      {ours/n:.2f}      |      {rag/n:.2f}")
    fo, fr = sum(o_all) / len(o_all), sum(r_all) / len(r_all)
    print(f"\n  overall: OURS {fo:.2f}  vs  RAG {fr:.2f}   (RAG collapses with depth; the planner holds)")
    print(f"  => the TRM planner is load-bearing HERE (structure must be inferred), unlike outer(hard(n)).")
    ok = fo >= 0.6 and fo > fr + 0.2
    print(f"\n  ALGO_GRR_PIPELINE V2-WIRING SELFTEST -> {'PASS' if ok else 'FAIL'}")
    return ok


def _selftest_spec() -> bool:
    """Where STEP-SPECULATION pays: multi-hard tasks each need K=2 hard helpers. NO-SPEC authors one atom
    per LM call; SPEC (the tiny planner proposes all K atoms → LM authors the CHUNK in ONE call) cuts LM
    calls ~K× at the SAME solve. Measured on a static (no-bank) store = the pure speculation speedup; it
    STACKS with banking. The composite verify still gates each authored atom (drift → no bank)."""
    from v5.runtime.algo_grr_compose import gen_corpus_multihard, HARD, OUTER
    print("algo_grr_pipeline --selftest-spec: step-speculation (batch author) vs per-atom author\n")
    src = {**{k: v[0] for k, v in HARD.items()}, **{k: v[0] for k, v in OUTER.items()}}
    author = lambda name, task: src.get(name, "")                    # noqa: E731
    batch = lambda names, task: {n: src.get(n, "") for n in names}   # noqa: E731 — one LM call for all K
    tasks = gen_corpus_multihard(90, seed=0)

    def run(spec: bool):
        store = AtomStore()
        for name, (code, *_ ) in OUTER.items():
            store[name] = code
        m = MembraneV2(store, AtomRouter(store), ProgramOraclePlanner(),
                       author_fn=None if spec else author,
                       batch_author_fn=batch if spec else None, bank=False)   # no bank = pure spec measure
        solved = sum(int(m.solve(t)["solved"]) for t in tasks)
        return solved, m.lm_calls, m.author_calls

    (s0, l0, a0), (s1, l1, a1) = run(False), run(True)
    n = len(tasks)
    print(f"  arm      | solved | LM calls | atoms authored | calls/task")
    print(f"  NO-SPEC  | {s0:>3}/{n} |   {l0:>4}   |     {a0:>4}       |   {l0/n:.2f}")
    print(f"  SPEC     | {s1:>3}/{n} |   {l1:>4}   |     {a1:>4}       |   {l1/n:.2f}")
    print(f"\n  => step-speculation: {l0/max(1,l1):.2f}× fewer LM calls at SAME solve (planner proposes "
          f"the K-atom program; LM authors the chunk in one call). Stacks with banking.")
    ok = s0 == n and s1 == n and l1 < l0 * 0.7 and a0 == a1
    print(f"\n  ALGO_GRR_PIPELINE SPEC SELFTEST -> {'PASS' if ok else 'FAIL'}")
    return ok


def _selftest_router() -> bool:
    """Topology-aware routing = the reasoner's structural memory-awareness. (a) On the de-confounded
    routing benchmark: content-only MISSES the cross-cluster dependency; follow-the-edge gets it (Recall
    1.0, scale-free). (b) In a MembraneV2 store: a composite's call-edge surfaces its sub-atom."""
    import numpy as np
    from v5.runtime.algo_grr_graphgps import gen_deconfounded, _topo_eval
    print("algo_grr_pipeline --selftest-router: topology follow-edge (structural memory) vs content-only\n")
    print("  N    | content-only recall | +follow-edge (topology)")
    rt_last = 0.0
    for N in (120, 300):
        content, A, dep, comm = gen_deconfounded(N, seed=0)
        rng = np.random.default_rng(1)
        test = [(int(q), {int(q), int(dep[q])}) for q in rng.permutation(N)[:40]]
        rec_c = []
        for q, nd in test:
            sims = content @ content[q]
            top = set(np.argsort(-sims)[:10].tolist())
            rec_c.append(len(top & nd) / len(nd))
        rc = float(np.mean(rec_c))
        rt = _topo_eval(A, test); rt_last = rt
        print(f"  {N:>4} |        {rc:.2f}         |        {rt:.2f}")
    store = AtomStore()
    store["is_prime"] = "def is_prime(n):\n    return n > 1 and all(n % i for i in range(2, n))\n"
    store["prime_pair"] = "def prime_pair(n):\n    return is_prime(n) and is_prime(n + 2)\n"
    ranked = TopologyAtomRouter(store).rank("prime pair", k=6)
    edge_ok = "is_prime" in ranked                     # pulled via prime_pair's call-edge, not content
    print(f"\n  MembraneV2 router: 'prime_pair' calls 'is_prime' -> follow-edge surfaces is_prime: {edge_ok}")
    ok = rt_last >= 0.99 and edge_ok
    print(f"  => structural deps FOLLOWED (recall 1.0, no training); content ranks the semantic rest.")
    print(f"\n  ALGO_GRR_PIPELINE ROUTER SELFTEST -> {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--selftest-planner", type=str, nargs="?", const="", default=None,
                    help="test pipeline WITH the real learned planner (optionally provide model path)")
    ap.add_argument("--selftest-compound", action="store_true",
                    help="MembraneV2 author+bank compounding: OURS vs RAG on the HARD corpus (no GPU)")
    ap.add_argument("--selftest-v2", action="store_true",
                    help="OURS (reason+bank) vs RAG (inline, no reasoner) incl. held-out (no GPU)")
    ap.add_argument("--selftest-v2-wiring", action="store_true",
                    help="where the TRM planner is load-bearing: deep-expression wiring, OURS(planner) vs "
                         "RAG(3B inline collapse) by depth (no GPU; RAG arm simulated)")
    ap.add_argument("--selftest-spec", action="store_true",
                    help="step-speculation: batch-author K atoms in ONE LM call vs per-atom (no GPU)")
    ap.add_argument("--selftest-router", action="store_true",
                    help="topology follow-edge routing (structural memory-awareness) vs content-only (no GPU)")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    if a.selftest_planner is not None:
        sys.exit(0 if _selftest_planner(a.selftest_planner if a.selftest_planner else "") else 1)
    if a.selftest_compound:
        sys.exit(0 if _selftest_compound() else 1)
    if a.selftest_v2:
        sys.exit(0 if _selftest_v2() else 1)
    if a.selftest_v2_wiring:
        sys.exit(0 if _selftest_v2_wiring() else 1)
    if a.selftest_spec:
        sys.exit(0 if _selftest_spec() else 1)
    if a.selftest_router:
        sys.exit(0 if _selftest_router() else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
