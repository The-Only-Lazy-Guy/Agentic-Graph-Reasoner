"""algo_grr_dcpd — Dual-Channel realization (v6): SYMBOLIC (exact structure) + SEMANTIC (explanation).

The v5 realizer PASTED whole atom bodies AND hard-coded the wiring -> the LM wrote nothing, so it could
not EXPLAIN its approach and the output was a spliced Frankenstein. v6 splits the two things an answer
actually contains (your DESIGN A/B brainstorm):

  SYMBOLIC channel  — the EXACT structure the model must not get wrong (verified atom bodies + the
                      call skeleton). Comes from the graph as an AST skeleton with typed HOLES. The LM
                      cannot corrupt it -> zero syntax hallucination on the hard parts. (DESIGN A γ=0 /
                      DESIGN B "AST fragments / syntax lattice".)
  SEMANTIC channel  — the EXPLANATION (why/how), narrated from the EXECUTION-GRAPH TRAVERSAL + the
                      nodes' own descriptions. Faithful by construction (it can only cite atoms that are
                      actually in the verified program), not a post-hoc rationalization.

Semantic channel is an INTERFACE, deliberately. Default = TEXT (measured winner: the z-wall /
softprompt-73->15 routing-collapse killed the last continuous-latent attempt — [[softprompt-latent-memory-result]]).
DESIGN A's continuous-latent channel (h_latent -> LM via soft-prompt/cross-attn) is a REGISTERED
alternative (LatentSemanticChannel) that must WIN `fair_ab` vs text before adoption — the door is built,
it has to earn the room.

No GPU: `--selftest` runs the whole mechanism with a stub filler + stub LM. Real grammar-constrained
hole-filling (Outlines / llama.cpp GBNF) drops into `fill_fn` on the `--run` path.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from v5.runtime.algo_grr_pipeline import AtomProgram, AtomStore, _render  # noqa: E402


# ── execution-graph helpers ─────────────────────────────────────────────────────
def atoms_in_wiring(w) -> list:
    """Call-order atom names in the wiring tree (the execution graph the TRM built)."""
    if isinstance(w, str):
        return []
    _, name, args = w
    inner = [a for arg in args for a in atoms_in_wiring(arg)]
    return inner + [name]     # deps first, then the caller (post-order = execution order)


# ── SYMBOLIC channel: AST skeleton with typed holes ──────────────────────────────
@dataclass
class Skeleton:
    """Exact structure from the graph. `prelude` (verified atom bodies) is IMMUTABLE — the LM never
    touches it. Only `holes` are LM-fillable, each confined by a grammar tag. This is the guarantee:
    a hallucinating filler can only damage a hole, never the verified closure, and the hole grammar +
    the verify gate catch even that."""
    entry: str
    prelude: str                              # verified atom sources (exact, from nodes)
    body_template: str                        # e.g. "return {wiring}"
    holes: dict = field(default_factory=dict)  # hole_name -> grammar tag ("call_expr", "identifier", ...)

    def assemble(self, fills: dict) -> str:
        body = self.body_template
        for h, val in fills.items():
            body = body.replace("{" + h + "}", val)
        return f"{self.prelude}\n\ndef {self.entry}(n):\n    {body}\n"


def build_skeleton(program: AtomProgram, store: AtomStore, entry: str) -> Skeleton:
    """Symbolic channel: pull each atom's VERIFIED source (exact) as the immutable prelude; leave the
    wiring as a single typed hole the LM fills (grammar = a call-expression over the program's atoms)."""
    prelude = "\n".join(store[a].rstrip("\n") for a in program.atoms if a in store)
    return Skeleton(entry=entry, prelude=prelude, body_template="return {wiring}",
                    holes={"wiring": f"call_expr over {{{', '.join(program.atoms)}, n}}"})


# stub fillers (the real one is a grammar-constrained LM decode) ------------------
def exact_fill(program: AtomProgram) -> Callable[[Skeleton], dict]:
    """Correct hole fill = the wiring the planner chose (what a grammar-constrained LM converges to)."""
    return lambda skel: {"wiring": _render(program.wiring)}


def hallucinating_fill(_program: AtomProgram) -> Callable[[Skeleton], dict]:
    """A filler that IGNORES the grammar and emits a re-derived / broken expression — models a raw LM
    left to write freely. Used to show the symbolic channel + verify catch it (the closure stays intact)."""
    return lambda skel: {"wiring": "n +"}          # syntactically broken glue


def grammar_valid(code: str) -> bool:
    """Does the emitted code parse? (Stand-in for the grammar/AST guarantee on the symbolic channel.)"""
    import ast
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


# ── SEMANTIC channel: interface + text impl + parked latent seam ─────────────────
class SemanticChannel:
    """Produces the NL explanation that rides alongside the symbolic emission."""

    name = "base"

    def narrate(self, program: AtomProgram, store: AtomStore, task: dict) -> str:
        raise NotImplementedError


class TextSemanticChannel(SemanticChannel):
    """Default (proven): narrate from the EXECUTION-GRAPH traversal + the nodes' own cards. Every line
    cites an atom that is ACTUALLY in the verified program -> faithful by construction."""

    name = "text"

    def narrate(self, program, store, task):
        order = [a for a in atoms_in_wiring(program.wiring) if a != "n"]
        meta = getattr(store, "meta", {})
        lines = [f"Approach for: {task.get('text', '')}"]
        for a in order:
            if a not in meta:
                lines.append(f"- {a}: (used)")
                continue
            nd = meta[a]
            ex = f"; e.g. {nd.examples[0]}" if nd.examples else ""
            prov = {"seed": "library atom", "authored": "authored+verified", "derived": "derived"}\
                .get(nd.provenance, nd.provenance)
            lines.append(f"- {nd.signature or a}: {nd.description} | {nd.approach} [{prov}{ex}]")
        lines.append(f"Composition: {_render(program.wiring)}")
        return "\n".join(lines)


class LatentSemanticChannel(SemanticChannel):
    """PARKED seam for DESIGN A's continuous semantic channel: the TRM's h_latent -> LM via
    soft-prompt / cross-attention. Needs a white-box runner + a TRM that emits h_latent. Registered so
    `fair_ab` can measure it the moment a runner exists. Adoption gate (honest, same one it failed
    before): beat TEXT on exact-emission + explanation-faithfulness + held-out, at scale, no routing
    collapse. Until then TextSemanticChannel is the default — not because latent is bad, because it
    hasn't earned it yet."""

    name = "latent"

    def __init__(self, inject_fn=None):
        self.inject_fn = inject_fn                 # (program, store, task, h_latent) -> text, on a real runner

    def narrate(self, program, store, task):
        if self.inject_fn is None:
            raise NotImplementedError(
                "continuous-latent semantic channel needs a white-box LM runner + TRM h_latent (v6 "
                "research). Use TextSemanticChannel until it wins fair_ab; the seam is here so it can.")
        return self.inject_fn(program, store, task)


_CHANNELS = {"text": TextSemanticChannel, "latent": LatentSemanticChannel}


def get_channel(name: str) -> SemanticChannel:
    return _CHANNELS[name]()


# ── the dual-channel realize ─────────────────────────────────────────────────────
def dual_channel_realize(program: AtomProgram, store: AtomStore, task: dict, *,
                         semantic: SemanticChannel | None = None,
                         fill_fn: Callable[[Skeleton], dict] | None = None) -> dict:
    """Symbolic (exact skeleton + LM-filled holes) + semantic (narration). Returns the LM-OWNED,
    explainable, syntactically-guaranteed output. The verified atom closure is exact; the LM authors
    the glue (in a hole) and the explanation -> the model OWNS + EXPLAINS its output, and cannot
    hallucinate the hard structure."""
    entry = task["entry"]
    skel = build_skeleton(program, store, entry)
    fill = (fill_fn or exact_fill(program))(skel)
    code = skel.assemble(fill)
    semantic = semantic or TextSemanticChannel()
    explanation = semantic.narrate(program, store, task)
    return dict(code=code, explanation=explanation, skeleton=skel,
                closure_intact=(skel.prelude in code))


def explanation_faithfulness(explanation: str, program: AtomProgram, vocab) -> float:
    """Precision of the explanation against the EXECUTED program: of the ATOM NAMES the explanation
    names (restricted to the known atom vocabulary — not every English word), what fraction are
    actually in the program. Narrate-from-graph = 1.0 by construction; a post-hoc free-form explanation
    that name-drops retrieved-but-unused atoms scores < 1.0 (hallucinated)."""
    used = {a for a in program.atoms if a != "n"}
    mentioned = {a for a in vocab if re.search(rf"\b{re.escape(a)}\b", explanation)}
    if not mentioned:
        return 0.0
    return len(mentioned & used) / len(mentioned)


def _seen_names(text: str) -> set:
    return set(re.findall(r"[a-z_][a-z0-9_]+", text))


# ── negative-edge MISTAKE pruning (symbolic pink-elephant fix) ────────────────────
@dataclass
class MistakeNode:
    """(-) signal: a task-shape -> a forbidden atom/approach + why. Used to PRUNE a candidate BEFORE
    generation (symbolic), so the LM never sees the trap in-context -> no pink-elephant, and the reason
    is available to narrate ('X fails here because ...'). This is DESIGN B's negative-edge check — the
    safe version of DESIGN A's latent repulsion (no hidden-state surgery)."""
    trigger_keys: list        # task keywords that arm this mistake
    forbid: str               # atom name / approach to drop from candidates
    why: str


def prune_candidates(candidates: list, mistakes: list, task_text: str) -> tuple:
    """Drop candidates that a triggered mistake forbids. Returns (kept, fired_reasons)."""
    q = set(_seen_names(task_text.lower()))
    fired = [m for m in mistakes if q & set(k.lower() for k in m.trigger_keys)]
    forbidden = {m.forbid for m in fired}
    kept = [c for c in candidates if c not in forbidden]
    return kept, [(m.forbid, m.why) for m in fired]


# ── fair A/B for the semantic channel (the gate the latent idea must pass) ────────
def fair_ab(channels: list, programs_tasks: list, store: AtomStore) -> dict:
    """Score each semantic channel on the SAME programs: mean explanation-faithfulness (+ raises if a
    channel isn't runnable yet). This is where the continuous-latent channel gets measured against text
    — identical tasks, identical metric — so 'if it works it ships' is a real, decidable claim."""
    vocab = set(store)
    out = {}
    for ch in channels:
        try:
            fs = []
            for prog, task in programs_tasks:
                expl = ch.narrate(prog, store, task)
                fs.append(explanation_faithfulness(expl, prog, vocab))
            out[ch.name] = dict(runnable=True, faithfulness=sum(fs) / len(fs))
        except NotImplementedError as e:
            out[ch.name] = dict(runnable=False, reason=str(e)[:80])
    return out


# ── no-GPU selftest ──────────────────────────────────────────────────────────────
def selftest() -> bool:
    from v5.runtime.algo_grr_compose import gen_corpus_hard, HARD, OUTER, OUTER_HELD
    from v5.runtime.algo_grr_pipeline import OraclePlanner
    print("algo_grr_dcpd --selftest: dual-channel realize (symbolic exact + semantic faithful)\n")

    store = AtomStore()
    for name, (code, _fn, desc) in {**OUTER, **OUTER_HELD, **HARD}.items():
        store.set_rich(name, code, description=desc.replace("{v}", "the value"), provenance="seed")
    tasks = gen_corpus_hard(24, seed=0)
    planner = OraclePlanner()

    # [1] SYMBOLIC channel: output is LM-owned (glue in a hole) yet verifies; closure stays exact
    solved = closure_ok = 0
    progs_tasks = []
    for t in tasks:
        prog = planner.plan(t)
        progs_tasks.append((prog, t))
        r = dual_channel_realize(prog, store, t)
        solved += int(t["verify_fn"](r["code"])[0] >= 1.0)
        closure_ok += int(r["closure_intact"])
    n = len(tasks)
    print(f"  [1] symbolic: dual-channel code verifies {solved}/{n} | closure exact-from-graph {closure_ok}/{n}")

    # [2] symbolic GUARANTEE: a hallucinating filler only damages the HOLE; the verified closure is
    #     never corrupted, and grammar+verify catch the bad glue.
    t0 = tasks[0]
    p0 = planner.plan(t0)
    bad = dual_channel_realize(p0, store, t0, fill_fn=hallucinating_fill(p0))
    bad_caught = (t0["verify_fn"](bad["code"])[0] < 1.0) and not grammar_valid(bad["code"])
    closure_survived = bad["closure_intact"]
    print(f"  [2] guarantee: hallucinated glue caught (grammar+verify)={bad_caught} | "
          f"verified closure survived intact={closure_survived}")

    # [3] SEMANTIC channel: narrate-from-graph is faithful (=1.0); a post-hoc free-form explanation
    #     that name-drops RANKED-but-unused atoms is < 1.0 (the hallucinated-explanation baseline).
    vocab = set(store)
    text_ch = TextSemanticChannel()
    faith = sum(explanation_faithfulness(text_ch.narrate(p, store, t), p, vocab) for p, t in progs_tasks) / n

    def posthoc(prog, task):                       # simulate an ungrounded explanation
        extra = [a for a in store if a not in prog.atoms][:3]
        return "We used " + ", ".join([a for a in prog.atoms if a != "n"] + extra)
    posthoc_faith = sum(explanation_faithfulness(posthoc(p, t), p, vocab) for p, t in progs_tasks) / n
    print(f"  [3] semantic faithfulness: narrate-from-graph {faith:.2f}  vs  post-hoc free-form {posthoc_faith:.2f}")

    # [4] negative-edge MISTAKE pruning (symbolic pink-elephant fix)
    mistakes = [MistakeNode(trigger_keys=["digits"], forbid="num_divisors",
                            why="'digits' task must not use the divisor-count atom")]
    cands = ["count_digits", "num_divisors", "is_even"]
    kept, fired = prune_candidates(cands, mistakes, "the number of digits in the value")
    prune_ok = ("num_divisors" not in kept) and fired
    print(f"  [4] mistake prune: kept={kept} fired={[f[0] for f in fired]} -> forbidden dropped={prune_ok}")

    # [5] fair A/B seam: text runs; latent is registered but NOT runnable (honest parked door)
    ab = fair_ab([TextSemanticChannel(), LatentSemanticChannel()], progs_tasks, store)
    seam_ok = ab["text"]["runnable"] and not ab["latent"]["runnable"]
    print(f"  [5] fair_ab seam: text={ab['text']} | latent={ab['latent']}")

    ok = (solved == n and closure_ok == n and bad_caught and closure_survived
          and faith >= 0.99 and posthoc_faith < faith and prune_ok and seam_ok)
    print(f"\n  => symbolic channel = exact + hallucination-proof; semantic channel = faithful (1.0) and")
    print(f"     swappable (latent seam present, must win fair_ab); mistakes prune symbolically.")
    print(f"\n  ALGO_GRR_DCPD SELFTEST -> {'PASS' if ok else 'FAIL'}")
    return ok


# ── real-LM path: grammar-constrained glue fill + free-form inline baseline ──────
def _valid_glue(expr: str, allowed: set) -> bool:
    """Grammar check for the hole: a single call-expression over the ALLOWED atom names + `n` only.
    (A software stand-in for GBNF/Outlines constrained decode — rejects anything the grammar forbids.)"""
    import ast
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError:
        return False
    for nd in ast.walk(tree):
        if isinstance(nd, ast.Call) and not (isinstance(nd.func, ast.Name) and nd.func.id in allowed):
            return False
        if isinstance(nd, ast.Name) and nd.id not in allowed and nd.id != "n":
            return False
        if isinstance(nd, (ast.Import, ast.Attribute)):
            return False
    return True


def make_glue_fill(gen):
    """LM fills the wiring HOLE: given the verified helper cards, write ONLY the composition expression.
    Validated against the hole grammar; an invalid emission is REJECTED (caught) and the skeleton repairs
    to the planner's exact wiring — so the symbolic channel always yields valid, verifying code."""
    def fill(skel: Skeleton, program: AtomProgram, store: AtomStore, task: dict):
        cards = "\n".join(store.card(a) for a in program.atoms if a in getattr(store, "meta", {}))
        prompt = (f"{cards}\nUsing ONLY the functions above, write the single Python expression for "
                  f"`{task['entry']}(n)` that computes: {task['text']}. Reply with ONLY the expression "
                  f"(e.g. `outer(inner(n))`), no `return`, no explanation.")
        raw = gen([prompt])[0].strip().splitlines()[0].replace("return ", "").strip(" `")
        allowed = {a for a in program.atoms if a != "n"}
        if _valid_glue(raw, allowed):
            return {"wiring": raw}, True
        return {"wiring": _render(program.wiring)}, False   # grammar rejected -> repair to exact
    return fill


def run_lm(model: str, n: int = 40):
    """Real 3B: DUAL-CHANNEL (exact closure + LM-filled validated glue + faithful narration) vs FREE-FORM
    inline (LM writes the whole entry). Measures the symbolic guarantee: dual-channel never ships a broken
    closure; free-form inline does."""
    import os
    os.environ["V5_HARD_VERIFY"] = "1"             # untrusted free-form 3B code -> hard-killable subprocess
    from v5.runtime.algo_grr_membrane import make_frozen_gen, _extract_code
    from v5.runtime.algo_grr_compose import gen_corpus_hard, HARD, OUTER, OUTER_HELD
    from v5.runtime.algo_grr_pipeline import OraclePlanner
    gen = make_frozen_gen(model, temperature=0.4, max_new_tokens=200)
    print(f"  model loaded — running {n} tasks (2 LM calls each: glue-fill + inline baseline)...", flush=True)

    store = AtomStore()
    for name, (code, _fn, desc) in {**OUTER, **OUTER_HELD, **HARD}.items():
        store.set_rich(name, code, description=desc.replace("{v}", "the value"), provenance="seed")
    tasks = gen_corpus_hard(n, seed=0)
    planner = OraclePlanner()
    glue = make_glue_fill(gen)

    dc_ok = dc_glue_rejected = faith_sum = 0
    inline_ok = inline_syntax_err = 0
    print(f"DCPD real-LM: dual-channel (exact closure + LM glue) vs free-form inline | n={n} lm={model}\n")
    for i, t in enumerate(tasks):
        prog = planner.plan(t)
        skel = build_skeleton(prog, store, t["entry"])
        fills, accepted = glue(skel, prog, store, t)
        dc_glue_rejected += int(not accepted)
        code = skel.assemble(fills)
        dc_ok += int(t["verify_fn"](code)[0] >= 1.0)
        faith_sum += explanation_faithfulness(TextSemanticChannel().narrate(prog, store, t), prog, set(store))
        # free-form inline baseline: LM writes the WHOLE entry (must reconstruct the hard helper itself)
        ip = (f"Write a Python function `{t['entry']}(n)` that computes {t['text']}. "
              f"Include any helpers. Return ONLY code.")
        icode = _extract_code(gen([ip])[0])
        inline_syntax_err += int(not grammar_valid(icode))
        inline_ok += int(t["verify_fn"](icode)[0] >= 1.0)
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{n}] dual-ok={dc_ok} (glue-rejected {dc_glue_rejected}) | inline-ok={inline_ok} "
                  f"(syntax-err {inline_syntax_err})", flush=True)
    print(f"\n  arm         | verifies | syntax errors shipped | explanation faithfulness")
    print(f"  DUAL-CHANNEL | {dc_ok:>3}/{n}  |          0            | {faith_sum/n:.2f}  "
          f"(glue rejected+repaired {dc_glue_rejected})")
    print(f"  FREE-INLINE  | {inline_ok:>3}/{n}  |         {inline_syntax_err:>2}            | n/a (no grounded explanation)")
    print(f"\n  => symbolic channel ships ZERO broken closures (helper bodies exact from graph); the LM only")
    print(f"     fills a grammar-checked hole. Free-form inline hallucinates syntax {inline_syntax_err}/{n}. Explanation")
    print(f"     faithfulness {faith_sum/n:.2f} (narrated from the executed program).")
    return dict(dc_ok=dc_ok, inline_ok=inline_ok, inline_syntax_err=inline_syntax_err, faith=faith_sum / n)


def main():
    ap = argparse.ArgumentParser(description="Dual-Channel realization (symbolic + semantic)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true", help="real-LM dual-channel vs free-form inline (needs --lm)")
    ap.add_argument("--lm", type=str, default="", help="HF model id for --run (e.g. Qwen/Qwen2.5-3B-Instruct)")
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.run:
        if not a.lm:
            raise SystemExit("--run needs --lm (real model). No-GPU mechanism check: --selftest")
        run_lm(a.lm, a.n)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
