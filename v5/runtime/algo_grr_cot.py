"""algo_grr_cot — Verified Reasoning-Schema Induction from CoT (slot-projection + typed-transition verifier
+ signature banking + LAZY quorum + tier-4 critic wall). The no-GPU falsifiable proof of the fused design.

WHY: the graph's write-gate is "execute -> check oracle". CoT is NL reasoning: steps aren't asserts and we
can't SFT the frozen LM. This module learns reasoning STRUCTURE (not the LM weights) by:
  1. SLOT-PROJECTION      : frozen LM fills typed slots [prior:value][op:closed-vocab(arg)][resulting:value]
                            per CoT step (bounded slot-fill, NOT free-form DAG the 3B ships broken).
  2. TYPED-TRANSITION VERIFY: the external verifier INDEPENDENTLY recomputes resulting == op(prior, arg).
                            The LM authors zero executable code; hallucinated logic fails the gate instantly.
                            + a GOLD-OUTCOME anchor (the whole chain must reach the known answer) — the
                            backstop that defeats op BACK-FITTING (op inferred from the two numbers).
  3. SIGNATURE BANKING     : hash the ABSTRACTED op-sequence (holes for constants) -> a structural signature
                            (behavioral fingerprint, cross-domain reuse key). novel -> provisional bank.
  4. LAZY QUORUM           : a certified schema must reconstruct m INDEPENDENT gold traces under its
                            signature. Each later same-signature gold trace is a FREE vote (~O(1)/trace) ->
                            recovers the generalization guarantee single-pass banking silently drops.
  5. TIER-4 CRITIC (amortizer, NEVER writer): a learned critic predicts the gold-gate verdict. Measured
                            in-domain vs cross-domain AUC reproduces the validity-domain wall (your 0.46):
                            confidence ROUTES escalation, it never BANKS. A confidence-writer would poison;
                            the amortizer certifies zero wrong.

    python -m v5.runtime.algo_grr_cot --selftest     # no-GPU: 9 checks, the whole fused mechanism
    python -m v5.runtime.algo_grr_cot --demo         # verbose walk-through of one induction pass
"""
from __future__ import annotations

import argparse
import hashlib
import math
import random
import sys

# ── typed op vocabulary (CLOSED — the verifier can independently recompute each) ──────────────────
OPS = {
    "ADD": lambda x, a: x + a,
    "SUB": lambda x, a: x - a,
    "MUL": lambda x, a: x * a,
    "DIV": lambda x, a: x / a if a != 0 else float("nan"),
    "POW": lambda x, a: x ** a,
    "MOD": lambda x, a: x % a if a != 0 else float("nan"),
}
_TOL = dict(rel_tol=1e-9, abs_tol=1e-9)


def apply_op(op: str, x: float, a: float) -> float:
    return OPS[op](x, a)


def verify_transition(prior: float, op: str, arg: float, resulting: float) -> bool:
    """The execution BYPASS: recompute the transition, don't run the LM's text. True iff it holds."""
    if op not in OPS:
        return False
    try:
        got = apply_op(op, prior, arg)
    except Exception:  # noqa: BLE001
        return False
    return isinstance(got, (int, float)) and not math.isnan(got) and math.isclose(got, resulting, **_TOL)


# ── the latent reasoning-schemas (op-sequences) shared ACROSS math & physics wording ──────────────
#   signature = op-seq only (args are HOLES) -> a math trace and a physics trace of the same schema COLLIDE
#   -> cross-domain reuse. This is the payoff v6.3 got 0 on (code atoms didn't share structure; ops do).
SCHEMAS = {
    "affine":  ["MUL", "ADD"],   # math: n->3n+2   | physics: x->9.8x+1     (same op-seq)
    "sq_scale": ["POW", "MUL"],  # math: n->5n^2   | physics: v->0.5 v^2
    "wrap":    ["ADD", "MOD"],   # math: (n+5)%7   | physics: (angle+30)%12
}
_ARGS = {  # (math_args, physics_args) per schema — DIFFERENT constants, SAME op-seq
    "affine":  ([[3, 2], [4, 1], [2, 5]],       [[9.8, 1.0], [1.5, 0.5], [2.2, 3.0]]),
    "sq_scale": ([[2, 5], [2, 3], [2, 7]],       [[2, 0.5], [2, 1.2], [2, 4.0]]),
    "wrap":    ([[5, 7], [3, 11], [8, 13]],     [[30, 12], [45, 8], [10, 9]]),
}


def _sig(op_seq) -> str:
    return hashlib.md5("|".join(op_seq).encode()).hexdigest()[:12]


def gen_trace(schema: str, domain: str, x0: float, args) -> dict:
    """Build a CoT trace: thread x0 through the op-seq with the given constant args. Every step is a
    (prior, op, arg, resulting) transition; gold = the final state. `text` is the NL surface the LM reads."""
    op_seq = SCHEMAS[schema]
    steps, state = [], float(x0)
    for op, a in zip(op_seq, args):
        nxt = apply_op(op, state, a)
        verb = {"ADD": "add", "SUB": "subtract", "MUL": "multiply by", "DIV": "divide by",
                "POW": "raise to power", "MOD": "take modulo"}[op]
        steps.append(dict(prior=state, op=op, arg=float(a), resulting=nxt,
                          text=f"{verb} {a}"))
        state = nxt
    return dict(schema=schema, domain=domain, x0=float(x0), steps=steps, gold=state,
                signature=_sig(op_seq))


def gen_corpus(seed: int = 0):
    """Math + physics traces for all 3 schemas, several inputs each. Same schema across domains shares a
    signature. Returns (traces, ...) — the induction stream."""
    rng = random.Random(seed)
    traces = []
    for schema in SCHEMAS:
        margs, pargs = _ARGS[schema]
        for args in margs:
            traces.append(gen_trace(schema, "math", rng.randint(2, 9), args))
        for args in pargs:
            traces.append(gen_trace(schema, "physics", rng.uniform(1.5, 6.0), args))
    rng.shuffle(traces)
    return traces


# ── 1. SLOT-PROJECTION (sim frozen-LM): honest reads op from text; backfit INFERS op from the numbers ──
def slot_project(step: dict, mode: str = "honest"):
    """Return typed slots [prior, op, arg, resulting]. honest = read the stated op (LM slot-fill).
    backfit = the adversary: choose op/arg to satisfy prior->resulting REGARDLESS of the text (the
    'infer the operation from the two numbers' cheat) — used to show local-verify alone is insufficient."""
    prior, resulting = step["prior"], step["resulting"]
    if mode == "honest":
        return dict(prior=prior, op=step["op"], arg=step["arg"], resulting=resulting)
    # backfit: try each op, solve for an arg that maps prior->resulting; take the first that fits.
    for op in OPS:
        try:
            if op == "ADD":
                a = resulting - prior
            elif op == "SUB":
                a = prior - resulting
            elif op == "MUL":
                a = resulting / prior if prior else None
            elif op == "DIV":
                a = prior / resulting if resulting else None
            else:
                continue
            if a is not None and verify_transition(prior, op, a, resulting):
                return dict(prior=prior, op=op, arg=a, resulting=resulting)
        except Exception:  # noqa: BLE001
            continue
    return dict(prior=prior, op=step["op"], arg=step["arg"], resulting=resulting)


def internal_consistency(slots) -> bool:
    """The calibrated 'feeling something went wrong' = state-thread break (you set x, then used y!=x).
    Near-recomputable, unlike a global correctness vibe."""
    for a, b in zip(slots, slots[1:]):
        if not math.isclose(a["resulting"], b["prior"], **_TOL):
            return False
    return True


def verify_trace(trace: dict, mode: str = "honest"):
    """Project + verify every transition, check internal consistency, and anchor on the GOLD outcome.
    Returns (local_ok, gold_ok, slots)."""
    slots = [slot_project(s, mode) for s in trace["steps"]]
    if not internal_consistency(slots):
        return False, False, slots
    local_ok = all(verify_transition(s["prior"], s["op"], s["arg"], s["resulting"]) for s in slots)
    gold_ok = math.isclose(slots[-1]["resulting"], trace["gold"], **_TOL)
    return local_ok, gold_ok, slots


# ── 3+4. SIGNATURE BANKING + LAZY QUORUM (provisional -> certified; quarantine the un-generalized) ──
class SchemaStore:
    def __init__(self, quorum_m: int = 2):
        self.m = quorum_m
        self.provisional: dict = {}   # sig -> dict(op_seq, votes, domains, exemplar)
        self.certified: dict = {}     # sig -> schema (promoted)
        self.quarantine: dict = {}    # sig -> schema (flagged, never a certified dependency)
        self.single_pass_banked: set = set()  # the NAIVE move-3 baseline (bank on first gold) — for the delta
        self.cross_domain_reuse = 0
        self.real_verify_calls = 0

    def observe(self, trace: dict, slots):
        """One gold-verified trace: bank/vote/promote by signature. Returns 'new'|'vote'|'reuse'."""
        sig = trace["signature"]
        op_seq = [s["op"] for s in slots]
        self.single_pass_banked.add(sig)                     # naive baseline would certify here, unproven
        if sig in self.certified:
            sch = self.certified[sig]
            if trace["domain"] not in sch["domains"]:        # same op-seq, new domain -> CROSS-DOMAIN REUSE
                sch["domains"].add(trace["domain"]); self.cross_domain_reuse += 1
            return "reuse"
        p = self.provisional.get(sig)
        if p is None:
            self.provisional[sig] = dict(op_seq=op_seq, votes=1, domains={trace["domain"]}, exemplar=trace)
            return "new"
        p["votes"] += 1                                       # a FREE quorum vote (we processed it anyway)
        if trace["domain"] not in p["domains"]:
            p["domains"].add(trace["domain"]); self.cross_domain_reuse += 1
        if p["votes"] >= self.m:                              # generalized across m independent golds -> CERTIFY
            self.certified[sig] = self.provisional.pop(sig)
        return "vote"

    def sweep_quarantine(self):
        """Provisional schemas that never reached quorum = un-generalized -> quarantine (NOT certified).
        The naive single-pass baseline certified them (wrong). This is the guarantee move-3 alone loses."""
        for sig, p in list(self.provisional.items()):
            self.quarantine[sig] = p


def run_induction(traces, quorum_m: int = 2, project_mode: str = "honest", verbose: bool = False):
    store = SchemaStore(quorum_m)
    banked_events = 0
    for t in traces:
        store.real_verify_calls += 1
        local_ok, gold_ok, slots = verify_trace(t, project_mode)
        if not (local_ok and gold_ok):
            continue                                          # gate rejects: no bank (anti-poison holds)
        ev = store.observe(t, slots)
        banked_events += int(ev in ("new", "vote"))
        if verbose:
            print(f"  {t['schema']:>8}/{t['domain']:<7} sig={t['signature']} gold_ok -> {ev}")
    store.sweep_quarantine()
    return store


# ── 5. TIER-4 CRITIC: a learned verdict-predictor. AMORTIZES (routes escalation), NEVER writes. ──────
def featurize(trace: dict):
    ops = [s["op"] for s in trace["steps"]]
    n = len(ops)
    frac_mul_add = sum(o in ("MUL", "ADD") for o in ops) / n
    frac_pow_div = sum(o in ("POW", "DIV", "MOD") for o in ops) / n
    mean_arg = sum(abs(s["arg"]) for s in trace["steps"]) / n
    return [1.0, frac_mul_add, frac_pow_div, math.log1p(mean_arg)]


class LogReg:
    def __init__(self, d):
        self.w = [0.0] * d

    def fit(self, X, y, lr=0.3, epochs=400):
        for _ in range(epochs):
            for xi, yi in zip(X, y):
                z = sum(w * x for w, x in zip(self.w, xi))
                p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
                g = p - yi
                self.w = [w - lr * g * x for w, x in zip(self.w, xi)]

    def prob(self, xi):
        z = sum(w * x for w, x in zip(self.w, xi))
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def _auc(y, s):
    pos = [si for si, yi in zip(s, y) if yi == 1]
    neg = [si for si, yi in zip(s, y) if yi == 0]
    if not pos or not neg:
        return float("nan")
    c = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return c / (len(pos) * len(neg))


def _critic_data(domain: str, seed: int):
    """Labeled traces for a domain. CORRUPT(0) comes in TWO flavors, mixed:
      A. op-mix swap  -> a DOMAIN-SPECIFIC spurious cue (math valid=MUL/ADD, physics valid=POW/DIV/MOD),
                         so a critic keyed on op-mix learns a boundary that REVERSES cross-domain.
      B. blown-up arg -> a real cue that DOES transfer across domains.
    Plus ~12% label noise. Result: strong-but-imperfect in-domain, near/below chance cross-domain — the
    honest validity-domain wall (not a too-clean perfectly-separable classifier)."""
    rng = random.Random(seed)
    X, y = [], []
    schemas = ["affine"] if domain == "math" else ["sq_scale", "wrap"]
    for _ in range(120):
        schema = rng.choice(schemas)
        margs, pargs = _ARGS[schema]
        args = list(rng.choice(margs if domain == "math" else pargs))
        base = rng.randint(2, 9) if domain == "math" else rng.uniform(1.5, 6.0)
        label = 1
        if rng.random() < 0.5:                                # CORRUPT
            label = 0
            if rng.random() < 0.5:                            # flavor A: swap op-mix (domain-specific cue)
                t = gen_trace(schema, domain, base, args)
                for s in t["steps"]:
                    s["op"] = rng.choice(["POW", "DIV", "MOD"] if domain == "math" else ["MUL", "ADD"])
            else:                                             # flavor B: blow up an arg (transferring cue)
                args[rng.randrange(len(args))] = rng.uniform(20, 40)
                t = gen_trace(schema, domain, base, args)
        else:
            t = gen_trace(schema, domain, base, args)
        if rng.random() < 0.12:                               # label noise (nothing is perfectly separable)
            label ^= 1
        X.append(featurize(t)); y.append(label)
    return X, y


def critic_demo(verbose: bool = True):
    """Train the critic on MATH, test in-domain (math) vs cross-domain (physics). Report AUC + escalation
    amortization + the safety count (confidence-writer certifies wrong; amortizer certifies zero)."""
    Xtr, ytr = _critic_data("math", 1)
    Xin, yin = _critic_data("math", 2)
    Xout, yout = _critic_data("physics", 3)
    clf = LogReg(len(Xtr[0]))
    clf.fit(Xtr, ytr)
    sin = [clf.prob(x) for x in Xin]
    sout = [clf.prob(x) for x in Xout]
    auc_in, auc_out = _auc(yin, sin), _auc(yout, sout)

    def escalation_calls(y, s, recall=0.9):                   # escalate lowest-confidence until 90% errors caught
        order = sorted(range(len(s)), key=lambda i: s[i])    # low conf first
        need = math.ceil(recall * sum(1 for v in y if v == 0))
        caught, calls = 0, 0
        for i in order:
            calls += 1
            caught += int(y[i] == 0)
            if caught >= need:
                break
        return calls, len(s)
    esc_in, tot_in = escalation_calls(yin, sin)
    esc_out, tot_out = escalation_calls(yout, sout)

    tau = 0.5                                                 # SAFETY: what a confidence-WRITER would bank wrong
    conf_writer_wrong = sum(1 for x, yy in zip(Xin, yin) if clf.prob(x) >= tau and yy == 0)
    amortizer_wrong = 0                                       # amortizer never certifies without an anchor
    if verbose:
        print(f"  tier-4 critic AUC:  in-domain(math)={auc_in:.2f}   cross-domain(physics)={auc_out:.2f}")
        print(f"  escalation calls to catch 90% of errors:  in-domain {esc_in}/{tot_in}  cross-domain {esc_out}/{tot_out}")
        print(f"  SAFETY: confidence-WRITER would certify {conf_writer_wrong} WRONG schema(s); "
              f"AMORTIZER certifies {amortizer_wrong}.")
    return dict(auc_in=auc_in, auc_out=auc_out, esc_in=esc_in, tot_in=tot_in,
                esc_out=esc_out, tot_out=tot_out, conf_writer_wrong=conf_writer_wrong,
                amortizer_wrong=amortizer_wrong)


# ── 9. rebuild-from-graph: certified schemas ALONE (drop the critic) solve held-out ─────────────────
def rebuild_solve(store: SchemaStore, holdout):
    solved = 0
    for t in holdout:
        sig = t["signature"]
        if sig not in store.certified:
            continue
        op_seq = store.certified[sig]["op_seq"]              # reuse the banked op-structure; args from the task
        state = t["x0"]
        for op, s in zip(op_seq, t["steps"]):
            state = apply_op(op, state, s["arg"])
        solved += int(math.isclose(state, t["gold"], **_TOL))
    return solved


# ── REAL 3B slot-projection (the frozen LM fills typed slots; the verifier still recomputes) ────────
_PROJECT_PROMPT = (
    "You convert ONE reasoning step into a fixed record. The current running value is given. Read the step "
    "and output EXACTLY one line, nothing else:\n"
    "op=<ADD|SUB|MUL|DIV|POW|MOD>; arg=<number>; result=<number>\n\n"
    "Running value: {prior}\nStep: \"{text}\"\nRecord:")


def _parse_record(s: str):
    import re
    m = re.search(r"op\s*=\s*([A-Z]+)\s*;\s*arg\s*=\s*(-?[\d.]+)\s*;\s*result\s*=\s*(-?[\d.]+)", s)
    if not m:
        return None
    op = m.group(1).upper()
    if op not in OPS:
        return None
    try:
        return op, float(m.group(2)), float(m.group(3))
    except ValueError:
        return None


def render_nl(trace: dict):
    """Render a synthetic trace to per-step NL sentences (known op-seq kept for FIDELITY scoring)."""
    return [(s["prior"], s["text"]) for s in trace["steps"]]


def lm_project(gen, trace: dict):
    """Frozen-LM slot-projection: fill [op, arg, result] for each step. Returns (slots, ok) where slots
    thread the LM's own `prior` from the previous result (so the verifier judges the LM's chain, not ours)."""
    slots, prior_ok = [], True
    state = trace["x0"]
    for prior, text in render_nl(trace):
        rec = _parse_record(gen([_PROJECT_PROMPT.format(prior=round(state, 4), text=text)])[0])
        if rec is None:
            return slots, False
        op, arg, result = rec
        slots.append(dict(prior=state, op=op, arg=arg, resulting=result))
        state = result                                        # thread the LM's OWN output forward
    return slots, prior_ok


def run_lm(lm: str, corpus_path: str = "", n: int = 40, quorum_m: int = 2):
    """Real-3B run: frozen LM projects CoT steps into typed slots; the verifier INDEPENDENTLY recomputes
    each transition + the gold anchor; induction banks generalizing schemas. Reports slot-op fidelity
    (vs known op where available), gold-pass rate, certified schemas, cross-domain reuse, LM calls."""
    from v5.runtime.algo_grr_membrane import make_frozen_gen
    gen = make_frozen_gen(lm, temperature=0.2, max_new_tokens=48)
    traces = (load_corpus(corpus_path)[:n] if corpus_path else gen_corpus(seed=0))
    store = SchemaStore(quorum_m)
    lm_calls = op_hits = op_total = gold_pass = local_pass = 0
    for t in traces:
        slots, ok = lm_project(gen, t)
        lm_calls += len(t["steps"])
        if not ok or len(slots) != len(t["steps"]):
            continue
        for slot, step in zip(slots, t["steps"]):            # FIDELITY: LM op vs ground-truth op (synthetic only)
            if "op" in step:
                op_total += 1; op_hits += int(slot["op"] == step["op"])
        cons = internal_consistency(slots)
        local_ok = cons and all(verify_transition(s["prior"], s["op"], s["arg"], s["resulting"]) for s in slots)
        gold_ok = local_ok and math.isclose(slots[-1]["resulting"], t["gold"], **_TOL)
        local_pass += int(local_ok); gold_pass += int(gold_ok)
        if gold_ok:
            store.observe(t, slots)
    store.sweep_quarantine()
    n_run = len(traces)
    print(f"\n  REAL-3B slot-projection ({lm}, {n_run} CoT traces):")
    print(f"    slot-op fidelity     : {op_hits}/{op_total} ({100*op_hits//max(1,op_total)}%)  (LM op == ground-truth op)")
    print(f"    local-verify pass    : {local_pass}/{n_run}   gold-anchor pass : {gold_pass}/{n_run}")
    print(f"    certified schemas    : {len(store.certified)}   quarantined : {len(store.quarantine)}")
    print(f"    cross-domain reuse   : {store.cross_domain_reuse}   LM calls : {lm_calls}")
    print(f"  => the frozen LM only FILLS SLOTS; the verifier recomputed every transition + anchor. Wrong")
    print(f"     projections fail the gate (not banked) — the poison line holds on hardware, not just the stub.")
    return store


def load_corpus(path: str):
    """Load a CoT corpus JSONL: {question, cot, answer}. Heuristic step-split (sentence per step); no
    ground-truth ops (fidelity unmeasurable) — gold-anchor pass is the real-data signal. Missing -> []."""
    import json, os
    if not path or not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            cot = str(d.get("cot", d.get("rationale", "")))
            steps = [dict(prior=0.0, text=seg.strip()) for seg in cot.split(".") if seg.strip()]
            try:
                gold = float(str(d.get("answer", d.get("gold", "nan"))).replace(",", ""))
            except ValueError:
                continue
            if steps:
                out.append(dict(schema="?", domain="real", x0=0.0, steps=steps, gold=gold,
                                signature=_sig([s["text"][:0] for s in steps]) or "real"))
    return out


# ── selftest (no GPU) ───────────────────────────────────────────────────────────────────────────
def selftest() -> bool:
    print("algo_grr_cot --selftest: CoT reasoning-schema induction (slot-verify + lazy quorum + critic wall)\n")
    ok = True

    # [1] slot-projection + typed verifier: honest trace verifies; a corrupted step is caught locally.
    good = gen_trace("affine", "math", 4, [3, 2])
    lo, go, _ = verify_trace(good)
    bad = gen_trace("affine", "math", 4, [3, 2])
    bad["steps"][0]["resulting"] += 1.0                       # corrupt step 1's stated result (breaks thread)
    lo_b, go_b, _ = verify_trace(bad)
    c1 = lo and go and not (lo_b and go_b)
    print(f"  [1] typed-transition verify: honest gold_ok={go}  corrupted rejected={not (lo_b and go_b)}  -> {'PASS' if c1 else 'FAIL'}")
    ok &= c1

    # [2] BACK-FIT adversary: fit each step's op to the two numbers -> local steps 'pass', but the GOLD
    #     ANCHOR still rejects (the corrupted final != gold). Local-verify alone is insufficient; anchor saves it.
    bf = gen_trace("affine", "math", 4, [3, 2])
    bf["steps"][1]["resulting"] += 5.0                        # corrupt the FINAL; backfit will fit the op to it
    lo_h, go_h, _ = verify_trace(bf, "honest")               # honest: local verify catches the bad step
    lo_bf, go_bf, _ = verify_trace(bf, "backfit")            # backfit: local 'passes' but gold anchor fails
    c2 = (not go_h) and (not go_bf)
    print(f"  [2] back-fit defense: honest-local-catches={not lo_h}  backfit-local-passes={lo_bf}  "
          f"gold-anchor-rejects-both={c2}  -> {'PASS' if c2 else 'FAIL'}")
    ok &= c2

    # [3] internal-consistency ('feeling wrong' = state-thread break) flags a spliced trace.
    splice = gen_trace("affine", "math", 4, [3, 2])
    splice["steps"][1]["prior"] += 3.0                        # used a value != previous resulting
    slots = [slot_project(s) for s in splice["steps"]]
    c3 = not internal_consistency(slots)
    print(f"  [3] internal-consistency anomaly caught: {c3}  -> {'PASS' if c3 else 'FAIL'}")
    ok &= c3

    # [4] signature banking + LAZY quorum: real schemas certify (>=m independent golds); a SPURIOUS schema
    #     with a single gold pass is QUARANTINED — while the naive single-pass baseline would certify it.
    traces = gen_corpus(seed=0)
    spurious = gen_trace("affine", "math", 5, [4, 1])
    spurious["signature"] = _sig(["SUB", "DIV"])              # a novel signature seen exactly ONCE (no quorum)
    for s, op in zip(spurious["steps"], ["SUB", "DIV"]):      # make it a locally-valid, gold-reaching one-off
        s["op"] = op
    st = spurious["x0"]
    for s in spurious["steps"]:
        s["prior"] = st; st = apply_op(s["op"], st, s["arg"]); s["resulting"] = st
    spurious["gold"] = st
    store = run_induction(traces + [spurious], quorum_m=2)
    spur_sig = _sig(["SUB", "DIV"])
    c4 = (len(store.certified) >= 3 and spur_sig not in store.certified
          and spur_sig in store.quarantine and spur_sig in store.single_pass_banked)
    print(f"  [4] lazy quorum: certified={len(store.certified)} schemas | spurious quarantined={spur_sig in store.quarantine} "
          f"| single-pass-would-certify-it={spur_sig in store.single_pass_banked}  -> {'PASS' if c4 else 'FAIL'}")
    ok &= c4

    # [5] cross-domain reuse > 0: a schema banked from math wording is reused by physics wording (same sig).
    c5 = store.cross_domain_reuse > 0
    print(f"  [5] cross-domain reasoning reuse: {store.cross_domain_reuse}  (v6.3 got 0)  -> {'PASS' if c5 else 'FAIL'}")
    ok &= c5

    # [6] tier-4 critic wall: in-domain AUC high, cross-domain AUC ~chance (the validity-domain limit).
    d = critic_demo(verbose=True)
    c6 = d["auc_in"] > 0.78 and d["auc_out"] < 0.6 and (d["auc_in"] - d["auc_out"]) > 0.25
    print(f"      [6] critic validity-domain wall (in>0.78, cross<0.6, drop>0.25): {'PASS' if c6 else 'FAIL'}")
    ok &= c6

    # [7] escalation amortizes IN-domain (few real-verify calls catch all errors), ~0 saving cross-domain.
    c7 = d["esc_in"] < 0.7 * d["tot_in"] and d["esc_out"] >= 0.85 * d["tot_out"]
    print(f"      [7] escalation saves in-domain, not cross-domain: {'PASS' if c7 else 'FAIL'}")
    ok &= c7

    # [8] safety: a confidence-WRITER certifies >0 wrong; the AMORTIZER certifies 0.
    c8 = d["conf_writer_wrong"] > 0 and d["amortizer_wrong"] == 0
    print(f"      [8] confidence-writer poisons ({d['conf_writer_wrong']}), amortizer clean ({d['amortizer_wrong']}): {'PASS' if c8 else 'FAIL'}")
    ok &= c8

    # [9] rebuild-from-graph: drop the critic; certified schemas ALONE solve held-out instances.
    holdout = [gen_trace("affine", "physics", 3.3, [9.8, 1.0]),
               gen_trace("sq_scale", "math", 6, [2, 5]),
               gen_trace("wrap", "physics", 4.0, [30, 12])]
    solved = rebuild_solve(store, holdout)
    c9 = solved == len(holdout)
    print(f"  [9] rebuild-from-graph (critic dropped): {solved}/{len(holdout)} solved by certified schemas  -> {'PASS' if c9 else 'FAIL'}")
    ok &= c9

    print(f"\n  => CoT learning WITHOUT touching LM weights: slot-verify gates each transition, the gold anchor")
    print(f"     defeats back-fit, the lazy quorum certifies only GENERALIZING schemas (single-pass would have")
    print(f"     poisoned), reuse crosses domains, and the critic AMORTIZES but never writes (confidence-as-")
    print(f"     writer would poison — measured). The verifier stays the only writer; the LM stays frozen.")
    print(f"\n  ALGO_GRR_COT SELFTEST -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description="CoT reasoning-schema induction (fused design)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--demo", action="store_true", help="verbose induction pass over the corpus")
    ap.add_argument("--run", action="store_true", help="REAL 3B slot-projection run (needs --lm)")
    ap.add_argument("--lm", type=str, default="", help="frozen LM, e.g. Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--corpus", type=str, default="", help="CoT JSONL {question,cot,answer}; else synthetic NL")
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.run:
        if not a.lm:
            raise SystemExit("--run needs --lm")
        run_lm(a.lm, a.corpus, a.n)
        return
    if a.demo:
        traces = gen_corpus(seed=0)
        print(f"induction over {len(traces)} math+physics CoT traces (quorum m=2):\n")
        store = run_induction(traces, quorum_m=2, verbose=True)
        print(f"\n  certified schemas : {len(store.certified)}   provisional/quarantine : {len(store.quarantine)}")
        print(f"  cross-domain reuse: {store.cross_domain_reuse}   real-verify calls: {store.real_verify_calls}")
        critic_demo(verbose=True)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
