"""algo_grr_spike_policy — the spiking graph layer as a MembraneSolver HOP POLICY.

Companion to algo_grr_spike.py (which plugs into MembraneV2's one-shot planner). This one plugs
into MembraneSolver's `policy_fn(task, selected, graph, retriever) -> [(node_id, score), ...]`,
which the solver calls ONCE PER HOP with the atoms it has already committed to. That loop is
already a settling loop with a verifier-gated stop, so the LIF dynamics map onto it directly.

WHY THIS CORPUS AND NOT THE OTHER ONE. Measured: MembraneV2's compose corpus has ZERO depend edges
(every atom independent; composition lives in the realizer), so a lateral graph term there has
nothing to act on and the idea is untestable, not wrong. The seed graph here has real structure:

    21 part_of edges into 4 concept hubs, and 7 depend edges between implementations --
    impl_is_perfect -> impl_sum_divisors -> impl_divisors  (a genuine 2-hop chain),
    impl_lcm -> impl_gcd, impl_is_palindrome_number -> impl_reverse_digits,
    impl_most_common -> impl_count_occurrences, impl_is_palindrome -> impl_reverse_string,
    impl_is_anagram -> impl_char_freq.

SELECTED ATOMS ARE CLAMPED FIRING. MembraneSolver's own comment says the baseline keeps a stable
query and "folding selected purposes back into the query reinforces the already-covered concept and
buries the missing complement -- that program-conditioning is the TRM policy's job (build B), which
retrieves the COMPLEMENT; the untrained baseline must not fake it." This layer does exactly that
STRUCTURALLY rather than by query rewriting: every already-selected atom is held at y=1, so it keeps
emitting lateral signal, and its depend-partners are excited into the ranking. Picking is_perfect
lights sum_divisors, which lights divisors -- the closure the realizer needs.

WHAT IT ADDS OVER SpreadingActivationRetriever (the existing edge baseline it must beat, NOT
re-derive). SAR propagates energy A_j = tanh(sum_i A_i * W_ij * C_j) with hand-set per-relation
weights. Three differences, and only the third is the real bet:
  1. discrete spikes + an adaptive per-node bar, instead of continuous activation then rank();
  2. homeostasis -- SAR has no theta, so a hub with many edges stays lit on every query (the graph
     form of "2/5/200 occur in every span"); here a hub habituates and stops dominating;
  3. ARBITRATION FROM STRUCTURE. SAR can only suppress through explicitly-negative relations
     (avoid_if/contradict/conflict) and THIS GRAPH HAS NONE -- its relations are exactly part_of
     and depend. So SAR can amplify but cannot say "these two are rivals for the same role, pick
     one". Competition here is derived from structure (siblings under a shared concept hub), so no
     one has to hand-author a negative edge for arbitration to exist.

PRIOR PRESERVATION IS EXACT, BY CONSTRUCTION. The returned score is
    score = base_score + accumulated lateral evidence
so at w_dep = w_comp = 0 the lateral term is identically 0 and the ranking IS the base retriever's
ranking, unchanged. (Scoring by v_peak instead would NOT preserve it: reset-to-zero plus tau means a
weak unit firing late can carry a larger membrane than a strong unit that fired at t=1.) The spikes
still do real work -- they GATE which units are allowed to emit lateral signal, and homeostasis
bounds how long a hub may keep emitting.

NOT CLAIMED: nothing is trained. w_dep/w_comp are hand-set scalars; algo_grr_spike.SpikingGraphLayer
carries the surrogate-gradient path if they are ever fit. AND `--compare` is currently INCONCLUSIVE
on this corpus -- three pre-existing defects stop the no-GPU scaleup path from scoring retrieval:

  1. DERIVE GATE IGNORES verify_fn (algo_grr_membrane.MembraneSolver.solve). `_coverage` accepts
     either `task["verify_fn"]` or tuple `tests`, but the derive branch calls
     `verify_code(code, task["entry"], task["tests"])` only. The compose corpus carries a verify_fn
     with `tests == []`, so derive scores 0.0 unconditionally. Measured: the exact code the derive
     path emits returns (1.0, 'all pass') from the task's own verify_fn while the solver records
     0.0 -- so every task needing an atom outside the seed graph is unsolvable. Structurally
     unable to pass, the same class as commit 761b3de.
  2. assemble_corpus HANDS make_stub_compiler THE FULL SOLUTION (`t["reference"]`, both prims
     DEFINED inline), against that function's own docstring ("a block that CALLS the selected
     atoms ... missing atom -> NameError -> verify fails"). Measured: coverage 1.0 with ZERO atoms
     selected, so the membrane is bypassed and every policy scores 100% for free.
  3. make_spreading_policy IS DEAD CODE (algo_grr_retrieval.py:291): it passes `base_retriever=`
     to a constructor whose parameter is `base`, so it raises TypeError on every call.

Together, (1) and (2) bracket the harness between "everyone wins for free" and "nobody can win";
neither end scores a retrieval policy. `_compare` therefore reports INCONCLUSIVE rather than
reading a 6.9-vs-7.0 lm/task gap at 0 solved as a result.

ALSO MEASURED, and it blunts the excitation half of the idea: `_curate` calls
`resolve_closure(graph, selected)`, which ALREADY walks depend edges transitively. Selecting
is_perfect deterministically pulls sum_divisors and divisors with no policy involved. So w_dep
largely duplicates an existing deterministic mechanism ("structure = traverse, content = rank",
per READ_THIS). The part nothing currently does is ARBITRATION -- w_comp.

    selftest:  python -m v5.runtime.algo_grr_spike_policy --selftest
    compare :  python -m v5.runtime.algo_grr_spike_policy --compare --n 60
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch                                                             # noqa: E402

from v5.runtime.algo_grr_spike import SpikingGraphLayer                  # noqa: E402


class SpikingGraphPolicy:
    """MembraneSolver policy_fn with LIF dynamics over the real typed graph."""

    def __init__(self, graph, base=None, *, top_k: int = 12, T: int = 4,
                 w_dep: float = 0.0, w_comp: float = 0.0, theta0: float = 0.15,
                 tau: float = 0.85, alpha: float = 0.6, beta: float = 0.02):
        from v5.runtime.algo_grr_retrieval import CachedTokenRetriever
        self.graph = graph
        self.base = base or CachedTokenRetriever(graph)
        self.top_k, self.T = top_k, T
        self.w_dep, self.w_comp = w_dep, w_comp
        self._layer_kw = dict(theta0=theta0, tau=tau, alpha=alpha, beta=beta,
                              w_dep=w_dep, w_comp=w_comp)
        self._sib: dict | None = None                     # concept hub -> member impls (cached)
        self._nodes_seen = 0
        self.stats = {"calls": 0, "steps": 0, "excited": 0, "reordered": 0}

    # ── structure ───────────────────────────────────────────────────────────────────────────
    def _siblings(self):
        """concept-hub membership, rebuilt when the graph grows (banking adds atoms + edges)."""
        if self._sib is None or len(self.graph.nodes) != self._nodes_seen:
            hub = defaultdict(set)
            for e in self.graph.edges:
                if e.relation == "part_of":
                    hub[e.dst].add(e.src)
            self._sib = {n: {m for h, ms in hub.items() if n in ms for m in ms if m != n}
                         for n in self.graph.nodes}
            self._nodes_seen = len(self.graph.nodes)
        return self._sib

    def _masks(self, names):
        """(dep, comp) over the candidate set. dep = real `depend` edges (either direction) --
        the composition closure. comp = shares a concept hub AND has no depend edge -- rivals for
        the same role. A depend partner is never a rival: needing B is the opposite of B being an
        alternative to A."""
        n = len(names)
        idx = {x: i for i, x in enumerate(names)}
        dep = torch.zeros(n, n)
        for e in self.graph.edges:
            if e.relation == "depend" and e.src in idx and e.dst in idx:
                dep[idx[e.src], idx[e.dst]] = 1.0
                dep[idx[e.dst], idx[e.src]] = 1.0
        sib = self._siblings()
        comp = torch.zeros(n, n)
        for i, a in enumerate(names):
            for b in sib.get(a, ()):
                j = idx.get(b)
                if j is not None and dep[i, j] == 0:
                    comp[i, j] = 1.0
        return dep, comp

    # ── the policy ──────────────────────────────────────────────────────────────────────────
    def __call__(self, task, selected, graph=None, retriever=None):
        self.stats["calls"] += 1
        sel = list(selected or [])
        ranked = self.base.rank(task["text"], exclude=set(sel))
        ranked = [(n, float(s)) for n, s in ranked][: self.top_k]
        if not ranked:
            return ranked
        # Candidate units + the already-selected atoms, which participate in the dynamics (they
        # emit) but are never returned (the solver would skip them anyway).
        names = [n for n, _ in ranked] + [s for s in sel if s in self.graph.nodes]
        base_s = torch.tensor([s for _, s in ranked] + [0.0] * (len(names) - len(ranked)),
                              dtype=torch.float32)
        n_cand = len(ranked)
        dep, comp = self._masks(names)
        layer = SpikingGraphLayer(**self._layer_kw)
        layer.reset_state(len(names))
        clamp = torch.zeros(len(names))
        clamp[n_cand:] = 1.0                              # selected atoms are CLAMPED FIRING
        U = self.w_comp * comp - self.w_dep * dep
        lat = torch.zeros(len(names))
        with torch.no_grad():
            for _t in range(self.T):
                y = layer.step(base_s, dep, comp)
                y = torch.maximum(y, clamp)               # a committed atom keeps emitting
                layer.y_prev = y
                # accumulated GRAPH evidence: the negated lateral term, so an excitatory (negative
                # U) edge adds and an inhibitory one subtracts. Identically zero when U == 0.
                lat = lat + (-(y @ U.T))
                self.stats["steps"] += 1
        score = base_s + lat
        out = sorted(((names[i], float(score[i])) for i in range(n_cand)),
                     key=lambda z: -z[1])
        if self.w_dep or self.w_comp:
            self.stats["excited"] += int((lat[:n_cand] > 1e-9).sum())
            if [n for n, _ in out] != [n for n, _ in ranked]:
                self.stats["reordered"] += 1
        return out


# ================================================================================================
# selftests
# ================================================================================================
def _selftest() -> bool:
    from v5.runtime.algo_grr_poison_test import load_seed
    print("algo_grr_spike_policy --selftest: LIF hop policy over the real seed graph\n")
    ok_all = True

    def chk(tag, cond, detail=""):
        nonlocal ok_all
        ok_all &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {tag}{('  — ' + detail) if detail else ''}")

    g = load_seed()
    task = {"text": "check whether a number is a perfect number"}

    # [1] PRIOR PRESERVATION — U=0 must return the base retriever's ranking, unchanged.
    from v5.runtime.algo_grr_retrieval import CachedTokenRetriever
    base = CachedTokenRetriever(g)
    p0 = SpikingGraphPolicy(g, base=base, w_dep=0.0, w_comp=0.0)
    got = p0(task, [])
    want = [(n, float(s)) for n, s in base.rank(task["text"], exclude=set())][: p0.top_k]
    chk("[1] U=0 reproduces the base ranking exactly",
        [n for n, _ in got] == [n for n, _ in want]
        and all(abs(a - b) < 1e-6 for (_, a), (_, b) in zip(got, want)),
        f"top3={[n for n, _ in got[:3]]}")

    # [2] STRUCTURE — the masks must come from the graph's real relations.
    names = ["impl_is_perfect", "impl_sum_divisors", "impl_divisors", "impl_is_prime"]
    dep, comp = p0._masks(names)
    chk("[2] dep mask == real depend edges (is_perfect–sum_divisors–divisors chain)",
        dep[0, 1] == 1 and dep[1, 2] == 1 and dep[0, 2] == 0,
        f"perfect~sum={dep[0,1]} sum~div={dep[1,2]} perfect~div={dep[0,2]} (2 hops, not 1)")
    chk("[2b] concept siblings compete, depend partners do not",
        comp[0, 3] == 1 and comp[0, 1] == 0,
        f"perfect~is_prime={comp[0,3]} perfect~sum_divisors={comp[0,1]}")

    # [3] COMPLEMENT RETRIEVAL — the property MembraneSolver's own comment says the policy owes it.
    #     The contrast is ACROSS w_dep with the same atom committed, not before/after committing:
    #     sum_divisors can already sit at rank 0 on token overlap alone, and then "moved up" is
    #     unmeasurable and the test passes or fails for reasons unrelated to the edge.
    p1 = SpikingGraphPolicy(g, base=base, w_dep=0.9, w_comp=0.0)
    sel = ["impl_is_perfect"]
    tgt = "impl_sum_divisors"
    s_on = dict(p1(task, sel))
    s_off = dict(p0(task, sel))
    gain = s_on.get(tgt, 0.0) - s_off.get(tgt, 0.0)
    # the excitation must reach the DIRECT partner and must not be a blanket lift of everything
    others = [n for n in s_off if n != tgt]
    blanket = sum(1 for n in others if s_on.get(n, 0.0) - s_off.get(n, 0.0) >= gain)
    chk("[3] committing an atom EXCITES its depend-complement specifically",
        gain > 1e-6 and blanket == 0,
        f"{tgt} score {s_off.get(tgt, 0):.3f} -> {s_on.get(tgt, 0):.3f} (+{gain:.3f}); "
        f"{blanket}/{len(others)} others gained as much")

    # [4] ARBITRATION — inhibition must reorder rivals; and it must be the INHIBITION doing it,
    #     so the same call with w_comp=0 has to leave the order alone.
    p2 = SpikingGraphPolicy(g, base=base, w_dep=0.0, w_comp=0.9)
    o_inh = [n for n, _ in p2(task, ["impl_is_perfect"])]
    o_off = [n for n, _ in p0(task, ["impl_is_perfect"])]
    chk("[4] inhibition arbitrates between concept siblings",
        o_inh != o_off, f"inhibited_top3={o_inh[:3]} free_top3={o_off[:3]}")

    # [5] NO NEGATIVE EDGES EXIST — the premise of the whole arbitration argument, measured.
    rels = {e.relation for e in g.edges}
    neg = rels & {"avoid_if", "contradict", "conflict", "refute", "trap_for"}
    chk("[5] graph has NO negative relations (so SAR cannot arbitrate)",
        not neg and rels == {"part_of", "depend"}, f"relations={sorted(rels)}")

    print(f"\n  ALGO_GRR_SPIKE_POLICY SELFTEST -> {'PASS' if ok_all else 'FAIL'}")
    return ok_all


def _compare(n: int = 60, w_dep: float = 0.9, w_comp: float = 0.5, max_hops: int = 4) -> bool:
    """Three policies, same corpus, same graph seed, same stub compiler. No GPU, no LM.

    Metric is solved AND lm/task: each hop triggers a coverage recompute (a compile call), so
    lm/task IS the search cost. The claim for the lateral graph is amortization -- equal solve at
    FEWER hops -- so a tie on solved with a lower lm/task is the win condition, not solved alone.
    """
    from v5.runtime.algo_grr_poison_test import load_seed
    from v5.runtime.algo_grr_membrane import make_stub_compiler
    from v5.runtime.algo_grr_retrieval import CachedTokenRetriever, make_spreading_policy
    from v5.runtime.algo_grr_scaleup import assemble_corpus, run_scaleup

    tasks, _ref_stubs = assemble_corpus(n_compose=n, mbpp_limit=0, seed=0)
    # CALL-ONLY RECIPES. assemble_corpus hands make_stub_compiler each task's `reference`, which is
    # the FULL solution with both prims DEFINED inline -- measured: coverage is 1.0 with ZERO atoms
    # selected, so the compiler never needs the membrane, every policy scores 100%, and a retrieval
    # comparison built on it is vacuous (all three arms tied at 60/60, lm/task 2.0, measuring
    # nothing). make_stub_compiler's own docstring asks for the opposite: "a `def <entry>(...)`
    # block that CALLS the selected atoms ... missing atom -> NameError -> verify fails". These
    # recipes call and never define, so coverage genuinely depends on what retrieval surfaced.
    from v5.runtime.algo_grr_compose import INNER, OUTER
    prim_src = {k: v[0] for k, v in {**INNER, **OUTER}.items()}
    stubs, needs = {}, {}
    for t in tasks:
        pr = t.get("_prims") or ()
        if len(pr) == 2:
            stubs[t["entry"]] = f"def {t['entry']}(n):\n    return {pr[1]}({pr[0]}(n))\n"
            needs[t["entry"]] = list(pr)
        else:
            stubs[t["entry"]] = t.get("reference", "")
            needs[t["entry"]] = []

    def compile_fn(spec):
        """Stub 'frozen compiler' that is RETRIEVAL-SENSITIVE and still solvable.

        Normal path: the entry body CALLS the prims and never defines them, so a prim the membrane
        failed to surface is a NameError and coverage really drops. Derive path: `spec['derive']`
        means the solver has exhausted its hops and is handing the LM a hole, so the missing prims
        are authored here (the "stub author correct" arm) -- otherwise a prim absent from the seed
        graph can NEVER be produced and the whole corpus scores 0/60, which is the other degenerate
        end of this harness. Retrieval quality then shows up where it should: in HOPS/lm-per-task,
        because a task whose prims are already banked reaches coverage without paying for derive."""
        atoms = spec.get("atoms", [])
        parts = [a["code"].rstrip("\n") for a in atoms if a.get("code")]
        if spec.get("derive"):
            have = {a.get("name") for a in atoms}
            parts += [prim_src[p] for p in needs.get(spec["entry"], [])
                      if p not in have and p in prim_src]
        closure = "\n\n".join(parts)
        body = stubs.get(spec["entry"], "")
        return (closure + "\n\n" + body) if closure else body

    holder = {}

    def arm_default(g):
        return None                                              # solver's built-in cosine policy

    def arm_spread(g):
        # NOT make_spreading_policy(): that adapter passes `base_retriever=` to a constructor
        # whose parameter is `base` (algo_grr_retrieval.py:291), so it raises TypeError on every
        # call and has never actually run. Constructing SAR directly keeps the baseline honest
        # instead of quietly dropping the arm.
        from v5.runtime.algo_grr_retrieval import SpreadingActivationRetriever
        sar = SpreadingActivationRetriever(g, base=CachedTokenRetriever(g), steps=5)
        return lambda task, selected, _g, _r: sar.rank(task["text"],
                                                       exclude=set(selected or ()))

    def arm_spike(g):
        holder["p"] = SpikingGraphPolicy(g, base=CachedTokenRetriever(g),
                                         w_dep=w_dep, w_comp=w_comp)
        return holder["p"]

    rows = []
    for tag, mk in [("cosine baseline (solver default)", arm_default),
                    ("SpreadingActivation (existing edge baseline)", arm_spread),
                    (f"SpikingGraphPolicy (w_dep={w_dep} w_comp={w_comp})", arm_spike)]:
        g = load_seed()                                          # FRESH graph per arm
        print(f"\n=== {tag} ===")
        res, _g = run_scaleup(g, tasks, compile_fn, policy_fn=mk(g),
                              max_hops=max_hops, verbose=False, report_every=10 ** 9)
        lm = res["per"][-1]["lm_per_task"] if res.get("per") else None
        rows.append((tag, res["solved"], lm, res["derived_reuse"]))
        print(f"  solved={res['solved']}/{len(tasks)}  lm/task={lm}  "
              f"deriv_reuse={res['derived_reuse']}  banked={res['banked']}")

    print(f"\n  SUMMARY                                        solved   lm/task  deriv_reuse")
    for tag, s, lm, dr in rows:
        print(f"    {tag:<44} {str(s):>4}/{len(tasks)}   {str(lm):>6}   {dr}")
    if "p" in holder:
        st = holder["p"].stats
        print(f"\n  spike stats: calls={st['calls']} steps={st['steps']} "
              f"excited={st['excited']} reordered_vs_base={st['reordered']}")
    _, s_cos, lm_cos, _ = rows[0]
    _, s_spr, lm_spr, _ = rows[1]
    _, s_spk, lm_spk, _ = rows[2]
    best_s, best_lm = max(s_cos, s_spr), min(lm_cos, lm_spr)
    # A degenerate run (nobody solves anything) must never read as a win: at 0 solved the lm/task
    # column is just how fast each arm exhausts its hop budget, and a 6.9-vs-7.0 gap there is noise
    # dressed as a result. Require the harness to have discriminated at all before judging.
    if not best_s:
        print("\n  INCONCLUSIVE — every arm solved 0; the harness is not scoring retrieval, so the "
              "lm/task column is meaningless here. See the derive-gate note in the docstring.")
        return False
    won = (s_spk > best_s) or (s_spk == best_s and lm_spk is not None and lm_spk < best_lm)
    print(f"\n  vs BEST BASELINE -> {'AHEAD' if won else 'NOT AHEAD (dynamics not load-bearing)'}")
    return won


def main() -> None:
    ap = argparse.ArgumentParser(description="Spiking graph layer as a MembraneSolver hop policy.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--n", type=int, default=60, help="corpus size for --compare")
    ap.add_argument("--w-dep", type=float, default=0.9, dest="w_dep")
    ap.add_argument("--w-comp", type=float, default=0.5, dest="w_comp")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    if a.compare:
        sys.exit(0 if _compare(n=a.n, w_dep=a.w_dep, w_comp=a.w_comp) else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
