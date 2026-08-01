"""algo_grr_spike — the STATEFUL GRAPH LAYER as a planner: cognition in the graph, LM as speech motor.

A leaky integrate-and-fire layer whose units ARE candidate atoms and whose lateral matrix IS the atom
call-graph. Replaces CandidatePlanner's hand-written rule ("one hard atom + a wrapper -> wire
outer(inner(n))") with settled dynamics:

    v(t)       = tau*v(t-1) + ff - y(t-1) @ U.T      leaky evidence + graph propagation
    y(t)       = 1[v(t) > theta(t)]                   discrete selection, count NOT fixed
    theta(t+1) = theta(t) + alpha*y(t) - beta         homeostasis: fire often -> harder to fire

Each mechanism replaces something this repo hand-tuned: tau/T a fixed cycle count; theta a fixed pick
budget / magic relevance bar; U the "+3.0 depend-neighbour boost, depth-1 only" rule (depth becomes a
time constant -- fire, reset, theta-bump -- so a walk cannot run away).

U IS THE CALL GRAPH. Sign convention follows the LIF form: POSITIVE U inhibits, NEGATIVE excites.
A depend/uses edge is EXCITATORY, so 2-hop composition emerges from the dynamics instead of being
enumerated. A same-class non-dependent pair is INHIBITORY, scaled by token overlap -- arbitration for
the measured "router mispicks the wrapper on token collisions" failure (CandidatePlanner+topo 25/40).

REAL: units are the real AtomStore's atoms; U comes from the real AST-derived `store.meta[n].depends`;
the readout is an AtomProgram consumed by the real MembraneV2 realize->verify->bank loop; `--compare`
runs the real `run_v2_compare` harness.

NOT CLAIMED: nothing here is trained. U is 2 tied scalars, the dynamics 4 hyperparameters, all hand-set
at prior-preserving defaults. A surrogate-gradient path exists so they CAN be trained; no training run
has happened, so no result here is a learned-reasoner result. ~6 scalars, not N^2, deliberately: a free
NxN lateral matrix is the same mistake as the _SessionRanker MLP (0.75 -> 0.62 on 34 pairs) and the
trained recall head (flipped near-tied spans; the frozen prior won).

PRIOR PRESERVATION: with w_dep = w_comp = 0 and one settle step the recurrence collapses to
y = 1[ff > theta0] -- a plain threshold on the router's ranking, i.e. today's behaviour. Selftest [1]
asserts it, and `--compare` reproduces the CandidatePlanner baseline exactly (51/60, 20/30, banked 5).

EXECUTION CHANNEL (`exec_feedback=True`, OFF by default): a spike on an atom with code is an ACTION.
When the lit set forms a complete program whose atoms are all in the store, it is realized and run
through the task's own verifier; a failure bumps theta on the blamed atom and the dynamics re-settle
(y_{t+1}=execute(a_t,y_t) on the real verifier). OFF by default because it spends verifier calls the
baseline does not -- see RetryWrapperControl, which matches its solve rate at slightly FEWER calls.

    selftest (no GPU, no LM):  python -m v5.runtime.algo_grr_spike --selftest
    end-to-end vs the baseline planners (no GPU, stub author + simulated inline 3B):
                               python -m v5.runtime.algo_grr_spike --compare
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch                                                            # noqa: E402
import torch.nn as nn                                                   # noqa: E402

from v5.runtime.algo_grr_pipeline import AtomProgram, _tok, _desc       # noqa: E402


# ================================================================================================
# the layer — LIF + lateral graph + homeostatic threshold. ~6 scalars, size-agnostic in N.
# ================================================================================================
class SpikingGraphLayer(nn.Module):
    """Stateful graph layer over a VARIABLE number of units (the candidate atoms of one task).

    There is deliberately NO input_weight matrix. The prototype's `input_weight [H, input_dim]`
    pins the unit count at construction time, but the candidate pool changes size every task (and
    grows as the store banks atoms). The drive is the retrieval score itself, so the layer is
    size-agnostic and adding an atom to the graph never invalidates the layer.
    """

    def __init__(self, tau: float = 0.85, alpha: float = 0.6, beta: float = 0.02,
                 theta0: float = 0.5, w_dep: float = 0.0, w_comp: float = 0.0,
                 reset_to_zero: bool = True, spike_mode: str = "hard",
                 surrogate_gamma: float = 5.0):
        super().__init__()
        if spike_mode not in {"hard", "surrogate"}:
            raise ValueError("spike_mode must be 'hard' or 'surrogate'")
        self.tau, self.alpha, self.beta, self.theta0 = tau, alpha, beta, theta0
        self.reset_to_zero = reset_to_zero
        self.spike_mode, self.surrogate_gamma = spike_mode, surrogate_gamma
        # THE learnable parameters: two tied lateral scalars. Not N^2 -- see module docstring.
        self.w_dep = nn.Parameter(torch.tensor(float(w_dep)))     # excitation along call-edges
        self.w_comp = nn.Parameter(torch.tensor(float(w_comp)))   # inhibition between rivals
        self.v = self.theta = self.y_prev = self.v_peak = None

    def reset_state(self, n: int, device=None):
        dev = device or self.w_dep.device
        self.v = torch.zeros(n, device=dev)
        self.theta = torch.full((n,), float(self.theta0), device=dev)
        self.y_prev = torch.zeros(n, device=dev)
        # v_peak = the membrane value AT THE MOMENT OF FIRING, i.e. the evidence that actually
        # caused the decision. Needed because reset_to_zero wipes v for exactly the units that
        # fired: reading `v` after a step ranks every winner at 0.0 and the readout degenerates to
        # tie-breaking by name. v_peak is the honest strength signal; v is the live integrator.
        self.v_peak = torch.zeros(n, device=dev)

    def _spike(self, v, theta):
        logits = v - theta
        if self.spike_mode == "hard":
            return (logits > 0).float()
        y_soft = torch.sigmoid(self.surrogate_gamma * logits)          # straight-through
        return (logits > 0).float() + (y_soft - y_soft.detach())

    def step(self, ff: torch.Tensor, dep: torch.Tensor, comp: torch.Tensor) -> torch.Tensor:
        """One LIF step. dep/comp are [N,N] 0/1 masks; the SIGNED lateral matrix is assembled from
        them as U = w_comp*comp - w_dep*dep, so excitation and inhibition share one term and one
        sign convention (positive U inhibits) instead of being two ad-hoc code paths."""
        U = self.w_comp * comp - self.w_dep * dep
        self.v = self.tau * self.v + ff - (self.y_prev @ U.T)
        y = self._spike(self.v, self.theta)
        self.v_peak = torch.maximum(self.v_peak, self.v * y)           # pre-reset firing strength
        if self.reset_to_zero:
            self.v = self.v * (1.0 - y)
        self.theta = self.theta + self.alpha * y - self.beta
        self.y_prev = y
        return y

    def suppress(self, mask: torch.Tensor, amount: float = 1.0):
        """Raise theta on the given units. The execution channel's failure signal: a program that
        FAILED its verifier makes exactly the atoms that fired harder to fire again, so the next
        settling step explores a different combination. Same homeostatic variable as habituation --
        a verified failure is just a strong, targeted dose of it."""
        self.theta = self.theta + amount * mask


# ================================================================================================
# U from the real call graph
# ================================================================================================
def lateral_masks(store, names: list[str], seed: set) -> tuple:
    """Build (dep, comp) [N,N] masks from the REAL atom call-graph.

    dep[i,j]  = 1 when i and j are call-graph neighbours (either direction). Composition edge.
    comp[i,j] = token-overlap Jaccard when i,j are same-class (both seed or both non-seed) and NOT
                call-graph neighbours. Rivals for the same role; the more alike, the harder they
                compete. Uses the pipeline's own tokenizer so the notion of "alike" is the same one
                the router already ranks by.
    """
    n = len(names)
    dep = torch.zeros(n, n)
    comp = torch.zeros(n, n)
    meta = getattr(store, "meta", {})
    toks = {}
    for x in names:
        code = store.get(x, "")
        toks[x] = set(_tok(x)) | set(_tok(_desc(code) if code else x))
    for i, a in enumerate(names):
        da = set(meta[a].depends) if a in meta else set()
        for j, b in enumerate(names):
            if i == j:
                continue
            db = set(meta[b].depends) if b in meta else set()
            if b in da or a in db:
                dep[i, j] = 1.0
            elif (a in seed) == (b in seed):
                ta, tb = toks[a], toks[b]
                u = len(ta | tb)
                comp[i, j] = (len(ta & tb) / u) if u else 0.0
    return dep, comp


# ================================================================================================
# the planner
# ================================================================================================
class SpikingGraphPlanner:
    """Drop-in for CandidatePlanner: `.plan(task, ranked) -> AtomProgram`.

    Candidate pool = the router's ranked store atoms UNION the names the proposer decodes. The
    union matters: a store-only pool could never name an atom that does not exist yet, which would
    silently kill MembraneV2's author-on-demand path (and with it the whole compounding result).
    The layer's job is SELECTION and ARBITRATION over that pool, not name invention.
    """

    def __init__(self, store, seed_names, proposer=None, *, top_k: int = 10, T: int = 6,
                 exec_feedback: bool = False, prop_drive: float = 1.0,
                 layer: SpikingGraphLayer | None = None):
        self.store = store
        self.seed = set(seed_names)
        self.proposer = proposer                 # e.g. NeuralDecodePlanner; supplies missing names
        self.top_k, self.T = top_k, T
        # DRIVE FOR PROPOSER-NAMED ATOMS. Measured: entering them at the LAST ranked slot's drive
        # (0.1 at top_k=10) starves the author-on-demand path -- an atom that does not exist yet
        # can only be named by the decoder, and at floor drive it needs ~6 tau-steps to cross
        # theta0, so it usually never fires: banked 2 of 5 HARD atoms, 24/60 stream vs the
        # CandidatePlanner baseline's 51/60. It also inverts the baseline's trust structure, where
        # the decoded hard atom is taken DIRECTLY and only the wrapper comes from the router. The
        # decoder infers STRUCTURE, the router matches CONTENT; neither is the other's subordinate.
        self.prop_drive = prop_drive
        self.exec_feedback = exec_feedback
        self.layer = layer or SpikingGraphLayer()
        self.stats = {"tasks": 0, "verify_calls": 0, "exec_rejects": 0, "settle_steps": 0,
                      "fired_total": 0, "fallback": 0}

    # ── candidate pool ──────────────────────────────────────────────────────────────────────
    def _pool(self, task: dict, ranked) -> tuple:
        """(names, ff, prop) — ff is the drive: reciprocal rank for router-surfaced atoms (monotone
        in the router's own preference, bounded in (0,1]), and `prop_drive` for names the proposer
        decoded. A proposer name that the router ALSO surfaced keeps the stronger of the two: two
        independent sources agreeing is more evidence, not less."""
        ranked = list(ranked or [])[: self.top_k]
        names, ff = list(ranked), [1.0 / (1.0 + r) for r in range(len(ranked))]
        prop = None
        if self.proposer is not None:
            try:
                prop = self.proposer.plan(task, ranked)
            except Exception:                                        # noqa: BLE001 — proposer is optional
                prop = None
            floor = ff[-1] if ff else self.prop_drive
            for a in (prop.atoms if prop else []):
                if a == "n":
                    continue
                # ROLE-SPLIT TRUST. prop_drive lifts only NON-SEED (hard/recurring) names. The
                # decoder generalises on the recurring atom but cannot emit a novel held-out
                # WRAPPER -- that is the measured NeuralDecode failure (stream 40/40, held-out
                # 6/40). Boosting proposer wrappers too collapses this planner into exactly that
                # planner: measured here 60/60 stream but 10/30 held-out, against CandidatePlanner's
                # 51/60 and 20/30. So wrappers stay on the ROUTER's ranking and the dynamics
                # arbitrate within that role instead of the router's bare argmax.
                d = floor if a in self.seed else self.prop_drive
                if a in names:
                    i = names.index(a)
                    ff[i] = max(ff[i], d)
                else:
                    names.append(a)
                    ff.append(d)
        return names, torch.tensor(ff, dtype=torch.float32), prop

    # ── readout ─────────────────────────────────────────────────────────────────────────────
    def _program(self, names, fired, v) -> AtomProgram | None:
        """Lit set -> program. The wiring shape is the corpus's standard outer(inner(n)); which
        atom plays which role comes from the seed/non-seed split (a seed atom is a wrapper), and
        the pick within each role is by v_peak -- the evidence at firing time, not the initial
        rank. Returns None when the dynamics did not light a complete program."""
        lit = [(i, n) for i, n in enumerate(names) if fired[i] > 0]
        hard = [(float(v[i]), n) for i, n in lit if n not in self.seed]
        wrap = [(float(v[i]), n) for i, n in lit if n in self.seed]
        if not hard or not wrap:
            return None
        h = max(hard)[1]
        w = max(wrap)[1]
        return AtomProgram(atoms=[h, w], wiring=("call", w, [("call", h, ["n"])]))

    # ── the settling loop ───────────────────────────────────────────────────────────────────
    def plan(self, task: dict, ranked=None) -> AtomProgram:
        self.stats["tasks"] += 1
        names, ff, prop = self._pool(task, ranked)
        if not names:
            self.stats["fallback"] += 1
            return prop or AtomProgram(atoms=["n"], wiring="n")
        dep, comp = lateral_masks(self.store, names, self.seed)
        L = self.layer
        L.reset_state(len(names))
        cum = torch.zeros(len(names))
        best = None
        with torch.no_grad():
            for _t in range(self.T):
                y = L.step(ff, dep, comp)
                cum = torch.clamp(cum + y, max=1.0)      # a unit that fired ONCE stays in the lit
                self.stats["settle_steps"] += 1          # set: reset_to_zero makes firing transient,
                self.stats["fired_total"] += int(y.sum())  # but a decision is not un-made by decay
                prog = self._program(names, cum, L.v_peak)
                if prog is None:
                    continue
                if not self.exec_feedback:
                    best = prog
                    break
                ok = self._try_execute(task, prog)
                if ok is None or ok:                     # unverifiable (atom missing) or verified
                    best = prog
                    break
                # Verified WRONG. The failure is evidence against the CONJUNCTION, so blame is
                # assigned to the least-supported member (lowest v_peak) and the better-supported
                # one stays live to recombine. Suppressing every member instead eliminates atoms
                # the evidence still backs, and the settling dead-ends after two pairs. This is a
                # heuristic credit assignment, not a derived rule -- flagged as such.
                self.stats["exec_rejects"] += 1
                members = [i for i, n in enumerate(names) if n in prog.atoms]
                blame = min(members, key=lambda i: float(L.v_peak[i]))
                idx = torch.zeros(len(names))
                idx[blame] = 1.0
                L.suppress(idx, amount=1.0)
                cum = cum * (1.0 - idx)
        if best is None:
            self.stats["fallback"] += 1
            return prop or AtomProgram(atoms=["n"], wiring="n")
        return best

    def _try_execute(self, task: dict, prog: AtomProgram):
        """Run the program for real. None = cannot judge (an atom is not in the store yet, so the
        realizer cannot build the closure -- the same constraint SearchPlanner has). Never fabricates
        a verdict when it cannot execute."""
        vf = task.get("verify_fn")
        if vf is None or any(a not in self.store for a in prog.atoms if a != "n"):
            return None
        from v5.runtime.algo_grr_pipeline import realize
        try:
            code = realize(prog, self.store, task["entry"])
            self.stats["verify_calls"] += 1
            return bool(vf(code)[0] >= 1.0)
        except Exception:                                            # noqa: BLE001
            return False


# ================================================================================================
# the control the execution-channel result has to beat
# ================================================================================================
class RetryWrapperControl:
    """VERIFIER-BUDGET-MATCHED CONTROL. CandidatePlanner + "if it fails verification, try the next
    wrapper in router order" -- the same verifier access the spiking planner's action channel gets,
    with NO membrane, NO threshold, NO lateral graph.

    This exists because the action-channel result is otherwise uninterpretable. The wrapper role has
    only 7 candidates in this corpus (4 OUTER + 3 OUTER_HELD); an arm that may re-query the scoring
    function can walk that list and will look excellent for reasons that have nothing to do with
    dynamics. If this control matches the spiking planner, the gain is "we handed it a verifier" and
    the LIF layer is decorative. The spiking planner has to win on SOLVE at equal verifier calls, or
    on VERIFIER CALLS at equal solve -- amortizing search is the whole claim."""

    def __init__(self, store, seed_names, ckpt, max_tries: int = 4):
        from v5.runtime.algo_grr_pipeline import CandidatePlanner
        self.inner = CandidatePlanner(store, ckpt, seed_names)
        self.store, self.seed, self.max_tries = store, set(seed_names), max_tries
        self.stats = {"tasks": 0, "verify_calls": 0, "exec_rejects": 0}

    def plan(self, task: dict, ranked=None) -> AtomProgram:
        self.stats["tasks"] += 1
        base = self.inner.plan(task, ranked)
        hard = [a for a in base.atoms if a != "n" and a not in self.seed]
        vf = task.get("verify_fn")
        if not hard or vf is None:
            return base
        from v5.runtime.algo_grr_pipeline import realize
        h = hard[0]
        wraps = [a for a in (ranked or []) if a in self.seed][: self.max_tries]
        for w in wraps:
            prog = AtomProgram(atoms=[h, w], wiring=("call", w, [("call", h, ["n"])]))
            if any(a not in self.store for a in prog.atoms):
                return prog                                  # unverifiable -> author-on-demand path
            try:
                self.stats["verify_calls"] += 1
                if vf(realize(prog, self.store, task["entry"]))[0] >= 1.0:
                    return prog
            except Exception:                                # noqa: BLE001
                pass
            self.stats["exec_rejects"] += 1
        return base


# ================================================================================================
# selftests — mechanism invariants, then the real end-to-end harness
# ================================================================================================
def _toy_store():
    """Names deliberately mirror the real corpus's SHAPE: multi-token, with collisions inside each
    role (`digit_twice`/`digit_thrice`, `outer_plus_one`/`outer_bump_one`). The real corpus is the
    same — HARD={num_partitions, derangements, josephus, catalan, mult_persistence},
    OUTER={is_prime, reverse_digits, digit_sum, num_divisors}, HELD={is_even, last_digit,
    count_digits} — so `digit_sum`/`count_digits`/`reverse_digits`/`last_digit` genuinely collide on
    tokens. That collision IS the measured CandidatePlanner+topo failure (router mispicks the
    wrapper, 25/40) and is exactly what the inhibition term arbitrates. Single-token toy names
    would have made comp identically zero and the test vacuous."""
    from v5.runtime.algo_grr_pipeline import AtomStore
    s = AtomStore()
    s["digit_twice"] = "def digit_twice(n):\n    return n * 2\n"
    s["digit_thrice"] = "def digit_thrice(n):\n    return n * 3\n"
    s["outer_plus_one"] = "def outer_plus_one(n):\n    return digit_twice(n) + 1\n"   # real dep edge
    s["outer_bump_one"] = "def outer_bump_one(n):\n    return n + 1\n"
    return s


def _selftest() -> bool:
    print("algo_grr_spike --selftest: LIF planner mechanism invariants\n")
    ok_all = True

    def chk(tag, cond, detail=""):
        nonlocal ok_all
        ok_all &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {tag}{('  — ' + detail) if detail else ''}")

    # [1] PRIOR PRESERVATION — U=0, T=1 must reduce to a threshold on the router's own ranking.
    L = SpikingGraphLayer(w_dep=0.0, w_comp=0.0, theta0=0.4, tau=0.85)
    L.reset_state(4)
    ff = torch.tensor([1.0, 0.5, 0.333, 0.25])                          # reciprocal-rank drive
    y = L.step(ff, torch.zeros(4, 4), torch.zeros(4, 4))
    chk("[1] U=0,T=1 == threshold on router score",
        torch.equal(y, (ff > 0.4).float()), f"y={y.tolist()} vs {(ff > 0.4).float().tolist()}")

    # [2] INHIBITION — two near-identical rivals. The stronger crosses first; the weaker would
    #     cross on the next step as tau-integration lifts it, and inhibition has to hold it down.
    #     alpha=beta=0 isolates the lateral term from homeostasis (otherwise the theta bump alone
    #     silences both units and the test passes for the wrong reason).
    comp = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    dep0 = torch.zeros(2, 2)
    ff2 = torch.tensor([0.60, 0.55])
    kw = dict(theta0=0.58, tau=0.9, alpha=0.0, beta=0.0)
    L_on = SpikingGraphLayer(w_comp=0.9, **kw)
    L_on.reset_state(2); y_on1 = L_on.step(ff2, dep0, comp)
    y_on = L_on.step(ff2, dep0, comp)
    L_off = SpikingGraphLayer(w_comp=0.0, **kw)
    L_off.reset_state(2); L_off.step(ff2, dep0, comp)
    y_off = L_off.step(ff2, dep0, comp)
    chk("[2] lateral inhibition suppresses the weaker rival",
        y_on1.tolist() == [1.0, 0.0] and y_on[1] == 0 and y_off[1] == 1,
        f"t1={y_on1.tolist()} inhibited={y_on.tolist()} free={y_off.tolist()}")

    # [3] EXCITATION — a sub-threshold atom on a call-edge from a firing atom must be pulled in.
    #     This is composition emerging from the dynamics rather than from enumeration.
    dep = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    ff3 = torch.tensor([0.9, 0.2])                                      # unit 1 alone cannot fire
    L_ex = SpikingGraphLayer(w_dep=0.8, theta0=0.5, tau=0.0)
    L_ex.reset_state(2)
    y1 = L_ex.step(ff3, dep, torch.zeros(2, 2))
    y2 = L_ex.step(ff3, dep, torch.zeros(2, 2))
    chk("[3] call-edge excitation pulls in the dep partner",
        y1[1] == 0 and y2[1] == 1, f"t1={y1.tolist()} t2={y2.tolist()}")

    # [4] HOMEOSTASIS — constant drive must NOT produce a constant fire; theta has to climb and
    #     break the loop. This is the anti-repetition property for the action channel.
    L_h = SpikingGraphLayer(alpha=0.6, beta=0.0, theta0=0.3, tau=0.0)
    L_h.reset_state(1)
    fires = [int(L_h.step(torch.tensor([0.8]), torch.zeros(1, 1), torch.zeros(1, 1))[0]) for _ in range(6)]
    chk("[4] homeostasis breaks a repetition loop", sum(fires) < 6, f"fires={fires}")

    # [5] SURROGATE GRADIENT — the tied scalars must actually receive gradient, else "trainable
    #     later" is a claim with nothing behind it. Needs TWO steps (the lateral term is y_prev@U.T,
    #     and y_prev is zero on the first step, so U has no path to the loss yet) and DISTINCT
    #     dep/comp masks (identical masks make U = w_comp*M - w_dep*M, which cancels at w_dep=w_comp
    #     and silently zeroes both grads).
    dep_g = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    comp_g = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    ff_g = torch.tensor([0.9, 0.8])                                     # both fire at t=1
    L_g = SpikingGraphLayer(w_dep=0.3, w_comp=0.3, spike_mode="surrogate", tau=0.0,
                            theta0=0.5, alpha=0.0, beta=0.0)
    L_g.reset_state(2)
    L_g.step(ff_g, dep_g, comp_g)
    L_g.step(ff_g, dep_g, comp_g).sum().backward()
    chk("[5] surrogate gradient reaches w_dep AND w_comp",
        L_g.w_dep.grad is not None and L_g.w_comp.grad is not None
        and float(L_g.w_dep.grad.abs()) > 0 and float(L_g.w_comp.grad.abs()) > 0,
        f"d/dw_dep={float(L_g.w_dep.grad):+.3f} d/dw_comp={float(L_g.w_comp.grad):+.3f}")

    # [6] REAL CALL GRAPH — U must come from the store's own AST-derived depends, not from names.
    s = _toy_store()
    names = ["digit_twice", "digit_thrice", "outer_plus_one", "outer_bump_one"]
    seeds = {"outer_plus_one", "outer_bump_one"}
    dep_m, comp_m = lateral_masks(s, names, seed=seeds)
    chk("[6] dep mask == real call graph (outer_plus_one calls digit_twice)",
        dep_m[2, 0] == 1 and dep_m[0, 2] == 1 and dep_m[1, 2] == 0,
        f"outer_plus_one~digit_twice={dep_m[2,0]} digit_thrice~outer_plus_one={dep_m[1,2]}")
    chk("[6b] token-colliding rivals compete, dep partners do not",
        comp_m[0, 1] > 0 and comp_m[2, 3] > 0 and comp_m[2, 0] == 0,
        f"twice~thrice={comp_m[0,1]:.2f} plus~bump={comp_m[2,3]:.2f} dep-pair={comp_m[2,0]:.2f}")

    # [7] PLANNER READOUT — a real AtomProgram of the corpus's outer(inner(n)) shape.
    p = SpikingGraphPlanner(s, seed_names=seeds, top_k=4,
                            layer=SpikingGraphLayer(theta0=0.2, tau=0.0))
    ranked = ["digit_twice", "outer_plus_one", "digit_thrice", "outer_bump_one"]
    prog = p.plan({"entry": "f"}, ranked=ranked)
    chk("[7] readout is a wired AtomProgram",
        isinstance(prog, AtomProgram) and len(prog.atoms) == 2
        and prog.wiring[0] == "call" and prog.wiring[1] in seeds,
        f"atoms={prog.atoms} wiring={prog.wiring}")

    # [8] EXECUTION CHANNEL — a verified-WRONG program must be abandoned for a different one, and
    #     the rejection must be counted. The verifier really executes the realized code. Target
    #     f(5)==11 is wrap_b(twice(5)); the dynamics' first pick (strongest drive on both roles)
    #     is wrap_a(twice(5))==21, so the channel has to reject once and re-settle to pass.
    task = {"entry": "f", "verify_fn": lambda c: (1.0 if _run(c) == 11 else 0.0,)}
    p2 = SpikingGraphPlanner(s, seed_names=seeds, top_k=4, T=8, exec_feedback=True,
                             layer=SpikingGraphLayer(theta0=0.2, tau=0.0))
    prog2 = p2.plan(task, ranked=ranked)
    got = _run(_realize_str(prog2, s, "f"))
    chk("[8] execution channel rejects wrong programs and re-settles",
        got == 11 and p2.stats["exec_rejects"] >= 1 and p2.stats["verify_calls"] >= 2,
        f"f(5)={got} rejects={p2.stats['exec_rejects']} verify_calls={p2.stats['verify_calls']}")

    print(f"\n  ALGO_GRR_SPIKE SELFTEST -> {'PASS' if ok_all else 'FAIL'}")
    return ok_all


def _realize_str(prog, store, entry):
    from v5.runtime.algo_grr_pipeline import realize
    return realize(prog, store, entry)


def _run(code, n: int = 5):
    ns = {}
    try:
        exec(code, ns)                                                  # noqa: S102
        return ns["f"](n)
    except Exception:                                                   # noqa: BLE001
        return None


def _compare(exec_feedback: bool = False, w_dep: float = 0.0, w_comp: float = 0.0) -> bool:
    """End-to-end on the REAL harness, ALL ARMS IN ONE PROCESS on the same corpus, same stub author,
    same simulated-3B inline baseline.

    The reference arm is CandidatePlanner -- the pure-neural planner this replaces. It is NOT
    `algo_grr_pipeline --selftest-v2`'s 60/60 30/30: that run passes no make_planner, so it falls
    through to OraclePlanner() reading task['_prims'], i.e. the oracle READ_THIS records as caught.
    Quoting it as the bar would be comparing a real planner against ground truth."""
    import random
    from v5.runtime.algo_grr_compose import gen_corpus_hard, HARD, OUTER, OUTER_HELD
    from v5.runtime.algo_grr_pipeline import run_v2_compare, CandidatePlanner, NeuralDecodePlanner

    src = {**{k: v[0] for k, v in HARD.items()},
           **{k: v[0] for k, v in OUTER.items()},
           **{k: v[0] for k, v in OUTER_HELD.items()}}
    author = lambda name, task: src.get(name, "")                       # noqa: E731
    stream = gen_corpus_hard(60, seed=0)
    holdout = gen_corpus_hard(30, seed=0, holdout=True)
    seed_names = set(OUTER) | set(OUTER_HELD)
    ckpt = Path(_ROOT) / "artifacts" / "planner_hard.pt"
    if not ckpt.exists():
        print(f"  planner checkpoint missing: {ckpt} — cannot run the pure-neural arms.")
        return False

    def make_inline():
        rng = random.Random(0)          # fresh per arm: the RAG baseline must be identical across
        def inline(task, retrieved=None):                               # arms, not drift with call order
            hard, outer = task["_prims"]
            if rng.random() < 0.35:
                return (f"{src[hard]}\n{src[outer]}\ndef {task['entry']}(n):\n"
                        f"    return {outer}({hard}(n))\n")
            return f"def {task['entry']}(n):\n    return n\n"
        return inline

    held = {}

    def make_spike(store):
        held["spike"] = SpikingGraphPlanner(
            store, seed_names, proposer=NeuralDecodePlanner(store, str(ckpt)),
            exec_feedback=exec_feedback, layer=SpikingGraphLayer(w_dep=w_dep, w_comp=w_comp))
        return held["spike"]

    def make_control(store):
        held["ctrl"] = RetryWrapperControl(store, seed_names, str(ckpt))
        return held["ctrl"]

    arms = [("CandidatePlanner (pure-neural baseline)",
             lambda st: CandidatePlanner(st, str(ckpt), seed_names)),
            ("RetryWrapperControl (verifier-budget-matched, no dynamics)", make_control),
            (f"SpikingGraphPlanner (exec={exec_feedback} w_dep={w_dep} w_comp={w_comp})", make_spike)]
    out = {}
    for tag, mk in arms:
        print(f"\n=== {tag} ===")
        out[tag] = run_v2_compare(stream, holdout, author, make_inline(),
                                  make_planner=mk, verbose=False)
    sp, ct = held["spike"].stats, held["ctrl"].stats
    base, ctrl, mine = (out[arms[0][0]], out[arms[1][0]], out[arms[2][0]])
    print(f"\n  spike stats: settle_steps={sp['settle_steps']} fired={sp['fired_total']} "
          f"fallback={sp['fallback']} verify_calls={sp['verify_calls']} rejects={sp['exec_rejects']}")
    print(f"  ctrl  stats: verify_calls={ct['verify_calls']} rejects={ct['exec_rejects']}")
    print(f"\n  SUMMARY            stream / held-out / banked / verify_calls")
    print(f"    baseline       : {base['ours_stream']:>3}/60   {base['ours_hold']:>3}/30    "
          f"{base['banked']}      0")
    print(f"    retry control  : {ctrl['ours_stream']:>3}/60   {ctrl['ours_hold']:>3}/30    "
          f"{ctrl['banked']}      {ct['verify_calls']}")
    print(f"    spike          : {mine['ours_stream']:>3}/60   {mine['ours_hold']:>3}/30    "
          f"{mine['banked']}      {sp['verify_calls']}")
    print(f"    RAG            : {base['rag_stream']:>3}/60   {base['rag_hold']:>3}/30    —      0")
    # The claim is amortization, so the bar is the CONTROL, not the baseline: beat it on solve at
    # equal verifier calls, or match its solve using fewer.
    won = (mine["ours_hold"] > ctrl["ours_hold"]
           or (mine["ours_hold"] == ctrl["ours_hold"] and sp["verify_calls"] < ct["verify_calls"]))
    print(f"\n  vs CONTROL -> {'AHEAD' if won else 'NOT AHEAD (dynamics not yet load-bearing)'}")
    return won


def main() -> None:
    ap = argparse.ArgumentParser(description="Spiking graph-layer planner (LIF over the atom call graph).")
    ap.add_argument("--selftest", action="store_true", help="mechanism invariants, no GPU/LM")
    ap.add_argument("--compare", action="store_true", help="end-to-end vs RAG on the real harness")
    ap.add_argument("--exec-feedback", action="store_true", dest="exec_feedback",
                    help="turn the ACTION channel on (verify in the settling loop; costs verifier calls)")
    ap.add_argument("--w-dep", type=float, default=0.0, dest="w_dep",
                    help="lateral EXCITATION along call-edges (0 = off = prior-preserving)")
    ap.add_argument("--w-comp", type=float, default=0.0, dest="w_comp",
                    help="lateral INHIBITION between token-colliding rivals (0 = off)")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    if a.compare:
        sys.exit(0 if _compare(exec_feedback=a.exec_feedback,
                               w_dep=a.w_dep, w_comp=a.w_comp) else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
