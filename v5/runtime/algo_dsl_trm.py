"""GRR-6 phase B: the TRM emits a DSL PROGRAM autoregressively — the piece where composition is real
reasoning, not a 4-way lookup.

At each decode step the reasoner, conditioned on the goal + partial program + the graph's atoms, picks:
an OP (FILTER/MAP/MAP2/KEEP_MAKEABLE/REDUCE), and its ARG (an ATOM pointer for the ops, an AGG for
REDUCE), halting at REDUCE. The realizer (algo_dsl) compiles the program to code that CALLS its atoms
(compose-forced). Imitating the reference programs (algo_dsl._PROGRAMS) teacher-forced = RECALL (0%
compositional generalization — it memorizes family->program). GRR-7 turns recall into REASONING with STaR
(expert iteration): SEARCH program-space (a bounded ENUMERATION of valid pipelines — pure policy-sampling
can't even explore an unseen family), VERIFY each against oracle I/O (NOT the reference program), KEEP the
fully-general ones, SFT the net on them so a plain decode jumps straight to the discovered program. This
low-variance imitation of verified solutions replaces GRPO's high-variance policy gradient (the wanderer),
ordered by a length curriculum (short programs before long). `--compo-gen-star` is the honest 2-number
test: zero-shot (recall) vs with-search (reasoning), across seeds so the variance is visible (selftest:
held-out max_prime_digitsum 0% zero-shot -> 100% with-search, identical across 3 seeds).

  selftest (no LM):           python -m v5.runtime.algo_dsl_trm --selftest
  compo-gen + search (molab): python -m v5.runtime.algo_dsl_trm --compo-gen-star --hard
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from v5.runtime.algo_dsl import _PROGRAMS, Op, atoms_of, realize_program

OPS = ["FILTER", "MAP", "MAP2", "KEEP_MAKEABLE", "REDUCE"]
AGGS = ["sum", "max", "count", "len"]
_REDUCE = OPS.index("REDUCE")


def program_to_steps(pipeline, atom_idx):
    """[(op_i, atom_i|-1, agg_i|-1)] teacher-forcing targets for a reference program."""
    steps = []
    for op in pipeline:
        oi = OPS.index(op.kind)
        if op.kind == "REDUCE":
            steps.append((oi, -1, AGGS.index(op.arg)))
        else:
            steps.append((oi, atom_idx[op.arg], -1))
    return steps


def _build():
    import torch
    import torch.nn as nn

    class ProgramDecoder(nn.Module):
        """Autoregressive: (goal, partial-program state, atoms) -> next (op, arg). GRU state carries the
        program so far; op/agg heads + an atom POINTER head choose the step."""
        def __init__(self, d_in=768, d=64):
            super().__init__()
            self.goal_proj = nn.Linear(d_in, d)
            self.atom_proj = nn.Linear(d_in, d)
            self.op_emb = nn.Embedding(len(OPS), d)
            self.agg_emb = nn.Embedding(len(AGGS), d)
            self.rnn = nn.GRUCell(d, d)
            self.op_head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, len(OPS)))
            self.agg_head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, len(AGGS)))
            self.q_atom = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
            self.d = d

        def _heads(self, g, state, A):
            h = torch.cat([g, state])
            return self.op_head(h), self.agg_head(h), A @ self.q_atom(h)      # op, agg, atom-pointer

        def loss(self, goal_vec, atom_vecs, steps):
            ce = nn.CrossEntropyLoss()
            g = self.goal_proj(goal_vec)
            A = self.atom_proj(atom_vecs)
            state = torch.zeros(self.d)
            total = torch.zeros(())
            for op_i, atom_i, agg_i in steps:
                op_l, agg_l, atom_l = self._heads(g, state, A)
                total = total + ce(op_l.unsqueeze(0), torch.tensor([op_i]))
                if op_i == _REDUCE:
                    total = total + ce(agg_l.unsqueeze(0), torch.tensor([agg_i]))
                    arg = self.agg_emb(torch.tensor(agg_i))
                    break
                total = total + ce(atom_l.unsqueeze(0), torch.tensor([atom_i]))
                arg = A[atom_i]
                state = self.rnn((self.op_emb(torch.tensor(op_i)) + arg).unsqueeze(0), state.unsqueeze(0))[0]
            return total

        def sample(self, goal_vec, atom_vecs, atom_names, temp=1.0, max_len=5):
            """Stochastic decode for RL: returns (program, sum log-prob). log-prob keeps grad (REINFORCE);
            the sampled indices are detached."""
            g = self.goal_proj(goal_vec)
            A = self.atom_proj(atom_vecs)
            state = torch.zeros(self.d)
            pipe, logp = [], torch.zeros(())
            for _ in range(max_len):
                op_l, agg_l, atom_l = self._heads(g, state, A)
                op = torch.distributions.Categorical(logits=op_l / temp).sample()
                logp = logp + torch.log_softmax(op_l, -1)[op]
                if OPS[int(op)] == "REDUCE":
                    agg = torch.distributions.Categorical(logits=agg_l / temp).sample()
                    logp = logp + torch.log_softmax(agg_l, -1)[agg]
                    pipe.append(Op("REDUCE", AGGS[int(agg)])); break
                ai = torch.distributions.Categorical(logits=atom_l / temp).sample()
                logp = logp + torch.log_softmax(atom_l, -1)[ai]
                pipe.append(Op(OPS[int(op)], atom_names[int(ai)]))
                state = self.rnn((self.op_emb(op) + A[ai]).unsqueeze(0), state.unsqueeze(0))[0]
            return pipe, logp

        @torch.no_grad()
        def decode(self, goal_vec, atom_vecs, atom_names, max_len=5):
            g = self.goal_proj(goal_vec)
            A = self.atom_proj(atom_vecs)
            state = torch.zeros(self.d)
            pipe = []
            for _ in range(max_len):
                op_l, agg_l, atom_l = self._heads(g, state, A)
                op = int(op_l.argmax())
                if OPS[op] == "REDUCE":
                    pipe.append(Op("REDUCE", AGGS[int(agg_l.argmax())])); break
                ai = int(atom_l.argmax())
                pipe.append(Op(OPS[op], atom_names[ai]))
                state = self.rnn((self.op_emb(torch.tensor(op)) + A[ai]).unsqueeze(0), state.unsqueeze(0))[0]
            return pipe

    return torch, nn, ProgramDecoder


def _family_len(fam: str) -> int:
    """Reference-program length (steps) — the curriculum axis (short programs before long)."""
    return len(_PROGRAMS[fam][1])


def _sft_steps(model, opt, traces, A, steps: int, seed: int = 0, clip: float = 1.0):
    """SFT gradient steps on an EXISTING model/opt (teacher-forced CE). Shared by warmstart imitation
    and STaR's imitate-your-own-successes update — same low-variance objective, no policy gradient."""
    import random
    import torch
    if not traces:
        return model
    rng = random.Random(seed)
    for _ in range(steps):
        gv, tgt = traces[rng.randrange(len(traces))]
        loss = model.loss(torch.as_tensor(gv, dtype=torch.float32), A, tgt)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)   # stabilize (lr-too-high diverged)
        opt.step()
    return model


def train_decoder(traces, atom_vecs, atom_names, d=64, steps=2000, lr=1e-3, seed=0):
    import torch
    torch, nn, ProgramDecoder = _build()
    torch.manual_seed(seed)
    model = ProgramDecoder(d_in=atom_vecs.shape[1], d=d)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    A = torch.as_tensor(atom_vecs, dtype=torch.float32)
    _sft_steps(model, opt, traces, A, steps, seed=seed)
    return model


def fast_reward(code: str, deps: str, name: str, n: int = 16, seed: int = 0) -> float:
    """In-process verify (no subprocess) returning the AGREEMENT FRACTION with the family oracle on n
    random inputs — a DENSE, reference-free reward (a close-but-wrong program scores 0.6, not 0), so RL
    has a gradient to climb. 1.0 iff fully general on this seed's inputs. Vary `seed` to re-check on FRESH
    inputs (the STaR keep-gate does — a kept program must be general on two disjoint input sets, not the
    reward's fixed set). Fast enough for many rollouts."""
    from v5.runtime.algo_compose_tasks import _FAMILIES, _FAMILIES_HARD, _sample
    fams = {**_FAMILIES, **_FAMILIES_HARD}
    if name not in fams:
        return 0.0
    kind, oracle = fams[name]
    ns: dict = {}
    try:
        exec((deps + "\n" + code) if deps else code, ns)
    except Exception:
        return 0.0
    fn = ns.get(name)
    if fn is None:
        return 0.0
    rng = np.random.default_rng(seed)
    match = 0
    for _ in range(n):
        args = _sample(kind, rng)
        try:
            match += int(fn(*args) == oracle(*args))
        except Exception:
            pass
    return match / n


def train_rl(model, tasks, atom_names, atom_vecs, fams, resolve_fn, embed_fn, K: int = 8, steps: int = 250,
             lr: float = 5e-4, temp: float = 1.2, seed: int = 0):
    """GRPO on VERIFY reward — the model SEARCHES program-space (samples K programs, rewards the ones
    that solve) and reinforces. Unlike imitation, it can DISCOVER a program for a task it has no
    reference for -> the recall->reasoning step."""
    import random
    import torch
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    A = torch.as_tensor(atom_vecs, dtype=torch.float32)
    tlist = [t for t in tasks if t.name in fams]
    rng = random.Random(seed)
    solved_ct = 0
    for _ in range(steps):
        t = rng.choice(tlist)
        goal = torch.as_tensor(np.asarray(list(embed_fn({"q": t.text}).values())[0], dtype=np.float32))
        samples = [model.sample(goal, A, atom_names, temp=temp) for _ in range(K)]
        rewards = []
        for pipe, _ in samples:
            try:
                code = realize_program(t.name, fams[t.name][0], pipe)
            except (ValueError, KeyError):
                rewards.append(0.0); continue
            rewards.append(fast_reward(code, resolve_fn(atoms_of(pipe)), t.name))
        R = torch.tensor(rewards)
        solved_ct += int(R.max() > 0)
        adv = R - R.mean()
        if float(adv.abs().sum()) < 1e-6:                    # all-same reward -> no gradient signal
            continue
        logps = torch.stack([lp for _, lp in samples])
        loss = -(adv * logps).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# STaR / expert-iteration — SEARCH + verify + imitate-your-own-successes (the STABLE alternative to GRPO).
# GRPO reinforced policy SAMPLES (high variance — the wanderer), and pure sampling can't even EXPLORE an
# unseen family: given a novel goal the warmstart is confidently wrong (all mass on the wrong atoms), so a
# 3-step exact program is ~1e-5 to stumble on. The action space is tiny, so the search is a BOUNDED
# ENUMERATION of valid pipelines, each VERIFIED against the family's oracle I/O (never the reference
# PROGRAM) — the DreamCoder "wake". The net then AMORTIZES it: SFT on the discovered programs so a plain
# decode jumps straight there (its job at inference). Deterministic search -> low variance across seeds.
# A length curriculum (short programs before long) orders the SFT. Enumeration is complete here, so it
# subsumes hindsight relabeling; the net-guided ordering + a verify budget are the levers when the space
# grows past enumeration (noted, not needed for the proof).
# ═══════════════════════════════════════════════════════════════════════════════

def _verify_pipe(pipe, fam_name, fams, resolve_fn, n: int = 32, seed: int = 0) -> float:
    """Realize a pipeline as a solution to `fam_name` and score agreement with its oracle I/O. Kind
    mismatch / no-REDUCE / wrong atom -> code errors -> 0.0. The verifier uses the task's I/O spec
    (legit), never the reference PROGRAM."""
    if fam_name not in fams:
        return 0.0
    try:
        code = realize_program(fam_name, fams[fam_name][0], pipe)
    except (ValueError, KeyError):
        return 0.0
    return fast_reward(code, resolve_fn(atoms_of(pipe)), fam_name, n=n, seed=seed)


def _is_general(pipe, fam_name, fams, resolve_fn, n: int = 32) -> bool:
    """Keep-gate = fully general on TWO disjoint random input sets (GRR-1's fuzz-generality bar, so the
    search consolidates only correct programs, never a benchmark-overfit)."""
    return (_verify_pipe(pipe, fam_name, fams, resolve_fn, n, seed=0) >= 1.0
            and _verify_pipe(pipe, fam_name, fams, resolve_fn, n, seed=7) >= 1.0)


def _valid_ops_for_kind(kind: str):
    """Prune the enumeration by input arity (the realizer would reject the rest anyway)."""
    if kind == "pairs":
        return ["MAP2"]
    if kind == "coins":
        return ["KEEP_MAKEABLE"]
    return ["FILTER", "MAP"]                                    # list / arrays -> unary transforms


def _enumerate_general(fam, fams, resolve_fn, atom_names, max_transforms=2, n_verify=32, cache=None):
    """The SEARCH: enumerate structurally-valid pipelines (<= max_transforms transform ops, each an
    (op, atom) pair, then a REDUCE(agg)) and keep the ones VERIFIED fully-general for `fam`. MDL: stop at
    the SHORTEST solving length and dedup by realized code — a longer program with a dead/no-op op (or one
    that's only-correct on weak inputs, e.g. digit_sum acting as identity on single-digit values) is
    rejected in favour of the minimal program. Without this the SFT target set is ambiguous AND can
    contain weak-input-only variants that pass search but fail the eval fuzz -> seed-dependent instability.
    Cached per family (net/seed-independent -> deterministic, hence stable)."""
    import itertools
    if cache is not None and fam in cache:
        return cache[fam]
    kind = fams[fam][0]
    ops = _valid_ops_for_kind(kind)
    slot = [(op, a) for op in ops for a in atom_names]
    found: list = []
    for n_t in range(max_transforms + 1):                      # ascending length -> first hit == MDL-minimal
        seen, level = set(), []
        for combo in itertools.product(slot, repeat=n_t):
            for agg in AGGS:
                pipe = [Op(op, a) for (op, a) in combo] + [Op("REDUCE", agg)]
                if not _is_general(pipe, fam, fams, resolve_fn, n_verify):
                    continue
                try:
                    code = realize_program(fam, kind, pipe)    # dedup truly-equivalent realizations
                except (ValueError, KeyError):
                    continue
                if code not in seen:
                    seen.add(code); level.append(pipe)
        if level:                                              # shortest length that solves -> take it, stop
            found = level
            break
    if cache is not None:
        cache[fam] = found
    return found


def _score_pipe(model, goal_t, A, atom_idx, pipe) -> float:
    """Net log-prob of a pipeline (teacher-forced scoring, no grad) — CHEAP (a few GRU steps), unlike a
    verify (exec + oracle calls). The guided search's ordering key."""
    import torch
    with torch.no_grad():
        return -float(model.loss(goal_t, A, program_to_steps(pipe, atom_idx)))


def _default_is_general(fams, resolve_fn, n_verify):
    """The hand-family verifier as a (pipe, fam) closure — the pluggable-verifier default. Factory
    domains (algo_dsl_gen) pass their own closure instead; the search machinery is domain-agnostic."""
    return lambda pipe, fam: _is_general(pipe, fam, fams, resolve_fn, n_verify)


def _guided_search(model, goal_vec, fam, fams, resolve_fn, atom_names, atom_vecs, max_transforms=2,
                   n_verify=32, budget=None, is_general=None):
    """Net-GUIDED search: the same ascending-length levels as _enumerate_general (MDL-minimality
    preserved), but WITHIN a level candidates are verified in the net's log-prob order, stopping at the
    first general hit (or after `budget` verifies). Returns (found, verifies_used). As the net
    consolidates, the true program ranks earlier -> verifies-to-solve DROPS: the measured amortization
    that lets search scale past brute force as atoms/families grow. EXHAUSTIVE per level — complete but
    only tractable to depth ~2-3; _beam_search is the deep-program engine."""
    import itertools
    import torch
    kind = fams[fam][0]
    ops = _valid_ops_for_kind(kind)
    slot = [(op, a) for op in ops for a in atom_names]
    atom_idx = {a: i for i, a in enumerate(atom_names)}
    g_t = torch.as_tensor(np.asarray(goal_vec, dtype=np.float32))
    A = torch.as_tensor(atom_vecs, dtype=torch.float32)
    chk = is_general or _default_is_general(fams, resolve_fn, n_verify)
    used = 0
    for n_t in range(max_transforms + 1):
        cands = [[Op(op, a) for (op, a) in combo] + [Op("REDUCE", agg)]
                 for combo in itertools.product(slot, repeat=n_t) for agg in AGGS]
        cands.sort(key=lambda p: -_score_pipe(model, g_t, A, atom_idx, p))
        for pipe in cands:
            if budget is not None and used >= budget:
                return [], used
            used += 1
            if chk(pipe, fam):
                return [pipe], used                       # first general hit at the minimal length
    return [], used


def _beam_search(model, goal_vec, fam, fams, resolve_fn, atom_names, atom_vecs, max_transforms=6,
                 n_verify=32, budget=None, B=12, is_general=None):
    """Net-BEAM search for DEEP programs (exhaustive levels die combinatorially: depth 6 over 9 atoms
    x 2 ops ~ 34M candidates). Depth-ascending like the exhaustive search (so the first general hit is
    still the shallowest = MDL-minimal within the explored set): at each depth, (1) TERMINATE the top-B
    prefixes with every REDUCE(agg), verify in the net's score order; (2) EXTEND each prefix by every
    (op, atom), keep the top-B partials by teacher-forced log-prob. Completeness is traded for the
    net's prior — the curriculum makes that prior real before deep families are attempted.
    Returns (found, verifies_used)."""
    import torch
    kind = fams[fam][0]
    ops = _valid_ops_for_kind(kind)
    slot = [(op, a) for op in ops for a in atom_names]
    atom_idx = {a: i for i, a in enumerate(atom_names)}
    g_t = torch.as_tensor(np.asarray(goal_vec, dtype=np.float32))
    A = torch.as_tensor(atom_vecs, dtype=torch.float32)
    chk = is_general or _default_is_general(fams, resolve_fn, n_verify)

    def score(steps):
        if not steps:
            return 0.0
        with torch.no_grad():                                # scoring only — no autograd graphs
            return -float(model.loss(g_t, A, steps))

    used = 0
    prefixes = [[]]
    for _depth in range(max_transforms + 1):
        terms = [p + [Op("REDUCE", agg)] for p in prefixes for agg in AGGS]
        terms.sort(key=lambda p: -_score_pipe(model, g_t, A, atom_idx, p))
        for pipe in terms:
            if budget is not None and used >= budget:
                return [], used
            used += 1
            if chk(pipe, fam):
                return [pipe], used                       # shallowest general hit in the beam
        ext = [p + [Op(op, a)] for p in prefixes for (op, a) in slot]
        ext.sort(key=lambda p: -score(program_to_steps(p, atom_idx)))
        prefixes = ext[:B]
    return [], used


def solve_with_search(model, task, fams, atom_names, atom_vecs, resolve_fn, embed_fn, atom_idx,
                      opt=None, sft_steps=150, budget=None, max_transforms=2, n_verify=32,
                      lr=5e-4, seed=0, is_general=None, beam=0):
    """The TRM inference primitive — the design's retrieve -> reason -> VERIFY -> update -> retry-until-
    solve, embodied: (1) DECODE (recall) + verify-general; (2) on fail, net-GUIDED SEARCH under a verify
    budget (exhaustive-guided by default; `beam=B` switches to beam search for deep programs);
    (3) CONSOLIDATE the discovery (SFT) so the next decode on this family jumps straight there.
    `is_general(pipe, fam)` plugs a domain verifier (factory families). Returns
    dict(solved, pipe, via='decode'|'search'|None, verifies)."""
    import torch
    gv = np.asarray(list(embed_fn({"q": task.text}).values())[0], dtype=np.float32)
    A = torch.as_tensor(atom_vecs, dtype=torch.float32)
    chk = is_general or _default_is_general(fams, resolve_fn, n_verify)
    pipe = model.decode(torch.as_tensor(gv), A, atom_names, max_len=max_transforms + 2)
    if chk(pipe, task.name):
        return dict(solved=True, pipe=pipe, via="decode", verifies=1)
    search = _beam_search if beam else _guided_search
    kw = dict(max_transforms=max_transforms, n_verify=n_verify, budget=budget, is_general=is_general)
    if beam:
        kw["B"] = beam
    found, used = search(model, gv, task.name, fams, resolve_fn, atom_names, atom_vecs, **kw)
    if not found:
        return dict(solved=False, pipe=None, via=None, verifies=used + 1)
    if opt is None:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    _sft_steps(model, opt, [(gv, program_to_steps(p, atom_idx)) for p in found], A, sft_steps, seed=seed)
    return dict(solved=True, pipe=found[0], via="search", verifies=used + 1)


def train_star(model, tasks, atom_names, atom_vecs, fams, resolve_fn, embed_fn, atom_idx,
               rounds=4, sft_steps=400, lr=5e-4, n_verify=32, max_transforms=2, curriculum=True,
               seed=0, cache=None, log=False):
    """Expert iteration: each round SEARCH (enumerate + verify) each active family for general programs,
    then SFT the net on (task goal -> discovered program). The curriculum unlocks longer-program families
    as rounds progress. Returns (model, history=[(round, families_discovered, n_active, pool_size)])."""
    import torch
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    A = torch.as_tensor(atom_vecs, dtype=torch.float32)
    cache = {} if cache is None else cache
    by_fam, goal_of = {}, {}
    for t in tasks:
        by_fam.setdefault(t.name, []).append(t)
        goal_of[id(t)] = np.asarray(list(embed_fn({"q": t.text}).values())[0], dtype=np.float32)
    maxlen = max((_family_len(f) for f in by_fam), default=3)
    hist = []
    for r in range(rounds):
        Lmax = min(maxlen, 2 + r) if curriculum else maxlen    # round0 -> len<=2, then unlock len 3...
        active = [f for f in by_fam if _family_len(f) <= Lmax] or list(by_fam)
        pool, discovered = [], 0
        for f in active:
            found = _enumerate_general(f, fams, resolve_fn, atom_names, max_transforms, n_verify, cache)
            if not found:
                continue
            discovered += 1
            for t in by_fam[f]:
                for prog in found:
                    pool.append((goal_of[id(t)], program_to_steps(prog, atom_idx)))
        _sft_steps(model, opt, pool, A, sft_steps, seed=seed + r)
        hist.append((r, discovered, len(active), len(pool)))
        if log:
            print(f"    STaR round {r}: search-discovered {discovered}/{len(active)} families | "
                  f"pool={len(pool)} traces", flush=True)
    return model, hist


def solve_dsl(model, task, input_kind, atom_names, atom_vecs, embed_fn, graph_path):
    """TRM decodes a program -> realize -> resolve deps through the graph -> fuzz-verify."""
    import torch
    from graph_core import MemoryGraph
    from v5.runtime.algo_compose_tasks import ALL_ATOMS
    from v5.runtime.algo_graph_mg import MGRetriever
    from v5.runtime.algo_quality import fuzz
    gv = torch.as_tensor(np.asarray(list(embed_fn({"q": task.text}).values())[0], dtype=np.float32))
    pipe = model.decode(gv, torch.as_tensor(atom_vecs, dtype=torch.float32), atom_names)
    try:
        code = realize_program(task.name, input_kind, pipe)
    except (ValueError, KeyError):
        return False, pipe
    retr = MGRetriever(MemoryGraph.load_json(graph_path), embed_fn)
    used = atoms_of(pipe)
    deps = retr.resolve_deps(used) if any(a in ALL_ATOMS or a in atom_names for a in used) else ""
    passed, total = fuzz(code, task.name, deps, n=40)
    return bool(total and passed == total), pipe


def _mpnet_setup(graph_path, hard):
    """(embed, atom_names, atom_idx, atom_vecs, fams, traces_by_fam) with REAL mpnet — shared by
    train_dsl_grr6 and the compo-gen eval."""
    from pathlib import Path
    from graph_core import MemoryGraph
    from v5.memory.store import make_mpnet_embedder
    from v5.runtime.algo_compose_tasks import gen_compose_tasks, seed_atom_graph
    from v5.runtime.algo_graph_mg import _fn_name
    if not Path(graph_path).exists():
        seed_atom_graph(graph_path, hard=hard)
    embed = make_mpnet_embedder()
    g = MemoryGraph.load_json(graph_path)
    impls = [(nid, n) for nid, n in g.nodes.items()
             if n.node_type == "implementation" and n.metadata.get("code")]
    atom_names = [_fn_name(n.metadata["code"]) or nid[len("impl_"):] for nid, n in impls]
    atom_idx = {a: i for i, a in enumerate(atom_names)}
    vecs = embed({nid: n.text for nid, n in impls})
    atom_vecs = np.asarray([vecs[nid] for nid, _ in impls], dtype=np.float32)
    fams = {f: (k, p) for f, (k, p) in _PROGRAMS.items() if atoms_of(p) <= set(atom_names)}
    ts = list(gen_compose_tasks(300, seed=0)) + list(gen_compose_tasks(300, seed=0, hard=hard))
    by_fam = {}
    for t in ts:
        if t.name in fams:
            by_fam.setdefault(t.name, []).append(
                (list(embed({"q": t.text}).values())[0], program_to_steps(fams[t.name][1], atom_idx)))
    return embed, atom_names, atom_idx, atom_vecs, fams, by_fam


def compo_gen_eval(graph_path: str, hard: bool = True, steps: int = 2500):
    """The REAL reasoning test (molab, real mpnet): leave-one-family-out. Train on the other families,
    test on the UNSEEN family — can the reasoner compose known ops in a novel combination? >0 here =
    systematic generalization (reasoning); 0 = recall. mpnet embeddings share sub-structure, so this is
    the fair test the synthetic embed couldn't be."""
    import tempfile
    from pathlib import Path
    from v5.runtime.algo_compose_tasks import gen_compose_tasks
    embed, atom_names, atom_idx, atom_vecs, fams, by_fam = _mpnet_setup(graph_path, hard)
    print(f"compo_gen (leave-one-family-out, real mpnet): {graph_path} | {len(fams)} families", flush=True)
    tot_solved = tot = 0
    for holdout in fams:
        traces = [tr for f in by_fam for tr in by_fam[f] if f != holdout]
        model = train_decoder(traces, atom_vecs, atom_names, steps=steps, seed=1)
        held = [t for t in (list(gen_compose_tasks(60, seed=999)) + list(gen_compose_tasks(60, seed=999, hard=hard)))
                if t.name == holdout][:12]
        solved = sum(int(solve_dsl(model, t, fams[t.name][0], atom_names, atom_vecs, embed, graph_path)[0])
                     for t in held)
        tot_solved += solved; tot += len(held)
        print(f"  holdout {holdout:22s} solve={solved}/{len(held)} ({solved/max(1,len(held)):.0%}) "
              f"(unseen family)", flush=True)
    print(f"  OVERALL compositional generalization: {tot_solved}/{tot} ({tot_solved/max(1,tot):.0%}) "
          f"-> {'REASONING (composes novel)' if tot_solved/max(1,tot) > 0.3 else 'RECALL (needs RL/curriculum)'}",
          flush=True)


def compo_gen_star(graph_path: str, hard: bool = True, warmstart_steps: int = 2000, rounds: int = 4,
                   seeds=(1, 2, 3)):
    """Leave-one-family-out WITH test-time search — the honest 2-number reasoning test (real mpnet).
    Per holdout family: (a) warmstart-imitate the OTHER families -> ZERO-SHOT decode holdout = the RECALL
    number (what compo_gen_eval reports); (b) STaR-adapt on holdout TASKS (enumerate + verify oracle I/O,
    the reference PROGRAM is never used) -> WITH-SEARCH decode = the REASONING-via-search number. Runs
    several seeds and reports the SPREAD, because GRPO's failure mode was high variance (the wanderer)."""
    from graph_core import MemoryGraph
    from v5.runtime.algo_compose_tasks import gen_compose_tasks
    from v5.runtime.algo_graph_mg import MGRetriever
    embed, atom_names, atom_idx, atom_vecs, fams, by_fam = _mpnet_setup(graph_path, hard)
    retr = MGRetriever(MemoryGraph.load_json(graph_path), embed)
    resolve_fn = lambda atoms: retr.resolve_deps(atoms) if atoms else ""
    search_cache: dict = {}                                     # enumeration is net/seed-independent -> share

    def _tasks_of(fam, seed, n):
        ts = list(gen_compose_tasks(80, seed=seed)) + list(gen_compose_tasks(80, seed=seed, hard=hard))
        return [t for t in ts if t.name == fam][:n]

    print(f"compo_gen_STaR (leave-one-out + test-time search, real mpnet): {graph_path} | "
          f"{len(fams)} families | seeds={list(seeds)}", flush=True)
    zs_all, ws_all = [], []
    for holdout in fams:
        zs_runs, ws_runs = [], []
        for sd in seeds:
            traces = [tr for f in by_fam if f != holdout for tr in by_fam[f]]
            model = train_decoder(traces, atom_vecs, atom_names, steps=warmstart_steps, seed=sd)
            held = _tasks_of(holdout, seed=777, n=12)              # eval instances (disjoint from adapt)
            kind = fams[holdout][0]
            zs = sum(int(solve_dsl(model, t, kind, atom_names, atom_vecs, embed, graph_path)[0]) for t in held)
            adapt = _tasks_of(holdout, seed=13, n=16)              # search on THESE (no reference program)
            model, _ = train_star(model, adapt, atom_names, atom_vecs, fams, resolve_fn, embed, atom_idx,
                                  rounds=rounds, seed=sd, cache=search_cache)
            ws = sum(int(solve_dsl(model, t, kind, atom_names, atom_vecs, embed, graph_path)[0]) for t in held)
            zs_runs.append(zs / max(1, len(held))); ws_runs.append(ws / max(1, len(held)))
        zsm, wsm = float(np.mean(zs_runs)), float(np.mean(ws_runs))
        zs_all.append(zsm); ws_all.append(wsm)
        print(f"  holdout {holdout:22s} zero-shot={zsm:.0%}  ->  with-search={wsm:.0%} "
              f"[min {min(ws_runs):.0%}, max {max(ws_runs):.0%}]  (unseen family)", flush=True)
    zsm, wsm = float(np.mean(zs_all)), float(np.mean(ws_all))
    print(f"  OVERALL  zero-shot(recall)={zsm:.0%}  ->  with-search(reasoning)={wsm:.0%}", flush=True)
    print(f"  verdict: {'STaR search RECOVERS the unseen composition' if wsm - zsm > 0.3 else 'no lift'} "
          f"(GRPO was unstable; STaR keeps only verified-general programs)", flush=True)


def train_dsl_grr6(graph_path: str, hard: bool = True, steps: int = 3000, d: int = 64, seed: int = 0,
                   out: str = "artifacts/grr6_dsl.pt"):
    """molab: train the DSL program decoder with REAL mpnet embeddings on the graph's atoms. Imitates
    the reference programs; eval held-out = decode->realize->verify solve over the covered families."""
    from pathlib import Path
    import torch
    from graph_core import MemoryGraph
    from v5.memory.store import make_mpnet_embedder
    from v5.runtime.algo_compose_tasks import gen_compose_tasks, seed_atom_graph
    from v5.runtime.algo_graph_mg import _fn_name
    if not Path(graph_path).exists():
        seed_atom_graph(graph_path, hard=hard)
    embed = make_mpnet_embedder()
    g = MemoryGraph.load_json(graph_path)
    impls = [(nid, n) for nid, n in g.nodes.items()
             if n.node_type == "implementation" and n.metadata.get("code")]
    atom_names = [_fn_name(n.metadata["code"]) or nid[len("impl_"):] for nid, n in impls]
    atom_idx = {a: i for i, a in enumerate(atom_names)}
    vecs = embed({nid: n.text for nid, n in impls})
    atom_vecs = np.asarray([vecs[nid] for nid, _ in impls], dtype=np.float32)
    fams = {f: (k, p) for f, (k, p) in _PROGRAMS.items() if atoms_of(p) <= set(atom_names)}
    ts = list(gen_compose_tasks(300, seed=0)) + list(gen_compose_tasks(300, seed=0, hard=hard))
    traces = [(list(embed({"q": t.text}).values())[0], program_to_steps(fams[t.name][1], atom_idx))
              for t in ts if t.name in fams]
    print(f"grr6 DSL train: {graph_path} | {len(atom_names)} atoms | {len(fams)} families | "
          f"{len(traces)} traces | steps={steps}", flush=True)
    model = train_decoder(traces, atom_vecs, atom_names, d=d, steps=steps, seed=seed)
    held = list(gen_compose_tasks(60, seed=999)) + list(gen_compose_tasks(60, seed=999, hard=hard))
    held = [t for t in held if t.name in fams]
    solved = sum(int(solve_dsl(model, t, fams[t.name][0], atom_names, atom_vecs, embed, graph_path)[0])
                 for t in held)
    print(f"  held-out ({len(held)} tasks, {len(fams)} families): program decode->realize->verify "
          f"solve={solved/max(1,len(held)):.0%}", flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "atom_names": atom_names, "atom_vecs": atom_vecs, "d": d,
                "fams": {f: k for f, (k, _p) in fams.items()}}, out)
    print(f"  DSL reasoner -> {out}  ({sum(p.numel() for p in model.parameters())} params)", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST — the reasoner DECODES a correct program per family (a real op+atom sequence) + solves.
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    import json
    import tempfile
    from pathlib import Path
    from v5.runtime.algo_compose_tasks import ALL_ATOMS, gen_compose_tasks
    print("algo_dsl_trm --selftest: TRM autoregressively decodes a DSL PROGRAM per family + solves\n")

    atom_names = ["is_prime", "digit_sum", "build_adj", "dijkstra", "edit_distance", "lcs_length",
                  "coin_change", "lis_length"]
    atom_idx = {a: i for i, a in enumerate(atom_names)}
    rng = np.random.default_rng(0)
    d_in = 64
    atom_vecs = rng.standard_normal((len(atom_names), d_in)).astype("float32")

    # families we cover with the DSL (need their atoms present)
    fams = {f: (k, p) for f, (k, p) in _PROGRAMS.items()
            if atoms_of(p) <= set(atom_names)}
    # specific before general: "increasing" precedes "subsequence" (max_lis's text contains
    # "longest-increasing-SUBSEQUENCE" — the general keyword would collapse both fams to one embedding)
    kw = [("largest digit-sum", "max_prime_digitsum"), ("prime", "sum_digitsum_primes"),
          ("edit", "sum_edit_distance"), ("increasing", "max_lis"),
          ("subsequence", "sum_lcs"), ("coin", "count_makeable")]
    fam_base = {f: rng.standard_normal(d_in).astype("float32") for f in fams}

    def embed(d):
        out = {}
        for k, text in d.items():
            f = next((f for w, f in kw if w in text.lower()), None)
            out[k] = ((fam_base[f] if f in fam_base else np.zeros(d_in, "float32"))
                      + 0.3 * rng.standard_normal(d_in)).astype("float32")
        return out

    # traces: (goal_vec, target steps) from generated tasks + the reference programs
    def make_traces(seed, n):
        ts = list(gen_compose_tasks(n, seed=seed)) + list(gen_compose_tasks(n, seed=seed, hard=True))
        out = []
        for t in ts:
            if t.name not in fams:
                continue
            gv = list(embed({"q": t.text}).values())[0]
            out.append((gv, program_to_steps(fams[t.name][1], atom_idx)))
        return out

    with tempfile.TemporaryDirectory() as td:
        gp = str(Path(td) / "g.json")
        nodes = [{"id": "concept_algorithms", "text": "algorithms", "node_type": "concept"}]
        for a in atom_names:
            nodes.append({"id": f"impl_{a}", "text": ALL_ATOMS[a][0], "node_type": "implementation",
                          "metadata": {"code": ALL_ATOMS[a][1]}})
        Path(gp).write_text(json.dumps({"metadata": {}, "nodes": nodes, "edges": []}))

        model = train_decoder(make_traces(0, 120), atom_vecs, atom_names, steps=2500)

        # [1] decode a full program for a task + it CALLS atoms (multi-step reasoning, not a pick)
        import torch
        t_ed = next(t for t in gen_compose_tasks(30, seed=1, hard=True) if t.name == "sum_edit_distance")
        pipe = model.decode(torch.as_tensor(np.asarray(list(embed({"q": t_ed.text}).values())[0], dtype=np.float32)),
                            torch.as_tensor(atom_vecs), atom_names)
        assert [o.kind for o in pipe] == ["MAP2", "REDUCE"] and pipe[0].arg == "edit_distance", pipe
        print(f"  [1] decoded program for sum_edit_distance: "
              f"{[(o.kind, o.arg) for o in pipe]} (a real op+atom SEQUENCE) -> PASS")

        # [2] held-out: decode -> realize -> solve, across families
        held = list(gen_compose_tasks(60, seed=999)) + list(gen_compose_tasks(60, seed=999, hard=True))
        held = [t for t in held if t.name in fams]
        solved = sum(int(solve_dsl(model, t, fams[t.name][0], atom_names, atom_vecs, embed, gp)[0])
                     for t in held)
        acc = solved / len(held)
        assert acc >= 0.75, acc
        print(f"  [2] held-out ({len(held)} tasks, {len(fams)} families): decode->realize->verify "
              f"solve={acc:.0%} -> PASS")

        # [3] a program is genuinely structured: >=2 decode steps, right ops+atoms+agg (not a lookup)
        lens = [len(model.decode(torch.as_tensor(np.asarray(list(embed({'q': t.text}).values())[0], dtype=np.float32)),
                                 torch.as_tensor(atom_vecs), atom_names)) for t in held[:12]]
        assert max(lens) >= 3, lens            # sum_digitsum_primes = FILTER,MAP,REDUCE = 3 steps
        print(f"  [3] programs are structured: decode length up to {max(lens)} steps (FILTER->MAP->REDUCE), "
              f"each an op+atom choice -> PASS")

        # [4] STaR: bounded SEARCH + verify RECOVERS a HELD-OUT family (compo-gen) + STABLE across seeds
        #     (the GRPO failure mode was high variance). Warmstart imitates the OTHER 5 families; the model
        #     never saw max_prime_digitsum's program (and pure policy-sampling never explores it). Zero-shot
        #     decode = recall; STaR-adapt (enumerate valid pipelines -> verify oracle I/O -> keep general ->
        #     SFT) discovers FILTER(is_prime)+MAP(digit_sum)+REDUCE(max) and consolidates it.
        resolve_fn = lambda atoms: "\n\n".join(ALL_ATOMS[a][1] for a in atoms if a in ALL_ATOMS)
        holdout = "max_prime_digitsum"
        held = [t for t in (list(gen_compose_tasks(60, seed=777)) + list(gen_compose_tasks(60, seed=777, hard=True)))
                if t.name == holdout][:10]
        adapt = [t for t in (list(gen_compose_tasks(60, seed=13)) + list(gen_compose_tasks(60, seed=13, hard=True)))
                 if t.name == holdout][:12]

        def _wo(sd):                                              # traces for every family EXCEPT holdout
            ts = list(gen_compose_tasks(120, seed=sd)) + list(gen_compose_tasks(120, seed=sd, hard=True))
            return [(list(embed({"q": t.text}).values())[0], program_to_steps(fams[t.name][1], atom_idx))
                    for t in ts if t.name in fams and t.name != holdout]

        zs_runs, ws_runs, cache = [], [], {}
        for sd in (1, 2, 3):
            warm = train_decoder(_wo(sd), atom_vecs, atom_names, steps=1800, seed=sd)
            zs = sum(int(solve_dsl(warm, t, fams[holdout][0], atom_names, atom_vecs, embed, gp)[0]) for t in held)
            star, _ = train_star(warm, adapt, atom_names, atom_vecs, fams, resolve_fn, embed, atom_idx,
                                 rounds=4, sft_steps=300, seed=sd, cache=cache)
            ws = sum(int(solve_dsl(star, t, fams[holdout][0], atom_names, atom_vecs, embed, gp)[0]) for t in held)
            zs_runs.append(zs); ws_runs.append(ws)
        H = len(held)
        assert np.mean(ws_runs) >= 0.7 * H, (zs_runs, ws_runs)          # search reliably solves the unseen family
        assert min(ws_runs) >= max(zs_runs), (zs_runs, ws_runs)         # search never underperforms zero-shot
        assert max(ws_runs) - min(ws_runs) <= 0.3 * H, ws_runs          # STABLE across seeds (unlike GRPO)
        print(f"  [4] held-out '{holdout}': zero-shot {zs_runs} -> with-search {ws_runs} (/{H}, 3 seeds) "
              f"-> STaR recovers the unseen composition + STABLE across seeds -> PASS")

        # [5] GUIDED search + solve_with_search (the until-solve inference primitive): verifies-to-solve
        #     DROPS after consolidation (net orders candidates -> measured amortization, the scale lever),
        #     and the second call on the same family flips via=search -> via=decode.
        torch_, _, PD = _build()
        torch_.manual_seed(0)
        cold = PD(d_in=d_in, d=64)
        t_lis = [t for t in gen_compose_tasks(60, seed=21, hard=True) if t.name == "max_lis"][:2]
        gv_lis = np.asarray(list(embed({"q": t_lis[0].text}).values())[0], dtype=np.float32)
        _, v_cold = _guided_search(cold, gv_lis, "max_lis", fams, resolve_fn, atom_names, atom_vecs)
        r1 = solve_with_search(cold, t_lis[0], fams, atom_names, atom_vecs, resolve_fn, embed, atom_idx)
        assert r1["solved"] and r1["via"] == "search", r1
        _, v_hot = _guided_search(cold, gv_lis, "max_lis", fams, resolve_fn, atom_names, atom_vecs)
        r2 = solve_with_search(cold, t_lis[1], fams, atom_names, atom_vecs, resolve_fn, embed, atom_idx)
        assert r2["solved"] and r2["via"] == "decode", r2
        # floor is 5: the 4 level-0 (REDUCE-only) candidates are always verified first, then the
        # consolidated net must rank the true program ~first within its level
        assert v_hot <= 7 and v_hot < v_cold, (v_cold, v_hot)
        print(f"  [5] guided search: verifies-to-solve {v_cold} (cold net) -> {v_hot} (consolidated); "
              f"solve_with_search via=search -> via=decode (until-solve primitive, amortized) -> PASS")

    print("\n  ALGO_DSL_TRM SELFTEST -> PASS  (the reasoner synthesizes programs; STaR turns recall into "
          "search-and-consolidate reasoning)")
    return True


def main():
    ap = argparse.ArgumentParser(description="GRR-6 phase B: TRM autoregressive DSL program decoder.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--train", action="store_true", help="train the DSL decoder with real mpnet (molab)")
    ap.add_argument("--compo-gen", action="store_true", help="leave-one-family-out (imitation): REASON or recall?")
    ap.add_argument("--compo-gen-star", action="store_true",
                    help="leave-one-family-out + STaR test-time search: the 2-number (recall vs reasoning) test")
    ap.add_argument("--graph", default="graphs/algo_reason_hard.json")
    ap.add_argument("--hard", action="store_true")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--rounds", type=int, default=5, help="STaR search/SFT rounds (--compo-gen-star)")
    ap.add_argument("--seeds", type=int, default=3, help="STaR seeds for the variance spread (--compo-gen-star)")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    if a.train:
        train_dsl_grr6(a.graph, hard=a.hard, steps=a.steps)
        return
    if a.compo_gen:
        compo_gen_eval(a.graph, hard=a.hard, steps=a.steps)
        return
    if a.compo_gen_star:
        compo_gen_star(a.graph, hard=a.hard, rounds=a.rounds, seeds=tuple(range(1, a.seeds + 1)))
        return
    ap.print_help()


if __name__ == "__main__":
    main()
