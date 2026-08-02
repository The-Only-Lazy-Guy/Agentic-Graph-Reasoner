"""algo_grr_editcur — a curriculum that teaches a tiny policy to EDIT the graph. No LM anywhere.

THE TEACHER IS EXECUTION, NOT A MODEL. A seed atom's code literally calls the atoms it depends on,
so Python's own AST says which `depend` edges are true, and running the realized closure says whether
the graph is currently good enough to compute the answer. That makes labels FREE and UNLIMITED:
corrupt a real graph in a known way, and the correction is known by construction. This matters
because every learned component in this repo so far died of data starvation (a ranker MLP on 34
pairs, a recall head on 28 examples, both beaten by their own frozen prior). A curriculum does not
have that failure mode -- it can generate as many labelled edits as the optimiser wants.

THE STUDENT NEVER SEES THE TEACHER'S CUE. The obvious feature -- "does node i's code contain node j's
name" -- solves the seed-graph levels outright and transfers NOTHING, because a session graph of
prose spans has no call sites. That is exactly the realizable-expert trap: an expert reading state
the student cannot see leaves an irreducible floor that looks like underfitting. So call-mention is
used ONLY to mint labels, and the policy sees only features that exist in ANY graph: token overlap,
co-firing, degree, common neighbours, path distance, novelty. Nothing here reads code as code.

CURRICULUM (each level generates unlimited instances from a REAL graph)
  L0 RESTORE : delete a true depend edge -> policy must ADD_EDGE it back. Verified by EXECUTING the
               repaired closure against the intact graph's own output. Cue-rich: an easy start.
  L1 REJECT  : inject a false depend edge -> policy must DROP_EDGE. Verified by closure precision
               against the AST truth (a poison edge drags dead atoms into the realized program).
  L2 MASKED  : L0 with every identifier rewritten to an opaque token. The surface cue is destroyed,
               so only the distributional features survive. This is the level that decides whether
               anything transfers -- passing L0 and failing L2 means the policy learned to read code.
  L3 SESSION : ZERO-SHOT transfer, no retraining. A real GSM8K session graph (prose spans + temporal
               `follows` edges, the shape membrane_session.SessionGraph writes) where the edit is a
               `related` edge from the cue-bearing span to the span holding the answer's digits.
               No AST exists here at all, which is the point.

EXPLORE -- searching for undiscovered actions. A graph of N nodes asserts O(N) edges out of N^2
possible, so almost every pair is unexplored territory. `propose()` samples that territory under
three biases, and the comparison between them IS the experiment:
  random   : uniform over non-edges. The control. If nothing beats this, the bias is decoration.
  hebbian  : pairs that CO-FIRE in the spiking layer but have no edge -- fire together, wire
             together. Structural plasticity read straight off the dynamics.
  novelty  : pairs whose homeostatic theta stayed low, i.e. nodes that rarely won anything and are
             therefore under-explored. Count-based exploration, for free, from a variable the layer
             already maintains.
A proposal is only kept if the verifier accepts it. REJECTED proposals are not discarded -- they are
written back as NEGATIVE edges (`not_depend`), which is the relation class this graph has never had
(measured: its relations are exactly part_of and depend) and precisely what lateral arbitration
needs. Exploration therefore manufactures the negative evidence that inhibition consumes.

    selftest : python -m v5.runtime.algo_grr_editcur --selftest
    train    : python -m v5.runtime.algo_grr_editcur --train
    explore  : python -m v5.runtime.algo_grr_editcur --explore
"""
from __future__ import annotations

import argparse
import ast
import glob
import random
import re
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch                                                              # noqa: E402
import torch.nn as nn                                                     # noqa: E402

from v5.runtime.algo_grr_spike import SpikingGraphLayer                   # noqa: E402

_WORD = re.compile(r"[^a-z0-9]+")


def _tok(s: str) -> set:
    return {t for t in _WORD.split((s or "").lower()) if len(t) > 2}


# ================================================================================================
# a graph the policy can actually edit  (generic over seed-graph atoms and session-graph spans)
# ================================================================================================
class EditGraph:
    """Minimal typed graph: id -> text, plus typed edges. Deliberately NOT AtomGraph/MemoryGraph --
    the curriculum has to run identically over code atoms and prose spans, and the only thing both
    substrates share is `nodes with text` + `typed edges`. Both real graphs are loaded into this."""

    def __init__(self):
        self.text: dict[str, str] = {}
        self.edges: set[tuple] = set()                    # (src, dst, relation)
        self._nbr: dict | None = None                     # invalidate-on-write neighbour cache

    def add(self, nid: str, text: str):
        self.text[nid] = text

    def link(self, a: str, b: str, rel: str = "depend"):
        self.edges.add((a, b, rel))
        self._nbr = None

    def unlink(self, a: str, b: str, rel: str = "depend"):
        self.edges.discard((a, b, rel))
        self._nbr = None

    def nbr_map(self) -> dict:
        """All-relations neighbour sets, built once per edge-set version. neighbours() rescans
        every edge, which is fine for 25 atoms and quadratic-in-disguise at session scale: a
        600-span graph scores ~600 candidates per source, so an uncached lookup turns the feature
        build into hundreds of millions of Python operations."""
        if self._nbr is None:
            m: dict = {n: set() for n in self.text}
            for a, b, _r in self.edges:
                if a in m:
                    m[a].add(b)
                if b in m:
                    m[b].add(a)
            self._nbr = m
        return self._nbr

    def has(self, a: str, b: str, rel: str | None = None) -> bool:
        return any(s == a and d == b and (rel is None or r == rel) for s, d, r in self.edges)

    def neighbours(self, n: str, rel: str | None = None) -> set:
        out = set()
        for s, d, r in self.edges:
            if rel is not None and r != rel:
                continue
            if s == n:
                out.add(d)
            elif d == n:
                out.add(s)
        return out

    def ids(self) -> list:
        return list(self.text)

    def copy(self) -> "EditGraph":
        g = EditGraph()
        g.text = dict(self.text)
        g.edges = set(self.edges)
        return g


def load_code_graph() -> tuple:
    """The real seed graph: 21 executable atoms. Returns (EditGraph, code_by_id, true_deps).
    true_deps comes from the AST -- the TEACHER, never a policy feature."""
    from v5.runtime.algo_grr_poison_test import load_seed
    src = load_seed()
    code = {nid: n.metadata.get("code", "") for nid, n in src.nodes.items()
            if nid.startswith("impl_") and n.metadata.get("code")}
    entry = {nid: src.nodes[nid].metadata.get("entry", nid[5:]) for nid in code}
    by_entry = {e: nid for nid, e in entry.items()}
    true_deps: dict[str, set] = {}
    for nid, c in code.items():
        called = {nd.func.id for nd in ast.walk(ast.parse(c))
                  if isinstance(nd, ast.Call) and isinstance(nd.func, ast.Name)}
        true_deps[nid] = {by_entry[x] for x in called if x in by_entry and by_entry[x] != nid}
    g = EditGraph()
    for nid, c in code.items():
        # node TEXT is the purpose string, never the source -- the policy must not be able to read
        # call sites out of it. The code lives in `code_by_id`, which only the verifier touches.
        g.add(nid, src.nodes[nid].text or entry[nid].replace("_", " "))
    for nid, deps in true_deps.items():
        for d in deps:
            g.link(nid, d, "depend")
    for e in src.edges:                                    # keep the concept hubs as real structure
        if e.relation == "part_of" and e.src in g.text:
            g.add(e.dst, e.dst.replace("concept_", "").replace("_", " "))
            g.link(e.src, e.dst, "part_of")
    return g, code, entry, true_deps


# ================================================================================================
# the verifier — real execution, no model
# ================================================================================================
def _closure(g: EditGraph, root: str, code: dict) -> list:
    order, seen = [], set()

    def visit(n):
        if n in seen or n not in code:
            return
        seen.add(n)
        for d in sorted(x for x in g.neighbours(n, "depend") if g.has(n, x, "depend")):
            visit(d)
        order.append(n)
    visit(root)
    return order


_PROBE_SETS = ((1, 4, 6, 12, 28),                                          # ints
               ([3, 1, 2, 3], [5], [2, 2, 7]),                             # lists
               ("racecar", "abc", "aabb"))                                 # strings


def run_atom(g: EditGraph, root: str, code: dict, entry: dict):
    """EXECUTE the atom under the graph's CURRENT closure. Returns a result tuple, or None if the
    graph is too broken to run (a missing depend edge is a NameError). Ground truth: no hand-written
    tests, no model.

    The probe set is chosen by TRYING each type. The seed graph is not all number theory -- it has
    list atoms (unique/flatten/most_common) and string atoms (reverse_string/char_freq/is_anagram),
    and an int-only probe made every one of them look unrunnable. Measured: that alone dragged the
    L0 execution check to 0.42 while ranking accuracy was 1.00, i.e. the verifier was reporting its
    own type error as a graph defect."""
    try:
        body = "\n\n".join(code[n] for n in _closure(g, root, code))
        ns: dict = {}
        exec(compile(body, "<editcur>", "exec"), ns)                      # noqa: S102 — the verifier
        fn = ns.get(entry[root])
        if not callable(fn):
            return None
    except Exception:                                                     # noqa: BLE001
        return None
    for probes in _PROBE_SETS:
        try:
            return tuple(repr(fn(p)) for p in probes)
        except Exception:                                                 # noqa: BLE001
            continue
    return None


# ================================================================================================
# features — every one computable on prose spans too. NO call-mention, NO code reading.
# ================================================================================================
N_FEAT = 7


def pair_features(g: EditGraph, i: str, j: str, cofire: dict, theta: dict,
                  nbrs: dict | None = None, toks: dict | None = None) -> list:
    ti = toks[i] if toks else _tok(g.text.get(i, ""))
    tj = toks[j] if toks else _tok(g.text.get(j, ""))
    jac = len(ti & tj) / len(ti | tj) if (ti or tj) else 0.0
    nb = nbrs if nbrs is not None else g.nbr_map()
    ni, nj = nb.get(i, set()), nb.get(j, set())
    n_all = max(1, len(g.text))
    common = len(ni & nj) / n_all
    dist = 0.0 if j in ni else (0.5 if (ni & nj) else 1.0)                # 1-hop / 2-hop / far
    return [jac, cofire.get((i, j), 0.0), len(ni) / n_all, len(nj) / n_all,
            common, dist, 0.5 * (theta.get(i, 0.0) + theta.get(j, 0.0))]


def pair_matrix(g: EditGraph, src: str, cands: list, cofire: dict, theta: dict) -> torch.Tensor:
    """Feature matrix for one decision, with the CUE-STRENGTH columns z-scored WITHIN the candidate
    set. This is the fix for a measured failure, not a flourish.

    Trained on raw features, the policy scored 0.88 on masked code graphs but 0.06 on prose spans --
    far BELOW token-overlap-alone (0.57) on the same probes. Cause: in a masked instance every node
    reads `n000 item`, so overlap is constant and carries no signal, and the gradient drives its
    weight toward zero; the policy cannot condition on "is overlap informative in THIS graph"
    because that is a property of the candidate set, not of any pair. Averaging the two regimes, it
    learned to distrust the one feature that transfers.

    Z-scoring within the candidate set makes the question scale-free: a constant column becomes all
    zeros and self-cancels, while a column that actually discriminates keeps its spread. The same
    learned weight then means "trust this cue to the extent it separates the candidates here",
    which is the property that has to survive the jump from code atoms to prose spans."""
    nb = g.nbr_map()
    tk = {src: _tok(g.text.get(src, ""))}
    for c in cands:
        tk[c] = _tok(g.text.get(c, ""))
    X = torch.tensor([pair_features(g, src, c, cofire, theta, nbrs=nb, toks=tk) for c in cands],
                     dtype=torch.float32)
    for col in (0, 1):                                        # token overlap, co-firing
        v = X[:, col]
        sd = float(v.std())
        X[:, col] = (v - v.mean()) / sd if sd > 1e-6 else torch.zeros_like(v)
    return X


def dynamics_stats(g: EditGraph, steps: int = 3, w_comp: float = 0.4,
                   max_probes: int = 120, seed: int = 0) -> tuple:
    """Run the spiking layer with every node used in turn as the query, and record (a) how often
    each PAIR fires together and (b) each node's final theta. Unsupervised: no labels, no model --
    just the layer's own behaviour on the graph it is sitting in. cofire is the hebbian explore
    bias; theta is the novelty bias."""
    ids = g.ids()
    n = len(ids)
    idx = {x: k for k, x in enumerate(ids)}
    adj = torch.zeros(n, n)
    for s, d, _r in g.edges:
        if s in idx and d in idx:
            adj[idx[s], idx[d]] = adj[idx[d], idx[s]] = 1.0
    toks = {x: _tok(g.text.get(x, "")) for x in ids}
    # The rival mask is a function of the graph, not of the probe, so it is built ONCE. Rebuilding
    # it inside the step loop is an [N,N] allocation per step per probe, which is what made this
    # unusable at session scale (N ~ 600 spans).
    comp = torch.ones(n, n) - adj - torch.eye(n)
    probes = ids
    if n > max_probes:                                  # subsample: co-fire is a rate, not a census
        probes = random.Random(seed).sample(ids, max_probes)
    cofire: dict = {}
    theta_sum = torch.zeros(n)
    for q in probes:
        tq = toks[q]
        drive = torch.tensor([len(tq & toks[x]) / max(1, len(tq | toks[x])) for x in ids],
                             dtype=torch.float32)
        layer = SpikingGraphLayer(theta0=0.12, tau=0.85, alpha=0.5, beta=0.02,
                                  w_dep=0.0, w_comp=w_comp)
        layer.reset_state(n)
        fired = torch.zeros(n)
        for _ in range(steps):
            y = layer.step(drive, adj, comp)
            fired = torch.clamp(fired + y, max=1.0)
        theta_sum += layer.theta
        lit = [ids[k] for k in range(n) if fired[k] > 0]
        for a in lit:
            for b in lit:
                if a != b:
                    cofire[(a, b)] = cofire.get((a, b), 0.0) + 1.0 / len(probes)
    theta = {ids[k]: float(theta_sum[k] / max(1, len(probes))) for k in range(n)}
    return cofire, theta


# ================================================================================================
# the policy — ~80 params, deliberately tiny
# ================================================================================================
class EditPolicy(nn.Module):
    def __init__(self, hidden: int = 8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N_FEAT, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def score_pairs(self, g, src, cands, cofire, theta):
        with torch.no_grad():
            return self(pair_matrix(g, src, cands, cofire, theta))


# ================================================================================================
# curriculum levels
# ================================================================================================
def _mask_graph(g: EditGraph) -> EditGraph:
    """L2: rewrite every node's text to an opaque token. Any policy that was really reading names
    collapses to chance here; a policy using structure/co-firing does not."""
    m = g.copy()
    for k, nid in enumerate(sorted(m.text)):
        m.text[nid] = f"n{k:03d} item"
    return m


def level_instances(level: int, g: EditGraph, true_deps: dict, rng: random.Random, n: int):
    """Yield (graph, source_node, candidates, gold, action). Unlimited by construction."""
    base = _mask_graph(g) if level == 2 else g
    srcs = [s for s, d in true_deps.items() if d]
    out = []
    for _ in range(n):
        s = rng.choice(srcs)
        if level in (0, 2):                                               # RESTORE a deleted edge
            gold = rng.choice(sorted(true_deps[s]))
            gr = base.copy()
            gr.unlink(s, gold, "depend")
            cands = [c for c in gr.ids() if c != s and not gr.has(s, c, "depend")]
            if gold not in cands:
                continue
            out.append((gr, s, cands, gold, "ADD_EDGE"))
        else:                                                             # L1: REJECT a poison edge
            gr = base.copy()
            bad = [c for c in gr.ids() if c != s and c not in true_deps[s]
                   and c in true_deps and not gr.has(s, c, "depend")]
            if not bad:
                continue
            poison = rng.choice(bad)
            gr.link(s, poison, "depend")
            cands = sorted(x for x in gr.neighbours(s, "depend") if gr.has(s, x, "depend"))
            out.append((gr, s, cands, poison, "DROP_EDGE"))
    return out


def train_policy(epochs: int = 60, per_epoch: int = 96, seed: int = 0, verbose: bool = True):
    """Train on L0+L1 only. L2 (masked) and L3 (session) are NEVER trained on -- they are the
    generalisation tests."""
    rng = random.Random(seed)
    g, code, entry, true_deps = load_code_graph()
    cofire, theta = dynamics_stats(g)
    pol = EditPolicy()
    opt = torch.optim.Adam(pol.parameters(), lr=5e-3)
    for ep in range(epochs):
        # CURRICULUM SCHEDULE. Measured first: training on L0+L1 alone gives L0 1.00 (chance 0.04)
        # but MASKED-L2 only 0.14 -- the policy had learned to read names, and that is precisely
        # what cannot transfer to a graph of prose spans. So the masked level is a TRAINING level
        # that ramps in, which is what makes this a curriculum rather than two tasks: start on the
        # cue-rich instances to get the action mechanics, then force the same head to solve the
        # same edits with the lexical cue destroyed. The zero-shot session graph stays held out.
        frac = min(0.5, 0.05 + 0.9 * ep / max(1, epochs - 1))
        n_hard = int(per_epoch * frac)
        n_easy = (per_epoch - n_hard) // 2
        insts = (level_instances(0, g, true_deps, rng, n_easy)
                 + level_instances(1, g, true_deps, rng, per_epoch - n_hard - n_easy)
                 + level_instances(2, g, true_deps, rng, n_hard))
        rng.shuffle(insts)
        tot = 0.0
        for gr, s, cands, gold, act in insts:
            if gold not in cands or len(cands) < 2:
                continue
            logits = pol(pair_matrix(gr, s, cands, cofire, theta))
            y = torch.tensor(cands.index(gold))
            # DROP_EDGE is the same ranking problem with the sign flipped: pick the edge that does
            # NOT belong. One head, two actions, so the policy cannot learn them as separate tasks.
            loss = nn.functional.cross_entropy((logits if act == "ADD_EDGE" else -logits)
                                               .unsqueeze(0), y.unsqueeze(0))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss)
        if verbose and (ep + 1) % 20 == 0:
            print(f"    epoch {ep+1:3d}  loss={tot / max(1, len(insts)):.4f}", flush=True)
    return pol, g, code, entry, true_deps, cofire, theta


def eval_level(pol, level, g, true_deps, cofire, theta, code=None, entry=None,
               n: int = 200, seed: int = 99) -> dict:
    rng = random.Random(seed)
    insts = level_instances(level, g, true_deps, rng, n)
    hit = ran = exe = 0
    for gr, s, cands, gold, act in insts:
        if gold not in cands or len(cands) < 2:
            continue
        sc = pol.score_pairs(gr, s, cands, cofire, theta)
        pick = cands[int((sc if act == "ADD_EDGE" else -sc).argmax())]
        ran += 1
        hit += int(pick == gold)
        if act == "ADD_EDGE" and code is not None:                        # EXECUTE the repair
            gr2 = gr.copy(); gr2.link(s, pick, "depend")
            exe += int(run_atom(gr2, s, code, entry) is not None
                       and run_atom(gr2, s, code, entry) == run_atom(g, s, code, entry))
    return {"n": ran, "acc": hit / max(1, ran), "chance": 1.0 / max(1, len(insts[0][2]) if insts else 1),
            "exec_ok": exe / max(1, ran) if code is not None and level != 1 else None}


# ================================================================================================
# EXPLORE — search the undiscovered part of the graph
# ================================================================================================
def propose(g: EditGraph, mode: str, cofire: dict, theta: dict, rng: random.Random,
            k: int = 20, pol=None) -> list:
    """Sample candidate edges from the N^2 - O(N) territory the graph has never asserted."""
    ids = g.ids()
    pool = [(a, b) for a in ids for b in ids
            if a != b and not g.has(a, b) and not g.has(b, a)]
    if not pool:
        return []
    if mode == "random":
        return rng.sample(pool, min(k, len(pool)))
    if mode == "hebbian":                                   # fire together but never wired
        return sorted(pool, key=lambda p: -cofire.get(p, 0.0))[:k]
    if mode == "novelty":                                   # both ends rarely won anything
        return sorted(pool, key=lambda p: theta.get(p[0], 0.0) + theta.get(p[1], 0.0))[:k]
    if mode == "policy":                                    # the trained edit policy proposes
        if pol is None:
            raise ValueError("mode='policy' needs a trained EditPolicy")
        by_src: dict = {}
        for a, b in pool:
            by_src.setdefault(a, []).append(b)
        scored = []
        for a, bs in by_src.items():
            if len(bs) < 2:
                continue
            s = pol.score_pairs(g, a, bs, cofire, theta)
            for j, b in enumerate(bs):
                scored.append((float(s[j]), a, b))
        scored.sort(reverse=True)
        return [(a, b) for _s, a, b in scored[:k]]
    raise ValueError(f"unknown explore mode: {mode}")


def hide_edges(g: EditGraph, true_deps: dict, rng: random.Random, n_hidden: int) -> tuple:
    """Remove n_hidden true depend edges. Exploration only means anything against a graph that is
    actually MISSING something: run against the intact seed graph, the unexplored pool contains no
    true edges by construction and every bias scores exactly 0 -- which is what the first version
    of this experiment measured, and it measured nothing."""
    all_true = [(s, d) for s, ds in true_deps.items() for d in ds]
    hidden = rng.sample(all_true, min(n_hidden, len(all_true)))
    gc = g.copy()
    for a, b in hidden:
        gc.unlink(a, b, "depend")
    return gc, set(hidden)


def explore_round(g: EditGraph, code: dict, entry: dict, hidden: set, mode: str,
                  rng: random.Random, k: int = 20, pol=None) -> dict:
    """Propose k unexplored edges, VERIFY each by execution, keep the winners, and write the losers
    back as `not_depend` -- the negative relation this graph has never had."""
    cofire, theta = dynamics_stats(g)
    cands = propose(g, mode, cofire, theta, rng, k=k, pol=pol)
    kept = neg = 0
    for a, b in cands:
        if a not in code:
            continue
        gold_out = run_atom(g, a, code, entry)
        trial = g.copy(); trial.link(a, b, "depend")
        out = run_atom(trial, a, code, entry)
        # An edge EARNS its place only if it repairs something (the atom could not run before and
        # now can). Merely not-breaking is not evidence -- that is what makes poison edges cheap.
        if out is not None and gold_out is None:
            g.link(a, b, "depend"); kept += 1
        else:
            g.link(a, b, "not_depend"); neg += 1
    hits = sum(1 for p in cands if p in hidden)
    return {"mode": mode, "proposed": len(cands), "true_edges_found": hits,
            "kept": kept, "negatives_written": neg,
            "precision": hits / max(1, len(cands))}


# ================================================================================================
# L3 — zero-shot transfer to a real GSM8K session graph (prose spans, no AST, no LM)
# ================================================================================================
def load_session_graph(n_problems: int = 40, span_words: int = 60) -> tuple:
    """Build the real session-graph SHAPE from cached GSM8K text: prose spans chained by temporal
    `follows` edges, exactly what membrane_session.SessionGraph writes when the KV cache spills.
    Read with pyarrow -- `import datasets` deadlocks after torch in this environment, and no model
    is loaded here at all."""
    import pyarrow as pa
    fs = sorted(glob.glob(str(Path(r"E:\cache\hf\datasets\openai___gsm8k") / "**" / "*.arrow"),
                          recursive=True))
    if not fs:
        raise FileNotFoundError("GSM8K arrow cache not found")
    tbl = pa.ipc.open_stream(pa.memory_map(fs[0], "rb")).read_all()
    qs = [tbl.column("question")[i].as_py() for i in range(min(n_problems, tbl.num_rows))]
    g = EditGraph()
    spans, owner, prev = [], {}, None
    for pi, q in enumerate(qs):
        words = q.split()
        for w0 in range(0, len(words), span_words):
            nid = f"span_{len(spans):03d}"
            g.add(nid, " ".join(words[w0:w0 + span_words]))
            owner[nid] = pi
            if prev is not None:
                g.link(prev, nid, "follows")
            prev = nid
            spans.append(nid)
    return g, qs, owner


def eval_session(pol, n_problems: int = 40, seed: int = 7, span_words: int = 25,
                 verbose: bool = True) -> dict:
    """ZERO-SHOT: the policy trained on code atoms is applied, with no retraining, to prose spans.
    Task: from the span holding a problem's CUE, add one `related` edge to the span holding that
    problem's answer digits. Gold = a different span owned by the same problem. Chance is measured,
    not assumed, and the policy is compared against the token-overlap feature alone -- if it cannot
    beat its own strongest input feature it has learned nothing transferable."""
    rng = random.Random(seed)
    g0, qs, owner = load_session_graph(n_problems, span_words=span_words)
    multi = [pi for pi in set(owner.values()) if sum(1 for v in owner.values() if v == pi) > 1]
    if not multi:
        return {"n": 0, "note": f"no multi-span problems at span_words={span_words}"}
    hit = base_hit = ran = 0
    chances = []
    for pi in multi:
        mine = sorted(s for s, o in owner.items() if o == pi)
        src, gold = mine[0], mine[1]
        # The same RESTORE edit as L0, transplanted to prose: cut the temporal link, then ask the
        # policy to reconnect the two spans of one problem out of the whole session. Without the
        # cut, `gold` is already adjacent and would be filtered out of the candidate set -- the
        # first run of this eval scored 0 instances for exactly that reason.
        g = g0.copy()
        g.unlink(src, gold, "follows")
        g.unlink(gold, src, "follows")
        cands = [c for c in g.ids() if c != src and not g.has(src, c) and not g.has(c, src)]
        if gold not in cands or len(cands) < 2:
            continue
        cofire, theta = dynamics_stats(g)
        sc = pol.score_pairs(g, src, cands, cofire, theta)
        jac = pair_matrix(g, src, cands, cofire, theta)[:, 0]
        ran += 1
        chances.append(1.0 / len(cands))
        hit += int(cands[int(sc.argmax())] == gold)
        base_hit += int(cands[int(jac.argmax())] == gold)
        if verbose and ran == 1:
            print(f"    (probe: {len(cands)} candidate spans, cut the follows edge "
                  f"{src}->{gold})", flush=True)
    if not ran:
        return {"n": 0, "note": "no eligible probes"}
    return {"n": ran, "policy": hit / ran, "token_overlap_only": base_hit / ran,
            "chance": sum(chances) / len(chances)}


# ================================================================================================
# selftest / entry points
# ================================================================================================
def _selftest() -> bool:
    print("algo_grr_editcur --selftest: curriculum machinery (no LM, real execution)\n")
    ok = True

    def chk(t, c, d=""):
        nonlocal ok
        ok &= bool(c)
        print(f"  [{'PASS' if c else 'FAIL'}] {t}{('  - ' + d) if d else ''}")

    g, code, entry, true_deps = load_code_graph()
    chk("[1] real seed graph loaded with AST-derived deps",
        len(code) >= 20 and sum(len(v) for v in true_deps.values()) >= 7,
        f"{len(code)} atoms, {sum(len(v) for v in true_deps.values())} true deps")

    # [2] the verifier must really break when the graph is broken, and really recover when repaired.
    root = "impl_is_perfect"
    good = run_atom(g, root, code, entry)
    broken = g.copy(); broken.unlink(root, "impl_sum_divisors", "depend")
    bad = run_atom(broken, root, code, entry)
    fixed = broken.copy(); fixed.link(root, "impl_sum_divisors", "depend")
    chk("[2] execution verifier: intact runs, cut edge breaks, repair restores",
        good is not None and bad is None and run_atom(fixed, root, code, entry) == good,
        f"intact={good} cut={bad}")

    # [3] the student must never see SOURCE. Note what this does and does not forbid: a purpose
    # string legitimately overlaps its dependency's name ("sum of divisors" contains "divisors"),
    # and that is feature[0] by design -- prose overlap is exactly the signal that also exists
    # between session spans, so it transfers. What must never leak is CODE, whose call sites are a
    # cue with no counterpart in a prose graph. L2 exists to measure what survives when even the
    # lexical signal is destroyed.
    leak = [n for n in g.ids()
            if "def " in g.text.get(n, "") or "return " in g.text.get(n, "")]
    chk("[3] node text carries purpose prose, never source code", not leak, f"leaky={leak[:3]}")

    # [4] masking really destroys the surface cue.
    m = _mask_graph(g)
    chk("[4] L2 masking removes all lexical identity",
        all(re.fullmatch(r"n\d{3} item", t) for t in m.text.values()))

    # [5] dynamics produce a real co-fire signal and real thetas.
    cofire, theta = dynamics_stats(g)
    chk("[5] spiking dynamics yield co-fire + theta signals",
        len(cofire) > 0 and len(theta) == len(g.ids()) and max(theta.values()) > 0,
        f"{len(cofire)} co-firing pairs, theta max {max(theta.values()):.3f}")

    # [6] explore modes must actually differ from each other, else the bias is decoration.
    rng = random.Random(0)
    props = {m_: set(map(tuple, propose(g, m_, cofire, theta, rng, k=15)))
             for m_ in ("random", "hebbian", "novelty")}
    chk("[6] the three explore biases propose different territory",
        props["hebbian"] != props["random"] and props["novelty"] != props["hebbian"],
        f"heb&rand={len(props['hebbian'] & props['random'])}, "
        f"heb&nov={len(props['hebbian'] & props['novelty'])}")

    # [7] curriculum instances are well-formed and unlimited.
    i0 = level_instances(0, g, true_deps, random.Random(1), 30)
    i1 = level_instances(1, g, true_deps, random.Random(1), 30)
    chk("[7] L0/L1 generate valid labelled instances",
        len(i0) > 20 and len(i1) > 20
        and all(gold in cands for _gr, _s, cands, gold, _a in i0 + i1),
        f"L0={len(i0)} L1={len(i1)}")

    print(f"\n  ALGO_GRR_EDITCUR SELFTEST -> {'PASS' if ok else 'FAIL'}")
    return ok


def _train_and_report(epochs: int = 60) -> bool:
    print("algo_grr_editcur --train: learn graph edits from execution, no LM\n")
    pol, g, code, entry, true_deps, cofire, theta = train_policy(epochs=epochs)
    print("\n  level                                    n    acc    chance  exec_ok")
    rows = []
    for lvl, tag in [(0, "L0 RESTORE  (trained)"), (1, "L1 REJECT   (trained)"),
                     (2, "L2 MASKED   (trained: the hard level)")]:
        r = eval_level(pol, lvl, g, true_deps, cofire, theta,
                       code=code if lvl != 1 else None, entry=entry)
        rows.append((tag, r))
        ex = "—" if r["exec_ok"] is None else f"{r['exec_ok']:.2f}"
        print(f"    {tag:<38} {r['n']:>3}   {r['acc']:.2f}   {r['chance']:.2f}     {ex}")
    print("\n  L3 SESSION (ZERO-SHOT — real GSM8K spans, no AST, no retraining)")
    s = eval_session(pol)
    if s.get("n"):
        print(f"    n={s['n']}  policy={s['policy']:.2f}  "
              f"token_overlap_only={s['token_overlap_only']:.2f}  chance={s['chance']:.2f}")
    else:
        print(f"    {s.get('note')}")
    return rows[0][1]["acc"] > rows[0][1]["chance"]


def _explore_report(trials: int = 8, n_hidden: int = 4, k: int = 20, epochs: int = 60) -> bool:
    print("algo_grr_editcur --explore: searching undiscovered edges\n")
    print(f"  Hide {n_hidden} true depend edges, let each bias propose {k} of the unexplored pool,")
    print("  and count how many of the HIDDEN edges it finds. Averaged over "
          f"{trials} independent hidings.\n")
    pol, _g, _c, _e, _td, _cf, _th = train_policy(epochs=epochs, verbose=False)
    g0, code, entry, true_deps = load_code_graph()
    n = len(g0.ids())
    pool_est = n * (n - 1) - 2 * len(g0.edges)
    print(f"  graph: {n} nodes, ~{pool_est} unexplored pairs, {n_hidden} needles\n")
    print("  mode      proposed  hidden_found  recall  precision  verified_keeps  negatives")
    out = {}
    for mode in ("random", "hebbian", "novelty", "policy"):
        agg = {"proposed": 0, "true_edges_found": 0, "kept": 0, "negatives_written": 0}
        for t in range(trials):
            rng = random.Random(1000 + t)
            gc, hidden = hide_edges(g0, true_deps, rng, n_hidden)
            r = explore_round(gc, code, entry, hidden, mode, rng, k=k, pol=pol)
            for key in agg:
                agg[key] += r[key]
        rec = agg["true_edges_found"] / max(1, n_hidden * trials)
        prec = agg["true_edges_found"] / max(1, agg["proposed"])
        out[mode] = rec
        print(f"  {mode:<9} {agg['proposed']:>8}  {agg['true_edges_found']:>12}  {rec:>6.3f}  "
              f"{prec:>9.3f}  {agg['kept']:>14}  {agg['negatives_written']:>9}")
    print("\n  random is the control: a bias that does not beat it is decoration.")
    best = max(out, key=out.get)
    print(f"  best bias: {best} (recall {out[best]:.3f} vs random {out['random']:.3f})")
    return out[best] > out["random"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Graph-edit curriculum (no LM).")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--explore", action="store_true")
    ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    if a.train:
        sys.exit(0 if _train_and_report(a.epochs) else 1)
    if a.explore:
        sys.exit(0 if _explore_report() else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
