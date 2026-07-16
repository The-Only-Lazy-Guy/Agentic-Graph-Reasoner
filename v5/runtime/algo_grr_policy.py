"""algo_grr_policy — the trained TRM retrieval policy (GRR-Tool build B2a).

The membrane's default retrieval is cosine, which is NOT program-conditioned: after selecting
`divisors` for "count prime divisors", re-embedding task+divisors reinforces the divisor cluster and
BURIES the missing complement `is_prime`. That is exactly what a learned policy fixes.

ComplementPolicy: a tiny pointer net that scores atoms conditioned on (task, atoms-selected-so-far),
trained to rank the STILL-MISSING atoms of a composition highest. It amortizes the compositions the
same way the rest of the stack amortizes search: ground-truth compositions supervise it; at inference
it drops into MembraneSolver.policy_fn and retrieves the COMPLEMENT of the partial program.

Embedder is injectable: HashEmbed (dense, deterministic, no deps) for the no-GPU selftest; mpnet on
molab. The LM is untouched — this trains only the tiny retrieval policy (owned, not the compiler).

    selftest (no GPU):  python -m v5.runtime.algo_grr_policy --selftest
"""
from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path

import numpy as np

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from graph_core import MemoryGraph, Node, Edge  # type: ignore  # noqa: E402
from v5.runtime.algo_grr_membrane import (  # noqa: E402
    MembraneSolver, TokenRetriever, make_stub_compiler, _tokens,
)


# ═══════════════════════════════════════════════════════════════════════════════
# HashEmbed — dense, deterministic bag-of-tokens embedding (no deps; mpnet on molab)
# ═══════════════════════════════════════════════════════════════════════════════

class HashEmbed:
    def __init__(self, dim: int = 96):
        self.dim = dim

    def __call__(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for t in _tokens(text):
            v[zlib.crc32(t.encode()) % self.dim] += 1.0
        n = float(np.linalg.norm(v))
        return v / n if n else v


# ═══════════════════════════════════════════════════════════════════════════════
# Ground-truth compositions over the seed graph (task text -> needed atom node ids)
# ═══════════════════════════════════════════════════════════════════════════════

# Labels are minimal SELECTION sets (what must be RETRIEVED); dependencies come free via the graph's
# depend-closure at realize time, so a closure-covered atom (lcm->gcd) is a single selection.
TRAIN_TASKS: list[tuple[str, list[str]]] = [
    # single-atom (recall)
    ("check whether a number is prime", ["impl_is_prime"]),
    ("greatest common divisor of two integers", ["impl_gcd"]),
    ("reverse the digits of an integer", ["impl_reverse_digits"]),
    ("largest sum of a contiguous subarray", ["impl_max_subarray_sum"]),
    ("remove duplicates preserving order", ["impl_unique"]),
    ("frequency of each character in a string", ["impl_char_freq"]),
    ("sum of the digits of an integer", ["impl_digit_sum"]),
    # closure-shortcut (single selection; dep pulled via the graph closure)
    ("least common multiple of two integers", ["impl_lcm"]),
    ("check if two strings are anagrams of each other", ["impl_is_anagram"]),
    ("check whether a string is a palindrome", ["impl_is_palindrome"]),
    ("check whether an integer is a palindrome", ["impl_is_palindrome_number"]),
    ("is n a perfect number", ["impl_is_perfect"]),
    ("the most common element in a list", ["impl_most_common"]),
    # genuine-2 compositions (two INDEPENDENT atoms — the complement-retrieval targets)
    ("count how many divisors of n are prime numbers", ["impl_divisors", "impl_is_prime"]),
    ("check whether the digit reversal of n is prime", ["impl_reverse_digits", "impl_is_prime"]),
    ("check whether the digit sum of n is prime", ["impl_digit_sum", "impl_is_prime"]),
    ("check whether the gcd of two numbers is prime", ["impl_gcd", "impl_is_prime"]),
    ("reversed digits of the largest contiguous subarray sum",
     ["impl_max_subarray_sum", "impl_reverse_digits"]),
    ("count the prime numbers among a list's unique elements", ["impl_unique", "impl_is_prime"]),
]


# ═══════════════════════════════════════════════════════════════════════════════
# ComplementPolicy — tiny pointer net (torch); scores atoms | (task, selected)
# ═══════════════════════════════════════════════════════════════════════════════

def _build():
    import torch
    import torch.nn as nn

    class ComplementPolicy(nn.Module):
        def __init__(self, d_in: int, d: int = 64):
            super().__init__()
            self.task = nn.Linear(d_in, d)
            self.sel = nn.Linear(d_in, d)
            self.atom = nn.Linear(d_in, d)
            self.q = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))

        def forward(self, x, A, sel_vec):
            # x:[d_in]  A:[N,d_in]  sel_vec:[d_in] (mean of selected atoms, zeros if none)
            query = self.q(torch.cat([self.task(x), self.sel(sel_vec)]))   # [d]
            return self.atom(A) @ query                                     # [N] logits
    return torch, nn, ComplementPolicy


def train_policy(graph: MemoryGraph, train_tasks, embed_fn, d: int = 64,
                 steps: int = 800, lr: float = 5e-3, seed: int = 0):
    """Deep-sup over prefixes: given (task, a prefix of the needed atoms), score the STILL-MISSING
    needed atoms high (BCE), everything else low. Teaches complement retrieval. Returns (model, ctx)."""
    torch, nn, ComplementPolicy = _build()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    atom_ids = [nid for nid in graph.nodes if graph.nodes[nid].node_type == "implementation"]
    idx = {nid: i for i, nid in enumerate(atom_ids)}
    A = torch.tensor(np.stack([embed_fn(graph.nodes[nid].text) for nid in atom_ids]))
    d_in = A.shape[1]

    # expand each task into (x, selected-subset, still-missing) over ALL SUBSETS of the needed set
    # (order-invariant: the policy must handle any selected state, not just fixed-order prefixes)
    from itertools import combinations
    samples = []
    for text, needed in train_tasks:
        need = [n for n in needed if n in idx]
        x = torch.tensor(embed_fn(text))
        for r in range(len(need) + 1):
            for sub in combinations(need, r):
                tgt = [idx[n] for n in need if n not in sub]
                sel_vec = A[[idx[n] for n in sub]].mean(0) if sub else torch.zeros(d_in)
                y = torch.zeros(len(atom_ids))
                for t in tgt:
                    y[t] = 1.0
                samples.append((x, sel_vec, y))

    model = ComplementPolicy(d_in, d)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    for step in range(steps):
        x, sel_vec, y = samples[rng.integers(len(samples))]
        logits = model(x, A, sel_vec)
        loss = bce(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
    return model, {"atom_ids": atom_ids, "idx": idx, "A": A, "d_in": d_in}


def make_policy_fn(model, ctx, embed_fn):
    """Wrap a trained ComplementPolicy as a MembraneSolver.policy_fn:
    (task, selected, graph, retriever) -> [(node_id, sigmoid_score), ...] ranked.
    FIXED atom set (ctx) — for a static graph. For the live loop where the graph GROWS, use
    make_graph_policy_fn instead (it re-embeds the current graph so derived atoms are scored)."""
    import torch
    atom_ids, idx, A = ctx["atom_ids"], ctx["idx"], ctx["A"]
    d_in = ctx["d_in"]

    def policy_fn(task, selected, graph, retriever):
        x = torch.tensor(embed_fn(task["text"]))
        sel = [idx[s] for s in selected if s in idx]
        sel_vec = A[sel].mean(0) if sel else torch.zeros(d_in)
        with torch.no_grad():
            scores = torch.sigmoid(model(x, A, sel_vec)).tolist()
        ranked = sorted(zip(atom_ids, scores), key=lambda z: -z[1])
        return ranked
    return policy_fn


def make_graph_policy_fn(model, embed_fn, cos_w: float = 0.5):
    """DEPLOYABLE policy_fn — re-embeds the CURRENT graph each call (so atoms DERIVED and banked during
    the loop are scored too) and RESIDUALS over cosine: final = cos_w·cosine(task,atom) + policy_sigmoid.
    The learned policy dominates the COMPLEMENT conditioning (its win: it lifts the atom the partial
    program still needs, which cosine buries), while cosine supplies base relevance so a brand-NEW atom
    the net never trained on stays findable. This is the version dropped into MembraneSolver."""
    import torch

    def policy_fn(task, selected, graph, retriever):
        impl = [nid for nid in graph.nodes if graph.nodes[nid].node_type == "implementation"]
        if not impl:
            return []
        A = torch.tensor(np.stack([embed_fn(graph.nodes[nid].text) for nid in impl]))
        x = torch.tensor(embed_fn(task["text"]))
        sel_ix = [impl.index(s) for s in selected if s in impl]
        sel_vec = A[sel_ix].mean(0) if sel_ix else torch.zeros(A.shape[1])
        with torch.no_grad():
            pol = torch.sigmoid(model(x, A, sel_vec)).tolist()
        cos = dict(retriever.rank(task["text"]))          # base relevance (novel-atom-safe)
        scored = [(nid, cos_w * cos.get(nid, 0.0) + pol[i]) for i, nid in enumerate(impl)]
        return sorted(scored, key=lambda z: -z[1])
    return policy_fn


def train_and_make_policy(graph, embed_fn=None, tasks=None, **kw):
    """Convenience: train ComplementPolicy on `graph` and return (model, deployable policy_fn).
    Default embedder = HashEmbed(256) (no deps; mpnet can be injected on molab)."""
    embed_fn = embed_fn or HashEmbed(dim=256)
    model, _ctx = train_policy(graph, tasks or TRAIN_TASKS, embed_fn, **kw)
    return model, make_graph_policy_fn(model, embed_fn)


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST — policy beats cosine at complement retrieval; drops into the membrane
# ═══════════════════════════════════════════════════════════════════════════════

def _seed_graph() -> MemoryGraph:
    from v5.runtime.algo_grr_seed import build_graph
    g = build_graph()
    nodes = {n["id"]: Node.from_dict(n) for n in g["nodes"]}
    edges = [Edge.from_dict(e) for e in g["edges"]]
    return MemoryGraph(nodes, edges, metadata=dict(g["metadata"]))


def _complement_rank(ranker, atom_id: str) -> int:
    """1-based rank of atom_id in a [(id,score),...] list (lower = better)."""
    for i, (nid, _s) in enumerate(ranker):
        if nid == atom_id:
            return i + 1
    return len(ranker) + 1


def _selftest() -> bool:
    print("algo_grr_policy --selftest: trained complement-retrieval policy vs cosine\n")
    graph = _seed_graph()
    embed = HashEmbed(dim=256)          # 96 collides on the string cluster; 256 is clean
    ok = True

    model, ctx = train_policy(graph, TRAIN_TASKS, embed, steps=2000, seed=0)
    policy_fn = make_graph_policy_fn(model, embed)     # the DEPLOYABLE growth-aware path
    cosine = TokenRetriever(graph)

    # ── [1] complement rank: after selecting the FIRST atom, where is the missing SECOND? ────
    # genuine-2 compositions (two independent atoms) — where cosine buries the complement.
    cases = [
        ("count how many divisors of n are prime numbers", "impl_divisors", "impl_is_prime"),
        ("check whether the gcd of two numbers is prime", "impl_gcd", "impl_is_prime"),
        ("reversed digits of the largest contiguous subarray sum",
         "impl_max_subarray_sum", "impl_reverse_digits"),
        ("count the prime numbers among a list's unique elements", "impl_unique", "impl_is_prime"),
    ]
    pol_ranks, cos_ranks = [], []
    for text, first, comp in cases:
        task = {"text": text}
        pr = _complement_rank(policy_fn(task, [first], graph, cosine), comp)
        cr = _complement_rank([(n, s) for n, s in cosine.rank(text, exclude={first})], comp)
        pol_ranks.append(pr); cos_ranks.append(cr)
        print(f"  {comp:26s} after {first:24s}  policy_rank={pr:2d}  cosine_rank={cr:2d}")
    avg_p, avg_c = np.mean(pol_ranks), np.mean(cos_ranks)
    top1 = sum(r == 1 for r in pol_ranks)
    print(f"  [1] complement rank: policy avg={avg_p:.2f} (top1 {top1}/{len(cases)}) vs "
          f"cosine avg={avg_c:.2f} -> {'PASS' if avg_p < avg_c and top1 >= 3 else 'FAIL'}")
    ok &= (avg_p < avg_c and top1 >= 3)

    # ── [2] policy drops into the membrane and solves compositions in <= cosine hops ──────────
    recipes = {
        "t_primediv": "def t_primediv(n):\n    return sum(1 for d in divisors(n) if is_prime(d))\n",
        "t_revprime": "def t_revprime(n):\n    return is_prime(reverse_digits(n))\n",
        "t_lcm": "def t_lcm(a, b):\n    return lcm(a, b)\n",
        "t_anag": "def t_anag(a, b):\n    return is_anagram(a, b)\n",
    }
    tasks = [
        # genuine-2 (cosine buries the complement -> wastes hops; policy retrieves it directly)
        dict(text="count how many divisors of n are prime numbers", entry="t_primediv",
             tests=[((12,), 2), ((30,), 3), ((7,), 1)]),
        dict(text="check whether the digit reversal of n is prime", entry="t_revprime",
             tests=[((13,), True), ((12,), False), ((20,), True)]),   # rev:31 T, 21 F, 2 T
        # closure-shortcut (single selection; policy should match cosine at 1 hop)
        dict(text="least common multiple of two integers", entry="t_lcm",
             tests=[((4, 6), 12), ((3, 5), 15)]),
        dict(text="check if two strings are anagrams of each other", entry="t_anag",
             tests=[(("listen", "silent"), True), (("abc", "abd"), False)]),
    ]
    compiler = make_stub_compiler(recipes)
    pol_hops = cos_hops = 0
    pol_solved = cos_solved = 0
    for t in tasks:
        rp = MembraneSolver(graph, compiler, policy_fn=policy_fn).solve(t)
        rc = MembraneSolver(graph, compiler).solve(t)
        pol_hops += sum(1 for e in rp["trace"] if "hop" in e)
        cos_hops += sum(1 for e in rc["trace"] if "hop" in e)
        pol_solved += int(rp["solved"]); cos_solved += int(rc["solved"])
    print(f"  [2] membrane solve: policy {pol_solved}/{len(tasks)} in {pol_hops} hops, "
          f"cosine {cos_solved}/{len(tasks)} in {cos_hops} hops -> "
          f"{'PASS' if pol_solved == len(tasks) and pol_hops <= cos_hops else 'FAIL'}")
    ok &= (pol_solved == len(tasks) and pol_hops <= cos_hops)

    # ── [3] GROWTH-aware: a NEWLY-BANKED atom (not in training) is scored by the deployable policy ──
    from graph_core import Node
    g2 = _seed_graph()
    g2.nodes["impl_nth_fibonacci"] = Node(
        id="impl_nth_fibonacci", text="the n-th fibonacci number", node_type="implementation",
        metadata={"code": "def nth_fibonacci(n):\n    a,b=0,1\n    for _ in range(n): a,b=b,a+b\n    return a\n",
                  "entry": "nth_fibonacci", "kind": "atom", "origin": "derived"})
    g2._rebuild_index()
    ranked = make_graph_policy_fn(model, embed)({"text": "the n-th fibonacci number"}, [], g2, TokenRetriever(g2))
    ids = [nid for nid, _s in ranked]
    scored_new = "impl_nth_fibonacci" in ids
    rank_new = (ids.index("impl_nth_fibonacci") + 1) if scored_new else 999
    # deployable bar = FINDABLE within the membrane's hop budget (6), not necessarily #1: the net never
    # trained on this atom, and the loop's verifier-gated speculative-add resolves the rest.
    findable = scored_new and rank_new <= 6
    print(f"  [3] growth-aware: derived atom scored={scored_new}, rank={rank_new} (<=6 findable) "
          f"-> {'PASS' if findable else 'FAIL'}")
    ok &= findable

    print(f"\n  ALGO_GRR_POLICY SELFTEST -> {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
