"""algo_grr_seed — build a CLEAN seed graph for the GRR-Tool (frozen-compiler + TRM-membrane) design.

Why this exists
---------------
The grown graphs (grr_grown / grown_graph*) are POLLUTED:
  - nodes are whole task-solutions banked under their entry-point name
    (`impl_similar_elements` = the raw MBPP prompt + a full solution),
  - reusable helpers are TRAPPED as nested inner functions instead of banked,
  - topology is FLAT: 371 part_of edges, ZERO depend edges -> nothing composes
    -> cross-task reuse is mechanically 0.

This module builds the opposite: a small set of VERIFIED, MINIMAL, genuinely
reusable primitive atoms with REAL depend/part_of topology and helper-granularity.
Each atom carries a concise PURPOSE string as its retrieval key (never a task
prompt). Composed atoms call their dependencies by bare name; the depend edge
records the composition so resolve_deps can assemble the closure at realize-time
(the proven compose-forced convention).

The graph is emitted in the exact graph_core schema so every existing tool
(MGRetriever / resolve_deps / subgraph / gnn_encoder) loads it unchanged.

    build:     python -m v5.runtime.algo_grr_seed --build --out graphs/grr_seed_clean.json
    selftest:  python -m v5.runtime.algo_grr_seed --selftest      (no GPU; verifies every atom + topology)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

# ═══════════════════════════════════════════════════════════════════════════════
# ATOM LIBRARY — each atom is (name, purpose, code, concept, depends, tests)
#   purpose : concise retrieval key (what it DOES), NOT a task prompt
#   code    : minimal verified body; calls deps by BARE name (closure resolved via depend edges)
#   depends : names of atoms this one calls (real composition structure)
#   tests   : (args_tuple, expected) pairs — used to VERIFY the atom (fuzz gate)
# ═══════════════════════════════════════════════════════════════════════════════

Atom = dict  # {name, purpose, code, concept, depends, tests}

ATOMS: list[Atom] = [
    # ── number theory ──────────────────────────────────────────────────────────
    dict(name="is_prime", concept="number_theory", depends=[],
         purpose="primality test — True iff n is a prime number",
         code=(
             "def is_prime(n):\n"
             "    if n < 2:\n"
             "        return False\n"
             "    i = 2\n"
             "    while i * i <= n:\n"
             "        if n % i == 0:\n"
             "            return False\n"
             "        i += 1\n"
             "    return True\n"),
         tests=[((1,), False), ((2,), True), ((17,), True), ((18,), False), ((97,), True)]),

    dict(name="gcd", concept="number_theory", depends=[],
         purpose="greatest common divisor of a and b",
         code=(
             "def gcd(a, b):\n"
             "    while b:\n"
             "        a, b = b, a % b\n"
             "    return a\n"),
         tests=[((12, 8), 4), ((17, 5), 1), ((100, 75), 25), ((0, 9), 9)]),

    dict(name="lcm", concept="number_theory", depends=["gcd"],
         purpose="least common multiple of a and b",
         code=(
             "def lcm(a, b):\n"
             "    return a * b // gcd(a, b)\n"),
         tests=[((4, 6), 12), ((3, 5), 15), ((12, 8), 24), ((7, 7), 7)]),

    dict(name="divisors", concept="number_theory", depends=[],
         purpose="sorted list of all positive divisors of n",
         code=(
             "def divisors(n):\n"
             "    out = []\n"
             "    i = 1\n"
             "    while i * i <= n:\n"
             "        if n % i == 0:\n"
             "            out.append(i)\n"
             "            if i != n // i:\n"
             "                out.append(n // i)\n"
             "        i += 1\n"
             "    return sorted(out)\n"),
         tests=[((12,), [1, 2, 3, 4, 6, 12]), ((7,), [1, 7]), ((1,), [1]), ((16,), [1, 2, 4, 8, 16])]),

    dict(name="sum_divisors", concept="number_theory", depends=["divisors"],
         purpose="sum of all positive divisors of n",
         code=(
             "def sum_divisors(n):\n"
             "    return sum(divisors(n))\n"),
         tests=[((12,), 28), ((6,), 12), ((7,), 8), ((1,), 1)]),

    dict(name="is_perfect", concept="number_theory", depends=["sum_divisors"],
         purpose="True iff n equals the sum of its proper divisors (perfect number)",
         code=(
             "def is_perfect(n):\n"
             "    return n > 0 and sum_divisors(n) - n == n\n"),
         tests=[((6,), True), ((28,), True), ((12,), False), ((1,), False)]),

    dict(name="digit_sum", concept="number_theory", depends=[],
         purpose="sum of the decimal digits of a non-negative integer n",
         code=(
             "def digit_sum(n):\n"
             "    s = 0\n"
             "    n = abs(n)\n"
             "    while n:\n"
             "        s += n % 10\n"
             "        n //= 10\n"
             "    return s\n"),
         tests=[((0,), 0), ((123,), 6), ((9999,), 36), ((10,), 1)]),

    dict(name="reverse_digits", concept="number_theory", depends=[],
         purpose="integer formed by reversing the decimal digits of n",
         code=(
             "def reverse_digits(n):\n"
             "    r = 0\n"
             "    n = abs(n)\n"
             "    while n:\n"
             "        r = r * 10 + n % 10\n"
             "        n //= 10\n"
             "    return r\n"),
         tests=[((123,), 321), ((100,), 1), ((0,), 0), ((5,), 5)]),

    dict(name="is_palindrome_number", concept="number_theory", depends=["reverse_digits"],
         purpose="True iff the integer n reads the same forwards and backwards",
         code=(
             "def is_palindrome_number(n):\n"
             "    return n >= 0 and n == reverse_digits(n)\n"),
         tests=[((121,), True), ((123,), False), ((7,), True), ((10,), False)]),

    # ── lists ──────────────────────────────────────────────────────────────────
    dict(name="unique", concept="lists", depends=[],
         purpose="list with duplicates removed, preserving first-seen order",
         code=(
             "def unique(xs):\n"
             "    seen = set()\n"
             "    out = []\n"
             "    for x in xs:\n"
             "        if x not in seen:\n"
             "            seen.add(x)\n"
             "            out.append(x)\n"
             "    return out\n"),
         tests=[(([1, 1, 2, 3, 2],), [1, 2, 3]), (([],), []), (([5, 5, 5],), [5])]),

    dict(name="flatten", concept="lists", depends=[],
         purpose="flatten a list of lists by one level",
         code=(
             "def flatten(xss):\n"
             "    out = []\n"
             "    for xs in xss:\n"
             "        for x in xs:\n"
             "            out.append(x)\n"
             "    return out\n"),
         tests=[(([[1, 2], [3]],), [1, 2, 3]), (([[], [4]],), [4]), (([],), [])]),

    dict(name="count_occurrences", concept="lists", depends=[],
         purpose="number of times value v appears in list xs",
         code=(
             "def count_occurrences(xs, v):\n"
             "    c = 0\n"
             "    for x in xs:\n"
             "        if x == v:\n"
             "            c += 1\n"
             "    return c\n"),
         tests=[(([1, 2, 2, 3], 2), 2), (([1, 2, 3], 9), 0), (([], 1), 0)]),

    dict(name="most_common", concept="lists", depends=["count_occurrences"],
         purpose="the element that appears most often in xs (first on ties)",
         code=(
             "def most_common(xs):\n"
             "    best = None\n"
             "    best_c = -1\n"
             "    for x in xs:\n"
             "        c = count_occurrences(xs, x)\n"
             "        if c > best_c:\n"
             "            best_c = c\n"
             "            best = x\n"
             "    return best\n"),
         tests=[(([1, 2, 2, 3],), 2), (([4],), 4), (([1, 1, 2, 2],), 1)]),

    dict(name="running_max", concept="lists", depends=[],
         purpose="prefix maxima — list whose i-th element is max(xs[:i+1])",
         code=(
             "def running_max(xs):\n"
             "    out = []\n"
             "    m = None\n"
             "    for x in xs:\n"
             "        m = x if m is None else (x if x > m else m)\n"
             "        out.append(m)\n"
             "    return out\n"),
         tests=[(([1, 3, 2, 5],), [1, 3, 3, 5]), (([],), []), (([2, 1],), [2, 2])]),

    dict(name="max_subarray_sum", concept="lists", depends=[],
         purpose="largest sum of any contiguous subarray (Kadane)",
         code=(
             "def max_subarray_sum(xs):\n"
             "    best = xs[0]\n"
             "    cur = xs[0]\n"
             "    for x in xs[1:]:\n"
             "        cur = x if cur < 0 else cur + x\n"
             "        if cur > best:\n"
             "            best = cur\n"
             "    return best\n"),
         tests=[(([-2, 1, -3, 4, -1, 2, 1, -5, 4],), 6), (([1, 2, 3],), 6), (([-1, -2],), -1)]),

    # ── strings ────────────────────────────────────────────────────────────────
    dict(name="reverse_string", concept="strings", depends=[],
         purpose="the string s reversed",
         code=(
             "def reverse_string(s):\n"
             "    return s[::-1]\n"),
         tests=[(("abc",), "cba"), (("",), ""), (("racecar",), "racecar")]),

    dict(name="is_palindrome", concept="strings", depends=["reverse_string"],
         purpose="True iff string s reads the same forwards and backwards",
         code=(
             "def is_palindrome(s):\n"
             "    return s == reverse_string(s)\n"),
         tests=[(("racecar",), True), (("abc",), False), (("",), True)]),

    dict(name="char_freq", concept="strings", depends=[],
         purpose="dict mapping each character of s to its count",
         code=(
             "def char_freq(s):\n"
             "    d = {}\n"
             "    for ch in s:\n"
             "        d[ch] = d.get(ch, 0) + 1\n"
             "    return d\n"),
         tests=[(("aab",), {"a": 2, "b": 1}), (("",), {}), (("xx",), {"x": 2})]),

    dict(name="is_anagram", concept="strings", depends=["char_freq"],
         purpose="True iff strings a and b are anagrams of each other",
         code=(
             "def is_anagram(a, b):\n"
             "    return char_freq(a) == char_freq(b)\n"),
         tests=[(("listen", "silent"), True), (("abc", "abd"), False), (("", ""), True)]),

    # ── search ─────────────────────────────────────────────────────────────────
    dict(name="binary_search", concept="search", depends=[],
         purpose="index of target t in sorted list xs, or -1 if absent",
         code=(
             "def binary_search(xs, t):\n"
             "    lo, hi = 0, len(xs) - 1\n"
             "    while lo <= hi:\n"
             "        mid = (lo + hi) // 2\n"
             "        if xs[mid] == t:\n"
             "            return mid\n"
             "        if xs[mid] < t:\n"
             "            lo = mid + 1\n"
             "        else:\n"
             "            hi = mid - 1\n"
             "    return -1\n"),
         tests=[(([1, 3, 5, 7], 5), 2), (([1, 3, 5, 7], 4), -1), (([], 1), -1), (([2], 2), 0)]),

    dict(name="merge_sorted", concept="search", depends=[],
         purpose="merge two sorted lists a and b into one sorted list",
         code=(
             "def merge_sorted(a, b):\n"
             "    i = j = 0\n"
             "    out = []\n"
             "    while i < len(a) and j < len(b):\n"
             "        if a[i] <= b[j]:\n"
             "            out.append(a[i]); i += 1\n"
             "        else:\n"
             "            out.append(b[j]); j += 1\n"
             "    out.extend(a[i:]); out.extend(b[j:])\n"
             "    return out\n"),
         tests=[(([1, 3, 5], [2, 4]), [1, 2, 3, 4, 5]), (([], [1]), [1]), (([1], []), [1])]),
]

CONCEPTS = {
    "number_theory": "number theory — primes, divisors, digits, gcd/lcm",
    "lists": "list processing — dedup, flatten, count, scan, subarray",
    "strings": "string processing — reversal, palindrome, character frequency, anagram",
    "search": "searching and merging over ordered data",
}


# ═══════════════════════════════════════════════════════════════════════════════
# VERIFY — assemble each atom with its dependency closure and run its tests
# ═══════════════════════════════════════════════════════════════════════════════

def _by_name() -> dict[str, Atom]:
    return {a["name"]: a for a in ATOMS}


def _dep_closure(name: str, index: dict[str, Atom], seen: set[str] | None = None) -> list[str]:
    """Topologically-ordered names of `name` and all transitive deps (deps first)."""
    seen = seen if seen is not None else set()
    order: list[str] = []
    def visit(n: str) -> None:
        if n in seen:
            return
        seen.add(n)
        for d in index[n]["depends"]:
            visit(d)
        order.append(n)
    visit(name)
    return order


def _assemble_source(name: str, index: dict[str, Atom]) -> str:
    """Full source = every atom in the dep closure, deps first (the realizer's job)."""
    return "\n".join(index[n]["code"] for n in _dep_closure(name, index))


def verify_atom(atom: Atom, index: dict[str, Atom]) -> tuple[bool, str]:
    """Execute the atom's dep-closure and check every test case. Returns (ok, detail)."""
    src = _assemble_source(atom["name"], index)
    ns: dict = {}
    try:
        exec(compile(src, f"<atom:{atom['name']}>", "exec"), ns)
    except Exception as e:  # noqa: BLE001
        return False, f"compile/exec error: {e!r}"
    fn: Callable | None = ns.get(atom["name"])
    if fn is None:
        return False, "entry function not defined after exec"
    for args, expected in atom["tests"]:
        try:
            got = fn(*args)
        except Exception as e:  # noqa: BLE001
            return False, f"call {atom['name']}{args!r} raised {e!r}"
        if got != expected:
            return False, f"{atom['name']}{args!r} -> {got!r}, expected {expected!r}"
    return True, f"{len(atom['tests'])} tests pass"


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD — emit the graph in the graph_core schema
# ═══════════════════════════════════════════════════════════════════════════════

def build_graph() -> dict:
    """Verify every atom, then build the clean seed graph dict (graph_core schema)."""
    index = _by_name()

    # 1. verify EVERY atom before it is allowed into the graph (the store-gate)
    for atom in ATOMS:
        ok, detail = verify_atom(atom, index)
        if not ok:
            raise ValueError(f"atom '{atom['name']}' FAILED verification: {detail}")
        # every declared dependency must exist
        for d in atom["depends"]:
            if d not in index:
                raise ValueError(f"atom '{atom['name']}' depends on unknown atom '{d}'")
            # and must actually be CALLED in the body (no phantom deps)
            if d not in atom["code"]:
                raise ValueError(f"atom '{atom['name']}' declares dep '{d}' but never calls it")

    nodes: list[dict] = []
    edges: list[dict] = []

    # 2. concept hubs
    for cid, ctext in CONCEPTS.items():
        nodes.append(dict(
            id=f"concept_{cid}", text=ctext, node_type="concept",
            confidence=0.5, importance=0.6, metadata={}, access_count=0, context_guard={},
        ))

    # 3. atom impl nodes + part_of (to concept) + depend (composition) edges
    for atom in ATOMS:
        nid = f"impl_{atom['name']}"
        nodes.append(dict(
            id=nid, text=atom["purpose"], node_type="implementation",
            confidence=0.9, importance=0.5,
            metadata=dict(
                code=atom["code"], entry=atom["name"], kind="atom", origin="seed",
                concept=atom["concept"], depends=list(atom["depends"]),
            ),
            access_count=0, context_guard={},
        ))
        edges.append(_edge(nid, f"concept_{atom['concept']}", "part_of"))
        for d in atom["depends"]:
            edges.append(_edge(nid, f"impl_{d}", "depend"))

    return {
        "metadata": {
            "seed": "grr_seed_clean", "built_by": "algo_grr_seed",
            "design": "frozen-compiler + TRM-membrane",
            "n_atoms": len(ATOMS), "n_concepts": len(CONCEPTS),
            "note": "verified primitive helpers; helper-granular; real depend topology",
        },
        "nodes": nodes,
        "edges": edges,
    }


def _edge(src: str, dst: str, relation: str) -> dict:
    return dict(src=src, dst=dst, relation=relation, strength=1.0, directed=True,
                metadata=dict(origin="seed"), confidence=1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    print("algo_grr_seed --selftest: clean seed graph (verified atoms + real topology)\n")
    index = _by_name()
    ok_all = True

    # [1] every atom verifies against its dep closure
    n_pass = 0
    for atom in ATOMS:
        ok, detail = verify_atom(atom, index)
        mark = "PASS" if ok else "FAIL"
        if not ok:
            ok_all = False
            print(f"  [atom] {atom['name']:22s} -> {mark}: {detail}")
        else:
            n_pass += 1
    print(f"  [1] atom verification: {n_pass}/{len(ATOMS)} pass -> {'PASS' if n_pass == len(ATOMS) else 'FAIL'}")

    # [2] build the graph (also runs the store-gate: dep existence + real-call checks)
    try:
        g = build_graph()
        print(f"  [2] build_graph: {len(g['nodes'])} nodes, {len(g['edges'])} edges -> PASS")
    except Exception as e:  # noqa: BLE001
        print(f"  [2] build_graph FAILED: {e!r}")
        return False

    # [3] topology sanity: depend edges present (composition, not flat), no dangling
    ids = {n["id"] for n in g["nodes"]}
    depend = [e for e in g["edges"] if e["relation"] == "depend"]
    part_of = [e for e in g["edges"] if e["relation"] == "part_of"]
    dangling = [e for e in g["edges"] if e["src"] not in ids or e["dst"] not in ids]
    n_composed = len({e["src"] for e in depend})
    print(f"  [3] topology: {len(depend)} depend, {len(part_of)} part_of, "
          f"{n_composed} composed atoms, {len(dangling)} dangling -> "
          f"{'PASS' if depend and not dangling else 'FAIL'}")
    if not depend or dangling:
        ok_all = False

    # [4] every atom part_of exactly one concept; every impl node reachable
    impls = [n for n in g["nodes"] if n["node_type"] == "implementation"]
    po_src = {e["src"] for e in part_of}
    orphan = [n["id"] for n in impls if n["id"] not in po_src]
    print(f"  [4] {len(impls)} atoms, {len(orphan)} orphan (no concept) -> "
          f"{'PASS' if not orphan else 'FAIL'}")
    if orphan:
        ok_all = False

    # [5] loads through the real graph_core.MemoryGraph unchanged
    try:
        import tempfile, os
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from graph_core import MemoryGraph  # type: ignore
        fd, tmp = tempfile.mkstemp(suffix=".json"); os.close(fd)
        Path(tmp).write_text(json.dumps(g), encoding="utf-8")
        mg = MemoryGraph.load_json(tmp)
        os.unlink(tmp)
        kept = len(mg.edges)
        print(f"  [5] graph_core.MemoryGraph.load_json: {len(mg.nodes)} nodes, "
              f"{kept}/{len(g['edges'])} edges kept -> "
              f"{'PASS' if kept == len(g['edges']) else 'FAIL'}")
        if kept != len(g["edges"]):
            ok_all = False
    except Exception as e:  # noqa: BLE001
        print(f"  [5] MemoryGraph load FAILED: {e!r}")
        ok_all = False

    print(f"\n  ALGO_GRR_SEED SELFTEST -> {'PASS' if ok_all else 'FAIL'}")
    return ok_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--out", default="graphs/grr_seed_clean.json")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(0 if _selftest() else 1)

    if a.build:
        g = build_graph()
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(g, indent=2), encoding="utf-8")
        depend = sum(1 for e in g["edges"] if e["relation"] == "depend")
        print(f"wrote {out}: {len(g['nodes'])} nodes, {len(g['edges'])} edges "
              f"({depend} depend), {len(ATOMS)} verified atoms, {len(CONCEPTS)} concepts")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
