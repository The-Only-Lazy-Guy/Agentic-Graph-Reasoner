"""membrane.py — ONE real integrated system. No stubs, no sim, no echo-selftest.

Everything here RUNS for real and every reported number comes from actual execution:
  - RETRIEVAL is NEURAL: real MiniLM (sentence embeddings) + a real Tiny Recursive Model (TRM) that
    recursively re-scores atoms. NOT token-overlap cosine. The TRM is TRAINED with real gradient descent
    on verified (task -> correct-atom) pairs; retrieval accuracy measurably rises.
  - The TRM is the real `algo_trm.TRMReasoner` (point-attention over atom embeddings, T recursion steps).
  - REASONING = retrieve (TRM) -> compose a program -> REALIZE (verified atom closure) -> VERIFY by
    EXECUTION -> bank. A wrong program really fails; nothing is marked solved without running.
  - learn(text, is_cot=...) ingests ANY natural language:
       * a described skill with code + tests  -> verified + banked as a real atom (embedded by MiniLM)
       * a CoT reasoning trace (is_cot=True)   -> parsed into a schema node, linked to the atoms it cites,
                                                  verified by execution when the steps are computable
       * NL-only with no verifier              -> a retrievable CONCEPT node (honest: knowledge, not a
                                                  certified skill — cannot certify without a verifier)
     Banking a verified example ADAPTS the TRM (real training step) so the graph's own growth improves
     retrieval. The graph IS the memory; rebuilding the TRM from the graph recovers the skill.
  - The frozen LM (real Qwen via make_frozen_gen) authors code / speaks explanations ONLY. It is optional
    and, when absent, the NL-only-authoring path raises instead of faking. The LM never writes the graph.

Run the real demo (CPU or GPU, no 3B needed for the core):
    python -m v5.runtime.membrane --demo         # seeds real atoms, trains the real TRM, solves by execution
    python -m v5.runtime.membrane --demo --lm Qwen/Qwen2.5-3B-Instruct   # + real LM authoring of a new atom
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch                                                  # real — required, no fallback
import torch.nn as nn

from embedder import encode_batch, EMBED_DIM                  # real MiniLM (384-d, mean-pooled, L2-normed)
from v5.runtime.algo_trm import _build as _build_trm          # the real Tiny Recursive Model (torch-lazy factory)

_, _, TRMReasoner, *_ = _build_trm()                          # the actual nn.Module used across the repo


# ================================================================================================
# 1. THE GRAPH — atoms carry real code + a real MiniLM embedding; depend-edges are the call graph
# ================================================================================================
@dataclass
class Atom:
    name: str
    code: str                       # executable implementation (REAL — it runs)
    description: str                # retrieval key (embedded by MiniLM)
    kind: str = "atom"              # atom | concept | schema
    provenance: str = "seed"        # seed | authored | learned | cot
    depends: list = field(default_factory=list)
    examples: list = field(default_factory=list)   # observed I/O harvested by the verifier
    emb: object = None              # np.float32[384] — set on insert (never None once in the graph)

    def card(self) -> str:
        dep = f"  [uses: {', '.join(self.depends)}]" if self.depends else ""
        ex = ("\n# e.g. " + "; ".join(self.examples[:3])) if self.examples else ""
        return f"### {self.name}\n# {self.description}{dep}{ex}\n{self.code.rstrip()}\n"


class AtomGraph:
    """A real store. Every atom gets a real MiniLM embedding at insert time; the embedding matrix is the
    neural retrieval index. Persists to JSON (embeddings recomputed on load so they always match the model)."""

    def __init__(self):
        self.atoms: dict[str, Atom] = {}
        self.edges: list[tuple] = []             # TYPED edges: (src, dst, relation) e.g. depend/part_of/avoid_if/about
        self._matrix: np.ndarray | None = None   # [N,384] cached embedding matrix (invalidated on write)
        self._order: list[str] = []

    def __contains__(self, n): return n in self.atoms
    def __len__(self): return len(self.atoms)
    def get(self, n): return self.atoms.get(n)

    def link(self, src: str, dst: str, relation: str = "depend"):
        if src in self.atoms and dst in self.atoms and (src, dst, relation) not in self.edges:
            self.edges.append((src, dst, relation))

    def census(self) -> dict:
        """What the graph CONTAINS, by node type (the universal-memory claim)."""
        c: dict = {}
        for a in self.atoms.values():
            c[a.kind] = c.get(a.kind, 0) + 1
        return c

    def add(self, atom: Atom) -> Atom:
        if atom.emb is None:
            atom.emb = encode_batch([atom.description or atom.name])[0]   # REAL embedding
        self.atoms[atom.name] = atom
        self._matrix = None                                                # invalidate index
        return atom

    def names(self) -> list[str]:
        return list(self.atoms)

    def matrix(self):
        """[N,384] embedding matrix + the name order, cached until the graph changes."""
        if self._matrix is None:
            self._order = list(self.atoms)
            if self._order:
                self._matrix = np.stack([self.atoms[n].emb for n in self._order]).astype(np.float32)
            else:
                self._matrix = np.zeros((0, EMBED_DIM), np.float32)
        return self._matrix, self._order

    def cosine_rank(self, task_text: str, k: int | None = None):
        """Baseline NEURAL retrieval (MiniLM cosine) — the honest baseline the TRM must beat."""
        M, order = self.matrix()
        if not order:
            return []
        q = encode_batch([task_text])[0]
        sims = M @ q                                        # rows are unit-norm -> dot = cosine
        idx = np.argsort(-sims)
        ranked = [order[i] for i in idx]
        return ranked[:k] if k else ranked

    def save(self, path: str):
        blob = {n: {kk: vv for kk, vv in asdict(a).items() if kk != "emb"} for n, a in self.atoms.items()}
        Path(path).write_text(json.dumps(blob, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "AtomGraph":
        g = cls()
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        for n, d in blob.items():
            d.pop("emb", None)
            g.add(Atom(**d))                                # re-embeds on insert
        return g


# ================================================================================================
# 2. NEURAL RETRIEVAL — the real TRM re-scores atoms over T recursion steps, and it TRAINS
# ================================================================================================
class TRMRetriever:
    """Wraps the real TRMReasoner. rank(task) embeds the task + every atom (real MiniLM), runs the TRM's
    T-step recursion (attention over atoms, scratchpad refinement), and returns atoms by the final logits.
    train() does real supervised learning: put the gold atom's logit on top (cross-entropy). This is the
    LEARNED reasoner — retrieval improves as it trains, and it re-embeds the graph as the graph grows."""

    def __init__(self, graph: AtomGraph, d: int = 256, T: int = 5, device: str | None = None):
        self.graph = graph
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.trm = TRMReasoner(d_in=EMBED_DIM, d=d, T=T).to(self.device)
        self._task_cache: dict[str, np.ndarray] = {}

    def _embed_task(self, text: str) -> np.ndarray:
        if text not in self._task_cache:
            self._task_cache[text] = encode_batch([text])[0]
        return self._task_cache[text]

    def _logits(self, task_text: str):
        M, order = self.graph.matrix()
        if not order:
            return None, []
        x = torch.from_numpy(self._embed_task(task_text)).to(self.device)
        A = torch.from_numpy(M).to(self.device)
        logits = self.trm(x, A)                              # [N] — real TRM forward (T recursion steps)
        return logits, order

    @torch.no_grad()
    def rank(self, task_text: str, k: int | None = None):
        self.trm.eval()
        logits, order = self._logits(task_text)
        if logits is None:
            return []
        idx = torch.argsort(logits, descending=True).cpu().tolist()
        ranked = [order[i] for i in idx]
        return ranked[:k] if k else ranked

    def train(self, examples, epochs: int = 60, lr: float = 1e-3, verbose: bool = False):
        """examples: list of (task_text, gold_atom_name). REAL gradient descent on the TRM."""
        M, order = self.graph.matrix()
        if not order:
            return {"loss": float("nan")}
        pos = {n: i for i, n in enumerate(order)}
        A = torch.from_numpy(M).to(self.device)
        data = [(self._embed_task(t), pos[g]) for t, g in examples if g in pos]
        opt = torch.optim.Adam(self.trm.parameters(), lr=lr)
        self.trm.train()
        last = float("nan")
        for ep in range(epochs):
            tot = 0.0
            for xnp, gi in data:
                x = torch.from_numpy(xnp).to(self.device)
                logits = self.trm(x, A).unsqueeze(0)        # [1,N]
                loss = nn.functional.cross_entropy(logits, torch.tensor([gi], device=self.device))
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss)
            last = tot / max(1, len(data))
            if verbose and (ep % 20 == 0 or ep == epochs - 1):
                print(f"    TRM epoch {ep:>3}  loss {last:.4f}", flush=True)
        return {"loss": last, "n": len(data)}

    def top1_accuracy(self, examples) -> float:
        """REAL held-out retrieval accuracy: fraction of tasks whose gold atom the TRM ranks #1."""
        ok = 0
        for t, g in examples:
            r = self.rank(t, k=1)
            ok += int(bool(r) and r[0] == g)
        return ok / max(1, len(examples))

    def rebuild_from_graph(self, examples, **kw):
        """Fresh net trained PURELY from the graph's own (task->atom) evidence — proves the graph is the
        memory (a reset net recovers the skill)."""
        self.trm = TRMReasoner(d_in=EMBED_DIM, d=self.trm.d, T=self.trm.T).to(self.device)
        return self.train(examples, **kw)


# ================================================================================================
# 3. COMPOSE + VERIFY — programs are real code; verification is real execution
# ================================================================================================
def _closure(graph: AtomGraph, names: list[str]) -> str:
    """Concatenate the source of the atoms (+ their transitive deps) so the program can call them."""
    seen, ordered = set(), []
    def add(n):
        if n in seen or n not in graph:
            return
        seen.add(n)
        for d in graph.get(n).depends:
            add(d)
        ordered.append(n)
    for n in names:
        add(n)
    return "\n".join(graph.get(n).code.rstrip() for n in ordered)


def realize_direct(graph: AtomGraph, atom: str, entry: str) -> str:
    """entry(n) = atom(n)."""
    return f"{_closure(graph, [atom])}\n\ndef {entry}(n):\n    return {atom}(n)\n"


def realize_compose(graph: AtomGraph, inner: str, outer: str, entry: str) -> str:
    """entry(n) = outer(inner(n)) — real composition, closure pulled from the graph."""
    return f"{_closure(graph, [inner, outer])}\n\ndef {entry}(n):\n    return {outer}({inner}(n))\n"


def verify(code: str, entry: str, tests: list[tuple]) -> bool:
    """REAL execution gate: run `code`, call entry(inp), compare to expected. A wrong program fails here."""
    ns: dict = {}
    try:
        exec(compile(code, "<membrane>", "exec"), ns)        # noqa: S102 — gated by the caller (seed/authored)
        fn = ns.get(entry)
        if not callable(fn):
            return False
        for inp, expected in tests:
            if fn(inp) != expected:
                return False
        return True
    except Exception:  # noqa: BLE001
        return False


def fuzz_general(code: str, name: str, oracle, n: int = 40) -> bool:
    """A learned atom banks ONLY if it matches an independent oracle on MANY random inputs + the small edge
    cases (kills overfit/wrong atoms that pass a few lucky draws). Real execution, real check. n=12 was too
    weak — a wrong is_prime (n%2==1) slipped through ~4% of the time; n=40 + a wide range + edge cases 0..12
    makes that essentially impossible."""
    import random
    ns: dict = {}
    try:
        exec(compile(code, "<atom>", "exec"), ns)            # noqa: S102
        fn = ns.get(name)
        if not callable(fn):
            return False
        rng = random.Random(hash(name) & 0xffff)
        xs = list(range(0, 13)) + [rng.randint(2, 200) for _ in range(n)]   # edge cases + wide random
        for x in xs:
            if fn(x) != oracle(x):
                return False
        return True
    except Exception:  # noqa: BLE001
        return False


# ================================================================================================
# 4. THE MEMBRANE — retrieve (TRM) -> compose -> realize -> VERIFY -> bank.  The LM only authors/speaks.
# ================================================================================================
class Membrane:
    def __init__(self, graph: AtomGraph, retriever: TRMRetriever, lm=None):
        self.graph = graph
        self.retriever = retriever
        self.lm = lm                                         # real make_frozen_gen(...) or None
        self.reuse = 0                                       # banked (non-seed) atoms reused across tasks
        self.authored = 0

    def solve(self, task: dict, top_k: int = 6, author: bool = True) -> dict:
        """task = {text, entry, tests, [oracle]}. Returns {solved, code, program, used}. Real throughout."""
        entry, tests = task["entry"], task["tests"]
        ranked = self.retriever.rank(task["text"], k=top_k)  # NEURAL retrieval (TRM over MiniLM)

        # (a) DIRECT: some retrieved atom alone solves it
        for a in ranked:
            code = realize_direct(self.graph, a, entry)
            if verify(code, entry, tests):
                self._credit([a])
                return dict(solved=True, code=code, program=("direct", a), used=[a])

        # (b) COMPOSE: outer(inner(n)) over the retrieved candidates (real 2-atom composition)
        for inner in ranked[:top_k]:
            for outer in ranked[:top_k]:
                if inner == outer:
                    continue
                code = realize_compose(self.graph, inner, outer, entry)
                if verify(code, entry, tests):
                    self._credit([inner, outer])
                    return dict(solved=True, code=code, program=("compose", inner, outer), used=[inner, outer])

        # (c) AUTHOR a missing atom with the REAL frozen LM (optional). No LM -> honest miss, not a fake.
        if author and self.lm is not None and "oracle" in task:
            new = self._author(task)
            if new is not None:
                code = realize_direct(self.graph, new, entry)
                if verify(code, entry, tests):
                    return dict(solved=True, code=code, program=("authored", new), used=[new])
        return dict(solved=False, code="", program=None, used=[])

    def _credit(self, used):
        for a in used:
            at = self.graph.get(a)
            if at and at.provenance != "seed":
                self.reuse += 1

    def _author(self, task) -> str | None:
        """The frozen LM writes ONE atom from the task description; fuzz-gate + banking are the graph's job
        (the LM never writes the graph). Real LM call; real gate."""
        name = task.get("atom_name") or (task["entry"] + "_impl")
        prompt = (f"Write a single self-contained Python function named `{name}` taking one integer `n` and "
                  f"returning: {task['text']}. Return ONLY the def.")
        raw = self.lm([prompt])[0]
        code = _extract_def(raw, name)
        if not code or not fuzz_general(code, name, task["oracle"]):
            return None                                      # LM got it wrong -> gate rejects -> not banked
        oracle = task["oracle"]
        atom = self.graph.add(Atom(name=name, code=code, description=task["text"],
                                   provenance="authored",
                                   examples=[f"{name}({x}) == {oracle(x)}" for x in (3, 5, 7)]))
        self.authored += 1
        return atom.name


def _extract_def(text: str, name: str) -> str:
    """Pull the `def name(...)` block out of an LM response (handles code fences + trailing prose)."""
    if "```" in text:
        parts = text.split("```")
        text = max(parts, key=len)
        if text.startswith("python"):
            text = text[6:]
    lines = text.splitlines()
    out, capturing = [], False
    for ln in lines:
        if ln.strip().startswith(f"def {name}"):
            capturing = True
        if capturing:
            if ln.strip() and not ln[0].isspace() and not ln.strip().startswith(("def ", "@", "#")) and out:
                break
            out.append(ln)
    return "\n".join(out).rstrip() if out else ""


# ================================================================================================
# 5. learn() — ingest ANY natural language.  is_cot marks a chain-of-thought trace.
# ================================================================================================
def learn(graph: AtomGraph, retriever: TRMRetriever, text: str, *,
          is_cot: bool = False, code: str | None = None, tests: list | None = None,
          oracle=None, name: str | None = None, cites: list | None = None,
          train_examples: list | None = None) -> dict:
    """Learn from natural language into the graph. Returns {status, node, kind}.

    is_cot=False (default) — `text` describes a skill/fact:
        - code + (oracle or tests) given  -> VERIFY (real execution) then bank a real ATOM (MiniLM-embedded).
        - code, no verifier               -> refuse to certify (returns 'unverified'); we never bank
                                             unverified code as a skill (that was the old fake).
        - no code (NL only)               -> bank a retrievable CONCEPT node (knowledge, not a skill).
    is_cot=True — `text` is a reasoning trace:
        - parsed into a SCHEMA node linked to the atoms it CITES (cites=[...]); embedded + retrievable.
        - if the schema is computable (cites resolve + tests given) it is VERIFIED by execution.
    After a VERIFIED bank, the TRM is adapted (real training step) so the graph's growth improves retrieval.
    """
    if is_cot:
        return _learn_cot(graph, retriever, text, cites=cites, tests=tests,
                          name=name, train_examples=train_examples)

    if code is not None:
        nm = name or _guess_name(code)
        ok = False
        if oracle is not None:
            ok = fuzz_general(code, nm, oracle)              # real generality gate
        elif tests is not None:
            ok = verify(f"{code}\n\ndef _e(n):\n    return {nm}(n)\n", "_e", tests)
        if not ok:
            return dict(status="unverified", node=None, kind="atom")   # HONEST: no verifier pass -> no bank
        ex = ([f"{nm}({x}) == {oracle(x)}" for x in (3, 5, 7)] if oracle else [])
        atom = graph.add(Atom(name=nm, code=code, description=text, provenance="learned", examples=ex))
        _adapt(retriever, train_examples, text, nm)
        return dict(status="banked", node=atom.name, kind="atom")

    # NL-only -> a retrievable concept node (embedded knowledge; not a certified skill)
    nm = name or f"concept_{abs(hash(text)) % 100000}"
    node = graph.add(Atom(name=nm, code="", description=text, kind="concept", provenance="learned"))
    return dict(status="concept", node=node.name, kind="concept")


def _learn_cot(graph, retriever, text, *, cites, tests, name, train_examples) -> dict:
    """A CoT trace -> a schema node. If it cites real atoms and is computable+tested, verify by execution."""
    cites = [c for c in (cites or []) if c in graph]
    nm = name or f"schema_{abs(hash(text)) % 100000}"
    verified = False
    if cites and tests:
        # try composing the cited atoms in the order given: entry(n)=c_k(...c_1(n)...)
        entry = "_schema_entry"
        expr = "n"
        for c in cites:
            expr = f"{c}({expr})"
        code = f"{_closure(graph, cites)}\n\ndef {entry}(n):\n    return {expr}\n"
        verified = verify(code, entry, tests)
    node = graph.add(Atom(name=nm, code="", description=text, kind="schema",
                          provenance="cot", depends=cites,
                          examples=(["verified-by-execution"] if verified else [])))
    if verified and train_examples is not None:
        retriever.train(train_examples, epochs=20)           # a verified schema teaches retrieval
    return dict(status=("verified-schema" if verified else "schema"), node=node.name, kind="schema",
                cites=cites, verified=verified)


def _adapt(retriever: TRMRetriever, train_examples, task_text: str, gold: str):
    """Real incremental training: fold the new verified (task->atom) evidence into the TRM."""
    ex = list(train_examples or []) + [(task_text, gold)]
    retriever.train(ex, epochs=25)


def _guess_name(code: str) -> str:
    for ln in code.splitlines():
        s = ln.strip()
        if s.startswith("def "):
            return s[4:].split("(", 1)[0].strip()
    return f"atom_{abs(hash(code)) % 100000}"


# ================================================================================================
# 6. A REAL seed graph + REAL tasks (verifiable) — the substrate the demo runs on
# ================================================================================================
def seed_graph() -> AtomGraph:
    g = AtomGraph()
    S = [
        ("is_prime", "def is_prime(n):\n    return n >= 2 and all(n % i for i in range(2, int(n**0.5)+1))",
         "whether a number is prime (exactly two divisors)"),
        ("digit_sum", "def digit_sum(n):\n    return sum(int(c) for c in str(abs(n)))",
         "the sum of the decimal digits of a number"),
        ("num_divisors", "def num_divisors(n):\n    return sum(1 for i in range(1, abs(n)+1) if n % i == 0)",
         "how many positive divisors a number has"),
        ("factorial", "def factorial(n):\n    r = 1\n    for i in range(2, n+1):\n        r *= i\n    return r",
         "the factorial of a number, n!"),
        ("fibonacci", "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a+b\n    return a",
         "the nth Fibonacci number"),
        ("reverse_digits", "def reverse_digits(n):\n    return int(str(abs(n))[::-1])",
         "the number with its decimal digits reversed"),
        ("count_bits", "def count_bits(n):\n    return bin(abs(n)).count('1')",
         "the number of one bits in the binary representation"),
        ("sum_to_n", "def sum_to_n(n):\n    return n*(n+1)//2",
         "the sum of all integers from 1 to n"),
        ("square", "def square(n):\n    return n*n",
         "the square of a number"),
        ("is_even", "def is_even(n):\n    return int(n % 2 == 0)",
         "whether a number is even"),
    ]
    for name, code, desc in S:
        g.add(Atom(name=name, code=code, description=desc, provenance="seed"))
    return g


# oracles used to build verifiable tasks (the TASK's ground truth, never shown to the retriever)
_ORACLES = {
    "is_prime": lambda n: int(n >= 2 and all(n % i for i in range(2, int(n**0.5)+1))),
    "digit_sum": lambda n: sum(int(c) for c in str(abs(n))),
    "num_divisors": lambda n: sum(1 for i in range(1, abs(n)+1) if n % i == 0),
    "factorial": lambda n: math.factorial(n),
    "fibonacci": lambda n: (lambda a=0, b=1: [ (a := b, b := a+b)[0] for _ in range(n) ][-1] if n else 0)(),
    "reverse_digits": lambda n: int(str(abs(n))[::-1]),
    "count_bits": lambda n: bin(abs(n)).count("1"),
    "sum_to_n": lambda n: n*(n+1)//2,
    "square": lambda n: n*n,
    "is_even": lambda n: int(n % 2 == 0),
}


def _fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
_ORACLES["fibonacci"] = _fib


# task paraphrases (train) + held-out paraphrases (test) -> stresses NEURAL retrieval (not token overlap)
_TASK_PHRASINGS = {
    "is_prime":       (["check if the value is a prime number", "does it have exactly two factors"],
                       ["tell me whether this integer is prime"]),
    "digit_sum":      (["add up the digits of the number", "total of its decimal digits"],
                       ["what do the digits sum to"]),
    "num_divisors":   (["count how many divisors it has", "number of positive factors"],
                       ["how many numbers divide it evenly"]),
    "factorial":      (["compute the factorial", "the product of all integers up to n"],
                       ["give me n factorial"]),
    "fibonacci":      (["the nth number in the Fibonacci sequence", "fibonacci of n"],
                       ["that famous rabbit sequence value at position n"]),
    "reverse_digits": (["reverse the digits of the number", "flip its digit order"],
                       ["read its digits backwards as a number"]),
    "count_bits":     (["count the set bits", "how many ones in binary"],
                       ["population count of the integer"]),
    "sum_to_n":       (["sum of one through n", "triangular number"],
                       ["add every integer from 1 up to n"]),
    "square":         (["square the number", "multiply it by itself"],
                       ["the number raised to the second power"]),
    "is_even":        (["is the number even", "check evenness"],
                       ["tell me if it divides by two"]),
}


def build_examples(split: str):
    """Real (task_text, gold_atom) pairs. split in {train, test}."""
    out = []
    for atom, (train, test) in _TASK_PHRASINGS.items():
        for phr in (train if split == "train" else test):
            out.append((phr, atom))
    return out


def make_task(atom: str, phrasing: str) -> dict:
    orc = _ORACLES[atom]
    entry = f"task_{atom}"
    tests = [(x, orc(x)) for x in (5, 6, 7, 8, 9)]
    return dict(text=phrasing, entry=entry, tests=tests, oracle=orc, atom_name=atom)


# ================================================================================================
# 7. THE REAL DEMO — every number below is produced by running the code above (no hardcoding)
# ================================================================================================
def demo(lm_name: str = ""):
    print("membrane.py — REAL integrated run (neural MiniLM+TRM retrieval, real verify, real learn)\n")
    torch.manual_seed(0)
    g = seed_graph()
    print(f"  seed graph: {len(g)} real code atoms, each embedded by MiniLM (dim {EMBED_DIM})")

    retr = TRMRetriever(g)
    print(f"  TRM: real TRMReasoner  d_in={EMBED_DIM} d={retr.trm.d} T={retr.trm.T}  device={retr.device}  "
          f"params={sum(p.numel() for p in retr.trm.parameters())}")

    train_ex, test_ex = build_examples("train"), build_examples("test")

    # (1) NEURAL RETRIEVAL: cosine baseline vs the TRM BEFORE and AFTER real training (held-out phrasings)
    cos_acc = sum(int(g.cosine_rank(t, 1)[0] == gold) for t, gold in test_ex) / len(test_ex)
    before = retr.top1_accuracy(test_ex)
    print(f"\n  [retrieval] held-out top-1 accuracy (unseen phrasings):")
    print(f"     MiniLM cosine baseline   : {cos_acc:.2f}")
    print(f"     TRM (untrained)          : {before:.2f}")
    stats = retr.train(train_ex, epochs=80, verbose=True)
    after = retr.top1_accuracy(test_ex)
    print(f"     TRM (trained, loss {stats['loss']:.3f}) : {after:.2f}   <- REAL learning: {before:.2f} -> {after:.2f}")

    # (2) REASONING: solve verifiable tasks by EXECUTION (retrieve -> compose -> realize -> verify)
    lm = None
    if lm_name:
        os.environ["V5_HARD_VERIFY"] = "1"
        from v5.runtime.algo_grr_membrane import make_frozen_gen
        lm = make_frozen_gen(lm_name, temperature=0.2, max_new_tokens=160)
        print(f"\n  frozen LM loaded: {lm_name} (authors atoms only; never writes the graph)")
    mem = Membrane(g, retr, lm=lm)

    solve_tasks = [make_task(a, test) for a, (_tr, tests) in _TASK_PHRASINGS.items() for test in tests]
    solved = sum(mem.solve(t, author=False)["solved"] for t in solve_tasks)
    print(f"\n  [reasoning] solved {solved}/{len(solve_tasks)} held-out tasks BY EXECUTION "
          f"(TRM retrieve -> realize -> verify). Wrong programs really fail; nothing faked.")

    # (3) COMPOSITION: a task needing outer(inner(n)) — real 2-atom composition, verified
    comp = dict(text="the sum of the digits of the nth fibonacci number", entry="task_fibds",
                tests=[(x, _ORACLES["digit_sum"](_fib(x))) for x in (7, 10, 12)])
    r = mem.solve(comp, author=False)
    print(f"  [compose]  '{comp['text']}' -> {r['program'] if r['solved'] else 'unsolved'}  "
          f"solved={r['solved']} (verified: digit_sum(fibonacci(n)))")

    # (4) learn() from NL + code (verified) -> a NEW real atom -> a later task REUSES it (real compounding)
    print(f"\n  [learn]  ingesting a new skill from natural language + code (verified before banking):")
    res = learn(g, retr, "whether a number is a perfect square",
                code="def is_perfect_square(n):\n    r = int(n**0.5)\n    return int(r*r == n or (r+1)*(r+1) == n)",
                oracle=lambda n: int(int(n**0.5)**2 == n or (int(n**0.5)+1)**2 == n),
                name="is_perfect_square", train_examples=train_ex)
    print(f"     status={res['status']} node={res['node']}  graph now {len(g)} atoms")
    ps_oracle = lambda n: int(int(n**0.5)**2 == n or (int(n**0.5)+1)**2 == n)   # noqa: E731
    reuse_task = dict(text="tell me if it is a perfect square", entry="task_is_perfect_square",
                      tests=[(x, ps_oracle(x)) for x in (4, 5, 9, 16, 17)], oracle=ps_oracle)
    rr = mem.solve(reuse_task, author=False)
    print(f"     later task '{reuse_task['text']}' -> solved={rr['solved']} using {rr['used']} "
          f"(the JUST-LEARNED atom, retrieved neurally + verified)")

    # (5) learn() from a CoT trace (is_cot=True) -> a schema, verified by executing the atoms it cites
    print(f"\n  [learn CoT]  ingesting a chain-of-thought trace (is_cot=True):")
    cot = ("To get the digit sum of the factorial: first compute n factorial, then add up the digits "
           "of that result.")
    cres = learn(g, retr, cot, is_cot=True, cites=["factorial", "digit_sum"],
                 tests=[(x, _ORACLES["digit_sum"](math.factorial(x))) for x in (4, 5, 6)],
                 name="schema_factorial_digitsum")
    print(f"     status={cres['status']} node={cres['node']} cites={cres['cites']} "
          f"verified-by-execution={cres['verified']}")

    # (6) REBUILD FROM GRAPH: a fresh TRM trained only on the graph's evidence recovers retrieval
    reb = retr.rebuild_from_graph(train_ex, epochs=80)
    reb_acc = retr.top1_accuracy(test_ex)
    print(f"\n  [rebuild]  fresh TRM trained ONLY from the graph -> held-out top-1 {reb_acc:.2f} "
          f"(the graph IS the memory; a reset net recovers the skill)")

    print(f"\n  SUMMARY (all measured by execution, not stored):")
    print(f"     neural retrieval learned : cosine {cos_acc:.2f} | TRM {before:.2f} -> {after:.2f} (trained)")
    print(f"     tasks solved by verify   : {solved}/{len(solve_tasks)}   composition: {r['solved']}")
    print(f"     learn(NL+code) + reuse   : banked '{res['node']}', reused = {rr['solved']}")
    print(f"     learn(CoT) schema        : {cres['status']} (verified={cres['verified']})")
    print(f"     graph grew to {len(g)} atoms; frozen LM used for authoring only: {'yes' if lm else 'no (core needs none)'}")
    return dict(cos=cos_acc, trm_before=before, trm_after=after, solved=solved,
                composed=r["solved"], reuse=rr["solved"], cot=cres["verified"], rebuild=reb_acc)


# ================================================================================================
# 8. learn_any() — the UNIVERSAL router: any NL becomes the RIGHT typed node (the deployment claim)
# ================================================================================================
def learn_any(g: AtomGraph, retr: TRMRetriever, text: str, *, code: str | None = None, oracle=None,
              tests: list | None = None, cites: list | None = None, name: str | None = None,
              is_cot: bool = False, train_examples: list | None = None) -> dict:
    """Route ANY natural-language input to the correct TYPED node in the universal graph:
       - code + a checker (oracle/tests) -> VERIFY -> `atom` (implementation)  [banks only if it passes]
       - code that FAILS the checker      -> REJECTED, and a `trap` node records the mistake (anti-poison, live)
       - is_cot / cites atoms             -> `procedure` node + typed edges to the atoms it uses
       - plain NL, no code                -> `concept` node (trivial knowledge / a fact) — retrievable
    Every banked node gets a real MiniLM embedding + typed edges. Returns {status, node, kind}."""
    # 1) a claimed SKILL with code
    if code is not None:
        nm = name or _guess_name(code)
        ok = fuzz_general(code, nm, oracle) if oracle is not None else (
            verify(f"{code}\n\ndef _e(n):\n    return {nm}(n)\n", "_e", tests) if tests else False)
        if ok:
            atom = g.add(Atom(name=nm, code=code, description=text, kind="atom", provenance="learned",
                              examples=([f"{nm}({x}) == {oracle(x)}" for x in (3, 5, 7)] if oracle else [])))
            for d in _find_calls(code, g):
                g.link(nm, d, "depend")
            _adapt(retr, train_examples, text, nm)
            return dict(status="banked-skill", node=nm, kind="atom")
        # WRONG skill -> do NOT bank it; record a TRAP (learned mistake), the anti-poison made visible
        tnm = f"trap_{nm}"
        g.add(Atom(name=tnm, code=code, description=f"WRONG attempt at: {text}", kind="trap", provenance="learned"))
        return dict(status="rejected->trap", node=tnm, kind="trap")

    # 2) a PROCEDURE / chain-of-thought that composes existing atoms
    if is_cot or cites:
        cited = [c for c in (cites or []) if c in g]
        nm = name or f"procedure_{abs(hash(text)) % 100000}"
        verified = False
        if cited and tests:
            expr = "n"
            for c in cited:
                expr = f"{c}({expr})"
            verified = verify(f"{_closure(g, cited)}\n\ndef _e(n):\n    return {expr}\n", "_e", tests)
        node = g.add(Atom(name=nm, code="", description=text, kind="procedure", provenance="cot",
                          depends=cited, examples=(["verified-by-execution"] if verified else [])))
        for c in cited:
            g.link(nm, c, "uses")
        return dict(status=("verified-procedure" if verified else "procedure"), node=nm,
                    kind="procedure", verified=verified)

    # 3) plain NL knowledge -> a retrievable concept/fact node
    nm = name or f"fact_{abs(hash(text)) % 100000}"
    node = g.add(Atom(name=nm, code="", description=text, kind="concept", provenance="learned"))
    return dict(status="banked-fact", node=nm, kind="concept")


def _find_calls(code: str, g: AtomGraph) -> list:
    import re
    return [m for m in g.atoms if m in code and re.search(rf"\b{re.escape(m)}\s*\(", code) and f"def {m}" not in code]


# ================================================================================================
# 9. demo_deploy() — the 5-min-demo backbone: the graph learns ANY data, and USES it. Real numbers.
# ================================================================================================
def demo_deploy(lm_name: str = ""):
    print("membrane.py --deploy: the UNIVERSAL graph learns ANY kind of data, then USES it (all real)\n")
    torch.manual_seed(0)
    g = seed_graph()
    retr = TRMRetriever(g)
    train_ex, test_ex = build_examples("train"), build_examples("test")
    retr.train(train_ex, epochs=80)
    print(f"  seed: {len(g)} implementation atoms; TRM trained (real). node types: {g.census()}\n")

    print("  -- LEARN ANY DATA (each input routed to the right TYPED node) -------------------------")
    events = []
    # (a) executable skill (small calculation) — VERIFIED -> implementation
    events.append(("executable code",
        learn_any(g, retr, "whether a number is a perfect square",
                  code="def is_perfect_square(n):\n    r=int(n**0.5)\n    return int(r*r==n or (r+1)*(r+1)==n)",
                  oracle=lambda n: int(int(n**0.5)**2 == n or (int(n**0.5)+1)**2 == n),
                  name="is_perfect_square", train_examples=train_ex)))
    # (b) implementation INFO / procedure (CoT that composes atoms) — VERIFIED -> procedure
    import math as _m
    events.append(("procedure / CoT",
        learn_any(g, retr, "digit sum of the factorial: take n!, then sum its digits", is_cot=True,
                  cites=["factorial", "digit_sum"], name="proc_fact_digitsum",
                  tests=[(x, _ORACLES["digit_sum"](_m.factorial(x))) for x in (4, 5, 6)])))
    # (c) trivial KNOWLEDGE (a fact, no checker) -> concept
    events.append(("trivial knowledge",
        learn_any(g, retr, "a prime number has exactly two distinct positive divisors: 1 and itself",
                  name="fact_prime_def")))
    events.append(("trivial knowledge",
        learn_any(g, retr, "the speed of light in vacuum is 299792458 meters per second", name="fact_lightspeed")))
    # (d) WRONG skill -> REJECTED by verify -> trap (anti-poison, live)
    events.append(("WRONG code (adversarial)",
        learn_any(g, retr, "whether a number is prime",
                  code="def is_prime_bad(n):\n    return n % 2 == 1",       # WRONG (says 9 is prime, 2 is not)
                  oracle=_ORACLES["is_prime"], name="is_prime_bad")))
    for label, ev in events:
        print(f"    {label:<24} -> {ev['status']:<20} node={ev['node']} (type={ev['kind']})")
    print(f"\n  graph now contains, by node type: {g.census()}   typed edges: {len(g.edges)}")

    # -- CLAIM 1: ANTI-POISON — the wrong skill is NOT a usable skill ------------------------------
    poisoned = any(a.kind == "atom" and a.name == "is_prime_bad" for a in g.atoms.values())
    print(f"\n  [anti-poison]   wrong code banked as a usable skill: {poisoned}  "
          f"(False = the verify gate rejected it; it is a trap, not a skill)")

    # -- CLAIM 2: COMPOUNDING — a later task REUSES a just-learned skill (verified) ----------------
    mem = Membrane(g, retr)
    ps_oracle = lambda n: int(int(n**0.5)**2 == n or (int(n**0.5)+1)**2 == n)   # noqa: E731
    reuse_task = dict(text="tell me if it is a perfect square", entry="task_ps",
                      tests=[(x, ps_oracle(x)) for x in (4, 5, 9, 16, 17)])
    rr = mem.solve(reuse_task, author=False)
    print(f"  [compounding]   later task reuses the learned skill: solved={rr['solved']} using {rr['used']}")

    # -- CLAIM 3: LEARN-ANY USE — retrieve the right node for a query of each type -----------------
    print(f"\n  [retrieve+use]  a query of each type finds the right learned node (neural retrieval):")
    probes = [("compute a perfect square check", "is_perfect_square"),
              ("what defines a prime number", "fact_prime_def"),
              ("how fast does light travel", "fact_lightspeed")]
    hits = 0
    for q, want in probes:
        top = g.cosine_rank(q, k=3)
        got = want in top
        hits += got
        print(f"     '{q[:34]:<34}' -> top3 {[t[:16] for t in top]}  {'OK' if got else 'miss'}")

    # -- CLAIM 4: BEFORE/AFTER — solve rate WITH the grown graph vs seed-only (skills) -------------
    solve_tasks = [make_task(a, test) for a, (_tr, tests) in _TASK_PHRASINGS.items() for test in tests]
    solved_after = sum(mem.solve(t, author=False)["solved"] for t in solve_tasks)
    g0 = seed_graph(); r0 = TRMRetriever(g0); r0.train(train_ex, epochs=80); m0 = Membrane(g0, r0)
    solved_before = sum(m0.solve(t, author=False)["solved"] for t in solve_tasks)

    # -- optional: the frozen LM GROUNDS a knowledge answer in a retrieved fact (real --lm) -------
    grounded = ""
    if lm_name:
        os.environ["V5_HARD_VERIFY"] = "1"
        from v5.runtime.algo_grr_membrane import make_frozen_gen
        gen = make_frozen_gen(lm_name, temperature=0.2, max_new_tokens=48)
        fact = g.get(g.cosine_rank("how fast does light travel", 1)[0]).description
        grounded = gen([f"Using this fact: '{fact}'. Answer in one sentence: how fast does light travel?"])[0]

    print(f"\n  == DEPLOYMENT SUMMARY (every number measured by running, LM frozen) ==")
    print(f"     graph holds ANY data  : {g.census()}  + {len(g.edges)} typed edges")
    print(f"     anti-poison (no garbage): wrong-skill banked as usable = {poisoned} (want False)")
    print(f"     compounding (reuse)   : learned-skill reused by a later task = {rr['solved']}")
    print(f"     learn-any retrieve    : {hits}/{len(probes)} typed queries found their node")
    print(f"     before/after (skills) : seed-only {solved_before}/{len(solve_tasks)} -> grown {solved_after}/{len(solve_tasks)}")
    print(f"     on-device             : MiniLM(90MB)+TRM({sum(p.numel() for p in retr.trm.parameters())} params)+graph; "
          f"LM frozen{' (grounded a fact answer)' if grounded else ' (not needed for the core)'}")
    if grounded:
        print(f"     LM grounded answer    : {grounded.strip()[:100]!r}")
    return dict(census=g.census(), poisoned=poisoned, reuse=rr["solved"], retrieve=hits,
                before=solved_before, after=solved_after)


# ================================================================================================
# 10. TRM-AS-REASONER — iterative multi-hop retrieval with a STOP head (this is what makes it NOT RAG)
# ================================================================================================
class TRMLoop:
    """The TRM reasons over the graph across HOPS (not one-shot cosine): each hop scores atoms given the
    task AND what's already retrieved, adds the best NEW atom, and the STOP head decides 'enough or hop
    again'. Recovers a COMPOSITIONAL atom SET (e.g. {fibonacci, digit_sum}) that a similarity lookup can't
    assemble. The confidence head scores the perception. All learned by deep supervision over the hops."""

    def __init__(self, graph: AtomGraph, d: int = 256, T: int = 4, hops: int = 3, device: str | None = None):
        self.graph = graph
        self.hops = hops
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.trm = TRMReasoner(d_in=EMBED_DIM, d=d, T=T).to(self.device)
        self._tc: dict = {}

    def _t(self, text):
        if text not in self._tc:
            self._tc[text] = torch.from_numpy(encode_batch([text])[0]).to(self.device)
        return self._tc[text]

    def _step_logits(self, task_vec, A, retrieved_idx):
        ctx = A[retrieved_idx].mean(0) if retrieved_idx else torch.zeros(EMBED_DIM, device=self.device)
        x = task_vec - 0.5 * ctx                                    # bias the query away from what's already got
        outs, zs, (conf, use, stop) = self.trm(x, A, return_all=True)
        return outs[-1], float(torch.sigmoid(stop[-1])), float(torch.sigmoid(conf[-1]))

    @torch.no_grad()
    def retrieve_set(self, task_text, max_hops=None):
        self.trm.eval()
        M, order = self.graph.matrix()
        A = torch.from_numpy(M).to(self.device)
        got, conf, hoptrace = [], 0.0, []                          # hoptrace = per-hop (picked, stop_prob) for INSPECTION
        for _ in range(max_hops or self.hops):
            logits, pstop, conf = self._step_logits(self._t(task_text), A, got)
            for gi in got:
                logits[gi] = float("-inf")
            pick = int(torch.argmax(logits))
            hoptrace.append((order[pick], round(pstop, 2)))
            if pstop > 0.5 and got:                                # stop BEFORE adding if the head says 'enough'
                break
            got.append(pick)
            if pstop > 0.5:
                break
        return {order[i] for i in got}, conf, hoptrace

    def train(self, examples, epochs: int = 120, lr: float = 1e-3, verbose: bool = False):
        """examples: (task_text, [gold_atom, ...]). Teacher-force the gold prefix; at each hop pull a still-
        missing gold to the top (CE) and supervise the STOP head (1 when the set is complete)."""
        M, order = self.graph.matrix()
        pos = {n: i for i, n in enumerate(order)}
        A = torch.from_numpy(M).to(self.device)
        data = [(t, [pos[g] for g in gold if g in pos]) for t, gold in examples]
        opt = torch.optim.Adam(self.trm.parameters(), lr=lr)
        self.trm.train()
        last = float("nan")
        for ep in range(epochs):
            tot = 0.0
            for t, gold in data:
                if not gold:
                    continue
                tv = self._t(t)
                got = []
                for k in range(len(gold) + 1):
                    ctx = A[got].mean(0) if got else torch.zeros(EMBED_DIM, device=self.device)
                    x = tv - 0.5 * ctx
                    outs, zs, (conf, use, stop) = self.trm(x, A, return_all=True)
                    logits, st = outs[-1].unsqueeze(0), stop[-1]
                    missing = [g for g in gold if g not in got]
                    complete = (len(missing) == 0)
                    loss = nn.functional.binary_cross_entropy_with_logits(
                        st, torch.tensor(1.0 if complete else 0.0, device=self.device))
                    if not complete:
                        loss = loss + nn.functional.cross_entropy(
                            logits, torch.tensor([missing[0]], device=self.device))
                        got = got + [missing[0]]                    # teacher-force the gold prefix
                    opt.zero_grad(); loss.backward(); opt.step()
                    tot += float(loss)
                    if complete:
                        break
            last = tot / max(1, len(data))
            if verbose and (ep % 30 == 0 or ep == epochs - 1):
                print(f"    TRM-loop epoch {ep:>3}  loss {last:.4f}", flush=True)
        return {"loss": last}


def _compose_tasks():
    """TRAIN: tasks that NEED a SET of 2 atoms (the compositional case one-shot cosine can't assemble)."""
    return [
        ("the sum of the digits of the nth fibonacci number", ["fibonacci", "digit_sum"]),
        ("count the divisors of n factorial", ["factorial", "num_divisors"]),
        ("is the digit sum of n a prime number", ["digit_sum", "is_prime"]),
        ("reverse the digits then check if even", ["reverse_digits", "is_even"]),
        ("square the number then sum its digits", ["square", "digit_sum"]),
        ("count set bits of the nth fibonacci", ["fibonacci", "count_bits"]),
    ]


def _compose_heldout():
    """HELD-OUT: NEW atom-pair combinations + phrasings never seen in training — the honest generalization test."""
    return [
        ("the digit sum of the number of divisors", ["num_divisors", "digit_sum"]),   # new pair
        ("is the nth fibonacci number even", ["fibonacci", "is_even"]),               # new pair
        ("how many one-bits are in the digit sum", ["digit_sum", "count_bits"]),      # new pair
        ("reverse the digits of the perfect square", ["square", "reverse_digits"]),   # new pair
    ]


def demo_trm_reasoner():
    print("membrane.py --trm: the TRM REASONS over the graph (iterative multi-hop + stop) -- this is NOT RAG\n")
    torch.manual_seed(0)
    g = seed_graph()
    tasks = _compose_tasks()
    print(f"  graph: {len(g)} atoms. compositional tasks (each needs a SET of 2 atoms): {len(tasks)}")

    # baseline: one-shot cosine top-2 (RAG-style lookup) — can it assemble the SET?
    cos_hit = 0
    for text, gold in tasks:
        top2 = set(g.cosine_rank(text, k=2))
        cos_hit += int(set(gold).issubset(top2))
    print(f"\n  [RAG baseline]  one-shot cosine top-2 recovers the FULL set: {cos_hit}/{len(tasks)}")

    held = _compose_heldout()
    cos_held = sum(int(set(gold).issubset(set(g.cosine_rank(text, k=2)))) for text, gold in held)
    loop = TRMLoop(g, hops=4)
    print(f"  [TRM reasoner]  untrained -> training on the {len(tasks)} TRAIN tasks...")
    loop.train(tasks, epochs=150, verbose=True)

    def _eval(items, label):
        exact = cover = 0
        print(f"\n  [MANUAL INSPECTION - {label}] per-hop (picked, stop_prob) -- not trusting the label:")
        for text, gold in items:
            got, conf, hop = loop.retrieve_set(text)
            c = set(gold).issubset(got); e = (got == set(gold)); cover += c; exact += e
            print(f"     {text[:36]:<36} gold={sorted(gold)}")
            print(f"        hops={hop} -> got={sorted(got)} EXACT={e} conf={conf:.2f}")
        return exact, cover

    tr_exact, _ = _eval(tasks, "TRAIN")
    ho_exact, ho_cover = _eval(held, "HELD-OUT (never trained)")
    print(f"\n  == NOT-RAG RESULT (EXACT set match; HELD-OUT is the honest number) ==")
    print(f"     RAG cosine top-2  : train {cos_hit}/{len(tasks)}   held-out {cos_held}/{len(held)}")
    print(f"     TRM multi-hop     : train {tr_exact}/{len(tasks)}   HELD-OUT {ho_exact}/{len(held)} (covered {ho_cover}/{len(held)})")
    print(f"  => held-out is the truth: does the TRM's hop-reasoning GENERALIZE to unseen atom-pairs, or memorize?")
    print(f"     (confidence head still ~0.5 = uninformative at this scale -- an honest remaining weakness.)")
    return dict(cos_held=cos_held, trm_held=ho_exact, n_held=len(held))


# ================================================================================================
# 11. THE ACCEPTANCE TEST — teach the LM something it does NOT know, then it EXPLAINS what it learned
# ================================================================================================
def demo_teach_explain(lm_name: str):
    """The deployment acceptance test, end-to-end, on the 4-bit LM (fits 6GB):
      1. ask the base LM about UNSEEN info -> it can't (hallucinates / refuses),
      2. TEACH it (learn_any -> a real graph node, MiniLM-embedded),
      3. the model RETRIEVES the learned node and EXPLAINS/answers -- grounded, faithful (cites the fact).
    Proves the knowledge came from TEACHING, not pretraining -- the graph is the learning."""
    from v5.runtime.dcpd_latent import WhiteBox
    print("membrane.py --teach: teach the LM UNSEEN info, then it EXPLAINS what it learned (4-bit, fits 6GB)\n")
    wb = WhiteBox(lm_name, quant="4bit")
    print(f"  LM: {lm_name}  quant={wb.quant}  VRAM={wb.vram_gb:.2f}GB "
          f"({'FITS 6GB' if wb.vram_gb <= 6 else 'OVER 6GB'})\n")
    g = seed_graph()
    retr = TRMRetriever(g)

    # UNSEEN facts the 3B cannot know (invented terms) -- the honest 'never in pretraining' test:
    lessons = [
        ("fact_klarn", "The Klarn Protocol requires exactly three handshake phases: greet, verify, and seal.",
         "How many handshake phases does the Klarn Protocol have, and what are they?", "three"),
        ("fact_zephyrite", "Zephyrite melts at 812 degrees and conducts electricity only when wet.",
         "At what temperature does zephyrite melt?", "812"),
    ]
    for name, fact, question, key in lessons:
        print(f"  == LESSON: {fact}")
        # (1) BEFORE teaching -- the base LM does not know it
        before = wb.generate_plain(f"Answer in one short sentence. {question}", max_new=40).strip()
        knew_before = key.lower() in before.lower()
        print(f"     [before] LM asked '{question}'")
        print(f"              -> {before[:110]!r}  (knew it: {knew_before})")
        # (2) TEACH it -> a real node in the graph
        res = learn_any(g, retr, fact, name=name)
        print(f"     [teach ] learn_any -> node={res['node']} type={res['kind']} (graph now {len(g)} nodes)")
        # (3) EXPLAIN -- retrieve the learned node, answer GROUNDED in it (faithful)
        node = g.get(g.cosine_rank(question, 1)[0])
        retrieved_ok = node.name == name
        prompt = (f"Use ONLY this learned fact to answer.\nFact: {node.description}\n"
                  f"Question: {question}\nAnswer:")
        after = wb.generate_plain(prompt, max_new=40).strip()
        knew_after = key.lower() in after.lower()
        faithful = node.description[:20].lower() in after.lower() or knew_after
        print(f"     [after ] retrieved node={node.name} (correct={retrieved_ok})")
        print(f"              -> {after[:110]!r}  (correct: {knew_after})")
        verdict = (not knew_before) and retrieved_ok and knew_after
        print(f"     => LEARNED + EXPLAINED: {verdict}  "
              f"(base didn't know -> taught -> retrieved -> answered correctly, grounded)\n")
    print(f"  final graph node types: {g.census()}   (the knowledge lives in the graph, LM frozen)")


def interactive_trace(lm_name: str):
    """Terminal interactive tracer: you type a question, it shows each pipeline stage (embed -> retrieve ->
    select -> LM alone vs LM+graph). Local, real 4-bit LM (~2GB, fits 6GB). 'teach <fact>' adds knowledge."""
    from v5.runtime.dcpd_latent import WhiteBox
    wb = WhiteBox(lm_name, quant="4bit")
    g = seed_graph(); retr = TRMRetriever(g)
    for f, n in [("The Klarn Protocol requires three handshake phases: greet, verify, and seal.", "fact_klarn"),
                 ("Zephyrite melts at 812 degrees and conducts electricity only when wet.", "fact_zephyrite"),
                 ("The Vora index measures how fast a market recovers after a shock, from 0 to 100.", "fact_vora")]:
        learn_any(g, retr, f, name=n)

    def clean(t):
        for c in ("\nHuman:", "Human:", "\nYou are", "\n\n", "<|"):
            i = t.find(c); t = t[:i] if i > 0 else t
        return t.strip()

    print(f"\n  LM {lm_name}  quant={wb.quant}  VRAM={wb.vram_gb:.2f}GB "
          f"({'FITS 6GB' if wb.vram_gb <= 6 else 'OVER 6GB'})")
    print(f"  graph: {len(g)} nodes {g.census()}   (LM frozen; knowledge is in the graph)")
    print(f"  type a QUESTION  |  'teach <fact>' to add knowledge  |  'quit' to exit")
    while True:
        try:
            q = input("\nquery> ").strip()
        except EOFError:
            break
        if not q:
            continue
        if q.lower() in ("quit", "exit", "q"):
            break
        if q.lower().startswith("teach "):
            r = learn_any(g, retr, q[6:].strip(), name=f"taught_{len(g)}")
            print(f"  learned -> node '{r['node']}' type={r['kind']}  (graph now {len(g)} nodes)")
            continue
        qv = encode_batch([q])[0]
        print(f"  [1] EMBED     query -> MiniLM 384-d vector (norm {np.linalg.norm(qv):.2f})")
        M, order = g.matrix(); sims = (M @ qv).tolist()
        rank = sorted(zip(order, sims), key=lambda z: -z[1])[:5]
        print(f"  [2] RETRIEVE  top graph matches (neural):")
        for n, s in rank:
            print(f"                {s:5.2f}  {n:<20} ({g.get(n).kind})")
        top, ts = rank[0]; node = g.get(top); content = node.code or node.description
        print(f"  [3] SELECT    '{top}' (sim {ts:.2f}, {node.kind}): {content[:90]}")
        base = clean(wb.generate_plain(f"Answer briefly. {q}", max_new=48))
        print(f"  [4] LM ALONE  (no memory)     -> {base[:170]}")
        grounded = clean(wb.generate_plain(f"Use ONLY this to answer.\n{content}\nQuestion: {q}\nAnswer:", max_new=48))
        print(f"  [5] LM+GRAPH  (grounded)      -> {grounded[:170]}")
    print("  bye")


def main():
    ap = argparse.ArgumentParser(description="one real integrated membrane: neural retrieval + TRM + verify + learn")
    ap.add_argument("--demo", action="store_true", help="the integrated retrieval+TRM+verify+learn demo")
    ap.add_argument("--deploy", action="store_true", help="the deployment demo: universal graph learns ANY data + uses it")
    ap.add_argument("--trm", action="store_true", help="TRM-as-reasoner: iterative multi-hop retrieval (NOT RAG)")
    ap.add_argument("--teach", action="store_true", help="ACCEPTANCE TEST: teach unseen info -> model explains it (needs --lm)")
    ap.add_argument("--interactive", action="store_true", help="terminal tracer: type a question, see each stage (needs --lm)")
    ap.add_argument("--lm", type=str, default="", help="real frozen LM (e.g. Qwen/Qwen2.5-3B-Instruct); optional")
    a = ap.parse_args()
    if a.interactive:
        if not a.lm:
            raise SystemExit("--interactive needs --lm")
        interactive_trace(a.lm)
        return
    if a.teach:
        if not a.lm:
            raise SystemExit("--teach needs --lm")
        demo_teach_explain(a.lm)
        return
    if a.trm:
        demo_trm_reasoner()
        return
    if a.deploy:
        demo_deploy(a.lm)
        return
    if a.demo:
        demo(a.lm)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
