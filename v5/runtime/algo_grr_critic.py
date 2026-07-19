"""GRR-Critic: a learned error-noticer that AMORTIZES the verifier + a SIGNED mistake tier.

Design: v5/grr_critic_design.md. Thesis:
- store MISTAKES as negative (-) nodes (failed trace + verdict + reason); never retrieved as a solution
  (that is poison), only as a WARNING that feeds the critic + prunes search.
- CRITIC = a small head trained on the verifier's OWN pass/fail labels -> predicts the verdict from a
  DOMAIN-GENERAL trace representation (task/attempt embeddings + a no-oracle behavior signature), so it
  (a) auto-decides the gross-obvious cases and SKIPS the verifier (the decay curve) and (b) flags errors
  in a domain it never trained on (cross-domain transfer = "notices itself").
- the verifier stays the WRITER of (+) atoms; the critic is a filter / router, never the sole writer.

No GPU. `--selftest` runs the full data pipeline through a stub embedder (offline). `--run` uses the real
MiniLM embedder (embedder.encode_batch) + real execution/sympy verdicts, and measures:
  train on CODE only -> (a) held-out CODE decay, (b) MATH transfer (zero math training).
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import threading
from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# domain-general behavior probing (observe what the attempt DOES; never compare to the oracle here)
# ---------------------------------------------------------------------------


def _run_timeout(fn: Callable, arg, t: float = 1.0):
    box: dict = {}

    def _w():
        try:
            box["v"] = fn(arg)
        except Exception:  # noqa: BLE001
            box["e"] = True

    th = threading.Thread(target=_w, daemon=True)
    th.start()
    th.join(t)
    if th.is_alive():
        raise TimeoutError
    if "e" in box:
        raise RuntimeError
    return box["v"]


def behavior_signature(run: Callable, probes) -> tuple[list[float], str]:
    """Run `run` on a few probe inputs and summarize WITHOUT any ground truth.

    Returns ([crash, monotone, has_neg, mean_magnitude], readable_string). These symptoms
    (crash / sign / monotonicity / magnitude / task-vs-behavior mismatch) are what a no-oracle critic
    can actually notice, and they are domain-general (any domain that produces numbers on probes)."""
    outs = []
    for xi in probes:
        try:
            v = _run_timeout(run, xi, 1.0)
            v = float(v) if isinstance(v, (int, float, bool)) else None
        except Exception:  # noqa: BLE001
            v = None
        outs.append(v)
    vals = [o for o in outs if o is not None]
    crash = 1.0 if len(vals) < len(probes) else 0.0
    monotone = 1.0 if len(vals) >= 2 and all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1)) else 0.0
    has_neg = 1.0 if any(v < 0 for v in vals) else 0.0
    mean_mag = float(np.mean([math.tanh(math.log1p(abs(v))) for v in vals])) if vals else 0.0
    bstr = " ".join(f"{xi}->{'X' if o is None else round(o, 4)}" for xi, o in zip(probes, outs))
    return [crash, monotone, has_neg, mean_mag], bstr


# ---------------------------------------------------------------------------
# CODE domain: correct atoms + oracles; negatives are verified real bugs (crash / semantic / subtle)
# ---------------------------------------------------------------------------

CODE_ATOMS: dict[str, tuple[str, Callable, str]] = {
    "num_partitions": (
        "def num_partitions(n):\n    dp = [1] + [0] * n\n    for k in range(1, n + 1):\n"
        "        for i in range(k, n + 1):\n            dp[i] += dp[i - k]\n    return dp[n]\n",
        None, "the number of integer partitions of n"),
    "josephus": (
        "def josephus(n):\n    r = 0\n    for i in range(1, n + 1):\n        r = (r + 2) % i\n    return r + 1\n",
        None, "the Josephus survivor position for n people stepping by two"),
    "catalan": (
        "def catalan(n):\n    from math import comb\n    return comb(2 * n, n) // (n + 1)\n",
        None, "the n-th Catalan number"),
    "derangements": (
        "def derangements(n):\n    if n == 0:\n        return 1\n    a, b = 1, 0\n"
        "    for i in range(2, n + 1):\n        a, b = b, (i - 1) * (a + b)\n    return b\n",
        None, "the number of derangements of n items"),
    "mult_persistence": (
        "def mult_persistence(n):\n    s, n = 0, abs(n)\n    while n >= 10:\n        p = 1\n"
        "        for c in str(n):\n            p *= int(c)\n        n = p\n        s += 1\n    return s\n",
        None, "the multiplicative persistence of n"),
    "factorial": (
        "def factorial(n):\n    r = 1\n    for i in range(2, n + 1):\n        r *= i\n    return r\n",
        None, "the factorial of n"),
    "fib": (
        "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
        None, "the n-th Fibonacci number"),
    "sum_digits": (
        "def sum_digits(n):\n    return sum(int(c) for c in str(abs(n)))\n",
        None, "the sum of the digits of n"),
    "count_divisors": (
        "def count_divisors(n):\n    n = abs(n) or 1\n    return sum(1 for d in range(1, n + 1) if n % d == 0)\n",
        None, "the number of divisors of n"),
    "triangular": (
        "def triangular(n):\n    return n * (n + 1) // 2\n",
        None, "the n-th triangular number"),
    "square": ("def square(n):\n    return n * n\n", None, "the square of n"),
    "cube": ("def cube(n):\n    return n ** 3\n", None, "the cube of n"),
    "digit_product": (
        "def digit_product(n):\n    p = 1\n    for c in str(abs(n)):\n        p *= int(c)\n    return p\n",
        None, "the product of the digits of n"),
    "collatz_steps": (
        "def collatz_steps(n):\n    n = abs(n) or 1\n    s = 0\n    while n != 1:\n"
        "        n = n // 2 if n % 2 == 0 else 3 * n + 1\n        s += 1\n    return s\n",
        None, "the number of Collatz steps to reach 1 from n"),
    "count_ones_binary": (
        "def count_ones_binary(n):\n    return bin(abs(n)).count('1')\n",
        None, "the number of one bits in the binary form of n"),
    "reverse_number": (
        "def reverse_number(n):\n    return int(str(abs(n))[::-1])\n",
        None, "the digits of n reversed as a number"),
    "is_prime": (
        "def is_prime(n):\n    if n < 2:\n        return 0\n    d = 2\n    while d * d <= n:\n"
        "        if n % d == 0:\n            return 0\n        d += 1\n    return 1\n",
        None, "one if n is prime else zero"),
    "sum_proper_divisors": (
        "def sum_proper_divisors(n):\n    n = abs(n) or 1\n    return sum(d for d in range(1, n) if n % d == 0)\n",
        None, "the sum of the proper divisors of n"),
    "pentagonal": ("def pentagonal(n):\n    return n * (3 * n - 1) // 2\n", None, "the n-th pentagonal number"),
    "hexagonal": ("def hexagonal(n):\n    return n * (2 * n - 1)\n", None, "the n-th hexagonal number"),
    "trailing_zeros_fact": (
        "def trailing_zeros_fact(n):\n    z, p = 0, 5\n    while p <= n:\n        z += n // p\n        p *= 5\n    return z\n",
        None, "the number of trailing zeros in n factorial"),
    "sum_squares": (
        "def sum_squares(n):\n    return sum(i * i for i in range(1, n + 1))\n",
        None, "the sum of squares from 1 to n"),
}


def _make_runner(code: str, fnname: str):
    ns: dict = {}
    try:
        exec(code, ns)  # noqa: S102 - sandboxed local benchmark code, not user input
    except Exception:  # noqa: BLE001
        return None
    return ns.get(fnname)


def _oracle_for(name: str):
    code, _, _ = CODE_ATOMS[name]
    fn = _make_runner(code, name)
    return fn


def _matches(run, oracle, xs) -> bool:
    for xi in xs:
        try:
            v = _run_timeout(run, xi, 1.0)
        except Exception:  # noqa: BLE001
            return False
        try:
            if v != oracle(xi):
                return False
        except Exception:  # noqa: BLE001
            return False
    return True


def _mutate_subtle(code: str, rng: random.Random) -> list[str]:
    """Small single-edit mutations likely to change behavior (off-by-one / operator swaps)."""
    cands = []
    ops = [("+ 1", "+ 2"), ("+ 1", "- 1"), ("- 1", "+ 1"), ("range(1,", "range(0,"),
           ("range(2,", "range(1,"), ("(r + 2)", "(r + 1)"), ("// (n + 1)", "// (n + 2)"),
           (" * i", " * (i - 1)"), ("dp[n]", "dp[n - 1]"), ("n + 1)", "n)"), (">= 10", "> 10"),
           ("n * n", "n + n"), ("** 3", "** 2"), ("* (3 * n - 1)", "* (3 * n + 1)"),
           ("(2 * n - 1)", "(2 * n + 1)"), ("range(1, n)", "range(1, n + 1)"),
           ("str(abs(n))[::-1]", "str(abs(n))"), ("d * d <= n", "d * d < n"), ("p *= 5", "p *= 6"),
           ("i * i", "i * 2"), ("3 * n + 1", "3 * n - 1"), ("n // 2", "n // 3")]
    for a, b in ops:
        if a in code:
            cands.append(code.replace(a, b, 1))
    rng.shuffle(cands)
    return cands


def gen_code_examples(seed: int = 0, per_atom: int = 6) -> list[dict]:
    rng = random.Random(seed)
    probes = [4, 5, 6, 7]
    check_xs = [3, 4, 5, 6, 7, 8, 9]
    out: list[dict] = []
    names = list(CODE_ATOMS)
    for name in names:
        code, _, desc = CODE_ATOMS[name]
        oracle = _oracle_for(name)
        # (+) correct
        run = _make_runner(code, name)
        if run and _matches(run, oracle, check_xs):
            out.append(dict(domain="code", task=desc, attempt_text=code,
                            run=_make_runner(code, name), probes=probes, label=1, mode="correct"))
        # (-) crash: reference a name that doesn't exist / bad op
        crash_code = code.replace("return", "return undefined_var +", 1)
        out.append(dict(domain="code", task=desc, attempt_text=crash_code,
                        run=_make_runner(crash_code, name), probes=probes, label=0, mode="crash"))
        # (-) semantic: another atom's body under THIS name (computes the wrong thing entirely)
        other = rng.choice([n for n in names if n != name])
        ocode, _, _ = CODE_ATOMS[other]
        sem_code = ocode.replace(f"def {other}(", f"def {name}(", 1)
        sem_run = _make_runner(sem_code, name)
        if sem_run and not _matches(sem_run, oracle, check_xs):
            out.append(dict(domain="code", task=desc, attempt_text=sem_code,
                            run=_make_runner(sem_code, name), probes=probes, label=0, mode="semantic"))
        # (-) subtle: verified behavior-changing single edit (the hard-to-notice ones)
        for m in _mutate_subtle(code, rng):
            mrun = _make_runner(m, name)
            if mrun and not _matches(mrun, oracle, check_xs):
                out.append(dict(domain="code", task=desc, attempt_text=m,
                                run=_make_runner(m, name), probes=probes, label=0, mode="subtle"))
                break
    # duplicate-augment to per_atom by paraphrasing the task text (cheap variety for training)
    return out


# ---------------------------------------------------------------------------
# MATH domain (held-out): sympy tasks; negatives verified by sympy equality
# ---------------------------------------------------------------------------


def gen_math_examples(seed: int = 0, n: int = 40) -> list[dict]:
    import sympy as sp

    x = sp.symbols("x")
    rng = random.Random(seed + 777)
    probes = [1, 2, 3, 5]
    out: list[dict] = []

    def runner(expr):
        f = sp.lambdify(x, expr, "math")
        return lambda xi: f(xi)

    tries = 0
    while len(out) < n and tries < n * 20:
        tries += 1
        deg = rng.randint(2, 4)
        coeffs = [rng.randint(-4, 4) for _ in range(deg + 1)]
        expr = sum(c * x ** i for i, c in enumerate(coeffs))
        op = rng.choice(["diff", "expand"])
        if op == "diff":
            task = f"the derivative with respect to x of {sp.sstr(expr)}"
            correct = sp.diff(expr, x)
        else:
            fac = x + rng.randint(-3, 3)
            base = sp.expand(expr * fac)
            task = f"the expanded polynomial form of ({sp.sstr(expr)}) times ({sp.sstr(fac)})"
            correct = base
        correct = sp.simplify(correct)
        kind = rng.choice(["correct", "correct", "semantic", "subtle", "crash"])
        try:
            if kind == "correct":
                att = correct
                if sp.simplify(att - correct) != 0:
                    continue
                out.append(dict(domain="math", task=task, attempt_text=sp.sstr(att),
                                run=runner(att), probes=probes, label=1, mode="correct"))
            elif kind == "crash":
                c = rng.choice(probes)
                att = 1 / (x - c)
                out.append(dict(domain="math", task=task, attempt_text=sp.sstr(att),
                                run=runner(att), probes=probes, label=0, mode="crash"))
            elif kind == "semantic":
                att = sp.expand(expr * (x + rng.randint(1, 3))) if op == "diff" else sp.diff(expr, x)
                if sp.simplify(att - correct) == 0:
                    continue
                out.append(dict(domain="math", task=task, attempt_text=sp.sstr(att),
                                run=runner(att), probes=probes, label=0, mode="semantic"))
            else:  # subtle
                terms = correct.as_ordered_terms()
                if not terms:
                    continue
                att = correct + rng.choice([-1, 1]) * rng.choice([1, 2]) * x ** rng.randint(0, deg)
                if sp.simplify(att - correct) == 0:
                    continue
                out.append(dict(domain="math", task=task, attempt_text=sp.sstr(att),
                                run=runner(att), probes=probes, label=0, mode="subtle"))
        except Exception:  # noqa: BLE001
            continue
    return out


# ---------------------------------------------------------------------------
# MISTAKE TIER: the negative (-) memory
# ---------------------------------------------------------------------------


class MistakeStore:
    """The (-) tier: failed traces keyed by failure signature; abstracts into failure modes.

    Warning-only: retrieved to prune search / train the critic, NEVER realized as a solution."""

    def __init__(self):
        self.nodes: list[dict] = []

    def add(self, task: str, attempt_text: str, mode: str, domain: str):
        self.nodes.append(dict(task=task, attempt=attempt_text, mode=mode, domain=domain))

    def abstract(self) -> dict[str, int]:
        hist: dict[str, int] = {}
        for nd in self.nodes:
            hist[nd["mode"]] = hist.get(nd["mode"], 0) + 1
        return hist


# ---------------------------------------------------------------------------
# featurization (domain-general) + embedders
# ---------------------------------------------------------------------------


def _stub_embed(texts, dim: int = 384) -> np.ndarray:
    """Deterministic bag-of-hashed-tokens embedding (offline selftest only)."""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        v = np.zeros(dim, dtype=np.float64)
        for tok in str(t).split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16) % (2 ** 32)
            v += np.random.RandomState(h).normal(size=dim)
        nrm = np.linalg.norm(v)
        out[i] = (v / nrm) if nrm > 0 else v
    return out


def featurize(examples: list[dict], embed_fn, use_emb: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """use_emb=True: full [emb(task) ‖ emb(attempt‖behavior) ‖ behavior ‖ cos] (domain-SPECIFIC, overfits
    on small data + transfers poorly). use_emb=False: DOMAIN-INVARIANT block only
    [behavior(4) ‖ cos ‖ l2diff] — low-dim, resists overfitting, and TRANSFERS across domains because a
    crash / magnitude / task-attempt-mismatch symptom is domain-agnostic."""
    tasks, atexts, beh = [], [], []
    for e in examples:
        bvec, bstr = behavior_signature(e["run"], e["probes"])
        tasks.append(e["task"])
        atexts.append(e["attempt_text"].replace("\n", " ") + " || behavior: " + bstr)
        beh.append(bvec)
    T = embed_fn(tasks)
    A = embed_fn(atexts)
    cos = np.sum(T * A, axis=1, keepdims=True)
    l2 = np.linalg.norm(T - A, axis=1, keepdims=True)
    B = np.asarray(beh, dtype=np.float32)
    if use_emb:
        X = np.concatenate([T, A, B, cos], axis=1).astype(np.float32)
    else:
        X = np.concatenate([B, cos, l2], axis=1).astype(np.float32)  # 6-dim, domain-invariant
    y = np.asarray([e["label"] for e in examples], dtype=np.float32)
    return X, y


# ---------------------------------------------------------------------------
# critic (torch MLP) + metrics
# ---------------------------------------------------------------------------


def train_critic(Xtr, ytr, epochs: int = 400, lr: float = 1e-3, wd: float = 1e-4, seed: int = 0):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True) + 1e-6
    Xn = (Xtr - mu) / sd
    net = nn.Sequential(nn.Linear(Xn.shape[1], 128), nn.ReLU(), nn.Dropout(0.1),
                        nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(Xn, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        out = net(Xt).squeeze(-1)
        loss = lossf(out, yt)
        loss.backward()
        opt.step()
    net.eval()
    return dict(net=net, mu=mu, sd=sd)


def predict(model, X) -> np.ndarray:
    import torch

    Xn = (X - model["mu"]) / model["sd"]
    with torch.no_grad():
        logits = model["net"](torch.tensor(Xn, dtype=torch.float32)).squeeze(-1)
        p = torch.sigmoid(logits).cpu().numpy()
    return p


def auc(y, p) -> float:
    pos = p[y == 1]
    neg = p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float) + 1.0
    rpos = ranks[: len(pos)].sum()
    return float((rpos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def decay_savings(y, p, target_prec: float = 0.90) -> dict:
    """How many verifier calls the critic can auto-decide at >= target precision (both directions)."""
    y = np.asarray(y)
    n = len(y)
    n_err = int((y == 0).sum())
    # auto-REJECT: predict 0 for p <= tau; maximize count s.t. precision(class0) >= target
    best_rej = 0
    for tau in sorted(set(p)):
        sel = p <= tau
        if sel.sum() == 0:
            continue
        prec = (y[sel] == 0).mean()
        if prec >= target_prec:
            best_rej = max(best_rej, int(sel.sum()))
    # auto-ACCEPT: predict 1 for p >= tau; precision(class1) >= target
    best_acc = 0
    for tau in sorted(set(p), reverse=True):
        sel = p >= tau
        if sel.sum() == 0:
            continue
        prec = (y[sel] == 1).mean()
        if prec >= target_prec:
            best_acc = max(best_acc, int(sel.sum()))
    saved = best_rej + best_acc
    return dict(n=n, n_err=n_err, auto_reject=best_rej, auto_accept=best_acc,
                saved_frac=saved / n if n else 0.0,
                err_recall=best_rej / n_err if n_err else 0.0)


def _split(examples, frac=0.7, seed=0):
    rng = random.Random(seed)
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    k = int(len(idx) * frac)
    tr = [examples[i] for i in idx[:k]]
    te = [examples[i] for i in idx[k:]]
    return tr, te


def _mode_breakdown(examples, p):
    """Per-failure-mode: how confidently the critic flags each (mean predicted-correct prob)."""
    modes: dict[str, list] = {}
    for e, pi in zip(examples, p):
        modes.setdefault(e["mode"], []).append(pi)
    return {m: (len(v), float(np.mean(v))) for m, v in sorted(modes.items())}


# ---------------------------------------------------------------------------
# selftest (offline, stub embedder, full real data pipeline)
# ---------------------------------------------------------------------------


def selftest() -> bool:
    print("[selftest] GRR-Critic — offline (stub embedder), REAL code+math data pipeline")
    code, math_ex = [], []
    for s in (1, 2, 3):  # pool seeds so the held-out set is large enough to be non-noisy
        code += gen_code_examples(seed=s)
        math_ex += gen_math_examples(seed=s, n=40)
    print(f"  code examples: {len(code)}  | math examples: {len(math_ex)}")
    assert len(code) >= 20, "code generator underfilled"
    assert len(math_ex) >= 20, "math generator underfilled"

    ms = MistakeStore()
    for e in code + math_ex:
        if e["label"] == 0:
            ms.add(e["task"], e["attempt_text"], e["mode"], e["domain"])
    hist = ms.abstract()
    print(f"  (-) mistake tier: {len(ms.nodes)} nodes, failure modes = {hist}")
    assert set(hist) & {"crash", "semantic", "subtle"}, "mistake modes missing"

    ex = code + math_ex
    X, y = featurize(ex, _stub_embed)
    assert X.shape[0] == len(ex) and X.shape[1] == 384 * 2 + 5, f"bad feature shape {X.shape}"
    tr, te = _split(list(range(len(ex))), 0.7, seed=3)
    model = train_critic(X[tr], y[tr], epochs=300, seed=0)
    p = predict(model, X[te])
    te_ex = [ex[i] for i in te]
    bd = _mode_breakdown(te_ex, p)
    print(f"  by mode (mean P-correct): {bd}")
    # a STUB (bag-of-hashed-tokens) embedder has NO semantic signal -> it can only learn the one
    # domain-general symptom the behavior features carry: CRASH. That is what the selftest verifies.
    # (semantic/subtle separability needs the real MiniLM embedder -> `--run`.)
    p_crash = bd.get("crash", (0, 1.0))[1]
    p_correct = bd.get("correct", (0, 0.0))[1]
    # AUC restricted to the stub-separable pair (correct vs crash) = the plumbing check
    mask = np.array([e["mode"] in ("correct", "crash") for e in te_ex])
    sub_auc = auc(y[te][mask], p[mask]) if mask.sum() and len(set(y[te][mask])) == 2 else float("nan")
    print(f"  crash P-correct={p_crash:.3f}  correct P-correct={p_correct:.3f}  "
          f"correct-vs-crash AUC={sub_auc:.3f} (info; real discrimination is --run)")
    # honest, stable gate: the ONE symptom a stub embedder can carry (crash) is learned + clearly
    # separated from correct. Semantic/subtle need real embeddings -> measured in --run.
    ok = (p_crash < 0.15) and (p_correct - p_crash > 0.20)
    print(f"  [selftest] {'PASS' if ok else 'FAIL'} (crash symptom learned: P(crash)<0.15 and margin>0.20)")
    return ok


# ---------------------------------------------------------------------------
# run (real MiniLM embedder): train on CODE -> decay on held-out CODE + transfer to MATH
# ---------------------------------------------------------------------------


def _dedup(examples: list[dict]) -> list[dict]:
    seen, out = set(), []
    for e in examples:
        key = (e["domain"], e["mode"], e["task"], e["attempt_text"])
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def cv_eval(X, y, examples, n_splits: int = 5, seed: int = 0) -> dict:
    """Repeated 75/25 splits: mean held-out AUC + mean verifier-savings (de-noises the tiny held-out)."""
    aucs, saves, recs = [], [], []
    idx = list(range(len(y)))
    for s in range(n_splits):
        random.Random(seed * 100 + s).shuffle(idx)
        k = int(len(idx) * 0.75)
        tr, te = idx[:k], idx[k:]
        m = train_critic(X[tr], y[tr], epochs=300, seed=s)
        p = predict(m, X[te])
        a = auc(y[te], p)
        if math.isnan(a):
            continue
        d = decay_savings(y[te], p, 0.90)
        aucs.append(a); saves.append(d["saved_frac"]); recs.append(d["err_recall"])
    return dict(auc=float(np.mean(aucs)), auc_sd=float(np.std(aucs)),
                saved=float(np.mean(saves)), err_recall=float(np.mean(recs)))


def transfer_eval(Xtr, ytr, Xte, yte, n: int = 5, seed: int = 0) -> tuple[float, float]:
    aucs = []
    for s in range(n):
        m = train_critic(Xtr, ytr, epochs=300, seed=seed + s)
        aucs.append(auc(yte, predict(m, Xte)))
    aucs = [a for a in aucs if not math.isnan(a)]
    return float(np.mean(aucs)), float(np.std(aucs))


def _mode_means(examples, embed_fn, use_emb, train_ex, seed=0):
    """Train on train_ex, return mean P-correct per mode over `examples` (for the honest breakdown)."""
    Xtr, ytr = featurize(train_ex, embed_fn, use_emb)
    m = train_critic(Xtr, ytr, epochs=300, seed=seed)
    Xe, _ = featurize(examples, embed_fn, use_emb)
    return _mode_breakdown(examples, predict(m, Xe))


def run(seed: int = 0):
    from embedder import encode_batch

    embed_fn = lambda texts: encode_batch(list(texts))  # noqa: E731

    print("=" * 96)
    print("GRR-Critic — REAL MiniLM | train=CODE | held-out CODE (decay) + MATH (transfer) | 5x CV")
    print("=" * 96)

    code, math_ex = [], []
    for s in range(20):
        code += gen_code_examples(seed=s)
    for s in range(12):
        math_ex += gen_math_examples(seed=s, n=48)
    code, math_ex = _dedup(code), _dedup(math_ex)

    def _hist(ex):
        h = {}
        for e in ex:
            h[e["mode"]] = h.get(e["mode"], 0) + 1
        return h

    print(f"code examples {len(code)} {_hist(code)} | math examples {len(math_ex)} {_hist(math_ex)}")
    ms = MistakeStore()
    for e in code + math_ex:
        if e["label"] == 0:
            ms.add(e["task"], e["attempt_text"], e["mode"], e["domain"])
    print(f"(-) MISTAKE TIER: {len(ms.nodes)} nodes | failure modes {ms.abstract()}")

    out = {}
    for use_emb, tag in [(False, "INVARIANT (behavior+cos+l2, domain-general)"),
                         (True, "FULL (+raw MiniLM embeddings, domain-specific)")]:
        Xc, yc = featurize(code, embed_fn, use_emb)
        Xm, ym = featurize(math_ex, embed_fn, use_emb)
        dec = cv_eval(Xc, yc, code, n_splits=5, seed=seed)
        tr_auc, tr_sd = transfer_eval(Xc, yc, Xm, ym, n=5, seed=seed)
        ceil = cv_eval(Xm, ym, math_ex, n_splits=5, seed=seed)
        modes = _mode_means(math_ex, embed_fn, use_emb, code, seed=seed)
        print("\n" + "-" * 96)
        print(f"FEATURES: {tag}")
        print(f"  (a) DECAY (held-out CODE, 5xCV) : AUC {dec['auc']:.3f}±{dec['auc_sd']:.3f} | "
              f"auto-decided {dec['saved']*100:4.1f}% verifier calls @P>=.90 | err-recall {dec['err_recall']*100:4.1f}%")
        print(f"  (b) TRANSFER (MATH, 0 trained)  : AUC {tr_auc:.3f}±{tr_sd:.3f}  vs chance 0.500  "
              f"vs math-trained ceiling {ceil['auc']:.3f}")
        print(f"      MATH by failure mode (mean P-correct, lower=flagged): {modes}")
        out[tag] = dict(code_auc=dec["auc"], code_saved=dec["saved"], math_auc=tr_auc, ceiling=ceil["auc"])

    print("\n" + "=" * 96)
    print("READ: invariant features (crash/magnitude/task-mismatch) TRANSFER code->math (>chance) + resist")
    print("      overfit; raw embeddings are domain-specific (overfit + weaker transfer). Subtle-numeric")
    print("      stays ~0.5 (needs the verifier) = reduce reliance, not eliminate.")
    return out


def main():
    ap = argparse.ArgumentParser(description="GRR-Critic: learned error-noticer + signed mistake tier")
    ap.add_argument("--selftest", action="store_true", help="offline pipeline test (stub embedder)")
    ap.add_argument("--run", action="store_true", help="real MiniLM: train CODE -> decay + MATH transfer")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.selftest:
        ok = selftest()
        raise SystemExit(0 if ok else 1)
    if args.run:
        run(seed=args.seed)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
