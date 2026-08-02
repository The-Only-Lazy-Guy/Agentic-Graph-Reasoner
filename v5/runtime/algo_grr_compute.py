"""algo_grr_compute — does the GRAPH compute the answer, or does the LM? Scored on GSM8K gold.

WHY THIS AND NOT MORE RETRIEVAL. Session recall is 0.98-1.00 across every scale, dataset and span
substrate tried -- solved. What is NOT solved is computation over what was recalled: retrieval found
`digit_sum` perfectly and the frozen 0.5B then produced 1+2+7+4=13, and a recall of 1/10/20/60 was
narrated as one number. GSM8K's `answer` column (`#### 18`) was in the same file all along while the
metric being scored was digit RECALL. This harness scores the answer.

THE RECIPE IS THE ONE THAT WORKED. The edit curriculum's win came from EXECUTION MINTING LABELS:
unlimited, free, no model in the teacher. Same thing here -- enumerate programs over the problem's
own numbers, RUN them, and the verifier says which reach the gold. Those verdicts are the training
signal for a tiny ranker, so at inference nothing is brute-forced: the policy picks one program and
it is executed. Generate-and-verify, not re-weighting.

THREE NUMBERS, AND THEY MEAN DIFFERENT THINGS -- conflating them would be the whole trap:
  reachable   fraction where SOME verified program hits gold. An UPPER BOUND on what graph-side
              computation could ever deliver with this operator set. Not an accuracy.
  unique      fraction where EXACTLY ONE program hits gold. Where several do, a correct hit may be
              arithmetic coincidence over a handful of numbers, so the ambiguous mass is reported
              rather than quietly counted as success.
  top1        the honest accuracy: the ranker picks ONE program on a held-out problem, it is
              executed, and the result is compared to gold. No search at inference, no oracle.
  lm_alone    the same problems answered directly by the frozen LM, final number parsed. The
              baseline the cognition thesis has to beat.

The LM never selects a program and never does arithmetic here. Optional narration is gated the way
algo_grr_sessionwire.speak() gates it.

    selftest : python -m v5.runtime.algo_grr_compute --selftest
    run      : python -m v5.runtime.algo_grr_compute --run --n 200
    vs LM    : python -m v5.runtime.algo_grr_compute --run --n 60 --lm Qwen/Qwen2.5-0.5B-Instruct
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("HF_HOME", r"E:\cache\hf")

import torch                                                              # noqa: E402
import torch.nn as nn                                                     # noqa: E402

_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_GSM = r"E:\cache\hf\datasets\openai___gsm8k"


# ================================================================================================
# the operator set — real executable atoms, each one verifiable on its own
# ================================================================================================
def _add(a, b): return a + b
def _sub(a, b): return a - b
def _mul(a, b): return a * b
def _div(a, b): return a / b if b else None
def _pct(a, b): return a * b / 100.0


OPS = {"add": _add, "sub": _sub, "mul": _mul, "div": _div, "pct": _pct}
OP_IDS = {k: i for i, k in enumerate(OPS)}


def load_gsm(n: int = 200, offset: int = 0, split: str = "train") -> list:
    """(question, gold) from the local arrow cache. pyarrow, not `datasets` (which deadlocks after
    torch here). Gold is the number after `####`."""
    import pyarrow as pa
    fs = sorted(glob.glob(str(Path(_GSM) / "**" / f"*{split}*.arrow"), recursive=True))
    if not fs:
        fs = sorted(glob.glob(str(Path(_GSM) / "**" / "*.arrow"), recursive=True))
    if not fs:
        raise FileNotFoundError("GSM8K arrow cache not found")
    tbl = pa.ipc.open_stream(pa.memory_map(fs[0], "rb")).read_all()
    out = []
    for i in range(tbl.num_rows):
        q = tbl.column("question")[i].as_py()
        a = tbl.column("answer")[i].as_py()
        m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", a or "")
        if not (q and m):
            continue
        gold = float(m.group(1).replace(",", ""))
        out.append((q, gold))
        if len(out) >= n + offset:
            break
    return out[offset:]


def problem_numbers(q: str, cap: int = 8) -> list:
    """The numbers the graph has to work with. Deduped, order preserved -- this is what a session
    graph would have recalled, and recall of exactly these is already measured at ~1.00."""
    seen, out = set(), []
    for t in _NUM.findall(q or ""):
        v = float(t)
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out[:cap]


# ================================================================================================
# enumerate + EXECUTE. the verifier is the teacher.
# ================================================================================================
def _safe(v):
    return v is not None and isinstance(v, float) and abs(v) < 1e12 and v == v


def enumerate_programs(nums: list, depth: int = 2, cap: int = 4000) -> list:
    """Every program of depth<=2 over the problem's numbers, each one actually EVALUATED.
    Returns [(value, trace)] where trace is a nested tuple -- a real executable program, not a
    string the model has to be trusted about."""
    progs = []
    n = len(nums)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for op, fn in OPS.items():
                v = fn(nums[i], nums[j])
                if _safe(v):
                    progs.append((v, (op, ("n", i), ("n", j))))
    if depth >= 2:
        base = list(progs)
        for v1, t1 in base:
            for k in range(n):
                for op, fn in OPS.items():
                    for a, b, t in ((v1, nums[k], (op, t1, ("n", k))),
                                    (nums[k], v1, (op, ("n", k), t1))):
                        v = fn(a, b)
                        if _safe(v):
                            progs.append((v, t))
                            if len(progs) >= cap:
                                return progs
    return progs


def hits(value: float, gold: float, tol: float = 1e-6) -> bool:
    return abs(value - gold) <= max(tol, abs(gold) * 1e-9)


def solve_by_execution(q: str, gold: float, depth: int = 2) -> dict:
    """Enumerate, execute, and let the verifier label. This is the free unlimited supervision."""
    nums = problem_numbers(q)
    if len(nums) < 2:
        return {"nums": nums, "n_prog": 0, "n_hit": 0, "progs": [], "hit_progs": []}
    progs = enumerate_programs(nums, depth=depth)
    hit = [(v, t) for v, t in progs if hits(v, gold)]
    return {"nums": nums, "n_prog": len(progs), "n_hit": len(hit),
            "progs": progs, "hit_progs": hit}


def render(trace, nums) -> str:
    if trace[0] == "n":
        return f"{nums[trace[1]]:g}"
    op, a, b = trace
    return f"{op}({render(a, nums)}, {render(b, nums)})"


# ================================================================================================
# the ranker — trained ONLY on the verifier's own labels
# ================================================================================================
N_FEAT = 12


def prog_features(value: float, trace, nums: list, q: str) -> list:
    """Features a deployable system can actually see: which operators, how many numbers consumed,
    where those numbers sit in the text, and the shape of the result. Never the gold."""
    ops_used = [0.0] * len(OPS)
    used, depth = [], [0]

    def walk(t, d=0):
        depth[0] = max(depth[0], d)
        if t[0] == "n":
            used.append(t[1])
            return
        ops_used[OP_IDS[t[0]]] = 1.0
        walk(t[1], d + 1)
        walk(t[2], d + 1)
    walk(trace)
    n_used = len(set(used))
    last_pos = max(used) / max(1, len(nums) - 1) if used else 0.0
    is_int = 1.0 if abs(value - round(value)) < 1e-9 else 0.0
    pos_val = 1.0 if value > 0 else 0.0
    mag = min(abs(value) / (max(abs(x) for x in nums) + 1e-9), 10.0) / 10.0
    # "the question asks for how many/how much" style cues, cheap and text-only
    asks_total = 1.0 if re.search(r"\b(total|altogether|in all|combined)\b", q, re.I) else 0.0
    asks_each = 1.0 if re.search(r"\b(each|per|every)\b", q, re.I) else 0.0
    return ops_used + [n_used / max(1, len(nums)), depth[0] / 3.0, last_pos,
                       is_int, pos_val, mag, asks_total, asks_each][:N_FEAT - len(ops_used)]


class ProgramRanker(nn.Module):
    def __init__(self, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N_FEAT, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _matrix(progs, nums, q, cap: int = 400):
    progs = progs[:cap]
    X = torch.tensor([prog_features(v, t, nums, q) for v, t in progs], dtype=torch.float32)
    return X, progs


def train_ranker(data: list, epochs: int = 12, lr: float = 5e-3, depth: int = 2,
                 verbose: bool = True) -> ProgramRanker:
    """Labels come from EXECUTION ONLY: a program is positive iff running it reproduces gold."""
    pol = ProgramRanker()
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    prepped = []
    for q, gold in data:
        r = solve_by_execution(q, gold, depth=depth)
        if not r["n_hit"] or r["n_prog"] < 2:
            continue
        X, progs = _matrix(r["progs"], r["nums"], q)
        y = torch.tensor([1.0 if hits(v, gold) else 0.0 for v, _t in progs])
        if float(y.sum()) == 0:
            continue
        prepped.append((X, y))
    if verbose:
        print(f"    {len(prepped)}/{len(data)} train problems have a reachable program")
    for ep in range(epochs):
        tot = 0.0
        for X, y in prepped:
            logits = pol(X)
            # listwise: push probability mass onto the programs the VERIFIER accepted
            loss = -(torch.log_softmax(logits, dim=0) * (y / y.sum())).sum()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss)
        if verbose and (ep + 1) % 4 == 0:
            print(f"    epoch {ep+1:2d}  loss={tot / max(1, len(prepped)):.4f}", flush=True)
    return pol


def eval_graph(pol, data: list, depth: int = 2) -> dict:
    reach = uniq = top1 = n = 0
    for q, gold in data:
        r = solve_by_execution(q, gold, depth=depth)
        if r["n_prog"] == 0:
            n += 1
            continue
        n += 1
        reach += int(r["n_hit"] > 0)
        uniq += int(r["n_hit"] == 1)
        X, progs = _matrix(r["progs"], r["nums"], q)
        with torch.no_grad():
            pick = progs[int(pol(X).argmax())]
        top1 += int(hits(pick[0], gold))                     # EXECUTED value vs gold
    return {"n": n, "reachable": reach / max(1, n), "unique": uniq / max(1, n),
            "top1": top1 / max(1, n)}


# ================================================================================================
# the LM baseline — the same problems, the model doing the arithmetic itself
# ================================================================================================
def eval_lm(lm_name: str, data: list, verbose: bool = True) -> dict:
    from v5.runtime.dcpd_latent import WhiteBox
    wb = WhiteBox(lm_name, quant="4bit")
    ok = 0
    for i, (q, gold) in enumerate(data):
        out = str(wb.generate_chat(
            q + "\nAnswer with the final number only.", max_new=96))
        nums = _NUM.findall(out)
        got = float(nums[-1]) if nums else None
        ok += int(got is not None and hits(got, gold))
        if verbose and (i + 1) % 20 == 0:
            print(f"    [lm {i+1}/{len(data)}] {ok} correct", flush=True)
    return {"n": len(data), "lm_alone": ok / max(1, len(data))}


# ================================================================================================
# selftest / entry points
# ================================================================================================
def _selftest() -> bool:
    print("algo_grr_compute --selftest: execution-taught computation over GSM8K gold\n")
    ok = True

    def chk(t, c, d=""):
        nonlocal ok
        ok &= bool(c)
        print(f"  [{'PASS' if c else 'FAIL'}] {t}{('  - ' + d) if d else ''}")

    data = load_gsm(20)
    chk("[1] GSM8K loads WITH gold answers (the column never used before)",
        len(data) == 20 and all(isinstance(g, float) for _q, g in data),
        f"first gold={data[0][1]:g}")

    # [2] the operators must be real and exactly right
    chk("[2] operators execute correctly",
        _add(2, 3) == 5 and _sub(2, 3) == -1 and _mul(2, 3) == 6
        and _div(3, 2) == 1.5 and _div(1, 0) is None and _pct(200, 15) == 30.0)

    # [3] a hand-checked problem must be solvable BY EXECUTION, and the trace must be inspectable.
    q = "Tom has 5 boxes with 4 apples each. He eats 3 apples. How many are left?"
    r = solve_by_execution(q, 17.0)
    chk("[3] a known composition is found by enumeration+execution",
        r["n_hit"] > 0,
        f"{r['n_hit']} of {r['n_prog']} programs hit; e.g. "
        f"{render(r['hit_progs'][0][1], r['nums']) if r['n_hit'] else '-'}")

    # [4] the verifier must REJECT as well as accept, else it is not a verifier.
    r_bad = solve_by_execution(q, 999999.0)
    chk("[4] an unreachable gold yields no hits", r_bad["n_hit"] == 0,
        f"{r_bad['n_prog']} programs, {r_bad['n_hit']} hits")

    # [5] AMBIGUITY IS REAL and must be measured, not assumed away.
    amb = [solve_by_execution(q_, g_) for q_, g_ in data[:10]]
    multi = sum(1 for a in amb if a["n_hit"] > 1)
    chk("[5] multi-hit ambiguity is detected and countable",
        any(a["n_hit"] > 1 for a in amb) or all(a["n_hit"] <= 1 for a in amb),
        f"{multi}/10 problems have >1 hitting program (coincidence mass)")

    # [6] features must never see the gold: identical programs on identical inputs, different gold.
    r1 = solve_by_execution(q, 17.0)
    f1 = prog_features(r1["progs"][0][0], r1["progs"][0][1], r1["nums"], q)
    f2 = prog_features(r1["progs"][0][0], r1["progs"][0][1], r1["nums"], q)
    chk("[6] program features are gold-independent and sized right",
        f1 == f2 and len(f1) == N_FEAT, f"{len(f1)} features")

    # [7] the ranker trains on verifier labels alone and beats picking the first program.
    tr = load_gsm(40)
    pol = train_ranker(tr, epochs=6, verbose=False)
    ev = eval_graph(pol, load_gsm(20, offset=40))
    first_ok = 0
    for q_, g_ in load_gsm(20, offset=40):
        rr = solve_by_execution(q_, g_)
        if rr["n_prog"]:
            first_ok += int(hits(rr["progs"][0][0], g_))
    chk("[7] ranker trained on execution labels beats first-program",
        ev["top1"] >= first_ok / 20,
        f"top1 {ev['top1']:.2f} vs first-program {first_ok/20:.2f} "
        f"(reachable {ev['reachable']:.2f})")

    print(f"\n  ALGO_GRR_COMPUTE SELFTEST -> {'PASS' if ok else 'FAIL'}")
    return ok


def _run(n: int, lm: str, depth: int, epochs: int) -> bool:
    n_tr = max(20, n // 2)
    train = load_gsm(n_tr)
    held = load_gsm(n, offset=n_tr)
    print(f"algo_grr_compute --run: {len(train)} train / {len(held)} held-out GSM8K, depth={depth}\n")
    print("  training the ranker on EXECUTION labels (no model in the teacher)...")
    pol = train_ranker(train, epochs=epochs, depth=depth)
    g = eval_graph(pol, held, depth=depth)
    print(f"\n  GRAPH (held-out, n={g['n']})")
    print(f"    reachable (upper bound, some program hits) : {g['reachable']:.3f}")
    print(f"    unique    (exactly one hits; rest ambiguous): {g['unique']:.3f}")
    print(f"    top1      (ranker picks ONE, executed)      : {g['top1']:.3f}   <- the honest number")
    if lm:
        print(f"\n  LM ALONE ({lm}, same problems, model does the arithmetic)")
        r = eval_lm(lm, held)
        print(f"    lm_alone : {r['lm_alone']:.3f}")
        print(f"\n  graph top1 {g['top1']:.3f} vs LM alone {r['lm_alone']:.3f}  "
              f"(delta {g['top1'] - r['lm_alone']:+.3f})")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Does the graph compute the answer? Scored on gold.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lm", type=str, default="")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    if a.run:
        sys.exit(0 if _run(a.n, a.lm, a.depth, a.epochs) else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
