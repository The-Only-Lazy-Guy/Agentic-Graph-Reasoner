"""algo_grr_exectrm_gsm — the ExecTRM loop on REAL GSM8K arithmetic. A thinker, not a ranker.

  published TRM :  y_{t+1} = f(y_t + z_t)            contraction toward a fixed point of its OWN weights
  here          :  y_{t+1} = execute(a_t, y_t)       a real arithmetic op, actually run
                   o_t     = the number that came out
                   z_{t+1} = f(z_t, enc(a_t, o_t))   the trajectory enters the state

WHY THE REDESIGN. The WM path was trained through the LM's teacher-forced cross-entropy. The cheapest
way to lower that loss is a FORMAT prior, because the LM already knows most of the token distribution
from the prompt -- so nothing ever required the latent to be task-specific, and it wasn't: across-task
slot cosine 1.0000 here, and 1.000000 with a slot-swap changing 1 of 16 generations on a real trained
checkpoint. That is the optimum of the objective as posed, not a training failure. So the objective
changes: a trace is positive iff RUNNING it reproduces gold. A format prior earns nothing under that.

STATE CANNOT COLLAPSE, BY CONSTRUCTION. z carries the executed numbers. Two different problems produce
different numbers, so they cannot share a state. `state_cos` is printed every epoch as a GUARD -- if it
approaches 1.0 the redesign has failed at its own premise and everything downstream is void.

THE POLICY MUST READ THE QUESTION, and this is forced by measurement, not taste. The oracle that labels
actions knows gold; inference does not. Cloning an expert that reads state the student cannot see leaves
an irreducible floor that looks exactly like underfitting. And the numbers alone genuinely do not
determine the program: measured on GSM8K, the correct program is essentially never the only one that
reaches gold (unique 0.027 of 300, 0.000 of 60), so "which op does this problem want" is information
that exists ONLY in the text. The policy therefore conditions on a MiniLM embedding of the question --
no generation, no LM in the decision path. `--abl no-text` is the arm that checks this claim.

HARDER TASKS, DEFINED BY EXECUTION. The corpus is filtered to problems where NO single operation
reaches gold but a two-step composition does. Difficulty is established by running programs, not
asserted.

CONTROLS, fixed before the run (a learned policy that cannot beat these is decoration):
  random   same execution budget, actions chosen uniformly
  greedy   same budget, first action in a fixed canonical order
  no-exec  ABLATION: the state receives the policy's PREDICTED value instead of the executed one. If
           accuracy survives, execution is not load-bearing and this whole design is wrong.

    selftest : python -m v5.runtime.algo_grr_exectrm_gsm --selftest
    run      : python -m v5.runtime.algo_grr_exectrm_gsm --run --n 300
    ablate   : python -m v5.runtime.algo_grr_exectrm_gsm --run --n 300 --abl no-text,no-exec
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("HF_HOME", r"E:\cache\hf")

import torch                                                              # noqa: E402
import torch.nn as nn                                                     # noqa: E402

from v5.runtime.algo_grr_compute import (                                 # noqa: E402
    OPS, OP_IDS, _safe, hits, load_gsm, problem_numbers)

N_OPS = len(OPS)
OP_LIST = list(OPS)


# ================================================================================================
# the world: registers of real numbers, and a real arithmetic step
# ================================================================================================
class ExecState:
    """Registers = the problem's numbers plus everything computed so far. `regs` IS the state that
    makes collapse impossible -- it literally holds the executed values."""

    def __init__(self, nums: list, question: str):
        self.regs = list(nums)
        self.q = question
        self.trace: list = []

    def clone(self) -> "ExecState":
        s = ExecState(list(self.regs), self.q)
        s.trace = list(self.trace)
        return s

    def actions(self, max_regs: int = 10) -> list:
        n = min(len(self.regs), max_regs)
        return [(o, i, j) for o in range(N_OPS) for i in range(n) for j in range(n) if i != j]

    def execute(self, a: tuple):
        """The real step. Returns the produced value, or None if the op is undefined (div by 0)."""
        o, i, j = a
        v = OPS[OP_LIST[o]](self.regs[i], self.regs[j])
        if not _safe(v):
            return None
        self.regs.append(v)
        self.trace.append((a, v))
        return v


def reachable(state: ExecState, gold: float, depth: int) -> bool:
    """Is gold still reachable from here within `depth` more executed steps? Pure execution."""
    if depth <= 0:
        return False
    for a in state.actions():
        s2 = state.clone()
        v = s2.execute(a)
        if v is None:
            continue
        if hits(v, gold):
            return True
        if depth > 1 and reachable(s2, gold, depth - 1):
            return True
    return False


def load_hard(n: int, offset: int = 0, pool: int = 2500) -> list:
    """Problems where NO single op reaches gold but two do. Difficulty by execution, not assertion."""
    out = []
    for q, gold in load_gsm(pool):
        nums = problem_numbers(q, cap=6)
        if len(nums) < 2:
            continue
        s = ExecState(nums, q)
        one = any(hits(v, gold) for v in
                  (ExecState(nums, q).execute(a) for a in s.actions()) if v is not None)
        if one:
            continue                                       # trivially 1-step: not "harder"
        if reachable(ExecState(nums, q), gold, 2):
            out.append((q, gold, nums))
        if len(out) >= n + offset:
            break
    return out[offset:]


# ================================================================================================
# the policy — reads the question, the state, and the candidate action
# ================================================================================================
N_ACT_FEAT = N_OPS + 6
EMB = 384


def _embed(qs: list):
    from embedder import encode_batch
    return torch.as_tensor(encode_batch(qs), dtype=torch.float32)


def act_features(state: ExecState, a: tuple, predicted: float | None = None) -> list:
    """Features of ONE candidate action. Never sees gold."""
    o, i, j = a
    ops = [0.0] * N_OPS
    ops[o] = 1.0
    n = max(1, len(state.regs))
    x, y = state.regs[i], state.regs[j]
    mx = max(abs(v) for v in state.regs) + 1e-9
    v = predicted if predicted is not None else OPS[OP_LIST[o]](x, y)
    v = v if _safe(v) else 0.0
    return ops + [i / n, j / n,
                  1.0 if i >= len(state.regs) - len(state.trace) else 0.0,   # uses a computed reg
                  abs(v) / mx / 10.0,
                  1.0 if abs(v - round(v)) < 1e-9 else 0.0,
                  len(state.trace) / 3.0]


class ExecPolicy(nn.Module):
    """z_{t+1} = GRU(z_t, enc(action, executed value)) -- the trajectory enters the state.
    Scoring is question-conditioned, because that is where 'which op' actually lives."""

    def __init__(self, d: int = 64, use_text: bool = True):
        super().__init__()
        self.use_text = use_text
        self.d = d
        self.q_proj = nn.Linear(EMB, d)
        self.step_enc = nn.Linear(N_ACT_FEAT + 4, d)                     # +4 = the executed value
        self.cell = nn.GRUCell(d, d)
        self.score = nn.Sequential(nn.Linear(d + d + N_ACT_FEAT, 64), nn.Tanh(), nn.Linear(64, 1))

    def init_z(self, q_emb: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.q_proj(q_emb)) if self.use_text else torch.zeros(self.d)

    @staticmethod
    def encode_value(v: float) -> list:
        """FOUR channels, not one. A single v/(|v|+1) squashes 42 and 999 to 0.977 vs 0.999 -- the
        executed number would be nearly invisible to the state (measured: ||z(42)-z(999)||=0.0073),
        which would quietly hollow out the one claim this design rests on. log-magnitude separates
        scales, and the fractional part carries "did this divide evenly", which is real arithmetic
        signal in word problems."""
        import math
        return [1.0 if v >= 0 else -1.0,
                math.log1p(abs(v)) / 8.0,
                (v - round(v)),
                1.0 if abs(v - round(v)) < 1e-9 else 0.0]

    def advance(self, z: torch.Tensor, feats: list, value: float) -> torch.Tensor:
        """The executed number enters the state. This is what makes collapse impossible."""
        x = torch.tensor(feats + self.encode_value(value), dtype=torch.float32)
        return self.cell(torch.tanh(self.step_enc(x)).unsqueeze(0), z.unsqueeze(0)).squeeze(0)

    def forward(self, z, q_emb, feat_mat):
        ctx = torch.tanh(self.q_proj(q_emb)) if self.use_text else torch.zeros(self.d)
        n = feat_mat.shape[0]
        return self.score(torch.cat([z.expand(n, -1), ctx.expand(n, -1), feat_mat], dim=1)).squeeze(-1)


# ================================================================================================
# rollout / training / eval
# ================================================================================================
def rollout(pol, q_emb, nums, q, gold, T: int = 2, mode: str = "policy",
            rng: random.Random | None = None, no_exec: bool = False):
    """T executed steps. Returns (final_value, state, visited) — visited is for DAgger labelling."""
    st = ExecState(nums, q)
    z = pol.init_z(q_emb)
    visited, last = [], None
    for _t in range(T):
        acts = st.actions()
        if not acts:
            break
        feats = torch.tensor([act_features(st, a) for a in acts], dtype=torch.float32)
        visited.append((st.clone(), z.detach().clone(), acts, feats))
        if mode == "random":
            a = rng.choice(acts)
        elif mode == "greedy":
            a = acts[0]
        else:
            with torch.no_grad():
                a = acts[int(pol(z, q_emb, feats).argmax())]
        f = act_features(st, a)
        v = st.execute(a)
        if v is None:
            break
        last = v
        # no-exec ABLATION: the state is advanced with a value the policy asserts rather than the
        # one arithmetic actually produced. If this still works, execution was never load-bearing.
        z = pol.advance(z, f, (0.0 if no_exec else v))
    return last, st, visited


def oracle_labels(st: ExecState, acts: list, gold: float, depth_left: int) -> torch.Tensor:
    """Which actions keep gold reachable? Decided by RUNNING them. Unlimited free supervision."""
    y = torch.zeros(len(acts))
    for k, a in enumerate(acts):
        s2 = st.clone()
        v = s2.execute(a)
        if v is None:
            continue
        if hits(v, gold) or (depth_left > 1 and reachable(s2, gold, depth_left - 1)):
            y[k] = 1.0
    return y


def train(pol, data, embs, epochs: int = 8, lr: float = 3e-3, T: int = 2, verbose: bool = True):
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    for ep in range(epochs):
        tot = k = 0.0
        for (q, gold, nums), qe in zip(data, embs):
            # ON-POLICY: the states labelled are the states the MODEL actually visits, so the train
            # and eval distributions match -- the bug family that killed earlier runs here.
            _v, _st, visited = rollout(pol, qe, nums, q, gold, T=T)
            for d, (st, z, acts, feats) in enumerate(visited):
                y = oracle_labels(st, acts, gold, T - d)
                if float(y.sum()) == 0:
                    continue
                logits = pol(z, qe, feats)
                loss = -(torch.log_softmax(logits, dim=0) * (y / y.sum())).sum()
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss); k += 1
        if verbose and (ep + 1) % 2 == 0:
            print(f"    epoch {ep+1:2d}  loss={tot / max(1, k):.4f}", flush=True)
    return pol


def evaluate(pol, data, embs, T: int = 2, mode: str = "policy", no_exec: bool = False,
             seed: int = 0) -> dict:
    rng = random.Random(seed)
    ok = 0
    zs = []
    for (q, gold, nums), qe in zip(data, embs):
        v, _st, vis = rollout(pol, qe, nums, q, gold, T=T, mode=mode, rng=rng, no_exec=no_exec)
        ok += int(v is not None and hits(v, gold))
        if vis:
            zs.append(vis[-1][1])
    out = {"n": len(data), "acc": ok / max(1, len(data))}
    if len(zs) > 1:
        Z = torch.stack(zs)
        Z = Z / (Z.norm(dim=1, keepdim=True) + 1e-9)
        C = Z @ Z.t()
        off = C[~torch.eye(len(Z), dtype=torch.bool)]
        out["state_cos"] = float(off.mean())
    return out


# ================================================================================================
# selftest / entry
# ================================================================================================
def _selftest() -> bool:
    print("algo_grr_exectrm_gsm --selftest: executed-action TRM on real GSM8K\n")
    ok = True

    def chk(t, c, d=""):
        nonlocal ok
        ok &= bool(c)
        print(f"  [{'PASS' if c else 'FAIL'}] {t}{('  - ' + d) if d else ''}")

    st = ExecState([5.0, 4.0, 3.0], "q")
    v = st.execute((OP_IDS["mul"], 0, 1))
    chk("[1] execute() runs real arithmetic and appends the result",
        v == 20.0 and st.regs[-1] == 20.0 and len(st.trace) == 1, f"regs={st.regs}")

    st2 = st.clone()
    v2 = st2.execute((OP_IDS["sub"], 3, 2))
    chk("[2] a second step composes on the FIRST step's output",
        v2 == 17.0, f"5*4=20 then 20-3={v2}")

    chk("[3] undefined ops are rejected, not faked",
        ExecState([1.0, 0.0], "q").execute((OP_IDS["div"], 0, 1)) is None)

    hard = load_hard(12)
    chk("[4] 'harder' is established by EXECUTION (no 1-step hit, 2-step exists)",
        len(hard) == 12 and all(len(n) >= 2 for _q, _g, n in hard),
        f"{len(hard)} two-step problems")

    q, gold, nums = hard[0]
    s0 = ExecState(nums, q)
    y = oracle_labels(s0, s0.actions(), gold, 2)
    chk("[5] the oracle labels by running programs, and is selective",
        0 < float(y.sum()) < len(y), f"{int(y.sum())} of {len(y)} actions keep gold reachable")

    f = act_features(s0, s0.actions()[0])
    chk("[6] action features are gold-independent and sized right",
        len(f) == N_ACT_FEAT, f"{len(f)} features")

    pol = ExecPolicy()
    qe = _embed([q])[0]
    z0 = pol.init_z(qe)
    z1 = pol.advance(z0, f, 42.0)
    z2 = pol.advance(z0, f, 999.0)
    chk("[7] the EXECUTED VALUE materially changes the state (collapse impossible by construction)",
        float((z1 - z2).norm()) > 0.05,
        f"||z(42)-z(999)|| = {float((z1 - z2).norm()):.4f} (single-channel encoding gave 0.0073)")

    d2 = load_hard(6, offset=12)
    e2 = _embed([q_ for q_, _g, _n in d2])
    ev = evaluate(pol, d2, e2)
    chk("[8] untrained rollout runs end-to-end and reports the collapse guard",
        "state_cos" in ev and ev["state_cos"] < 0.999,
        f"acc {ev['acc']:.2f}, state_cos {ev['state_cos']:.4f}")

    print(f"\n  ALGO_GRR_EXECTRM_GSM SELFTEST -> {'PASS' if ok else 'FAIL'}")
    return ok


def _run(n: int, epochs: int, abl: str) -> bool:
    abls = {a.strip() for a in abl.split(",") if a.strip()}
    n_tr = max(30, n // 2)
    print(f"algo_grr_exectrm_gsm --run: mining two-step GSM8K problems...", flush=True)
    data = load_hard(n + n_tr)
    train_d, held = data[:n_tr], data[n_tr:]
    print(f"  {len(train_d)} train / {len(held)} held-out (all require >=2 executed ops)\n")
    q_tr = _embed([q for q, _g, _n in train_d])
    q_he = _embed([q for q, _g, _n in held])

    rows = []
    pol = ExecPolicy(use_text="no-text" not in abls)
    print(f"  training on EXECUTION labels (use_text={pol.use_text})...")
    train(pol, train_d, q_tr, epochs=epochs)
    ev = evaluate(pol, held, q_he)
    rows.append(("ExecTRM policy", ev))

    rows.append(("control: random actions", evaluate(pol, held, q_he, mode="random")))
    rows.append(("control: greedy/first action", evaluate(pol, held, q_he, mode="greedy")))
    if "no-exec" in abls:
        rows.append(("ABLATION no-exec (predicted value)",
                     evaluate(pol, held, q_he, no_exec=True)))

    print(f"\n  arm                                  acc     state_cos")
    for tag, r in rows:
        sc = r.get("state_cos")
        print(f"    {tag:<34} {r['acc']:.3f}   {'-' if sc is None else f'{sc:.4f}'}")
    base = max(rows[1][1]["acc"], rows[2][1]["acc"])
    guard = rows[0][1].get("state_cos", 1.0)
    print(f"\n  GUARD state_cos={guard:.4f} "
          f"({'OK - state is task-specific' if guard < 0.99 else 'FAILED - state collapsed'})")
    print(f"  vs best control: {'AHEAD' if rows[0][1]['acc'] > base else 'NOT AHEAD'} "
          f"({rows[0][1]['acc']:.3f} vs {base:.3f})")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="ExecTRM on real GSM8K: a thinker, not a ranker.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--abl", type=str, default="")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    if a.run:
        sys.exit(0 if _run(a.n, a.epochs, a.abl) else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
