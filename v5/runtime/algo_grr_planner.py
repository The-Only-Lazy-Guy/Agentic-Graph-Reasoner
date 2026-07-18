"""algo_grr_planner — the LEARNED planner (#2): infer an atom-PROGRAM (structure) from the NL task.

Replaces MembraneV2's OraclePlanner. The composition-ceiling result showed a frozen 3B collapses to 3%
wiring a depth-5 expression; given the structure the realizer is 100%. This module LEARNS to emit that
structure from language: a small seq2seq-with-attention reads the NL task and decodes a prefix-serialised
atom-program (ops over atoms + the input leaf). realize() (algo_grr_pipeline) turns it into code; verify
grades it. Held-out expressions test generalisation.

  input:  "one more than the negation of the sum of the product of n and n and n"
  output: inc neg add mul n n n            (prefix; fixed arities -> unambiguous tree)

Trained on shallow depths, evaluated DEEPER (structure generalisation, not memorisation).

    python -m v5.runtime.algo_grr_planner --selftest   # no-GPU: train + held-out realized-verify rate
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from v5.runtime.algo_grr_wiring import gen_expr, to_words, oracle, verify, _UNARY, _BINARY  # noqa: E402

# program token vocab: ops (fixed arity) + leaf + specials
_OPS = [_UNARY[k][1] for k in _UNARY] + [_BINARY[k][1] for k in _BINARY]   # dbl inc sq neg add mul sub
_ARITY = {**{_UNARY[k][1]: 1 for k in _UNARY}, **{_BINARY[k][1]: 2 for k in _BINARY}, "n": 0}
_PTOK = ["<pad>", "<bos>", "<eos>", "n"] + _OPS
_P2I = {t: i for i, t in enumerate(_PTOK)}


def _prog_tokens(t) -> list[str]:
    """Prefix serialisation of the expression tree -> program tokens (over atom names)."""
    if t[0] == "n":
        return ["n"]
    if t[0] in _UNARY:
        return [_UNARY[t[0]][1]] + _prog_tokens(t[1])
    return [_BINARY[t[0]][1]] + _prog_tokens(t[1]) + _prog_tokens(t[2])


def _tokens_to_wiring(toks: list[str]):
    """Inverse: prefix program tokens -> AtomProgram wiring tree (+ atom list). Robust to truncation."""
    from v5.runtime.algo_grr_pipeline import AtomProgram
    it = iter(toks)

    def parse():
        try:
            tok = next(it)
        except StopIteration:
            return "n"                              # pad a truncated program with the leaf
        if tok == "n":
            return "n"
        ar = _ARITY.get(tok, 0)
        return ("call", tok, [parse() for _ in range(ar)])

    wiring = parse()

    def atoms_of(w, acc):
        if isinstance(w, tuple):
            acc.append(w[1])
            for a in w[2]:
                atoms_of(a, acc)
    acc: list = []
    atoms_of(wiring, acc)
    # dedup keep order
    seen, atoms = set(), []
    for a in acc:
        if a not in seen:
            seen.add(a); atoms.append(a)
    return AtomProgram(atoms=atoms or ["n"], wiring=wiring)


def _word_vocab(plans_words):
    vocab = {"<pad>": 0, "<unk>": 1}
    for w in plans_words:
        for tok in w.replace("(", " ").replace(")", " ").split():
            vocab.setdefault(tok.lower(), len(vocab))
    return vocab


def _build():
    import torch
    import torch.nn as nn

    class Seq2Seq(nn.Module):
        """GRU encoder + attention decoder. Reads word tokens, decodes program tokens."""

        def __init__(self, n_words, n_prog, d=128):
            super().__init__()
            self.we = nn.Embedding(n_words, d, padding_idx=0)
            self.enc = nn.GRU(d, d, batch_first=True, bidirectional=True)
            self.pe = nn.Embedding(n_prog, d, padding_idx=0)
            self.dec = nn.GRUCell(d + 2 * d, d)
            self.att = nn.Linear(2 * d, d)
            self.out = nn.Linear(d + 2 * d, n_prog)
            self.h0 = nn.Linear(2 * d, d)
            self.d = d

        def encode(self, w):
            H, hn = self.enc(self.we(w))            # H:[B,L,2d]
            h = torch.tanh(self.h0(torch.cat([hn[0], hn[1]], -1)))
            return H, h

        def step(self, tok, h, H, wmask):
            x = self.pe(tok)                         # [B,d]
            sc = (self.att(H) @ h.unsqueeze(-1)).squeeze(-1)     # [B,L]
            sc = sc.masked_fill(~wmask, -1e9)
            a = torch.softmax(sc, -1).unsqueeze(1)   # [B,1,L]
            ctx = (a @ H).squeeze(1)                 # [B,2d]
            h = self.dec(torch.cat([x, ctx], -1), h)
            logit = self.out(torch.cat([h, ctx], -1))
            return logit, h

    return torch, nn, Seq2Seq


def _make_data(depths, n, seed, wvocab=None):
    rng = random.Random(seed)
    words, progs = [], []
    for i in range(n):
        d = rng.choice(depths)
        t = gen_expr(d, random.Random(seed * 7 + i))
        words.append(to_words(t))
        progs.append(["<bos>"] + _prog_tokens(t) + ["<eos>"])
    if wvocab is None:
        wvocab = _word_vocab(words)
    return words, progs, wvocab


def train_planner(model, words, progs, wvocab, steps=3000, lr=2e-3, seed=0):
    import torch
    rng = random.Random(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = torch.nn.CrossEntropyLoss(ignore_index=0)

    def enc_words(w):
        ids = [wvocab.get(t.lower(), 1) for t in w.replace("(", " ").replace(")", " ").split()]
        return ids or [1]
    for it in range(steps):
        i = rng.randrange(len(words))
        w = torch.tensor([enc_words(words[i])])
        p = torch.tensor([[_P2I[t] for t in progs[i]]])
        H, h = model.encode(w)
        wmask = torch.ones(1, w.shape[1], dtype=torch.bool)
        loss = 0.0
        for j in range(p.shape[1] - 1):
            logit, h = model.step(p[:, j], h, H, wmask)
            loss = loss + lossf(logit, p[:, j + 1])
        loss = loss / (p.shape[1] - 1)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 600 == 0 or it == steps - 1:
            print(f"  [planner it {it:4d}] loss {loss.item():.4f}", flush=True)
    return model, enc_words


def plan_program(model, enc_words, wvocab, task_words, max_len=40):
    import torch
    model.eval()
    with torch.no_grad():
        w = torch.tensor([enc_words(task_words)])
        H, h = model.encode(w)
        wmask = torch.ones(1, w.shape[1], dtype=torch.bool)
        tok = torch.tensor([_P2I["<bos>"]])
        out = []
        for _ in range(max_len):
            logit, h = model.step(tok, h, H, wmask)
            nxt = int(logit.argmax(-1))
            t = _PTOK[nxt]
            if t == "<eos>":
                break
            if t not in ("<bos>", "<pad>"):
                out.append(t)
            tok = torch.tensor([nxt])
    return out


def plan_by_search(model, enc_words, task_words, verify_fn, beam=10, max_len=40):
    """Net-GUIDED VERIFIED search (the GRR-7 recipe): beam-decode top-`beam` candidate programs with the
    seq2seq as a GUIDE, then VERIFY each -> return the first that passes. Search does the reasoning; the
    net cuts the budget; verify gates. Generalises to depth where flat decode degrades. Returns
    (AtomProgram|None, n_verifies, solved)."""
    import torch
    model.eval()
    with torch.no_grad():
        w = torch.tensor([enc_words(task_words)])
        H, h0 = model.encode(w)
        wmask = torch.ones(1, w.shape[1], dtype=torch.bool)
        beams = [([_P2I["<bos>"]], 0.0, h0)]
        done = []
        for _ in range(max_len):
            nxt = []
            for toks, lp, h in beams:
                if toks[-1] == _P2I["<eos>"]:
                    done.append((toks, lp)); continue
                logit, h2 = model.step(torch.tensor([toks[-1]]), h, H, wmask)
                logp = torch.log_softmax(logit, -1)[0]
                tv, ti = logp.topk(beam)
                for v, i in zip(tv.tolist(), ti.tolist()):
                    nxt.append((toks + [i], lp + v, h2))
            if not nxt:
                break
            beams = sorted(nxt, key=lambda x: x[1], reverse=True)[:beam]
        done += [(t, l) for t, l, _ in beams]
    cands = sorted(done, key=lambda x: x[1], reverse=True)[:beam]
    verifies = 0
    for toks, _ in cands:
        prog = _tokens_to_wiring([_PTOK[i] for i in toks if _PTOK[i] not in ("<bos>", "<eos>", "<pad>")])
        verifies += 1
        if verify_fn(_realize_prog(prog)):
            return prog, verifies, True
    return None, verifies, False


def _selftest() -> bool:
    print("algo_grr_planner --selftest: LEARNED planner infers atom-programs from NL (no GPU)\n")
    torch, nn, Seq2Seq = _build()
    torch.manual_seed(0)
    tr_words, tr_progs, wvocab = _make_data([1, 2, 3], 1500, seed=1)
    model = Seq2Seq(len(wvocab), len(_PTOK))
    model, enc_words = train_planner(model, tr_words, tr_progs, wvocab, steps=3500)

    def eval_depth(d, n=60, seed=500):
        flat = srch = budget = 0
        for i in range(n):
            t = gen_expr(d, random.Random(seed + i))
            words = to_words(t)
            # (a) flat greedy decode
            prog = _tokens_to_wiring(plan_program(model, enc_words, wvocab, words))
            flat += int(verify(_realize_prog(prog), t))
            # (b) net-guided VERIFIED search
            vf = lambda body, t=t: verify(body, t)  # noqa: E731
            _p, nv, solved = plan_by_search(model, enc_words, words, vf, beam=10)
            srch += int(solved); budget += nv
        return flat / n, srch / n, budget / n

    print("\n  HELD-OUT (unseen):  depth | flat-decode verify | net-guided SEARCH verify | avg verifies")
    frows, srows = [], []
    for d in (1, 2, 3, 4, 5):
        fv, sv, bud = eval_depth(d)
        frows.append(fv); srows.append(sv)
        tag = "" if d <= 3 else "  (DEEPER than train)"
        print(f"                       {d}    |       {fv:.2f}         |         {sv:.2f}           |   {bud:.1f}{tag}")
    fo, so = sum(frows) / len(frows), sum(srows) / len(srows)
    ok = so - fo > 0.15 and srows[1] > 0.9        # search must clearly beat flat + solve in-range depth-2
    print(f"\n  overall: flat-decode {fo:.2f}  ->  net-guided-search {so:.2f}   (frozen-LM free-form was 0.03@d5)")
    print(f"  -> {'PASS' if ok else 'FAIL'}: net-GUIDED VERIFIED SEARCH clearly beats flat decode (search reasons,")
    print(f"     net guides, verify gates = GRR-7). HONEST: deep extrapolation (d>=4) is BUDGET-bound — beam=10 +")
    print(f"     OOD net-guidance can't contain the program; raise beam / train on deeper depths (GRR-8 tradeoff).")
    print(f"\n  ALGO_GRR_PLANNER SELFTEST -> {'PASS' if ok else 'FAIL'}")
    return ok


def _realize_prog(prog) -> str:
    """Entry glue only (verify() prepends the verified helper closure)."""
    from v5.runtime.algo_grr_pipeline import _render
    return f"def solve(n):\n    return {_render(prog.wiring)}\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
