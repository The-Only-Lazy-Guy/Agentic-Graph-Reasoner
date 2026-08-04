"""algo_grr_nstm_sup -- the NSTM prototype's ACTUAL design (supervised, cross-entropy on a known
target) on real SWE-bench data, testing whether slots can carry context the prompt no longer holds.

WHY THIS EXISTS AND THE EARLIER NSTM RUNS DO NOT COUNT AS A TEST OF THE IDEA.
The prototype notebook trains its slots with F.cross_entropy against a KNOWN TARGET TOKEN, on a task
that REQUIRES memory (remember two entities, then answer which one the query asks for). I built
something else: REINFORCE on verified outcomes, over 3 instances, on a task whose optimal policy was a
fixed author->test sequence. That is a high-variance estimator with almost no data, on a task where
the slots had nothing unique to hold -- and it collapsed, which says nothing about the notebook's
design. This runs the notebook's design.

THE TASK, and it is real: 2079 SWE-bench hunks are a single line replaced by a single line (22.7% of
9152 real hunks, measured by scripts/mine_verified_schemas.py). Given the issue and the old line,
produce the new line. Known target, so cross-entropy applies directly, exactly as in the notebook.

THE THREE ARMS -- this is simultaneously the KV-cache test:
    A  full        the LM reads ISSUE + OLD LINE           (upper bound; most prompt tokens)
    B  truncated   the LM reads OLD LINE only              (the issue is simply gone)
    C  nstm        the LM reads OLD LINE only, and the ISSUE is written into the SLOTS
If C recovers A while paying B's token cost, the slots are genuinely carrying context the prompt no
longer holds -- which is the memory-compression claim, stated as something measurable rather than
asserted. If C == B, the slots carry nothing and the claim fails on real data with a clean signal and
no excuse about estimator variance.

PRIOR PRESERVATION: W_out is zero-init, so at step 0 arm C is EXACTLY arm B and any gain is
attributable to training rather than to a different starting point.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("HF_HOME", r"E:\cache\hf")

import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_VER = Path(_ROOT) / "artifacts" / "repair_schemas_verified.json"


def load_pairs(n: int = 2000, seed: int = 0):
    """Real (issue, old_line, new_line) triples from verified single-line SWE-bench fixes."""
    if not _VER.exists():
        raise FileNotFoundError(f"{_VER} missing -- run scripts/mine_verified_schemas.py")
    rows = [r for r in json.loads(_VER.read_text(encoding="utf-8"))["instances"]
            if r["schema"] == "replace_line"]
    out = []
    for r in rows:
        p = r.get("params") or {}
        o, nw = p.get("old_line", ""), p.get("new_line", "")
        if o.strip() and nw.strip() and o != nw and len(o) < 200 and len(nw) < 200:
            out.append({"issue": r["problem"], "old": o, "new": nw,
                        "instance_id": r["instance_id"]})
    random.Random(seed).shuffle(out)
    return out[:n]


class SupNSTM(nn.Module):
    """The notebook's module, sized for a real LM. Slots are written from the ISSUE (the context the
    prompt is not allowed to carry in arm C) and emit a gated logit residual while the answer decodes.

    Low-rank output: d_slot -> rank -> vocab. A dense d_slot x 151936 head was 19.4M params and blew
    the VRAM budget; the rank bottleneck keeps the same interface at a fraction of the cost."""

    def __init__(self, d_model: int, vocab: int, num_slots: int = 4, top_k: int = 2,
                 d_slot: int = 128, rank: int = 32):
        super().__init__()
        self.num_slots, self.top_k, self.d_slot = num_slots, top_k, d_slot
        self.h_in = nn.Linear(d_model, d_slot)
        self.ctx_in = nn.Linear(384, d_slot)
        self.W_Q = nn.Linear(d_slot, d_slot, bias=False)
        self.W_K = nn.Linear(d_slot, d_slot, bias=False)
        self.W_V = nn.Linear(d_slot, d_slot, bias=False)
        self.W_g = nn.Linear(d_slot, 1)
        self.gru = nn.GRUCell(d_slot, d_slot)
        self.down = nn.Linear(d_slot, rank, bias=False)
        self.up = nn.Linear(rank, vocab, bias=False)
        nn.init.zeros_(self.up.weight)          # PRIOR PRESERVATION: residual == 0 at init

    def init_slots(self, device):
        return torch.zeros(1, self.num_slots, self.d_slot, device=device)

    def _sparse(self, s):
        if self.top_k >= s.size(-1):
            return F.softmax(s, -1)
        v, i = torch.topk(s, self.top_k, -1)
        m = torch.full_like(s, float("-inf"))
        m.scatter_(-1, i, v)
        return F.softmax(m, -1)

    def write(self, slots, emb):
        """Fold the ISSUE embedding into the slots before decoding begins."""
        v = torch.tanh(self.ctx_in(emb)).view(1, 1, self.d_slot)
        a = self._sparse((self.W_Q(v) * self.W_K(slots)).sum(-1) / math.sqrt(self.d_slot))
        u = a.unsqueeze(-1) * self.W_V(v)
        B, K, D = slots.shape
        return self.gru(u.reshape(B * K, D), slots.reshape(B * K, D)).reshape(B, K, D)

    def step(self, h, slots):
        x = torch.tanh(self.h_in(h)).view(1, 1, self.d_slot)
        a = self._sparse((self.W_Q(x) * self.W_K(slots)).sum(-1) / math.sqrt(self.d_slot))
        u = a.unsqueeze(-1) * self.W_V(x)
        B, K, D = slots.shape
        new = self.gru(u.reshape(B * K, D), slots.reshape(B * K, D)).reshape(B, K, D)
        ra = F.softmax((self.W_Q(x) * self.W_K(new)).sum(-1) / math.sqrt(self.d_slot), -1)
        c = (ra.unsqueeze(-1) * new).sum(1)
        g = torch.sigmoid(self.W_g(x.view(1, self.d_slot)))
        return new, g, self.up(self.down(c))


def token_f1(pred: str, gold: str) -> float:
    """Partial credit on the generated line. Exact-match sits at ~0.01 for EVERY arm including
    full-context, so it is a FLOOR metric here: it cannot reveal a generation gain that exists but is
    short of character-perfect. F1 over token multisets shows whether the decode is getting closer."""
    import collections
    p = collections.Counter(re.findall(r"[\w\.]+|\S", pred or ""))
    g = collections.Counter(re.findall(r"[\w\.]+|\S", gold or ""))
    inter = sum((p & g).values())
    if not inter:
        return 0.0
    pr, rc = inter / max(1, sum(p.values())), inter / max(1, sum(g.values()))
    return 2 * pr * rc / (pr + rc)


def prompt_for(arm: str, issue: str, old: str) -> str:
    """Arm A carries the issue in the PROMPT. Arms B and C do not -- in C it lives in the slots."""
    if arm == "full":
        return (f"Bug report:\n{issue[:700]}\n\nThis line is wrong:\n{old}\n"
                f"Write the corrected line, and nothing else:\n")
    return f"This line is wrong:\n{old}\nWrite the corrected line, and nothing else:\n"


def teacher_forced_loss(lm, nstm, arm, issue_emb, prompt: str, target: str, train: bool):
    """Cross-entropy on the REAL target line, exactly as the notebook trains. The LM is frozen; only
    NSTM receives gradient, and only through the logit residual."""
    tok = lm.tok
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt")
    ids = (ids if torch.is_tensor(ids) else ids["input_ids"]).to(lm.device)
    tgt = tok(target, return_tensors="pt", add_special_tokens=False).input_ids.to(lm.device)
    full = torch.cat([ids, tgt], 1)
    with torch.no_grad():
        out = lm.model(full, output_hidden_states=True)
    logits = out.logits[0, ids.shape[1] - 1: full.shape[1] - 1, :].float()   # predict each target tok
    if arm == "nstm":
        h = out.hidden_states[-1][0, ids.shape[1] - 1: full.shape[1] - 1, :].float()
        slots = nstm.write(nstm.init_slots(lm.device), issue_emb)
        res = []
        for t in range(h.shape[0]):
            slots, g, dz = nstm.step(h[t:t + 1], slots)
            res.append(g * dz)
        logits = logits + torch.cat(res, 0)
    return F.cross_entropy(logits, tgt[0])


@torch.no_grad()
def generate_line(lm, nstm, arm, issue_emb, prompt: str, max_new: int = 48) -> str:
    """Actually DECODE the replacement line with the residual live, so the result can be scored as a
    REPAIR (exact match against the maintainer's line) rather than only as cross-entropy. Greedy, so
    the comparison between arms is deterministic. KV-cached: the earlier full-forward-per-token loop
    is what pushed VRAM past budget."""
    tok = lm.tok
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt")
    ids = (ids if torch.is_tensor(ids) else ids["input_ids"]).to(lm.device)
    slots = nstm.write(nstm.init_slots(lm.device), issue_emb) if arm == "nstm" else None
    start, past, cur = ids.shape[1], None, ids
    for _ in range(max_new):
        out = lm.model(cur, past_key_values=past, use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :].float()
        if arm == "nstm":
            slots, g, dz = nstm.step(out.hidden_states[-1][:, -1, :].float(), slots)
            logits = logits + g * dz
        nxt = logits.argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], 1)
        cur = nxt
        if nxt.item() == tok.eos_token_id:
            break
    return tok.decode(ids[0, start:], skip_special_tokens=True).strip()


def embed(texts):
    from embedder import encode_batch
    return np.asarray(encode_batch(texts), dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description="Supervised NSTM: can slots carry the dropped context?")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lm", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if not a.run:
        ap.print_help(); return

    rows = load_pairs(a.n, seed=a.seed)
    cut = int(len(rows) * 0.8)
    tr, held = rows[:cut], rows[cut:]
    print(f"{len(rows)} real single-line SWE-bench fixes  |  train {len(tr)}  held {len(held)}\n")
    torch.manual_seed(a.seed)
    from v5.runtime.dcpd_latent import WhiteBox
    lm = WhiteBox(a.lm, quant="4bit")
    ie_tr = embed([r["issue"][:600] for r in tr])
    ie_h = embed([r["issue"][:600] for r in held])

    def evaluate(arm, nstm, gen_cap: int = 120, shuffle: bool = False):
        """CE on every held example; generation on a capped slice (identical cap across arms).

        shuffle=True is THE FALSIFIER: each example is given ANOTHER example's issue embedding. A
        zero-init residual trained on 1120 examples could simply have learned a generic "python line"
        prior that helps no matter which issue it saw -- that would be indistinguishable from memory in
        the headline numbers. If shuffled slots recover just as much, it is a prior, not memory. If
        they degrade, the slots genuinely carry INSTANCE-SPECIFIC content."""
        tot, n, em, f1, g_n = 0.0, 0, 0, 0.0, 0
        idx = list(range(len(held)))
        if shuffle:
            rnd = random.Random(1234)
            idx = idx[:]
            rnd.shuffle(idx)
            # guarantee nobody keeps their own issue
            idx = [idx[(k + 1) % len(idx)] if idx[k] == k else idx[k] for k in range(len(idx))]
        with torch.no_grad():
            for i, r in enumerate(held):
                e = torch.tensor(ie_h[idx[i]], device=lm.device)
                pr = prompt_for(arm, r["issue"], r["old"])
                tot += float(teacher_forced_loss(lm, nstm, arm, e, pr, r["new"], train=False)); n += 1
                if g_n < gen_cap:
                    out = generate_line(lm, nstm, arm, e, pr)
                    em += int(out.strip() == r["new"].strip())
                    f1 += token_f1(out, r["new"])
                    g_n += 1
        return tot / max(1, n), em / max(1, g_n), f1 / max(1, g_n)

    nstm = SupNSTM(lm.model.config.hidden_size, lm.model.config.vocab_size).to(lm.device)
    print(f"SupNSTM: {sum(p.numel() for p in nstm.parameters()):,} params\n")

    res, em, f1 = {}, {}, {}
    for arm in ("full", "truncated"):
        res[arm], em[arm], f1[arm] = evaluate(arm if arm == "full" else "trunc", nstm)
        print(f"  arm {arm:10s} held CE {res[arm]:.4f}   EM {em[arm]:.3f}   token-F1 {f1[arm]:.3f}",
              flush=True)

    # arm C: train ONLY the slots to recover what truncation removed
    opt = torch.optim.Adam(nstm.parameters(), lr=1e-4)
    print(f"\n  training NSTM (LM frozen, CE on the real target line)...", flush=True)
    for ep in range(a.epochs):
        run, k = 0.0, 0
        for i, r in enumerate(tr):
            e = torch.tensor(ie_tr[i], device=lm.device)
            loss = teacher_forced_loss(lm, nstm, "nstm", e,
                                       prompt_for("trunc", r["issue"], r["old"]), r["new"], True)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(nstm.parameters(), 1.0)
            opt.step()
            run += float(loss); k += 1
            if k % 50 == 0:
                print(f"    epoch {ep+1} [{k}/{len(tr)}] train CE {run/k:.4f}", flush=True)
        print(f"    epoch {ep+1} done, train CE {run/max(1,k):.4f}", flush=True)
    res["nstm"], em["nstm"], f1["nstm"] = evaluate("nstm", nstm)
    res["shuf"], em["shuf"], f1["shuf"] = evaluate("nstm", nstm, shuffle=True)

    print(f"\n{'=' * 74}")
    print("SUPERVISED NSTM -- can slots carry context the prompt no longer holds?")
    print(f"  A full      (issue in prompt)     CE {res['full']:.4f}   EM {em['full']:.3f}   token-F1 {f1['full']:.3f}")
    print(f"  B truncated (issue GONE)          CE {res['truncated']:.4f}   EM {em['truncated']:.3f}   token-F1 {f1['truncated']:.3f}")
    print(f"  C nstm      (issue in SLOTS)      CE {res['nstm']:.4f}   EM {em['nstm']:.3f}   token-F1 {f1['nstm']:.3f}")
    print(f"  D shuffled  (WRONG issue in slots) CE {res['shuf']:.4f}   EM {em['shuf']:.3f}   token-F1 {f1['shuf']:.3f}")
    de = em["nstm"] - em["truncated"]
    print(f"\n  REPAIRS: slots vs truncated {de:+.3f}   (full-context ceiling {em['full']:.3f})")
    gap = res["truncated"] - res["full"]
    rec = res["truncated"] - res["nstm"]
    print(f"\n  context is worth        : {gap:+.4f} CE  (B - A)")
    print(f"  slots recovered         : {rec:+.4f} CE  (B - C)")
    print(f"  fraction of the context the slots carry: "
          f"{(rec / gap if abs(gap) > 1e-6 else 0):.1%}")
    spec = res["shuf"] - res["nstm"]
    print(f"\n  FALSIFIER -- slots given the WRONG issue: CE {res['shuf']:.4f} vs {res['nstm']:.4f}")
    print(f"    instance-specific content worth {spec:+.4f} CE")
    print(f"    -> {'GENUINE MEMORY (wrong issue hurts)' if spec > 0.05 else 'A GENERIC PRIOR, NOT MEMORY (wrong issue is just as good)'}")
    print(f"  token-F1 generation: B {f1['truncated']:.3f} -> C {f1['nstm']:.3f} (ceiling A {f1['full']:.3f})")
    print(f"{'=' * 74}")


if __name__ == "__main__":
    main()
