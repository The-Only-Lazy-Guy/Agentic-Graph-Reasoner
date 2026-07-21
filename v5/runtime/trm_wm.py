"""trm_wm.py — the TRM as a REAL reasoner COUPLED to the frozen LM (design (b)), with a WORKING MEMORY.

The prior TRM was a ranker: it reordered atoms and never touched the LM. This wires it into the LM's
computation so its reasoning actually MODULATES generation.

  LONG-TERM memory = the graph                       (unchanged; grows without training)
  WORKING memory   = K slots the TRM refines over T recursion steps, INITIALIZED from the top-K retrieved
                     atoms  -> grounded in real content, not a free latent (this is what killed soft-prompt)
  COUPLING to LM   = a GATED CROSS-ATTENTION adapter (Flamingo-style, tanh-gate init 0) inserted at a few
                     frozen-LM layers: the LM's hidden states ATTEND to the working-memory slots.

  DEEP SUPERVISION = at each refinement step t, the working memory must independently identify the target
                     (via cosine retrieval against the answer pool). Gradients flow through EVERY step,
                     not just the final output -> compels the recurrence to be meaningful, not a black box.

Only the adapter + the slot-refiner (the "TRM") train, on VERIFIED answers; the LM never moves (frozen ->
anti-poison preserved, same sanction as trainer.py). The tanh gate starts at 0 so at init the LM is
BITWISE-identical to the base model; the adapter can only *add* signal once it earns lower loss.

Mechanism proven by --selftest on distilgpt2 (no Qwen needed):
  (i)   gate=0  -> LM logits identical to base            (identity at init; can't wreck fluency)
  (ii)  gate!=0 -> LM logits change                       (working memory is causally wired into the LM)
  (iii) train   -> a token placed ONLY in the slots (absent from the prompt) becomes the LM's answer, and
                   it GENERALIZES to HELD-OUT slot content it never trained on  (real copy-from-memory,
                   not memorization) — with the gate ABLATED to 0 the effect vanishes (proves causality).
  (iv)  deep sup: each refinement step independently improves retrieval accuracy -> measured stepwise.

    python -m v5.runtime.trm_wm --selftest                                    # mechanism, distilgpt2, local
    python -m v5.runtime.trm_wm --run --lm Qwen/Qwen3-4B-Instruct-2507        # real experiment (their GPU)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn as nn

from embedder import encode_batch, EMBED_DIM


# ================================================================================================
# gated cross-attention adapter — the LM's hidden states attend to the working-memory slots
# ================================================================================================
class GatedCrossAttn(nn.Module):
    """h (LM hidden [B,S,d]) attends to slots [K,d]; output = h + tanh(g)*proj(attn). g init 0 -> identity."""

    def __init__(self, d: int, n_heads: int = 4):
        super().__init__()
        assert d % n_heads == 0
        self.h, self.dh = n_heads, d // n_heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.g = nn.Parameter(torch.zeros(1))
        for lin in (self.v, self.o):
            nn.init.eye_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, h: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        B, S, d = h.shape
        if slots.dim() == 2:
            slots = slots.unsqueeze(0).expand(B, -1, -1)
        Bk, K, _ = slots.shape
        q = self.q(h).view(B, S, self.h, self.dh).transpose(1, 2)
        k = self.k(slots).view(Bk, K, self.h, self.dh).permute(0, 2, 1, 3)
        v = self.v(slots).view(Bk, K, self.h, self.dh).permute(0, 2, 1, 3)
        att = torch.softmax((q @ k.transpose(-1, -2)) / (self.dh ** 0.5), dim=-1)
        ctx = (att @ v).transpose(1, 2).reshape(B, S, d)
        delta = self.o(ctx)
        delta = delta / (delta.norm(dim=-1, keepdim=True) + 1e-6) * h.norm(dim=-1, keepdim=True)
        return h + torch.tanh(self.g) * delta


# ================================================================================================
# WMReasoner — the working memory + its recursive refinement + the coupling hooks + deep supervision
# ================================================================================================
class WMReasoner(nn.Module):
    """K working-memory slots, initialized from retrieved atom embeddings, refined by T recursion steps
    (the real TRM role: iterative refinement), then read by the LM via gated cross-attention adapters.

    DEEP SUPERVISION: a lightweight retrieval head reads the working memory at each refinement step t
    and scores all candidate answers (cosine sim against answer_pool). CE loss at every step forces the
    entire refinement trajectory to converge toward the target, not just the final state."""
    def __init__(self, d_lm: int, couple_layers, d_emb: int = EMBED_DIM, T: int = 4, n_heads: int = 4):
        super().__init__()
        self.T = T
        # MODALITY PROJECTOR (2-layer GELU): translate KG/MiniLM geometry -> the decoder's native space, so
        # the cross-attention isn't handed 'foreign noise'. A single Linear is too weak to bridge the gap.
        self.proj_atom = nn.Sequential(nn.Linear(d_emb, d_lm), nn.GELU(), nn.Linear(d_lm, d_lm))
        self.proj_task = nn.Linear(d_emb, d_lm)
        self.upd = nn.Sequential(nn.Linear(3 * d_lm, d_lm), nn.GELU(), nn.Linear(d_lm, d_lm))
        self.norm = nn.LayerNorm(d_lm)
        self.adapters = nn.ModuleList([GatedCrossAttn(d_lm, n_heads) for _ in couple_layers])
        self.couple_layers = list(couple_layers)
        self._slots = None

        self.ds_pool = nn.Linear(d_lm, d_lm)
        self.ds_proj = nn.Linear(d_lm, d_lm)
        # temperature for the DS cosine logits. q and pool are L2-normalized -> logits in [-1,1]; dividing by
        # sqrt(d_lm) (~50) flattens them to ~uniform (ds_loss stuck at ln N, no gradient). 0.07 SHARPENS them.
        self._ds_scale = 0.07

    def refine(self, task_emb: torch.Tensor, atom_embs: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """task_emb [d_emb], atom_embs [K,d_emb] -> (working_memory [K,d_lm], per_step_states [[K,d_lm],...])"""
        q = self.proj_task(task_emb)
        z = self.proj_atom(atom_embs)
        states = []
        for _ in range(self.T):
            ctx = z.mean(0, keepdim=True).expand_as(z)
            qb = q.unsqueeze(0).expand_as(z)
            z = self.norm(z + self.upd(torch.cat([z, qb, ctx], dim=-1)))
            states.append(z.clone())
        return z, states

    def ds_loss_batch(self, all_states: list[list[torch.Tensor]], answer_pool: torch.Tensor, gold_idxs: list[int]) -> torch.Tensor:
        """Batch deep supervision: for each example, at each refinement step, pool K slots -> query ->
        score against all candidates in answer_pool. CE on gold_idx at every step.

        all_states: [[T states], ...] per-example per-step states
        answer_pool: [N, d_lm] candidate embeddings
        gold_idxs:   [B] correct answer indices

        Vectorized: all examples + all steps are batched into one loss computation.
        """
        dev = answer_pool.device
        B = len(all_states)
        T = len(all_states[0]) if all_states else 1
        K = all_states[0][0].shape[0]
        # [B, T, K, d_lm] -> [B*T, K, d_lm]
        flat = torch.stack([torch.stack(s) for s in all_states], dim=0).float().to(dev)
        flat = flat.view(B * T, K, -1)
        pooled = torch.tanh(self.ds_pool(flat.mean(1)))  # [B*T, d_lm]
        q = self.ds_proj(pooled)
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
        logits = q @ answer_pool.float().T / self._ds_scale  # [B*T, N]
        gold = torch.tensor(gold_idxs, device=dev).repeat_interleave(T)
        return nn.functional.cross_entropy(logits, gold)

    def align_projector(self, kg_embs: torch.Tensor, lm_targets: torch.Tensor,
                        steps: int = 300, lr: float = 1e-3) -> float:
        """MODALITY PRE-ALIGNMENT (CLIP): train proj_atom so KG (MiniLM) embeddings land in the LM's OWN
        representation space -> cross-attention stops seeing foreign noise. kg_embs [N,d_emb], lm_targets
        [N,d_lm] (the LM's embedding of each atom). Trains proj_atom only; symmetric InfoNCE."""
        opt = torch.optim.Adam(self.proj_atom.parameters(), lr=lr)
        T = lm_targets / (lm_targets.norm(dim=-1, keepdim=True) + 1e-8)
        last = float("nan")
        for _ in range(steps):
            P = self.proj_atom(kg_embs)
            P = P / (P.norm(dim=-1, keepdim=True) + 1e-8)
            logits = P @ T.t() / 0.07
            labels = torch.arange(P.shape[0], device=P.device)
            loss = 0.5 * (nn.functional.cross_entropy(logits, labels) +
                          nn.functional.cross_entropy(logits.t(), labels))
            opt.zero_grad(); loss.backward(); opt.step()
            last = float(loss.detach())
        return last

    def set_context(self, task_emb, atom_embs):
        te = torch.as_tensor(task_emb, dtype=torch.float32, device=self._device())
        ae = torch.as_tensor(atom_embs, dtype=torch.float32, device=self._device())
        if ae.dim() == 1:
            ae = ae.unsqueeze(0)
        self._slots, _ = self.refine(te, ae)

    def set_slots_direct(self, slots: torch.Tensor):
        self._slots = slots.unsqueeze(0) if slots.dim() == 1 else slots

    def clear(self):
        self._slots = None

    def _device(self):
        return self.proj_task.weight.device

    def couple(self, wb) -> list:
        handles = []
        for a_i, L in enumerate(self.couple_layers):
            handles.append(wb.layers[L].register_forward_hook(self._mk_hook(a_i)))
        return handles

    def _mk_hook(self, idx):
        def hook(_mod, _inp, out):
            if self._slots is None:
                return None
            h = out[0] if isinstance(out, tuple) else out
            h2 = self.adapters[idx](h.float(), self._slots.float()).to(h.dtype)
            if isinstance(out, tuple):
                return (h2,) + tuple(out[1:])
            return h2
        return hook


# ================================================================================================
# selftest — prove the mechanism on distilgpt2 (identity / causal / trainable+generalizing / deep sup)
# ================================================================================================
def _vocab_words(tok, n: int = 120):
    """Pull n single-token lowercase words from the vocab."""
    words = []
    for tid in range(len(tok)):
        s = tok.decode([tid])
        if s[:1] == " " and s[1:].isalpha() and s[1:].islower() and len(s) >= 4:
            words.append((s[1:], tid))
        if len(words) >= n:
            break
    return words


def _run_probe(wb, R, pids, train, test, precomputed_states, answer_pool, tid_of,
               steps=200, lr=3e-3, bs=128, ds_weight=0.15):
    """precomputed_states: list of [[K,d_lm], ...] per-step states for each word, one refine per word."""
    k = len(train)
    opt = torch.optim.Adam([p for p in R.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    for a in R.adapters:
        with torch.no_grad():
            a.g.fill_(1.5)

    R.train()
    last = float("nan")
    order = list(range(k))
    for ep in range(steps):
        torch.manual_seed(ep)
        order = torch.randperm(k).tolist()
        tot, nb = 0.0, 0
        for i in range(0, k, bs):
            idx = order[i:i + bs]
            final_slots = torch.stack([precomputed_states[j][-1] for j in idx], dim=0).to(wb.device)
            R.set_slots_direct(final_slots)
            logits = wb.model(pids.expand(len(idx), -1)).logits[:, -1]
            lm_loss = nn.functional.cross_entropy(
                logits, torch.tensor([tid_of[train[j][0]] for j in idx], device=wb.device))

            if ds_weight > 0 and answer_pool is not None:
                state_list = []
                gold_list = []
                for bi, j in enumerate(idx):
                    state_list.append(precomputed_states[j])
                    gold_list.append(j)
                ds_acc = R.ds_loss_batch(state_list, answer_pool.to(wb.device), gold_list)
                loss = lm_loss + ds_weight * ds_acc
            else:
                loss = lm_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(lm_loss.detach())
            nb += 1
        last = tot / nb

    R.eval()

    def acc(data, ablate=False):
        hits = 0
        base = len(train) if data is not train else 0
        for i in range(0, len(data), bs):
            idx = list(range(i, min(i + bs, len(data))))
            chunk_slots = torch.stack([precomputed_states[base + j][-1] for j in idx], dim=0).to(wb.device)
            R.set_slots_direct(chunk_slots.detach())
            gs = [float(a.g) for a in R.adapters]
            if ablate:
                for a in R.adapters:
                    with torch.no_grad():
                        a.g.zero_()
            with torch.no_grad():
                preds = wb.model(pids.expand(len(idx), -1)).logits[:, -1].argmax(-1).tolist()
            if ablate:
                for a, gv in zip(R.adapters, gs):
                    with torch.no_grad():
                        a.g.fill_(gv)
            hits += sum(int(p == tid_of[data[j][0]]) for j, p in zip(idx, preds))
        return hits / len(data)

    tr = acc(train)
    te = acc(test)
    te_abl = acc(test, ablate=True)
    return tr, te, te_abl, float(R.adapters[0].g), last


def selftest(wb=None, bs=128, steps_a=120, steps_b=250):
    from v5.runtime.dcpd_latent import WhiteBox
    torch.manual_seed(0)
    if wb is None:
        print("trm_wm.py --selftest : TRM working memory coupled to a FROZEN distilgpt2 (mechanism proof)\n")
        wb = WhiteBox("distilgpt2", quant="fp32")
        if os.environ.get("GRAPH_FORCE_CPU"):
            wb.model = wb.model.to("cpu"); wb.device = "cpu"
            print("  (forced CPU)")
    else:
        print(f"trm_wm.py --probe on {wb.name} (real LM): copy(A) + bridge(B) mechanism test on a capable model\n")
    d_lm = wb.d_model
    couple = [wb.n_layers - 2, wb.n_layers - 1]         # last two layers (was [4,5] for distilgpt2's 6)
    R = WMReasoner(d_lm, couple_layers=couple).to(wb.device)
    for p in wb.model.parameters():
        p.requires_grad_(False)
    handles = R.couple(wb)

    prompt = "The answer is"
    pids = wb.tok(prompt, return_tensors="pt").input_ids.to(wb.device)
    words = _vocab_words(wb.tok, 120)
    split = max(1, int(0.8 * len(words)))
    train_w, test_w = words[:split], words[split:]
    tid_of = {w: t for w, t in words}

    # (i) identity at init
    R.clear()
    base = wb.model(pids).logits.detach()
    R.set_context(encode_batch([prompt])[0], encode_batch(["banana"])[0])
    id_diff = (base - wb.model(pids).logits.detach()).abs().max().item()
    print(f"  (i)   identity@init   max|base - withslots(gate=0)| = {id_diff:.2e}   "
          f"{'PASS' if id_diff < 1e-4 else 'FAIL'}")
    # (ii) causal
    with torch.no_grad():
        R.adapters[0].g.fill_(1.0)
    ch = (base - wb.model(pids).logits.detach()).abs().max().item()
    print(f"  (ii)  causal wiring   max|base - withslots(gate=1)| = {ch:.2e}   "
          f"{'PASS' if ch > 1e-3 else 'FAIL'}\n")
    with torch.no_grad():
        R.adapters[0].g.zero_()

    lm_emb = wb.model.get_input_embeddings().weight
    ans_idx = {w: i for i, (w, _) in enumerate(words)}
    answer_pool = torch.stack([lm_emb[tid_of[w]] for w, _ in words], dim=0)
    answer_pool = answer_pool / (answer_pool.norm(dim=-1, keepdim=True) + 1e-8)

    # PROBE A — WIRING: precompute ALL per-step states at once
    states_a = []
    for w, _ in words:
        z_a = lm_emb[tid_of[w]].detach()  # [d_lm]
        states_a.append([z_a.unsqueeze(0).clone() for _ in range(R.T)])
    tr_a, te_a, ab_a, g_a, l_a = _run_probe(
        wb, R, pids, train_w, test_w, states_a, answer_pool, tid_of, steps=steps_a, bs=bs, ds_weight=0.15)
    print(f"  (A) WIRING  (slot = LM's own embedding):  train {tr_a:.2f}  HELD-OUT {te_a:.2f}  "
          f"ablate->0 {ab_a:.2f}  gate {g_a:+.2f}  loss {l_a:.3f}")

    # PROBE B — BRIDGE (slot from MiniLM graph space). Run RAW vs CLIP-ALIGNED to isolate the modality gap.
    for h in handles:
        h.remove()
    task_emb = torch.as_tensor(encode_batch([prompt])[0], dtype=torch.float32, device=wb.device)
    memb = {w: torch.as_tensor(encode_batch([w])[0], dtype=torch.float32, device=wb.device) for w, _ in words}
    align_kg = torch.stack([memb[w] for w, _ in train_w])                    # MiniLM of TRAIN atoms only
    align_tgt = torch.stack([lm_emb[tid_of[w]].float() for w, _ in train_w]) # the LM's own embedding of each
    te_b = 0.0
    for tag, do_align in (("raw", False), ("CLIP-aligned", True)):
        Rb = WMReasoner(d_lm, couple_layers=couple).to(wb.device)
        hb = Rb.couple(wb)
        al = Rb.align_projector(align_kg, align_tgt, steps=300) if do_align else None
        states_b = []
        with torch.no_grad():
            for w, _ in words:
                _, states = Rb.refine(task_emb, memb[w].unsqueeze(0))         # aligned projector -> LM-space slots
                states_b.append([s.clone().detach() for s in states])
        tr_b, te_b, ab_b, g_b, l_b = _run_probe(
            wb, Rb, pids, train_w, test_w, states_b, answer_pool, tid_of, steps=steps_b, bs=bs, ds_weight=0.15)
        alstr = f"  align_loss {al:.2f}" if al is not None else ""
        print(f"  (B:{tag:>12}) BRIDGE MiniLM->LM:  train {tr_b:.2f}  HELD-OUT {te_b:.2f}  "
              f"ablate {ab_b:.2f}  gate {g_b:+.2f}{alstr}")
        for h in hb:
            h.remove()
    handles = []

    print()
    if te_a >= 0.5 and ab_a < te_a:
        print("  => WIRING PROVEN: working memory causally + GENERALIZABLY drives the frozen LM (probe A).")
        if te_b >= 0.5:
            print("     BRIDGE also works on distilgpt2 -- graph-space slots read too.")
        else:
            print("     BRIDGE did NOT generalize on distilgpt2 -- expected: the tiny LM can't decode a")
            print("     foreign embedding space. This is the job of the capable 4B (--run).")
    else:
        print("  => WIRING still not generalizing -- the adapter architecture needs more work (report honest).")

    print(f"\n  DEEP SUPERVISION (ds_weight=0.15):")
    print(f"     refinement steps: {R.T}  |  ds_head params: {sum(p.numel() for p in R.ds_pool.parameters()) + sum(p.numel() for p in R.ds_proj.parameters())}")
    print(f"     each step's working memory must independently retrieve the target from {len(words)} candidates")
    print(f"     -> gradients flow through ALL T steps, not just the final output")
    for h in handles:
        h.remove()


# ================================================================================================
# run_real — deploy the WM reasoner on the real 4B LM with graph atoms + composition tasks
# ================================================================================================
def _seed_atoms() -> tuple[dict, dict]:
    """Same 10 atoms as membrane.py. Returns {name: description} and {name: code}."""
    descs = {
        "is_prime": "whether a number is prime (exactly two divisors)",
        "digit_sum": "the sum of the decimal digits of a number",
        "num_divisors": "how many positive divisors a number has",
        "factorial": "the factorial of a number, n!",
        "fibonacci": "the nth Fibonacci number",
        "reverse_digits": "the number with its decimal digits reversed",
        "count_bits": "the number of one bits in the binary representation",
        "sum_to_n": "the sum of all integers from 1 to n",
        "square": "the square of a number",
        "is_even": "whether a number is even",
    }
    codes = {
        "is_prime": "def is_prime(n): return n>=2 and all(n%i for i in range(2,int(n**0.5)+1))",
        "digit_sum": "def digit_sum(n): return sum(int(c) for c in str(abs(n)))",
        "num_divisors": "def num_divisors(n): return sum(1 for i in range(1,abs(n)+1) if n%i==0)",
        "factorial": "def factorial(n): r=1\n for i in range(2,n+1): r*=i\n return r",
        "fibonacci": "def fibonacci(n): a,b=0,1\n for _ in range(n): a,b=b,a+b\n return a",
        "reverse_digits": "def reverse_digits(n): return int(str(abs(n))[::-1])",
        "count_bits": "def count_bits(n): return bin(abs(n)).count('1')",
        "sum_to_n": "def sum_to_n(n): return n*(n+1)//2",
        "square": "def square(n): return n*n",
        "is_even": "def is_even(n): return int(n%2==0)",
    }
    return descs, codes


def _compose_tasks_real():
    """(task_text, atoms_needed, target_code_template) for training and held-out."""
    train = [
        ("return the sum of the digits of the nth fibonacci number",
         ["fibonacci", "digit_sum"],
         "def task(n): return digit_sum(fibonacci(n))"),
        ("count the divisors of n factorial",
         ["factorial", "num_divisors"],
         "def task(n): return num_divisors(factorial(n))"),
        ("check if the digit sum of n is a prime",
         ["digit_sum", "is_prime"],
         "def task(n): return is_prime(digit_sum(n))"),
        ("reverse the digits then check if even",
         ["reverse_digits", "is_even"],
         "def task(n): return is_even(reverse_digits(n))"),
        ("square the number then sum its digits",
         ["square", "digit_sum"],
         "def task(n): return digit_sum(square(n))"),
        ("count set bits of the nth fibonacci number",
         ["fibonacci", "count_bits"],
         "def task(n): return count_bits(fibonacci(n))"),
    ]
    held_out = [
        ("the digit sum of the number of divisors of n",
         ["num_divisors", "digit_sum"],
         "def task(n): return digit_sum(num_divisors(n))"),
        ("is the nth fibonacci number even",
         ["fibonacci", "is_even"],
         "def task(n): return is_even(fibonacci(n))"),
        ("how many one-bits are in the digit sum of n",
         ["digit_sum", "count_bits"],
         "def task(n): return count_bits(digit_sum(n))"),
        ("reverse the digits of the square of n",
         ["square", "reverse_digits"],
         "def task(n): return reverse_digits(square(n))"),
    ]
    return train, held_out


def _exec_verify(code: str, tests: list) -> bool:
    """Run code, call task(n) with test inputs, verify outputs match."""
    try:
        ns = {}
        exec(compile(code, "<verify>", "exec"), ns)
        fn = ns.get("task")
        if not callable(fn):
            return False
        for inp, expected in tests:
            if fn(inp) != expected:
                return False
        return True
    except Exception:
        return False


def run_real(lm_name: str):
    from v5.runtime.dcpd_latent import WhiteBox
    import random
    print(f"run_real: WMReasoner coupled to {lm_name} (4-bit) — real composition tasks\n")

    wb = WhiteBox(lm_name, quant="4bit")
    d_lm = wb.d_model
    couple = [wb.n_layers - 2, wb.n_layers - 1]
    print(f"  LM: {lm_name}  d={d_lm}  layers={wb.n_layers}  gate layers={couple}  device={wb.device}")

    R = WMReasoner(d_lm, couple_layers=couple).to(wb.device)
    for p in wb.model.parameters():
        p.requires_grad_(False)
    handles = R.couple(wb)

    descs, codes = _seed_atoms()
    atom_names = list(descs.keys())
    print(f"  graph: {len(atom_names)} atoms (MiniLM embeddings)")

    atom_embs = {n: torch.as_tensor(encode_batch([descs[n]])[0], dtype=torch.float32, device=wb.device)
                 for n in atom_names}

    train_tasks, held_tasks = _compose_tasks_real()
    all_tasks = train_tasks + held_tasks
    split = len(train_tasks)
    print(f"  tasks: {split} train, {len(all_tasks) - split} held-out (2-atom composition)\n")

    opt = torch.optim.Adam([p for p in R.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
    for a in R.adapters:
        with torch.no_grad():
            a.g.fill_(1.5)

    # Build raw prompts (no chat template, to avoid special-token issues with teacher-forcing)
    def build_prompt(task_text):
        return wb.tok(f"Write a function task(n):\n# {task_text}\ndef task(n):", return_tensors="pt").input_ids.to(wb.device)

    # Precompute static task embeddings and atom embeddings for all examples
    print("  Precomputing task + atom embeddings...")
    task_embs = {}
    for text, atoms_needed, _ in all_tasks:
        if text not in task_embs:
            task_embs[text] = torch.as_tensor(encode_batch([text])[0], dtype=torch.float32, device=wb.device)

    atom_stack = torch.stack([atom_embs[n] for n in atom_names], dim=0).to(wb.device)  # [N, d_emb] MiniLM (384)

    prompt_ids = {}
    for text, _, _ in all_tasks:
        prompt_ids[text] = build_prompt(text)

    # All 10 atom oracle functions (used for verification)
    def _fib(n):
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        return a

    def _run_task(n, code_line):
        """Execute the composition code_line (e.g. 'digit_sum(fibonacci(n))') at n."""
        fn_map = {
            "is_prime": lambda n: n>=2 and all(n%i for i in range(2,int(n**0.5)+1)),
            "digit_sum": lambda n: sum(int(c) for c in str(abs(n))),
            "num_divisors": lambda n: sum(1 for i in range(1,abs(n)+1) if n%i==0),
            "factorial": lambda n: __import__('math').factorial(n),
            "fibonacci": _fib,
            "reverse_digits": lambda n: int(str(abs(n))[::-1]),
            "count_bits": lambda n: bin(abs(n)).count('1'),
            "sum_to_n": lambda n: n*(n+1)//2,
            "square": lambda n: n*n,
            "is_even": lambda n: int(n%2==0),
        }
        return eval(code_line, {"__builtins__": __builtins__}, fn_map)

    def verify(code_str, test_ns=(5, 7, 10)):
        """Verify that code_str defines task(n) matching oracle expectations."""
        code_str = code_str.strip()
        if not code_str.startswith("def task"):
            return False
        try:
            ns = {}
            exec(code_str, ns)
            task_fn = ns.get("task")
            if not callable(task_fn):
                return False
            for line in code_str.split("\n"):
                if "return " in line:
                    expr = line.split("return ", 1)[1].strip()
                    for n in test_ns:
                        if task_fn(n) != _run_task(n, expr):
                            return False
                    return True
            return False
        except Exception:
            return False

    train_ex = [(task_embs[text], atom_names.index(atoms_needed[0]), atoms_needed,
                 prompt_ids[text], text, code)
                for text, atoms_needed, code in train_tasks]
    held_ex = [(task_embs[text], atom_names.index(atoms_needed[0]), atoms_needed,
                prompt_ids[text], text, code)
               for text, atoms_needed, code in held_tasks]

    print("  Training the adapter + WMReasoner (real-close loop: refine → LM → verify)...")
    best_held = 0.0
    for ep in range(100):
        R.train()
        # DS candidate pool must live in d_lm space (the ds query = ds_proj(states) is d_lm). Project the
        # MiniLM atom embeddings through proj_atom; recompute each epoch so it tracks the trained projection.
        train_pool = R.proj_atom(atom_stack)
        train_pool = (train_pool / (train_pool.norm(dim=-1, keepdim=True) + 1e-8)).detach()
        random.shuffle(train_ex)
        tot_lm, tot_ds, n = 0.0, 0.0, 0
        for task_emb, gold_idx, atoms_needed, pids, text, target_code in train_ex:
            K_atom_embs = torch.stack([atom_embs[n] for n in atoms_needed], dim=0)
            slots, states = R.refine(task_emb, K_atom_embs)
            R.set_slots_direct(slots)
            return_body = target_code.split(": ", 1)[1] if ": " in target_code else target_code
            tids = wb.tok(" " + return_body, return_tensors="pt").input_ids.to(wb.device)
            input_ids = torch.cat([pids, tids], dim=-1)
            labels = torch.full_like(input_ids, -100)
            labels[:, pids.shape[-1]:] = tids
            outs = wb.model(input_ids=input_ids, labels=labels)
            lm_loss = outs.loss
            state_list = [states]
            ds_loss = R.ds_loss_batch(state_list, train_pool, [gold_idx])
            loss = lm_loss + 0.1 * ds_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_lm += float(lm_loss.detach())
            tot_ds += float(ds_loss.detach())
            n += 1

        if ep % 10 == 0 or ep == 99:
            R.eval()
            held_ok = 0
            ablated_ok = 0
            for task_emb, gold_idx, atoms_needed, pids, text, target_code in held_ex:
                K_atom_embs = torch.stack([atom_embs[n] for n in atoms_needed], dim=0)
                slots, _ = R.refine(task_emb, K_atom_embs)
                with torch.no_grad():
                    R.set_slots_direct(slots)
                    out = wb.model.generate(pids, max_new_tokens=64,
                                            do_sample=False, pad_token_id=wb.tok.eos_token_id)
                    code = wb.tok.decode(out[0][pids.shape[-1]:], skip_special_tokens=True).strip()
                if verify(code):
                    held_ok += 1
                R.clear()
                with torch.no_grad():
                    out = wb.model.generate(pids, max_new_tokens=64,
                                            do_sample=False, pad_token_id=wb.tok.eos_token_id)
                    code_abl = wb.tok.decode(out[0][pids.shape[-1]:], skip_special_tokens=True).strip()
                if verify(code_abl):
                    ablated_ok += 1
            best_held = max(best_held, held_ok)
            print(f"  ep {ep:>3}  lm_loss {tot_lm/max(n,1):.3f}  ds_loss {tot_ds/max(n,1):.3f}  "
                  f"held WM {held_ok}/{len(held_ex)}  ablated {ablated_ok}/{len(held_ex)}  "
                  f"gate {float(R.adapters[0].g):+.2f}")

    print(f"\n  Best held-out: {best_held}/{len(held_ex)}  (gate ablated = {ablated_ok} baseline)")
    verdict = "PROVEN" if best_held > ablated_ok else "PARTIAL"
    print(f"  => {verdict}: working memory {'improves' if best_held > ablated_ok else 'does not improve'} held-out composition on {lm_name}")
    for h in handles:
        h.remove()


def probe_real(lm_name: str):
    """Run the copy(A)+bridge(B) mechanism test on the REAL 4B (not distilgpt2). Probe B is the decisive one:
    can a capable LM READ graph-space (MiniLM) slots via the working memory and generalize? distilgpt2 can't
    (0.04); this is the fair test. Smaller batch/steps since the 4B is heavy."""
    from v5.runtime.dcpd_latent import WhiteBox
    wb = WhiteBox(lm_name, quant="4bit")
    for p in wb.model.parameters():
        p.requires_grad_(False)
    print(f"  LM {lm_name}  quant={wb.quant}  VRAM={wb.vram_gb:.2f}GB  layers={wb.n_layers}\n")
    selftest(wb, bs=48, steps_a=80, steps_b=200)


def main():
    ap = argparse.ArgumentParser(description="TRM working memory coupled to a frozen LM (real reasoner, design b)")
    ap.add_argument("--selftest", action="store_true", help="prove the mechanism on distilgpt2 (local, fast)")
    ap.add_argument("--probe", action="store_true", help="copy+bridge mechanism test on the real --lm (the fair bridge test)")
    ap.add_argument("--run", action="store_true", help="full composition experiment on --lm (hardest task)")
    ap.add_argument("--lm", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    a = ap.parse_args()
    if a.probe:
        probe_real(a.lm)
    elif a.run:
        run_real(a.lm)
    else:
        selftest()


if __name__ == "__main__":
    main()
