"""algo_grr_nstm -- THE LM MAKES EVERY DECISION; a slot memory modulates its LOGITS; the graph, the
self-built tool bank, and the real verifier stay exactly where they were.

This is the user's NSTM notebook idea, rebuilt on the real stack and pointed at the failure this
session actually measured.

WHAT CHANGED, AND WHY IT IS NOT THE NULL RESULT I WARNED ABOUT.
I argued against NSTM because the notebook trains it with F.cross_entropy on the LM's next token --
that is lm_loss, the exact signal that made this project's slots task-invariant (slot_cos 1.0000).
That objection dies here: the LM EMITS THE DECISIONS AS TEXT and NSTM is trained by REINFORCE ON
VERIFIED OUTCOMES (did the tool call work, did the file turn out to be the right one, did the REAL
test suite pass). No cross-entropy on tokens anywhere. The slots are shaped by consequences, not by
imitation.

WHY THE OLD DESIGN HAD TO GO. The previous controller's job was "pick a tool, pick one of 24 candidate
strings", dominated by a +0.45 reward for landing on the gold file. That is RETRIEVAL -- the exact
thing this project said it did not want ("NOT A RETRIEVER/RANKER but a cognitive thinker") -- and it
loses to a plain grep, which reaches gold 0.786 of the time. Measured end to end, the controller
opened the wrong file on the one instance the hand-driven pipeline solves, destroying the solve.
So the decisions move to the LM, and the learned part moves to where there IS cognitive work:
carrying state across steps, in particular WHAT ALREADY FAILED AND WHY.

THE TASK IS NOW FAILURE-DRIVEN REVISION, not file lookup:
  author a tool -> the REAL verifier runs -> feed the REAL failure back -> revise.
Later decisions structurally depend on earlier observations, which is the property this project's own
ablation history says is required for an observation-conditioned module to matter at all. No lexical
baseline substitutes for it: grep cannot read a traceback.

PRIOR PRESERVATION IS BY CONSTRUCTION. W_out is zero-init, so at step 0 the logit residual is exactly
zero and the system IS the frozen LM. The ablation `--abl no-nstm` is therefore the exact incumbent,
not an approximation of it. Four learned components in this project degraded their own baselines by
skipping this.

FALSIFIER, pre-registered: across-instance slot cosine must stay BELOW 0.99. If the slots collapse to
a constant the way every earlier latent here did, this is the same null and it should be reported as
one rather than rescued.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("HF_HOME", r"E:\cache\hf")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from v5.runtime.algo_grr_swetools import (container_file_text, instance_tests, t_grep, t_run_tests,
                                          test_command)
from v5.runtime.algo_grr_toolsmith import (ToolBank, _excerpt, gate_edit, run_edit_tool, strip_fences,
                                           vram_check)


# ── NSTM: slot memory that modulates the frozen LM's logits ──────────────────────────────────────
class NSTM(nn.Module):
    """K slots, sparse write, gated logit residual. Structure follows the prototype notebook; the
    differences are the ones that decide whether it works here.

      * W_out is ZERO-INIT -> delta_z == 0 at step 0 -> the model is exactly the frozen LM. The
        no-nstm ablation is then the true incumbent by construction.
      * slots persist ACROSS TOOL STEPS, not just across tokens in one forward pass. That is the
        point: what carries between steps is what failed last time.
      * trained by REINFORCE on verified outcomes -- never by cross-entropy on tokens.
    """

    def __init__(self, d_model: int, vocab_size: int, num_slots: int = 4, top_k: int = 2,
                 d_slot: int = 128, action_ids: list | None = None):
        super().__init__()
        self.d_model, self.vocab_size = d_model, vocab_size
        self.num_slots, self.top_k, self.d_slot = num_slots, top_k, d_slot
        # project the LM's wide hidden state down to a small slot space: a 3B has d_model 2048, and
        # K slots at full width would dwarf every learned component in this project.
        self.h_in = nn.Linear(d_model, d_slot)
        self.W_Q = nn.Linear(d_slot, d_slot, bias=False)
        self.W_K = nn.Linear(d_slot, d_slot, bias=False)
        self.W_V = nn.Linear(d_slot, d_slot, bias=False)
        self.W_g = nn.Linear(d_slot, 1)
        self.slot_gru = nn.GRUCell(d_slot, d_slot)
        self.obs_in = nn.Linear(384, d_slot)          # MiniLM view of a real observation
        # ACTION-RESTRICTED RESIDUAL. The first version emitted a residual over the FULL 151936-token
        # vocabulary: W_out alone was 128 x 151936 = 19.4M params (97% of the module), it materialised
        # a full-vocab tensor plus autograd graph per decoded token, and it pushed the run to 6.44 GB
        # against a 6.0 GB budget. It was also aimed at the wrong thing. The measured failure is not
        # which CODE tokens get written -- the LM authors correct patches -- it is that the model never
        # sequences to test(). That is a choice among a handful of action words. So the residual is
        # scattered onto ONLY the action tokens; the LM writes all content unmodulated.
        self.action_ids = list(action_ids or [])
        n_act = max(1, len(self.action_ids))
        self.register_buffer("act_idx", torch.tensor(self.action_ids or [0], dtype=torch.long))
        self.W_out = nn.Linear(d_slot, n_act, bias=False)
        nn.init.zeros_(self.W_out.weight)             # PRIOR PRESERVATION: delta_z == 0 at init

    def init_slots(self, device, dtype=torch.float32):
        return torch.zeros(1, self.num_slots, self.d_slot, device=device, dtype=dtype)

    def _sparse(self, scores):
        if self.top_k is None or self.top_k >= scores.size(-1):
            return F.softmax(scores, dim=-1)
        topv, topi = torch.topk(scores, self.top_k, dim=-1)
        masked = torch.full_like(scores, float("-inf"))
        masked.scatter_(-1, topi, topv)
        return F.softmax(masked, dim=-1)

    def write_observation(self, slots, obs_emb):
        """Fold a REAL tool observation into the slots between steps. This is the cross-step channel:
        a failed test's traceback lands here and is still present when the next tool call is decoded."""
        v = torch.tanh(self.obs_in(obs_emb)).view(1, 1, self.d_slot)
        q, k = self.W_Q(v), self.W_K(slots)
        a = self._sparse((q * k).sum(-1) / math.sqrt(self.d_slot))
        upd = a.unsqueeze(-1) * self.W_V(v)
        B, K, D = slots.shape
        return self.slot_gru(upd.reshape(B * K, D), slots.reshape(B * K, D)).reshape(B, K, D)

    def step(self, h_t, slots):
        """One decoded token: read the LM state, update slots, emit a gated logit residual."""
        h = torch.tanh(self.h_in(h_t)).view(1, 1, self.d_slot)
        q, k = self.W_Q(h), self.W_K(slots)
        a = self._sparse((q * k).sum(-1) / math.sqrt(self.d_slot))
        upd = a.unsqueeze(-1) * self.W_V(h)
        B, K, D = slots.shape
        new = self.slot_gru(upd.reshape(B * K, D), slots.reshape(B * K, D)).reshape(B, K, D)
        rq = self.W_Q(h)
        ra = F.softmax((rq * self.W_K(new)).sum(-1) / math.sqrt(self.d_slot), dim=-1)
        c = (ra.unsqueeze(-1) * new).sum(1)                       # [1, d_slot]
        g = torch.sigmoid(self.W_g(h.view(1, self.d_slot)))       # [1, 1]
        return new, g, self.W_out(c), a                           # dz is [1, n_action_tokens]

    def scatter_residual(self, logits, g, dz):
        """Add the gated residual onto the ACTION tokens only, in place of a full-vocab add.
        Returns logits unchanged when there are no action ids, so this can never silently no-op
        into something that looks trained."""
        if not self.action_ids:
            return logits
        return logits.index_add(1, self.act_idx, g * dz)


# ── the LM decides. every action is decoded text, constrained to what actually exists ─────────────
ACTION_HEAD = re.compile(r"([a-z_]+)\s*\(", re.S)


def parse_action(text: str, valid: list):
    """The LM's own words ARE the decision. Parsing is exact and unforgiving -- an unparseable or
    unknown action is a real failure the model must recover from, never silently repaired into
    something valid, which would hide the model's mistakes inside the harness.

    The argument is extracted by BALANCED-PAREN scanning, not a regex. A non-greedy `(.*?)\\)` stopped
    at the first inner ')' -- so `author("def edit(text): ...")` was truncated to `"def edit(text`,
    and EVERY authored tool was rejected as "did not contain def edit(text)". Measured: 12 of 12
    actions rejected across 3 instances, which read like an LM failure and was entirely a parser bug.
    Python code is full of parentheses; anything parsing code must count them."""
    text = text or ""
    for m in ACTION_HEAD.finditer(text):
        tool = m.group(1).strip()
        if tool not in valid:
            continue
        depth, i, start = 1, m.end(), m.end()
        in_s, esc = None, False
        while i < len(text) and depth:
            c = text[i]
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif in_s:
                if c == in_s:
                    in_s = None
            elif c in "\"'":
                in_s = c
            elif c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            i += 1
        if depth:                                   # unterminated call: take the rest of the text
            arg = text[start:]
        else:
            arg = text[start:i - 1]
        arg = arg.strip()
        if len(arg) >= 2 and arg[0] == arg[-1] and arg[0] in "\"'":
            arg = arg[1:-1]
        return tool, arg, ""
    m = ACTION_HEAD.search(text)
    if m:
        return None, None, f"unknown tool {m.group(1)!r}; valid tools are {valid}"
    return None, None, "no action of the form tool(arg) was emitted"


class NSTMAgent:
    """Frozen LM + NSTM. The LM emits actions; NSTM shifts its logits; real tools execute; real
    observations return; the verifier decides. Only NSTM has gradients."""

    def __init__(self, lm, nstm: NSTM, use_nstm: bool = True):
        self.lm, self.nstm, self.use_nstm = lm, nstm, use_nstm

    @torch.no_grad()
    def _encode_obs(self, text: str):
        from embedder import encode_batch
        return torch.tensor(encode_batch([(text or "")[:300]])[0], dtype=torch.float32,
                            device=self.nstm.W_out.weight.device)

    def decode_action(self, prompt: str, slots, max_new: int = 320, temperature: float = 0.7,
                      sample: bool = True):
        """Decode one action token by token, adding NSTM's gated residual to each step's logits.
        Returns (text, slots, logps, mean_gate). Grad flows ONLY through the residual."""
        tok, model = self.lm.tok, self.lm.model
        msgs = [{"role": "user", "content": prompt}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        ids = (enc if torch.is_tensor(enc) else enc["input_ids"]).to(self.lm.device)
        start, logps, gates = ids.shape[1], [], []
        # KV CACHE. The first version re-ran the FULL forward with output_hidden_states=True for every
        # token, which is O(n^2) and holds every layer's activations -- it drove the run to 6.44 GB and
        # tripped the 6.0 GB budget check. Feeding one token at a time against past_key_values keeps
        # the per-step cost flat.
        past, cur = None, ids
        for _ in range(max_new):
            with torch.no_grad():
                out = model(cur, past_key_values=past, use_cache=True, output_hidden_states=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :].float()
            if self.use_nstm:
                h = out.hidden_states[-1][:, -1, :].float()
                slots, g, dz, _ = self.nstm.step(h, slots)
                logits = self.nstm.scatter_residual(logits, g, dz)  # 0 at init -> exactly the base LM
                gates.append(g.view(-1))
            probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
            nxt = torch.multinomial(probs, 1) if sample else logits.argmax(-1, keepdim=True)
            if self.use_nstm:
                logps.append(torch.log(probs.gather(-1, nxt).clamp_min(1e-9)).view(-1))
            ids = torch.cat([ids, nxt], 1)
            cur = nxt
            if nxt.item() == tok.eos_token_id:
                break
        del past
        text = tok.decode(ids[0, start:], skip_special_tokens=True).strip()
        mg = float(torch.stack(gates).mean()) if gates else 0.0
        return text, slots, logps, mg


def slot_signature(slots) -> np.ndarray:
    return slots.detach().mean(1).view(-1).float().cpu().numpy()


# ── the environment: real repo, real graph, real self-built tools, real verifier ──────────────────
def build_prompt(issue: str, path: str, src: str, history: list, tools: list, banked: list,
                 have_patch: bool = False) -> str:
    """SLOTS REPLACE CONTEXT, they do not duplicate it.

    The first version re-sent the issue, the full ~2.6k-char code excerpt and the whole history on
    EVERY step, so each step re-filled a long KV cache from scratch -- and then added slots on top.
    That is the inverse of the thesis: NSTM was pure overhead, paying for a second copy of state the
    prompt was already carrying. Once a patch exists the code excerpt has done its job (the next
    decision is whether to VERIFY, not what to write), so it is dropped and the recurrent slots carry
    what mattered. Measured token cost of the excerpt is reported by --run.
    """
    hist = "\n".join(f"  step {i}: {h}" for i, h in enumerate(history[-2:])) or "  (nothing yet)"
    bank_txt = "\n".join(f"  - {d}" for d in banked[:3]) or "  (bank empty)"
    if have_patch:
        return (f"You are fixing a real bug in {path}.\n\nBUG REPORT:\n{issue[:600]}\n\n"
                f"WHAT HAS HAPPENED SO FAR:\n{hist}\n\n"
                f"You ALREADY HAVE a patch that applies and parses. It has NOT been verified.\n"
                f"Choose ONE action:\n"
                f"  test()       run the REAL test suite on your current patch\n"
                f"  author(...)  replace it with a different `def edit(text: str) -> str`\n"
                f"  stop()       give up\n\n"
                f"Emit exactly ONE action call and nothing else.")
    # NO ANGLE-BRACKET PLACEHOLDERS. The first version wrote `author(<python>)` as the syntax spec and
    # the 3B copied it literally -- every action parsed to the string "<python>" and all 12 attempts
    # across 3 instances were rejected. A concrete worked example is copied usefully; a placeholder is
    # copied uselessly.
    return (f"You are fixing a real bug in the file {path}.\n\nBUG REPORT:\n{issue[:1100]}\n\n"
            f"RELEVANT CODE FROM {path} (copy `old` strings from here EXACTLY, indentation included):\n"
            f"{_excerpt(src, issue)}\n\n"
            f"WHAT HAS HAPPENED SO FAR:\n{hist}\n\n"
            f"VERIFIED TOOLS ALREADY IN THE GRAPH (reusable):\n{bank_txt}\n\n"
            f"Choose ONE action:\n"
            f"  author(...)  supply a python function `def edit(text: str) -> str`. `text` is the\n"
            f"               ENTIRE current contents of {path}; return the ENTIRE fixed contents.\n"
            f"               Make one targeted change; do not rewrite the file.\n"
            f"  reuse(...)   a short query; retrieves and applies a verified tool from the graph\n"
            f"  test()       run the REAL test suite against your current patch\n"
            f"  stop()       give up\n\n"
            f"Example of a correct action:\n"
            f"author(def edit(text: str) -> str:\n"
            f"    old = \"        if isinstance(value, bytes):\"\n"
            f"    new = \"        if isinstance(value, (bytes, memoryview)):\"\n"
            f"    return text.replace(old, new, 1))\n\n"
            f"Emit exactly ONE action call and nothing else.")


def run_episode(agent: NSTMAgent, row: dict, bank: ToolBank, max_steps: int = 4,
                sample: bool = True, verbose: bool = False):
    """One real trajectory. Every decision is the LM's decoded text; every observation is real."""
    inst, gold, issue = row["instance_id"], row["gold"], row["problem"]
    src = container_file_text(inst, gold)
    if not src:
        return {"error": "no container file"}, [], []
    dev = agent.nstm.W_out.weight.device
    slots = agent.nstm.init_slots(dev)
    state = {"repo": row["repo"], "instance_id": inst}
    history, logps, sigs, ptoks = [], [], [], 0
    tools = ["author", "reuse", "test", "stop"]
    banked = [bank.g.atoms[n].description for n in getattr(bank.g, "atoms", {})][:3]

    for step in range(max_steps):
        prompt = build_prompt(issue, gold, src, history, tools, banked,
                              have_patch=bool(state.get("patch_tool")))
        ptoks += len(agent.lm.tok(prompt).input_ids)
        text, slots, lp, gate = agent.decode_action(prompt, slots, sample=sample)
        logps += lp
        sigs.append(slot_signature(slots))
        tool, arg, err = parse_action(text, tools)
        if tool is None:
            obs = f"REJECTED: {err}"
        elif tool == "stop":
            history.append("stop()")
            break
        elif tool == "author":
            code = strip_fences(arg)
            if "def edit" not in code:
                obs = "REJECTED: your action did not contain `def edit(text)`"
            else:
                ok, res = run_edit_tool(code, src)
                if not ok:
                    obs = f"TOOL RAISED: {str(res)[:200]}"
                else:
                    good, note = gate_edit(src, res)
                    if not good:
                        obs = f"GATE REJECTED: {note[:200]}"
                    else:
                        state["patch_tool"] = (gold, code)
                        state["authored_code"] = code
                        obs = f"OK: tool applies and parses ({note}). You can now call test()."
        elif tool == "reuse":
            hits = bank.retrieve(arg or issue, k=3)
            obs = "no banked tool applied here"
            for nm, code, cos in hits:
                ok, res = run_edit_tool(code, src)
                if ok and gate_edit(src, res)[0]:
                    state["patch_tool"] = (gold, code)
                    state["reused_tool"] = nm
                    obs = f"OK: reused banked tool {nm} (cos {cos:.2f}). You can now call test()."
                    break
        elif tool == "test":
            if not state.get("patch_tool"):
                obs = "REJECTED: no patch yet -- author() or reuse() first"
            else:
                ok, out = t_run_tests(state, None)
                state["tests_passed"] = ok
                tail = " | ".join([l for l in out.strip().splitlines() if l.strip()][-6:])
                obs = ("PASS: the real test suite passed." if ok
                       else f"FAIL: the real test suite still fails. Output:\n{tail[:700]}")
        history.append(f"{text[:80]} -> {obs[:400]}")
        if verbose:
            print(f"    [{step}] {(tool or '?')}: {obs[:150]}", flush=True)
        # THE CROSS-STEP CHANNEL: the real observation is written into the slots, so the next action
        # is decoded with the failure already in memory.
        slots = agent.nstm.write_observation(slots, agent._encode_obs(obs))
        if state.get("tests_passed"):
            break

    # MEASUREMENT, NOT A DECISION -- and labelled as such wherever it is reported. The baseline LM
    # authors a valid patch and then keeps authoring instead of ever calling test(), so the episode
    # ends with a patch that was never verified. Running the verifier once at the end separates two
    # very different failures: "the authored tool was wrong" from "the model never sequenced to
    # verification". `tests_passed` stays the agent's own result; this writes a separate key so the
    # two can never be conflated in a report.
    if state.get("patch_tool") and "tests_passed" not in state:
        ok, _ = t_run_tests(state, None)
        state["tests_passed_unprompted"] = ok
    state["prompt_tokens"] = ptoks
    return state, logps, sigs


def reward(state: dict) -> float:
    """Verified only. Passing the REAL suite dominates; producing a gated patch is partial credit.

    The +0.3 for ACTUALLY INVOKING the verifier is what makes this trainable at all. Without it every
    episode authored a patch, none reached test(), and every reward was ~0.2 -- REINFORCE subtracts
    the mean, so the advantage was ~0 and NSTM received essentially NO GRADIENT across 9 episodes.
    That run could not have tested the idea either way. Rewarding the act of verifying (not its
    outcome, which is scored separately and far higher) creates the variance the estimator needs, and
    it rewards exactly the behaviour that was missing.
    """
    r = 0.0
    if state.get("patch_tool"):
        r += 0.2
    if "tests_passed" in state:            # the agent chose to verify -- outcome-independent
        r += 0.3
    if state.get("tests_passed"):
        r += 1.0
    return r


def train(agent: NSTMAgent, rows: list, bank: ToolBank, epochs: int = 3, lr: float = 1e-4,
          verbose: bool = True):
    """REINFORCE with a mean baseline, on NSTM parameters ONLY. No loss touches the LM."""
    opt = torch.optim.Adam(agent.nstm.parameters(), lr=lr)
    for ep in range(epochs):
        recs = []
        for row in rows:
            st, lp, _ = run_episode(agent, row, bank, sample=True)
            recs.append((reward(st), lp))
        live = [(r, lp) for r, lp in recs if lp]
        base = sum(r for r, _ in live) / max(1, len(live))
        if live and any(abs(r - base) > 1e-8 for r, _ in live):
            loss = torch.stack([-(r - base) * torch.stack(lp).sum() for r, lp in live]).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.nstm.parameters(), 1.0)
            opt.step()
        if verbose:
            print(f"  [nstm] epoch {ep + 1}  mean reward {base:.3f}", flush=True)
    return agent


def _selftest() -> bool:
    print("algo_grr_nstm --selftest: LM decides, NSTM modulates, verifier rules\n")
    ok = True

    def chk(t, c, d=""):
        nonlocal ok
        ok &= bool(c)
        print(f"  [{'PASS' if c else 'FAIL'}] {t}{('  - ' + d) if d else ''}")

    n = NSTM(d_model=64, vocab_size=100, num_slots=4, top_k=2, d_slot=32, action_ids=[5, 6, 7, 8])
    s = n.init_slots("cpu")
    h = torch.randn(1, 64)
    s2, g, dz, a = n.step(h, s)
    chk("[1] PRIOR PRESERVED: the logit residual is exactly zero at init (W_out zero-init)",
        float(dz.abs().max()) == 0.0, f"max|dz| = {float(dz.abs().max()):.2e}")
    base = torch.randn(1, 100)
    chk("[1b] scatter leaves the logits bit-identical at init",
        torch.equal(n.scatter_residual(base.clone(), g, dz), base))
    with torch.no_grad():
        n.W_out.weight.fill_(1.0)
    s3, g3, dz3, _ = n.step(h, s)
    out = n.scatter_residual(base.clone(), g3, dz3)
    touched = (out != base).nonzero()[:, 1].tolist()
    chk("[1c] a TRAINED residual touches ONLY the action tokens, not the whole vocabulary",
        touched == [5, 6, 7, 8], f"changed token ids {touched}")
    chk("[1d] residual is action-sized, not vocab-sized (was 128x151936 = 19.4M params)",
        n.W_out.weight.shape[0] == 4, f"W_out {tuple(n.W_out.weight.shape)}")
    chk("[2] slots still MOVE at init even though the residual is zero",
        float((s2 - s).abs().max()) > 1e-6)
    chk("[3] write is SPARSE: only top_k slots receive a write",
        int((a > 1e-6).sum()) <= n.top_k, f"{int((a > 1e-6).sum())} of {n.num_slots} slots")

    o1 = n.write_observation(s, torch.randn(384))
    o2 = n.write_observation(s, torch.randn(384))
    chk("[4] DIFFERENT observations drive slots to DIFFERENT states (the cross-step channel)",
        float((o1 - o2).abs().max()) > 1e-6, f"delta {float((o1 - o2).abs().max()):.4f}")

    t, a_, e = parse_action('author("def edit(text): return text")', ["author", "test", "stop"])
    chk("[5] the LM's own text IS the decision (parsed, not chosen from a menu)",
        t == "author" and "def edit" in a_)
    # the bug that made 12 of 12 real actions look like LM failures: a non-greedy regex stopped at the
    # first INNER ')' and truncated every authored tool.
    code = "def edit(text):\n    return text.replace('a', 'b')"
    t2, a2, _ = parse_action(f'author("{code}")', ["author", "test", "stop"])
    chk("[5b] balanced-paren scan keeps code containing parentheses INTACT",
        t2 == "author" and a2.strip().endswith("'b')"), (a2 or "")[-28:].replace("\n", " "))
    t, _, e = parse_action("please run the tests now", ["author", "test", "stop"])
    chk("[6] an unparseable action is a REAL failure, never silently repaired", t is None and e)
    t, _, e = parse_action("delete_everything('/')", ["author", "test", "stop"])
    chk("[7] an unknown tool is rejected outright", t is None and "unknown tool" in e)

    r_none, r_patch = reward({}), reward({"patch_tool": 1})
    r_tried = reward({"patch_tool": 1, "tests_passed": False})    # verified, and it FAILED
    r_pass = reward({"patch_tool": 1, "tests_passed": True})
    chk("[8] reward is verified-only and dominated by the REAL test suite",
        r_none == 0.0 and r_patch == 0.2 and r_pass == 1.5 and r_pass > r_tried)
    chk("[8b] INVOKING the verifier is rewarded even when it fails -- this is the reward VARIANCE "
        "whose absence gave NSTM ~0 gradient over 9 episodes",
        r_tried > r_patch, f"patch-only {r_patch} vs verified-but-failed {r_tried}")

    tot = sum(p.numel() for p in NSTM(2048, 151936, d_slot=128, action_ids=[1, 2, 3, 4]).parameters())
    chk("[9] the module is small again now the residual is action-sized, not vocab-sized",
        tot < 1_000_000, f"{tot:,} params (was 19,907,713 with a full-vocab W_out)")

    print(f"\n  ALGO_GRR_NSTM -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description="LM decides; NSTM slots modulate its logits.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--lm", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--abl", type=str, default="", help="no-nstm = the exact frozen-LM incumbent")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    if not a.run:
        ap.print_help(); return

    CACHED = ["django__django-10924", "django__django-11133", "django__django-11999"]
    rows = [r for r in json.loads((Path(_ROOT) / "artifacts" / "swebench_loc_big.json")
                                  .read_text(encoding="utf-8")) if r["instance_id"] in CACHED]
    torch.manual_seed(a.seed); random.seed(a.seed)
    from v5.runtime.dcpd_latent import WhiteBox
    print(f"loading {a.lm} (4-bit)...", flush=True)
    lm = WhiteBox(a.lm, quant="4bit")
    print(f"  VRAM {vram_check('after LM load'):.2f} GB", flush=True)

    d_model = lm.model.config.hidden_size
    # the tokens the residual is allowed to touch: the first token of each action word, in the forms
    # the model actually emits (bare and space-prefixed). Everything else decodes unmodulated.
    act_ids = sorted({lm.tok(v, add_special_tokens=False).input_ids[0]
                      for w in ("author", "reuse", "test", "stop") for v in (w, " " + w)})
    print(f"action tokens the residual may touch: {act_ids}", flush=True)
    nstm = NSTM(d_model, lm.model.config.vocab_size, num_slots=4, top_k=2, d_slot=128,
                action_ids=act_ids).to(lm.device)
    use = "no-nstm" not in a.abl
    print(f"NSTM: {sum(p.numel() for p in nstm.parameters()):,} params  active={use}", flush=True)
    agent = NSTMAgent(lm, nstm, use_nstm=use)
    bank = ToolBank()

    if use:
        train(agent, rows, bank, epochs=a.epochs)

    print(f"\n{'=' * 74}\nEVAL -- every decision is the LM's; NSTM only shifts logits\n{'=' * 74}",
          flush=True)
    solved, patched, sigs, unprompted, ptok = 0, 0, [], 0, 0
    for row in rows:
        st, _, sg = run_episode(agent, row, bank, max_steps=a.steps, sample=False, verbose=True)
        sigs += sg[-1:]
        ptok += st.get("prompt_tokens", 0)
        patched += int(bool(st.get("patch_tool")))
        if st.get("tests_passed"):
            solved += 1
        if st.get("tests_passed") or st.get("tests_passed_unprompted"):
            unprompted += 1
            bank.bank(row["problem"], row["repo"], row["gold"], st.get("authored_code", ""), "verified")
        print(f"  {row['instance_id']}: patched={bool(st.get('patch_tool'))} "
              f"AGENT_CALLED_TEST={'tests_passed' in st} "
              f"PASSES={bool(st.get('tests_passed') or st.get('tests_passed_unprompted'))}", flush=True)

    print(f"\n  gated patches                      : {patched}/{len(rows)}")
    print(f"  REAL SOLVES (agent ran the verifier): {solved}/{len(rows)}")
    print(f"  patch passes when verified for it   : {unprompted}/{len(rows)}   "
          f"[harness-initiated check, NOT an agent decision]")
    print(f"  PROMPT TOKENS fed to the LM        : {ptok}  ({ptok / max(1, len(rows)):.0f}/instance)"
          f"   <- the load the slots are supposed to remove")
    print(f"  banked tools  : {len(bank)}")
    if use and len(sigs) > 1:
        M = np.stack(sigs)
        M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        cs = [float(M[i] @ M[j]) for i in range(len(M)) for j in range(i + 1, len(M))]
        mc = float(np.mean(cs))
        print(f"\n  FALSIFIER across-instance slot cosine: {mc:.4f}  "
              f"-> {'COLLAPSED (same null as every earlier latent)' if mc > 0.99 else 'slots stay task-specific'}")
    print(f"  VRAM {vram_check('end of run'):.2f} GB")


if __name__ == "__main__":
    main()
