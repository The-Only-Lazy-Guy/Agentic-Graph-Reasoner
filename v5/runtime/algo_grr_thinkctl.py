"""algo_grr_thinkctl — THE THINKER AS CONTROLLER. It drives real tools over a real repo, its state is
advanced by real observations, and the LM only fills holes it cannot.

DESIGNED AGAINST THE SPECIFIC FAILURES THIS PROJECT MEASURED, not against the word "TRM":

  trained through lm_loss -> the latent became a CONSTANT (across-task slot cosine 1.0000, slot-swap
      changed 1 of 16). Here NO loss touches the LM. The reward is whether a tool call actually
      worked and whether the edit survived its gates.
  output was a score over candidates -> re-ranking, null in every arm tried (spiking planner, hop
      policy, slot gate, RL descent, CommitHead). Here the output is an ACTION that changes the
      world: which tool, and which argument. Running it moves the state somewhere new.
  recursion was a contraction that washed the input out ("the T-cycle recurrence is a fixed-point
      attractor" -- algo_trm's own comment). Here z is advanced by a GRU from the REAL OBSERVATION
      the tool returned, so two different repos cannot share a trajectory.
  latent injection into the LM -> dead across four attempts. Here the interface is TEXT in both
      directions: the LM reads a rendered trace, and returns a string.
  the task had no room -> every no-exec ablation came back exactly null. Here `--abl no-obs` blinds
      the observation features, and the controller then cannot know whether its last grep found 1
      file or 40, so it MUST degrade. That is the falsifier.

ARGUMENTS ARE POINTED AT, NOT GENERATED. Tool arguments come from a candidate set derived from the
issue text and from what previous tools returned (identifiers in the issue, paths grep printed). The
controller picks an index. Nothing can hallucinate a path that does not exist. The LM is asked for
exactly one thing -- the replacement CODE in an edit -- which is the only argument no pointer can
supply.
"""
from __future__ import annotations

import argparse
import json
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

from v5.runtime.algo_grr_swetools import (_repo_dir, swe_registry, t_edit, t_find_def, t_grep,
                                          t_list_dir, t_read_file)

TOOLS = ["grep", "find_def", "read_file", "edit", "stop"]
N_TOOL = len(TOOLS)
N_OBS = 10
IDT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def obs_features(state: dict, blind: bool = False) -> np.ndarray:
    """What the last tool call actually returned. This is the ONLY thing that distinguishes step t
    from step t-1, so blinding it (--abl no-obs) leaves the controller with nothing to act on."""
    if blind:
        return np.zeros(N_OBS, dtype=np.float32)
    hits = state.get("last_hits", [])
    return np.array([
        1.0 if state.get("open_file") else 0.0,
        min(len(hits), 40) / 40.0,
        1.0 if len(hits) == 1 else 0.0,              # a unique hit is the strong signal
        1.0 if state.get("last_ok") else 0.0,
        1.0 if state.get("patch") else 0.0,
        1.0 if state.get("edit_ambiguous") else 0.0,
        1.0 if state.get("edit_unparsable") else 0.0,
        min(state.get("steps", 0), 8) / 8.0,
        min(len(state.get("seen_files", ())), 8) / 8.0,
        1.0 if state.get("gold_open") else 0.0,      # TRAIN-ONLY reward shaping, zeroed at eval
    ], dtype=np.float32)


class ThinkerController(nn.Module):
    """z_{t+1} = GRU(z_t, [action_emb, observation]) -> (tool logits, argument pointer).

    Small on purpose: this project has degraded its own baseline four times by putting thousands of
    parameters on a few hundred examples. Capacity is not the lever here; the observation channel is.
    """

    def __init__(self, d: int = 48, n_arg: int = 24):
        super().__init__()
        self.d, self.n_arg = d, n_arg
        self.tool_emb = nn.Embedding(N_TOOL, 16)
        self.step_in = nn.Linear(16 + N_OBS, d)
        self.cell = nn.GRUCell(d, d)
        self.q_proj = nn.Linear(384, d)                       # the goal, via MiniLM
        self.tool_head = nn.Linear(2 * d, N_TOOL)
        self.arg_head = nn.Linear(2 * d + 384, 1)             # pointer: score each candidate

    def init_z(self, goal: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.q_proj(goal))

    def advance(self, z, tool_idx: int, obs: np.ndarray) -> torch.Tensor:
        x = torch.cat([self.tool_emb(torch.tensor(tool_idx)),
                       torch.tensor(obs, dtype=torch.float32)])
        return self.cell(torch.tanh(self.step_in(x)).unsqueeze(0), z.unsqueeze(0)).squeeze(0)

    def tool_logits(self, z, goal):
        return self.tool_head(torch.cat([z, torch.tanh(self.q_proj(goal))]))

    def arg_logits(self, z, goal, cand_emb: torch.Tensor):
        ctx = torch.cat([z, torch.tanh(self.q_proj(goal))]).unsqueeze(0).expand(cand_emb.shape[0], -1)
        return self.arg_head(torch.cat([ctx, cand_emb], dim=1)).squeeze(-1)


def arg_candidates(issue: str, state: dict, n: int = 24) -> list:
    """Pointer targets: identifiers named in the issue, plus paths previous tools surfaced. The
    controller can only choose from things that actually exist in the task or in an observation."""
    cands = []
    seen = set()
    for t in IDT.findall(issue)[:400]:
        if t not in seen and len(t) > 3:
            seen.add(t)
            cands.append(t)
    for h in state.get("last_hits", [])[:12]:
        p = h.split(":")[0]
        if p not in seen:
            seen.add(p)
            cands.append(p)
    return cands[:n]


def run_episode(ctl, issue: str, repo: str, instance_id: str, gold: str | None = None,
                max_steps: int = 6, sample: bool = False, blind: bool = False, lm=None):
    """One real trajectory: pick a tool, point at an argument, RUN it, fold the observation back in."""
    from embedder import encode_batch
    goal = torch.tensor(encode_batch([issue[:1000]])[0], dtype=torch.float32)
    state = {"repo": repo, "instance_id": instance_id, "steps": 0, "seen_files": set()}
    z = ctl.init_z(goal)
    logps, trace = [], []
    for _ in range(max_steps):
        tl = ctl.tool_logits(z, goal)
        if sample:
            d = torch.distributions.Categorical(logits=tl)
            ti = int(d.sample()); logps.append(d.log_prob(torch.tensor(ti)))
        else:
            ti = int(tl.argmax())
        tool = TOOLS[ti]
        if tool == "stop":
            trace.append(("stop", "", True, ""))
            break
        cands = arg_candidates(issue, state)
        if not cands:
            break
        ce = torch.tensor(encode_batch(cands), dtype=torch.float32)
        al = ctl.arg_logits(z, goal, ce)
        if sample:
            d = torch.distributions.Categorical(logits=al)
            ai = int(d.sample()); logps.append(d.log_prob(torch.tensor(ai)))
        else:
            ai = int(al.argmax())
        arg = cands[ai]

        if tool == "grep":
            ok, obs = t_grep(state, arg)
            state["last_hits"] = state.get("last_grep", [])
        elif tool == "find_def":
            ok, obs = t_find_def(state, arg)
            state["last_hits"] = [l for l in obs.splitlines() if ":" in l]
        elif tool == "read_file":
            path = arg if arg.endswith(".py") else (state.get("last_hits") or [""])[0].split(":")[0]
            ok, obs = t_read_file(state, path) if path else (False, "no path to read")
            if ok:
                state["seen_files"].add(path)
        else:  # edit -- the ONE place the LM is consulted
            if not state.get("open_file"):
                ok, obs = False, "no file open"
            else:
                anchor, new = propose_edit(issue, state, lm)
                ok, obs = t_edit(state, (anchor, new)) if anchor else (False, "no anchor")
                state["edit_ambiguous"] = "ambiguous" in obs
                state["edit_unparsable"] = "not parse" in obs
        state["last_ok"] = ok
        state["steps"] += 1
        if gold is not None:
            state["gold_open"] = (state.get("open_file") == gold)
        z = ctl.advance(z, ti, obs_features(state, blind))
        trace.append((tool, str(arg)[:60], ok, obs[:120]))
    return state, logps, trace


def propose_edit(issue: str, state: dict, lm=None):
    """The LM's ONLY job: given the open file and the issue, return replacement code for one anchor.

    The anchor itself is chosen deterministically (a unique line mentioning an identifier from the
    issue), so the LM cannot pick where to cut -- only what to write. Without an LM this returns an
    identity edit, which still exercises the gates and is the honest no-LM baseline."""
    txt = state.get("open_text", "")
    toks = set(IDT.findall(issue))
    anchor = None
    for line in txt.splitlines():
        s = line.strip()
        if len(s) > 20 and txt.count(line) == 1 and any(t in line for t in toks):
            anchor = line
            break
    if anchor is None or lm is None:
        return anchor, anchor
    prompt = (f"Issue:\n{issue[:700]}\n\nFile {state['open_file']} contains this line:\n{anchor}\n\n"
              f"Rewrite ONLY that line to fix the issue. Output the replacement line and nothing else.")
    out = str(lm.generate_chat(prompt, max_new=96)).strip().splitlines()
    new = out[0] if out else anchor
    indent = anchor[:len(anchor) - len(anchor.lstrip())]
    return anchor, (indent + new.strip())


def reward(state: dict, gold: str | None) -> float:
    """Verified, dense, and NEVER from an LM. Progress toward a real patch, each term observable."""
    r = 0.0
    if state.get("open_file"):
        r += 0.1
    if gold and state.get("open_file") == gold:
        r += 0.4
    if state.get("patch"):
        r += 0.5
    return r


def train(ctl, data, epochs: int = 6, lr: float = 3e-3, blind: bool = False, verbose: bool = True):
    opt = torch.optim.Adam(ctl.parameters(), lr=lr)
    for ep in range(epochs):
        rs = []
        for row in data:
            st, lp, _ = run_episode(ctl, row["problem"], row["repo"], row["instance_id"],
                                    gold=row["gold"], sample=True, blind=blind)
            rs.append((reward(st, row["gold"]), lp))
        base = sum(r for r, _ in rs) / max(1, len(rs))
        live = [(r, lp) for r, lp in rs if lp]
        if live:
            loss = torch.stack([-(r - base) * torch.stack(lp).sum() for r, lp in live]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if verbose and (ep + 1) % 2 == 0:
            print(f"    epoch {ep+1:2d}  mean reward {base:.3f}", flush=True)
    return ctl


def _selftest() -> bool:
    print("algo_grr_thinkctl --selftest: thinker drives REAL tools, state comes from REAL observations\n")
    ok = True

    def chk(t, c, d=""):
        nonlocal ok
        ok &= bool(c)
        print(f"  [{'PASS' if c else 'FAIL'}] {t}{('  - ' + d) if d else ''}")

    if not _repo_dir("django/django").is_dir():
        print("  no checkout; skipping")
        return False
    ctl = ThinkerController()
    issue = ("QuerySet.only() after select_related() crashes on proxy models. "
             "The error comes from RelatedPopulator when init_list is built.")
    st, lp, trace = run_episode(ctl, issue, "django/django", "django__django-11999", sample=True)
    chk("[1] the episode runs REAL tools and returns a real trace",
        len(trace) > 0 and all(len(x) == 4 for x in trace), f"{len(trace)} steps")
    chk("[2] tools actually executed (observations are non-empty)",
        any(x[3] for x in trace), " | ".join(f"{x[0]}:{x[2]}" for x in trace[:4]))

    a = obs_features({"open_file": "x", "last_hits": ["a"], "last_ok": True, "steps": 2})
    b = obs_features({"open_file": None, "last_hits": [], "last_ok": False, "steps": 0})
    chk("[3] observation features distinguish real states", not np.allclose(a, b))
    chk("[4] --abl no-obs really blinds them (the falsifier has teeth)",
        np.allclose(obs_features({"open_file": "x", "last_hits": ["a"]}, blind=True),
                    np.zeros(N_OBS)))

    z0 = ctl.init_z(torch.randn(384))
    z1 = ctl.advance(z0, 0, a)
    z2 = ctl.advance(z0, 0, b)
    chk("[5] the OBSERVATION moves the state (no contraction to a fixed point)",
        float((z1 - z2).norm()) > 1e-3, f"||dz|| = {float((z1 - z2).norm()):.4f}")

    cands = arg_candidates(issue, {"last_hits": ["django/db/models/query.py:12"]})
    chk("[6] arguments are POINTED AT, drawn from the issue and from observations",
        "RelatedPopulator" in cands and any(c.endswith(".py") for c in cands),
        f"{len(cands)} candidates")

    st2 = {"repo": "django/django", "open_file": "django/db/models/query.py"}
    t_read_file(st2, "django/db/models/query.py")
    anc, new = propose_edit(issue, st2, lm=None)
    chk("[7] anchor is chosen deterministically; no-LM path is an identity edit",
        anc is not None and anc == new, (anc or "")[:60])

    chk("[8] reward is verified progress only, never an LM judgement",
        reward({"open_file": "a"}, "b") == 0.1
        and reward({"open_file": "b"}, "b") == 0.5
        and reward({"open_file": "b", "patch": 1}, "b") == 1.0)

    chk("[9] registry has the real tools", set(swe_registry()) >= {"grep", "edit", "run_tests"})
    print(f"\n  ALGO_GRR_THINKCTL -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description="Thinker as controller over real tools.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--abl", type=str, default="")
    ap.add_argument("--lm", type=str, default="")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    if a.run:
        rows = [r for r in json.loads((Path(_ROOT) / "artifacts" / "swebench_loc_big.json")
                                      .read_text(encoding="utf-8"))
                if r["repo"] == "django/django"][:a.n]
        blind = "no-obs" in a.abl
        random.Random(0).shuffle(rows)
        tr, held = rows[: int(len(rows) * 0.7)], rows[int(len(rows) * 0.7):]
        lm = None
        if a.lm:
            from v5.runtime.dcpd_latent import WhiteBox
            lm = WhiteBox(a.lm, quant="4bit")
        print(f"algo_grr_thinkctl: {len(tr)} train / {len(held)} held django instances  "
              f"blind={blind}  lm={a.lm or 'none'}\n")
        ctl = ThinkerController()
        train(ctl, tr, blind=blind)
        tot = {"opened": 0, "gold": 0, "patched": 0}
        for row in held:
            st, _, _ = run_episode(ctl, row["problem"], row["repo"], row["instance_id"],
                                   gold=None, blind=blind, lm=lm)
            tot["opened"] += int(bool(st.get("open_file")))
            tot["gold"] += int(st.get("open_file") == row["gold"])
            tot["patched"] += int(bool(st.get("patch")))
        n = max(1, len(held))
        print(f"\n  held-out ({n} instances, gold NEVER visible at eval)")
        print(f"    opened a file        {tot['opened'] / n:.3f}")
        print(f"    opened the GOLD file {tot['gold'] / n:.3f}")
        print(f"    produced a patch that applies AND parses {tot['patched'] / n:.3f}")
        sys.exit(0)
    ap.print_help()


if __name__ == "__main__":
    main()
