"""algo_grr_thinkctl -- THE THINKER AS A DOMAIN-GENERAL CONTROLLER. It drives real tools over real
data, its state is advanced by real observations, and the LM only fills holes it cannot: one anchored
rewrite in code, or a faithful narration of a decision the controller already made. Nothing about the
controller, the observation channel, or the training loop is SWE-specific -- that was the gap the user
called out directly: "this entire system still does best on SWE tasks... make the thinker actually
generally smart and make the LM speak anything based on the thinker and graph information and tools."

DESIGNED AGAINST THE SPECIFIC FAILURES THIS PROJECT MEASURED, not against the word "TRM":
  trained through lm_loss -> the latent became a CONSTANT (across-task slot cosine 1.0000). Here NO
      loss touches the LM. Reward is verified progress; narration is never trained at all.
  output was a score over candidates -> re-ranking, null in every arm tried. Here the output is an
      ACTION that changes the world: which tool, and which argument.
  recursion was a contraction that washed the input out. Here z is advanced by a GRU from the REAL
      OBSERVATION the tool returned -- now the observation itself is a MiniLM embedding of whatever
      text the tool actually printed, so the same mechanism reads a grep hit list, a "no definition"
      message, or a retrieved paragraph without any domain-specific featurisation.
  latent injection into the LM -> dead across four attempts. Here the interface is TEXT both ways.
  the task had no room -> every no-exec ablation came back null on single-shot tasks. Both domains
      here are multi-step with a real state to accumulate.

WHAT "DOMAIN-GENERAL" MEANS CONCRETELY: a `Domain` bundles a tool registry, a pointer-candidate
function, a verified reward/metrics function, a state initializer, and a "what did we decide" reader.
ThinkerController never sees any of that -- its constructor only takes n_tool (the action count), and
every input to it is either a discrete index or a 384-d MiniLM embedding, which exists in identical
shape for a repo, a paragraph set, or anything else with real, verifiable tools. Adding a third domain
means writing a Domain, not touching this file's controller code.

ARGUMENTS ARE POINTED AT, NOT GENERATED, in both domains: code identifiers/paths from the issue and
from grep/find_def hits; entity phrases from the question and from retrieved paragraphs. The LM is
asked for exactly two things it cannot be pointed at: replacement CODE in an edit, and the WORDING of
an already-fixed decision (speak()). It never chooses which file, which answer, or whether a step
worked -- that stays entirely inside the pointer mechanism and the real tool observations.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import os
os.environ.setdefault("HF_HOME", r"E:\cache\hf")

import numpy as np
import torch
import torch.nn as nn

from v5.runtime.algo_grr_swetools import _repo_dir, t_edit, t_find_def, t_grep, t_read_file, t_run_tests
from v5.runtime.algo_grr_qatools import t_answer, t_read, t_retrieve

IDT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")                       # code identifiers / paths
ENT = re.compile(r"[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*){0,3}")  # prose entity phrases
_STOP = {"The", "This", "That", "These", "Those", "It", "If", "When", "What", "Which", "Who", "Whom",
         "A", "An", "In", "On", "At", "By", "For", "With", "And", "Or", "But", "Is", "Was", "Were", "Are"}


def text_candidates(texts: list, pattern: "re.Pattern", n: int = 24, min_len: int = 3,
                    stop: set | None = None) -> list:
    """Pointer targets drawn from real text the task or a tool already produced -- never generated.
    The only domain-specific knob is which PATTERN counts as 'a thing you could point at' (identifiers
    for code, capitalized phrases for prose); the mechanism that turns text into candidates is shared."""
    stop = stop or ()
    seen, out = set(), []
    for t in texts:
        for m in pattern.findall(t or ""):
            if m not in seen and len(m) >= min_len and m not in stop:
                seen.add(m)
                out.append(m)
    return out[:n]


# ── the controller: domain-blind by construction ─────────────────────────────────────────────────
class ThinkerController(nn.Module):
    """z_{t+1} = GRU(z_t, [action_emb, observation_embedding, ok, step_frac, repeat]) -> (tool
    logits, argument pointer).

    The observation is now a MiniLM embedding of the RAW TEXT the tool returned, not a hand-built
    numeric feature vector. That is what makes this domain-general: a 10-dim feature like
    "edit_ambiguous?" only means anything for code-edit tools, but "the embedding of what the tool
    said" is defined identically whether the tool is grep or a HotpotQA paragraph retriever.

    Small on purpose: this project has degraded its own baseline four times by putting thousands of
    parameters on a few hundred examples. Capacity is not the lever here; the observation channel is.
    """

    def __init__(self, n_tool: int, d: int = 48, obs_dim: int = 384):
        super().__init__()
        self.n_tool, self.d, self.obs_dim = n_tool, d, obs_dim
        self.tool_emb = nn.Embedding(n_tool, 16)
        self.obs_proj = nn.Linear(obs_dim, 16)
        self.step_in = nn.Linear(16 + 16 + 3, d)
        self.cell = nn.GRUCell(d, d)
        self.q_proj = nn.Linear(obs_dim, d)                    # the goal, via MiniLM
        self.tool_head = nn.Linear(2 * d, n_tool)
        self.arg_head = nn.Linear(2 * d + obs_dim, 1)          # pointer: score each candidate

    def init_z(self, goal: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.q_proj(goal))

    def advance(self, z, tool_idx: int, ok: float, step_frac: float, repeat: float,
                obs_emb: torch.Tensor) -> torch.Tensor:
        te = self.tool_emb(torch.tensor(tool_idx))
        oe = torch.tanh(self.obs_proj(obs_emb))
        extra = torch.tensor([float(ok), float(step_frac), float(repeat)], dtype=torch.float32)
        x = torch.tanh(self.step_in(torch.cat([te, oe, extra])))
        return self.cell(x.unsqueeze(0), z.unsqueeze(0)).squeeze(0)

    def tool_logits(self, z, goal):
        return self.tool_head(torch.cat([z, torch.tanh(self.q_proj(goal))]))

    def arg_logits(self, z, goal, cand_emb: torch.Tensor):
        ctx = torch.cat([z, torch.tanh(self.q_proj(goal))]).unsqueeze(0).expand(cand_emb.shape[0], -1)
        return self.arg_head(torch.cat([ctx, cand_emb], dim=1)).squeeze(-1)


class Domain:
    """Everything task-specific lives here. The controller, run_episode, and train() never branch on
    which domain they are in -- only Domain's own callables know that."""

    def __init__(self, name: str, tools: dict, tool_names: list, arg_fn, reward_fn, init_state,
                decision_fn, metrics_fn, max_steps: int = 6, no_arg_tools: frozenset = frozenset(),
                available_fn=None):
        self.name = name
        self.tools = tools                # {tool_name: fn(state, arg) -> (ok, obs_text)}
        self.tool_names = tool_names      # ordered; index IS the action id ("stop" handled specially)
        self.arg_fn = arg_fn              # fn(goal_text, state) -> list[str] pointer candidates
        self.reward_fn = reward_fn        # fn(state, gold) -> float, VERIFIED, never from an LM
        self.init_state = init_state      # fn(row) -> state dict
        self.decision_fn = decision_fn    # fn(state) -> str, "what did we decide" (for speak())
        self.metrics_fn = metrics_fn      # fn(state, gold) -> {name: 0/1 or float}, all verified
        self.max_steps = max_steps
        # tools whose fn ignores `arg` entirely (e.g. SWE's edit, whose anchor is deterministic) --
        # a reviewer caught that sampling+scoring a pointer for these trains REINFORCE on a choice
        # with zero causal effect on the outcome, pure noise in the gradient. Skip the pointer step
        # for these; "stop" is always in this set since it never took an argument to begin with.
        self.no_arg_tools = frozenset(no_arg_tools) | {"stop"}
        # fn(state) -> set of currently-callable tool names. These are the tools' OWN documented
        # preconditions ("no file open; read_file first", "no patch to test") -- the same conditions
        # they already check and return as errors. Masking them makes an invalid action unpickable
        # instead of a wasted step. That is the tool API's real contract, not privileged information:
        # nothing here depends on gold, and the agent could read the identical constraint off the
        # error string. Measured effect on the SWE chain: ~2.4% of random 8-step trajectories ever
        # reached run_tests-with-a-patch unmasked.
        self.available_fn = available_fn


# ── SWE domain: real django checkout, real tools, + the graph-retrieval tool that was missing ─────
_GRAPH_CACHE: dict = {}


def _repo_graph(repo: str):
    """The repo's small-world AtomGraph (directories as worlds), built LIVE from the same checkout the
    other tools scan -- not from the cached path-embedding artifact, which was mined from a different
    tree snapshot and would silently mismatch. Cached per repo per process; the SWE domain here only
    ever touches one repo, so this is one embedding pass, not one per instance."""
    if repo in _GRAPH_CACHE:
        return _GRAPH_CACHE[repo]
    from embedder import encode_batch
    from v5.runtime.membrane import build_repo_graph
    root = _repo_dir(repo)
    files = [p.relative_to(root).as_posix() for p in root.rglob("*.py")]
    g = None
    if files:
        embs = encode_batch([f.replace("/", " ").replace("_", " ") for f in files])
        path_emb = {f: np.asarray(e, dtype=np.float32) for f, e in zip(files, embs)}
        g = build_repo_graph(files, path_emb, worlds=True)
    _GRAPH_CACHE[repo] = g
    return g


def propose_edit(issue: str, state: dict, lm=None):
    """The LM's ONE job inside the tool loop: given the open file and the issue, return replacement
    code for one anchor. The anchor is chosen DETERMINISTICALLY (a unique line mentioning an
    identifier from the issue), so the LM cannot pick WHERE to cut, only WHAT to write -- the DCPD
    split. Without an LM this returns an identity edit: the honest no-LM baseline that still exercises
    the uniqueness/parse gates."""
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


def make_swe_domain(lm=None, max_steps: int = 6, use_locate: bool = True, verify: bool = False) -> Domain:
    def t_grep_tool(state, arg):
        ok, obs = t_grep(state, arg)
        if ok:
            state["last_hits"] = state.get("last_grep", [])
        return ok, obs

    def t_find_def_tool(state, arg):
        ok, obs = t_find_def(state, arg)
        if ok:
            state["last_hits"] = [l for l in obs.splitlines() if ":" in l]
        return ok, obs

    def t_read_file_tool(state, arg):
        path = arg if arg.endswith(".py") else (state.get("last_hits") or [""])[0].split(":")[0]
        if not path:
            return False, "no path to read"
        return t_read_file(state, path)

    def t_locate_tool(state, arg):
        """GRAPH retrieval: route the query through the repo's small-world AtomGraph instead of
        scanning text. This is the mechanism `--locate` already measured at ~0.53 held-INSTANCE
        accuracy in membrane.py -- committed last as the gap ("opened GOLD is 0.083... should call the
        localization thinker AS A TOOL instead of raw grep"). O(W): compares against world centroids,
        not every file."""
        from embedder import encode_batch
        from v5.runtime.membrane import locate_thinker
        g = _repo_graph(state["repo"])
        if g is None:
            return False, "graph unavailable for this repo"
        q = np.asarray(encode_batch([(arg or state.get("goal_text", ""))[:1000]])[0], dtype=np.float32)
        res = locate_thinker(g, q, top_w=10, T=3, per_step=8)
        ranked = res.get("ranked") or []
        if ranked:
            state["last_hits"] = [f"{f}:1" for f in ranked[:12]]
        return bool(ranked), "\n".join(ranked[:12]) if ranked else "no candidates"

    def t_edit_tool(state, arg):
        if not state.get("open_file"):
            return False, "no file open; read_file first"
        anchor, new = propose_edit(state.get("goal_text", ""), state, lm)
        if anchor is None:
            return False, "no anchor found for this issue in the open file"
        return t_edit(state, (anchor, new))

    def t_run_tests_tool(state, arg):
        """THE real verifier: run the instance's actual SWE-bench Docker container and its actual
        test suite. Everything else in this domain is navigation or a syntax-level gate; this is the
        only tool that can say a patch actually FIXED anything, not just that it applied and parsed.
        No arg -- always the instance's default test command, never a controller-authored shell string.

        KNOWN LIMITATION, found live and not yet fixed: the default command is `pytest`, which is
        wrong for django (it ships no pytest at all; its real runner is tests/runtests.py) and
        possibly other repos here too. Confirmed real: an LM-authored edit on django__django-11999
        applied and parsed, but the test command failed with "No module named pytest" -- that
        correctly reports as tests_passed=False now (a prior version of the ok-heuristic misread
        that exact failure as a pass). So tests_passed=True can be trusted; tests_passed=False on a
        non-pytest repo does NOT yet mean the fix was wrong -- it may just mean the command never ran.
        Proper per-repo test targeting needs real SWE-bench FAIL_TO_PASS metadata this project's mined
        dataset does not currently carry (only instance_id/repo/gold/problem)."""
        ok, obs = t_run_tests(state, None)
        state["tests_passed"] = ok
        return ok, obs

    names = ["grep", "find_def", "read_file", "edit", "stop"]
    tools = {"grep": t_grep_tool, "find_def": t_find_def_tool, "read_file": t_read_file_tool,
              "edit": t_edit_tool}
    no_arg = {"edit"}
    if use_locate:
        names = ["grep", "find_def", "read_file", "locate", "edit", "stop"]
        tools["locate"] = t_locate_tool
    if verify:
        # gated OFF by default: docker run --pull=never fails fast on an uncached image, but the
        # controller could still spend real step-budget on a tool that's guaranteed to fail for
        # every instance except the ones actually cached locally, which would just add noise to
        # ordinary training runs that were never trying to reach this tool.
        names = [n for n in names if n != "stop"] + ["run_tests", "stop"]
        tools["run_tests"] = t_run_tests_tool
        no_arg.add("run_tests")

    def arg_fn(goal_text, state):
        # tool-discovered paths go FIRST and are never truncated away. A reviewer found the previous
        # order (issue words first, hits appended after) meant text_candidates' own n=24 cap already
        # filled up from issue prose alone in ~96% of real issues (mean 23.9 qualifying words), so the
        # one thing grep/find_def/locate actually exist to produce was silently unscoreable.
        seen, cands = set(), []
        for h in state.get("last_hits", [])[:12]:
            p = h.split(":")[0]
            if p not in seen:
                seen.add(p)
                cands.append(p)
        for t in text_candidates([goal_text], IDT, n=24):
            if t not in seen and len(cands) < 24:
                seen.add(t)
                cands.append(t)
        return cands

    def available_fn(state):
        """The tools' OWN preconditions, which they already enforce and report as errors."""
        av = {"grep", "find_def", "stop"}
        if use_locate:
            av.add("locate")
        if state.get("last_hits"):                 # read_file has nothing to open without a hit
            av.add("read_file")
        if state.get("open_file"):                 # "no file open; read_file first"
            av.add("edit")
        if verify and state.get("patch"):          # "no patch to test"
            av.add("run_tests")
        return av

    def reward_fn(state, gold):
        # SHAPING BUG FIXED: the old terms paid +0.5 for patching ANY file but only +0.4 for finding
        # the RIGHT one, so "open something, patch it" (0.6) strictly dominated the actual objective.
        # The policy learned exactly that -- measured: patched 0.958, opened_gold 0.000. Patching the
        # wrong file is now worth almost nothing, so targeting is the only route to the big reward.
        # gold is used HERE only, at training time; run_episode never puts it in state and no tool
        # can read it, so this is reward shaping on training labels, not leakage into the policy.
        on_gold = bool(gold) and state.get("open_file") == gold
        r = 0.0
        if state.get("open_file"):
            r += 0.05
        if on_gold:
            r += 0.45
        if state.get("patch"):
            r += 0.5 if on_gold else 0.05
        if state.get("tests_passed"):
            # dominant term, deliberately: passing the REAL test suite is the only signal here that
            # means "fixed," not "applied and parsed." Only reachable when verify=True.
            r += 1.0
        # small cost per REDUNDANT step. Without it, nothing opposed the degenerate loop actually
        # observed in traces (read_file with the same failing arg 5-7 times in a row): repeats were
        # free, so a policy that secured the cheap +0.1 and then idled was never pushed off it.
        r -= 0.02 * float(state.get("repeats", 0))
        return r

    def init_state(row):
        return {"repo": row["repo"], "instance_id": row["instance_id"], "goal_text": row["goal"]}

    def decision_fn(state):
        if state.get("patch"):
            p, old, new = state["patch"]
            return f"In {p}, replace `{old.strip()[:70]}` with `{new.strip()[:70]}`"
        return state.get("open_file") or ""

    def metrics_fn(state, gold):
        return {"opened": float(bool(state.get("open_file"))),
                "opened_gold": float(state.get("open_file") == gold),
                "patched": float(bool(state.get("patch"))),
                "tests_passed": float(bool(state.get("tests_passed")))}

    return Domain("swe", tools, names, arg_fn, reward_fn, init_state, decision_fn, metrics_fn, max_steps,
                  no_arg_tools=frozenset(no_arg), available_fn=available_fn)


# ── HotpotQA domain: real multi-hop QA, proving the SAME controller is not SWE-specific ───────────
def make_qa_domain(max_steps: int = 5) -> Domain:
    names = ["retrieve", "read", "answer", "stop"]
    tools = {"retrieve": t_retrieve, "read": t_read, "answer": t_answer}

    def arg_fn(goal_text, state):
        bodies = [b for _, b in state.get("retrieved", [])]
        ents = text_candidates([goal_text] + bodies, ENT, n=23, min_len=3, stop=_STOP)
        return [goal_text[:200]] + ents                       # hop 1 can point at the whole question

    # gold, for this domain, is the (answer, gold_titles) pair -- kept OUT of state entirely (a
    # reviewer found gold_titles was being written into state at init; nothing read it mid-episode,
    # but it was a landmine for the next QA tool that "simplifies" init_state to dict(row)). reward_fn/
    # metrics_fn receive it only from OUTSIDE, from row['gold'], same discipline as the SWE domain.
    def _em(pred, gold_answer):
        # EXACT match on normalized text only -- a reviewer found the prior `a in b or b in a`
        # substring check let "John" pass against gold "Johnson", and "American" pass against gold
        # "American Idol". Real metric-leakage into the training reward, not just the report.
        pn = re.sub(r"[^a-z0-9 ]", "", (pred or "").strip().lower())
        gn = re.sub(r"[^a-z0-9 ]", "", (gold_answer or "").strip().lower())
        return bool(pn) and bool(gn) and pn == gn

    def available_fn(state):
        """t_read/t_answer are meaningless before anything has been retrieved -- and they say so
        ("not retrieved yet: X"). Same masking discipline as the SWE domain, no gold involved."""
        av = {"retrieve", "stop"}
        if state.get("retrieved"):
            av |= {"read", "answer"}
        return av

    def reward_fn(state, gold):
        answer_gold, _titles_gold = gold
        r = 0.1 * min(len(state.get("retrieved", [])), 2)
        if state.get("open_para"):
            r += 0.1
        if _em(state.get("answer"), answer_gold):
            r += 0.6
        r -= 0.02 * float(state.get("repeats", 0))
        return r

    def init_state(row):
        return {"paras": row["paras"], "goal_text": row["goal"]}

    def decision_fn(state):
        return state.get("answer") or ""

    def metrics_fn(state, gold):
        answer_gold, titles_gold = gold
        got_titles = {t for t, _ in state.get("retrieved", [])}
        return {"retrieved_any": float(bool(state.get("retrieved"))),
                "retrieved_gold_para": float(bool(titles_gold) and bool(got_titles & titles_gold)),
                "answered": float(bool(state.get("answer"))),
                "exact_match": float(_em(state.get("answer"), answer_gold))}

    return Domain("hotpotqa", tools, names, arg_fn, reward_fn, init_state, decision_fn, metrics_fn,
                  max_steps, available_fn=available_fn)


# ── LM speaks. Exactly once per episode, and NEVER decides anything ────────────────────────────────
def speak(goal_text: str, trace: list, decision: str, lm=None) -> str:
    """The LM's only remaining job, shared by BOTH domains: put the controller's ALREADY-FIXED
    decision into words, grounded in what the tools actually found. It does not choose the decision --
    `decision` is fixed by the pointer mechanism before this is ever called. Without an LM this is the
    honest baseline: the decision stated plainly, unchanged."""
    if not decision:
        return ""
    if lm is None:
        return decision
    evid = " | ".join(f"{t}({a}): {o[:80]}" for t, a, ok, o in trace if ok and o)[:700]
    prompt = (f"Task: {goal_text[:400]}\n\nWhat the tools found:\n{evid}\n\n"
              f"The determined answer is: {decision}\n\n"
              f"State this as one plain sentence, using only the evidence above. "
              f"Do not change the answer, only phrase it.")
    try:
        out = str(lm.generate_chat(prompt, max_new=64)).strip()
    except Exception:                                          # noqa: BLE001
        return decision
    return out or decision


def speak_faithful(decision: str, spoken: str) -> bool:
    """Verified, not LM-judged: did narration keep the decision's own content, or quietly invent a
    different one? At least half (rounding UP, so 3-of-3 tokens needs 2, not 1 -- a reviewer caught
    the prior floor division letting a short decision pass on one generic word) of the decision's
    content tokens (len>=4) must survive into speech. Still a loose, substring-based lower bound, not
    a tight verification -- read a 0.8-ish faithfulness number as "did not obviously invent a
    different answer," not as proof the phrasing preserved every nuance."""
    toks = re.findall(r"[A-Za-z0-9_./]{4,}", decision)
    if not toks:
        return True
    s = spoken.lower()
    hit = sum(1 for w in toks if w.lower() in s)
    return hit >= max(1, -(-len(toks) // 2))


# ── the loop. domain-blind: it only ever calls through Domain's callables ──────────────────────────
def run_episode(ctl: ThinkerController, domain: Domain, row: dict, sample: bool = False,
                blind: bool = False):
    """One real trajectory: pick a tool, point at an argument, RUN it, fold the REAL observation back
    into z. `gold` is never placed in `state` and never passed to any tool -- reward/metrics compare
    the returned state against row['gold'] from OUTSIDE, after the episode is over."""
    from embedder import encode_batch
    goal_text = row["goal"]
    goal = torch.tensor(encode_batch([goal_text[:1000]])[0], dtype=torch.float32)
    state = domain.init_state(row)
    z = ctl.init_z(goal)
    logps, trace, tried = [], [], set()
    for _ in range(domain.max_steps):
        tl = ctl.tool_logits(z, goal)
        if domain.available_fn is not None:
            avail = domain.available_fn(state)
            mask = torch.tensor([0.0 if n in avail else -1e9 for n in domain.tool_names])
            tl = tl + mask
        if sample:
            d = torch.distributions.Categorical(logits=tl)
            ti = int(d.sample()); logps.append(d.log_prob(torch.tensor(ti)))
        else:
            ti = int(tl.argmax())
        tool = domain.tool_names[ti]
        if tool == "stop":
            trace.append(("stop", "", True, ""))
            break
        if tool in domain.no_arg_tools:
            arg = None
        else:
            cands = domain.arg_fn(goal_text, state)
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

        ok, obs = domain.tools[tool](state, arg)
        state["steps"] = state.get("steps", 0) + 1
        rep = (tool, arg) in tried
        if rep:
            state["repeats"] = state.get("repeats", 0) + 1
        tried.add((tool, arg))
        if blind:
            oe, sig = torch.zeros(ctl.obs_dim), (0.0, 0.0, 0.0)
        else:
            oe = torch.tensor(encode_batch([(obs or "")[:300]])[0], dtype=torch.float32)
            sig = (float(ok), state["steps"] / domain.max_steps, float(rep))
        z = ctl.advance(z, ti, *sig, oe)
        trace.append((tool, str(arg)[:60], ok, (obs or "")[:120]))
    return state, logps, trace


def train(ctl: ThinkerController, domain: Domain, data: list, epochs: int = 6, lr: float = 3e-3,
         blind: bool = False, verbose: bool = True, batch: int = 8, ent_w: float = 0.01):
    """REINFORCE with a mean-reward baseline, updated per MINIBATCH.

    The previous version did ONE opt.step() per EPOCH -- 6 epochs meant SIX gradient updates for the
    whole run. That, not "not enough data", is why every SWE number in this file sat near zero and why
    the policy collapsed into degenerate repeat loops: it was barely optimized at all. Per-minibatch
    updates give ~epochs*len(data)/batch steps instead (6x7=42 at the old defaults, and --epochs now
    actually buys something).

    ent_w: entropy bonus. With ~6 gradient steps the old setup could not collapse fast enough for
    anyone to notice it would; with real optimization it collapses to a single action immediately
    unless exploration is held open. Standard REINFORCE practice, not a workaround.
    """
    opt = torch.optim.Adam(ctl.parameters(), lr=lr)
    hist = []
    for ep in range(epochs):
        rs, ep_r = [], []
        for i, row in enumerate(data):
            st, lp, _ = run_episode(ctl, domain, row, sample=True, blind=blind)
            r = domain.reward_fn(st, row["gold"])
            rs.append((r, lp))
            ep_r.append(r)
            if len(rs) >= batch or i == len(data) - 1:
                live = [(rr, l) for rr, l in rs if l]
                if live:
                    base = sum(rr for rr, _ in live) / len(live)
                    pol = torch.stack([-(rr - base) * torch.stack(l).sum() for rr, l in live]).mean()
                    ent = torch.stack([-torch.stack(l).sum() for _, l in live]).mean()
                    (pol + ent_w * ent).backward()
                    torch.nn.utils.clip_grad_norm_(ctl.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad()
                rs = []
        m = sum(ep_r) / max(1, len(ep_r))
        hist.append(m)
        if verbose and (ep + 1) % 2 == 0:
            print(f"    [{domain.name}] epoch {ep + 1:2d}  mean reward {m:.3f}", flush=True)
    return ctl


# ── data loaders: real rows only, shaped to Domain.init_state's expectations ───────────────────────
# held-REPO convention (pytest, sphinx) matches the rest of this project's SWE-bench work
# (membrane.py's --locate, scripts/loc_thinker_sweep.py) so numbers are comparable across files.
_HELD_REPOS = ("pytest-dev/pytest", "sphinx-doc/sphinx")


def _load_swe_rows(n_per_repo: int = 8, seed: int = 0):
    """Real, OPEN, multi-repo SWE-bench -- not a django-only slice (django is 668 of 1725 real mined
    instances; a plain random sample would still be ~39% django). Stratified per-repo sampling so the
    other 11 repos are actually represented. Returns (train, held_instance, held_repo):
      - held_instance: unseen ISSUES from repos the controller trained on elsewhere.
      - held_repo: repos NEVER seen in training at all -- the deployment case, reported even when bad.
    All real checkouts confirmed present at E:/swebench_src for all 12 repos before this was written."""
    rows = json.loads((Path(_ROOT) / "artifacts" / "swebench_loc_big.json").read_text(encoding="utf-8"))
    by_repo: dict = {}
    for r in rows:
        by_repo.setdefault(r["repo"], []).append(r)
    rng = random.Random(seed)

    def mk(r):
        return {"goal": r["problem"], "gold": r["gold"], "repo": r["repo"], "instance_id": r["instance_id"]}

    held_r = []
    for repo in _HELD_REPOS:
        rs = by_repo.get(repo, [])[:]
        rng.shuffle(rs)
        held_r += [mk(r) for r in rs[:n_per_repo]]

    pool = []
    for repo, rs in by_repo.items():
        if repo in _HELD_REPOS:
            continue
        rs = rs[:]
        rng.shuffle(rs)
        pool += [mk(r) for r in rs[:n_per_repo]]
    rng.shuffle(pool)
    split = int(len(pool) * 0.7)
    return pool[:split], pool[split:], held_r


def _load_qa_rows(n: int) -> list:
    from v5.runtime.membrane import load_hotpot
    data = load_hotpot(n)
    # gold is (answer, gold_titles) -- never split across state and an outside value; see make_qa_domain.
    rows = [{"goal": q, "gold": (ans, gold), "paras": paras, "qid": qid, "type": typ}
             for qid, q, ans, typ, paras, gold in data]
    random.Random(0).shuffle(rows)
    return rows


def _run_until_real(ctl, dom, row, tries=6):
    """A FRESH, untrained controller's first action is close to uniform random, so 'sample the very
    first step as stop' is a real coin-flip outcome, not a bug -- reseed and retry rather than let the
    selftest be flaky on whichever global RNG state happened to precede it."""
    for seed in range(tries):
        torch.manual_seed(seed)
        st, lp, trace = run_episode(ctl, dom, row, sample=True)
        if any(x[3] for x in trace):
            return st, lp, trace
    return st, lp, trace


def _selftest() -> bool:
    print("algo_grr_thinkctl --selftest: ONE controller, TWO real domains, no domain-specific weights\n")
    ok = True

    def chk(t, c, d=""):
        nonlocal ok
        ok &= bool(c)
        print(f"  [{'PASS' if c else 'FAIL'}] {t}{('  - ' + d) if d else ''}")

    # ---- generality of the mechanism, checked structurally, not just by claim ----
    chk("[1] text_candidates draws real pointer targets, filtered by pattern+stopwords",
        "RelatedPopulator" in text_candidates(["RelatedPopulator crashes on init_list"], IDT)
        and "The" not in text_candidates(["The Eiffel Tower is in Paris"], ENT, stop=_STOP))

    ctl_swe = ThinkerController(n_tool=6)
    ctl_qa = ThinkerController(n_tool=4)
    chk("[2] the SAME class instantiates for domains with DIFFERENT action counts",
        ctl_swe.n_tool == 6 and ctl_qa.n_tool == 4 and type(ctl_swe) is type(ctl_qa))

    g0 = ctl_qa.init_z(torch.randn(384))
    obs_a = torch.tensor(np.random.RandomState(0).randn(384), dtype=torch.float32)
    obs_b = torch.tensor(np.random.RandomState(1).randn(384), dtype=torch.float32)
    z1 = ctl_qa.advance(g0, 0, 1.0, 0.2, 0.0, obs_a)
    z2 = ctl_qa.advance(g0, 0, 1.0, 0.2, 0.0, obs_b)
    chk("[3] the OBSERVATION EMBEDDING moves the state (no fixed-point contraction)",
        float((z1 - z2).norm()) > 1e-3, f"||dz|| = {float((z1 - z2).norm()):.4f}")
    zb = ctl_qa.advance(g0, 0, 0.0, 0.0, 0.0, torch.zeros(384))
    chk("[4] --abl no-obs really blinds it: a zeroed signal is reachable and distinct from real obs",
        float((zb - z1).norm()) > 1e-3)

    chk("[5] speak() never touches the decision with no LM (honest baseline)",
        speak("q", [], "django/db/models/query.py", lm=None) == "django/db/models/query.py")
    chk("[6] speak_faithful is a VERIFIED string check, not an LM judgement",
        speak_faithful("open query.py", "The controller opened query.py to look for the bug.")
        and not speak_faithful("open query.py", "The controller closed the terminal."))

    # ---- SWE domain: real repo, real graph tool ----
    if _repo_dir("django/django").is_dir():
        dom = make_swe_domain()
        chk("[7] SWE domain includes the graph-retrieval tool ('locate') the last commit flagged missing",
            "locate" in dom.tool_names)
        g = _repo_graph("django/django")
        chk("[8] the repo's small-world AtomGraph builds for real, from the SAME checkout the tools scan",
            g is not None and len(g.atoms) > 100 and len(g.worlds) > 1,
            f"{len(g.atoms) if g else 0} files, {len(g.worlds) if g else 0} worlds")
        row = {"goal": "QuerySet.only() after select_related() crashes on proxy models. "
                       "RelatedPopulator builds init_list incorrectly.",
               "gold": "django/db/models/query.py", "repo": "django/django",
               "instance_id": "django__django-11999"}
        st, lp, trace = _run_until_real(ctl_swe, dom, row)
        chk("[9] SWE episode runs REAL tools end to end", len(trace) > 0 and any(x[3] for x in trace),
            " | ".join(f"{x[0]}:{x[2]}" for x in trace[:4]))
        chk("[10] SWE reward is verified progress only, never an LM judgement",
            dom.reward_fn({"open_file": "a"}, "b") == 0.05
            and dom.reward_fn({"open_file": "b"}, "b") == 0.5
            and dom.reward_fn({"open_file": "b", "patch": 1}, "b") == 1.0)
        chk("[10i] patching the WRONG file no longer outscores finding the RIGHT one "
            "(the shaping bug that produced patched 0.958 / opened_gold 0.000)",
            dom.reward_fn({"open_file": "wrong", "patch": 1}, "gold")
            < dom.reward_fn({"open_file": "gold"}, "gold"),
            f"wrong+patch={dom.reward_fn({'open_file': 'w', 'patch': 1}, 'g'):.2f} vs "
            f"gold={dom.reward_fn({'open_file': 'g'}, 'g'):.2f}")
        # a reviewer found a real bug here: issue text alone routinely fills text_candidates' n=24 cap
        # (mean 23.9 qualifying words across 100 real issues), so a hit appended AFTER truncation was
        # silently dropped ~96% of the time -- the pointer could never reach a file grep/locate found.
        long_issue = " ".join(f"distinctidentifierword{i}" for i in range(40))
        cands = dom.arg_fn(long_issue, {"last_hits": ["django/db/models/query.py:12"]})
        chk("[10b] a tool-discovered hit SURVIVES even when issue text alone would fill the cap",
            "django/db/models/query.py" in cands, f"{len(cands)} candidates, hit present={'django/db/models/query.py' in cands}")
        chk("[10c] 'edit' skips the pointer step entirely (its arg was sampled but never read)",
            "edit" in dom.no_arg_tools and dom.tools["edit"].__code__.co_argcount == 2)
        dom_v = make_swe_domain(verify=True)
        chk("[10d] verify=True adds the REAL Docker test-suite tool as a no-arg action",
            "run_tests" in dom_v.tool_names and "run_tests" in dom_v.no_arg_tools
            and dom_v.reward_fn({"tests_passed": True}, None) >= 1.0)
        chk("[10e] verify=False (the default) never offers it -- ordinary runs are unaffected",
            "run_tests" not in dom.tool_names)
        av0 = dom.available_fn({})
        av1 = dom.available_fn({"last_hits": ["a.py:1"], "open_file": "a.py"})
        chk("[10f] tool masking enforces the tools' OWN preconditions (no gold involved)",
            "edit" not in av0 and "read_file" not in av0
            and "edit" in av1 and "read_file" in av1 and "grep" in av0,
            f"empty state -> {sorted(av0)}")
        chk("[10g] run_tests is masked until a patch exists",
            "run_tests" not in dom_v.available_fn({"open_file": "a.py", "last_hits": ["a.py:1"]})
            and "run_tests" in dom_v.available_fn({"patch": ("a.py", "x", "y")}))
        chk("[10h] redundant steps are penalised (the degenerate repeat loop had no cost before)",
            dom.reward_fn({"open_file": "a", "repeats": 3}, None)
            < dom.reward_fn({"open_file": "a", "repeats": 0}, None))
    else:
        print("  [SKIP] no django checkout; SWE-domain checks skipped")

    # ---- HotpotQA domain: real questions, proving this is not SWE-specific ----
    hp = Path(_ROOT) / "artifacts" / "hotpot_multihop.json"
    if hp.exists():
        rows = _load_qa_rows(20)
        dom_qa = make_qa_domain()
        st, lp, trace = _run_until_real(ctl_qa, dom_qa, rows[0])
        chk("[11] HotpotQA episode runs the SAME controller/loop code over REAL paragraphs",
            len(trace) > 0 and any(x[3] for x in trace),
            " | ".join(f"{x[0]}:{x[2]}" for x in trace[:4]))
        chk("[12] QA reward is verified string comparison only, never an LM judgement",
            dom_qa.reward_fn({"answer": "paris", "retrieved": [(1, 2)]}, ("Paris", set())) > 0.6)
        chk("[12b] QA exact-match is EXACT, not substring (a reviewer found 'John' passing "
            "against gold 'Johnson')",
            not dom_qa.reward_fn({"answer": "John", "retrieved": []}, ("Johnson", set())) > 0.15
            and not dom_qa.reward_fn({"answer": "American", "retrieved": []},
                                     ("American Idol", set())) > 0.15)
        chk("[12c] gold never sits in state -- QA's init_state carries no gold field",
            "gold" not in str(dom_qa.init_state(rows[0]).keys()).lower())
        # not every real question names an entity ("What science fantasy series..." does not), so pick
        # a real row that does to check the POINTED, not the row that happens to be shuffled first.
        row_e = next((r for r in rows if len(dom_qa.arg_fn(r["goal"], {"retrieved": []})) > 1), rows[0])
        cands = dom_qa.arg_fn(row_e["goal"], {"retrieved": []})
        chk("[13] QA arguments are POINTED AT (question text / entities), never generated",
            row_e["goal"][:50] in cands[0] and len(cands) > 1, f"{cands[:4]}")
    else:
        print(f"  [SKIP] {hp} missing; HotpotQA-domain checks skipped")

    print(f"\n  ALGO_GRR_THINKCTL -> {'PASS' if ok else 'FAIL'}")
    return ok


def _eval_held(ctl, domain, rows, blind, lm, label):
    import collections
    tot = collections.Counter()
    n, spoken_n, faithful = 0, 0, 0
    for row in rows:
        st, _, trace = run_episode(ctl, domain, row, blind=blind)
        for k, v in domain.metrics_fn(st, row["gold"]).items():
            tot[k] += float(v)
        n += 1
        decision = domain.decision_fn(st)
        if lm is not None and decision:
            spoken = speak(row["goal"], trace, decision, lm)
            spoken_n += 1
            faithful += int(speak_faithful(decision, spoken))
    n = max(1, n)
    print(f"\n  {label} ({n} instances, gold NEVER visible during the episode)")
    for k, v in tot.items():
        print(f"    {k:<22} {v / n:.3f}")
    if spoken_n:
        print(f"    {'LM narration faithful':<22} {faithful / spoken_n:.3f}  (n={spoken_n})")


def main():
    ap = argparse.ArgumentParser(description="Thinker as a domain-general controller over real tools.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--domain", choices=["swe", "hotpot"], default="swe")
    ap.add_argument("--n", type=int, default=40, help="HotpotQA: total questions.")
    ap.add_argument("--n-per-repo", type=int, default=8,
                    help="SWE: real instances sampled per repo (stratified across all 12 open "
                         "SWE-bench repos, not just django).")
    ap.add_argument("--abl", type=str, default="")
    ap.add_argument("--lm", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--verify", action="store_true",
                    help="SWE: add the real Docker test-suite tool. Off by default -- only a handful "
                         "of SWE-bench images are cached locally, and docker run --pull=never makes "
                         "every other instance fail fast rather than pull, so this is safe to try but "
                         "will mostly report 'no image' outside cached instances.")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    if a.run:
        blind = "no-obs" in a.abl
        use_locate = "no-locate" not in a.abl
        lm = None
        if a.lm:
            from v5.runtime.dcpd_latent import WhiteBox
            lm = WhiteBox(a.lm, quant="4bit")

        held_r = []
        if a.domain == "swe":
            tr, held, held_r = _load_swe_rows(n_per_repo=a.n_per_repo, seed=a.seed)
            domain = make_swe_domain(lm=lm, use_locate=use_locate, verify=a.verify)
            repo_line = (f"repos: {sorted({r['repo'] for r in tr + held})} train/held-I, "
                        f"{list(_HELD_REPOS)} held-REPO (never trained on)")
        else:
            rows = _load_qa_rows(a.n)
            domain = make_qa_domain()
            tr, held = rows[: int(len(rows) * 0.7)], rows[int(len(rows) * 0.7):]
            repo_line = ""

        print(f"algo_grr_thinkctl: domain={a.domain}  {len(tr)} train / {len(held)} held-I / "
              f"{len(held_r)} held-REPO  blind={blind}  locate={use_locate if a.domain == 'swe' else 'n/a'}"
              f"  lm={a.lm or 'none'}  seed={a.seed}")
        if repo_line:
            print(f"  {repo_line}")
        print()
        # unseeded, REINFORCE over sparse-reward episodes is genuinely seed-sensitive -- one unlucky
        # init converged to an always-stop policy (0.000 on every metric). Seeding makes runs
        # reproducible and comparable across ablations; it does not hide instability, it controls for it.
        torch.manual_seed(a.seed)
        ctl = ThinkerController(n_tool=len(domain.tool_names))
        train(ctl, domain, tr, epochs=a.epochs, blind=blind)

        _eval_held(ctl, domain, held, blind, lm, "held-INSTANCE" if a.domain == "swe" else "held-out")
        if held_r:
            _eval_held(ctl, domain, held_r, blind, lm,
                      "held-REPO (unseen repos entirely -- the deployment case)")
        sys.exit(0)
    ap.print_help()


if __name__ == "__main__":
    main()
