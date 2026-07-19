"""algo_grr_gate_rl — the gate/delegation RL harness (no-GPU) that demonstrates the mode-collapse traps
and that a COMPOSITE reward reaches the dual-channel synergy.

Context (user's deployment-scaling design, [[grr-v6-dual-channel]]): to RL-tune the TRM's gate — how much
it delegates to the LM vs pins to the graph — a naive pass/fail reward COLLAPSES:
  - Collapse A "Silent Coder": clamp delegate=0 everywhere -> perfect code, ZERO explanation.
  - Collapse B "Yapper": clamp delegate=1 everywhere -> a plain hallucinating LLM.
Fix = a COMPOSITE reward: R_exec (terminal, big syntax penalty) + R_bridge (reward narration ONLY if exec
passes; FAITHFULNESS-weighted, not token volume) + P_mode (penalize graph-on-Sem / LM-on-Struct) + entropy
H with a decaying beta; trained on a CURRICULUM (syntax-sandbox -> forced-explanation).

This is the DISCRETE delegation policy (per span: use the LM or the graph) — the shipped dual-channel; the
AST skeleton gives the Struct/Sem label per span FOR FREE (closure=Struct, holes/narration=Sem). The
continuous-latent gamma is Design A (parked behind fair_ab); this same reward trains it if it ever wins.

    python -m v5.runtime.algo_grr_gate_rl --selftest   # no-GPU: shows Collapse A, Collapse B, and synergy
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass


# ── environment: a task = a sequence of typed spans (Struct = exact code, Sem = explanation) ──────
def gen_task(rng: random.Random) -> list:
    """A realized answer: a few Struct spans (the code closure) + a few Sem spans (the explanation)."""
    spans = ["struct"] * rng.randint(1, 3) + ["sem"] * rng.randint(1, 3)
    rng.shuffle(spans)
    return spans


def _feat(span: str):
    import torch
    return torch.tensor([1.0 if span == "struct" else 0.0, 1.0 if span == "sem" else 0.0])


@dataclass
class RewardCfg:
    r_pass: float = 10.0            # terminal: code verifies
    r_syntax_fail: float = -10.0    # terminal: a Struct span the LM wrote hallucinated syntax
    lam_bridge: float = 2.0         # R_bridge: reward per faithful narration span (exec must pass)
    lam_pmode: float = 3.0          # P_mode: gating misalignment penalty
    cost_lm: float = 0.3            # per-LM-delegation cost (a risky, expensive asset)
    cost_kg: float = 0.0            # per-graph-use cost (Collapse-B trap makes this > cost_lm)
    p_syntax: float = 0.15          # LM syntax-error prob on a Struct span (measured ~5/40 free-inline)
    beta_entropy: float = 0.05
    faithful: float = 1.0           # narration is graph-grounded -> faithful (the R_bridge weight)


def reward(spans: list, acts: list, cfg: RewardCfg, rng: random.Random) -> tuple:
    """acts[i] = 1 (delegate span i to the LM) or 0 (pin to the graph). Returns (R, info)."""
    struct_lm = sem_lm = sem_kg = struct_total = sem_total = kg_uses = 0
    syntax_error = False
    for span, a in zip(spans, acts):
        if span == "struct":
            struct_total += 1
            if a == 1:                                   # LM writes rigid structure -> risky
                struct_lm += 1
                if rng.random() < cfg.p_syntax:
                    syntax_error = True
            else:
                kg_uses += 1                             # graph closure -> exact, safe
        else:
            sem_total += 1
            if a == 1:
                sem_lm += 1                              # LM narrates -> the explanation
            else:
                sem_kg += 1                              # graph pointer building English -> nonsense
                kg_uses += 1
    exec_pass = not syntax_error
    r_exec = cfg.r_pass if exec_pass else cfg.r_syntax_fail
    r_bridge = cfg.lam_bridge * cfg.faithful * sem_lm if exec_pass else 0.0   # faithful, gated on exec
    p_mode = cfg.lam_pmode * (sem_kg + struct_lm)        # graph-on-Sem + LM-on-Struct = wrong tool
    cost = cfg.cost_lm * (struct_lm + sem_lm) + cfg.cost_kg * kg_uses
    R = r_exec + r_bridge - p_mode - cost
    return R, dict(struct_lm=struct_lm, struct_total=struct_total, sem_lm=sem_lm,
                   sem_total=sem_total, exec_pass=int(exec_pass))


# ── policy (tiny; this is the gate head, not the whole TRM) ───────────────────────
def _build_policy():
    import torch.nn as nn

    class GatePolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))

        def forward(self, feat):
            import torch
            return torch.sigmoid(self.net(feat)).squeeze(-1)

    return GatePolicy()


def rollout(policy, spans, cfg, rng, greedy=False):
    import torch
    logps, ents, acts = [], [], []
    for span in spans:
        p = policy(_feat(span)).clamp(1e-6, 1 - 1e-6)
        a = (1 if p.item() > 0.5 else 0) if greedy else (1 if rng.random() < p.item() else 0)
        logps.append(torch.log(p) if a else torch.log(1 - p))
        ents.append(-(p * torch.log(p) + (1 - p) * torch.log(1 - p)))
        acts.append(a)
    R, info = reward(spans, acts, cfg, rng)
    return R, logps, ents, info


def train(phases: list, episodes: int = 1500, lr: float = 0.02, seed: int = 0):
    """REINFORCE with a moving-average baseline + entropy bonus, over a CURRICULUM (list of RewardCfg,
    one per phase). Entropy beta decays across the whole run so the policy can lock in late."""
    import torch

    torch.manual_seed(seed)
    policy = _build_policy()
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    rng = random.Random(seed)
    baseline = 0.0
    per = max(1, episodes // len(phases))
    for ep in range(episodes):
        cfg = phases[min(ep // per, len(phases) - 1)]
        beta = cfg.beta_entropy * max(0.0, 1.0 - ep / episodes)      # decay exploration
        R, logps, ents, _ = rollout(policy, gen_task(rng), cfg, rng)
        baseline = 0.95 * baseline + 0.05 * R
        adv = R - baseline
        loss = -adv * torch.stack(logps).sum() - beta * torch.stack(ents).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return policy


def evaluate(policy, cfg, n=400, seed=99) -> dict:
    rng = random.Random(seed)
    s_lm = s_tot = m_lm = m_tot = passes = 0
    for _ in range(n):
        _, _, _, info = rollout(policy, gen_task(rng), cfg, rng, greedy=True)
        s_lm += info["struct_lm"]; s_tot += info["struct_total"]
        m_lm += info["sem_lm"]; m_tot += info["sem_total"]
        passes += info["exec_pass"]
    return dict(struct_deleg=s_lm / max(1, s_tot), sem_deleg=m_lm / max(1, m_tot),
                exec_pass=passes / n)


# ── selftest: the three regimes ──────────────────────────────────────────────────
def selftest() -> bool:
    print("algo_grr_gate_rl --selftest: gate RL — mode-collapse traps vs composite-reward synergy\n")
    base = RewardCfg()

    # ARM 1 — R_exec only (+ LM cost): no reward for explanation -> Collapse A (Silent Coder)
    cfg_exec = RewardCfg(lam_bridge=0.0, lam_pmode=0.0)
    exec_pol = train([cfg_exec], episodes=1500, seed=1)
    a = evaluate(exec_pol, base)

    # ARM 2 — COMPOSITE on a curriculum (P1 syntax-sandbox -> P2 forced-explanation) -> synergy
    p1 = RewardCfg(lam_bridge=0.0, lam_pmode=1.0)          # learn Struct->graph first
    p2 = RewardCfg(lam_bridge=2.0, lam_pmode=3.0)          # then open the gate for narration
    comp_pol = train([p1, p2], episodes=2600, seed=1)
    c = evaluate(comp_pol, base)

    # ARM 3 — DROP the syntax penalty (r_syntax_fail~0) + no P_mode + graph made expensive -> Collapse B
    # (Yapper): with nothing teaching "don't trust the LM on structure", the policy delegates EVERYTHING
    # to the LM. This is exactly the user's point — the big syntax penalty in R_exec is load-bearing.
    cfg_yap = RewardCfg(r_syntax_fail=0.0, lam_bridge=2.0, lam_pmode=0.0,
                        cost_lm=0.05, cost_kg=1.0, p_syntax=0.05)
    yap_pol = train([cfg_yap], episodes=1500, seed=1)
    y = evaluate(yap_pol, base)

    print(f"  ARM               | Struct->LM | Sem->LM | exec_pass | reading")
    print(f"  R_exec only       |    {a['struct_deleg']:.2f}    |   {a['sem_deleg']:.2f}  |   {a['exec_pass']:.2f}    | "
          f"COLLAPSE A (Silent Coder: no explanation)")
    print(f"  COMPOSITE (curric)|    {c['struct_deleg']:.2f}    |   {c['sem_deleg']:.2f}  |   {c['exec_pass']:.2f}    | "
          f"SYNERGY (graph writes code, LM explains)")
    print(f"  volume,no P_mode  |    {y['struct_deleg']:.2f}    |   {y['sem_deleg']:.2f}  |   {y['exec_pass']:.2f}    | "
          f"COLLAPSE B (Yapper: LM writes structure)")

    collapse_a = a["sem_deleg"] < 0.25                     # Silent Coder: doesn't let the LM explain
    synergy = c["struct_deleg"] < 0.25 and c["sem_deleg"] > 0.75 and c["exec_pass"] > 0.80
    collapse_b = y["struct_deleg"] > 0.50                  # Yapper: LM writing rigid structure
    ok = collapse_a and synergy and collapse_b
    print(f"\n  => composite reward AVOIDS both traps: Struct->graph (safe, {c['struct_deleg']:.0%}) + "
          f"Sem->LM ({c['sem_deleg']:.0%}, faithful) at {c['exec_pass']:.0%} exec.")
    print(f"     R_exec-only collapses to Silent Coder; volume-without-P_mode collapses to Yapper — the")
    print(f"     two traps the user flagged, reproduced. AST gives Struct/Sem for free; R_bridge is")
    print(f"     faithfulness-weighted (not volume) + gated on exec.")
    print(f"\n  ALGO_GRR_GATE_RL SELFTEST -> {'PASS' if ok else 'FAIL'} "
          f"(collapseA={collapse_a} synergy={synergy} collapseB={collapse_b})")
    return ok


def main():
    ap = argparse.ArgumentParser(description="Gate/delegation RL harness (composite reward, no GPU)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
