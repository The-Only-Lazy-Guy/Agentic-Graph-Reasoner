"""V5 Stage 2: train the graph-attention projections (learn to look, then write).

Stage 2 of the adapter staging plan. Stage 1 trained the aux heads on a frozen
loop representation; Stage 2 trains the cross-attention projections so the loop
learns WHERE to attend and HOW to write graph signal into the LM residual stream,
while keeping generation stable. NOT yet answer-quality training.

Split into two sub-stages (avoids learning bad attention + bad writing at once):

  Stage 2A — learn to LOOK
    unfreeze: W_q, W_k, W_v, K_proj, V_proj (attention routing)
    frozen:   W_o, gate (residual write), heads, overlay, LM, GNN
    loss:     per-loop weighted attention CE (planning->plan_anchor,
              evidence->evid_anchor); later loops weighted higher.

  Stage 2B — learn to WRITE
    unfreeze: + W_o, gate (residual write)
    frozen:   heads, overlay, LM, GNN  (Q/K/V stay trainable)
    loss:     attention CE + residual-magnitude penalty (keep ||gate*W_o(A)||/||h||
              bounded) + head-retention (heads frozen here -> retention is free).

The residual gate starts small (~0.02) so writing grows gradually. Generation
stability vs the base LM (the KL / catastrophic-rate check) is validated
separately by `v5.perturbation_baseline` (which runs the LM); this in-flow
trainer uses the residual-magnitude penalty + head retention as stability proxies.

    python -m v5.training.stage2      # synthetic 2A -> 2B run
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from v5.cross_attention import V5AttentionAdapter
from v5.training.stage1 import Stage1Example, synthetic_examples

GATE_INIT = 0.02
_EPS = 1e-8


# ── freeze protocols ─────────────────────────────────────────────────────────

def _set(mod, flag):
    for p in mod.parameters():
        p.requires_grad_(flag)


def prepare_stage2a(adapter: V5AttentionAdapter) -> List[Tensor]:
    """2A: train attention routing (Q/K/V); freeze write (W_o/gate), heads, overlay."""
    _set(adapter, False)
    for blk in (adapter.planning_block, adapter.evidence_block):
        _set(blk.K_proj, True)
        _set(blk.V_proj, True)
        blk.proj.W_q.requires_grad_(True)
        blk.proj.W_k.requires_grad_(True)
        blk.proj.W_v.requires_grad_(True)
    return [p for p in adapter.parameters() if p.requires_grad]


def prepare_stage2b(adapter: V5AttentionAdapter) -> List[Tensor]:
    """2B: also train the residual write (W_o + gate); Q/K/V stay trainable."""
    prepare_stage2a(adapter)
    for blk in (adapter.planning_block, adapter.evidence_block):
        blk.proj.W_o.requires_grad_(True)
        blk.proj.gate.requires_grad_(True)
    return [p for p in adapter.parameters() if p.requires_grad]


# ── attention-routing loss ───────────────────────────────────────────────────

def _loop_weights(n: int) -> List[float]:
    """Later loops weighted higher (early loops may explore)."""
    if n <= 1:
        return [1.0]
    return [0.5 + 0.5 * (i / (n - 1)) for i in range(n)]   # 0.5 -> 1.0


def attention_ce(attn_history: List[Tensor], target_onehot: Tensor, pool_mask: Tensor) -> Tensor:
    """Per-loop weighted CE of the softmax attention toward the gold pool anchors.

    attn is already softmax over the pool (out-of-pool ~ 0). target may mark >1
    node (soft target). Returns 0 (grad-safe) if pool/target empty.
    """
    if not attn_history or pool_mask is None or not pool_mask.any():
        return attn_history[0].sum() * 0.0 if attn_history else torch.zeros((), requires_grad=True)
    tgt = (target_onehot * pool_mask.float()).clamp(min=0)
    if float(tgt.sum().item()) == 0.0:
        return attn_history[0].sum() * 0.0
    tgt = tgt / tgt.sum()
    w = _loop_weights(len(attn_history))
    loss = attn_history[0].sum() * 0.0
    for wr, attn in zip(w, attn_history):
        logp = torch.log(attn.clamp_min(_EPS))
        loss = loss + wr * (-(tgt * logp).sum())
    return loss / sum(w)


# ── trainer ──────────────────────────────────────────────────────────────────

@dataclass
class Stage2Config:
    sub_stage: str = "2A"          # "2A" or "2B"
    epochs: int = 200
    lr: float = 2e-4               # low: attention CE over a small pool diverges at 1e-3
    weight_decay: float = 5e-4     # bound K/V projections so the write doesn't inflate
    grad_clip: float = 0.5
    lambda_delta: float = 0.5      # residual-magnitude penalty weight (2B)
    target_write_ratio: float = 0.05   # keep ||gate*W_o(A)||/||h|| around here
    log_every: int = 50


class Stage2Trainer:
    def __init__(self, adapter: V5AttentionAdapter, config: Optional[Stage2Config] = None):
        self.adapter = adapter
        self.cfg = config or Stage2Config()
        prep = prepare_stage2b if self.cfg.sub_stage == "2B" else prepare_stage2a
        self.params = prep(adapter)
        self.opt = torch.optim.AdamW(self.params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.cfg.epochs)

    def _step(self, ex: Stage1Example):
        kv = ex.graph_kv
        _, ps, _ = self.adapter.run_planning(ex.h_init, ex.goal, kv, ex.node_ids, task_frame=ex.task_frame)
        _, es, _ = self.adapter.run_evidence(ps.h_r, ex.goal, kv, ex.node_ids, task_frame=ex.task_frame)

        loss = ex.h_init.sum() * 0.0
        if ex.plan_anchor is not None:
            loss = loss + attention_ce(ps.attn_history, ex.plan_anchor, kv.planning_mask.unsqueeze(0))
        if ex.evid_anchor is not None:
            loss = loss + attention_ce(es.attn_history, ex.evid_anchor, kv.evidence_mask.unsqueeze(0))

        # residual-magnitude penalty (2B): keep the write near target, not runaway
        wr = (ps.write_ratios or []) + (es.write_ratios or [])
        mean_wr = sum(wr) / max(1, len(wr))
        if self.cfg.sub_stage == "2B":
            ratio = torch.tensor(0.0, requires_grad=True)
            # differentiable proxy: penalize gate^2 above target (gate is the learnable knob)
            for blk in (self.adapter.planning_block, self.adapter.evidence_block):
                over = torch.relu(blk.proj.gate.abs() - self.cfg.target_write_ratio * 10)
                ratio = ratio + over ** 2
            loss = loss + self.cfg.lambda_delta * ratio
        return loss, mean_wr

    def train(self, examples: List[Stage1Example]):
        for ep in range(self.cfg.epochs):
            self.adapter.train()
            tot = 0.0; mwr = 0.0
            for ex in examples:
                loss, wr = self._step(ex)
                self.opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.params, self.cfg.grad_clip)
                self.opt.step()
                tot += float(loss.item()); mwr += wr
            self.sched.step()
            if (ep + 1) % self.cfg.log_every == 0 or ep == 0:
                print(f"  [{self.cfg.sub_stage}] epoch {ep+1:3d}  attn_loss={tot/len(examples):.4f}  "
                      f"mean_write_ratio={mwr/len(examples):.4f}")

    @torch.no_grad()
    def evaluate(self, examples: List[Stage1Example]) -> dict:
        """Attention precision: does the final-loop attention argmax hit a gold
        pool anchor? Plus mean residual write ratio."""
        self.adapter.eval()
        plan_hit = plan_n = evid_hit = evid_n = 0
        wrs = []
        for ex in examples:
            _, ps, _ = self.adapter.run_planning(ex.h_init, ex.goal, ex.graph_kv, ex.node_ids, task_frame=ex.task_frame)
            _, es, _ = self.adapter.run_evidence(ps.h_r, ex.goal, ex.graph_kv, ex.node_ids, task_frame=ex.task_frame)
            if ex.plan_anchor is not None and ps.attn_history:
                plan_n += 1
                idx = ps.attn_history[-1].argmax().item()
                plan_hit += int(ex.plan_anchor[0, idx].item() == 1.0)
            if ex.evid_anchor is not None and es.attn_history:
                evid_n += 1
                idx = es.attn_history[-1].argmax().item()
                evid_hit += int(ex.evid_anchor[0, idx].item() == 1.0)
            wrs += (ps.write_ratios or []) + (es.write_ratios or [])
        return {
            "plan_attn_precision": plan_hit / max(1, plan_n),
            "evid_attn_precision": evid_hit / max(1, evid_n),
            "mean_write_ratio": sum(wrs) / max(1, len(wrs)),
            "plan_gate": float(self.adapter.planning_block.proj.gate.item()),
            "evid_gate": float(self.adapter.evidence_block.proj.gate.item()),
        }


def run():
    torch.manual_seed(7)
    device = torch.device("cpu")
    lm_dim = 128
    adapter = V5AttentionAdapter(r_plan=3, r_evidence=3, lm_hidden_dim=lm_dim, gate_init=GATE_INIT).to(device)
    examples = synthetic_examples(15, device, lm_dim)
    print(f"Stage 2 synthetic: {len(examples)} examples  gate_init={GATE_INIT}")

    print("\n--- Stage 2A: learn to LOOK (train Q/K/V; W_o/gate frozen) ---")
    t2a = Stage2Trainer(adapter, Stage2Config(sub_stage="2A", epochs=200, lr=2e-4))
    print(f"trainable tensors: {len(t2a.params)}")
    before = t2a.evaluate(examples)
    print("BEFORE:", {k: round(v, 3) for k, v in before.items()})
    t2a.train(examples)
    after2a = t2a.evaluate(examples)
    print("AFTER 2A:", {k: round(v, 3) for k, v in after2a.items()})

    print("\n--- Stage 2B: learn to WRITE (train W_o + gate; Q/K/V stay on) ---")
    t2b = Stage2Trainer(adapter, Stage2Config(sub_stage="2B", epochs=150, lr=2e-4, lambda_delta=0.5))
    print(f"trainable tensors: {len(t2b.params)}")
    t2b.train(examples)
    after2b = t2b.evaluate(examples)
    print("AFTER 2B:", {k: round(v, 3) for k, v in after2b.items()})

    print("\n=== Stage 2 success criteria ===")
    ok_plan = after2a["plan_attn_precision"] >= 0.9
    ok_evid = after2a["evid_attn_precision"] >= 0.9
    ok_write = after2b["mean_write_ratio"] <= 0.35
    print(f"  planning attention precision >=0.9 : {after2a['plan_attn_precision']:.2f}  {'OK' if ok_plan else 'FAIL'}")
    print(f"  evidence attention precision >=0.9 : {after2a['evid_attn_precision']:.2f}  {'OK' if ok_evid else 'FAIL'}")
    print(f"  residual write ratio bounded       : {after2b['mean_write_ratio']:.3f}  {'OK' if ok_write else 'FAIL'}")
    print(f"  gates (plan/evid)                  : {after2b['plan_gate']:.3f} / {after2b['evid_gate']:.3f}")
    assert ok_plan and ok_evid and ok_write
    print("\nSTAGE 2 (2A routing + 2B gated write) OK on synthetic.")
    print("Generation stability vs base LM is checked by v5.perturbation_baseline.")
    return after2b


if __name__ == "__main__":
    run()
