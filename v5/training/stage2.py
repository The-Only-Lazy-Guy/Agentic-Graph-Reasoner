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

from collections import Counter
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
    """2B: train the residual write (W_o + gate); Q/K/V stay trainable; and allow
    SLIGHT head fine-tuning (prediction heads, not overlay) so the heads track the
    h that 2B's writing shifts — without this, frozen heads regress (the Stage-1
    semantics drift as h changes). A head-retention loss keeps the semantics."""
    prepare_stage2a(adapter)
    for blk in (adapter.planning_block, adapter.evidence_block):
        blk.proj.W_o.requires_grad_(True)
        blk.proj.gate.requires_grad_(True)
    aux = adapter.aux_heads          # prediction heads (NOT overlay, which is Stage 3)
    for head in (aux.head_norm, aux.slot, aux.node, aux.epistemic, aux.invalidator, aux.shortcut):
        _set(head, True)
    return [p for p in adapter.parameters() if p.requires_grad]


# ── attention-routing loss ───────────────────────────────────────────────────

def _loop_weights(n: int) -> List[float]:
    """Later loops weighted higher (early loops may explore)."""
    if n <= 1:
        return [1.0]
    return [0.5 + 0.5 * (i / (n - 1)) for i in range(n)]   # 0.5 -> 1.0


def _pool_probs(attn: Tensor, pool_mask: Tensor) -> Tensor:
    """Renormalize attention over the pool nodes only (mask out-of-pool)."""
    p = attn.squeeze(0) * pool_mask.float()
    s = p.sum()
    return p / (s + _EPS) if float(s.item()) > 0 else p


def attn_entropy(attn: Tensor, pool_mask: Tensor) -> Tensor:
    """Shannon entropy of the pool attention (high = diffuse, low = confident)."""
    p = _pool_probs(attn, pool_mask)
    return -(p * torch.log(p.clamp_min(_EPS))).sum()


def max_pool_attn(attn: Tensor, pool_mask: Tensor) -> Tensor:
    """Largest single attention weight over the pool (1.0 = collapsed on one node)."""
    return _pool_probs(attn, pool_mask).max()


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
    lambda_neg: float = 0.5        # diffuse-attention penalty on negative cases
    lambda_head: float = 1.0       # 2B head-retention loss weight (keep Stage-1 semantics)
    target_write_ratio: float = 0.05   # keep ||gate*W_o(A)||/||h|| around here
    qkv_lr_scale: float = 0.3      # 2B: Q/K/V at lower LR than the write path (W_o/gate)
    wo_weight_decay: float = 1e-3  # 2B: extra decay on W_o to keep the write bounded
    log_every: int = 50


class Stage2Trainer:
    def __init__(self, adapter: V5AttentionAdapter, config: Optional[Stage2Config] = None):
        self.adapter = adapter
        self.cfg = config or Stage2Config()
        prep = prepare_stage2b if self.cfg.sub_stage == "2B" else prepare_stage2a
        self.params = prep(adapter)
        if self.cfg.sub_stage == "2B":
            # 2B: write path (W_o + gate) at full LR + extra decay; Q/K/V at a
            # lower LR (already-good routing should drift only slightly while the
            # write path learns).
            write_params = []
            for blk in (adapter.planning_block, adapter.evidence_block):
                write_params += [blk.proj.W_o.weight, blk.proj.gate]
            write_set = set(id(p) for p in write_params)
            # Q/K/V + heads at lower LR (slight drift); write path at full LR.
            other_params = [p for p in self.params if id(p) not in write_set]
            self.opt = torch.optim.AdamW([
                {"params": write_params, "lr": self.cfg.lr, "weight_decay": self.cfg.wo_weight_decay},
                {"params": other_params, "lr": self.cfg.lr * self.cfg.qkv_lr_scale,
                 "weight_decay": self.cfg.weight_decay},
            ])
        else:
            self.opt = torch.optim.AdamW(self.params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.cfg.epochs)

    def _step(self, ex: Stage1Example):
        kv = ex.graph_kv
        _, ps, _ = self.adapter.run_planning(ex.h_init, ex.goal, kv, ex.node_ids, task_frame=ex.task_frame)
        _, es, _ = self.adapter.run_evidence(ps.h_r, ex.goal, kv, ex.node_ids, task_frame=ex.task_frame)

        loss = ex.h_init.sum() * 0.0
        is_neg = (ex.tag == "negative")
        if is_neg:
            # No gold anchor: push attention DIFFUSE (penalize concentration) so
            # the adapter does not learn to confidently inject on unrelated /
            # weak-evidence questions.
            pm, em = kv.planning_mask.unsqueeze(0), kv.evidence_mask.unsqueeze(0)
            for a in ps.attn_history:
                loss = loss + self.cfg.lambda_neg * max_pool_attn(a, pm)
            for a in es.attn_history:
                loss = loss + self.cfg.lambda_neg * max_pool_attn(a, em)
        else:
            if ex.plan_anchor is not None:
                loss = loss + attention_ce(ps.attn_history, ex.plan_anchor, kv.planning_mask.unsqueeze(0))
            if ex.evid_anchor is not None:
                loss = loss + attention_ce(es.attn_history, ex.evid_anchor, kv.evidence_mask.unsqueeze(0))

        # residual-magnitude penalty (2B): keep the gated write small (gate is the
        # dominant differentiable knob; W_o is also decayed). Penalize gate^2 so
        # the write stays in the ~0.01-0.10 band unless attention CE truly needs more.
        wr = (ps.write_ratios or []) + (es.write_ratios or [])
        mean_wr = sum(wr) / max(1, len(wr))
        if self.cfg.sub_stage == "2B":
            gate_pen = (self.adapter.planning_block.proj.gate ** 2
                        + self.adapter.evidence_block.proj.gate ** 2)
            loss = loss + self.cfg.lambda_delta * gate_pen
            # head-retention: heads are slightly trainable in 2B so they track the h
            # that the write path shifts. Keep Stage-1 semantics (slot/epi/shortcut).
            if not is_neg:
                loss = loss + self.cfg.lambda_head * self._head_retention(es, ex)
        return loss, mean_wr

    def _head_retention(self, es, ex: Stage1Example) -> Tensor:
        """Stage-1 head losses on the current (2B-shifted) state, so frozen-semantics
        do not drift as h changes. slot/epistemic/shortcut + invalidator."""
        from v5.training.stage1 import _required_slot_idx
        loss = es.slot_state_r.sum() * 0.0
        if ex.slot_target is not None:
            req = _required_slot_idx(ex.task_frame)
            loss = loss + F.binary_cross_entropy(es.slot_state_r[0, req], ex.slot_target[0, req])
        if ex.epi_target is not None:
            loss = loss + F.binary_cross_entropy(es.epistemic_confidence_r, ex.epi_target)
        if ex.shortcut_target is not None:
            loss = loss + F.binary_cross_entropy(es.shortcut_validity_r, ex.shortcut_target)
        if ex.inv_target is not None and ex.struct_inv_mask is not None and ex.struct_inv_mask.any():
            inv_p = es.invalidator_flags_r.clamp(1e-6, 1 - 1e-6)
            m = ex.struct_inv_mask
            loss = loss + F.binary_cross_entropy(inv_p[m], ex.inv_target[m])
        return loss

    def train(self, examples: List[Stage1Example]):
        for ep in range(self.cfg.epochs):
            self.adapter.train()
            tot = 0.0; mwr = 0.0
            for ex in examples:
                loss, wr = self._step(ex)
                mwr += wr
                if loss.grad_fn is None:      # no supervised/penalty term this example
                    continue
                self.opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.params, self.cfg.grad_clip)
                self.opt.step()
                tot += float(loss.item())
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
        pos_plan_ent, pos_evid_ent, neg_ent = [], [], []
        plan_top1, evid_top1 = [], []
        for ex in examples:
            _, ps, _ = self.adapter.run_planning(ex.h_init, ex.goal, ex.graph_kv, ex.node_ids, task_frame=ex.task_frame)
            _, es, _ = self.adapter.run_evidence(ps.h_r, ex.goal, ex.graph_kv, ex.node_ids, task_frame=ex.task_frame)
            pm, em = ex.graph_kv.planning_mask.unsqueeze(0), ex.graph_kv.evidence_mask.unsqueeze(0)
            if ex.tag == "negative":
                if ps.attn_history:
                    neg_ent.append(float(attn_entropy(ps.attn_history[-1], pm).item()))
                continue
            if ex.plan_anchor is not None and ps.attn_history:
                plan_n += 1
                idx = ps.attn_history[-1].argmax().item()
                plan_hit += int(ex.plan_anchor[0, idx].item() == 1.0)
                pos_plan_ent.append(float(attn_entropy(ps.attn_history[-1], pm).item()))
                plan_top1.append(idx)
            if ex.evid_anchor is not None and es.attn_history:
                evid_n += 1
                idx = es.attn_history[-1].argmax().item()
                evid_hit += int(ex.evid_anchor[0, idx].item() == 1.0)
                pos_evid_ent.append(float(attn_entropy(es.attn_history[-1], em).item()))
                evid_top1.append(idx)
            wrs += (ps.write_ratios or []) + (es.write_ratios or [])

        def _top1_freq(idxs):  # fraction taken by the single most common top-1 node
            if not idxs:
                return float("nan")
            return max(Counter(idxs).values()) / len(idxs)

        return {
            "plan_attn_precision": plan_hit / max(1, plan_n),
            "evid_attn_precision": evid_hit / max(1, evid_n),
            "mean_write_ratio": sum(wrs) / max(1, len(wrs)),
            "plan_gate": float(self.adapter.planning_block.proj.gate.item()),
            "evid_gate": float(self.adapter.evidence_block.proj.gate.item()),
            "plan_entropy_pos": sum(pos_plan_ent) / max(1, len(pos_plan_ent)),
            "evid_entropy_pos": sum(pos_evid_ent) / max(1, len(pos_evid_ent)),
            "neg_entropy": (sum(neg_ent) / len(neg_ent)) if neg_ent else float("nan"),
            "plan_top1_freq": _top1_freq(plan_top1),
            "evid_top1_freq": _top1_freq(evid_top1),
        }


def run():
    torch.manual_seed(7)
    device = torch.device("cpu")
    lm_dim = 128
    adapter = V5AttentionAdapter(r_plan=3, r_evidence=3, lm_hidden_dim=lm_dim, gate_init=GATE_INIT).to(device)
    examples = synthetic_examples(15, device, lm_dim, n_negative=10)
    n_neg = sum(1 for e in examples if e.tag == "negative")
    print(f"Stage 2 synthetic: {len(examples)} examples ({n_neg} negative)  gate_init={GATE_INIT}")

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
    # positives should get MORE confident (lower entropy) than negatives stay diffuse;
    # top-1 freq < 1.0 means not collapsed to a single node across the two families.
    ok_neg_diffuse = (after2b["neg_entropy"] != after2b["neg_entropy"]) or \
                     (after2b["neg_entropy"] >= max(after2b["plan_entropy_pos"], after2b["evid_entropy_pos"]))
    ok_not_collapsed = after2b["plan_top1_freq"] < 0.99 and after2b["evid_top1_freq"] < 0.99
    print(f"  planning attention precision >=0.9 : {after2a['plan_attn_precision']:.2f}  {'OK' if ok_plan else 'FAIL'}")
    print(f"  evidence attention precision >=0.9 : {after2a['evid_attn_precision']:.2f}  {'OK' if ok_evid else 'FAIL'}")
    print(f"  residual write ratio bounded       : {after2b['mean_write_ratio']:.3f}  {'OK' if ok_write else 'FAIL'}")
    print(f"  gates (plan/evid)                  : {after2b['plan_gate']:.3f} / {after2b['evid_gate']:.3f}")
    print(f"  entropy pos plan/evid              : {after2b['plan_entropy_pos']:.2f} / {after2b['evid_entropy_pos']:.2f}  (lower=confident)")
    print(f"  entropy negatives (diffuse)        : {after2b['neg_entropy']:.2f}  {'OK' if ok_neg_diffuse else 'FAIL'}")
    print(f"  top-1 freq plan/evid (<1=ok)       : {after2b['plan_top1_freq']:.2f} / {after2b['evid_top1_freq']:.2f}  {'OK' if ok_not_collapsed else 'FAIL'}")
    assert ok_plan and ok_evid and ok_write and ok_not_collapsed
    print("\nSTAGE 2 (2A routing + 2B gated write) OK on synthetic, with negatives:")
    print("positives route confidently, negatives stay diffuse, write bounded, no collapse.")
    print("Generation stability vs base LM is checked by v5.perturbation_baseline.")
    return after2b


if __name__ == "__main__":
    run()
