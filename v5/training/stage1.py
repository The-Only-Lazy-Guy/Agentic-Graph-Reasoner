"""V5 staged training — Stage 1: train aux heads on a frozen loop representation.

Stage 1 of the adapter staging plan (see v5_PROGRESS.md):

    Stage 1  freeze LM + freeze GNN, train aux heads only      <- THIS FILE
    Stage 2  unfreeze cross-attention projections (W_q/W_k/W_v/W_o, K/V proj)
    Stage 3  unfreeze StateOverlayHead
    Stage 4  LoRA on selected LM layers
    Stage 5  optional GNN fine-tuning

NB: this is ADAPTER staged training (progressive unfreezing of the V5 module).
It is NOT the V4 graph-edit data staging (generate -> collect-but-don't-apply ->
train -> apply safe edits -> harder batch). One is neural optimization, the other
is graph-corpus curriculum.

The trainability test (v5/training/trainability_test.py) proved this stage works
on synthetic data; this module makes the trainer reusable and adds the
V4-generated-trace data path.

Two data sources, same `Stage1Example` interface:
  - synthetic_examples(): constructed teacher labels (runs now, no LM needed)
  - corpus_examples():   from the Phase 15 V4 corpus + real mpnet embeddings +
                         real frozen-Qwen prefill h_init (needs the real stack;
                         see realstack_test.py for the component wiring)

    python -m v5.training.stage1            # synthetic smoke run
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from v5.cross_attention import V5AttentionAdapter
from v5.goal_encoder import SLOT_ID
from v5.subgraph import GraphMemoryKV


# ── example interface ────────────────────────────────────────────────────────

@dataclass
class Stage1Example:
    """One training example. Any label may be None to skip that head's loss."""
    h_init: Tensor                       # [1, lm_dim] frozen LM prefill anchor (or synthetic)
    graph_kv: GraphMemoryKV              # frozen GNN output for this subgraph
    goal: Tensor                         # [1, GOAL_DIM]
    node_ids: List[str]
    task_frame: dict

    plan_anchor: Optional[Tensor] = None   # [1, N] planning node target
    evid_anchor: Optional[Tensor] = None   # [1, N] evidence node target
    slot_target: Optional[Tensor] = None   # [1, NUM_SLOTS]
    epi_target: Optional[Tensor] = None    # [1, N]
    inv_target: Optional[Tensor] = None    # [1, N] (structural nodes only)
    shortcut_target: Optional[Tensor] = None  # [1, 1]
    struct_inv_mask: Optional[Tensor] = None  # [1, N] bool, structural invalidator nodes
    tag: str = ""                          # free label (e.g. family) for reporting


# ── freeze protocol ──────────────────────────────────────────────────────────

def prepare_stage1(adapter: V5AttentionAdapter, freeze_overlay: bool = False) -> List[Tensor]:
    """Freeze everything except the prediction heads; return trainable params.

    Stage 1 trains the aux prediction heads (slot/node/epistemic/invalidator/
    shortcut + head_norm). The cross-attention projections (W_q/W_k/W_v/W_o and
    K/V proj) are frozen — they are Stage 2. The StateOverlayHead is frozen here
    when `freeze_overlay=True` (strict staging: overlay is Stage 3); the proven
    recipe also works with it trainable, so it defaults to trainable.

    The LM and GNN are external and must be frozen by the caller (they are not
    part of the adapter).
    """
    for blk in (adapter.planning_block, adapter.evidence_block):
        for grp in (blk.proj, blk.K_proj, blk.V_proj):
            for p in grp.parameters():
                p.requires_grad_(False)
    if freeze_overlay:
        for p in adapter.aux_heads.overlay.parameters():
            p.requires_grad_(False)
    return [p for p in adapter.parameters() if p.requires_grad]


# ── loss helpers ─────────────────────────────────────────────────────────────

def pool_ce(scores: Tensor, target_onehot: Tensor, pool_mask: Tensor) -> Tensor:
    """Cross-entropy of softmax(scores over pool) toward the target node(s).

    Bounded/stable, unlike sigmoid-BCE on unbounded cumulative node scores.
    target_onehot may mark >1 node (soft target). Returns 0 (grad-safe) if the
    pool or target is empty.
    """
    if pool_mask is None or not pool_mask.any():
        return scores.sum() * 0.0
    masked = scores.masked_fill(~pool_mask, -1e9)
    logp = torch.log_softmax(masked, dim=-1)
    tgt = (target_onehot * pool_mask).clamp(min=0)
    if float(tgt.sum().item()) == 0.0:
        return scores.sum() * 0.0
    tgt = tgt / tgt.sum()
    return -(tgt * logp).sum()


# ── trainer ──────────────────────────────────────────────────────────────────

@dataclass
class Stage1Config:
    epochs: int = 300
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 0.5
    freeze_overlay: bool = False
    # per-head loss weights
    w_plan: float = 1.0
    w_evid: float = 1.0
    w_slot: float = 2.0
    w_epi: float = 1.0
    w_inv: float = 2.0
    w_shortcut: float = 1.0
    log_every: int = 40


class Stage1Trainer:
    """Trains the aux heads on a frozen loop representation (AdamW + cosine)."""

    def __init__(self, adapter: V5AttentionAdapter, config: Optional[Stage1Config] = None):
        self.adapter = adapter
        self.cfg = config or Stage1Config()
        self.params = prepare_stage1(adapter, freeze_overlay=self.cfg.freeze_overlay)
        self.opt = torch.optim.AdamW(self.params, lr=self.cfg.lr,
                                     weight_decay=self.cfg.weight_decay)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=self.cfg.epochs)

    def _losses(self, ex: Stage1Example):
        kv = ex.graph_kv
        _, ps, _ = self.adapter.run_planning(
            ex.h_init, ex.goal, kv, ex.node_ids, task_frame=ex.task_frame)
        _, es, _ = self.adapter.run_evidence(
            ps.h_r, ex.goal, kv, ex.node_ids, task_frame=ex.task_frame)
        cfg = self.cfg
        loss = ex.h_init.sum() * 0.0
        if ex.plan_anchor is not None:
            loss = loss + cfg.w_plan * pool_ce(
                ps.node_scores_r, ex.plan_anchor, kv.planning_mask.unsqueeze(0))
        if ex.evid_anchor is not None:
            loss = loss + cfg.w_evid * pool_ce(
                es.node_scores_r, ex.evid_anchor, kv.evidence_mask.unsqueeze(0))
        if ex.slot_target is not None:
            req = _required_slot_idx(ex.task_frame)
            loss = loss + cfg.w_slot * F.binary_cross_entropy(
                es.slot_state_r[0, req], ex.slot_target[0, req])
        if ex.epi_target is not None:
            loss = loss + cfg.w_epi * F.binary_cross_entropy(
                es.epistemic_confidence_r, ex.epi_target)
        if ex.inv_target is not None and ex.struct_inv_mask is not None and ex.struct_inv_mask.any():
            inv_p = es.invalidator_flags_r.clamp(1e-6, 1 - 1e-6)
            m = ex.struct_inv_mask
            loss = loss + cfg.w_inv * F.binary_cross_entropy(inv_p[m], ex.inv_target[m])
        if ex.shortcut_target is not None:
            loss = loss + cfg.w_shortcut * F.binary_cross_entropy(
                es.shortcut_validity_r, ex.shortcut_target)
        return loss

    def train(self, examples: List[Stage1Example]) -> List[dict]:
        history = []
        for ep in range(self.cfg.epochs):
            self.adapter.train()
            tot = 0.0
            for ex in examples:
                loss = self._losses(ex)
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.params, self.cfg.grad_clip)
                self.opt.step()
                tot += float(loss.item())
            self.sched.step()
            mean = tot / max(1, len(examples))
            history.append({"epoch": ep + 1, "mean_loss": mean})
            if (ep + 1) % self.cfg.log_every == 0 or ep == 0:
                print(f"  epoch {ep+1:3d}  mean_loss={mean:.4f}")
        return history

    @torch.no_grad()
    def evaluate(self, examples: List[Stage1Example]) -> Dict[str, float]:
        from v5.exit_condition import fallback_needed
        self.adapter.eval()
        agg = {k: 0 for k in ("plan", "evid", "slot", "epi", "inv", "sc")}
        cnt = {k: 0 for k in agg}
        fb = {}
        fbn = {}
        for ex in examples:
            _, ps, _ = self.adapter.run_planning(
                ex.h_init, ex.goal, ex.graph_kv, ex.node_ids, task_frame=ex.task_frame)
            _, es, _ = self.adapter.run_evidence(
                ps.h_r, ex.goal, ex.graph_kv, ex.node_ids, task_frame=ex.task_frame)
            if ex.plan_anchor is not None:
                cnt["plan"] += 1
                agg["plan"] += int(ex.plan_anchor[0, ps.node_scores_r.argmax()].item() == 1.0)
            if ex.evid_anchor is not None:
                cnt["evid"] += 1
                agg["evid"] += int(ex.evid_anchor[0, es.node_scores_r.argmax()].item() == 1.0)
            if ex.slot_target is not None:
                req = _required_slot_idx(ex.task_frame)
                cnt["slot"] += 1
                agg["slot"] += int(((es.slot_state_r[0, req] - ex.slot_target[0, req]).abs() < 0.5).all().item())
            if ex.epi_target is not None:
                cnt["epi"] += 1
                agg["epi"] += int(((es.epistemic_confidence_r > 0.5).float() == ex.epi_target).all().item())
            if ex.inv_target is not None and ex.struct_inv_mask is not None and ex.struct_inv_mask.any():
                cnt["inv"] += 1
                m = ex.struct_inv_mask
                pred = (es.invalidator_flags_r[m] > 0.5).float()
                agg["inv"] += int((pred == ex.inv_target[m]).all().item())
            if ex.shortcut_target is not None:
                cnt["sc"] += 1
                agg["sc"] += int((es.shortcut_validity_r.item() > 0.5) == ex.shortcut_target.item())
            tag = ex.tag or "all"
            fb.setdefault(tag, 0); fbn.setdefault(tag, 0)
            fbn[tag] += 1; fb[tag] += int(fallback_needed(es, ex.task_frame))
        out = {f"{k}_acc": (agg[k] / cnt[k] if cnt[k] else float("nan")) for k in agg}
        for tag in fb:
            out[f"fallback_{tag}"] = fb[tag] / max(1, fbn[tag])
        return out


def _required_slot_idx(task_frame: dict) -> List[int]:
    slots = (task_frame or {}).get("required_slots") or []
    return [SLOT_ID.get(str(s), SLOT_ID["unknown"]) for s in slots]


# ── synthetic data path (runs now) ───────────────────────────────────────────

def synthetic_examples(n_per_family: int, device, lm_dim: int,
                       n_negative: int = 0) -> List[Stage1Example]:
    """Build Stage1Examples from the trainability synthetic generator.

    Two positive families (applicable/blocked) + optional negatives (no-graph /
    weak-evidence; tag=='negative', no gold anchor).
    """
    from v5.training.trainability_test import (
        make_tasks, encode_graph, TASK_FRAME, NODE_IDS, STRUCT_INV_IDX)
    from v5.goal_encoder import encode_task_frame, GoalEncoder

    kv = encode_graph(device)
    goal = encode_task_frame(TASK_FRAME, device, GoalEncoder().to(device).eval())
    tasks = make_tasks(n_per_family, device, seed=42, n_negative=n_negative)
    struct = torch.zeros(1, len(NODE_IDS), dtype=torch.bool, device=device)
    for i in STRUCT_INV_IDX:
        struct[0, i] = True

    examples = []
    for t in tasks:
        examples.append(Stage1Example(
            h_init=t.h_init, graph_kv=kv, goal=goal,
            node_ids=NODE_IDS, task_frame=TASK_FRAME,
            plan_anchor=t.plan_anchor, evid_anchor=t.evid_anchor,
            slot_target=t.slot_target, epi_target=t.epi_target,
            inv_target=t.inv_target, shortcut_target=t.shortcut_target,
            struct_inv_mask=struct, tag=t.family,
        ))
    return examples


# ── V4-corpus data path (needs the real stack) ───────────────────────────────

def corpus_examples(corpus_path, gnn=None, embedder=None, h_init_provider: Optional[Callable] = None,
                    device=None, lm_dim: int = 128):
    """Build Stage1Examples from the Phase 15 V4 corpus (delegates to the bridge).

    The Phase 15/17 bridge (`v5/training/bridge.py`) does the real conversion:
    parse labels via Phase15Dataset, build the subgraph + frozen-GNN GraphMemoryKV,
    split the anchor mask into planning vs evidence by pool, and pull h_init from
    the provider.

    gnn/embedder/h_init_provider default to test mocks so the converter runs on
    the real corpus without a LM. Real training passes a frozen RGCNEncoder, an
    mpnet embedder (transformers.AutoModel), and a frozen-Qwen h_init provider.

    NOTE (substrate gap, see v5_PROGRESS.md "What remains"): the current corpus
    has ~0% planning-pool labels — base-graph anchors are mostly fact/claim
    (evidence pool). Planning labels appear once V4 writes the reasoning substrate
    (strategy/failure_pattern/epistemic_state/...) into the graph. Run
    `python -m v5.training.bridge` for the per-head coverage report.
    """
    from v5.training.bridge import corpus_to_stage1_examples
    return corpus_to_stage1_examples(
        corpus_path, gnn=gnn, embedder=embedder,
        h_init_provider=h_init_provider, device=device, lm_dim=lm_dim)


# ── synthetic smoke run ──────────────────────────────────────────────────────

def run():
    torch.manual_seed(7)
    device = torch.device("cpu")
    lm_dim = 128
    adapter = V5AttentionAdapter(r_plan=3, r_evidence=3, lm_hidden_dim=lm_dim).to(device)
    examples = synthetic_examples(15, device, lm_dim)
    print(f"Stage 1 synthetic scaffold: {len(examples)} examples  lm_dim={lm_dim}")

    trainer = Stage1Trainer(adapter, Stage1Config(epochs=300, lr=1e-3))
    print(f"trainable param tensors: {len(trainer.params)}  (cross-attn projections frozen)")
    before = trainer.evaluate(examples)
    print("\nBEFORE:", {k: round(v, 2) for k, v in before.items()})
    print("\ntraining (Stage 1: heads only, frozen loop projections)...")
    trainer.train(examples)
    after = trainer.evaluate(examples)
    print("\nAFTER: ", {k: round(v, 2) for k, v in after.items()})

    assert after["slot_acc"] >= 0.9 and after["epi_acc"] >= 0.9 and after["inv_acc"] >= 0.9
    assert after["sc_acc"] >= 0.9 and after["plan_acc"] >= 0.9 and after["evid_acc"] >= 0.9
    assert after.get("fallback_applicable", 1.0) <= 0.1
    assert after.get("fallback_blocked", 0.0) >= 0.9
    print("\nSTAGE 1 SCAFFOLD OK — heads train to target on the frozen loop representation")
    return True


if __name__ == "__main__":
    run()
