"""Phase 16 training loop: LoRA + AuxHeads on V5AttentionAdapter.

Training targets per sample (from Phase15Dataset):
  1. Node attention   BCE(attn_weights_plan, anchor_mask)  — plan block
  2. Node attention   BCE(attn_weights_evid, anchor_mask)  — evid block
  3. Slot fill        BCE(slot_state_r, slot_fill_target)
  4. Epistemic        BCE(epistemic_confidence_r, epistemic_target)
  5. Invalidator      BCE(invalidator_flags_r, invalidator_target)
  6. Shortcut         BCE(shortcut_validity_r, shortcut_valid)

LM language modeling loss is excluded here — LoRA cross-entropy is a separate
supervised fine-tuning pass (Phase 17) once the aux heads converge.

Usage:
    trainer = Phase16Trainer(adapter, gnn, goal_encoder, embedder, device=device)
    trainer.train(corpus_path="artifacts/phase15/phase15_corpus.jsonl", epochs=5)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

from v5.adapter import GraphAttentionInjector
from v5.cross_attention import V5AttentionAdapter
from v5.gnn_encoder import RGCNEncoder
from v5.goal_encoder import GoalEncoder, encode_task_frame
from v5.subgraph import build_active_subgraph
from v5.training.dataset import Phase15Dataset, Phase15Sample, phase15_collate_fn

log = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-2
    epochs: int = 5
    r_plan: int = 4
    r_evidence: int = 6
    # Loss weights
    # node + slot converge with fake embeddings; epistemic/invalidator need real LM hidden states
    w_node: float = 1.0
    w_slot: float = 2.0
    w_epistemic: float = 0.1   # scale down until Phase 17 (real Qwen3 h_init)
    w_invalidator: float = 0.1
    w_shortcut: float = 0.5
    log_every: int = 5   # steps


class FakeEmbedder:
    """Zero BERT embeddings — placeholder until real BERT is integrated.

    Replace with a real sentence-transformers / BERT encoder before Phase 17.
    Produces [N, 768] float32 zeros on the target device.
    """
    def __init__(self, device: torch.device):
        self.device = device
        self.dim = 768

    def embed(self, texts: List[str]) -> Tensor:
        return torch.zeros(len(texts), self.dim, device=self.device)


class Phase16Trainer:
    """Trains aux heads (+ LoRA if attached) on Phase 15 corpus supervision.

    Does NOT load or freeze Qwen3 — training targets come from corpus signals
    only (node attention, slot fill, epistemic, invalidator, shortcut).
    This lets aux heads be pre-trained cheaply before hooking into the full LM.

    When Qwen3 is available, subclass or extend to add LM loss.
    """

    def __init__(
        self,
        adapter: V5AttentionAdapter,
        gnn: RGCNEncoder,
        goal_encoder: GoalEncoder,
        embedder=None,
        device: Optional[torch.device] = None,
        config: Optional[TrainingConfig] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.adapter = adapter.to(self.device)
        self.gnn = gnn.to(self.device)
        self.goal_encoder = goal_encoder.to(self.device)
        self.embedder = embedder or FakeEmbedder(self.device)
        self.cfg = config or TrainingConfig()

        # Only train adapter params (gnn and goal_encoder are frozen at this phase)
        self.gnn.eval()
        for p in self.gnn.parameters():
            p.requires_grad_(False)
        self.goal_encoder.eval()
        for p in self.goal_encoder.parameters():
            p.requires_grad_(False)

        trainable = [p for p in self.adapter.parameters() if p.requires_grad]
        self.opt = AdamW(trainable, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

    def train(self, corpus_path: str | Path, epochs: Optional[int] = None) -> List[dict]:
        """Run Phase 16 training. Returns per-epoch loss log."""
        epochs = epochs or self.cfg.epochs
        ds = Phase15Dataset(corpus_path)
        loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=phase15_collate_fn)

        history = []
        for ep in range(epochs):
            ep_losses = []
            for step, batch in enumerate(loader):
                sample: Phase15Sample = batch[0]
                loss, breakdown = self._train_step(sample)
                if loss is None:
                    continue

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.adapter.parameters(), max_norm=1.0)
                self.opt.step()

                ep_losses.append(breakdown)
                if (step + 1) % self.cfg.log_every == 0:
                    avg = {k: sum(d[k] for d in ep_losses) / len(ep_losses) for k in ep_losses[0]}
                    log.info(f"ep={ep+1} step={step+1} " + " ".join(f"{k}={v:.4f}" for k, v in avg.items()))

            if ep_losses:
                avg = {k: sum(d[k] for d in ep_losses) / len(ep_losses) for k in ep_losses[0]}
                history.append({"epoch": ep + 1, **avg})
                print(f"Epoch {ep+1}/{epochs} — " + " ".join(f"{k}={v:.4f}" for k, v in avg.items()))

        return history

    @torch.enable_grad()
    def _train_step(self, sample: Phase15Sample):
        """Single sample forward + loss. Returns (total_loss, breakdown_dict)."""
        self.adapter.train()

        node_ids = sample.node_ids
        N = len(node_ids)
        if N == 0:
            return None, {}

        # ── embed nodes ──────────────────────────────────────────────────────
        texts = [sample.node_texts.get(nid, "") for nid in node_ids]
        text_emb_tensor = self.embedder.embed(texts)   # [N, 768]

        # Convert to dict for build_active_subgraph
        text_emb_dict = {nid: text_emb_tensor[i].tolist() for i, nid in enumerate(node_ids)}

        # ── build fake MemoryGraph stub ───────────────────────────────────────
        graph = _FakeGraph(node_ids, sample.node_types)

        # ── build active subgraph + GNN forward ──────────────────────────────
        active_sg = build_active_subgraph(
            graph, node_ids, text_emb_dict, self.device, sample.task_frame
        )
        with torch.no_grad():
            graph_kv = self.gnn.encode_to_kv(active_sg.encoder_inputs, active_sg)

        with torch.no_grad():
            goal = encode_task_frame(sample.task_frame, self.device, self.goal_encoder)

        # ── fake hidden state (d_lm=2560, random) ────────────────────────────
        # At real Phase 17 this comes from Qwen3 prefill hidden state
        h_init = torch.randn(1, 2560, device=self.device) * 0.02

        # ── run planning + evidence blocks ────────────────────────────────────
        h_plan, plan_state, plan_logs = self.adapter.run_planning(
            h_init, goal, graph_kv, node_ids,
            r_max=self.cfg.r_plan, task_frame=sample.task_frame,
        )
        h_evid, evid_state, evid_logs = self.adapter.run_evidence(
            h_plan, goal, graph_kv, node_ids,
            r_max=self.cfg.r_evidence, task_frame=sample.task_frame,
        )

        # ── supervision targets ───────────────────────────────────────────────
        anchor_t = torch.tensor(sample.anchor_mask, dtype=torch.float32, device=self.device).unsqueeze(0)   # [1, N]
        epi_t = torch.tensor(sample.epistemic_target, dtype=torch.float32, device=self.device).unsqueeze(0) # [1, N]
        inv_t = torch.tensor(sample.invalidator_target, dtype=torch.float32, device=self.device).unsqueeze(0) # [1, N]
        slot_t = torch.tensor(sample.slot_fill_target, dtype=torch.float32, device=self.device).unsqueeze(0) # [1, NUM_SLOTS]
        shortcut_t = torch.tensor([[sample.shortcut_valid]], dtype=torch.float32, device=self.device)        # [1, 1]

        # ── losses ────────────────────────────────────────────────────────────
        cfg = self.cfg

        # Node attention: node_scores_r is cumulative attn weights (sum = r per iter,
        # can exceed [0,1]). Apply sigmoid to get proper probabilities.
        plan_scores = torch.sigmoid(plan_state.node_scores_r)   # [1, N] in (0,1)
        evid_scores = torch.sigmoid(evid_state.node_scores_r)
        node_loss_plan = F.binary_cross_entropy(plan_scores, anchor_t)
        node_loss_evid = F.binary_cross_entropy(evid_scores, anchor_t)
        node_loss = (node_loss_plan + node_loss_evid) * 0.5

        slot_loss = F.binary_cross_entropy(evid_state.slot_state_r, slot_t)

        # Epistemic and invalidator targets are sparse: ~1-2 positive nodes out of N.
        # pos_weight = (N - n_pos) / max(n_pos, 1) balances the class imbalance.
        epi_n_pos = epi_t.sum().clamp(min=1.0)
        epi_pos_w = ((N - epi_n_pos) / epi_n_pos).clamp(max=10.0)
        epi_loss = F.binary_cross_entropy(
            evid_state.epistemic_confidence_r,
            epi_t,
            weight=(epi_t * (epi_pos_w - 1) + 1),   # per-element weight
        )

        inv_n_pos = inv_t.sum().clamp(min=1.0)
        inv_pos_w = ((N - inv_n_pos) / inv_n_pos).clamp(max=10.0)
        inv_loss = F.binary_cross_entropy(
            evid_state.invalidator_flags_r.clamp(1e-6, 1 - 1e-6),
            inv_t,
            weight=(inv_t * (inv_pos_w - 1) + 1),
        )

        sc_loss = F.binary_cross_entropy(evid_state.shortcut_validity_r, shortcut_t)

        total = (
            cfg.w_node * node_loss
            + cfg.w_slot * slot_loss
            + cfg.w_epistemic * epi_loss
            + cfg.w_invalidator * inv_loss
            + cfg.w_shortcut * sc_loss
        )

        breakdown = {
            "total": total.item(),
            "node": node_loss.item(),
            "slot": slot_loss.item(),
            "epistemic": epi_loss.item(),
            "invalidator": inv_loss.item(),
            "shortcut": sc_loss.item(),
        }
        return total, breakdown


# ── fake graph stub ───────────────────────────────────────────────────────────

class _FakeNode:
    """Minimal node object accepted by build_active_subgraph / build_encoder_inputs."""
    def __init__(self, nid: str, node_type: str):
        self.node_id = nid
        self.node_type = node_type
        self.text = ""
        self.confidence = 0.5
        self.metadata = {}


class _FakeGraph:
    """Minimal graph accepted by build_active_subgraph.

    Phase 15 corpus anchors have no edge information. Edges are empty.
    At Phase 17 the real MemoryGraph will be loaded from the session file.
    """
    def __init__(self, node_ids: List[str], node_types: Dict[str, str]):
        self.nodes = {
            nid: _FakeNode(nid, node_types.get(nid, "unknown"))
            for nid in node_ids
        }
        self.edges = []   # no edge data in corpus anchors
