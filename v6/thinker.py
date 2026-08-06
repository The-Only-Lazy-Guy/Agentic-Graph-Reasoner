"""The thinker: SlotDIM state + a per-step RETRIEVAL GATE over the worlds graph.

    step t:  q_t   = W_ret(slot_state)          query built FROM memory's own state
             g_t   = sigmoid(W_gate(slot_state))  "is a lookup worth it right now"
             hits  = top-m over graph node embeddings
             write hits into the slots as new write events
             (the LM then reads the slots via CrossDIM)

Memory decides what to look up based on what it currently holds. That is what keeps the vocabulary
OPEN: hits come from the graph, so a node added after training is reachable, and novelty does not
require a bigger fixed menu.

THE GATE HAS NO GRADIENT FROM CE, and this is the load-bearing caveat. Nothing in next-token loss
rewards "retrieve the right thing" -- under CE the gate will settle on always-retrieve or
never-retrieve and both look fine. The training signal has to come from the VERIFIER
(retrieve -> compose -> verify -> reinforce). Until that loop exists, run the gate in `always` mode
and treat retrieval quality as a separate measurement, because a learned-but-unsupervised gate is
exactly the kind of component that silently does nothing while appearing to work.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from v6.slotdim import SlotDIMv3


class RetrievalGate(nn.Module):
    """Decides WHETHER to retrieve and WHAT, from the current slot state."""

    def __init__(self, d_slot: int, d_node: int = 384):
        super().__init__()
        self.W_ret = nn.Linear(d_slot, d_node, bias=False)   # memory state -> query
        self.W_gate = nn.Linear(d_slot, 1)                   # memory state -> "worth a lookup?"
        self.W_gate.bias.data.fill_(1.0)                     # start ~0.73: retrieve by default

    def forward(self, slot_state: torch.Tensor, node_embs: torch.Tensor, m: int = 2,
                mode: str = "always"):
        """slot_state [d_slot]; node_embs [N, d_node] -> (hit_idx [m], gate scalar).

        mode 'always' ignores the learned gate for SELECTION (still returns its value so it can be
        logged and later trained); 'gated' lets it suppress retrieval. Default is 'always' because
        the gate is unsupervised under CE -- see the module docstring.
        """
        q = self.W_ret(slot_state)
        q = q / q.norm().clamp_min(1e-6)
        nb = node_embs / node_embs.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        sims = nb @ q
        idx = torch.topk(sims, k=min(m, sims.shape[0])).indices
        gate = torch.sigmoid(self.W_gate(slot_state)).squeeze(-1)
        if mode == "gated" and float(gate) < 0.5:
            idx = idx[:0]
        return idx, gate


class Thinker(nn.Module):
    """Slots + retrieval gate. Emits the slot table the LM reads; never emits text itself."""

    def __init__(self, d_slot: int, k: int = 8, n_heads: int = 2, n_steps: int = 2,
                 d_node: int = 384, curvature: bool = True, integral: bool = True):
        super().__init__()
        self.n_steps = n_steps
        self.slots = SlotDIMv3(d_slot, k=k, n_heads=n_heads,
                               curvature=curvature, integral=integral)
        self.gate = RetrievalGate(d_slot, d_node=d_node)
        self.node_proj = nn.Linear(d_node, d_slot, bias=False)   # graph rows -> write events

    def forward(self, seed: torch.Tensor, node_embs: torch.Tensor, mode: str = "always"):
        """seed [T0, d_slot] initial write events (e.g. the task); node_embs [N, 384] graph rows.

        Returns (slot_table [k, d_slot], trace) where trace records, per step, which nodes were
        retrieved and what the gate said -- so retrieval can be audited instead of assumed.
        """
        writes = seed if seed.dim() == 2 else seed.unsqueeze(0)
        M = self.slots(writes)
        trace = []
        for _ in range(self.n_steps):
            state = M.mean(0)
            idx, g = self.gate(state, node_embs, mode=mode)
            trace.append({"idx": idx.tolist(), "gate": float(g)})
            if idx.numel():
                hits = self.node_proj(node_embs[idx])         # (m, d_slot)
                writes = torch.cat([writes, hits], dim=0)     # retrieved nodes become write events
                M = self.slots(writes)
        return M, trace
