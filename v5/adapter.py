"""Hook-based injection of V5AttentionAdapter into Qwen3-4B.

Registers forward hooks on transformer block layers 8 and 20.
Hook captures the hidden state, runs the recurrent attention block,
returns updated hidden state. LM weights stay frozen.

Usage:
    adapter = V5AttentionAdapter()
    gnn = RGCNEncoder()
    goal_enc = GoalEncoder()
    injector = GraphAttentionInjector(adapter, gnn, goal_enc)

    # At inference time:
    injector.prepare_session(graph, node_ids, text_embeddings, task_frame)
    # Then load Qwen3 and call model.generate() — hooks fire automatically.
    with injector.inject(model):
        output = model.generate(input_ids, ...)

    loop_logs = injector.get_loop_logs()  # per-iteration log entries
"""
from __future__ import annotations

import contextlib
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from v5.cross_attention import V5AttentionAdapter
from v5.gnn_encoder import GNN_HIDDEN_DIM, RGCNEncoder
from v5.goal_encoder import GoalEncoder, encode_task_frame
from v5.loop_state import LoopState
from v5.subgraph import ActiveSubgraph, GraphMemoryKV, build_active_subgraph

# Qwen3-4B: 36 transformer layers (0-indexed)
PLANNING_LAYER = 8
EVIDENCE_LAYER = 20


class GraphAttentionInjector:
    """Manages GNN pre-computation, hook registration, and loop log collection.

    One injector instance per inference session. Call prepare_session() once
    per question, then use inject() context manager during model.generate().
    """

    def __init__(
        self,
        adapter: V5AttentionAdapter,
        gnn: RGCNEncoder,
        goal_encoder: GoalEncoder,
        device: Optional[torch.device] = None,
    ):
        self.adapter = adapter
        self.gnn = gnn
        self.goal_encoder = goal_encoder
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Populated by prepare_session()
        self._active_subgraph: Optional[ActiveSubgraph] = None
        self._graph_kv: Optional[GraphMemoryKV] = None       # GNN output, session-cached
        self._goal: Optional[Tensor] = None                  # [1, GOAL_DIM]
        self._task_frame: Optional[dict] = None
        self._r_plan: int = 4
        self._r_evidence: int = 6

        # Populated during forward pass
        self._loop_logs: List[dict] = []
        self._plan_state: Optional[LoopState] = None
        self._evid_state: Optional[LoopState] = None

        # Hook-call accounting + run-once guards.
        # During model.generate() the prefill pass fires each layer hook once;
        # decode steps (seq_len==1) are skipped. These guards also prevent a
        # second prefill-shaped pass (beam search / chunked prefill) from
        # re-running the recurrent loops within one session.
        self._plan_ran: bool = False
        self._evid_ran: bool = False
        self._plan_hook_calls: int = 0
        self._evid_hook_calls: int = 0
        self.run_once_per_session: bool = True

        self._hooks: List = []

    def prepare_session(
        self,
        graph,
        node_ids: List[str],
        text_embeddings: Dict[str, List[float]],
        task_frame: dict,
        r_plan: int = 4,
        r_evidence: int = 6,
    ) -> None:
        """Pre-compute GNN embeddings, masks, and goal vector for this session.

        GNN runs once. K, V, planning_mask, evidence_mask cached in GraphMemoryKV.

        Args:
            graph: MemoryGraph instance (already pre-filtered to top-K nodes)
            node_ids: ordered node IDs in the subgraph
            text_embeddings: {node_id: [768-float]} pre-computed BERT embeddings
            task_frame: dict with task_family, question_mode, required_slots
        """
        self._task_frame = task_frame
        self._r_plan = r_plan
        self._r_evidence = r_evidence
        self._loop_logs = []
        self._plan_state = None
        self._evid_state = None
        self._plan_ran = False
        self._evid_ran = False
        self._plan_hook_calls = 0
        self._evid_hook_calls = 0

        # Build ActiveSubgraph (encoder inputs + planning/evidence masks)
        self._active_subgraph = build_active_subgraph(
            graph, node_ids, text_embeddings, self.device, task_frame
        )

        # GNN forward pass (once per session) — raw embeddings only
        self.gnn.eval()
        with torch.no_grad():
            self._graph_kv = self.gnn.encode_to_kv(
                self._active_subgraph.encoder_inputs,
                self._active_subgraph,
            )

        self.goal_encoder.eval()
        with torch.no_grad():
            self._goal = encode_task_frame(task_frame, self.device, self.goal_encoder)

    def _planning_hook(self, module, args, output):
        """Forward hook for layer 8 (planning block). Uses planning_mask.

        Skips decode steps (seq_len == 1) — hooks run only during prefill.
        Uses the LAST token's hidden state as the reasoning anchor (most
        context-rich position during prompt processing).
        """
        if self._graph_kv is None:
            return output

        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output

        # Skip token-by-token decode steps; only run during prefill
        if h.shape[1] == 1:
            return output

        # Run-once guard: skip a second prefill-shaped pass in one session
        if self.run_once_per_session and self._plan_ran:
            return output

        self._plan_hook_calls += 1
        h_anchor = h[:, -1, :]   # [B, d_lm] — last token, not first

        h_updated, state, logs = self.adapter.run_planning(
            h=h_anchor,
            goal=self._goal,
            graph_kv=self._graph_kv,
            r_max=self._r_plan,
            task_frame=self._task_frame,
        )
        self._plan_state = state
        self._plan_ran = True
        self._loop_logs.extend(logs)

        h_new = h.clone()
        h_new[:, -1, :] = h_updated   # write back to last position
        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        return h_new

    def _evidence_hook(self, module, args, output):
        """Forward hook for layer 20 (evidence block). Uses evidence_mask.

        Same prefill-only + last-token anchor rules as planning hook.
        """
        if self._graph_kv is None:
            return output

        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output

        if h.shape[1] == 1:
            return output

        # Run-once guard: skip a second prefill-shaped pass in one session
        if self.run_once_per_session and self._evid_ran:
            return output

        self._evid_hook_calls += 1
        h_anchor = h[:, -1, :]   # [B, d_lm]

        h_updated, state, logs = self.adapter.run_evidence(
            h=h_anchor,
            goal=self._goal,
            graph_kv=self._graph_kv,
            r_max=self._r_evidence,
            task_frame=self._task_frame,
        )
        self._evid_state = state
        self._evid_ran = True
        self._loop_logs.extend(logs)

        h_new = h.clone()
        h_new[:, -1, :] = h_updated
        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        return h_new

    @contextlib.contextmanager
    def inject(self, model: nn.Module):
        """Context manager: register hooks, yield, then remove them."""
        layer_modules = _get_transformer_layers(model)
        if len(layer_modules) <= max(PLANNING_LAYER, EVIDENCE_LAYER):
            raise ValueError(
                f"Model has only {len(layer_modules)} transformer layers; "
                f"expected >{max(PLANNING_LAYER, EVIDENCE_LAYER)}"
            )

        h_plan = layer_modules[PLANNING_LAYER].register_forward_hook(self._planning_hook)
        h_evid = layer_modules[EVIDENCE_LAYER].register_forward_hook(self._evidence_hook)
        self._hooks = [h_plan, h_evid]
        try:
            yield self
        finally:
            for h in self._hooks:
                h.remove()
            self._hooks = []

    def get_loop_logs(self) -> List[dict]:
        return list(self._loop_logs)

    def get_hook_call_counts(self) -> Dict[str, int]:
        """Per-session hook fire counts. With run_once_per_session=True both
        should be exactly 1 after a single generate() call."""
        return {
            "planning": self._plan_hook_calls,
            "evidence": self._evid_hook_calls,
        }

    def get_fallback_needed(self) -> bool:
        """True if evidence loop exited via max_loops with incomplete state."""
        from v5.exit_condition import fallback_needed
        if self._evid_state is None:
            return False
        return fallback_needed(self._evid_state)


def _get_transformer_layers(model: nn.Module) -> List[nn.Module]:
    """Locate the ordered list of transformer decoder blocks in Qwen3-4B.

    Tries common attribute paths used by Hugging Face Qwen3 and other
    decoder-only models. Raises ValueError if not found.
    """
    # Qwen3: model.model.layers
    for attr_path in [
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("layers",),
    ]:
        m = model
        try:
            for attr in attr_path:
                m = getattr(m, attr)
            if isinstance(m, (nn.ModuleList, list)) and len(m) > 0:
                return list(m)
        except AttributeError:
            continue
    raise ValueError(
        "Cannot locate transformer layer list in model. "
        "Check _get_transformer_layers() for this model architecture."
    )
