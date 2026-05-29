"""R-GCN encoder: MemoryGraph -> per-node embeddings (K, V for cross-attention).

Architecture:
  Input  per node: BERT text embedding (768, frozen) + node_type embedding (64)
                   + confidence scalar (1) + status embedding (16) = 849 dims
  Project: Linear(849, 256) -> LayerNorm
  Layer 1: RGCNConv(256, 256, num_relations=12) -> ReLU -> Dropout(0.1)
  Layer 2: RGCNConv(256, 256, num_relations=12) -> LayerNorm
  Output:  [N x 256] per-node embeddings

Edge type IDs are stable integers from reasoning.graph_relations.RELATION_TYPE_ID.
Unknown relation strings map to type_id 5 (RELATED) as fallback.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import RGCNConv

from reasoning.graph_relations import RELATION_TYPE_ID

# Node types and their indices (stable order — append only)
NODE_TYPE_VOCAB = [
    "fact", "claim", "strategy", "failure_pattern", "solved_subgoal",
    "reasoning_atom", "control_rule", "epistemic_state", "procedure",
    "application", "reasoning_chain", "unknown",
]
NODE_TYPE_ID: Dict[str, int] = {t: i for i, t in enumerate(NODE_TYPE_VOCAB)}
NUM_NODE_TYPES = len(NODE_TYPE_VOCAB)

# Epistemic status vocab for epistemic_state nodes
EPISTEMIC_STATUS_VOCAB = ["verified", "supported", "uncertain", "mismatched", "unknown"]
EPISTEMIC_STATUS_ID: Dict[str, int] = {s: i for i, s in enumerate(EPISTEMIC_STATUS_VOCAB)}
NUM_EPISTEMIC_STATUSES = len(EPISTEMIC_STATUS_VOCAB)

NUM_RELATIONS = len(RELATION_TYPE_ID)   # 12
TEXT_EMBED_DIM = 768
NODE_TYPE_EMBED_DIM = 64
EPISTEMIC_EMBED_DIM = 16
CONFIDENCE_DIM = 1
INPUT_DIM = TEXT_EMBED_DIM + NODE_TYPE_EMBED_DIM + EPISTEMIC_EMBED_DIM + CONFIDENCE_DIM  # 849
GNN_HIDDEN_DIM = 256


def _node_type_id(node_type: str) -> int:
    return NODE_TYPE_ID.get(node_type, NODE_TYPE_ID["unknown"])


def _epistemic_status_id(status: str) -> int:
    return EPISTEMIC_STATUS_ID.get(status, EPISTEMIC_STATUS_ID["unknown"])


def _relation_type_id(relation: str) -> int:
    return RELATION_TYPE_ID.get(relation, RELATION_TYPE_ID.get("related", 5))


class GraphEncoderInputs:
    """Tensors extracted from a MemoryGraph subgraph for GNN forward pass.

    All tensors live on the same device as the caller.
    """
    def __init__(
        self,
        node_ids: List[str],
        text_embeddings: Tensor,        # [N, 768]  frozen BERT embeddings
        node_type_ids: Tensor,          # [N]       long
        epistemic_status_ids: Tensor,   # [N]       long
        confidences: Tensor,            # [N, 1]    float
        edge_index: Tensor,             # [2, E]    long  (src, dst row-major)
        edge_type: Tensor,              # [E]       long  relation type IDs
    ):
        self.node_ids = node_ids
        self.text_embeddings = text_embeddings
        self.node_type_ids = node_type_ids
        self.epistemic_status_ids = epistemic_status_ids
        self.confidences = confidences
        self.edge_index = edge_index
        self.edge_type = edge_type

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def device(self) -> torch.device:
        return self.text_embeddings.device


class RGCNEncoder(nn.Module):
    """Two-layer R-GCN that produces [N x GNN_HIDDEN_DIM] node embeddings.

    These are used as fixed K, V matrices for the cross-attention loops in
    the LM adapter. GNN runs once per session; K/V are cached.
    """

    def __init__(
        self,
        text_embed_dim: int = TEXT_EMBED_DIM,
        node_type_embed_dim: int = NODE_TYPE_EMBED_DIM,
        epistemic_embed_dim: int = EPISTEMIC_EMBED_DIM,
        hidden_dim: int = GNN_HIDDEN_DIM,
        num_relations: int = NUM_RELATIONS,
        dropout: float = 0.1,
    ):
        super().__init__()
        input_dim = text_embed_dim + node_type_embed_dim + epistemic_embed_dim + CONFIDENCE_DIM

        self.node_type_embed = nn.Embedding(NUM_NODE_TYPES, node_type_embed_dim)
        self.epistemic_embed = nn.Embedding(NUM_EPISTEMIC_STATUSES, epistemic_embed_dim)

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.conv1 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, inputs: GraphEncoderInputs) -> Tensor:
        """Return [N x hidden_dim] node embeddings."""
        type_emb = self.node_type_embed(inputs.node_type_ids)           # [N, 64]
        epi_emb = self.epistemic_embed(inputs.epistemic_status_ids)     # [N, 16]
        x = torch.cat([
            inputs.text_embeddings,   # [N, 768]
            type_emb,                 # [N, 64]
            epi_emb,                  # [N, 16]
            inputs.confidences,       # [N, 1]
        ], dim=-1)                    # [N, 849]

        x = self.input_proj(x)                                          # [N, 256]

        # Layer 1 with residual
        h1 = self.conv1(x, inputs.edge_index, inputs.edge_type)
        h1 = self.norm1(self.dropout(self.act(h1)) + x)

        # Layer 2 with residual
        h2 = self.conv2(h1, inputs.edge_index, inputs.edge_type)
        h2 = self.norm2(h2 + h1)

        return h2   # [N, 256]


def build_encoder_inputs(
    graph,
    node_ids: List[str],
    text_embeddings: Dict[str, List[float]],
    device: torch.device,
) -> GraphEncoderInputs:
    """Convert a MemoryGraph subgraph into GNN-ready tensors.

    Args:
        graph: MemoryGraph instance
        node_ids: ordered list of node IDs to include (subgraph)
        text_embeddings: pre-computed {node_id: [768-float]} from embedder
        device: target torch device
    """
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    N = len(node_ids)

    text_vecs = []
    ntype_ids = []
    epi_ids = []
    confs = []

    for nid in node_ids:
        node = graph.nodes.get(nid)
        vec = text_embeddings.get(nid)
        if vec is None:
            vec = [0.0] * TEXT_EMBED_DIM
        text_vecs.append(vec)

        if node is not None:
            ntype_ids.append(_node_type_id(node.node_type))
            conf = float(getattr(node, "confidence", 0.5) or 0.5)
            meta = getattr(node, "metadata", {}) or {}
            status = meta.get("status", "unknown") if isinstance(meta, dict) else "unknown"
            epi_ids.append(_epistemic_status_id(status))
            confs.append(conf)
        else:
            ntype_ids.append(_node_type_id("unknown"))
            epi_ids.append(_epistemic_status_id("unknown"))
            confs.append(0.5)

    # Build edge tensors — only include edges where both endpoints are in subgraph
    src_list, dst_list, rel_list = [], [], []
    for edge in graph.edges:
        s_idx = id_to_idx.get(edge.src)
        d_idx = id_to_idx.get(edge.dst)
        if s_idx is None or d_idx is None:
            continue
        src_list.append(s_idx)
        dst_list.append(d_idx)
        rel_list.append(_relation_type_id(edge.relation))

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long, device=device)
        edge_type = torch.tensor(rel_list, dtype=torch.long, device=device)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        edge_type = torch.zeros((0,), dtype=torch.long, device=device)

    return GraphEncoderInputs(
        node_ids=node_ids,
        text_embeddings=torch.tensor(text_vecs, dtype=torch.float32, device=device),
        node_type_ids=torch.tensor(ntype_ids, dtype=torch.long, device=device),
        epistemic_status_ids=torch.tensor(epi_ids, dtype=torch.long, device=device),
        confidences=torch.tensor([[c] for c in confs], dtype=torch.float32, device=device),
        edge_index=edge_index,
        edge_type=edge_type,
    )
