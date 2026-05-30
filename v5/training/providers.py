"""Real providers for V5 Stage 1 training: mpnet embeddings + frozen-Qwen h_init.

These replace the mocks (ZeroEmbedder, MockHInitProvider) in the bridge so Stage 1
can train on REAL graph node embeddings and REAL LM hidden states. Both use
transformers AutoModel — sentence_transformers segfaults when co-loaded with
torch_geometric on this machine (see v5_PROGRESS.md). Run heavy combos with
KMP_DUPLICATE_LIB_OK=TRUE.

  RealEmbedder              — all-mpnet-base-v2, 768-d, mean-pooled + L2-normalized
  FrozenQwenHInitProvider   — frozen Qwen prefill hidden state at an anchor layer
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import Tensor

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"   # 768-d, matches GNN
DEFAULT_LM = "Qwen/Qwen2.5-1.5B"                          # HF, hidden=1536, 28 layers
ANCHOR_LAYER = 8   # take the prefill hidden state at this transformer layer's output


class RealEmbedder:
    """mpnet-768 node embedder via transformers AutoModel + mean pooling."""

    def __init__(self, device: torch.device, repo: str = EMBED_MODEL):
        from transformers import AutoTokenizer, AutoModel
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(repo)
        self.model = AutoModel.from_pretrained(repo).to(device).eval()
        self.dim = self.model.config.hidden_size
        assert self.dim == 768, f"expected 768-d embedder, got {self.dim}"

    @torch.no_grad()
    def embed_nodes(self, node_texts: Dict[str, str]) -> Dict[str, List[float]]:
        ids = list(node_texts.keys())
        if not ids:
            return {}
        enc = self.tok([node_texts[i] for i in ids], padding=True,
                       truncation=True, max_length=256, return_tensors="pt").to(self.device)
        out = self.model(**enc).last_hidden_state            # [B, T, 768]
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        return {i: emb[k].tolist() for k, i in enumerate(ids)}


class FrozenQwenHInitProvider:
    """(question, task_frame) -> [1, hidden] frozen-Qwen prefill hidden state.

    Loads Qwen once, frozen. For each question it runs a single prefill forward
    with output_hidden_states and returns the LAST token's hidden state at
    `anchor_layer` — the same anchor the planning hook uses (last token, prefill).
    hidden_states is (embeddings, layer_0_out, ..., layer_{N-1}_out), so the
    output of transformer layer L is hidden_states[L+1]; we index that.

    Results are cached per question (the corpus reuses questions across rows).
    `hidden_size` is the LM width — build the adapter with lm_hidden_dim=this.
    """

    def __init__(self, model_name: str = DEFAULT_LM, anchor_layer: int = ANCHOR_LAYER,
                 device: Optional[torch.device] = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.anchor_layer = anchor_layer
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.hidden_size = self.model.config.hidden_size
        n_layers = self.model.config.num_hidden_layers
        assert anchor_layer < n_layers, f"anchor_layer {anchor_layer} >= {n_layers} layers"
        self._cache: Dict[str, Tensor] = {}

    @torch.no_grad()
    def __call__(self, question: str, task_frame: dict) -> Tensor:
        if question in self._cache:
            return self._cache[question]
        enc = self.tok(question, return_tensors="pt", truncation=True,
                       max_length=512).to(self.device)
        out = self.model(**enc, output_hidden_states=True)
        # output of transformer layer `anchor_layer` == hidden_states[anchor_layer + 1]
        h = out.hidden_states[self.anchor_layer + 1][:, -1, :]   # [1, hidden]
        h = h.detach().to(self.device)
        self._cache[question] = h
        return h
