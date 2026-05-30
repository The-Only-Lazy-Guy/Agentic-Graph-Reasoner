"""V5 Phase 17 minimal real-stack test.

Proves the loop logs stay sane with REAL components (no training, no GPU spend
beyond one frozen prefill):

  real mpnet-768 embeddings
    + real base graph (graphs/*.json) for evidence-pool nodes
    + injected reasoning-substrate nodes for planning-pool nodes
    + real frozen decoder LM (Qwen2.5-1.5B) prefill hidden states
    + V5 adapter via the real GraphAttentionInjector hook path
    -> loop logs

Why injected substrate nodes: the base knowledge graphs use node types
(fact/claim/theorem/...) that populate the EVIDENCE pool, but the PLANNING pool
types (strategy/failure_pattern/epistemic_state) are the reasoning substrate V4
writes into the graph over time. To exercise both pools on real data we load a
real base graph and inject a few real-text substrate nodes — exactly how V5 sees
the graph in production (base knowledge + V4-learned substrate).

The frozen LM is Qwen2.5-1.5B (HF format, supports forward hooks). The real 4B
target is GGUF (llama.cpp) and needs an HF-format export or llama-cpp hooks
before it can be swapped in — Phase 18 concern. Adapter hidden dim is read from
the loaded model's config, so the swap is a config change.

    python -m v5.realstack_test
    python -m v5.realstack_test --graph graphs/cs1.json --model Qwen/Qwen2.5-0.5B-Instruct
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List

import torch

from graph_core import MemoryGraph, Node, Edge
from reasoning.graph_relations import Rel
from v5.adapter import GraphAttentionInjector, PLANNING_LAYER, EVIDENCE_LAYER
from v5.cross_attention import V5AttentionAdapter
from v5.gnn_encoder import RGCNEncoder
from v5.goal_encoder import GoalEncoder

EMBED_MODEL = "all-mpnet-base-v2"        # 768-dim, matches GNN TEXT_EMBED_DIM
DEFAULT_LM = "Qwen/Qwen2.5-1.5B"         # HF format, hidden=1536, 28 layers


# ── real embedder ────────────────────────────────────────────────────────────

class RealEmbedder:
    """mpnet-768 sentence embedder via transformers AutoModel + mean pooling.

    Uses transformers directly rather than sentence_transformers: on this
    machine the sentence_transformers native stack segfaults when co-loaded
    with torch_geometric / the LM. Mean-pooled, L2-normalized embeddings match
    sentence-transformers/all-mpnet-base-v2 semantics.
    """
    def __init__(self, device: torch.device):
        from transformers import AutoTokenizer, AutoModel
        self.device = device
        repo = f"sentence-transformers/{EMBED_MODEL}"
        self.tok = AutoTokenizer.from_pretrained(repo)
        self.model = AutoModel.from_pretrained(repo).to(device).eval()
        self.dim = self.model.config.hidden_size
        assert self.dim == 768, f"expected 768-dim embedder, got {self.dim}"

    @torch.no_grad()
    def embed_nodes(self, node_texts: Dict[str, str]) -> Dict[str, List[float]]:
        ids = list(node_texts.keys())
        enc = self.tok([node_texts[i] for i in ids], padding=True,
                       truncation=True, return_tensors="pt").to(self.device)
        out = self.model(**enc).last_hidden_state            # [B, T, 768]
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        return {i: emb[k].tolist() for k, i in enumerate(ids)}


# ── build a real-ish test subgraph ───────────────────────────────────────────

def build_test_graph(base_path: str):
    """Load a real base graph, take a few evidence nodes, inject substrate nodes.

    Returns (graph, node_ids, question, task_frame).
    """
    g = MemoryGraph.load_json(base_path)

    # Real evidence-pool nodes from the base graph (fact / claim / application)
    EVID_TYPES = {"fact", "claim", "application"}
    evid_ids = [nid for nid, n in g.nodes.items() if n.node_type in EVID_TYPES][:6]

    # Inject reasoning-substrate planning-pool nodes (real text, V4-style)
    substrate = [
        Node(
            id="bsearch_strategy",
            text="Strategy: apply binary search to locate a target in a sorted array "
                 "by halving the search interval each step.",
            node_type="strategy", confidence=0.8, importance=0.9,
            metadata={"status": "uncertain"},
        ),
        Node(
            id="unsorted_array_failure",
            text="Failure pattern: binary search returns wrong results when the input "
                 "array is not sorted under the comparison used.",
            node_type="failure_pattern", confidence=0.85, importance=0.9,
            metadata={"status": "uncertain"},
        ),
        Node(
            id="bsearch_applicability_epi",
            text="It is currently uncertain whether the array is guaranteed sorted, "
                 "so binary search applicability is unverified.",
            node_type="epistemic_state", confidence=0.6, importance=0.7,
            metadata={"status": "uncertain"},
        ),
        Node(
            id="sorted_precondition_verified",
            text="The sorted-order precondition for binary search has been verified "
                 "for this input.",
            node_type="epistemic_state", confidence=0.9, importance=0.8,
            metadata={"status": "verified"},
        ),
    ]
    for n in substrate:
        g.nodes[n.id] = n

    # The failure pattern structurally invalidates the strategy
    g.edges.append(Edge(
        src="unsorted_array_failure", dst="bsearch_strategy",
        relation=Rel.INVALIDATED_BY, strength=0.9,
    ))

    node_ids = [n.id for n in substrate] + evid_ids

    question = ("Is binary search applicable to find a target in this array, "
                "and what precondition must hold?")
    task_frame = {
        "task_family": "algorithm_applicability",
        "question_mode": "direct_relationship",
        "required_slots": ["verdict", "reason"],
    }
    return g, node_ids, question, task_frame


# ── real-stack run ───────────────────────────────────────────────────────────

def run(graph_path: str = "graphs/algo3_binary_search.json",
        model_name: str = DEFAULT_LM,
        device_str: str = None):
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}  embedder={EMBED_MODEL}  lm={model_name}")

    # 1. real graph + injected substrate
    g, node_ids, question, task_frame = build_test_graph(graph_path)
    print(f"\nsubgraph nodes ({len(node_ids)}):")
    for nid in node_ids:
        print(f"  {nid:34s} {g.nodes[nid].node_type:16s} "
              f"status={g.nodes[nid].metadata.get('status','-')}")

    # 2. real mpnet embeddings
    print("\nembedding nodes with mpnet...")
    embedder = RealEmbedder(device)
    node_texts = {nid: g.nodes[nid].text for nid in node_ids}
    text_emb = embedder.embed_nodes(node_texts)
    print(f"  embedded {len(text_emb)} nodes, dim={len(next(iter(text_emb.values())))}")

    # 3. real frozen LM
    print(f"\nloading frozen LM {model_name}...")
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
    cfg = AutoConfig.from_pretrained(model_name)
    lm_dim = cfg.hidden_size
    n_layers = cfg.num_hidden_layers
    assert n_layers > max(PLANNING_LAYER, EVIDENCE_LAYER), \
        f"model has {n_layers} layers; need >{max(PLANNING_LAYER, EVIDENCE_LAYER)}"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"  loaded: hidden={lm_dim}, layers={n_layers} (hooks at L{PLANNING_LAYER}/L{EVIDENCE_LAYER})")

    # 4. V5 adapter sized to the LM
    adapter = V5AttentionAdapter(r_plan=4, r_evidence=6, lm_hidden_dim=lm_dim).to(device).eval()
    gnn = RGCNEncoder().to(device).eval()
    goal_enc = GoalEncoder().to(device).eval()
    injector = GraphAttentionInjector(adapter, gnn, goal_enc, device=device)

    # 5. prepare session (GNN runs once, caches GraphMemoryKV)
    injector.prepare_session(g, node_ids, text_emb, task_frame, r_plan=4, r_evidence=6)
    kv = injector._graph_kv
    print("\npool routing (from real GNN encode):")
    for i, nid in enumerate(node_ids):
        print(f"  {nid:34s} plan={bool(kv.planning_mask[i])!s:5} "
              f"evid={bool(kv.evidence_mask[i])!s:5} inv={kv.invalidator_flags[i].item():.0f}")

    # 6. real prefill forward with hooks active
    print("\nrunning real Qwen prefill with V5 hooks...")
    inputs = tok(question, return_tensors="pt").to(device)
    with injector.inject(model):
        with torch.no_grad():
            model(**inputs)   # single prefill pass fires L8/L20 hooks once each

    # 7. inspect loop logs
    logs = injector.get_loop_logs()
    print(f"\nhook call counts: {injector.get_hook_call_counts()}")
    print(f"loop log entries: {len(logs)}")

    plan_logs = [e for e in logs if e["layer"] == PLANNING_LAYER]
    evid_logs = [e for e in logs if e["layer"] == EVIDENCE_LAYER]

    def show(tag, entries):
        print(f"\n=== {tag} ({len(entries)} iters) ===")
        for e in entries:
            tops = [f"{n}:{w:.3f}" for n, w in e["top_nodes"] if w > -1e8]
            print(f"  loop {e['loop']}: exit={e['exit_reason']}  top={tops}")

    show(f"PLANNING (L{PLANNING_LAYER})", plan_logs)
    show(f"EVIDENCE (L{EVIDENCE_LAYER})", evid_logs)

    # 8. sanity assertions (deterministic invariants)
    assert injector.get_hook_call_counts() == {"planning": 1, "evidence": 1}, \
        "hooks should fire exactly once each in one prefill"
    if plan_logs:
        assert plan_logs[-1]["exit_reason"] is not None
    if evid_logs:
        assert evid_logs[-1]["exit_reason"] is not None

    # planning top nodes must be planning-pool; evidence top nodes evidence-pool
    plan_pool = {nid for i, nid in enumerate(node_ids) if bool(kv.planning_mask[i])}
    evid_pool = {nid for i, nid in enumerate(node_ids) if bool(kv.evidence_mask[i])}
    for e in plan_logs:
        for n, w in e["top_nodes"]:
            if w > -1e8:
                assert n in plan_pool, f"planning attended out-of-pool node {n}"
    for e in evid_logs:
        for n, w in e["top_nodes"]:
            if w > -1e8:
                assert n in evid_pool, f"evidence attended out-of-pool node {n}"

    fb = injector.get_fallback_needed()
    print(f"\nfallback_needed (untrained heads): {fb}")
    print("\nREAL-STACK TEST PASSED — loop logs sane with real embeddings + real h_init")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="graphs/algo3_binary_search.json")
    ap.add_argument("--model", default=DEFAULT_LM)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    run(args.graph, args.model, args.device)
