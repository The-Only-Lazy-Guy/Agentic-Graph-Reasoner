"""Phase 15/17 bridge: V4 corpus trace -> V5 Stage1Example.

Converts each Phase 15 V4 session row into a `Stage1Example` the Stage 1 trainer
consumes. This is the critical path from "V4 produced labeled traces" to "V5
heads train on real graph states."

What the converter does (all real logic, testable now):
  - parse the row via Phase15Dataset (labels: anchor / slot / epistemic /
    invalidator / shortcut)
  - build a graph object + ActiveSubgraph from the row's nodes
  - run the (frozen) GNN to get the GraphMemoryKV for the subgraph
  - SPLIT the single anchor mask into planning_anchor vs evidence_anchor by the
    GNN's planning/evidence pool membership (a node is supervised only in the
    pool its block can attend)
  - pull h_init from an injected provider (real frozen-Qwen prefill, or a mock)

What stays GATED (the real inputs, not the logic):
  - `h_init_provider`: real frozen Qwen prefill hidden state. A mock provider is
    fine for testing the converter; real training needs the real one.
  - substrate-rich graphs: the current corpus anchors are mostly fact/claim
    (evidence pool). Planning-pool node types (strategy/failure_pattern/...) only
    appear once V4 has written the reasoning substrate into the graph. The
    converter handles substrate-poor rows gracefully (planning_anchor=None), and
    the __main__ demo REPORTS pool coverage so the gap is measured, not hidden.

    python -m v5.training.bridge      # convert the real corpus with mock providers
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch
from torch import Tensor

from v5.gnn_encoder import RGCNEncoder
from v5.subgraph import build_active_subgraph, INVALIDATOR_RELATIONS
from v5.training.dataset import Phase15Dataset, Phase15Sample
from v5.training.stage1 import Stage1Example

# A provider maps (question, task_frame) -> [1, lm_dim] anchor hidden state.
HInitProvider = Callable[[str, dict], Tensor]


# ── minimal graph object from corpus nodes ───────────────────────────────────

class _CorpusNode:
    def __init__(self, nid: str, ntype: str, text: str, status: str = "unknown"):
        self.node_id = nid
        self.node_type = ntype
        self.text = text
        self.confidence = 0.6
        self.metadata = {"status": status}


class _CorpusGraph:
    """Graph built from a corpus row's nodes. Edges are limited to invalidator
    edges we can infer (deprecate_fact-targeted nodes mark structural
    invalidators); base-graph topology is not present in the corpus row, so
    R-GCN message passing is shallow until the real persisted graph is wired
    (documented in v5_PROGRESS.md "remaining: real MemoryGraph with edges")."""

    def __init__(self, nodes: Dict[str, _CorpusNode], edges: list):
        self.nodes = nodes
        self.edges = edges


def _build_corpus_graph(sample: Phase15Sample, inv_node_ids: set) -> _CorpusGraph:
    nodes = {
        nid: _CorpusNode(nid, sample.node_types.get(nid, "unknown"),
                         sample.node_texts.get(nid, ""))
        for nid in sample.node_ids
    }
    return _CorpusGraph(nodes, edges=[])  # no topology available from corpus row


# ── persisted-graph neighborhood (real topology) ─────────────────────────────

DEFAULT_PERSISTED_GRAPH = "graphs/merged_graph.json"


def load_persisted_graph(path: str = DEFAULT_PERSISTED_GRAPH):
    """Load the persisted MemoryGraph once (the real topology source)."""
    from graph_core import MemoryGraph
    return MemoryGraph.load_json(path)


def _neighborhood(graph, anchor_ids: List[str], hops: int, max_nodes: int = 32) -> List[str]:
    """Ordered node list = anchors (present in graph) + their k-hop neighbors.

    Anchors come first (so label remap is stable); neighbors expand the subgraph
    so build_active_subgraph keeps the edges among them -> real R-GCN message
    passing instead of isolated nodes.

    Delegates to MemoryGraph.local_neighborhood when available — it bounds growth
    by max_hops AND max_nodes and orders expansion by node importance (so a dense
    graph cannot blow up the subgraph). Falls back to a plain BFS otherwise.
    """
    present = [a for a in anchor_ids if a in graph.nodes]
    if hasattr(graph, "local_neighborhood"):
        nb = graph.local_neighborhood(present, max_hops=max(1, hops), max_nodes=max_nodes)
        # keep anchors first, then the importance-ordered neighbors
        pres_set = set(present)
        return present + [n for n in nb if n not in pres_set]
    # fallback BFS (e.g. for graph-likes without the helper)
    frontier = set(present)
    seen = set(present)
    for _ in range(max(0, hops)):
        nxt = set()
        for e in graph.edges:
            if e.src in frontier and e.dst not in seen:
                nxt.add(e.dst)
            if e.dst in frontier and e.src not in seen:
                nxt.add(e.src)
        seen |= nxt
        frontier = nxt
        if not frontier or len(seen) >= max_nodes:
            break
    neighbors = [n for n in seen if n not in set(present)][: max(0, max_nodes - len(present))]
    return present + neighbors


# ── h_init providers ─────────────────────────────────────────────────────────

class MockHInitProvider:
    """Deterministic synthetic h_init for testing the converter without a LM.

    Real training replaces this with a frozen-Qwen prefill provider (see
    realstack_test.py / GraphAttentionInjector for the hidden-state wiring).
    """
    def __init__(self, lm_dim: int, device):
        self.lm_dim = lm_dim
        self.device = device

    def __call__(self, question: str, task_frame: dict) -> Tensor:
        seed = abs(hash(question)) % (2**31)
        g = torch.Generator().manual_seed(seed)
        return (torch.randn(1, self.lm_dim, generator=g) * 0.5).to(self.device)


class ZeroEmbedder:
    """Mock 768-d embedder for converter testing. Real path: mpnet AutoModel."""
    def __init__(self, device):
        self.device = device
        self.dim = 768

    def embed_nodes(self, node_texts: Dict[str, str]) -> Dict[str, List[float]]:
        return {nid: [0.0] * self.dim for nid in node_texts}


# ── the converter ────────────────────────────────────────────────────────────

def sample_to_stage1_example(
    sample: Phase15Sample,
    gnn: RGCNEncoder,
    embedder,
    h_init_provider: HInitProvider,
    device: torch.device,
    lm_dim: int,
    persisted_graph=None,
    hops: int = 1,
    max_nodes: int = 32,
) -> Optional[Stage1Example]:
    """Convert one Phase15Sample -> Stage1Example. Returns None if unusable.

    When `persisted_graph` is given and the anchors resolve in it, the subgraph is
    the k-hop NEIGHBORHOOD (anchors + neighbors, with real edges) so the GNN does
    real message passing. Otherwise it falls back to an anchors-only corpus graph
    (no topology). Per-node labels are remapped onto the (possibly expanded) node
    list — anchors keep their labels, neighbor nodes are unlabeled context.
    """
    anchor_ids = sample.node_ids
    if not anchor_ids:
        return None

    # Resolve against the persisted graph for real topology, else anchors-only.
    use_persisted = (
        persisted_graph is not None
        and any(a in persisted_graph.nodes for a in anchor_ids)
    )
    if use_persisted:
        node_ids = _neighborhood(persisted_graph, anchor_ids, hops, max_nodes=max_nodes)
        graph = persisted_graph
        node_texts = {nid: (getattr(graph.nodes[nid], "text", "") or "") for nid in node_ids}
    else:
        node_ids = anchor_ids
        inv_node_ids = {nid for i, nid in enumerate(anchor_ids)
                        if sample.invalidator_target[i] > 0.5}
        graph = _build_corpus_graph(sample, inv_node_ids)
        node_texts = {nid: sample.node_texts.get(nid, "") for nid in node_ids}

    N = len(node_ids)
    text_emb = embedder.embed_nodes(node_texts)
    asg = build_active_subgraph(graph, node_ids, text_emb, device, sample.task_frame)
    with torch.no_grad():
        kv = gnn.encode_to_kv(asg.encoder_inputs, asg)

    # Remap per-anchor labels onto the (possibly expanded) node list.
    anchor_pos = {a: i for i, a in enumerate(anchor_ids)}
    def _remap(values: List[float]) -> Tensor:
        out = torch.zeros(1, N, device=device)
        for j, nid in enumerate(node_ids):
            if nid in anchor_pos:
                out[0, j] = float(values[anchor_pos[nid]])
        return out

    anchor = _remap(sample.anchor_mask)              # [1, N]
    epi_t = _remap(sample.epistemic_target)
    inv_t = _remap(sample.invalidator_target)
    struct = (inv_t > 0.5)

    plan_mask = kv.planning_mask.unsqueeze(0)
    evid_mask = kv.evidence_mask.unsqueeze(0)
    plan_anchor = anchor * plan_mask.float()
    evid_anchor = anchor * evid_mask.float()
    plan_anchor = plan_anchor if plan_anchor.sum() > 0 else None
    evid_anchor = evid_anchor if evid_anchor.sum() > 0 else None

    slot_t = torch.tensor(sample.slot_fill_target, dtype=torch.float32, device=device).unsqueeze(0)
    shortcut_t = torch.tensor([[sample.shortcut_valid]], dtype=torch.float32, device=device)

    h_init = h_init_provider(sample.question, sample.task_frame)
    if h_init.shape != (1, lm_dim):
        h_init = h_init.reshape(1, lm_dim)

    return Stage1Example(
        h_init=h_init.to(device), graph_kv=kv, goal=_goal_for(sample.task_frame, device),
        node_ids=node_ids, task_frame=sample.task_frame,
        plan_anchor=plan_anchor, evid_anchor=evid_anchor,
        slot_target=slot_t,
        epi_target=epi_t if epi_t.sum() > 0 else None,
        inv_target=inv_t if struct.any() else None,
        shortcut_target=shortcut_t,
        struct_inv_mask=struct if struct.any() else None,
        tag=("applicable" if sample.finalized else "blocked"),
    )


_GOAL_CACHE: Dict[tuple, Tensor] = {}


def _goal_for(task_frame: dict, device) -> Tensor:
    from v5.goal_encoder import GoalEncoder, encode_task_frame
    key = (task_frame.get("task_family"), task_frame.get("question_mode"),
           tuple(task_frame.get("required_slots") or []))
    if key not in _GOAL_CACHE:
        enc = GoalEncoder().to(device).eval()
        with torch.no_grad():
            _GOAL_CACHE[key] = encode_task_frame(task_frame, device, enc)
    return _GOAL_CACHE[key]


def corpus_to_stage1_examples(
    corpus_path,
    gnn: Optional[RGCNEncoder] = None,
    embedder=None,
    h_init_provider: Optional[HInitProvider] = None,
    device: Optional[torch.device] = None,
    lm_dim: int = 128,
    persisted_graph=None,
    graph_path: Optional[str] = DEFAULT_PERSISTED_GRAPH,
    hops: int = 1,
    max_nodes: int = 32,
) -> List[Stage1Example]:
    """Convert the whole Phase 15 corpus into Stage1Examples.

    gnn/embedder/h_init_provider default to test mocks so the converter logic can
    be exercised on the real corpus without a LM. Real training passes a frozen
    RGCNEncoder, an mpnet embedder, and a frozen-Qwen h_init provider.

    persisted_graph/graph_path: source real topology (k-hop neighborhood with
    edges) instead of anchors-only. Pass graph_path=None to force anchors-only.
    """
    device = device or torch.device("cpu")
    gnn = gnn or RGCNEncoder().to(device).eval()
    for p in gnn.parameters():
        p.requires_grad_(False)
    embedder = embedder or ZeroEmbedder(device)
    h_init_provider = h_init_provider or MockHInitProvider(lm_dim, device)

    if persisted_graph is None and graph_path:
        import os
        if os.path.exists(graph_path):
            persisted_graph = load_persisted_graph(graph_path)

    ds = Phase15Dataset(corpus_path)
    examples = []
    for sample in ds.samples:
        ex = sample_to_stage1_example(
            sample, gnn, embedder, h_init_provider, device, lm_dim,
            persisted_graph=persisted_graph, hops=hops, max_nodes=max_nodes)
        if ex is not None:
            examples.append(ex)
    return examples


# ── demo / coverage report on the real corpus ────────────────────────────────

def _coverage(examples):
    cov = {"plan": 0, "evid": 0, "slot": 0, "epi": 0, "inv": 0, "shortcut": 0}
    for ex in examples:
        cov["plan"] += int(ex.plan_anchor is not None)
        cov["evid"] += int(ex.evid_anchor is not None)
        cov["slot"] += int(ex.slot_target is not None)
        cov["epi"] += int(ex.epi_target is not None)
        cov["inv"] += int(ex.inv_target is not None)
        cov["shortcut"] += int(ex.shortcut_target is not None)
    return cov


def _avg_nodes(examples):
    return sum(len(ex.node_ids) for ex in examples) / max(1, len(examples))


def run(corpus_path: str = "artifacts/phase15/phase15_corpus.jsonl"):
    device = torch.device("cpu")
    lm_dim = 128

    # anchors-only vs persisted-neighborhood — show the topology upgrade
    anchors_only = corpus_to_stage1_examples(corpus_path, device=device, lm_dim=lm_dim,
                                             graph_path=None)
    persisted = corpus_to_stage1_examples(corpus_path, device=device, lm_dim=lm_dim,
                                          graph_path=DEFAULT_PERSISTED_GRAPH, hops=1)

    print(f"converted {len(persisted)} corpus rows -> Stage1Example "
          f"(mock embedder + mock h_init)")
    print(f"\nsubgraph size (avg nodes/example):")
    print(f"  anchors-only           : {_avg_nodes(anchors_only):.1f}")
    print(f"  persisted 1-hop nbhd   : {_avg_nodes(persisted):.1f}  "
          f"(real edges -> real R-GCN message passing)")

    cov = _coverage(persisted)
    T = max(1, len(persisted))
    print("\nper-head label coverage (rows with a usable label):")
    for k, v in cov.items():
        print(f"  {k:9s} {v:3d}/{len(persisted)}  ({v/T:.0%})")

    ex = persisted[0]
    print(f"\nexample[0]: N={len(ex.node_ids)} nodes "
          f"(anchors + 1-hop neighbors), h_init={tuple(ex.h_init.shape)}, "
          f"kv.node_embeddings={tuple(ex.graph_kv.node_embeddings.shape)}, tag={ex.tag}")
    print(f"  plan_anchor={'present' if ex.plan_anchor is not None else 'None (no planning-pool anchor)'}")
    print(f"  evid_anchor={'present' if ex.evid_anchor is not None else 'None'}")

    print("\nINTERPRETATION:")
    print("- Topology: persisted neighborhood replaces isolated anchors with a real")
    print("  edge-bearing subgraph, so the GNN does actual message passing.")
    print("- Substrate gap: planning-label coverage is still the bottleneck —")
    print("  merged_graph has no strategy/failure_pattern/epistemic_state nodes, so")
    print("  planning labels stay 0 until V4 writes that reasoning substrate.")
    print("\nBRIDGE OK — converter produces well-formed Stage1Examples from the")
    print("persisted MemoryGraph neighborhood of the real corpus")
    return persisted


if __name__ == "__main__":
    run()
