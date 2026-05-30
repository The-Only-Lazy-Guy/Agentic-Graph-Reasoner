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
) -> Optional[Stage1Example]:
    """Convert one Phase15Sample -> Stage1Example. Returns None if unusable."""
    node_ids = sample.node_ids
    N = len(node_ids)
    if N == 0:
        return None

    # nodes flagged by deprecate_fact patches act as structural invalidators here
    inv_node_ids = {nid for i, nid in enumerate(node_ids)
                    if sample.invalidator_target[i] > 0.5}
    graph = _build_corpus_graph(sample, inv_node_ids)

    text_emb = embedder.embed_nodes({nid: sample.node_texts.get(nid, "") for nid in node_ids})
    asg = build_active_subgraph(graph, node_ids, text_emb, device, sample.task_frame)
    with torch.no_grad():
        kv = gnn.encode_to_kv(asg.encoder_inputs, asg)

    anchor = torch.tensor(sample.anchor_mask, dtype=torch.float32, device=device).unsqueeze(0)  # [1,N]
    plan_mask = kv.planning_mask.unsqueeze(0)
    evid_mask = kv.evidence_mask.unsqueeze(0)

    # Split the single anchor mask by pool: a node is supervised only in the pool
    # its block attends. None when that pool has no anchored node (partial labels).
    plan_anchor = anchor * plan_mask.float()
    evid_anchor = anchor * evid_mask.float()
    plan_anchor = plan_anchor if plan_anchor.sum() > 0 else None
    evid_anchor = evid_anchor if evid_anchor.sum() > 0 else None

    epi_t = torch.tensor(sample.epistemic_target, dtype=torch.float32, device=device).unsqueeze(0)
    inv_t = torch.tensor(sample.invalidator_target, dtype=torch.float32, device=device).unsqueeze(0)
    struct = (inv_t > 0.5)
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
) -> List[Stage1Example]:
    """Convert the whole Phase 15 corpus into Stage1Examples.

    gnn/embedder/h_init_provider default to test mocks so the converter logic can
    be exercised on the real corpus without a LM. Real training passes a frozen
    RGCNEncoder, an mpnet embedder, and a frozen-Qwen h_init provider.
    """
    device = device or torch.device("cpu")
    gnn = gnn or RGCNEncoder().to(device).eval()
    for p in gnn.parameters():
        p.requires_grad_(False)
    embedder = embedder or ZeroEmbedder(device)
    h_init_provider = h_init_provider or MockHInitProvider(lm_dim, device)

    ds = Phase15Dataset(corpus_path)
    examples = []
    for sample in ds.samples:
        ex = sample_to_stage1_example(sample, gnn, embedder, h_init_provider, device, lm_dim)
        if ex is not None:
            examples.append(ex)
    return examples


# ── demo / coverage report on the real corpus ────────────────────────────────

def run(corpus_path: str = "artifacts/phase15/phase15_corpus.jsonl"):
    device = torch.device("cpu")
    lm_dim = 128
    examples = corpus_to_stage1_examples(corpus_path, device=device, lm_dim=lm_dim)
    print(f"converted {len(examples)} corpus rows -> Stage1Example (mock embedder + mock h_init)")

    # label coverage — quantifies the substrate gap honestly
    cov = {"plan": 0, "evid": 0, "slot": 0, "epi": 0, "inv": 0, "shortcut": 0}
    for ex in examples:
        cov["plan"] += int(ex.plan_anchor is not None)
        cov["evid"] += int(ex.evid_anchor is not None)
        cov["slot"] += int(ex.slot_target is not None)
        cov["epi"] += int(ex.epi_target is not None)
        cov["inv"] += int(ex.inv_target is not None)
        cov["shortcut"] += int(ex.shortcut_target is not None)
    T = max(1, len(examples))
    print("\nper-head label coverage (rows with a usable label):")
    for k, v in cov.items():
        print(f"  {k:9s} {v:3d}/{len(examples)}  ({v/T:.0%})")

    # shape sanity on the first example
    ex = examples[0]
    N = len(ex.node_ids)
    print(f"\nexample[0]: N={N} nodes, h_init={tuple(ex.h_init.shape)}, "
          f"kv.node_embeddings={tuple(ex.graph_kv.node_embeddings.shape)}, tag={ex.tag}")
    print(f"  plan_anchor={'present' if ex.plan_anchor is not None else 'None (no planning-pool anchor)'}")
    print(f"  evid_anchor={'present' if ex.evid_anchor is not None else 'None'}")

    print("\nINTERPRETATION: planning-label coverage is the substrate gap — base-graph")
    print("corpus anchors are mostly evidence-pool (fact/claim). Planning labels rise")
    print("once V4 writes strategy/failure_pattern/epistemic_state substrate into the graph.")
    print("\nBRIDGE OK — converter produces well-formed Stage1Examples from the real corpus")
    return examples


if __name__ == "__main__":
    run()
