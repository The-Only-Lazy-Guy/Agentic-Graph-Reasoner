# V5 Architecture: Graph↔LM Cross-Attention Design

> **Status**: Design finalized (2026-05-29). Implementation begins after Phase 15 corpus collection.
> **Reference**: All decisions in this document were resolved through an explicit design interview before any code was written.

---

## 1. Problem Statement

The V4 architecture connects the LM to the graph **symbolically** — the graph is searched via TF-IDF + semantic retrieval, results are injected into the system prompt as text, and the LM reasons over them. The graph is *consulted* but not *deeply integrated*.

The core limitation:

```
query quality  →  retrieval quality  →  reasoning quality
```

In V4, the query is the raw question. The LM has no learned mechanism for forming a *better* query based on its current reasoning state or goal.

**V5 goal**: Make the query vector itself a learned, goal-conditioned reasoning instrument. The LM should learn *where to look* based on *what it is trying to solve*, not just *what words appear in the question*.

The philosophical shift:

```
V4: LM reasons, graph is referenced
V5: LM and graph co-reason via differentiable attention
```

---

## 2. Design Principles

These are non-negotiable constraints for the entire V5 design:

1. **The LM must stay a pure reasoner.** The LM should never become a memory store. Its core reasoning circuits must not be corrupted by graph content. The graph is structured *external* memory — the LM queries it, not the other way around.

2. **The graph is object-oriented.** Nodes have typed roles (`fact`, `claim`, `strategy`, `procedure`, `failure_pattern`, `solved_subgoal`). The architecture must be type-aware, not just embedding-aware.

3. **No dynamic procedure creation.** The LM can create `StrategyNode`s (reusable recipes) but cannot write Python code for new `ProcedureNode`s. Procedures are developer-authored tools. Strategies are model-authored compositions.

4. **GPU budget is constrained.** We rent GPUs. Training must be efficient. Overfitting risk is real on small corpora. Every trainable parameter must earn its keep.

5. **Phase 15 corpus feeds Phase 16 training.** No training before corpus exists. Architecture design runs in parallel with corpus collection.

---

## 3. Attention Level Decision

We identified three levels of LM↔graph attention:

| Level | Description | Training target |
|---|---|---|
| **Level 1** | Vanilla cross-attention. Q = LM hidden state, K/V = graph embeddings. Learns semantic relevance. | What matches context? |
| **Level 2** | Policy-aware attention. Q is goal-conditioned. Learns useful-next-step retrieval. | What leads to successful reasoning? |
| **Level 3** | Attention as heuristic search. Attention dynamics replace explicit graph traversal. Differentiable search guidance. | Where to look next? |

**Decision: Target Level 2, with Level 3 as the north star for a later phase.**

Rationale:
- Level 1 is what V4 approximates with TF-IDF. Not enough.
- Level 3 is the ultimate goal but requires much more training data and infrastructure.
- Level 2 is the correct next step: goal-conditioned Q vectors trained on reasoning trajectories.

---

## 4. Graph→LM Bridge Architecture

**Decision: GNN → typed node embeddings → LM cross-attends.**

```
MemoryGraph (typed nodes: fact, claim, strategy, failure_pattern, ...)
        │
        ▼
   GNN encoder
   (type-aware message passing)
        │
        ▼
  Per-node embeddings [N × d_gnn]
  (K, V matrices for cross-attention)
        │
        ▼
LM cross-attention at injection layers
  Q = f(LM hidden state, TaskFrame goal vector)
  K, V = GNN node embeddings
        │
        ▼
  Context vector injected into LM hidden state
```

Why GNN on the graph side (not raw text embeddings):
- The GNN can propagate information across typed edges (`leveraged`, `entails`, `contradicts`, `overlaps`)
- A `strategy` node's embedding will be informed by the `fact` nodes it was built from
- Type-aware message passing lets `failure_pattern` nodes repel attention (they encode anti-patterns) while `strategy` nodes attract it during planning
- This is not possible with static per-node text embeddings

---

## 5. Goal Conditioning

**Decision: Use `TaskFrame` from the existing `micro_controller` as the goal signal.**

The `micro_controller` already produces, at query time:
- `task_family` (enum: `algorithm_applicability`, `direct_judgment`, `procedure_execution`, ...)
- `question_mode` (e.g., `explain`, `compare`, `verify`, `design`)
- `required_slots` (the specific sub-questions the model must fill)
- `task_signature` (a deterministic fingerprint of the task type)
- `subgoals` (ordered list of reasoning steps)

These are encoded into a **goal vector** `g`:

```python
g = encode_task_frame(task_family, question_mode, required_slots)
# g: [d_goal] vector

Q = W_q(concat(h_lm, g))
# h_lm: LM hidden state at injection layer
# W_q: LoRA-trained projection
# Q: goal-conditioned query vector
```

This means the *same question* asked with different goals would produce different Q vectors and therefore attend to different graph nodes — which is exactly the intended behavior (see the "explain architecture" vs. "debug failure" example from the design discussion).

**Upgrade condition**: If during Phase 15 corpus analysis we find that `TaskFrame` signals are too coarse to distinguish fine-grained retrieval needs, we will extend `micro_controller` to produce richer goal encodings. This is explicitly deferred.

---

## 6. Injection Points

**Decision: Two injection points — Layer ~8 (plan pass) and Layer ~20 (evidence pass).**

Qwen3-4B has 36 transformer layers.

```
Layer 0-7:   Token processing, early contextualization
Layer 8:     ◄── INJECTION POINT 1: Goal-setting / planning pass
Layer 9-19:  Reasoning integration
Layer 20:    ◄── INJECTION POINT 2: Evidence retrieval pass
Layer 21-35: Answer formation
```

### Injection Point 1 — Layer 8: Plan Pass

- **Node pool**: `StrategyNode`, `FailurePatternNode`, `ControlRuleNode`, `ReasoningChainNode`, `EpistemicStateNode` (when `status=uncertain` or `requires_evidence_before_shortcut=True`)
- **Purpose**: At this early stage, the LM has processed the question but hasn't committed to a reasoning path. It attends broadly to *strategy*, *anti-pattern*, and *deductive chain* nodes to select an appropriate reasoning plan. Epistemic states with unresolved open questions surface here to pre-warn the model before it commits to a shortcut.
- **Effect**: The retrieved strategy context is added to the hidden state, biasing subsequent layers toward the correct reasoning structure for this task family.
- **Analogy**: "Before I start — what worked before? What traps should I avoid? What am I not yet certain about?"

### Injection Point 2 — Layer 20: Evidence Pass

- **Node pool**: `FactNode`, `ClaimNode`, `ApplicationNode`, `SolvedSubgoalNode`, `EpistemicStateNode` (when `status=verified` or `status=supported`)
- **Purpose**: Mid-reasoning, the LM has partially formulated its approach. It queries for specific factual evidence AND the graph's belief-status on that evidence. A high-confidence `EpistemicStateNode` pointing to a `SolvedSubgoalNode` signals the model can safely shortcut. A low-confidence or invalidated one signals it must verify further.
- **Effect**: The retrieved evidence + epistemic signal is added to the hidden state, providing both facts and confidence metadata before the model commits to an answer.
- **Analogy**: "Given my plan — what does the graph say about this claim? And how confident is the graph in that answer?"

### Why two different node pools?

This naturally encodes the `plan → verify` reasoning structure that the V4 micro_controller enforces symbolically. V5 makes this structure *differentiable* and *learnable*.

A model that correctly plans (Layer 8) but retrieves wrong evidence (Layer 20) is penalized by the trajectory reward. A model that retrieves wrong strategies but luckily finds correct facts still fails to generalize. Only complete trajectories are rewarded.

---

## 7. What Gets Trained

**Decision: LoRA on cross-attention Q, K, V projection layers only + GNN weights.**

```
Base Qwen3-4B:          FROZEN (36 layers, ~4B params)
Cross-attn Q projection: LoRA (r=16, α=32) ← trained
Cross-attn K projection: LoRA (r=16, α=32) ← trained  
Cross-attn V projection: LoRA (r=16, α=32) ← trained
GNN encoder:             Fully trained (small, ~10-50M params)
Goal encoder (TaskFrame):Fully trained (MLP, ~1M params)
```

Estimated trainable parameters: **~5-20M** out of ~4B total. Approximately 0.25% of the model.

Why this scope:
- The LM's core feed-forward and self-attention layers stay completely frozen → reasoning circuits are preserved
- The LoRA on Q/K/V projections teaches the LM *how to form graph queries* without teaching it *what to think*
- The GNN is fully trained because it needs to learn type-aware structural embeddings from scratch
- Full LoRA across all attention layers risks destroying the base model's reasoning capability and would require massive compute

**The invariant**: After training, if you remove the graph entirely and zero out the cross-attention context vectors, the model should perform identically to the frozen base model on pure reasoning tasks. The graph improves performance; its absence doesn't degrade it.

---

## 8. Training Signal

**Decision: Full trajectory reward — node selection path + final answer correctness.**

A training example is a complete tuple:

```python
trajectory = {
    "question": str,
    "task_frame": TaskFrame,
    "layer8_nodes_selected": List[NodeId],   # which nodes got high attention weight at L8
    "layer20_nodes_selected": List[NodeId],  # which nodes got high attention weight at L20
    "final_answer": str,
    "reward": float  # computed by trajectory_reward()
}
```

### Reward Function

```python
def trajectory_reward(traj, graph) -> float:
    r = 0.0

    # Component 1: Answer correctness (primary signal)
    r += answer_score(traj.final_answer, traj.question)  # 0.0 to 1.0

    # Component 2: Layer 8 node quality
    # Did the model attend to strategy nodes matching its task_family?
    for nid in traj.layer8_nodes_selected:
        node = graph.nodes[nid]
        if node.type == "strategy" and node.task_family == traj.task_frame.task_family:
            r += 0.15  # bonus
        elif node.type == "failure_pattern":
            r += 0.10  # attending to anti-patterns is also valuable

    # Component 3: Layer 20 node quality
    # Did the model attend to facts/claims that actually support the answer?
    for nid in traj.layer20_nodes_selected:
        node = graph.nodes[nid]
        if node.type in ("fact", "claim") and is_supporting_evidence(node, traj.final_answer):
            r += 0.20

    # Component 4: Penalty for attending to irrelevant or contradicting nodes
    for nid in traj.layer8_nodes_selected + traj.layer20_nodes_selected:
        node = graph.nodes[nid]
        if node.type == "fact" and contradicts(node, traj.final_answer):
            r -= 0.30

    return max(0.0, min(1.0, r))
```

### Training method: Contrastive + Offline Supervised

**Offline supervised**: Positive examples from Phase 15 corpus sessions where the model succeeded. The node retrieval paths from those sessions are treated as supervised targets.

**Contrastive**: For each positive trajectory, construct a negative by:
1. Swapping the Layer 8 strategy node for a mismatched strategy (wrong task family)
2. Swapping the Layer 20 fact node for a `FailurePatternNode` from the same domain
3. Verify that the negative trajectory produces a worse answer (or a wrong one)

The contrastive loss pushes Q vectors to separate good trajectories from bad ones in representation space:

```
L_contrastive = max(0, margin - reward(positive) + reward(negative))
L_supervised  = cross_entropy(predicted_answer, correct_answer)
L_total       = α * L_supervised + β * L_contrastive
```

Hyperparameters `α` and `β` are tuned during Phase 16 experiments.

---

## 9. Base Model

**Decision: Qwen3-4B.**

Rationale:
- Already benchmarked extensively in V4 (14-case broad sweep, vacuum/sound paraphrase cases, Dijkstra preconditions, etc.)
- We have a precise understanding of its failure modes and strengths
- 4B parameters is trainable on a single rented A100 (40GB) with LoRA + gradient checkpointing
- Qwen3 family has strong instruction-following needed for the OpenCode protocol
- Upgrading to 7B is possible later if 4B plateaus

Qwen3-4B layer count: **36 layers**, hidden dimension: **2560**.
Injection layers: Layer 8 and Layer 20 (confirmed against this architecture).

---

## 10. Sequencing Plan

```
Phase 15: Corpus Collection (CURRENT)
  │  Collect (question, task_frame, node_path, answer, reward) tuples
  │  Corpus must record: which nodes were retrieved, in which order, with what attention weight
  │  Target: 2,000-10,000 complete trajectories
  │
  ├─ PARALLEL: V5 architecture implementation
  │    Implement: GNN encoder, goal encoder, cross-attention injection adapter
  │    Verify: forward pass works, layer 8/20 hooks fire correctly
  │    No training yet.
  │
  ▼
Phase 16: Cross-Attention Training
  │  Use Phase 15 corpus as training data
  │  Train: GNN + LoRA (Q, K, V at L8 and L20)
  │  Validate: trajectory reward improves vs. frozen baseline
  │
  ▼
Phase 17: Evaluation & Integration
     Replace V4 symbolic retrieval with trained cross-attention module
     Run full benchmark comparison: V4 retrieval vs. V5 cross-attention
     Evaluate: answer quality, node selection precision, trajectory coherence
```

---

## 11. Open Questions (Deferred)

These were explicitly deferred and must be revisited before Phase 16 begins:

1. **GNN architecture**: What message-passing scheme? (GCN, GAT, RGCN for typed edges?) How many layers? What hidden dimension? → Resolve at Phase 16 start.

2. **Attention weight extraction**: How do we record which nodes got high attention weight during inference for corpus collection? Hook-based extraction or explicit logging? → Resolve at Phase 15 start.

3. **Graph size at inference**: The full `MemoryGraph` may have thousands of nodes. Cross-attention over all nodes is O(N²). Do we pre-filter to a candidate set (e.g., top-K from TF-IDF) before cross-attention? → Likely yes. Exact K and filtering strategy TBD.

4. **TaskFrame upgrade threshold**: If Phase 15 corpus analysis reveals TaskFrame signals are too coarse, what specific fields get added to `micro_controller`? → Monitor during Phase 15.

5. **Contrastive negative construction**: Exact algorithm for constructing hard negatives from the existing graph. Do we use `FailurePatternNode`s as ready-made negatives? → Resolve at Phase 16 data pipeline design.

6. **Answer scoring function**: ~~Must decide before training~~ → **RESOLVED**: `reasoning/scoring.py` implements heuristic + embedding + LLM judge fallback. See `answer_score()` and `trajectory_reward()`.

---

## 12. What This Is NOT

To prevent scope creep:

- **Not a separate planner**: There is no new planner module. The cross-attention mechanism IS the learned planning signal.
- **Not dynamic procedure creation**: The LM still cannot write Python code for new `ProcedureNode`s. It creates `StrategyNode`s (composition recipes) only.
- **Not full fine-tuning**: The base LM is frozen except for LoRA on the two cross-attention layers. We are not training a new model.
- **Not replacing the graph**: The `MemoryGraph`, `post_processing`, `micro_controller`, and all V4 infrastructure remain intact. V5 adds a differentiable retrieval layer on top.

---

## 13. Meta-Reasoning Control Layer (V5 Addition)

> Proposed and approved 2026-05-29. Implemented in `reasoning/schemas.py` and `reasoning/graph_relations.py`.

The graph previously stored *what is known*. This section adds *how confidently it is known*, *what invalidates it*, *what it requires*, and *where it transfers*.

### Node Type: `epistemic_state`

Stores the system's belief status about a claim, subgoal, or reasoning path.

```json
{
  "id": "epi_dijkstra_negative_edge_001",
  "type": "epistemic_state",
  "target_node_id": "claim_dijkstra_negative_edge_invalid",
  "status": "verified",
  "confidence": 0.94,
  "support_level": "mechanistic + textbook fact",
  "open_questions": [],
  "known_risks": ["Special DAG shortest path algorithms may confuse the answer"],
  "invalidators": ["Question is about DAG-specific shortest path, not normal Dijkstra"],
  "requires_evidence_before_shortcut": false,
  "last_verified_by": ["fact_dijkstra_nonnegative", "reasoning_greedy_invariant"]
}
```

**Why this is critical**: Without `epistemic_state`, the graph is a bag of confident-looking nodes. The model has no mechanism to distinguish:
- A fact that has been mechanistically verified from a fact that was inferred once in one session
- A shortcut that is always safe from one that fires only in specific conditions
- A solved subgoal that can be reused from one that requires re-verification

**GNN role**:
- Attended to at **Layer 8** when `status=uncertain` or `open_questions` is non-empty → pre-warns the model before it commits to a shortcut
- Attended to at **Layer 20** when `status=verified` or `status=supported` → signals the model can trust the nearby evidence nodes

---

### Edge Type: `invalidated_by`

Connects a claim/strategy/shortcut to a condition that makes it unsafe.

```
claim: "Use Dijkstra for shortest path"
  ──invalidated_by──>
condition: "Graph has negative edges (not just negative cycles)"
```

**Semantics**: "This claim is UNSAFE to use when the condition described by the destination node is true."

**GNN role**: During the Layer 20 evidence pass, if the model attends to a shortcut node AND that node has an active `invalidated_by` edge whose destination matches the current question context, the trajectory reward penalizes finalization without verification.

**Enforcement**: `SolvedSubgoalNode` already has `valid_when` and `invalid_when` as text lists. `invalidated_by` edges make these first-class graph structure so the GNN can propagate the invalidation signal.

---

### Edge Type: `requires_slot`

Connects a strategy/procedure/answer to required task-frame slots.

```
strategy_algorithm_applicability
  ──requires_slot──> verdict
  ──requires_slot──> reason
  ──requires_slot──> alternative
  ──requires_slot──> caveat
```

**Semantics**: "This strategy cannot produce a complete answer unless the named slot is filled."

**Why graph structure instead of prompt instructions**: The GNN can propagate "missing slot" signals directly from the graph. When the model attends to a strategy node at Layer 8, the GNN also surfaces the `requires_slot` targets, telling the model what it must find before it can finalize — without relying on the system prompt to enumerate slot requirements.

---

### Edge Type: `transfers_to`

Stores analogical transfer between reasoning structures and domains.

```
reasoning_atom: "monotonic invariant allows binary search"
  ──transfers_to──>
application: "parametric search"

strategy: "rank via cumulative frequency"
  ──transfers_to──>
application: "Fenwick tree leaderboard design"
```

**Semantics**: "The reasoning structure of the source applies analogically to the problem described by the destination."

**GNN role**: During the Layer 8 planning pass, if the model attends to a strategy node, the GNN also propagates through `transfers_to` edges to surface analogical candidates as secondary planning anchors. This enables cross-domain reasoning without the model needing to explicitly recognize the analogy from text alone.

---

### Full example with epistemic control flow

**Question**: "Can I use Dijkstra if there are negative edges but no negative cycle?"

**Graph activates**:
```
fact: Dijkstra requires non-negative weights
failure_pattern: 'no negative cycle means Dijkstra works' (known misconception)
reasoning_chain: negative edge → breaks greedy finalization → wrong answer
epistemic_state: status=verified, confidence=0.94
  ──invalidated_by──> condition: 'question is about DAG shortest path, not general Dijkstra'
strategy_algorithm_applicability
  ──requires_slot──> verdict
  ──requires_slot──> reason
  ──requires_slot──> alternative
  ──requires_slot──> caveat
```

**Model concludes**:
```
verdict   = no
reason    = greedy invariant breaks on negative edge
alternative = Bellman-Ford / DAG shortest path if applicable
caveat    = no negative cycle is sufficient for Bellman-Ford, not Dijkstra
confidence  = high (epistemic_state verified)
```

This is better reasoning, not just better retrieval.

---

### Registry

All valid edge relation types are defined in [`reasoning/graph_relations.py`](file:///E:/PROJECT/graph_v5/reasoning/graph_relations.py) with:
- Full semantic documentation per relation
- Numeric GNN type IDs (`RELATION_TYPE_ID` dict) — stable, append-only
- Groupings: `PLANNING_PASS_RELATIONS`, `EVIDENCE_PASS_RELATIONS`, `NEGATIVE_RELATIONS`, `POSITIVE_RELATIONS`
- Helper functions: `relation_type_id()`, `is_negative()`, `is_positive()`

---

*Document authored: 2026-05-29. Section 13 added 2026-05-29. Revision required before Phase 16 begins.*
