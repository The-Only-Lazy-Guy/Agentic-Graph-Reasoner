Below is a design proposal that fits your current NGR direction and directly extends the old route-walk/planner idea into a stronger **graph-native traversal + evidence compiler + answer agent**.

---

# Design Proposal: Evidence-Collecting Graph Traversal for Editing and Answering

## 1. Purpose

The goal is to upgrade the current Neural Graph Reasoner from only choosing graph edit actions into a system that can also **collect structured evidence during traversal**, use that evidence to make safer graph updates, and reuse the same collected evidence to answer user queries.

This proposal extends the current NGR roadmap:

```text
NGR-v1 = multi-step graph edit policy
NGR-v2 = graph policy + text writer
NGR-v3 = graph-grounded answer/reasoning agent
```

Your project already defines NGR-v3 as a graph-grounded answering agent with modes like `EDIT_GRAPH`, `ANSWER_PROBE`, `VERIFY_ANSWER`, and `ANSWER_USER`, where the answer loop routes through graph regions, expands frontiers, inspects paths, builds temporary session state, drafts an answer, verifies support, and then answers. 

This proposal makes that idea more concrete.

---

# 2. Main Thesis

The new design is better than the old LM-planner route-walk design because it separates three things:

```text
1. Graph policy:
   decides where to traverse, what evidence matters, and what structure/action to use.

2. Evidence compiler:
   converts visited nodes, edges, paths, conflicts, and summaries into an LLM-readable packet.

3. LLM/text writer:
   writes natural language only after the graph policy has selected structure and evidence.
```

This matches your current rule that the LM should not be the planner. In the current task list, the graph policy owns action IDs, node pointers, region pointers, relation classes, and stop/continue decisions, while the system serializes JSON. The LM is allowed only for writing node/update/conflict/summary text, not choosing tool sequence, targets, or relation labels. 

So the new design keeps the good part of the old planner — flexible reasoning over graph paths — but removes the fragile part — letting the LM hallucinate actions, IDs, or relations.

---

# 3. Old Design vs New Design

## Old design: LM planner / hybrid route-walk

The old design was roughly:

```text
signal/query
→ retrieve candidate nodes
→ LM planner decides tool/action JSON
→ repair/fallback sometimes fixes bad actions
→ editor applies graph mutation
```

Strengths:

```text
flexible
easy to prototype
LLM can explain why it chose actions
good for early exploration
```

Weaknesses:

```text
can hallucinate node IDs
can output invalid JSON
hard to train with RL
fallbacks can accidentally become training labels
slow and VRAM-heavy
weak action/target validity guarantees
```

Your uploaded task list explicitly says the project has moved away from training a tiny LM to generate planner JSON and is now focused on a graph-native policy that reasons over graph state and chooses discrete tool/action/target/relation decisions. 

## New design: Evidence-Collecting Neural Graph Reasoner

The new design is:

```text
query/signal
→ anchor retrieval
→ graph-native traversal
→ collect evidence nodes/edges/paths
→ compile evidence packet
→ graph policy decides edit or answer structure
→ LLM writes text only
→ validator checks grounding
→ commit or answer
```

Strengths:

```text
no hallucinated IDs because targets are pointers
better schema validity
better RL training target
evidence paths are reusable for both editing and answering
LLM gets structured context instead of raw graph dump
safer long-term memory mutation
```

This is directly aligned with the expected win of GraphPolicyNet over the old LM planner: better schema validity, target validity, speed, RL trainability, and hallucinated-ID avoidance. 

---

# 4. Core New Component: Evidence-Collecting Traversal

## 4.1 Current traversal state

The environment should already track state fields like:

```text
signal
step
budget_left
active_regions
anchor_nodes
visited_nodes
frontier_nodes
candidate_nodes
candidate_edges
candidate_paths
candidate_edits
tool_history
last_action
done
```

This matches the planned `GraphPolicyEnv` state fields in your task list. 

The proposal is to make these fields more useful by treating traversal as **evidence collection**, not only candidate expansion.

---

## 4.2 Traversal actions

Use these primitive actions:

```text
route_regions
expand_frontier
inspect_path
find_conflicts
propose_edit
stop
```

These are already listed as the supported primitive actions for the graph policy environment. 

But we add one conceptual rule:

> Every traversal action must produce structured evidence artifacts.

For example:

```text
expand_frontier
→ adds candidate nodes and edges

inspect_path
→ adds evidence paths and relation-chain interpretations

find_conflicts
→ adds contradiction paths and uncertainty notes

propose_edit
→ uses collected evidence to create a shadow edit
```

---

# 5. Evidence Objects

During traversal, collect four object types.

## 5.1 Evidence nodes

```json
{
  "id": "fenwick_prefix_query",
  "text": "Fenwick tree supports prefix sum query in O(log n).",
  "node_type": "concept",
  "confidence": 0.94,
  "importance": 0.87,
  "retrieval_score": 0.82,
  "visited_step": 2,
  "role": "supporting_evidence"
}
```

## 5.2 Evidence edges

```json
{
  "src": "fenwick_prefix_query",
  "dst": "fenwick_range_sum",
  "relation": "supports",
  "strength": 0.91,
  "confidence": 0.88,
  "role": "support_path_edge"
}
```

## 5.3 Evidence paths

```json
{
  "path_id": "P1",
  "nodes": [
    "fenwick_core_operations",
    "fenwick_prefix_query",
    "fenwick_range_sum"
  ],
  "relations": ["supports", "supports"],
  "score": 0.89,
  "interpretation": "Fenwick core operations support prefix queries, and prefix queries support the range-sum formula."
}
```

## 5.4 Conflict records

```json
{
  "conflict_id": "C1",
  "node_a": "fenwick_direct_range_storage",
  "node_b": "fenwick_range_sum",
  "relation": "contradicts",
  "severity": 0.81,
  "interpretation": "The direct-storage claim conflicts with the prefix-difference range-sum explanation."
}
```

The important idea: the graph policy does not only collect nodes. It collects **paths with meaning**.

---

# 6. New Component: Graph Context Compiler

Add a file:

```text
graph_context_compiler.py
```

Its job:

```text
TraversalResult
→ ranked paths
→ conflict notes
→ evidence summaries
→ answer/edit policy
→ LLM-readable EvidencePacket
```

This is the missing bridge between graph traversal and deep LLM understanding.

The compiler should not dump raw graph JSON. It should create a compact packet like:

```text
QUERY:
How does Fenwick tree answer range sum?

ANCHORS:
[B] Range sum [l,r] = prefix(r) - prefix(l-1).
confidence: 0.96

SUPPORTING PATHS:
P1: D --supports--> A --supports--> B
Interpretation:
Fenwick tree's core operation is prefix sum. Since range sum can be reduced to two prefix sums, B is strongly supported.

CONFLICTS:
C1: C contradicts B.
Interpretation:
C claims Fenwick tree directly stores all range sums, which is likely misleading.

ANSWER POLICY:
Use B as the main answer.
Mention C only as a correction if useful.
```

This is much better for the LLM than:

```json
{"nodes": [...], "edges": [...]}
```

because the LLM sees the **reasoning skeleton**, not just disconnected facts.

---

# 7. Proposed Architecture

```text
User signal/query
   ↓
Anchor selection
   ↓
Evidence-collecting graph traversal
   ↓
Temporary session graph
   ↓
Graph Context Compiler
   ↓
Mode decision:
      EDIT_GRAPH / ANSWER_USER / VERIFY_ANSWER
   ↓
Graph policy chooses structure
   ↓
LLM writes text only
   ↓
Grounding validator
   ↓
Answer or commit proposal
```

For editing:

```text
signal
→ traverse affected region
→ collect evidence paths
→ propose shadow edit
→ score consistency/QA/retrieval improvement
→ commit only if safe
```

For answering:

```text
question
→ traverse relevant region
→ collect evidence paths
→ compile answer context
→ draft grounded answer
→ verify answer support
→ answer without mutating long-term graph
```

This matches the existing NGR-v3 rule that answer mode may create temporary claims, calculations, hypotheses, evidence paths, and answer plans, but must not directly mutate long-term memory. Permanent graph edits require a separate validator and commit gate. 

---

# 8. Is This Better Than the Old Design?

Yes — but with one condition.

It is better **if the graph policy controls traversal/action/targets**, and the LLM only receives compiled evidence for text generation.

It is not better if you simply traverse more nodes and dump more text into the LLM.

## Why it is better

### 1. Better grounding

Old design:

```text
LLM sees retrieved chunks and guesses relationships.
```

New design:

```text
LLM sees nodes + edges + paths + relation interpretations + conflicts.
```

The LLM no longer has to infer the structure alone.

---

### 2. Better target validity

Old design:

```text
LLM may generate node IDs or invalid targets.
```

New design:

```text
policy selects candidate pointers.
system serializes IDs.
```

This follows your current candidate-limited targeting rule: the model should select candidate indices/pointers and never be trained to generate node IDs. 

---

### 3. Better trainability

Old design:

```text
train LM to emit JSON/tool plans
hard to RL
fragile schema
fallback pollution risk
```

New design:

```text
train graph policy on discrete actions, pointers, relations, stop decisions
then use offline RL / best-of-N / Q-learning
```

Your roadmap already says offline RL should start with simpler methods like pairwise action ranking, Q-learning, actor-critic, and best-of-N trajectory ranking, not PPO/GRPO until the environment is stable. 

---

### 4. Better reuse

The same traversal can support both:

```text
graph editing
question answering
verification
```

This is the main upgrade. Traversal is no longer just “find candidates for edit.” It becomes a reusable reasoning substrate.

---

# 9. Main Risk

The main risk is **over-traversal**.

If every update triggers the whole DAG, you get:

```text
too much context
too much noise
expensive traversal
possible graph corruption
LLM confused by irrelevant evidence
```

So traversal must be:

```text
bounded
relation-aware
priority-scored
evidence-gated
budget-limited
```

Recommended defaults:

```text
max_depth: 3
max_visited_nodes: 64
max_paths: 12
max_conflicts: 5
min_path_score: 0.20
top_k_frontier_per_step: 8
```

---

# 10. Relation Composition Rules

The context compiler should translate relation chains into meaning.

Example rules:

```text
supports + supports
→ indirect support

supports + part_of
→ supports part of broader concept

contradicts + supports
→ weakens downstream claim

refines + supports
→ supports a more precise version

example_of + part_of
→ example illustrates one subpart

part_of + summary
→ parent summary may need refresh
```

This is where your graph becomes stronger than normal RAG.

Normal RAG says:

```text
Here are top-k chunks.
```

Your system says:

```text
Here are the evidence paths, what they mean, what conflicts exist, and which claims should be trusted.
```

---

# 11. Proposed Data Schema

## 11.1 TraversalResult

```json
{
  "query": "...",
  "mode": "ANSWER_USER",
  "anchors": [],
  "visited_nodes": [],
  "visited_edges": [],
  "evidence_paths": [],
  "conflicts": [],
  "candidate_edits": [],
  "tool_history": [],
  "budget_used": 0
}
```

## 11.2 EvidencePacket

```json
{
  "query": "...",
  "task_type": "answer_user",
  "anchor_nodes": [],
  "supporting_paths": [],
  "conflicting_paths": [],
  "key_claims": [],
  "uncertainty_notes": [],
  "answer_policy": {
    "prefer_nodes": [],
    "avoid_overtrusting": [],
    "must_mention_uncertainty": false
  }
}
```

## 11.3 AnswerPlan

```json
{
  "main_claims": [],
  "supporting_paths_used": [],
  "conflicts_handled": [],
  "missing_information": [],
  "insufficient_evidence": false,
  "final_answer_plan": "..."
}
```

This connects well with your current NGR-v3 answer trajectory schema, which already includes `tool_steps`, `evidence_nodes`, `evidence_paths`, `session_edits`, answer text/confidence, and proposed long-term edits. 

---

# 12. Training Plan

## Stage A — Add evidence collection to NGR-v1a/v1b

Do not jump directly to answer mode.

First, make traversal produce:

```text
evidence_nodes
evidence_edges
evidence_paths
conflict_records
path_scores
```

This respects the current roadmap warning: do not skip stages, do not jump straight to answer mode, do not let the text writer become the planner, and do not reintroduce fallback winners as labels. 

---

## Stage B — Train path usefulness

Create labels from corruption tasks:

```text
masked edge recovery → path that connects src/dst is useful
false claim → contradiction path is useful
duplicate signal → coverage path is useful
summarize_cluster → dense local cluster paths are useful
```

Train heads:

```text
path_usefulness_head
conflict_relevance_head
evidence_node_role_head
stop_after_evidence_head
```

---

## Stage C — Build Graph Context Compiler

Implement deterministic compiler first.

No neural model needed at first.

Inputs:

```text
TraversalResult
```

Outputs:

```text
EvidencePacket as text + JSON
```

---

## Stage D — Add answer mode

Create:

```text
graph_answer_env.py
qa_answer_probe_ngraph.py
train_graph_answer_policy.py
eval_graph_answer_policy.py
```

These are already listed as planned NGR-v3 files. 

---

# 13. Evaluation Metrics

For edit mode:

```text
final edit accuracy
commit_f1
session_edge_f1
memory_attachment_f1
no_op_accuracy
invalid_action_rate
unsafe mutation rate
```

For answer mode:

```text
answer correctness
evidence node usage
evidence path correctness
unsupported claim rate
insufficient-evidence honesty
overconfidence penalty
```

These answer rewards match the existing NGR-v3 reward list: answer correctness, expected term recall, evidence node usage, evidence path correctness, insufficient-evidence honesty, unsupported claim penalty, and overconfidence penalty. 

---

# 14. Recommended Version Name

I would call this:

```text
NGR-v3.EC — Evidence-Collecting Graph Reasoner
```

or:

```text
NGR-v3-GCC — Graph Context Compiler
```

My preferred name:

```text
NGR-v3-GCC: Graph-Grounded Answering with Graph Context Compiler
```

Because the key innovation is not only traversal. The key innovation is:

```text
traversal result → structured evidence packet → grounded LLM answer
```

---

# 15. Final Verdict

Yes, this design is better than the old design.

But the reason is specific:

```text
Old design:
LLM planner reasons directly and may hallucinate structure.

Current NGR design:
graph policy chooses structure, but mostly for editing.

Proposed design:
graph policy traverses and collects evidence;
Graph Context Compiler converts paths into meaning;
LLM writes grounded answers or text fields only.
```

This gives you the best of both:

```text
graph-native validity + LLM language ability
```

without letting the LLM become the planner again.

The shortest summary:

```text
Do not make the LLM walk the graph blindly.
Make the graph policy walk the graph, collect evidence paths, compile them into a reasoning packet, and let the LLM verbalize the answer from that packet.
```
