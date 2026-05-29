
For your goal, the answerer should be closer to:

```text
dynamic graph search
+ session graph mutation
+ path reasoning
+ hypothesis refinement
+ local learned edit operators
+ final answer from evidence structure
```

The key idea:

```text
Reasoning = building, testing, and compressing paths in the graph.
```

Not just retrieving nodes.

---

# Better design: Graph Reasoning Session

Instead of a fixed pipeline, make the answerer run a **task session loop**.

```text
Question
  ↓
Create SessionGraph
  ↓
While not resolved:
    choose reasoning frontier
    retrieve / expand / inspect path
    propose local structure with PRED
    update SessionGraph
    score evidence paths
    detect contradiction / missing bridge / uncertainty
    decide next move
  ↓
Build answer from final SessionGraph
  ↓
Optional commit proposal
```

This loop can run for 3 steps or 50 steps depending on the task.

---

# What is a “reasoning path”?

A graph path can represent a chain of reasoning.

Example:

```text
A supports B
B depends_on C
C contradicts D
```

This path tells the answerer:

```text
A indirectly challenges D through B and C.
```

So a path is not just navigation. It is a **structured argument**.

A reasoning path should store:

```text
nodes visited
edges traversed
relation composition
confidence
evidence text
open questions
contradictions found
```

Example:

```json
{
  "path_id": "p17",
  "nodes": ["optimization_claim", "dp_dependency", "counterexample"],
  "edges": [
    ["optimization_claim", "contradicts", "dp_dependency"],
    ["counterexample", "supports", "dp_dependency"]
  ],
  "path_type": "contradiction_evidence",
  "confidence": 0.82,
  "status": "active"
}
```

Then the answerer can reason over paths, not just nodes.

---

# SessionGraph should store reasoning, not only edits

The session graph should contain several types of nodes:

```text
Evidence nodes:
  facts retrieved from main graph

Hypothesis nodes:
  temporary guesses made during reasoning

Question nodes:
  unresolved subquestions

Bridge nodes:
  connections between distant graph regions

Conflict nodes:
  contradictions or tension points

Conclusion nodes:
  partial answers

Edit nodes:
  proposed main-graph updates
```

So the session graph becomes the answerer’s **working mind**.

Example:

```text
User asks:
  "Why does this optimization fail?"

SessionGraph:
  Q0 = user question
  H0 = optimization may remove dependency
  E0 = DP recurrence depends on previous state
  E1 = greedy shortcut ignores future state
  C0 = H0 contradicts E0
  A0 = final explanation candidate
```

Edges:

```text
Q0 seeks A0
H0 explains Q0
E0 supports C0
H0 contradicts E0
C0 supports A0
```

That is much more powerful than a flat retrieved context.

---

# Better Answerer-v1: dynamic but not fully agentic

You said agentic can wait until v2. Good.

So v1 should be **dynamic**, but with controlled policies.

Not:

```text
LLM freely chooses any action
```

Instead:

```text
Algorithmic loop with learned scoring and bounded actions
```

The loop can still be flexible.

## Answerer-v1 loop

```text
1. Initialize session graph with question node.
2. Retrieve initial anchors from main graph.
3. Add anchors into session graph as evidence nodes.
4. Create frontier from:
   - anchor nodes
   - session hypotheses
   - unresolved question nodes
   - high-uncertainty edges
5. Repeatedly expand the best frontier item.
6. Each expansion may:
   - retrieve neighbors
   - run PRED on local region
   - create hypothesis node
   - create bridge node
   - verify edge
   - summarize a cluster
   - mark a path as useful/useless
7. Stop when answer confidence is high or budget is exhausted.
8. Compose answer from best evidence paths.
```

This is not a direct chain. It is a **graph search over reasoning states**.

---

# The important object: Reasoning Frontier

At every step, the answerer keeps a frontier.

Each frontier item is something worth investigating:

```text
frontier item types:
  node_to_expand
  path_to_extend
  hypothesis_to_test
  contradiction_to_resolve
  missing_bridge_to_find
  uncertain_edge_to_verify
  cluster_to_summarize
```

Each item has a score:

```text
frontier_score =
  relevance_to_question
  + path_confidence
  + information_gain
  + contradiction_value
  + novelty
  - repetition_penalty
  - uncertainty_penalty
  - path_length_penalty
```

So the answerer does not blindly expand one hop. It chooses what seems useful.

---

# How paths help reasoning

Paths can support different reasoning modes.

## 1. Support chain

```text
A supports B
B supports C
therefore A indirectly supports C
```

Used for explanation.

## 2. Dependency chain

```text
A depends_on B
B depends_on C
therefore A requires C
```

Used for “why does this need that?”

## 3. Contradiction chain

```text
A supports B
B contradicts C
therefore A is evidence against C
```

Used for debugging false claims.

## 4. Part-whole chain

```text
A part_of B
B part_of C
therefore A is inside C
```

Used for hierarchical understanding.

## 5. Bridge path

```text
A related_to Bridge
Bridge related_to B
```

Used to connect distant concepts.

This is where the graph gives you advantage over plain LLM context.

---

# Relation composition table

The answerer should learn or implement a soft relation algebra.

Example:

```text
support + support       → support
support + contradict    → contradict
contradict + support    → contradict
part_of + part_of       → part_of
depend + depend         → depend
example_of + part_of    → example_of
related + anything      → weak/unknown
```

But do not make this a hard heuristic forever. In v1 it can be a transparent scoring prior. Later a learned model can replace it.

This helps the answerer score paths:

```text
Path:
  optimization removes dependency
  contradicts
  DP requires dependency

Conclusion:
  optimization is invalid
```

---

# Where current PRED fits

Current PRED should be used as a **local graph operator**.

Not:

```text
PRED answers the question
```

But:

```text
PRED looks at a local region and proposes session graph edits
```

For example, at step 7:

```text
Current focus:
  question node
  memory node A
  memory node B
  hypothesis H

Run PRED:
  proposes H contradicts A
  proposes B supports H
  proposes new bridge node
```

Then EXEC applies that into the session graph.

Current PRED is still useful even with weaknesses, because the session loop can treat its outputs as tentative. Your current work already keeps the executor deterministic and avoids hiding predictor failures in executor heuristics, which is exactly the right foundation. 

---

# Graph modification during session

During the task, the answerer should modify the **session graph** continuously.

Actions:

```text
ADD_TEMP_NODE
ADD_TEMP_EDGE
UPDATE_EDGE_CONFIDENCE
MARK_EDGE_TENTATIVE
MARK_EDGE_VERIFIED
MARK_CONTRADICTION
CREATE_BRIDGE
SUMMARIZE_PATH
SUMMARIZE_CLUSTER
ADD_PENDING_COMMIT
REJECT_DRAFT
```

Example:

```text
Step 1:
  retrieve DP dependency nodes

Step 2:
  PRED proposes:
    optimization_claim contradicts dp_dependency

Step 3:
  verifier says confidence medium
  session edge status = tentative

Step 4:
  frontier adds:
    verify contradiction

Step 5:
  retrieval expands to counterexample node

Step 6:
  path found:
    counterexample supports dp_dependency
    optimization_claim contradicts dp_dependency

Step 7:
  contradiction edge becomes verified

Step 8:
  answer says:
    "It fails because it breaks the dependency required by the recurrence..."
```

This is real graph reasoning.

---

# The answerer should reason over paths and subgraphs

Instead of feeding the LLM raw nodes, feed it a **reasoning packet**.

Bad packet:

```text
Here are 20 retrieved nodes. Answer.
```

Good packet:

```text
Question:
  Why does optimization X fail?

Best evidence paths:
  Path 1:
    X removes state dependency
    contradicts
    DP recurrence requires previous state

  Path 2:
    Counterexample supports DP dependency

Open uncertainty:
  Relation between X and greedy shortcut is tentative

Conclusion candidate:
  X fails because it removes a dependency needed for correctness.
```

This makes the LLM’s job easier and more grounded.

---

# Dynamic Answerer-v1 algorithm

Here is the better v1:

```python
def answerer_v1(question, main_graph):
    session = SessionGraph()
    session.add_question(question)

    anchors = retrieve_anchors(question, main_graph, k=8)
    session.import_memory_nodes(anchors)

    frontier = init_frontier(question, anchors, session)

    for step in range(MAX_STEPS):
        item = frontier.pop_best()

        if item.type == "expand_node":
            neighbors = retrieve_neighbors(item.node_id, main_graph)
            session.import_memory_nodes(neighbors)
            frontier.add_from_neighbors(neighbors)

        elif item.type == "extend_path":
            paths = extend_reasoning_path(item.path, main_graph, session)
            session.add_paths(paths)
            frontier.add_from_paths(paths)

        elif item.type == "propose_structure":
            local_view = build_local_view(question, session, item.focus)
            goal = pred.predict(local_view)
            draft = executor.apply_to_session(goal, session)
            session.add_draft(draft)
            frontier.add_from_draft(draft)

        elif item.type == "verify_edge":
            result = verifier.verify(item.edge, session, main_graph)
            session.update_edge_status(item.edge, result)
            frontier.add_from_verification(result)

        elif item.type == "summarize_cluster":
            summary = summarize_cluster(item.cluster, session)
            session.add_temp_node(summary)
            frontier.add_from_summary(summary)

        if answer_ready(session, question):
            break

    packet = build_reasoning_packet(question, session)
    answer = llm_write_answer(packet)
    commit_plan = propose_commits(session)

    return answer, session, commit_plan
```

This loop is controlled, but not direct.

---

# What decides “answer ready”?

Not one fixed step count.

Use conditions:

```text
answer_ready if:
  at least one strong evidence path exists
  no critical contradiction unresolved
  enough relevant nodes have been explored
  answer confidence above threshold
  or budget exhausted
```

For hard tasks, it explores longer.

For easy tasks, it stops early.

---

# Path search strategies

Use multiple path strategies, not only BFS.

## 1. Relevance beam search

Keeps top-k paths related to the question.

## 2. Contradiction search

Looks specifically for paths ending in contradiction/refute/conflict edges.

## 3. Bridge search

Looks for nodes that connect two separated regions.

## 4. Dependency tracing

Follows `depend`, `part_of`, `requires`, `causes`.

## 5. Evidence gathering

Follows support/example/refine edges.

Different questions use different path modes.

Example:

```text
"Why does this fail?"
  prioritize contradiction + dependency paths

"How are A and B related?"
  prioritize bridge paths

"Explain concept X"
  prioritize support + part_of + summary paths

"Is this claim true?"
  prioritize support vs contradiction balance
```

This is how the graph gives task-specific reasoning advantage.

---

# Session graph as short-term memory

The session graph should not only store final edits. It stores thinking history.

```text
SessionGraph:
  question node
  retrieved nodes
  active hypotheses
  candidate paths
  rejected paths
  verified edges
  tentative edges
  summaries
  final answer node
```

That lets the answerer remember what it already tried.

Example:

```text
Do not retrieve the same region again.
Do not repeat rejected path.
Use verified contradiction in final answer.
Use tentative edge only as weak evidence.
```

This is much better than an LLM context window alone.

---

# PRED with context

You asked earlier about adding context as parameter.

In this dynamic design, context is essential.

PRED should receive:

```text
local signal
candidate spans
retrieved memory nodes
session graph local neighborhood
active reasoning path
question embedding
current mode
```

So instead of:

```text
PRED(signal, spans, memory)
```

you eventually want:

```text
PRED(signal, spans, memory, session_context, active_path, mode)
```

For v1, fake this by converting session context into memory-like nodes.

For v2/PRED-v4, add a real context encoder.

---

# Why current PRED failures are okay

Current PRED still has long_decompose problems. The latest verifier work improved false-edge suppression but did not solve strict long_decompose. That means PRED is not reliable enough to be the whole answerer. But it can still be a draft proposer inside a larger reasoning loop.

Use statuses:

```text
PRED edge:
  draft

Verifier agrees:
  verified

Path evidence supports it:
  promoted

Contradiction found:
  rejected or marked conflict
```

So the answerer can recover from imperfect PRED.

---

# The true advantage of graph reasoning

The answerer should use the graph for these things:

```text
1. Finding distant evidence through paths
2. Testing hypotheses against existing memory
3. Building temporary reasoning chains
4. Detecting contradictions
5. Creating bridge concepts
6. Summarizing subgraphs into reusable abstractions
7. Updating long-term memory after the task
```

That is much stronger than top-k retrieval.

---

# Final improved design

The answerer should be:

```text
A dynamic graph reasoning system.
It does not follow a fixed chain.
It repeatedly chooses a frontier item,
expands or edits the session graph,
scores reasoning paths,
and stops only when the session graph contains enough verified structure to answer.
```

The central object is not the retrieved context.

The central object is:

```text
SessionGraph + active reasoning paths
```

That is where complex thinking happens.

So the better v1 design is:

```text
Question
  ↓
SessionGraph initialized
  ↓
Dynamic frontier search over main graph + session graph
  ↓
PRED proposes local edits when useful
  ↓
EXEC applies edits to session graph
  ↓
Paths are scored, verified, extended, summarized
  ↓
Answer is written from best evidence paths
  ↓
Only safe edits are proposed for main graph commit
```

That is the direction I think will actually use your graph to its full advantage.
