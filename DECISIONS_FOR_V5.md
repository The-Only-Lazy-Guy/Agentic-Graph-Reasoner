Design discussion summary: from slow tool-calling to recurrent graph-aware reasoning

We started from the bottleneck in the current system: the model’s tool use is slow and fragile because it depends on text decoding.

LM decodes tool call
→ formats JSON
→ parser validates
→ graph tool executes
→ result is inserted back
→ LM continues

This is expensive because every reasoning step requires the LLM to generate a correct tool-call format before the graph can be used. The original idea was to add cross-attention between the language model and the graph so the model could access graph knowledge more directly, instead of repeatedly calling retrieval tools.

At first, we considered cross-attention as a replacement for retrieval. But we clarified that the graph is not just passive memory. In the project proposal, the graph already contains typed nodes such as fact, claim, application, strategy, solved subgoal, reasoning atom, failure pattern, control rule, and procedure, and the system is designed around task frames, evidence reading, tool execution, checking, audit trails, and graph edits.  

That changed the architecture direction.

The graph should not be treated as ordinary RAG memory. It should be treated as an object-oriented reasoning runtime.

⸻

Key insight 1: the graph is executable memory, not just retrieved text

We realized that graph nodes can be more than information chunks.

Some nodes behave like:

facts
strategies
procedures
tools
derived reasoning
solved subgoals
failure patterns
control rules
epistemic states

Some nodes can also call or depend on other nodes. Edges are weighted and biased, so the graph has a kind of control flow.

This means a solved_subgoal or derived_reasoning node can skip repeated reasoning, but only if its preconditions match and no invalidator applies.

So the correct model became:

Graph runtime = reasoning/control/execution layer
LLM = language synthesis + raw derivation layer

Not:

LLM absorbs the whole graph and does everything internally

⸻

Key insight 2: cross-attention helps, but only over an active subgraph

We then asked: how does the model know what to look for?

The answer was: cross-attention itself does not search the whole graph well. The model’s hidden states act as queries, while graph node embeddings act as keys and values:

LM hidden state = Q
graph node embeddings = K, V
attention = softmax(QKᵀ)V

But this only works if the graph has already been narrowed down to a relevant active subgraph.

Therefore, the architecture needs:

Question
→ TaskFrame
→ candidate subgraph selection
→ graph/GNN encoding
→ cross-attention over active graph state

Cross-attention is not the search engine. It is the model’s working-memory interface.

⸻

Key insight 3: use a GNN to locate and encode the active subgraph

We then considered using a GNN.

The conclusion was that a GNN is very suitable for locating the active subgraph because the graph is typed, weighted, biased, and relational.

The GNN should not run over the entire graph every step. Instead, the system should first retrieve or activate a candidate region, then run the GNN over that local region.

Suggested flow:

Question + TaskFrame
→ cheap anchor retrieval / graph expansion
→ candidate subgraph
→ heterogeneous GNN
→ node scores, edge scores, action scores, shortcut scores
→ active graph state

The GNN’s job is to answer:

Which nodes are relevant?
Which edges should be followed?
Which strategy nodes apply?
Which failure patterns matter?
Which solved subgoals can shortcut reasoning?
Which epistemic states support or weaken a claim?
Which invalidators are active?

So the GNN became the graph-side reasoning selector, while cross-attention became the LM-side interface to the selected graph state.

⸻

Key insight 4: add epistemic control to make shortcuts safe

We then discussed what new node/edge types would improve general reasoning.

The strongest addition was:

Node:
- epistemic_state
Edges:
- invalidated_by
- requires_slot
- transfers_to

The purpose is to make the graph aware of trust, uncertainty, conditions, and analogy.

A shortcut node should not simply mean:

I have seen this before, so answer immediately.

It should mean:

Use this shortcut only if:
- preconditions match
- invalidators do not fire
- required answer slots are covered
- epistemic confidence is high enough
- dependencies are available

This turns solved subgoals and derived reasoning nodes into safe reusable reasoning capsules instead of dangerous cached answers.

⸻

Key insight 5: one-shot graph attention is too shallow

After looking at the V5 architecture idea, we first saw it as:

GNN encodes graph
→ Layer 8 cross-attention for planning
→ Layer 20 cross-attention for evidence
→ generate answer

This was already better than V4 because it replaced prompt-injected retrieval with learned graph attention.

But then we asked whether one graph query is enough.

The answer was probably no, because reasoning is iterative:

choose strategy
→ detect failure pattern
→ check epistemic state
→ fill slots
→ verify invalidators
→ use evidence
→ answer

So we introduced multiple latent graph-attention loops.

The key distinction was:

Bad:
external graph retrieval once per token
Good:
one candidate subgraph selection
→ multiple internal graph-attention loops

This gives reasoning depth without returning to slow JSON tool calls.

⸻

Final architecture we reached

The final design is:

Question + TaskFrame
        │
        ▼
1× GNN forward pass
        │
        ▼
Fixed graph K,V embeddings
(node embeddings + typed edge information)
        │
        ▼
Layer 8: Planning loop
R_plan = 1–4 iterations
Each iteration:
Q_r = Wq(h_r ‖ goal_vector ‖ slot_state_r)
A_r = attend(Q_r, K, V)
Main attended nodes:
- strategy nodes
- failure pattern nodes
- control rule nodes
- epistemic state nodes
- shortcut / derived reasoning candidates
Updates:
- hidden state
- slot state
- node scores
- shortcut validity
- invalidator flags
Exit when:
- planning slots converge
- attention stabilizes
- shortcut is verified
- no invalidator blocks the chosen path
        │
        ▼
Layer 9–19:
Frozen LM reasoning integration
        │
        ▼
Layer 20: Evidence loop
R_evidence = 1–6 iterations
Each iteration:
Q_r = Wq(h_r ‖ goal_vector ‖ slot_state_r)
A_r = attend(Q_r, K, V)
Main attended nodes:
- fact nodes
- claim nodes
- application nodes
- solved subgoal nodes
- derived reasoning nodes
- epistemic state nodes
Updates:
- evidence coverage
- confidence
- slot completion
- contradiction risk
- invalidator status
Exit when:
- all required slots are filled
- epistemic confidence is high
- no active invalidator remains
- checker risk is low
        │
        ▼
Layer 21–35:
Answer generation from solved reasoning state
        │
        ▼
Checker
        │
        ▼
If checker fails:
fallback to V4 external tool loop

The important upgrade is:

V4:
LLM repeatedly calls graph tools through decoded JSON
V5:
GNN selects/encodes active graph
→ LM performs iterative latent reasoning over graph state
→ explicit tools are only fallback

⸻

Why this design is better

The final architecture improves the system in four ways.

First, it reduces latency because common reasoning no longer needs repeated JSON tool calls.

Second, it preserves multi-step reasoning because the model can loop internally over the active graph state.

Third, it keeps auditability because each loop can log its top attended nodes, slot state, confidence, invalidators, and exit reason.

Fourth, it keeps safety because the symbolic checker and V4 fallback still exist when latent reasoning is insufficient.

So the final design is not “cross-attention replaces tools.”

A more accurate description is:

TaskFrame-conditioned recurrent graph attention over a typed executable memory graph, using a GNN-encoded active subgraph, with epistemic/invalidator-aware slot checking and fallback symbolic tool execution.

That is the core design we arrived at.8
