# Task Classifier Helper — Design Doc

**Status:** design only. Not implemented. See Phase 14-16 in `v4_PROGRESS.md`.

**Problem:** v4 adds 8-10× latency overhead on easy questions (47s vs 5s on
"what is binary search") for zero quality gain. On hard questions the overhead
is justified (+17pp concept coverage). A gating classifier would route easy
questions through a lightweight path and hard questions through the full pipeline.

---

## 1. What it decides

Per incoming question, the classifier outputs a **pipeline config**:

```python
@dataclass
class PipelineConfig:
    enable_activation: bool     # Phase 2: graph task frame
    enable_coverage_check: bool # Phase 3: coverage self-judge
    enable_hypothesis_gate: bool # Phase 4: verify-before-finalize
    enable_plan_tree: bool      # Phase 9: adaptive plan tree
    enable_procedures: bool     # Phase 10: invoke_procedure tool
    polish_answer: bool         # Phase 12: extra LLM polish call
    run_reflection: bool        # Phase 14: post-session reflection
    max_steps_override: int     # cap step budget (e.g., 5 for easy, 25 for hard)
```

Cheaper configs skip expensive phases. The cheapest useful config:
```python
FAST_CONFIG = PipelineConfig(
    enable_activation=False,
    enable_coverage_check=False,
    enable_hypothesis_gate=False,
    enable_plan_tree=False,
    enable_procedures=False,
    polish_answer=False,
    run_reflection=False,
    max_steps_override=5,
)
```

## 2. Input features

The classifier sees the question BEFORE any graph retrieval. Features:

| Feature | Type | Source |
|---|---|---|
| `question_text` | str | user input |
| `question_length` | int | `len(question)` |
| `question_tokens` | int | whitespace token count |
| `num_sub_questions` | int | count of "(1)" / "(2)" style markers |
| `has_code_request` | bool | regex for "implement" / "code" / "template" |
| `has_design_request` | bool | regex for "design" / "architecture" / "system" |
| `has_multi_constraint` | bool | regex for "requirements:" / "must" / "at least" |
| `domain_keywords` | List[str] | top matching domain tags (CS, math, bio, etc.) |
| `estimated_answer_type` | str | "factual" / "design" / "proof" / "implementation" |

Optionally, AFTER anchor retrieval (fast, ~50ms):

| Feature | Type | Source |
|---|---|---|
| `top_anchor_types` | Counter | node_types of top-5 anchors |
| `has_failure_pattern_in_anchors` | bool | any `failure_pattern` in top-k |
| `anchor_mean_similarity` | float | mean cosine sim of top-5 |
| `anchor_type_diversity` | float | unique_types / k |

## 3. Output space

Three complexity levels is enough:

| Level | Config | When |
|---|---|---|
| `trivial` | `FAST_CONFIG` | Pure recall, definition, single-concept |
| `moderate` | Activation + polish on; rest off; max_steps=12 | Multi-concept, comparison, short implementation |
| `complex` | Everything on; max_steps=25 | Multi-part design, novel synthesis, adversarial constraints |

## 4. Model architecture

**Option A — Rule-based (bootstrap):**

```python
def classify_complexity(question: str, anchors: List[str], graph: MemoryGraph) -> str:
    tokens = len(question.split())
    subs = len(re.findall(r'\(\d\)|\d\.', question))
    if tokens < 30 and subs <= 1:
        return "trivial"
    if subs >= 3 or any(w in question.lower() for w in ["design", "architecture", "distributed"]):
        return "complex"
    return "moderate"
```

Good enough to start collecting labeled data. Not the goal; just bootstraps
the training set.

**Option B — Small fine-tuned model (target):**

- **Base:** Qwen3-0.6B or similar small LM
- **Task:** 3-class classification from question text
- **Training data source:** the distillation corpus from Phase 15. For each
  session, derive the label from observed metrics:
  - `trivial`: steps ≤ 4 AND tool_calls ≤ 10 AND no failures AND no hypotheses
  - `moderate`: 5 ≤ steps ≤ 12 OR hypotheses > 0
  - `complex`: steps > 12 OR failures > 0 OR plan_tree used
- **Training format:** `{"text": question, "label": level}`
- **Estimated data:** ~300 labeled sessions to get 80%+ F1. Collect via
  repeated benchmark runs across the full graph + question bank.

**Option C — Embedding-based classifier:**

- Embed question with the same embedder used for anchor retrieval
- Train a 2-layer MLP on top: `embed(question) → [trivial/moderate/complex]`
- Fastest inference (<1ms), smallest model, no LLM needed at classify time

## 5. Integration point

```python
def answer_query_v4(
    *,
    question: str,
    graph: MemoryGraph,
    controller: ...,
    auto_config: bool = True,  # when True, classifier sets the pipeline config
    classifier: Optional[TaskClassifier] = None,
    **manual_overrides,
) -> V4Packet:
    if auto_config and classifier is not None:
        config = classifier.classify(question, graph)
        # manual_overrides still take precedence
        config.update(manual_overrides)
    else:
        config = manual_overrides
    # Pass config fields to the existing enable_* params...
```

The classifier runs BEFORE graph retrieval (Option A/C) or AFTER anchor
retrieval (Option B with anchor features). Either way it adds <50ms.

## 6. Training pipeline

```
Phase 15 corpus (sessions.jsonl)
    → label_sessions.py  (auto-derive labels from metrics)
    → train_classifier.py (fine-tune Qwen3-0.6B or train MLP)
    → classifier.onnx / classifier.pt
    → TaskClassifier.classify(question) in answerer_v4
```

## 7. Evaluation

- **Metric:** weighted F1 across the 3 levels
- **Ablation:** compare v4 with auto_config vs always-complex vs always-trivial
- **Key invariant:** complex questions MUST NOT be downgraded to trivial
  (quality loss is worse than latency gain). False-complex is fine (just slower).
- **Monitoring:** log `(predicted_level, actual_steps, actual_quality)` per
  session; alert if predicted=trivial but actual steps > 10.

## 8. Deferred / future

- **Per-phase gating** — instead of 3 levels, predict a bitmask over
  individual phases. More flexible but harder to label and train.
- **Online learning** — update the classifier from production sessions
  without manual relabeling. Needs a reward signal (answer quality score).
- **Confidence threshold** — if the classifier is uncertain (e.g., softmax
  entropy > 0.8), fall back to `moderate` as a safe default.

---

## 9. Functionalities summary

| Functionality | Input | Output | When |
|---|---|---|---|
| **Classify question complexity** | question text (+ optional anchors) | trivial / moderate / complex | before graph retrieval |
| **Produce pipeline config** | complexity level | PipelineConfig dataclass | before answer_query_v4 main loop |
| **Override manual flags** | PipelineConfig + user kwargs | merged config | at answer_query_v4 entry |
| **Collect training labels** | corpus sessions.jsonl | labeled dataset | offline, from distillation corpus |
| **Train the classifier** | labeled dataset | model checkpoint | offline |
| **Monitor in production** | (predicted, actual metrics) | alerts / drift detection | per-session logging |
