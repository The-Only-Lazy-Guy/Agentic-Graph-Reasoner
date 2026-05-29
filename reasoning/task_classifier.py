"""Task complexity classifier — routes questions through the right v4 pipeline config.

Option C from CLASSIFIER_DESIGN.md: embed the question with the same embedder
used for anchor retrieval, run a 2-layer MLP on top to predict
trivial/moderate/complex. Sub-millisecond inference, no LLM needed.

Three modes:
  1. **Trained MLP** — loaded from a checkpoint. Best quality once trained.
  2. **Rule-based bootstrap** — deterministic heuristics. Good enough to
     start collecting labeled data. Used as fallback when no checkpoint exists.
  3. **Always-complex** — safe default when classifier confidence is low.

Pipeline configs per level control which v4 phases fire, keeping easy tasks
fast and hard tasks thorough.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_TORCH_MODULE = None
_NN_MODULE = None
_TORCH_IMPORT_ATTEMPTED = False


def _load_torch_modules() -> Tuple[Optional[Any], Optional[Any]]:
    """Import torch lazily so rule-based callers never pay the startup cost."""
    global _TORCH_MODULE, _NN_MODULE, _TORCH_IMPORT_ATTEMPTED
    if _TORCH_IMPORT_ATTEMPTED:
        return _TORCH_MODULE, _NN_MODULE
    _TORCH_IMPORT_ATTEMPTED = True
    try:
        import torch
        import torch.nn as nn
    except Exception:
        _TORCH_MODULE = None
        _NN_MODULE = None
    else:
        _TORCH_MODULE = torch
        _NN_MODULE = nn
    return _TORCH_MODULE, _NN_MODULE


# ---------------------------------------------------------------------------
# Pipeline config — what answer_query_v4 receives
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Which v4 phases to enable for this question."""
    enable_activation: bool = True
    enable_coverage_check: bool = True     # not a direct v4 param; signals caller intent
    enable_hypothesis_gate: bool = True    # same
    enable_plan_tree: bool = False
    enable_procedures: bool = False
    polish_answer: bool = True
    run_reflection_inline: bool = False
    collect_corpus: bool = True
    apply_graph_edits: bool = False
    max_steps: int = 20

    def to_v4_kwargs(self) -> Dict[str, Any]:
        """Return the subset of fields that map directly to answer_query_v4 params."""
        return {
            "enable_activation": self.enable_activation,
            "enable_plan_tree": self.enable_plan_tree,
            "enable_procedures": self.enable_procedures,
            "polish_answer": self.polish_answer,
            "run_reflection_inline": self.run_reflection_inline,
            "collect_corpus": self.collect_corpus,
            "apply_graph_edits": self.apply_graph_edits,
            "max_steps": self.max_steps,
        }


# Pre-built configs per complexity level
TRIVIAL_CONFIG = PipelineConfig(
    enable_activation=False,
    enable_coverage_check=False,
    enable_hypothesis_gate=False,
    enable_plan_tree=False,
    enable_procedures=False,
    polish_answer=False,
    run_reflection_inline=False,
    collect_corpus=True,
    max_steps=12,
)

MODERATE_CONFIG = PipelineConfig(
    enable_activation=True,
    enable_coverage_check=True,
    enable_hypothesis_gate=True,
    enable_plan_tree=False,
    enable_procedures=False,
    polish_answer=True,
    run_reflection_inline=False,
    collect_corpus=True,
    max_steps=18,
)

COMPLEX_CONFIG = PipelineConfig(
    enable_activation=True,
    enable_coverage_check=True,
    enable_hypothesis_gate=True,
    enable_plan_tree=True,
    enable_procedures=True,
    polish_answer=True,
    run_reflection_inline=True,
    collect_corpus=True,
    apply_graph_edits=False,
    max_steps=30,
)

LEVEL_TO_CONFIG = {
    "trivial": TRIVIAL_CONFIG,
    "moderate": MODERATE_CONFIG,
    "complex": COMPLEX_CONFIG,
}


# ---------------------------------------------------------------------------
# Complexity levels
# ---------------------------------------------------------------------------

LEVELS = ("trivial", "moderate", "complex")
LEVEL_TO_IDX = {lv: i for i, lv in enumerate(LEVELS)}
IDX_TO_LEVEL = {i: lv for lv, i in LEVEL_TO_IDX.items()}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

_SUB_QUESTION_RE = re.compile(r"\(\d+\)|\d+\.\s")
_DESIGN_KW = re.compile(r"\b(design|architect|system|distribut|service|scalab|deploy)\b", re.I)
_CODE_KW = re.compile(r"\b(implement|code|template|function|class|algorithm|write\s+a)\b", re.I)
_CONSTRAINT_KW = re.compile(r"\b(must|require|at\s+least|constraint|within\s+\d|concurrent)\b", re.I)


def extract_text_features(question: str) -> Dict[str, float]:
    """Cheap text features computed before any graph retrieval."""
    q = question or ""
    tokens = q.split()
    return {
        "length_chars": float(len(q)),
        "length_tokens": float(len(tokens)),
        "num_sub_questions": float(len(_SUB_QUESTION_RE.findall(q))),
        "has_design_request": float(bool(_DESIGN_KW.search(q))),
        "has_code_request": float(bool(_CODE_KW.search(q))),
        "has_multi_constraint": float(bool(_CONSTRAINT_KW.search(q))),
        "question_mark_count": float(q.count("?")),
        "newline_count": float(q.count("\n")),
    }


# ---------------------------------------------------------------------------
# Rule-based bootstrap (Option A fallback)
# ---------------------------------------------------------------------------

def classify_rules(question: str) -> Tuple[str, float]:
    """Heuristic complexity classification. Returns (level, confidence).

    Good enough to bootstrap training data. NOT the target for deployment.
    """
    feats = extract_text_features(question)
    score = 0.0
    # Long questions with sub-parts are likely complex.
    score += min(feats["num_sub_questions"] * 0.3, 0.9)
    # Design/architecture keywords.
    score += feats["has_design_request"] * 0.3
    # Multiple constraints.
    score += feats["has_multi_constraint"] * 0.25
    # Code requests → moderate.
    score += feats["has_code_request"] * 0.15
    # Short, single-question → trivial.
    if feats["length_tokens"] < 25 and feats["num_sub_questions"] <= 1:
        score -= 0.4

    if score >= 0.6:
        return "complex", min(score, 1.0)
    if score >= 0.2:
        return "moderate", min(1.0, 0.5 + score * 0.3)
    return "trivial", min(1.0, max(0.5, 1.0 - score))


# ---------------------------------------------------------------------------
# MLP model (Option C)
# ---------------------------------------------------------------------------

EMBED_DIM = 384  # matches the project's embedder output
HIDDEN_DIM = 64
NUM_CLASSES = len(LEVELS)


def _build_mlp_model(nn_module: Any) -> Any:
    """2-layer MLP: embed(384) -> hidden(64) -> classes(3)."""
    return nn_module.Sequential(
        nn_module.Linear(EMBED_DIM, HIDDEN_DIM),
        nn_module.ReLU(),
        nn_module.Dropout(0.1),
        nn_module.Linear(HIDDEN_DIM, NUM_CLASSES),
    )


# ---------------------------------------------------------------------------
# TaskClassifier — unified interface
# ---------------------------------------------------------------------------

DEFAULT_CHECKPOINT_PATH = Path("cache/task_classifier/classifier.pt")


class TaskClassifier:
    """Unified classifier with trained-MLP primary + rule-based fallback.

    Usage:
        classifier = TaskClassifier.load()          # tries checkpoint
        level, conf, config = classifier.classify("What is binary search?")
        pkt = answer_query_v4(**config.to_v4_kwargs(), ...)
    """

    def __init__(
        self,
        model: Optional[Any] = None,           # ComplexityMLP or None
        confidence_threshold: float = 0.5,
    ):
        self.model = model
        self.confidence_threshold = confidence_threshold
        self._embedder_loaded = False
        self._encode_fn = None

    @classmethod
    def load(cls, checkpoint_path: Optional[Path] = None) -> "TaskClassifier":
        """Load a trained checkpoint if available; else rule-based fallback."""
        ckpt = Path(checkpoint_path or DEFAULT_CHECKPOINT_PATH)
        if not ckpt.exists():
            return cls(model=None)
        try:
            torch_mod, nn_mod = _load_torch_modules()
            if torch_mod is None or nn_mod is None:
                return cls(model=None)
            model = _build_mlp_model(nn_mod)
            state = torch_mod.load(str(ckpt), map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()
            return cls(model=model)
        except Exception:
            return cls(model=None)

    def _ensure_embedder(self):
        if not self._embedder_loaded:
            from embedder import encode_one
            self._encode_fn = encode_one
            self._embedder_loaded = True

    def classify(
        self,
        question: str,
        **overrides: Any,
    ) -> Tuple[str, float, PipelineConfig]:
        """Classify a question and return (level, confidence, config).

        If the trained model exists and is confident enough, use it.
        Otherwise fall back to rules. overrides are merged into the
        resulting PipelineConfig (so callers can force specific flags).
        """
        level: str
        confidence: float

        if self.model is not None:
            try:
                level, confidence = self._classify_mlp(question)
            except Exception:
                level, confidence = classify_rules(question)
            else:
                if confidence < self.confidence_threshold:
                    # Low confidence → safe fallback to moderate.
                    level = "moderate"
                    confidence = self.confidence_threshold
        else:
            level, confidence = classify_rules(question)

        config = PipelineConfig(**asdict(LEVEL_TO_CONFIG[level]))
        # Apply any caller overrides.
        for k, v in overrides.items():
            if hasattr(config, k):
                setattr(config, k, v)
        return level, confidence, config

    def _classify_mlp(self, question: str) -> Tuple[str, float]:
        torch_mod, _nn_mod = _load_torch_modules()
        if torch_mod is None:
            raise RuntimeError("torch unavailable for MLP classification")
        self._ensure_embedder()
        emb = self._encode_fn(question)
        x = torch_mod.from_numpy(emb).unsqueeze(0).float()
        with torch_mod.no_grad():
            logits = self.model(x)
            probs = torch_mod.softmax(logits, dim=-1).squeeze(0)
        idx = int(probs.argmax().item())
        conf = float(probs[idx].item())
        return IDX_TO_LEVEL[idx], conf

    def classify_batch(
        self,
        questions: List[str],
    ) -> List[Tuple[str, float]]:
        """Batch classify (MLP path only). Falls back to rules per-item if no model."""
        if self.model is None:
            return [classify_rules(q) for q in questions]
        torch_mod, _nn_mod = _load_torch_modules()
        if torch_mod is None:
            return [classify_rules(q) for q in questions]
        self._ensure_embedder()
        embs = np.stack([self._encode_fn(q) for q in questions])
        x = torch_mod.from_numpy(embs).float()
        with torch_mod.no_grad():
            logits = self.model(x)
            probs = torch_mod.softmax(logits, dim=-1)
        results: List[Tuple[str, float]] = []
        for i in range(len(questions)):
            idx = int(probs[i].argmax().item())
            conf = float(probs[i, idx].item())
            results.append((IDX_TO_LEVEL[idx], conf))
        return results


# ---------------------------------------------------------------------------
# Training data utilities
# ---------------------------------------------------------------------------

def label_from_metrics(metrics: Dict[str, Any]) -> str:
    """Auto-derive a complexity label from observed session metrics.

    This is the labeling function for the training pipeline:
    corpus sessions.jsonl → label_sessions.py → labeled dataset.
    """
    steps = metrics.get("steps", 0)
    tool_calls = metrics.get("tool_call_count", 0)
    # Quality signals from the corpus row
    quality = metrics.get("quality", {}) if isinstance(metrics.get("quality"), dict) else {}
    complexity_proxy = quality.get("complexity_proxy_score", 0)

    if steps <= 4 and tool_calls <= 10 and complexity_proxy == 0:
        return "trivial"
    if steps > 12 or complexity_proxy >= 2:
        return "complex"
    return "moderate"


def build_training_set(
    corpus_path: Path,
) -> List[Dict[str, Any]]:
    """Read sessions.jsonl and produce a labeled training set.

    Returns list of {"question": str, "label": str, "session_id": str}.
    Only includes finalized sessions.
    """
    rows: List[Dict[str, Any]] = []
    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not row.get("quality", {}).get("finalized"):
                continue
            question = row.get("input", {}).get("question", "")
            if not question:
                continue
            label = label_from_metrics({
                "steps": row.get("metrics", {}).get("steps", 0),
                "tool_call_count": row.get("metrics", {}).get("tool_call_count", 0),
                "quality": row.get("quality", {}),
            })
            rows.append({
                "question": question,
                "label": label,
                "session_id": row.get("session_id", ""),
            })
    return rows
