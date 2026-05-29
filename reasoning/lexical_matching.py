"""Shared lexical matching helpers.

This module centralizes the deterministic lexical heuristics used by Phase 3C
and 3E. The functions intentionally preserve the old behavior at each call
site; this is a cleanup boundary, not a scoring redesign.
"""
from __future__ import annotations

import math
import re
from typing import Callable, Iterable, Optional, Sequence, Set


DEFAULT_CONTENT_STOPWORDS = frozenset({
    "about", "above", "after", "again", "against", "also", "before", "being",
    "could", "every", "from", "have", "into", "more", "must", "need", "only",
    "should", "than", "that", "the", "their", "then", "there", "this", "with",
    "would", "your", "using", "under", "when", "what", "where", "which",
})


def normalize_text(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def lexical_tokens(
    text: object,
    *,
    min_chars: int = 3,
    stopwords: Iterable[str] = (),
) -> Set[str]:
    stop = set(stopwords)
    return {
        tok.lower()
        for tok in re.findall(r"[A-Za-z0-9_]+", str(text))
        if len(tok) >= min_chars and tok.lower() not in stop
    }


def lexical_overlap(a: object, b: object, *, min_chars: int = 3) -> float:
    ta = lexical_tokens(a, min_chars=min_chars)
    tb = lexical_tokens(b, min_chars=min_chars)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / math.sqrt(len(ta) * len(tb))


def content_tokens(
    text: object,
    *,
    min_chars: int = 4,
    stopwords: Iterable[str] = DEFAULT_CONTENT_STOPWORDS,
) -> Set[str]:
    return lexical_tokens(text, min_chars=min_chars, stopwords=stopwords)


def has_token_overlap(
    text: object,
    question: object,
    *,
    min_hits: int = 1,
    min_chars: int = 4,
    stopwords: Iterable[str] = DEFAULT_CONTENT_STOPWORDS,
) -> bool:
    text_tokens = content_tokens(text, min_chars=min_chars, stopwords=stopwords)
    if not text_tokens:
        return False
    question_tokens = content_tokens(question, min_chars=min_chars, stopwords=stopwords)
    return len(text_tokens & question_tokens) >= min_hits


def constraint_addressed(
    constraint: str,
    haystack: str,
    *,
    task_concept_extractor: Optional[Callable[[str], Optional[str]]] = None,
    min_keyword_chars: int = 5,
    max_keywords: int = 6,
    max_required_hits: int = 2,
) -> bool:
    text = normalize_text(constraint).lower()
    hay = str(haystack or "").lower()
    if task_concept_extractor is not None:
        task_concept = task_concept_extractor(constraint)
        if task_concept:
            return task_concept.lower() in hay
    if "long long" in text or "int64" in text:
        return any(tok in hay for tok in ("long long", "int64", "int64_t"))
    if "same effect" in text and "server" in text:
        return any(tok in hay for tok in ("same effect", "same result", "does not change"))
    if "loss minimum" in text and "convergence" in text:
        return ("minimum" in hay or "loss" in hay) and "convergence" in hay
    if "true/false" in text and "monotone" in text:
        return any(tok in hay for tok in ("true", "false", "always increases", "always decreases"))
    if "shared state" in text or "shared resources" in text:
        return "shared" in hay
    if "sum, prefix, suffix, and best" in text or "sum prefix suffix and best" in text:
        return (
            "segment tree" in hay
            and any(tok in hay for tok in ("prefix", "pref"))
            and any(tok in hay for tok in ("suffix", "suff"))
            and any(tok in hay for tok in ("best", "maximum subarray", "max subarray", "max_sum", "max sum"))
            and any(tok in hay for tok in ("sum", "total sum", "total_sum"))
        )
    if "cross-boundary merge rule" in text or "left.suffix + right.prefix" in text:
        return any(tok in hay for tok in ("left.suffix + right.prefix", "cross-boundary", "cross boundary"))
    if "all-negative" in text or "non-empty" in text:
        return any(tok in hay for tok in ("maximum element", "largest element", "max element", "not 0"))
    if "negative edge" in text or "dijkstra" in text:
        return "negative" in hay and ("dijkstra" in hay or "bellman" in hay or "unsafe" in hay or "invalid" in hay)
    if "edge-active intervals over time" in text or "rollback-capable dsu" in text:
        return (
            _contains_any(hay, "rollback", "undo", "revert")
            and _contains_any(hay, "dsu", "union-find", "union find", "disjoint set")
            and _contains_any(hay, "segment tree over time", "time segment tree", "divide and conquer over time", "time-axis", "offline")
            and _contains_any(hay, "active interval", "time interval", "edge lifetime", "interval")
        )
    if "segment tree beats" in text or "count_max" in text or "second-max" in text or "second max" in text:
        return (
            "segment tree" in hay
            and _contains_any(hay, "second max", "second maximum", "second-largest")
            and _contains_any(hay, "count of max", "max count", "count_max", "cnt max")
            and _contains_any(hay, "sum", "range sum", "node sum")
        )
    if "current maxima" in text and "second max" in text:
        return _contains_any(
            hay,
            "only current maxima change",
            "cap only the maxima",
            "between the current max and second max",
            "second max < x",
        )
    if "durable local payment state machine" in text or "payment intent" in text:
        return (
            _contains_any(hay, "durable", "state machine", "payment intent")
            and _contains_any(hay, "idempotency key", "idempotency")
        )
    if "querying psp state" in text or "psp state" in text:
        return (
            _contains_any(hay, "psp status", "query the psp", "query psp", "external status lookup", "reconcile")
            and _contains_any(hay, "retry", "replay", "dedupe", "deduplication")
        )
    if "backfill historical data" in text and "verify parity" in text:
        return (
            _contains_any(hay, "backfill", "historical copy", "initial copy")
            and _contains_any(hay, "dual write", "cdc", "change data capture", "live tail", "live writes")
            and _contains_any(hay, "verify", "verification", "consistency check", "compare", "checksums", "validate")
            and _contains_any(hay, "cutover", "cut over", "writer switch", "source of truth")
        )
    if "rollback viable" in text or "old-good source of truth" in text:
        return _contains_any(hay, "rollback", "roll back") and _contains_any(hay, "source of truth", "old monolith", "old-good", "authoritative")
    if "single-writer ownership" in text or "partition ownership" in text:
        return (
            _contains_any(hay, "single-writer", "single writer per sku", "serialize writes per sku", "serialized writes", "partition owner", "partitioned ownership")
            and _contains_any(hay, "oversell", "prevent oversell")
        )
    if "reservation lifecycle" in text or "authoritative source of truth" in text:
        has_lifecycle = _contains_any(hay, "reservation", "hold", "reserve", "confirm", "release", "expire", "expiration")
        has_authority = _contains_any(hay, "source of truth", "authoritative", "cache is derived", "not just cache")
        return has_lifecycle and has_authority
    keys = [tok for tok in re.findall(r"[a-z0-9_]+", text) if len(tok) >= min_keyword_chars]
    if not keys:
        return True
    checked = keys[:max_keywords]
    hits = sum(1 for tok in checked if tok in hay)
    return hits >= max(1, min(max_required_hits, len(checked) // 2))


def matches_packet_constraint(
    claim: str,
    hard_constraints: Sequence[str],
    *,
    task_concept_extractor: Optional[Callable[[str], Optional[str]]] = None,
) -> bool:
    claim_text = normalize_text(claim)
    if not claim_text:
        return False
    claim_hay = claim_text.lower()
    return any(
        constraint_addressed(
            claim_text,
            constraint.lower(),
            task_concept_extractor=task_concept_extractor,
        )
        or constraint_addressed(
            constraint,
            claim_hay,
            task_concept_extractor=task_concept_extractor,
        )
        for constraint in hard_constraints
    )


def _contains_any(haystack: str, *terms: str) -> bool:
    return any(term in haystack for term in terms)
