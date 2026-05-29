"""ExcessiveSearchDetector — flags long stretches of search_nodes calls
without intervening read_node calls.

Observed failure mode in v4 hard-task traces (2026-05-25): the model issues
many reworded search_nodes calls trying to find a procedure that doesn't
exist in the graph. Even with Jaccard dedupe (Phase 13a) catching exact
semantic duplicates, the model can still drift across slightly-different
queries for ~10+ steps before giving up.

This MP fires on `post_dispatch`. It walks `ctx.tool_call_log` backward
from the end and counts consecutive search_nodes calls. If that run length
hits SEARCH_RUN_THRESHOLD, it emits a WARN signal nudging the model to
either commit to what it has found or record_failure.

once_per_session=False so the warning fires each turn the threshold is
crossed (otherwise the first warning might be missed if the model is
mid-cycle of generating yet another search).
"""
from __future__ import annotations

from typing import List

from reasoning.meta import MetaContext, MetaProcedure
from reasoning.signals import Signal


SEARCH_RUN_THRESHOLD = 5   # 5 consecutive searches with no reads = nudge


def _count_trailing_searches(tool_call_log: List[dict]) -> int:
    """Return the number of consecutive search_nodes calls at the end of the log."""
    n = 0
    for entry in reversed(tool_call_log):
        name = entry.get("name")
        if name == "search_nodes":
            n += 1
        elif name in ("read_node", "expand_neighbors"):
            # A read or expand breaks the search-only streak.
            break
        else:
            # Other tool calls (mark_done, create_object, etc.) don't reset
            # but don't count toward the streak either.
            continue
    return n


def _detect_excessive_search(ctx: MetaContext) -> List[Signal]:
    run_len = _count_trailing_searches(ctx.tool_call_log)
    if run_len < SEARCH_RUN_THRESHOLD:
        return []
    # Recent search queries, for the warning message context.
    recent_queries: List[str] = []
    for entry in reversed(ctx.tool_call_log):
        if entry.get("name") != "search_nodes":
            if recent_queries:
                break
            continue
        q = (entry.get("args") or {}).get("query") or ""
        recent_queries.append(q)
        if len(recent_queries) >= 3:
            break
    sample = " | ".join(f"{q!r}" for q in reversed(recent_queries))
    return [Signal(
        id=f"excessive_search_iter_{ctx.current_iteration}_run_{run_len}",
        type="excessive_search",
        severity="warn",
        message=(
            f"You have issued {run_len} consecutive search_nodes calls without "
            f"reading a node in between. Recent queries: {sample}. "
            "If the graph does not contain what you are looking for, switch to "
            "read_node / expand_neighbors on results you already have, or call "
            "record_failure(approach='looked for X', condition='not in graph', "
            "mechanism='no node matches after N attempts') and proceed with the "
            "evidence you have."
        ),
        emitted_at_step=ctx.current_iteration,
        emitted_by="excessive_search_detector",
        metadata={"run_length": run_len, "threshold": SEARCH_RUN_THRESHOLD},
        sticky=True,
        once=False,
    )]


def build_excessive_search_detector() -> MetaProcedure:
    return MetaProcedure(
        id="meta_excessive_search_detector",
        name="ExcessiveSearchDetector",
        purpose=(
            f"Flag {SEARCH_RUN_THRESHOLD}+ consecutive search_nodes calls "
            f"with no intervening read_node / expand_neighbors call."
        ),
        fires_on="post_dispatch",
        predicate=_detect_excessive_search,
        once_per_session=False,
        priority=40,
    )
