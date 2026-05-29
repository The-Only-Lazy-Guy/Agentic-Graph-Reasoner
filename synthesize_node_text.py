from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from graph_core import MemoryGraph


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def clean(text: str, max_len: int) -> str:
    return normalize_text(text)[:max_len].rstrip()


def _load_graph(graph_cache: Dict[str, MemoryGraph], graph_path: str) -> MemoryGraph:
    if graph_path not in graph_cache:
        graph_cache[graph_path] = MemoryGraph.load_json(str(graph_path))
    return graph_cache[graph_path]


def _memory_text(graph: MemoryGraph, node_id: str, max_len: int) -> str:
    node = graph.nodes.get(str(node_id))
    return clean(str(node.text) if node is not None else "", max_len)


def _best_matching_memory_id(memory_ids: Sequence[str], support_text: str, graph: MemoryGraph) -> str | None:
    support_clean = clean(support_text, 120)
    for mem_id in memory_ids:
        if _memory_text(graph, mem_id, 120) == support_clean:
            return str(mem_id)

    support_tokens = set(normalize_text(support_text).lower().split())
    best_id = None
    best_score = -1
    for mem_id in memory_ids:
        mem_tokens = set(normalize_text(_memory_text(graph, mem_id, 120)).lower().split())
        score = len(support_tokens & mem_tokens)
        if score > best_score:
            best_score = score
            best_id = str(mem_id)
    return best_id


def apply_template_synthesis(
    row: Mapping[str, Any],
    reconciled_slots: Sequence[Mapping[str, Any]],
    *,
    graph_cache: Dict[str, MemoryGraph] | None = None,
) -> list[dict[str, Any]]:
    graph_cache = graph_cache if graph_cache is not None else {}
    task_type = str(row.get("task_type", ""))
    graph = _load_graph(graph_cache, str(row.get("graph_path", "")))
    memory_ids = [str(x) for x in (row.get("initial_memory_node_ids", []) or []) if str(x)]

    synthesized: list[dict[str, Any]] = [dict(slot) for slot in reconciled_slots]
    used_slots = [(i, slot) for i, slot in enumerate(synthesized) if str(slot.get("span_text", "") or "").strip()]

    if task_type == "mixed_add_link":
        if len(used_slots) >= 2 and memory_ids:
            _, source_slot = used_slots[0]
            _, new_slot = used_slots[1]
            source_text = str(source_slot.get("span_text", "") or "")
            dst_text = _memory_text(graph, memory_ids[0], 110)
            new_slot["span_text"] = clean(
                f"{source_text} This supports a new note related to {dst_text}.",
                220,
            )
        return synthesized

    if task_type == "multi_region_attach":
        if len(used_slots) >= 2 and len(memory_ids) >= 2:
            bridge_slot = None
            non_bridge_slots = []
            for _, slot in used_slots:
                if bool(slot.get("is_bridge")) or str(slot.get("node_type", "")) == "bridge":
                    if bridge_slot is None:
                        bridge_slot = slot
                else:
                    non_bridge_slots.append(slot)
            if bridge_slot is None or not non_bridge_slots:
                return synthesized
            support_text = str(non_bridge_slots[0].get("span_text", "") or "")
            support_mem = _best_matching_memory_id(memory_ids, support_text, graph)
            if support_mem is not None:
                other_mem = next((m for m in memory_ids if m != support_mem), None)
                if other_mem is not None:
                    text_a = _memory_text(graph, support_mem, 90)
                    text_b = _memory_text(graph, other_mem, 90)
                    bridge_slot["span_text"] = clean(
                        f"{text_a} and {text_b} are connected by a shared bridge concept.",
                        180,
                    )
        return synthesized

    return synthesized
