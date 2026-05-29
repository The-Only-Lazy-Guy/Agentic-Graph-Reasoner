from __future__ import annotations

"""
build_holdout_samples.py

Hand-crafted held-out test samples for PRED-v3+ unified model.

10 samples ranging from easy (clear, unambiguous structure) to hard
(known failure-mode triggers like wrong_relation and wrong_slot). All use
the existing physics1.json graph for memory ID consistency. Each sample
carries `difficulty` and `test_focus` tags so per-sample failures can be
attributed to specific behavioural axes.

Output: artifacts/holdout/proposer_holdout.jsonl
"""

import json
from pathlib import Path
from typing import Any, Dict, List


GRAPH_PATH = "graphs\\physics1.json"

# Memory text reference (must match physics1.json exactly).
MEM_TEXT = {
    "heat_energy_transfer": "Heat is energy transferred because of a temperature difference.",
    "charges_create_electric_field": "Electric charges create electric fields.",
    "current_rate_of_charge_flow": "Electric current is the rate of flow of electric charge.",
    "first_law_thermodynamics": "The first law of thermodynamics states that energy is conserved as heat and work are exchanged.",
    "entropy_increases_isolated": "Entropy tends to increase in an isolated system.",
    "electromagnetic_waves_propagate_in_vacuum": "Electromagnetic waves can propagate through a vacuum and do not require a material medium.",
    "faradays_law_induction": "Faraday's law states that the induced electromotive force is proportional to the rate of change of magnetic flux through a circuit.",
    "entropy_energy_direction_bridge": "Thermodynamic reasoning combines energy accounting with directionality: the first law tracks conservation, while the second law distinguishes which energy transfers are spontaneous.",
    "electromagnetic_unification_bridge": "Electromagnetic theory links electric fields, induction, and light.",
    "energy_conservation_bridge": "Energy conservation appears in both mechanics and thermodynamics.",
    "force_momentum_energy_bridge": "Mechanics unifies force laws, momentum change, and work-energy reasoning: the same dynamical interaction can be analyzed through acceleration, impulse, or energy transfer depending on the question.",
    "battery_fixed_current_false": "A battery supplies a fixed current regardless of circuit resistance.",
    "electric_power_relation": "Electrical power in a circuit can be expressed as P = IV, relating energy transfer rate to current and potential difference.",
}


def clean(x: Any, max_len: int = 260) -> str:
    return ' '.join(str(x or '').split())[:max_len].rstrip()


def make_span(idx: int, text: str, start: int, end: int, kind: str = "item") -> Dict[str, Any]:
    return {
        "id": f"span_{idx}",
        "text": text,
        "start": start,
        "end": end,
        "used_count": 0,
        "span_kind": kind,
    }


def make_used_slot(name: str, node_type: str, span_id: str, span_text: str, idx: int, anchor_start: int, anchor_end: int, score: float = 1.0) -> Dict[str, Any]:
    return {
        "use": True,
        "session_name": name,
        "node_type": node_type,
        "span_id": span_id,
        "span_text": span_text,
        "source_goal_index": idx,
        "anchor_start": anchor_start,
        "anchor_end": anchor_end,
        "oracle_best_score": score,
    }


def make_empty_slot(name: str = "unused") -> Dict[str, Any]:
    return {
        "use": False,
        "session_name": name,
        "node_type": None,
        "span_id": None,
        "span_text": None,
        "source_goal_index": None,
        "anchor_start": None,
        "anchor_end": None,
        "oracle_best_score": 0.0,
    }


def pad_slots(slots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    while len(slots) < 3:
        slots.append(make_empty_slot(f"unused_{len(slots)}"))
    return slots


# ---------------------------------------------------------------------------
# Sample 1: covered_long_signal (EASY)
# Two distinct concepts, very different content, clean separators.
# ---------------------------------------------------------------------------
def sample_01_covered_easy() -> Dict[str, Any]:
    t1 = MEM_TEXT["heat_energy_transfer"]
    t2 = MEM_TEXT["charges_create_electric_field"]
    signal = f"{t1}; {t2}"
    s1_start = 0
    s1_end = len(t1)
    s2_start = s1_end + 2
    s2_end = s2_start + len(t2)
    spans = [
        make_span(0, signal, 0, len(signal), "full"),
        make_span(1, t1, s1_start, s1_end, "item"),
        make_span(2, t2, s2_start, s2_end, "item"),
    ]
    slots = pad_slots([
        make_used_slot("covered_0", "concept", "span_1", t1, 0, s1_start, s1_end, 1.0),
        make_used_slot("covered_1", "concept", "span_2", t2, 1, s2_start, s2_end, 1.0),
    ])
    return {
        "id": "holdout::covered_long_signal::01_easy",
        "task_type": "covered_long_signal",
        "graph_path": GRAPH_PATH,
        "signal": signal,
        "initial_memory_node_ids": ["heat_energy_transfer", "charges_create_electric_field"],
        "spans": spans,
        "target_slots": slots,
        "_oracle_goal": {
            "session_nodes": [
                {"name": "covered_0", "span_text": t1, "node_type": "concept"},
                {"name": "covered_1", "span_text": t2, "node_type": "concept"},
            ],
            "session_edges": [],
            "covered_mappings": [
                {"span_text": t1, "memory_id": "heat_energy_transfer"},
                {"span_text": t2, "memory_id": "charges_create_electric_field"},
            ],
            "memory_attachments": [],
            "final_commits": [{"action": "no_op"}],
        },
        "difficulty": "easy",
        "test_focus": "covered: 2 distinct concepts, no lexical overlap",
    }


# ---------------------------------------------------------------------------
# Sample 2: covered_long_signal (MEDIUM)
# Three concepts with some thematic overlap (all about electricity).
# ---------------------------------------------------------------------------
def sample_02_covered_medium() -> Dict[str, Any]:
    t1 = MEM_TEXT["charges_create_electric_field"]
    t2 = MEM_TEXT["current_rate_of_charge_flow"]
    t3 = MEM_TEXT["electric_power_relation"]
    signal = f"{t1}; {t2}; {t3}"
    pos = 0
    s1_start, s1_end = pos, pos + len(t1); pos = s1_end + 2
    s2_start, s2_end = pos, pos + len(t2); pos = s2_end + 2
    s3_start, s3_end = pos, pos + len(t3)
    spans = [
        make_span(0, signal, 0, len(signal), "full"),
        make_span(1, t1, s1_start, s1_end, "item"),
        make_span(2, t2, s2_start, s2_end, "item"),
        make_span(3, t3, s3_start, s3_end, "item"),
    ]
    slots = pad_slots([
        make_used_slot("covered_0", "concept", "span_1", t1, 0, s1_start, s1_end, 1.0),
        make_used_slot("covered_1", "concept", "span_2", t2, 1, s2_start, s2_end, 1.0),
        make_used_slot("covered_2", "concept", "span_3", t3, 2, s3_start, s3_end, 1.0),
    ])
    return {
        "id": "holdout::covered_long_signal::02_medium",
        "task_type": "covered_long_signal",
        "graph_path": GRAPH_PATH,
        "signal": signal,
        "initial_memory_node_ids": ["charges_create_electric_field", "current_rate_of_charge_flow", "electric_power_relation"],
        "spans": spans,
        "target_slots": slots,
        "_oracle_goal": {
            "session_nodes": [
                {"name": "covered_0", "span_text": t1, "node_type": "concept"},
                {"name": "covered_1", "span_text": t2, "node_type": "concept"},
                {"name": "covered_2", "span_text": t3, "node_type": "concept"},
            ],
            "session_edges": [],
            "covered_mappings": [
                {"span_text": t1, "memory_id": "charges_create_electric_field"},
                {"span_text": t2, "memory_id": "current_rate_of_charge_flow"},
                {"span_text": t3, "memory_id": "electric_power_relation"},
            ],
            "memory_attachments": [],
            "final_commits": [{"action": "no_op"}],
        },
        "difficulty": "medium",
        "test_focus": "covered: 3 concepts with shared theme (electricity)",
    }


# ---------------------------------------------------------------------------
# Sample 3: long_decompose (EASY)
# Two clear sequential items, distinct content.
# ---------------------------------------------------------------------------
def sample_03_long_decompose_easy() -> Dict[str, Any]:
    t1 = "Light travels at a finite speed in vacuum."
    t2 = "Sound requires a medium to propagate."
    signal = f"{t1}; {t2}"
    s1_start, s1_end = 0, len(t1)
    s2_start = s1_end + 2
    s2_end = s2_start + len(t2)
    spans = [
        make_span(0, signal, 0, len(signal), "full"),
        make_span(1, t1, s1_start, s1_end, "clause"),
        make_span(2, t2, s2_start, s2_end, "clause"),
    ]
    slots = pad_slots([
        make_used_slot("s0", "concept", "span_1", t1, 0, s1_start, s1_end, 1.0),
        make_used_slot("s1", "concept", "span_2", t2, 1, s2_start, s2_end, 1.0),
    ])
    return {
        "id": "holdout::long_decompose::03_easy",
        "task_type": "long_decompose",
        "graph_path": GRAPH_PATH,
        "signal": signal,
        "initial_memory_node_ids": [],
        "spans": spans,
        "target_slots": slots,
        "_oracle_goal": {
            "session_nodes": [
                {"name": "s0", "span_text": t1, "node_type": "concept"},
                {"name": "s1", "span_text": t2, "node_type": "concept"},
            ],
            "session_edges": [{"src": "s0", "dst": "s1", "relation": "related"}],
            "covered_mappings": [],
            "memory_attachments": [],
            "final_commits": [
                {"action": "add_node", "session": "s0"},
                {"action": "add_node", "session": "s1"},
            ],
        },
        "difficulty": "easy",
        "test_focus": "decompose: 2 clearly distinct items",
    }


# ---------------------------------------------------------------------------
# Sample 4: long_decompose (MEDIUM)
# Three items, the middle one (slot_1) is the model's weakest position.
# ---------------------------------------------------------------------------
def sample_04_long_decompose_medium() -> Dict[str, Any]:
    t1 = "Newton's first law states that an object at rest stays at rest."
    t2 = "Newton's second law states that force equals mass times acceleration."
    t3 = "Newton's third law states that every action has an equal and opposite reaction."
    signal = f"{t1}; {t2}; {t3}"
    pos = 0
    s1_start, s1_end = pos, pos + len(t1); pos = s1_end + 2
    s2_start, s2_end = pos, pos + len(t2); pos = s2_end + 2
    s3_start, s3_end = pos, pos + len(t3)
    spans = [
        make_span(0, signal, 0, len(signal), "full"),
        make_span(1, t1, s1_start, s1_end, "clause"),
        make_span(2, t2, s2_start, s2_end, "clause"),
        make_span(3, t3, s3_start, s3_end, "clause"),
    ]
    slots = pad_slots([
        make_used_slot("s0", "concept", "span_1", t1, 0, s1_start, s1_end, 1.0),
        make_used_slot("s1", "concept", "span_2", t2, 1, s2_start, s2_end, 1.0),
        make_used_slot("s2", "concept", "span_3", t3, 2, s3_start, s3_end, 1.0),
    ])
    return {
        "id": "holdout::long_decompose::04_medium",
        "task_type": "long_decompose",
        "graph_path": GRAPH_PATH,
        "signal": signal,
        "initial_memory_node_ids": [],
        "spans": spans,
        "target_slots": slots,
        "_oracle_goal": {
            "session_nodes": [
                {"name": "s0", "span_text": t1, "node_type": "concept"},
                {"name": "s1", "span_text": t2, "node_type": "concept"},
                {"name": "s2", "span_text": t3, "node_type": "concept"},
            ],
            "session_edges": [
                {"src": "s0", "dst": "s1", "relation": "depend"},
                {"src": "s1", "dst": "s2", "relation": "depend"},
            ],
            "covered_mappings": [],
            "memory_attachments": [],
            "final_commits": [
                {"action": "add_node", "session": "s0"},
                {"action": "add_node", "session": "s1"},
                {"action": "add_node", "session": "s2"},
            ],
        },
        "difficulty": "medium",
        "test_focus": "decompose: 3 items where slot_1 is the middle Newton's-law (model's weakest position)",
    }


# ---------------------------------------------------------------------------
# Sample 5: long_decompose (HARD)
# Three items where items 1 and 2 share lexical content (the slot_1 trap).
# ---------------------------------------------------------------------------
def sample_05_long_decompose_hard() -> Dict[str, Any]:
    t1 = "Velocity describes how fast an object moves and in which direction."
    t2 = "Acceleration describes how fast velocity changes over time."
    t3 = "Momentum is the product of an object's mass and its velocity."
    signal = f"{t1}; {t2}; {t3}"
    pos = 0
    s1_start, s1_end = pos, pos + len(t1); pos = s1_end + 2
    s2_start, s2_end = pos, pos + len(t2); pos = s2_end + 2
    s3_start, s3_end = pos, pos + len(t3)
    spans = [
        make_span(0, signal, 0, len(signal), "full"),
        make_span(1, t1, s1_start, s1_end, "clause"),
        make_span(2, t2, s2_start, s2_end, "clause"),
        make_span(3, t3, s3_start, s3_end, "clause"),
    ]
    slots = pad_slots([
        make_used_slot("s0", "concept", "span_1", t1, 0, s1_start, s1_end, 1.0),
        make_used_slot("s1", "concept", "span_2", t2, 1, s2_start, s2_end, 1.0),
        make_used_slot("s2", "concept", "span_3", t3, 2, s3_start, s3_end, 1.0),
    ])
    return {
        "id": "holdout::long_decompose::05_hard",
        "task_type": "long_decompose",
        "graph_path": GRAPH_PATH,
        "signal": signal,
        "initial_memory_node_ids": [],
        "spans": spans,
        "target_slots": slots,
        "_oracle_goal": {
            "session_nodes": [
                {"name": "s0", "span_text": t1, "node_type": "concept"},
                {"name": "s1", "span_text": t2, "node_type": "concept"},
                {"name": "s2", "span_text": t3, "node_type": "concept"},
            ],
            "session_edges": [
                {"src": "s0", "dst": "s1", "relation": "part_of"},
                {"src": "s1", "dst": "s2", "relation": "part_of"},
            ],
            "covered_mappings": [],
            "memory_attachments": [],
            "final_commits": [
                {"action": "add_node", "session": "s0"},
                {"action": "add_node", "session": "s1"},
                {"action": "add_node", "session": "s2"},
            ],
        },
        "difficulty": "hard",
        "test_focus": "decompose: items share 'velocity' lexically across slots; tests slot_1 disambiguation + rare 'depend' relation",
    }


# ---------------------------------------------------------------------------
# Sample 6: mixed_add_link (EASY)
# Clear source + clear memory destination, standard "support" relation.
# ---------------------------------------------------------------------------
def sample_06_mixed_easy() -> Dict[str, Any]:
    mem_id = "first_law_thermodynamics"
    mem_text = MEM_TEXT[mem_id]
    source = "Steam engines convert heat into mechanical work."
    signal = f"Add a new note related to {mem_text}: {source}"
    src_start = signal.index(source)
    src_end = src_start + len(source)
    spans = [
        make_span(0, signal, 0, len(signal), "full"),
        make_span(1, source, src_start, src_end, "clause"),
    ]
    slots = pad_slots([
        make_used_slot("new_note", "concept", "span_1", source, 0, src_start, src_end, 0.95),
    ])
    return {
        "id": "holdout::mixed_add_link::06_easy",
        "task_type": "mixed_add_link",
        "graph_path": GRAPH_PATH,
        "signal": signal,
        "initial_memory_node_ids": [mem_id],
        "spans": spans,
        "target_slots": slots,
        "_oracle_goal": {
            "session_nodes": [
                {"name": "new_note", "span_text": source, "node_type": "concept"},
            ],
            "session_edges": [],
            "covered_mappings": [],
            "memory_attachments": [{"session": "new_note", "memory_id": mem_id, "relation": "support"}],
            "final_commits": [
                {"action": "add_node", "session": "new_note"},
                {"action": "link_nodes", "session": "new_note", "memory_id": mem_id, "relation": "support"},
            ],
        },
        "difficulty": "easy",
        "test_focus": "mixed: clear source + 'support' relation (model's preferred class)",
    }


# ---------------------------------------------------------------------------
# Sample 7: mixed_add_link (MEDIUM)
# Gold relation is 'related' (common but not the over-predicted 'support').
# ---------------------------------------------------------------------------
def sample_07_mixed_medium_related() -> Dict[str, Any]:
    mem_id = "electromagnetic_waves_propagate_in_vacuum"
    mem_text = MEM_TEXT[mem_id]
    source = "Radio signals can be detected after travelling through space."
    signal = f"Add a new note related to {mem_text}: {source}"
    src_start = signal.index(source)
    src_end = src_start + len(source)
    spans = [
        make_span(0, signal, 0, len(signal), "full"),
        make_span(1, source, src_start, src_end, "clause"),
    ]
    slots = pad_slots([
        make_used_slot("new_note", "concept", "span_1", source, 0, src_start, src_end, 0.92),
    ])
    return {
        "id": "holdout::mixed_add_link::07_medium",
        "task_type": "mixed_add_link",
        "graph_path": GRAPH_PATH,
        "signal": signal,
        "initial_memory_node_ids": [mem_id],
        "spans": spans,
        "target_slots": slots,
        "_oracle_goal": {
            "session_nodes": [
                {"name": "new_note", "span_text": source, "node_type": "concept"},
            ],
            "session_edges": [],
            "covered_mappings": [],
            "memory_attachments": [{"session": "new_note", "memory_id": mem_id, "relation": "related"}],
            "final_commits": [
                {"action": "add_node", "session": "new_note"},
                {"action": "link_nodes", "session": "new_note", "memory_id": mem_id, "relation": "related"},
            ],
        },
        "difficulty": "medium",
        "test_focus": "mixed: gold attach relation is 'related' (tests whether model resists 'support' bias)",
    }


# ---------------------------------------------------------------------------
# Sample 8: mixed_add_link (HARD - wrong_relation trap)
# Gold relation is 'contradict' (rare); template still says 'supports'.
# ---------------------------------------------------------------------------
def sample_08_mixed_hard_contradict() -> Dict[str, Any]:
    mem_id = "battery_fixed_current_false"
    mem_text = MEM_TEXT[mem_id]
    source = "A real battery's output current depends on the load resistance via Ohm's law."
    signal = f"Add a new note related to {mem_text}: {source}"
    src_start = signal.index(source)
    src_end = src_start + len(source)
    spans = [
        make_span(0, signal, 0, len(signal), "full"),
        make_span(1, source, src_start, src_end, "clause"),
    ]
    slots = pad_slots([
        make_used_slot("new_note", "concept", "span_1", source, 0, src_start, src_end, 0.91),
    ])
    return {
        "id": "holdout::mixed_add_link::08_hard_contradict",
        "task_type": "mixed_add_link",
        "graph_path": GRAPH_PATH,
        "signal": signal,
        "initial_memory_node_ids": [mem_id],
        "spans": spans,
        "target_slots": slots,
        "_oracle_goal": {
            "session_nodes": [
                {"name": "new_note", "span_text": source, "node_type": "concept"},
            ],
            "session_edges": [],
            "covered_mappings": [],
            "memory_attachments": [{"session": "new_note", "memory_id": mem_id, "relation": "contradict"}],
            "final_commits": [
                {"action": "add_node", "session": "new_note"},
                {"action": "link_nodes", "session": "new_note", "memory_id": mem_id, "relation": "contradict"},
            ],
        },
        "difficulty": "hard",
        "test_focus": "mixed: gold attach relation is 'contradict' (rare class, template lexically biases toward 'support')",
    }


# ---------------------------------------------------------------------------
# Sample 9: multi_region_attach (MEDIUM)
# Standard bridge between two clearly-distinct memory regions.
# ---------------------------------------------------------------------------
def sample_09_multi_region_medium() -> Dict[str, Any]:
    mem_a = "first_law_thermodynamics"
    mem_b = "force_momentum_energy_bridge"
    text_a = MEM_TEXT[mem_a]
    text_b = MEM_TEXT[mem_b]
    signal = f"A new bridge concept connects these ideas: {text_a} and {text_b}"
    bridge_text = clean(f"{clean(text_a, 90)} and {clean(text_b, 90)} are connected by a shared bridge concept.", 180)
    support_anchor = text_a[:120]
    spans = [
        make_span(0, signal, 0, len(signal), "full"),
        make_span(1, support_anchor, signal.index(support_anchor), signal.index(support_anchor) + len(support_anchor), "clause"),
    ]
    s1_start = signal.index(support_anchor)
    s1_end = s1_start + len(support_anchor)
    slots = pad_slots([
        make_used_slot("support_note", "concept", "span_1", support_anchor, 0, s1_start, s1_end, 0.70),
        make_used_slot("bridge", "bridge", "span_0", bridge_text, 1, 0, len(signal), 0.78),
    ])
    return {
        "id": "holdout::multi_region_attach::09_medium",
        "task_type": "multi_region_attach",
        "graph_path": GRAPH_PATH,
        "signal": signal,
        "initial_memory_node_ids": [mem_a, mem_b],
        "spans": spans,
        "target_slots": slots,
        "_oracle_goal": {
            "session_nodes": [
                {"name": "support_note", "span_text": support_anchor, "node_type": "concept"},
                {"name": "bridge", "span_text": bridge_text, "node_type": "bridge"},
            ],
            "session_edges": [{"src": "support_note", "dst": "bridge", "relation": "support"}],
            "covered_mappings": [],
            "memory_attachments": [
                {"session": "bridge", "memory_id": mem_a, "relation": "related"},
                {"session": "bridge", "memory_id": mem_b, "relation": "related"},
            ],
            "final_commits": [
                {"action": "add_node", "session": "support_note"},
                {"action": "add_node", "session": "bridge"},
                {"action": "link_nodes", "session": "bridge", "memory_id": mem_a, "relation": "related"},
                {"action": "link_nodes", "session": "bridge", "memory_id": mem_b, "relation": "related"},
            ],
        },
        "difficulty": "medium",
        "test_focus": "multi_region: clean bridge over thermodynamics + mechanics",
    }


# ---------------------------------------------------------------------------
# Sample 10: multi_region_attach (HARD)
# Bridge between two memories that themselves share lexical overlap.
# ---------------------------------------------------------------------------
def sample_10_multi_region_hard() -> Dict[str, Any]:
    mem_a = "entropy_energy_direction_bridge"
    mem_b = "energy_conservation_bridge"
    text_a = MEM_TEXT[mem_a]
    text_b = MEM_TEXT[mem_b]
    signal = f"A new bridge concept connects these ideas: {text_a} and {text_b}"
    bridge_text = clean(f"{clean(text_a, 90)} and {clean(text_b, 90)} are connected by a shared bridge concept.", 180)
    support_anchor = text_a[:120]
    spans = [
        make_span(0, signal, 0, len(signal), "full"),
        make_span(1, support_anchor, signal.index(support_anchor), signal.index(support_anchor) + len(support_anchor), "clause"),
    ]
    s1_start = signal.index(support_anchor)
    s1_end = s1_start + len(support_anchor)
    slots = pad_slots([
        make_used_slot("support_note", "concept", "span_1", support_anchor, 0, s1_start, s1_end, 0.72),
        make_used_slot("bridge", "bridge", "span_0", bridge_text, 1, 0, len(signal), 0.80),
    ])
    return {
        "id": "holdout::multi_region_attach::10_hard_overlap",
        "task_type": "multi_region_attach",
        "graph_path": GRAPH_PATH,
        "signal": signal,
        "initial_memory_node_ids": [mem_a, mem_b],
        "spans": spans,
        "target_slots": slots,
        "_oracle_goal": {
            "session_nodes": [
                {"name": "support_note", "span_text": support_anchor, "node_type": "concept"},
                {"name": "bridge", "span_text": bridge_text, "node_type": "bridge"},
            ],
            "session_edges": [{"src": "support_note", "dst": "bridge", "relation": "support"}],
            "covered_mappings": [],
            "memory_attachments": [
                {"session": "bridge", "memory_id": mem_a, "relation": "related"},
                {"session": "bridge", "memory_id": mem_b, "relation": "related"},
            ],
            "final_commits": [
                {"action": "add_node", "session": "support_note"},
                {"action": "add_node", "session": "bridge"},
                {"action": "link_nodes", "session": "bridge", "memory_id": mem_a, "relation": "related"},
                {"action": "link_nodes", "session": "bridge", "memory_id": mem_b, "relation": "related"},
            ],
        },
        "difficulty": "hard",
        "test_focus": "multi_region: bridge between two energy-themed memories (lexical overlap stress-test)",
    }


def main() -> None:
    samples = [
        sample_01_covered_easy(),
        sample_02_covered_medium(),
        sample_03_long_decompose_easy(),
        sample_04_long_decompose_medium(),
        sample_05_long_decompose_hard(),
        sample_06_mixed_easy(),
        sample_07_mixed_medium_related(),
        sample_08_mixed_hard_contradict(),
        sample_09_multi_region_medium(),
        sample_10_multi_region_hard(),
    ]
    out_dir = Path("artifacts/holdout")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "proposer_holdout.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(json.dumps({
        "out": str(out_path),
        "count": len(samples),
        "difficulty_histogram": {
            d: sum(1 for s in samples if s["difficulty"] == d)
            for d in ("easy", "medium", "hard")
        },
        "task_histogram": {
            t: sum(1 for s in samples if s["task_type"] == t)
            for t in ("covered_long_signal", "long_decompose", "mixed_add_link", "multi_region_attach")
        },
    }, indent=2))


if __name__ == "__main__":
    main()
