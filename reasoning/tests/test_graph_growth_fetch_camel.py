from __future__ import annotations

from v5.graph_grower.fetch_camel import row_to_doc, _canon


def _row(message_1="What is the orbital velocity?", message_2=None, sub_topic="Orbits"):
    return {
        "role_1": "Physicist_RoleType.ASSISTANT",
        "topic;": "Mechanics",
        "sub_topic": sub_topic,
        "message_1": message_1,
        "message_2": message_2 if message_2 is not None else ("v" * 300),
    }


def test_maps_domain_mode_and_combines_text():
    doc = row_to_doc(_row(message_1="Q", message_2="A" * 300), 3, domain="physics")
    assert doc is not None
    assert doc["id"] == "camel_physics_000003"
    assert doc["mode"] == "cot"
    assert doc["domain"] == "physics"
    assert "Q" in doc["text"] and "A" in doc["text"]
    assert doc["meta"]["sub_topic"] == "Orbits"


def test_keyword_filter_requires_match():
    r = _row(message_1="Compute the orbital velocity", message_2="apply Newton " * 50)
    assert row_to_doc(r, 0, domain="physics", keywords=["orbital"]) is not None
    assert row_to_doc(r, 0, domain="physics", keywords=["photosynthesis"]) is None


def test_drops_short_solution():
    assert row_to_doc(_row(message_2="too short"), 0, domain="physics") is None


def test_canon_normalizes_science_labels():
    assert _canon("chemistry") == "chem"
    assert _canon("biology") == "bio"
    assert _canon("physics") == "physics"
