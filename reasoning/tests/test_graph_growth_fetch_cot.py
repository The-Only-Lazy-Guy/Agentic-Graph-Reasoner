from __future__ import annotations

from v5.graph_grower.fetch_cot import row_to_doc


def _row(domain="math", problem="P", reasoning=None, solution="S", source="src"):
    return {
        "domain": domain,
        "problem": problem,
        "deepseek_reasoning": reasoning if reasoning is not None else ("r" * 300),
        "deepseek_solution": solution,
        "source": source,
    }


def test_maps_domain_mode_and_combines_text():
    doc = row_to_doc(_row(domain="Code", problem="Q", solution="A"), 7)
    assert doc is not None
    assert doc["id"] == "ot114k_000007"
    assert doc["mode"] == "cot"
    assert doc["domain"] == "algo"            # Code -> algo
    assert "Q" in doc["text"] and "A" in doc["text"]
    assert doc["meta"]["ot_domain"] == "code"


def test_domain_filter_excludes_other_domains():
    assert row_to_doc(_row(domain="science"), 0, ot_domains=["math", "code"]) is None
    assert row_to_doc(_row(domain="math"), 0, ot_domains=["math", "code"]) is not None


def test_keyword_filter_requires_match():
    r = _row(problem="Compute the orbital velocity", reasoning="apply Newton " * 50)
    assert row_to_doc(r, 0, keywords=["orbital"]) is not None
    assert row_to_doc(r, 0, keywords=["photosynthesis"]) is None


def test_drops_short_reasoning():
    assert row_to_doc(_row(reasoning="too short"), 0) is None


def test_science_maps_to_science_label():
    doc = row_to_doc(_row(domain="Science"), 1)
    assert doc["domain"] == "science"
