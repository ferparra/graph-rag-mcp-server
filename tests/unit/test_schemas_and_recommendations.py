"""Tests for MCP typed schemas and recommendation logic."""

from __future__ import annotations

from src.recommendations import recommend_for_low_confidence
from src.schemas import Diagnostics, SmartSearchResponse


def test_smart_search_response_defaults():
    response = SmartSearchResponse(query="test")

    assert response.schema_version == "1.0"
    assert response.status == "ok"
    assert response.hits == []
    assert response.diagnostics.retrieval_method == "unknown"


def test_recommendations_low_similarity_and_categorical_intent():
    diagnostics = Diagnostics(
        distances_mean=0.7,
        query_intent="categorical",
        expansion_depth=0,
        circuit_breaker_state="closed",
    )

    recs = recommend_for_low_confidence(diagnostics)
    codes = [r.code for r in recs]

    assert "add_entities" in codes
    assert "narrow_with_tag" in codes
    assert "rephrase_question" in codes


def test_recommendations_includes_health_when_cb_open():
    diagnostics = Diagnostics(circuit_breaker_state="open")

    recs = recommend_for_low_confidence(diagnostics)
    codes = [r.code for r in recs]

    assert "check_index_health" in codes

