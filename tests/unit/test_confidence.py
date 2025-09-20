"""Tests for composite confidence aggregation."""

from src.confidence import compute_composite_confidence


def test_retrieval_confidence_only():
    confidence = compute_composite_confidence(distances=[0.2, 0.4, 0.3], citation_count=0)
    # Mean distance = 0.3 -> confidence = 0.7
    assert round(confidence, 3) == 0.7


def test_combined_signals_average():
    confidence = compute_composite_confidence(
        distances=[0.2, 0.3],
        completeness_score=0.8,
        answer_confidence=0.6,
        citation_count=1,
    )
    # Retrieval=0.75, completeness=0.8, answer=0.6 -> average = 0.7166...
    assert round(confidence, 3) == 0.717


def test_bonus_for_multiple_citations():
    confidence = compute_composite_confidence(
        distances=[0.3, 0.3],
        citation_count=3,
    )
    # Base retrieval = 0.7, +0.05 bonus capped at 0.75
    assert round(confidence, 3) == 0.75


def test_empty_inputs_return_zero():
    assert compute_composite_confidence(distances=[]) == 0.0
