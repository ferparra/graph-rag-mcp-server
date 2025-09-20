"""Recommendation engine for degraded smart search responses."""

from __future__ import annotations

from typing import List

try:
    from schemas import Diagnostics, Recommendation
except ImportError:  # pragma: no cover - package import
    from .schemas import Diagnostics, Recommendation


def recommend_for_low_confidence(diagnostics: Diagnostics) -> List[Recommendation]:
    """Generate up to three recommendations when status is degraded."""

    recs: List[Recommendation] = []
    added_codes: set[str] = set()

    def _add(rec: Recommendation) -> None:
        if rec.code not in added_codes:
            recs.append(rec)
            added_codes.add(rec.code)

    # Always include rephrase recommendation with high priority
    _add(
        Recommendation(
            message="Rephrase the question with clearer intent or synonyms.",
            code="rephrase_question",
        )
    )

    distances_mean = diagnostics.distances_mean or 1.0

    if distances_mean > 0.55:
        _add(
            Recommendation(
                message="Add one or two concrete entities or proper nouns to make the query more specific.",
                code="add_entities",
            )
        )

    if diagnostics.query_intent in {"categorical", "graph"}:
        _add(
            Recommendation(
                message="Constrain the search with a relevant tag such as #project or #area.",
                code="narrow_with_tag",
            )
        )

    if diagnostics.expansion_depth is not None and diagnostics.expansion_depth < 1:
        _add(
            Recommendation(
                message="Increase the number of results (k) to retrieve more supporting context.",
                code="increase_k",
            )
        )

    if diagnostics.circuit_breaker_state in {"open", "half_open"}:
        _add(
            Recommendation(
                message="Check the index health; the retrieval backend recently failed requests.",
                code="check_index_health",
            )
        )

    # Cap recommendations at three, ensuring the high-priority items stay
    return recs[:3]
