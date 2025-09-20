"""Confidence aggregation utilities for smart search responses."""

from __future__ import annotations

from typing import Iterable, Optional


def clamp(value: float, *, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, value))


def compute_composite_confidence(
    *,
    distances: Iterable[Optional[float]],
    completeness_score: Optional[float] = None,
    answer_confidence: Optional[float] = None,
    citation_count: int = 0,
) -> float:
    """Aggregate multiple signals into a capped composite confidence score."""

    parts = []

    filtered_distances = [d for d in distances if isinstance(d, (int, float))]
    if filtered_distances:
        mean_distance = sum(filtered_distances) / len(filtered_distances)
        retrieval_conf = clamp(1.0 - mean_distance)
        parts.append(retrieval_conf)

    if isinstance(completeness_score, (int, float)) and completeness_score > 0:
        parts.append(clamp(float(completeness_score)))

    if isinstance(answer_confidence, (int, float)) and answer_confidence > 0:
        parts.append(clamp(float(answer_confidence)))

    if not parts:
        confidence = 0.0
    else:
        confidence = clamp(sum(parts) / len(parts))

    if citation_count >= 2:
        confidence = clamp(confidence + 0.05)

    return confidence

