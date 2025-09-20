"""Unit tests for time filtering utilities."""

from datetime import datetime, timedelta, timezone

import pytest

from src.time_filters import (
    filter_notes_by_time,
    parse_natural_language_time_range,
    parse_structured_time_filter,
)


def _anchor() -> datetime:
    return datetime(2024, 5, 22, 15, 30, tzinfo=timezone.utc)


def test_parse_natural_language_last_days():
    anchor = _anchor()
    result = parse_natural_language_time_range("last 3 days", now=anchor)

    assert result is not None
    assert result.mode == "modified"
    assert result.end == anchor
    assert result.start == anchor - timedelta(days=3)


def test_parse_natural_language_created_last_week():
    anchor = _anchor()
    result = parse_natural_language_time_range("created last week", now=anchor)

    assert result is not None
    assert result.mode == "created"

    start_of_this_week = anchor.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=anchor.weekday())
    expected_start = start_of_this_week - timedelta(weeks=1)

    assert result.start == expected_start
    assert result.end == start_of_this_week


def test_parse_structured_time_filter_last_two_weeks():
    anchor = _anchor()
    payload = {"last": 2, "unit": "weeks", "mode": "both"}
    result = parse_structured_time_filter(payload, now=anchor)

    assert result is not None
    assert result.mode == "both"
    assert result.end == anchor
    assert result.start == anchor - timedelta(weeks=2)


def test_filter_notes_by_time_respects_mode():
    anchor = _anchor()
    notes = [
        {
            "id": "recently_modified",
            "meta": {"file.mtime": (anchor - timedelta(hours=2)).timestamp()},
        },
        {
            "id": "older_note",
            "meta": {"file.mtime": (anchor - timedelta(days=10)).timestamp()},
        },
        {
            "id": "recently_created",
            "meta": {
                "file.mtime": (anchor - timedelta(days=10)).timestamp(),
                "file.ctime": (anchor - timedelta(days=1)).timestamp(),
            },
        },
    ]

    time_range = parse_structured_time_filter({"last": 2, "unit": "days", "mode": "created"}, now=anchor)
    assert time_range is not None

    filtered = filter_notes_by_time(notes, time_range)
    filtered_ids = {note["id"] for note in filtered}

    assert filtered_ids == {"recently_created"}


def test_parse_structured_time_filter_validation_errors():
    with pytest.raises(ValueError):
        parse_structured_time_filter({"last": 3}, now=_anchor())

