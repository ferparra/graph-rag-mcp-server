"""Utilities for parsing and applying time-based note filters."""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Literal, Optional, cast

from pydantic import BaseModel, Field, ValidationError, model_validator

try:  # Optional dependency used for richer date parsing when available
    from dateutil import parser as dateutil_parser  # type: ignore
except Exception:  # pragma: no cover - fall back when dateutil is unavailable
    dateutil_parser = None  # type: ignore

TimeFilterMode = Literal["created", "modified", "both"]


@dataclass
class TimeRange:
    """Concrete start/end window with target metadata mode."""

    start: Optional[datetime]
    end: Optional[datetime]
    mode: TimeFilterMode


class StructuredTimeFilter(BaseModel):
    """Structured representation of a time window supplied by tools."""

    start: Optional[datetime] = Field(default=None, description="Start of the time window")
    end: Optional[datetime] = Field(default=None, description="End of the time window")
    last: Optional[int] = Field(default=None, description="Look-back quantity (e.g. 7)")
    unit: Optional[Literal["minutes", "hours", "days", "weeks", "months", "years"]] = Field(
        default=None,
        description="Unit for look-back quantity"
    )
    mode: TimeFilterMode = Field(default="modified", description="Which timestamp to evaluate")

    @model_validator(mode="after")
    def validate_configuration(self) -> "StructuredTimeFilter":
        if self.start and self.end and self.start > self.end:
            raise ValueError("start must be before end")
        if self.last is not None and self.last <= 0:
            raise ValueError("last must be a positive integer")
        if self.last is not None and self.unit is None:
            raise ValueError("unit is required when last is provided")
        if self.last is not None and (self.start or self.end):
            raise ValueError("last cannot be combined with start/end")
        return self


def parse_structured_time_filter(
    payload: Optional[Dict[str, Any]],
    *,
    now: Optional[datetime] = None
) -> Optional[TimeRange]:
    """Validate and convert structured payload into a concrete time range."""

    if not payload:
        return None

    try:
        structured = StructuredTimeFilter.model_validate(payload)
    except ValidationError as exc:  # Re-raise with clearer message for callers
        raise ValueError(f"Invalid structured time filter: {exc}") from exc

    return _resolve_structured_time_filter(structured, now=now)


def parse_natural_language_time_range(
    text: Optional[str],
    *,
    now: Optional[datetime] = None
) -> Optional[TimeRange]:
    """Parse a natural language description into a concrete time range."""

    if not text:
        return None

    cleaned = text.strip()
    if not cleaned:
        return None

    lowered = cleaned.lower()
    mode: TimeFilterMode = "modified"

    # Allow the user to scope which timestamp to inspect via leading keyword
    for prefix, candidate_mode in (
        ("created", "created"),
        ("modified", "modified"),
        ("updated", "modified"),
        ("edited", "modified"),
    ):
        if lowered.startswith(prefix):
            mode = candidate_mode  # type: ignore[assignment]
            lowered = lowered[len(prefix):].strip(" ,:-")
            if not lowered:
                # Phrases like "created" alone are ambiguous
                return None
            break

    now_utc = _ensure_utc(now or datetime.now(timezone.utc))

    # Handle explicit range syntax first (between X and Y)
    between_match = re.match(r"^between\s+(.+?)\s+and\s+(.+)$", lowered)
    if between_match:
        start_str, end_str = between_match.groups()
        start_dt = _parse_datetime_string(start_str.strip())
        end_dt = _parse_datetime_string(end_str.strip())
        if start_dt and end_dt:
            return TimeRange(start=_ensure_utc(start_dt), end=_ensure_utc(end_dt), mode=mode)
        return None

    # Since / after X (open-ended)
    if lowered.startswith("since "):
        candidate = _parse_datetime_string(lowered[6:].strip())
        if candidate:
            return TimeRange(start=_ensure_utc(candidate), end=None, mode=mode)
        return None
    if lowered.startswith("after "):
        candidate = _parse_datetime_string(lowered[6:].strip())
        if candidate:
            # Exclusive after -> shift by a microsecond to avoid re-including boundary
            start_dt = _ensure_utc(candidate) + timedelta(microseconds=1)
            return TimeRange(start=start_dt, end=None, mode=mode)
        return None
    if lowered.startswith("before "):
        candidate = _parse_datetime_string(lowered[7:].strip())
        if candidate:
            end_dt = _ensure_utc(candidate) - timedelta(microseconds=1)
            return TimeRange(start=None, end=end_dt, mode=mode)
        return None

    # Simple keywords: today/yesterday/this week/month/year
    if lowered in {"today", "today's"}:
        start_dt = _start_of_day(now_utc)
        return TimeRange(start=start_dt, end=now_utc, mode=mode)
    if lowered in {"yesterday", "yday"}:
        end_dt = _start_of_day(now_utc)
        start_dt = end_dt - timedelta(days=1)
        return TimeRange(start=start_dt, end=end_dt, mode=mode)
    if lowered in {"this week"}:
        start_dt = _start_of_week(now_utc)
        return TimeRange(start=start_dt, end=now_utc, mode=mode)
    if lowered in {"last week"}:
        end_dt = _start_of_week(now_utc)
        start_dt = end_dt - timedelta(weeks=1)
        return TimeRange(start=start_dt, end=end_dt, mode=mode)
    if lowered in {"this month"}:
        start_dt = _start_of_month(now_utc)
        return TimeRange(start=start_dt, end=now_utc, mode=mode)
    if lowered in {"last month"}:
        end_dt = _start_of_month(now_utc)
        start_dt = _shift_months(end_dt, -1)
        return TimeRange(start=start_dt, end=end_dt, mode=mode)
    if lowered in {"this year"}:
        start_dt = _start_of_year(now_utc)
        return TimeRange(start=start_dt, end=now_utc, mode=mode)
    if lowered in {"last year"}:
        end_dt = _start_of_year(now_utc)
        start_dt = end_dt.replace(year=end_dt.year - 1)
        return TimeRange(start=start_dt, end=end_dt, mode=mode)

    # Generic "last/past/in the last" expressions
    range_match = re.match(
        r"^(?:in\s+the\s+|over\s+the\s+|during\s+the\s+|last\s+|past\s+)?"  # qualifiers
        r"(\d+)\s+"
        r"(minute|minutes|hour|hours|day|days|week|weeks|month|months|year|years)"
        r"(?:\s+ago)?$",
        lowered,
    )
    if range_match:
        quantity = int(range_match.group(1))
        unit = range_match.group(2)
        start_dt = _subtract_units(now_utc, quantity, unit)
        return TimeRange(start=start_dt, end=now_utc, mode=mode)

    # Fallback: direct date value (treated as that day)
    candidate = _parse_datetime_string(lowered)
    if candidate:
        start_dt = _start_of_day(_ensure_utc(candidate))
        end_dt = start_dt + timedelta(days=1)
        return TimeRange(start=start_dt, end=end_dt, mode=mode)

    return None


def filter_notes_by_time(notes: Iterable[Dict[str, Any]], time_range: TimeRange) -> List[Dict[str, Any]]:
    """Return notes whose timestamps fall within the requested window."""

    filtered: List[Dict[str, Any]] = []
    for note in notes:
        meta = note.get("meta", {}) or {}
        if _note_matches_time_range(meta, time_range):
            filtered.append(note)
    return filtered


def _resolve_structured_time_filter(structured: StructuredTimeFilter, *, now: Optional[datetime]) -> Optional[TimeRange]:
    if structured.start or structured.end:
        start_dt = _ensure_utc(structured.start) if structured.start else None
        end_dt = _ensure_utc(structured.end) if structured.end else None
        return TimeRange(start=start_dt, end=end_dt, mode=structured.mode)

    if structured.last is not None and structured.unit:
        now_utc = _ensure_utc(now or datetime.now(timezone.utc))
        start_dt = _subtract_units(now_utc, structured.last, structured.unit)
        return TimeRange(start=start_dt, end=now_utc, mode=structured.mode)

    return None


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _start_of_day(dt: datetime) -> datetime:
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def _start_of_week(dt: datetime) -> datetime:
    # ISO week: Monday is weekday 0
    return _start_of_day(dt) - timedelta(days=dt.weekday())


def _start_of_month(dt: datetime) -> datetime:
    return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _start_of_year(dt: datetime) -> datetime:
    return dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)


def _shift_months(dt: datetime, delta_months: int) -> datetime:
    year = dt.year + (dt.month + delta_months - 1) // 12
    month = (dt.month + delta_months - 1) % 12 + 1
    last_day = calendar.monthrange(year, month)[1]
    day = min(dt.day, last_day)
    return dt.replace(year=year, month=month, day=day)


def _subtract_units(dt: datetime, quantity: int, unit: str) -> datetime:
    unit = unit.lower()
    if unit.startswith("minute"):
        delta = timedelta(minutes=quantity)
        return dt - delta
    if unit.startswith("hour"):
        delta = timedelta(hours=quantity)
        return dt - delta
    if unit.startswith("day"):
        delta = timedelta(days=quantity)
        return dt - delta
    if unit.startswith("week"):
        delta = timedelta(weeks=quantity)
        return dt - delta
    if unit.startswith("month"):
        return _shift_months(dt, -quantity)
    if unit.startswith("year"):
        return dt.replace(year=dt.year - quantity)
    # Unknown unit, fallback to days for safety
    return dt - timedelta(days=quantity)


def _parse_datetime_string(value: str) -> Optional[datetime]:
    if not value:
        return None

    cleaned = value.strip()
    if not cleaned:
        return None

    # Prefer ISO formats first
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError:
        pass

    # Common fallback formats
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue

    if dateutil_parser is not None:
        try:
            return cast(datetime, dateutil_parser.parse(cleaned))
        except Exception:
            return None

    return None


def _parse_datetime_value(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OSError, ValueError):
            return None
    if isinstance(value, str):
        parsed = _parse_datetime_string(value)
        if parsed:
            return _ensure_utc(parsed)
    return None


def _note_matches_time_range(meta: Dict[str, Any], time_range: TimeRange) -> bool:
    targets: List[TimeFilterMode]
    if time_range.mode == "both":
        targets = ["created", "modified"]
    else:
        targets = [time_range.mode]

    for target in targets:
        timestamp = _extract_timestamp(meta, target)
        if timestamp is None:
            continue

        ts_utc = _ensure_utc(timestamp)
        if time_range.start and ts_utc < time_range.start:
            continue
        if time_range.end and ts_utc > time_range.end:
            continue
        return True

    return False


def _extract_timestamp(meta: Dict[str, Any], target: TimeFilterMode) -> Optional[datetime]:
    candidate_keys: List[str]
    if target == "created":
        candidate_keys = [
            "created",
            "created_at",
            "createdAt",
            "file.ctime",
            "file.ctime_iso",
        ]
    elif target == "modified":
        candidate_keys = [
            "modified",
            "modified_at",
            "updated",
            "updated_at",
            "file.mtime",
            "file.mtime_iso",
        ]
    else:
        candidate_keys = []

    for key in candidate_keys:
        if key not in meta:
            continue
        parsed = _parse_datetime_value(meta.get(key))
        if parsed:
            return parsed

    return None
