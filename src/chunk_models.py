from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


def _parse_delimited(value: Any) -> List[str]:
    """Normalize comma-delimited metadata fields into clean lists."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [segment.strip() for segment in str(value).split(',') if segment.strip()]


def _normalize_header_context(value: Any) -> List[str]:
    """Ensure header context metadata is always a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(entry) for entry in value if str(entry)]
    if isinstance(value, str):
        if not value:
            return []
        # Support either JSON-like strings or delimiter-based strings
        if value.startswith('[') and value.endswith(']'):
            try:
                import json

                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [str(entry) for entry in parsed if str(entry)]
            except Exception:
                return [segment.strip() for segment in value.strip('[]').split(',') if segment.strip()]
        return [segment.strip() for segment in value.split('>') if segment.strip()]
    return [str(value)]


def _normalize_importance(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


class ChunkHit(BaseModel):
    """Normalized representation of a retrieved chunk across all search paths."""

    chunk_id: str
    note_id: Optional[str] = None
    vault: Optional[str] = None
    path: Optional[str] = None
    title: Optional[str] = None
    header_text: Optional[str] = None
    text: str = ""
    distance: Optional[float] = None
    final_score: Optional[float] = None
    retrieval_method: str = "vector_search"
    importance_score: Optional[float] = None
    meta: Dict[str, Any] = Field(default_factory=dict)
    chunk_uri: Optional[str] = None
    chunk_info: Dict[str, Any] = Field(default_factory=dict)
    note_info: Dict[str, Any] = Field(default_factory=dict)
    relationships: List[Dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    @classmethod
    def from_store_row(
        cls,
        *,
        row: Dict[str, Any],
        distance: Optional[float] = None,
        final_score: Optional[float] = None,
        retrieval_method: str = "vector_search",
        relationships: Optional[List[Dict[str, Any]]] = None,
        extra_chunk_info: Optional[Dict[str, Any]] = None,
    ) -> "ChunkHit":
        chunk_id = str(row.get("id") or row.get("chunk_id") or "")
        meta = dict(row.get("meta") or {})
        if not chunk_id:
            chunk_id = str(meta.get("chunk_id") or "")
        document = row.get("document")
        if document is None:
            document = row.get("text")
        text = str(document) if document is not None else ""

        note_id = meta.get("note_id")
        vault = meta.get("vault")
        path = meta.get("path")
        title = meta.get("title")
        header_text = meta.get("header_text")
        importance_score = _normalize_importance(meta.get("importance_score"))

        chunk_info: Dict[str, Any] = {
            "chunk_id": chunk_id,
            "chunk_type": meta.get("chunk_type"),
            "importance_score": importance_score,
            "header_context": _normalize_header_context(meta.get("parent_headers")),
            "retrieval_method": retrieval_method,
        }
        if extra_chunk_info:
            chunk_info.update(extra_chunk_info)

        note_info: Dict[str, Any] = {
            "note_id": note_id,
            "title": title,
            "path": path,
            "vault": vault,
            "tags": _parse_delimited(meta.get("tags")),
            "links_to": _parse_delimited(meta.get("links_to")),
            "contains_links": _parse_delimited(meta.get("contains_links")),
        }

        return cls(
            chunk_id=chunk_id,
            note_id=str(note_id) if note_id else None,
            vault=str(vault) if vault else None,
            path=str(path) if path else None,
            title=str(title) if title else None,
            header_text=str(header_text) if header_text else None,
            text=text,
            distance=float(distance) if isinstance(distance, (int, float)) else None,
            final_score=float(final_score) if isinstance(final_score, (int, float)) else None,
            retrieval_method=retrieval_method,
            importance_score=importance_score,
            meta=meta,
            chunk_info=chunk_info,
            note_info=note_info,
            relationships=relationships or [],
        )


ChunkHitList: type[list[ChunkHit]] = List[ChunkHit]
