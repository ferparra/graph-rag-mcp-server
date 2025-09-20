"""Typed response schemas for MCP tools."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

SchemaVersion = Literal["1.0"]


class Citation(BaseModel):
    note_title: str
    chunk_id: str
    supporting_quote: Optional[str] = None
    uri: Optional[str] = None


class Diagnostics(BaseModel):
    retrieval_method: Literal["vector_search", "graph_expansion", "hybrid", "unknown"] = "unknown"
    query_intent: Literal[
        "semantic",
        "graph",
        "categorical",
        "specific",
        "analytical",
        "unknown",
    ] = "unknown"
    intent_confidence: float = 0.0
    distances_mean: Optional[float] = None
    expansion_depth: Optional[int] = None
    used_cache: bool = False
    retries: int = 0
    circuit_breaker_state: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    message: str
    code: Literal[
        "add_entities",
        "narrow_with_tag",
        "increase_k",
        "rephrase_question",
        "open_note",
        "check_index_health",
        "reindex_vault",
    ]


class SmartSearchHit(BaseModel):
    id: str
    text: str
    meta: Dict[str, object]
    chunk_uri: Optional[str] = None


class SmartSearchResponse(BaseModel):
    schema_version: SchemaVersion = "1.0"
    status: Literal["ok", "degraded", "error"] = "ok"
    query: str
    hits: List[SmartSearchHit] = Field(default_factory=list)
    total_results: int = 0
    answer: Optional[str] = None
    citations: List[Citation] = Field(default_factory=list)
    confidence: float = 0.0
    diagnostics: Diagnostics = Field(default_factory=Diagnostics)
    recommendations: List[Recommendation] = Field(default_factory=list)
    explanation: Optional[str] = None
    error: Optional[str] = None

