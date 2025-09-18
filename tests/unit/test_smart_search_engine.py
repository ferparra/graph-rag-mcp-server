from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from chunk_models import ChunkHit  # noqa: E402
from mcp_server import SmartSearchEngine  # noqa: E402


def make_engine() -> SmartSearchEngine:
    engine = SmartSearchEngine(unified_store=MagicMock(), searcher=MagicMock())
    engine._vault_paths = [Path("/vault")]  # ensure URI generation works if needed
    engine._default_vault_name = "vault"
    return engine


def test_chunk_hits_from_raw_merges_metadata_and_document() -> None:
    engine = make_engine()
    engine.unified_store.fetch_chunks.return_value = {
        "chunk-1": {
            "id": "chunk-1",
            "meta": {"chunk_id": "chunk-1", "title": "Title from store"},
            "document": "Stored text",
        }
    }

    raw_hits: list[Dict[str, Any]] = [
        {
            "id": "chunk-1",
            "meta": {"note_id": "note-1", "chunk_type": "section"},
            "distance": 0.2,
        }
    ]

    chunk_hits = engine._chunk_hits_from_raw(raw_hits, default_method="vector_search")

    assert len(chunk_hits) == 1
    hit = chunk_hits[0]
    assert hit.chunk_id == "chunk-1"
    assert hit.meta["title"] == "Title from store"
    assert hit.meta["note_id"] == "note-1"
    assert hit.text == "Stored text"
    assert hit.chunk_info["chunk_type"] == "section"
    engine.unified_store.fetch_chunks.assert_called_once_with(["chunk-1"], include_docs=True)


def test_deduplicate_hits_prefers_higher_score_and_merges_relationships() -> None:
    engine = make_engine()

    hit_low = ChunkHit(
        chunk_id="chunk-1",
        text="",
        retrieval_method="vector_search",
        meta={},
        chunk_info={},
        note_info={},
        final_score=0.4,
        relationships=[{"type": "parent"}],
    )
    hit_high = ChunkHit(
        chunk_id="chunk-1",
        text="",
        retrieval_method="graph_neighbor",
        meta={},
        chunk_info={},
        note_info={},
        final_score=0.8,
        relationships=[{"type": "child"}],
    )

    deduped = engine._deduplicate_hits([hit_low, hit_high])

    assert len(deduped) == 1
    combined = deduped[0]
    assert combined.final_score == 0.8
    rel_types = {rel["type"] for rel in combined.relationships}
    assert rel_types == {"parent", "child"}
