from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.unified_store import UnifiedStore


def make_store() -> UnifiedStore:
    return UnifiedStore(client_dir=Path("/tmp"), collection_name="test", embed_model="test-model")


def test_fetch_chunks_deduplicates_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    store = make_store()
    mock_collection = MagicMock()
    mock_collection.get.return_value = {
        "ids": ["chunk-1", "chunk-2"],
        "metadatas": [
            {"chunk_id": "chunk-1", "title": "One"},
            {"chunk_id": "chunk-2", "title": "Two"},
        ],
        "documents": ["Doc one", "Doc two"],
    }
    monkeypatch.setattr(store, "_collection", lambda: mock_collection)

    result: Dict[str, Dict[str, Any]] = store.fetch_chunks(["chunk-1", "chunk-2", "chunk-1"])

    assert list(result.keys()) == ["chunk-1", "chunk-2"]
    assert result["chunk-1"]["document"] == "Doc one"
    assert result["chunk-2"]["meta"]["title"] == "Two"
    mock_collection.get.assert_called_once_with(ids=["chunk-1", "chunk-2"], include=['metadatas', 'documents'])


def test_fetch_chunks_handles_missing_records(monkeypatch: pytest.MonkeyPatch) -> None:
    store = make_store()
    mock_collection = MagicMock()
    mock_collection.get.return_value = {
        "ids": ["chunk-1"],
        "metadatas": [{"chunk_id": "chunk-1"}],
        "documents": ["Doc one"],
    }
    monkeypatch.setattr(store, "_collection", lambda: mock_collection)

    result = store.fetch_chunks(["chunk-1", "chunk-missing"], include_docs=True)

    assert "chunk-1" in result
    assert "chunk-missing" not in result
