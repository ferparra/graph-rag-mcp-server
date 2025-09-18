from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from dspy_rag import EnhancedVaultSearcher  # noqa: E402


class DummyPrediction:
    def __init__(self) -> None:
        self.answer = "enhanced answer"
        self.context = "context"
        self.query_intent = "semantic"
        self.intent_confidence = 0.9
        self.retrieval_method = "enhanced"
        self.citations = ["chunk-1"]
        self.confidence = 0.8


async def run(coro):
    return await coro


def test_enhanced_vault_searcher_awaits_predictor() -> None:
    searcher = EnhancedVaultSearcher.__new__(EnhancedVaultSearcher)
    searcher.unified_store = MagicMock()
    searcher.search_store = MagicMock()
    searcher.optimization_manager = None
    searcher.adaptive_rag = SimpleNamespace()
    searcher.complex_handler = None
    searcher.legacy_rag = MagicMock()

    async_prediction = DummyPrediction()
    forward = AsyncMock(return_value=async_prediction)
    searcher.adaptive_rag.forward = forward  # type: ignore[attr-defined]

    result = asyncio.run(searcher.ask("What is new?", use_enhanced=True))

    forward.assert_awaited_once()
    assert result["answer"] == "enhanced answer"
    assert result["method"] == "enhanced_adaptive_rag"
