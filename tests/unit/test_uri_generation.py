from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from mcp_server import SmartSearchEngine  # noqa: E402


def make_engine(vault_paths: list[Path]) -> SmartSearchEngine:
    engine = SmartSearchEngine(unified_store=MagicMock(), searcher=MagicMock())
    engine._vault_paths = vault_paths
    engine._default_vault_name = vault_paths[0].name if vault_paths else "Vault"
    return engine


def test_generate_chunk_uri_preserves_nested_structure() -> None:
    vault_root = Path("/data/obsidian")
    engine = make_engine([vault_root])

    metadata: Dict[str, Any] = {
        "path": str(vault_root / "Projects/AI/Notebook.md"),
        "header_text": "Research Notes",
        "title": "Notebook",
    }

    uri = engine.generate_chunk_uri(metadata)

    assert uri.startswith("obsidian://open?")
    assert "vault=obsidian" in uri
    assert "file=Projects/AI/Notebook.md" in uri
    assert uri.endswith("#Research%20Notes")


def test_generate_chunk_uri_handles_windows_path() -> None:
    vault_root = Path("C:/vault")
    engine = make_engine([vault_root])

    metadata = {
        "path": r"C:\vault\Reference\Index.md",
        "header_text": "Jump Table",
    }

    uri = engine.generate_chunk_uri(metadata)

    assert "file=Reference/Index.md" in uri
    assert uri.endswith("#Jump%20Table")


def test_generate_chunk_uri_falls_back_to_note_id() -> None:
    engine = make_engine([Path("/notes")])
    metadata = {
        "path": "",
        "note_id": "Docs/Readme.md",
    }

    uri = engine.generate_chunk_uri(metadata)

    assert "file=Docs/Readme.md" in uri
    assert "#" not in uri


def test_generate_chunk_uri_encodes_complex_anchor() -> None:
    engine = make_engine([Path("/vault")])

    metadata = {
        "path": "/vault/notes/example.md",
        "header_text": "Complex Header #123!",
    }

    uri = engine.generate_chunk_uri(metadata)

    assert uri.endswith("#Complex%20Header%20%23123%21")
