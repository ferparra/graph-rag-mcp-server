from __future__ import annotations
import os
import sys
import importlib
from pathlib import Path
from typing import Any, Dict, List
import asyncio
import json

# Ensure repo root in sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class AttrDict(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(item) from e
    def __setattr__(self, key, value):
        self[key] = value


def _to_attrdict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return AttrDict({k: _to_attrdict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [ _to_attrdict(v) for v in obj ]
    return obj


def call_tool(tool_obj, /, **kwargs):
    """Invoke a FastMCP FunctionTool (or plain callable) and normalize output.

    For FastMCP tools: await .run with a single dict argument and parse JSON
    into AttrDict or native list/str for convenient access in tests.
    """
    run_attr = getattr(tool_obj, "run", None)
    if callable(run_attr):
        coro_or_res = run_attr(dict(kwargs))
        try:
            # Await if coroutine
            if asyncio.iscoroutine(coro_or_res):
                res = asyncio.run(coro_or_res)
            else:
                res = coro_or_res
        except RuntimeError:
            # If we're already in an event loop, create a new loop
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                res = loop.run_until_complete(coro_or_res)
            finally:
                loop.close()
                asyncio.set_event_loop(None)

        # Parse ToolResult -> content
        contents = getattr(res, "content", None)
        if contents and len(contents) > 0:
            text = getattr(contents[0], "text", None)
            if text is not None:
                # Try JSON first
                try:
                    data = json.loads(text)
                    return _to_attrdict(data)
                except Exception:
                    # Fallback to raw text
                    return text
        return res
    # Plain callable fallback
    return tool_obj(**kwargs)


def reset_app_state_to_test_dir(test_dir: Path):
    """Point settings.vaults to test_dir and reinit global app_state."""
    # Set env so downstream components also see it if they read afresh
    os.environ["OBSIDIAN_RAG_VAULTS"] = str(test_dir)

    from src import config as cfg
    # Override settings in-place
    cfg.settings.vaults = [test_dir]

    # (Re)import mcp_server and reset app_state, ensuring existing stores are closed first
    # If already imported, close open resources to avoid Oxigraph lock contention
    if "src.mcp_server" in sys.modules:
        mcp_server = sys.modules["src.mcp_server"]
        try:
            # Best-effort close of existing RDF graph store
            if hasattr(mcp_server, "app_state") and hasattr(mcp_server.app_state, "graph_store"):
                mcp_server.app_state.graph_store.close()
        except Exception:
            pass
        mcp_server = importlib.reload(mcp_server)
    else:
        import src.mcp_server as mcp_server

    # Now that module-level app_state has reinitialized, return module
    return mcp_server


def reindex_all(mcp_server_module) -> None:
    res = call_tool(mcp_server_module.reindex_vault, target="all", full_reindex=True)
    if not getattr(res, "success", False):
        raise RuntimeError(f"Reindex failed: {res}")


def assert_true(cond: bool, message: str):
    if not cond:
        raise AssertionError(message)


def test_search_notes(mcp, test_dir: Path):
    tests = [
        {
            "query": "earth verification code",
            "expect_contains": "Earth",
        },
        {
            "query": "mars verification code",
            "expect_contains": "Mars",
        },
        {
            "query": "CONST_TOKEN_ALPHA_earth_93e8a4",
            "expect_contains": "Earth",
        },
        {
            "query": "CONST_TOKEN_ALPHA_mars_77c2d1",
            "expect_contains": "Mars",
        },
    ]

    for t in tests:
        res = call_tool(mcp.search_notes, query=t["query"], k=3)
        assert_true(res.total_results > 0, f"No results for query: {t['query']}")
        top_texts = [h.get("meta", {}).get("title", "") for h in res.hits]
        assert_true(
            any(t["expect_contains"].lower() in x.lower() for x in top_texts),
            f"Expected '{t['expect_contains']}' in top hits for query '{t['query']}', got {top_texts}")


def test_graph_neighbors(mcp):
    # Expect Earth and Mars to be neighbors via Link Map and mutual links
    res_earth = call_tool(mcp.graph_neighbors, note_id_or_title="Earth", depth=1)
    res_mars = call_tool(mcp.graph_neighbors, note_id_or_title="Mars", depth=1)

    def names(nodes: List[Dict[str, Any]]):
        return [n.get("title") or n.get("id") for n in nodes]

    n_earth = names(res_earth.nodes)
    n_mars = names(res_mars.nodes)

    assert_true(any("Mars" in (x or "") for x in n_earth), f"Mars not found among Earth neighbors: {n_earth}")
    assert_true(any("Earth" in (x or "") for x in n_mars), f"Earth not found among Mars neighbors: {n_mars}")


def test_get_subgraph(mcp):
    sg = call_tool(mcp.get_subgraph, seed_notes=["Earth"], depth=1)
    assert_true(isinstance(sg.nodes, list), "Subgraph nodes not a list")
    assert_true(len(sg.nodes) >= 2, f"Expected at least 2 nodes in subgraph, got {len(sg.nodes)}")


def test_backlinks_and_tags(mcp, test_dir: Path):
    # Backlinks: both planets are linked from Link Map
    bl_earth = call_tool(mcp.get_backlinks, note_id_or_path="Earth")
    bl_mars = call_tool(mcp.get_backlinks, note_id_or_path="Mars")
    bl_earth_titles = [b.get("title") or b.get("id") or b for b in bl_earth]
    bl_mars_titles = [b.get("title") or b.get("id") or b for b in bl_mars]
    assert_true(any("Link Map" in (x or "") for x in bl_earth_titles), f"Link Map not in Earth backlinks: {bl_earth_titles}")
    assert_true(any("Link Map" in (x or "") for x in bl_mars_titles), f"Link Map not in Mars backlinks: {bl_mars_titles}")

    # Tags: topic/planets should include both Earth and Mars
    notes_with_tag = call_tool(mcp.get_notes_by_tag, tag="topic/planets")
    titles = [n.get("title") or n.get("id") for n in notes_with_tag]
    assert_true(any("Earth" in (t or "") for t in titles), f"Earth not found by tag: {titles}")
    assert_true(any("Mars" in (t or "") for t in titles), f"Mars not found by tag: {titles}")


def test_read_and_properties(mcp, test_dir: Path):
    earth_rel = Path("rag_mcp_server_test_content/planets/Earth.md")
    earth = call_tool(mcp.read_note, note_path=str(earth_rel))
    assert_true("Earth" in earth.title, f"Unexpected title: {earth.title}")
    assert_true("93E8A4" in earth.content, "Expected content not found in Earth note")

    fm = call_tool(mcp.get_note_properties, note_path=str(earth_rel))
    assert_true("tags" in fm, "Expected tags in frontmatter")

    updated = call_tool(mcp.update_note_properties, note_path=str(earth_rel), properties={"test_flag": True}, merge=True)
    assert_true(updated.get("test_flag") is True, f"Frontmatter not updated: {updated}")


def test_create_add_archive(mcp, test_dir: Path):
    created = call_tool(mcp.create_note,
        title="Transient Test Note",
        content="Ephemeral content for CRUD eval. #test/suite",
        folder="scratch",
        tags=["test/suite"],
        enrich=False,
    )
    created_path = Path(created["path"])
    assert_true(created_path.exists(), f"Created note missing: {created_path}")

    # Add content
    call_tool(mcp.add_content_to_note, note_path=str(created_path), content="\nAdded line.")
    readback = call_tool(mcp.read_note, note_path=str(created_path))
    assert_true("Added line." in readback.content, "Content append failed")

    # Archive it
    archived = call_tool(mcp.archive_note, note_path=str(created_path))
    arch_path = Path(archived)
    assert_true(arch_path.exists(), f"Archived note missing: {arch_path}")


def test_list_notes(mcp):
    lst = call_tool(mcp.list_notes, limit=1000)
    assert_true(len(lst) > 0, "list_notes returned empty list")


def main():
    # Ensure corpus baseline exists
    from scripts.build_test_content import write_baseline, BASELINE_DIR
    import tempfile
    import shutil

    write_baseline()

    # Use a temporary working directory copied from baseline for this run only
    with tempfile.TemporaryDirectory(prefix="rag_mcp_eval_", dir=str(ROOT)) as tmpdir:
        work_dir = Path(tmpdir) / "rag_mcp_server_test_content"
        shutil.copytree(BASELINE_DIR, work_dir)

        test_dir = work_dir
        mcp = reset_app_state_to_test_dir(test_dir)
        reindex_all(mcp)

        # Run evals
        test_search_notes(mcp, test_dir)
        test_graph_neighbors(mcp)
        test_get_subgraph(mcp)
        test_backlinks_and_tags(mcp, test_dir)
        test_read_and_properties(mcp, test_dir)
        test_create_add_archive(mcp, test_dir)
        test_list_notes(mcp)

        print("All MCP evals passed ✅")


if __name__ == "__main__":
    main()
