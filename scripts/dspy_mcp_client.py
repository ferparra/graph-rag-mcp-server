#!/usr/bin/env python3
"""
Quick DSPy client for Graph RAG MCP Server.

Runs a smart search against your Obsidian vault via the server's smart_search tool
and prints concise results. Defaults to calling the server tool directly in-process
for simplicity. You can also attempt HTTP MCP mode if you have the HTTP server running.

Usage examples:
  - Direct (in-process):
      uv run scripts/dspy_mcp_client.py \
        --vault /path/to/your/vault \
        --chroma /path/to/.chroma_db \
        --query "What are my health goals?"

  - HTTP MCP (if server running on localhost:8765):
      uv run scripts/dspy_mcp_client.py \
        --mode mcp-http --url http://localhost:8765 \
        --vault /path/to/your/vault \
        --query "What are my health goals?"

Notes:
  - Ensure your vault has been indexed: `uv run scripts/reindex.py all`
  - For HTTP mode, start the server: `uv run graph-rag-mcp-http --port 8765`
  - For stdio mode, you can adapt this script or your MCP client to spawn:
      `uvx --python 3.13 --from . graph-rag-mcp-stdio`
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import asyncio
from pathlib import Path
from typing import Any, Dict, List
import re


def _abs(path: str) -> str:
    return str(Path(path).expanduser().absolute())


def _redact_uri(uri: str) -> str:
    """Redact potentially sensitive query params (e.g., vault name) from Obsidian URIs."""
    try:
        # Replace vault=... value with 'REDACTED' while keeping other params intact
        return re.sub(r"(vault=)[^&#]+", r"\1REDACTED", uri)
    except Exception:
        return uri


def _print_results(payload: Dict[str, Any], redact_sensitive: bool = True) -> None:
    query = payload.get("query", "")
    status = payload.get("status", "unknown")
    strategy = payload.get("strategy_used", "?")
    explanation = payload.get("explanation", "")
    hits = payload.get("hits", []) or []
    total = payload.get("total_results", len(hits))
    confidence = payload.get("confidence")
    diagnostics = payload.get("diagnostics") or {}
    recommendations: List[Dict[str, Any]] = payload.get("recommendations") or []

    print(f"Query: {query}")
    line = f"Status: {status}"
    if isinstance(confidence, (int, float)):
        line += f" | Confidence: {confidence:.2f}"
    print(line)

    intent = diagnostics.get("query_intent", "unknown")
    intent_conf = diagnostics.get("intent_confidence")
    retrieval = diagnostics.get("retrieval_method", strategy)
    retries = diagnostics.get("retries", 0)
    cb_state = diagnostics.get("circuit_breaker_state") or "unknown"

    intent_bits = intent
    if isinstance(intent_conf, (int, float)):
        intent_bits += f" ({intent_conf:.2f})"
    print(f"Intent: {intent_bits} | Method: {retrieval} | Retries: {retries} | CB: {cb_state}")

    if explanation:
        print(f"Explanation: {explanation}")
    print(f"Results: {total}")

    if recommendations:
        print("Recommendations:")
        for rec in recommendations:
            message = rec.get("message", "")
            code = rec.get("code", "")
            print(f"  - {message} [{code}]")
    print("")

    for i, hit in enumerate(hits, start=1):
        # The smart_search tool returns enhanced hits with chunk_uri and info
        title = None
        chunk_uri = hit.get("chunk_uri")
        text = hit.get("text", "").strip()

        # Prefer note_info.title when present (enhanced path)
        note_info = hit.get("note_info") or {}
        if isinstance(note_info, dict):
            title = note_info.get("title") or title

        # Fallback to raw meta title when running via direct/basic path
        if not title:
            meta = hit.get("meta") or {}
            if isinstance(meta, dict):
                title = meta.get("title")

        # Final fallback
        title = title or "Untitled"

        # Snippet
        snippet = (text[:180] + "…") if len(text) > 180 else text

        print(f"[{i}] {title}")
        if chunk_uri:
            safe_uri = _redact_uri(chunk_uri) if redact_sensitive else chunk_uri
            print(f"    URI: {safe_uri}")
        print(f"    Snippet: {snippet}")
        print("")


async def _run_direct(query: str, vault_path: str, k: int) -> Dict[str, Any]:
    """Call the server tool in-process (no MCP transport)."""
    # Set env before importing server/settings so they pick up
    os.environ.setdefault("OBSIDIAN_RAG_VAULTS", vault_path)

    # Optionally point Chroma dir via env (already handled by caller if provided)
    # Keep DSPy optimization optional; disabled by default for speed
    os.environ.setdefault("OBSIDIAN_RAG_DSPY_OPTIMIZE_ENABLED", "false")

    # Reduce noisy logs and external telemetry to avoid leaking paths
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    os.environ.setdefault("CHROMADB_ANONYMIZED_TELEMETRY", "False")
    os.environ.setdefault("POSTHOG_DISABLED", "true")

    # Import server after env is set
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import logging
    import mcp_server as mcp_mod  # type: ignore
    smart_search = mcp_mod.smart_search  # type: ignore
    # Quiet common loggers unless caller asks otherwise (handled in main)
    logging.getLogger("graph_rag_mcp").setLevel(logging.WARNING)
    logging.getLogger("unified_store").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.ERROR)
    logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.ERROR)
    logging.getLogger("cache_manager").setLevel(logging.WARNING)
    logging.getLogger("dspy_optimizer").setLevel(logging.WARNING)
    logging.getLogger("dspy_programs").setLevel(logging.WARNING)
    logging.getLogger("dspy_rag").setLevel(logging.WARNING)

    async def _call(vf: str | None) -> Dict[str, Any]:
        # FastMCP FunctionTool exposes .run(...)
        args: Dict[str, Any] = {"query": query, "k": k}
        if vf:
            args["vault_filter"] = vf
        res = await smart_search.run(args)  # type: ignore[attr-defined]

        # Normalize possible ToolResult into a plain dict
        contents = getattr(res, "content", None)
        if contents and len(contents) > 0:
            text = getattr(contents[0], "text", None)
            if text is not None:
                try:
                    return json.loads(text)
                except Exception:
                    return {"query": query, "strategy_used": "unknown", "hits": [], "total_results": 0, "raw": text}
        if isinstance(res, dict):
            return res
        return json.loads(json.dumps(res, default=str))

    # First try with provided vault filter; if empty, retry without filter
    payload = await _call(vault_path)
    if int(payload.get("total_results", 0)) == 0 and vault_path:
        fallback = await _call(None)
        # Keep explanation to indicate fallback
        fallback.setdefault("explanation", "")
        fallback["explanation"] = (fallback["explanation"] + " | Fallback: no vault match; searched all vaults").strip()
        return fallback
    return payload


async def _run_http(query: str, vault_path: str, k: int, url: str) -> Dict[str, Any]:
    """Attempt to call smart_search via MCP HTTP client.

    Requires `mcp` Python package (declared in project deps). The exact client API
    may vary by version; this tries a common pattern.
    """
    try:
        # Lazy import so direct mode works without mcp installed in other environments
        from mcp.client.session import ClientSession  # type: ignore
        from mcp.transport.http import HTTPConnection  # type: ignore

        async with ClientSession(HTTPConnection(url)) as session:  # type: ignore[arg-type]
            # Ensure the tool exists (optional)
            try:
                tools = await session.list_tools()  # type: ignore[attr-defined]
                names = [getattr(t, "name", None) or t.get("name") for t in tools]
                if "smart_search" not in names:
                    print("Warning: smart_search tool not listed; attempting call anyway.")
            except Exception:
                # Not fatal; proceed to call directly
                pass

            # Call the tool
            async def _call(vf: str | None):
                args = {"query": query, "k": k}
                if vf:
                    args["vault_filter"] = vf
                return await session.call_tool("smart_search", arguments=args)  # type: ignore[attr-defined]

            result = await _call(vault_path)

            # The client may return a typed model or a dict; normalize to dict
            def _normalize(res: Any) -> Dict[str, Any]:
                if hasattr(res, "model_dump"):
                    return res.model_dump()  # type: ignore[no-any-return]
                if isinstance(res, dict):
                    return res
                if hasattr(res, "content") and isinstance(res.content, (dict, list)):
                    return {"query": query, "strategy_used": "unknown", "hits": res.content, "total_results": len(res.content)}  # type: ignore[dict-item]
                return json.loads(json.dumps(res, default=str))

            payload = _normalize(result)
            if int(payload.get("total_results", 0)) == 0 and vault_path:
                payload = _normalize(await _call(None))
                payload.setdefault("explanation", "")
                payload["explanation"] = (payload["explanation"] + " | Fallback: no vault match; searched all vaults").strip()
            return payload

    except Exception as e:
        raise RuntimeError(f"HTTP MCP call failed: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run smart search via Graph RAG MCP Server")
    parser.add_argument("--query", default="What are my health goals?", help="Search query")
    parser.add_argument("--vault", required=True, help="Path to Obsidian vault root")
    parser.add_argument("--chroma", default=None, help="Path to ChromaDB directory (optional)")
    parser.add_argument("--k", type=int, default=6, help="Number of results")
    parser.add_argument("--no-redact", action="store_true", help="Print full URIs (including vault name)")
    parser.add_argument("--mode", choices=["direct", "mcp-http"], default="direct", help="Invocation mode")
    parser.add_argument("--url", default="http://localhost:8765", help="MCP HTTP URL (for mcp-http mode)")
    parser.add_argument("--verbose", action="store_true", help="Verbose logs (may print paths)")
    parser.add_argument("--json", action="store_true", help="Print full JSON response instead of formatted output")
    args = parser.parse_args()

    vault_path = _abs(args.vault)

    # If user provided a Chroma path, export it so server code picks it up
    if args.chroma:
        os.environ["OBSIDIAN_RAG_CHROMA_DIR"] = _abs(args.chroma)
    else:
        # Default to repo-local .chroma_db if available
        repo_root = Path(__file__).resolve().parents[1]
        default_chroma = repo_root / ".chroma_db"
        if default_chroma.exists():
            os.environ.setdefault("OBSIDIAN_RAG_CHROMA_DIR", str(default_chroma))

    # Optional verbose logging
    if args.verbose:
        import logging
        logging.getLogger("graph_rag_mcp").setLevel(logging.INFO)
        logging.getLogger("unified_store").setLevel(logging.INFO)
        logging.getLogger("sentence_transformers").setLevel(logging.INFO)
        logging.getLogger("chromadb").setLevel(logging.INFO)
        logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.INFO)

    try:
        if args.mode == "mcp-http":
            payload = asyncio.run(_run_http(args.query, vault_path, args.k, args.url))
        else:
            payload = asyncio.run(_run_direct(args.query, vault_path, args.k))
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            _print_results(payload, redact_sensitive=not args.no_redact)
    except Exception as e:
        # Sanitize errors to avoid leaking local paths or env
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
