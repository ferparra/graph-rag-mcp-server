#!/usr/bin/env python3
"""Entry points for Graph RAG MCP Server."""

def run_stdio():
    """Run MCP server via stdio for Claude Desktop."""
    from .mcp_server import run_stdio as _run_stdio
    _run_stdio()

def run_http():
    """Run MCP server via HTTP for Cursor and other clients."""
    from .mcp_server import run_http as _run_http
    _run_http()

def run_main():
    """Run main entry point."""
    from .main import main
    main()

def run_installer():
    """Run the installer."""
    from .install import main
    main()