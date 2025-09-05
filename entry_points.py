#!/usr/bin/env python3
"""Entry points for Graph RAG MCP Server."""
import sys
from pathlib import Path

def run_stdio():
    """Run MCP server via stdio for Claude Desktop."""
    # Ensure package imports resolve when installed
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))

    from src.mcp_server import run_stdio as _run_stdio
    _run_stdio()

def run_http():
    """Run MCP server via HTTP for Cursor and other clients."""
    # Ensure package imports resolve when installed
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))

    from src.mcp_server import run_http as _run_http
    _run_http()

def run_main():
    """Run main entry point."""
    # Add project root to path for imports
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    import main
    main.main()

def run_installer():
    """Run the installer."""
    # Add project root to path for imports
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    import install
    install.main()
