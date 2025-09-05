#!/usr/bin/env python3
"""
Graph RAG MCP Server for Obsidian
Main entry point for running the MCP server
"""
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.mcp_server import run_stdio

def main():
    """Run the MCP server via stdio for Claude Desktop integration."""
    print("🚀 Starting Graph RAG MCP Server for Obsidian...")
    run_stdio()

if __name__ == "__main__":
    main()
