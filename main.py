#!/usr/bin/env python3
"""
Graph RAG MCP Server for Obsidian
Main entry point for running the MCP server
"""
import sys
import logging

from src.mcp_server import run_stdio

def main():
    """Run the MCP server via stdio for Claude Desktop integration."""
    # Log to stderr to avoid corrupting MCP stdio on stdout
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    logging.getLogger("graph_rag_mcp").info("Starting Graph RAG MCP Server for Obsidian (stdio)")
    run_stdio()

if __name__ == "__main__":
    main()
