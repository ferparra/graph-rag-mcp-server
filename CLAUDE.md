# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Graph RAG MCP Server for Obsidian - A local-first system with unified ChromaDB store that combines vector search and graph relationships using metadata, powered by Gemini 2.5 Flash for intelligent Q&A over Obsidian vaults. The system exposes its capabilities through the Model Context Protocol (MCP) for integration with Claude Desktop.

## Essential Commands

### Development Setup
```bash
# Install dependencies
uv sync

# Setup environment 
cp .env.example .env
# Edit .env to add GEMINI_API_KEY

# Type checking (core source only, scripts excluded)
uv run pyrefly check
```

### Indexing Operations
```bash
# Full indexing with unified store (vector search + graph relationships)
uv run scripts/reindex.py all

# Index using unified store (same as 'all')
uv run scripts/reindex.py unified

# Index ChromaDB only (vector-only mode)
uv run scripts/reindex.py chroma

# Check indexing status
uv run scripts/reindex.py status

# Real-time file watching (auto-reindex on changes)
uv run scripts/reindex_watch.py start
```

### Running the MCP Server
```bash
# Start MCP server for Claude Desktop (stdio)
uv run main.py

# Alternative direct run
uv run src/mcp_server.py
```

## Architecture

### Unified Store Design
The system uses a unified ChromaDB architecture that combines vector search with graph relationships stored as metadata:

- **ChromaDB**: Vector search over semantically chunked document content using sentence transformers
- **Graph Metadata**: Relationships (links, tags, chunk hierarchies) stored as ChromaDB metadata for colocated querying
- **Simplified Architecture**: Single database eliminates cross-database synchronization and simplifies maintenance

### Intelligent Semantic Chunking
The system implements intelligent semantic chunking that preserves the natural structure of Obsidian notes:

- **Hierarchical Chunking**: Respects markdown headers (H1-H6), sections, paragraphs, lists, code blocks, and tables
- **Context Preservation**: Each chunk maintains parent header hierarchy for full context
- **Importance Scoring**: Chunks are scored based on header level, position, link density, and content type
- **Graph Relationships**: Chunks are linked via sequential and hierarchical relationships stored in ChromaDB metadata
- **Multi-hop Retrieval**: RAG system can expand from vector search hits to related chunks via graph traversal

### Core Components

**Data Flow**: `Obsidian Files → fs_indexer → Unified ChromaDB Store → Enhanced DSPy RAG → MCP Tools`

1. **src/fs_indexer.py**: Parses Markdown files, extracts frontmatter, wikilinks `[[...]]`, and tags `#...`. Basic file parsing and metadata extraction.

2. **src/semantic_chunker.py**: Intelligent chunking module that creates semantically meaningful chunks based on markdown structure (headers, sections, lists, code blocks). Calculates importance scores and preserves hierarchical context.

3. **src/chroma_store.py**: ChromaDB wrapper for vector operations. Uses sentence-transformers for embeddings with support for both semantic and character chunking strategies.

4. **src/unified_store.py**: UnifiedStore class extending ChromaDB with graph relationship capabilities. Stores links, tags, and chunk hierarchies as metadata for efficient querying and graph traversal.

5. **src/dspy_rag.py**: Enhanced RAG implementation with EnhancedVaultSearcher that includes adaptive query routing, multi-strategy retrieval, and ReAct agent for complex queries.

6. **src/dspy_programs.py**: Advanced DSPy programs with query intent classification, adaptive retrieval, and self-optimization capabilities.

7. **src/dspy_optimizer.py**: MIPROv2 optimization with persistent state management for continuous improvement across uvx connections.

8. **src/dspy_agent.py**: ReAct agent for complex multi-hop reasoning with graph traversal tools.

9. **src/mcp_server.py**: FastMCP server exposing 17 tools for Claude Desktop integration, including DSPy optimization controls.

10. **src/config.py**: Pydantic-based configuration with environment variable support and DSPy optimization settings.

### Metadata Schema Design
The system stores graph relationships as ChromaDB metadata:
```python
# Note-level metadata
{
    "note_id": "unique_note_id",
    "title": "Note Title",
    "links_to": "note2,note3,note4",  # Comma-separated
    "backlinks_from": "note5,note6",
    "tags": "tag1,tag2,tag3"
}

# Chunk-level metadata (semantic chunking)
{
    "chunk_id": "unique_chunk_id",
    "chunk_type": "section",
    "header_text": "Introduction",
    "header_level": 2,
    "importance_score": 0.8,
    "sequential_next": "next_chunk_id",
    "sequential_prev": "prev_chunk_id",
    "parent_chunk": "parent_chunk_id",
    "child_chunks": "child1,child2"
}
```

### MCP Tools Categories
- **Search & Q&A**: `search_notes`, `answer_question`, `graph_neighbors`, `get_subgraph`
- **Note Operations**: `list_notes`, `read_note`, `get_note_properties`, `update_note_properties`, `add_content_to_note`
- **Graph Navigation**: `get_backlinks`, `get_notes_by_tag`
- **Management**: `archive_note`, `create_folder`

## Configuration

### Key Environment Variables
- `GEMINI_API_KEY`: Required for Q&A functionality
- `OBSIDIAN_RAG_CHROMA_DIR`: ChromaDB storage directory (default: `.chroma_db`)
- `OBSIDIAN_RAG_*`: Prefixed settings for embedding models, chunk sizes, etc.

### Semantic Chunking Configuration
- `OBSIDIAN_RAG_CHUNK_STRATEGY`: "semantic" (default) or "character"
- `OBSIDIAN_RAG_SEMANTIC_MIN_CHUNK_SIZE`: Minimum chunk size (default: 100)
- `OBSIDIAN_RAG_SEMANTIC_MAX_CHUNK_SIZE`: Maximum chunk size (default: 3000)
- `OBSIDIAN_RAG_SEMANTIC_MERGE_THRESHOLD`: Merge small chunks threshold (default: 200)
- `OBSIDIAN_RAG_SEMANTIC_INCLUDE_CONTEXT`: Include parent headers (default: true)

### Default Behavior
- Vault path defaults to parent directory of project
- Uses `all-MiniLM-L6-v2` for embeddings
- Semantic chunking with hierarchical structure preservation
- Gemini 2.5 Flash for generation with multi-hop retrieval
- **Enhanced DSPy RAG with self-optimization enabled** (see [DSPy Enhancements](docs/DSPY_ENHANCEMENTS.md))

### DSPy Optimization Settings
- `OBSIDIAN_RAG_DSPY_STATE_DIR`: Persistent directory for optimized programs (default: `.dspy_state`)
- `OBSIDIAN_RAG_DSPY_OPTIMIZE_ENABLED`: Enable/disable optimization (default: `true`)
- `OBSIDIAN_RAG_DSPY_AUTO_MODE`: MIPROv2 mode - light/medium/heavy (default: `light`)
- `OBSIDIAN_RAG_DSPY_OPTIMIZATION_INTERVAL_HOURS`: Hours between optimizations (default: `168`)
- `OBSIDIAN_RAG_DSPY_MAX_EXAMPLES`: Max examples for optimization (default: `50`)

## Important Implementation Notes

### Unified Store Benefits
ChromaDB metadata provides both vector search and graph capabilities:
- **Vector Search**: Semantic similarity search over semantically chunked content
- **Graph Relationships**: Links, tags, and hierarchies stored as metadata for fast filtering
- **Colocated Data**: Vectors and relationships in same database for optimal performance
- **Multi-hop Retrieval**: Vector search → Metadata-based graph expansion → Context enrichment
- **Simplified Architecture**: Single database eliminates synchronization complexity

### Google Genai SDK Usage
**Critical**: Uses the NEW `google-genai` SDK, not the deprecated `google-generativeai`. This is configured in pyproject.toml and used throughout dspy_rag.py.

### Type Safety
- Strongly typed with Pydantic models throughout
- pyrefly type checker configured to exclude scripts/ directory (which use dynamic imports)
- All pyrefly ignore comments are intentional for runtime dynamic behavior

### File Processing
- Supports `.md`, `.markdown`, `.txt`, and `.excalidraw` files
- Extracts text labels from Excalidraw JSON for indexing
- Frontmatter parsed with python-frontmatter library
- Wikilinks resolved by title/stem matching across vault

### MCP Integration
- FastMCP framework with stdio transport for Claude Desktop
- Pydantic models for tool inputs/outputs ensure type safety
- Real-time graph and vector updates when notes change