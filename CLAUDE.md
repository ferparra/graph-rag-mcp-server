---
connection_strength: 0.5
enrichment_model: gemini-2.5-flash
enrichment_version: '1.0'
key_concepts:
- Graph RAG MCP Server for Obsidian 2. Dual Database Architecture (ChromaDB & RDFLib)
  3. Model Context Protocol (MCP) Integration 4. Intelligent Q&A over Obsidian Vaults
  5. Development and Indexing Operations
last_enriched: '2025-08-23T20:57:21.272237'
para_category: AI/Knowledge Management System
para_confidence: 1.0
para_reasoning: The note describes a specific software development project ("Graph
  RAG MCP Server for Obsidian"), outlining its purpose, components, and providing
  essential development and indexing commands. The file path also clearly indicates
  a dedicated folder for this project. This aligns perfectly with the definition of
  a 'project' in the PARA method.
para_type: project
potential_links:
- '[[README]]'
related_topics:
- Obsidian Retrieval Augmented Generation (RAG) Vector Databases Graph Databases Large
  Language Models (LLMs) Model Context Protocol (MCP) Local-first software development
tags:
- area/ai/rag
- para/project
- para/project/ai/knowledge-management-system
- project/knowledge-management/obsidian
- tech/database/graph/rdflib
- tech/database/vector/chromadb
- tech/llm/gemini
- tech/protocol/mcp
- type/project-overview
- type/technical-guide
---

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Graph RAG MCP Server for Obsidian - A local-first system that combines ChromaDB vector search, RDFLib semantic graph relationships, and Gemini 2.5 Flash for intelligent Q&A over Obsidian vaults. The system exposes its capabilities through the Model Context Protocol (MCP) for integration with Claude Desktop.

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
# Full indexing (both ChromaDB and RDF graph)
uv run scripts/reindex.py all

# Index ChromaDB only (vector search)
uv run scripts/reindex.py chroma

# Index RDF graph only (relationships)
uv run scripts/reindex.py rdf

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

### Dual Database Design
The system uses a dual database architecture that treats each database as specialized for different query patterns:

- **ChromaDB**: Vector search over semantically chunked document content using sentence transformers
- **RDFLib + Oxigraph**: Semantic graph relationships using W3C RDF standards and SPARQL queries (powered by Oxigraph's embedded RocksDB backend)

### Intelligent Semantic Chunking
The system implements intelligent semantic chunking that preserves the natural structure of Obsidian notes:

- **Hierarchical Chunking**: Respects markdown headers (H1-H6), sections, paragraphs, lists, code blocks, and tables
- **Context Preservation**: Each chunk maintains parent header hierarchy for full context
- **Importance Scoring**: Chunks are scored based on header level, position, link density, and content type
- **Graph Relationships**: Chunks are linked via sequential and hierarchical relationships in the RDF graph
- **Multi-hop Retrieval**: RAG system can expand from vector search hits to related chunks via graph traversal

### Core Components

**Data Flow**: `Obsidian Files → fs_indexer → ChromaDB + RDF Graph → DSPy RAG → MCP Tools`

1. **src/fs_indexer.py**: Parses Markdown files, extracts frontmatter, wikilinks `[[...]]`, and tags `#...`. Basic file parsing and metadata extraction.

2. **src/semantic_chunker.py**: Intelligent chunking module that creates semantically meaningful chunks based on markdown structure (headers, sections, lists, code blocks). Calculates importance scores and preserves hierarchical context.

3. **src/chroma_store.py**: ChromaDB wrapper for vector operations. Uses sentence-transformers for embeddings with support for both semantic and character chunking strategies.

4. **src/graph_store.py**: RDFGraphStore class using rdflib with Oxigraph backend (embedded RocksDB). Implements custom ontology with namespaces (VAULT, NOTES, TAGS, CHUNKS) and SPARQL queries for graph traversal. Stores chunk-level relationships. Oxigraph provides native SPARQL 1.1 support with much faster query performance than RDFLib's default engine.

5. **src/dspy_rag.py**: RAG implementation using DSPy framework with Gemini 2.5 Flash via the modern `google-genai` SDK. Includes SemanticRetriever for multi-hop graph-enhanced retrieval.

6. **src/mcp_server.py**: FastMCP server exposing 15 tools for Claude Desktop integration.

7. **src/config.py**: Pydantic-based configuration with environment variable support and semantic chunking settings.

### RDF Ontology Design
The system uses semantic web standards with custom namespaces:
```turtle
@prefix vault: <http://obsidian-vault.local/ontology#>
@prefix notes: <http://obsidian-vault.local/notes/>
@prefix tags: <http://obsidian-vault.local/tags/>
@prefix chunks: <http://obsidian-vault.local/chunks/>

# Note relationships
notes:note_id a vault:Note ;
    vault:hasTitle "Title" ;
    vault:linksTo notes:other_note ;
    vault:hasTag tags:tag_name .

# Chunk relationships (semantic chunking)
chunks:chunk_id a vault:Chunk ;
    vault:belongsToNote notes:note_id ;
    vault:chunkType "section" ;
    vault:hasHeader "Introduction" ;
    vault:headerLevel 2 ;
    vault:importanceScore 0.8 ;
    vault:followedBy chunks:next_chunk ;
    vault:hasParentSection chunks:parent_chunk .
```

### MCP Tools Categories
- **Search & Q&A**: `search_notes`, `answer_question`, `graph_neighbors`, `get_subgraph`
- **Note Operations**: `list_notes`, `read_note`, `get_note_properties`, `update_note_properties`, `add_content_to_note`
- **Graph Navigation**: `get_backlinks`, `get_notes_by_tag`
- **Management**: `archive_note`, `create_folder`

## Configuration

### Key Environment Variables
- `GEMINI_API_KEY`: Required for Q&A functionality
- `RDF_DB_PATH`: Base path for RDF store (default: `.vault_graph.db` - Oxigraph will create `.vault_graph_oxigraph/` directory)
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

## Important Implementation Notes

### Database Synergy
ChromaDB and RDF work together - not as fallbacks but as complementary systems:
- ChromaDB: Semantic similarity search over semantically chunked content
- RDF Graph (Oxigraph): Structural relationships (backlinks, tags, hierarchy) + chunk relationships with native SPARQL 1.1 execution
- Multi-hop Retrieval: Vector search → Graph expansion → Context enrichment
- Combined retrieval dramatically enhances RAG context quality and relevance
- Oxigraph provides 10-100x faster SPARQL query performance compared to RDFLib's default Python-based engine

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