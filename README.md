# Graph RAG MCP Server for Obsidian

A powerful local-first Graph-RAG system that combines **ChromaDB** vector search, **RDFLib** semantic graph relationships, and **Gemini 2.5 Flash** for intelligent Q&A over your Obsidian vault.

## 🌟 Features

- **📊 Dual Database Architecture**: ChromaDB for semantic search + RDFLib with SQLite for graph relationships
- **🤖 RAG-Powered Q&A**: Ask questions about your vault using Gemini 2.5 Flash
- **🕸️ Graph Navigation**: Explore backlinks, tags, and note relationships
- **🔄 Real-time Sync**: File watcher for automatic indexing
- **🏠 Local-First**: All processing happens locally (except Gemini API calls)
- **📝 MCP Integration**: Full Model Context Protocol support for Claude Desktop
- **🔒 Strongly Typed**: Pydantic models throughout for reliability

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Obsidian      │    │   ChromaDB      │    │   RDFLib +      │
│     Vault       │───▶│  (Vector DB)    │    │   SQLite        │
│                 │    │                 │    │  (Graph DB)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       ▼                       ▼
         │              ┌─────────────────┐    ┌─────────────────┐
         │              │      DSPy       │    │   SPARQL Queries│
         └─────────────▶│   RAG Engine    │◀──▶│  (Neighbors,    │
                        │  + Gemini 2.5   │    │   Subgraphs)    │
                        └─────────────────┘    └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │   MCP Server    │
                        │   (FastMCP)     │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ Claude Desktop  │
                        │   Integration   │
                        └─────────────────┘
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
uv sync
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 3. Index Your Vault

```bash
# Full indexing (ChromaDB + RDF Graph)
uv run scripts/reindex.py all

# Or ChromaDB only
uv run scripts/reindex.py chroma

# Or RDF graph only
uv run scripts/reindex.py rdf
```

### 4. Run MCP Server

```bash
# For Claude Desktop (stdio)
uv run main.py

# Or run directly
uv run src/mcp_server.py
```

### 5. Configure Claude Desktop

Add to your MCP settings:

```json
{
  "mcpServers": {
    "graph-rag-obsidian": {
      "command": "uv",
      "args": ["run", "python", "main.py"],
      "cwd": "/path/to/graph-rag-mcp-server",
      "env": {
        "GEMINI_API_KEY": "your_api_key_here"
      }
    }
  }
}
```

## 🛠️ Available Commands

### Indexing Scripts

```bash
# Full indexing
uv run scripts/reindex.py all

# ChromaDB only
uv run scripts/reindex.py chroma

# RDF graph only  
uv run scripts/reindex.py rdf

# Check status
uv run scripts/reindex.py status
```

### Real-time Watching

```bash
# Start file watcher
uv run scripts/reindex_watch.py start

# Test file detection
uv run scripts/reindex_watch.py test
```

## 🔧 MCP Tools

The server exposes these tools for Claude:

### Search & Q&A
- **`search_notes`**: Vector search across vault
- **`answer_question`**: RAG-powered Q&A with citations
- **`graph_neighbors`**: Find related notes via graph
- **`get_subgraph`**: Extract note subgraphs

### Note Operations  
- **`list_notes`**: Browse vault contents
- **`read_note`**: Get full note content
- **`get_note_properties`**: Read frontmatter
- **`update_note_properties`**: Modify frontmatter
- **`add_content_to_note`**: Append content

### Graph Navigation
- **`get_backlinks`**: Find notes linking to target
- **`get_notes_by_tag`**: Find notes by tag

### Management
- **`archive_note`**: Move notes to archive
- **`create_folder`**: Create directories

## ⚙️ Configuration

Key settings in `.env`:

```bash
# Required
GEMINI_API_KEY=your_key_here

# Optional RDF configuration
RDF_DB_PATH=/custom/path/to/vault_graph.db
RDF_STORE_IDENTIFIER=my_vault_graph

# Optional customization
OBSIDIAN_RAG_EMBEDDING_MODEL=all-MiniLM-L6-v2
OBSIDIAN_RAG_GEMINI_MODEL=gemini-2.5-flash
OBSIDIAN_RAG_MAX_CHARS=1800
OBSIDIAN_RAG_OVERLAP=200
```

## 🏃‍♂️ Usage Examples

### Search Your Vault
```python
# Vector search
results = search_notes("machine learning algorithms", k=5)

# Q&A with context
answer = answer_question("What did I learn about transformers?")
```

### Explore Relationships
```python
# Find related notes
neighbors = graph_neighbors("Deep Learning", depth=2)

# Get backlinks
backlinks = get_backlinks("Neural Networks")

# Find by tag
tagged_notes = get_notes_by_tag("ai")
```

### Manage Notes
```python
# Read note
content = read_note("Research/AI Progress.md")

# Update properties
update_note_properties("Research/AI Progress.md", {
    "status": "completed",
    "tags": ["ai", "research", "finished"]
})
```

## 🔍 How It Works

1. **File Parsing**: Extracts markdown content, frontmatter, wikilinks, and tags
2. **Vector Indexing**: Chunks text and stores embeddings in ChromaDB  
3. **RDF Graph Building**: Creates semantic triples for notes, links, and tags in SQLite
4. **RAG Pipeline**: DSPy + Gemini for grounded question answering
5. **MCP Interface**: Exposes all capabilities via Model Context Protocol

## 🆕 What's New in RDFLib Version

### Advantages of RDFLib + SQLite
- **No External Dependencies**: Embedded SQLite database, no server setup required
- **Semantic Web Standards**: Uses W3C RDF and SPARQL standards
- **Flexible Schema**: Easy to extend with new relationship types
- **Persistent Storage**: SQLite provides ACID compliance and durability
- **Lightweight**: Much smaller footprint than graph databases
- **Query Power**: SPARQL provides powerful semantic graph queries

### RDF Schema
The system uses a custom ontology:
```turtle
@prefix vault: <http://obsidian-vault.local/ontology#> .
@prefix notes: <http://obsidian-vault.local/notes/> .
@prefix tags: <http://obsidian-vault.local/tags/> .

# Notes and their properties
notes:my_note a vault:Note ;
    vault:hasTitle "My Note Title" ;
    vault:hasPath "/path/to/note.md" ;
    vault:linksTo notes:other_note ;
    vault:hasTag tags:important .

# Tags
tags:important a vault:Tag ;
    vault:hasName "important" .
```

## 🛡️ Security & Privacy

- **Local-First**: All data processing happens on your machine
- **API Calls**: Only Gemini API for text generation (optional)
- **No Data Leakage**: Vault content never leaves your control
- **Path Validation**: Prevents directory traversal attacks

## 🧪 Testing

```bash
# Test indexing
uv run scripts/reindex.py status

# Test file watching  
uv run scripts/reindex_watch.py test

# Test RDF queries
uv run python -c "
from src.graph_store import RDFGraphStore
from src.config import settings
store = RDFGraphStore(settings.rdf_db_path, settings.rdf_store_identifier)
stats = store.get_stats()
print(f'Graph stats: {stats}')
store.close()
"
```

## 🔧 Troubleshooting

### RDF Database Issues
- Check disk space for `.vault_graph.db`
- Try full reindex: `uv run scripts/reindex.py rdf`
- Database is automatically created on first run

### ChromaDB Issues
- Check disk space for `.chroma_db/`
- Try full reindex: `uv run scripts/reindex.py all --full`

### Gemini API Issues
- Verify API key: `uv run scripts/reindex.py status`
- Check rate limits and quotas

## 📁 Project Structure

```
graph-rag-mcp-server/
├── src/
│   ├── config.py          # Configuration management
│   ├── fs_indexer.py      # File parsing & chunking
│   ├── chroma_store.py    # Vector database ops
│   ├── graph_store.py     # RDF graph database ops  
│   ├── dspy_rag.py        # RAG with Gemini
│   └── mcp_server.py      # MCP server & tools
├── scripts/
│   ├── reindex.py         # Batch indexing
│   └── reindex_watch.py   # Real-time watching
├── main.py                # Entry point
├── pyproject.toml         # Dependencies
└── .env.example           # Configuration template
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality  
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

---

**Built with modern python stack**: Pydantic, ChromaDB, RDFLib, DSPy, FastMCP, and the latest google-genai SDK.