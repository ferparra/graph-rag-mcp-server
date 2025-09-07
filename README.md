# Graph RAG MCP Server for Obsidian

A powerful local-first Graph-RAG system that combines **ChromaDB** unified store with metadata-based graph relationships and **Gemini 2.5 Flash** for intelligent Q&A over your Obsidian vault.

## 🌟 Features

- **📊 Unified Database Architecture**: ChromaDB with metadata-based graph relationships for both semantic search and graph traversal
- **🧠 Intelligent Semantic Chunking**: Respects markdown structure (headers, sections, lists, code blocks)
- **🎯 PARA Taxonomy Classification**: AI-powered organization using Projects, Areas, Resources, Archive system
- **🤖 RAG-Powered Q&A**: Multi-hop retrieval with Gemini 2.5 Flash
- **🕸️ Graph Navigation**: Explore backlinks, tags, and semantic relationships
- **🔄 Real-time Sync**: File watcher for automatic indexing
- **🏠 Local-First**: All processing happens locally (except Gemini API calls)
- **📝 Multi-Client MCP Support**: Works with Claude Desktop, Cursor, and Raycast
- **⚡ Dual Transport Modes**: stdio for Claude Desktop, HTTP for other clients
- **🛠️ Automated Installation**: One-command setup with client detection
- **🔒 Strongly Typed**: Pydantic models throughout for reliability

## 🎯 Supported MCP Clients

- **🤖 Claude Desktop**: Full stdio integration with automatic configuration
- **📝 Cursor**: HTTP mode with MCP extension support  
- **⚡ Raycast**: HTTP API with custom extension templates
- **🔌 Any MCP Client**: Standard MCP protocol support (stdio/HTTP)

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   Obsidian      │    │   ChromaDB      │
│     Vault       │───▶│ (Unified Store) │
│                 │    │ Vector + Graph  │
└─────────────────┘    │   Metadata      │
         │              └─────────────────┘
         │                       │
         │                       ▼
         │              ┌─────────────────┐
         │              │      DSPy       │
         └─────────────▶│   RAG Engine    │
                        │  + Gemini 2.5   │
                        │ (Multi-hop      │
                        │  Retrieval)     │
                        └─────────────────┘
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

### Automated Installation (Recommended)

The easiest way to get started with Claude Desktop, Cursor, or Raycast:

```bash
# Interactive setup wizard
uv run install.py

# Or non-interactive with your settings
uv run install.py --vault "/path/to/your/vault" --api-key "your_key"
```

The installer will:
- ✅ Detect your installed MCP clients (Claude Desktop, Cursor, Raycast)
- ⚙️ Configure each client automatically  
- 📦 Install all dependencies
- 🧪 Test the installation
- 📝 Create environment configuration

### Manual Installation

If you prefer manual setup:

#### 1. Install uv (if not installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh  # macOS/Linux
```

#### 2. Install Dependencies
```bash
uv sync
```

#### 3. Configure Environment
```bash
cp configs/.env.example .env
# Edit .env and add your GEMINI_API_KEY and vault paths
```

#### 4. Index Your Vault
```bash
# Full indexing (unified ChromaDB store)
uv run scripts/reindex.py all

# Check indexing status
uv run scripts/reindex.py status
```

#### 5. Configure Your MCP Client

**Claude Desktop (stdio mode):**
```json
{
  "mcpServers": {
    "graph-rag-obsidian": {
      "command": "uvx",
      "args": ["--python", "3.13", "--from", ".", "graph-rag-mcp-stdio"],
      "cwd": "/path/to/graph-rag-mcp-server",
      "env": {
        "GEMINI_API_KEY": "your_api_key_here",
        "OBSIDIAN_RAG_VAULTS": "/path/to/your/vault"
      }
    }
  }
}
```

**Cursor (HTTP mode):**
```bash
# Start HTTP server
uv run graph-rag-mcp-http

# Configure Cursor MCP extension to use http://localhost:8765
```

**Raycast (HTTP mode):**
```bash
# Start HTTP server  
uv run graph-rag-mcp-http

# Install generated Raycast extension
```

For detailed configuration instructions, see [SETUP.md](SETUP.md).

## 🛠️ Available Commands

### MCP Server Modes

```bash
# Interactive installer
uv run install.py

# Claude Desktop (stdio mode)
uvx --python 3.13 --from . graph-rag-mcp-stdio

# Cursor/Raycast (HTTP mode)  
uv run graph-rag-mcp-http

# HTTP with custom port
uv run graph-rag-mcp-http --port 9000

# Direct stdio runs (alternative)
uv run main.py                    # stdio mode
uv run src/mcp_server.py         # stdio mode  
```

### Indexing Scripts

```bash
# Full indexing
uv run scripts/reindex.py all

# ChromaDB unified store
uv run scripts/reindex.py unified

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

### PARA Taxonomy Enrichment

Enhance your vault with intelligent PARA system classification using DSPy:

```bash
# Analyze current vault taxonomy state
uv run scripts/enrich_para_taxonomy.py analyze --sample 100

# Preview enrichment (dry run) on sample notes
uv run scripts/enrich_para_taxonomy.py enrich --limit 10 --dry-run

# Apply enrichment to specific notes
uv run scripts/enrich_para_taxonomy.py enrich "path/to/note.md" --apply

# Bulk enrichment with filters
uv run scripts/enrich_para_taxonomy.py enrich --limit 50 --folder "Projects" --apply

# FULL VAULT ENRICHMENT (new!)
# Preview entire vault enrichment
uv run scripts/enrich_para_taxonomy.py enrich-all --dry-run

# Apply to entire vault (skips already enriched by default)
uv run scripts/enrich_para_taxonomy.py enrich-all --apply

# Force re-enrichment of entire vault
uv run scripts/enrich_para_taxonomy.py enrich-all --apply --force-all

# Customize batch size for large vaults
uv run scripts/enrich_para_taxonomy.py enrich-all --apply --batch-size 25
```

**PARA Classification Features:**
- 🎯 **Intelligent Classification**: Uses Gemini 2.5 Flash to classify notes into Projects, Areas, Resources, Archive
- 🏷️ **Hierarchical Tags**: Suggests structured tags like `#para/project/ai/automation`
- 🔗 **Relationship Discovery**: Finds potential links between related notes with validation
- 💡 **Concept Extraction**: Identifies key concepts and themes
- 🛡️ **Safe Updates**: Only adds frontmatter, never modifies content
- 📊 **Confidence Scoring**: Provides reasoning and confidence for classifications
- ⚡ **Batch Processing**: Process entire vaults efficiently with configurable batch sizes
- 🔄 **Smart Deduplication**: Avoids reprocessing already enriched notes

## 🔧 MCP Tools

The server exposes these tools for Claude:

### Search & Q&A
- **`search_notes`**: Vector search across vault
- **`answer_question`**: RAG-powered Q&A with citations
- **`graph_neighbors`**: Find related notes via graph
- **`get_subgraph`**: Extract note subgraphs

### Note Operations  
- **`create_note`**: Create new notes with auto-enriched frontmatter
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
- **`reindex_vault`**: Reindex unified ChromaDB store
- **`enrich_notes`**: Apply PARA taxonomy enrichment to notes

## ⚙️ Configuration

Key settings in `.env`:

```bash
# Required
GEMINI_API_KEY=your_key_here

# ChromaDB configuration
OBSIDIAN_RAG_CHROMA_DIR=/custom/path/to/.chroma_db
OBSIDIAN_RAG_COLLECTION=vault_collection

# Optional customization
OBSIDIAN_RAG_EMBEDDING_MODEL=all-MiniLM-L6-v2
OBSIDIAN_RAG_GEMINI_MODEL=gemini-2.5-flash

# Semantic chunking configuration
OBSIDIAN_RAG_CHUNK_STRATEGY=semantic  # or "character" for simple chunking
OBSIDIAN_RAG_SEMANTIC_MIN_CHUNK_SIZE=100
OBSIDIAN_RAG_SEMANTIC_MAX_CHUNK_SIZE=3000
OBSIDIAN_RAG_SEMANTIC_MERGE_THRESHOLD=200
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
# Create a new note with auto-enrichment
note = create_note(
    title="Machine Learning Breakthrough",
    content="# Key Findings\n\nDiscovered new optimization technique...",
    folder="Research",
    tags=["ml", "optimization"],
    para_type="project",  # Hint for PARA classification
    enrich=True  # Apply AI enrichment
)

# Read note
content = read_note("Research/AI Progress.md")

# Update properties
update_note_properties("Research/AI Progress.md", {
    "status": "completed",
    "tags": ["ai", "research", "finished"]
})
```

### Creating Notes with Auto-Enrichment

The `create_note` tool creates properly formatted Obsidian notes with:
- Clean YAML frontmatter
- Automatic PARA classification (if content provided)
- Intelligent tag suggestions
- Potential link discovery
- Timestamps and metadata

**Example created note:**
```markdown
---
created: '2025-08-23T20:30:00.000000'
modified: '2025-08-23T20:30:00.000000'
para_type: project
para_category: ai/research
para_confidence: 0.85
key_concepts:
- Machine Learning Optimization
- Gradient Descent Improvements
- Performance Benchmarking
tags:
- ml
- optimization
- para/project
- para/project/ai/research
- tech/ai/ml/optimization
potential_links:
- '[[Optimization Techniques]]'
- '[[Research Log 2025]]'
enrichment_version: '1.0'
last_enriched: '2025-08-23T20:30:00.000000'
enrichment_model: gemini-2.5-flash
---

# Machine Learning Breakthrough

Your content here...
```

### PARA Enrichment Workflow

**Step 1: Analyze your vault**
```bash
uv run scripts/enrich_para_taxonomy.py analyze --sample 100
```
Shows current taxonomy state and enrichment potential.

**Step 2: Test on subset (dry run)**
```bash
uv run scripts/enrich_para_taxonomy.py enrich --limit 5 --dry-run
```
Preview classifications without making changes.

**Step 3: Apply enrichment**
```bash
# Start small
uv run scripts/enrich_para_taxonomy.py enrich --limit 20 --apply

# Scale up
uv run scripts/enrich_para_taxonomy.py enrich --limit 100 --apply
```

**Example enriched note:**
```yaml
---
para_type: project
para_category: AI/Automation  
para_confidence: 0.9
key_concepts:
  - AI Agent Development
  - Computer Use Automation
  - Grounded AI Systems
tags:
  - "#project/ai/automation"
  - "#area/ai/development"
potential_links:
  - "Related Project Name"
enrichment_version: "1.0"
last_enriched: "2025-08-23T17:59:32"
---
# Your original note content remains unchanged
```

## 🔍 How It Works

1. **File Parsing**: Extracts markdown content, frontmatter, wikilinks, and tags
2. **Semantic Chunking**: Intelligent chunking based on markdown structure (headers, sections, lists)
3. **Vector Indexing**: Stores semantic chunks with embeddings in ChromaDB  
4. **Graph Metadata**: Stores notes, links, tags, and chunk relationships as ChromaDB metadata
5. **PARA Classification**: Uses DSPy + Gemini to classify notes into Projects, Areas, Resources, Archive
6. **RAG Pipeline**: Multi-hop retrieval combining vector search with graph traversal
7. **MCP Interface**: Exposes all capabilities via Model Context Protocol

## 🆕 What's New

### Unified ChromaDB Architecture (Latest Update!)
- **⚡ Simplified Architecture**: Single database for both vector search and graph relationships
- **🗄️ Metadata-based Graph**: Efficient graph storage using ChromaDB's native metadata capabilities
- **🔧 Colocated Data**: Vector embeddings and graph relationships stored together for optimal query performance
- **📦 Reduced Dependencies**: No need for separate RDF libraries or stores
- **🚀 Performance**: Streamlined queries with direct metadata access
- **💾 Efficient Storage**: Single store with optimized embedding and metadata storage

### Enhanced PARA Taxonomy
- **🤖 AI-Powered Classification**: Automatic categorization into Projects, Areas, Resources, Archive
- **🏷️ Smart Tagging**: Hierarchical tags like `#para/project/ai/automation`
- **🔗 Validated Wikilinks**: Only suggests links to existing notes in your vault
- **📊 Batch Processing**: Process entire vaults with configurable batch sizes
- **🎯 Obsidian-Native**: Clean YAML frontmatter without markdown formatting

### ChromaDB Metadata Schema
The system stores graph relationships as ChromaDB metadata:
```python
# Note-level metadata
metadata = {
    "note_id": "my_note",
    "title": "My Note Title",
    "path": "/path/to/note.md",
    "tags": "important,ai,project",
    "links_to": "other_note,related_note",
    "backlinks_from": "source_note,another_note",
    "vault": "my_vault"
}

# Chunk-level metadata (semantic chunking)
chunk_metadata = {
    "chunk_id": "my_note#chunk_0",
    "chunk_type": "section",
    "header_text": "Introduction",
    "header_level": 2,
    "importance_score": 0.8,
    "sequential_next": "my_note#chunk_1",
    "sequential_prev": "",
    "parent_chunk": "my_note#header_0",
    "child_chunks": "my_note#chunk_1,my_note#chunk_2",
    "sibling_chunks": "my_note#chunk_3",
    "semantic_chunk": True
}
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

# Test unified store
uv run python -c "
from src.unified_store import UnifiedStore
from src.config import settings
store = UnifiedStore(
    client_dir=settings.chroma_dir,
    collection_name=settings.collection,
    embed_model=settings.embedding_model
)
stats = store.get_stats()
print(f'Store stats: {stats}')
"
```

## 🔧 Troubleshooting

### ChromaDB Issues
- ChromaDB stores data in `.chroma_db/` directory
- Check disk space and permissions
- Try full reindex: `uv run scripts/reindex.py unified --full`
- Database is automatically created on first run

### Unified Store Issues
- Check disk space for `.chroma_db/`
- Try full reindex: `uv run scripts/reindex.py unified --full`
- Ensure notes are properly deduplicated (fixed in latest version)

### Gemini API Issues
- Verify API key: `uv run scripts/reindex.py status`
- Check rate limits and quotas
- For enrichment errors, try smaller batch sizes

### Enrichment Issues
- Use `--dry-run` to preview changes first
- Check note has content (empty files are skipped)
- Reduce batch size if hitting API limits: `--batch-size 10`

## 📁 Project Structure

```
graph-rag-mcp-server/
├── src/
│   ├── config.py               # Configuration management
│   ├── fs_indexer.py           # File parsing & metadata extraction
│   ├── semantic_chunker.py     # Intelligent markdown-aware chunking
│   ├── chroma_store.py         # Vector database operations
│   ├── unified_store.py        # ChromaDB unified operations (vectors + graph metadata)
│   ├── dspy_rag.py             # RAG engine with Gemini 2.5 Flash
│   └── mcp_server.py           # FastMCP server & tool definitions
├── scripts/
│   ├── reindex.py              # Database indexing utilities
│   ├── reindex_watch.py        # Real-time file monitoring
│   ├── enrich_para_taxonomy.py # PARA classification & enrichment
│   └── migrate_rdf_store.py    # (removed)
├── configs/
│   ├── claude-desktop.json     # Claude Desktop MCP configuration template
│   ├── cursor-mcp.json         # Cursor MCP configuration template  
│   ├── raycast-config.json     # Raycast extension configuration template
│   └── .env.example            # Environment configuration template
├── install.py                  # Automated installer & configurator
├── main.py                     # Alternate MCP server entry point (stdio)
├── pyproject.toml              # Dependencies & entry points (uv managed)
├── SETUP.md                    # Comprehensive setup guide
└── README.md                   # Project overview & quick start
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality  
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

---

**Built with modern python stack**: Pydantic, ChromaDB, DSPy, FastMCP, and the latest google-genai SDK.

### Deterministic MCP Evals (DSPy-assisted)

This repo includes a tiny on-disk test corpus at `rag_mcp_server_test_content/` and a runner that executes deterministic evals for the MCP tools. The evals use distinctive phrases and low-temperature LLM configs so results are stable across embedding and Gemini model choices.

- Build/refresh the test corpus: `uv run scripts/build_test_content.py`
- Run evals for MCP tools: `uv run scripts/run_mcp_evals.py`

The runner uses an immutable baseline snapshot at `rag_mcp_server_test_content_baseline/`, copies it into a temporary working directory for the duration of the run, points the vault to that temporary copy, reinitializes app state, reindexes, and then checks:
- `search_notes` returns the expected notes for Earth/Mars queries consistently
- `graph_neighbors`, `get_subgraph`, `get_backlinks`, `get_notes_by_tag`
- CRUD: `create_note`, `add_content_to_note`, `archive_note`
- `read_note`, `get_note_properties`, `update_note_properties`, `list_notes`

If you want to use a different embedding or Gemini model, set `OBSIDIAN_RAG_EMBEDDING_MODEL` or `OBSIDIAN_RAG_GEMINI_MODEL` and re-run the evals. The test corpus is crafted to remain unambiguous across models. The baseline directory is never mutated; the working copy is temporary and cleaned up after the run.
