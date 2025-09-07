# 🚀 Graph RAG MCP Server - Setup Guide

This guide will help you set up the Graph RAG MCP Server with Claude Desktop, Cursor, and Raycast using `uv` (Astral's fast Python package manager).

## 📋 Prerequisites

### 1. Install `uv` (Required)
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Get a Gemini API Key
- Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
- Create a new API key
- Save it for configuration

### 3. Locate Your Obsidian Vault
Find the path to your Obsidian vault. Common locations:
- macOS: `~/Library/Mobile Documents/iCloud~md~obsidian/Documents/YourVault`
- Windows: `C:\Users\YourName\Documents\ObsidianVault`
- Linux: `~/Documents/ObsidianVault`

## 🎯 Quick Install (Automated)

The easiest way to set up the MCP server for all clients:

```bash
# Clone the repository
git clone https://github.com/yourusername/graph-rag-mcp-server
cd graph-rag-mcp-server

# Install dependencies
uv sync

# Run the installer for all clients
uv run install.py all

# Or install for specific clients
uv run install.py claude cursor raycast

# With custom vault path
uv run install.py all --vault "/path/to/your/vault" --api-key "your-gemini-key"
```

The installer will:
- ✅ Check for `uv` installation
- ✅ Install all Python dependencies
- ✅ Create `.env` file with your configuration
- ✅ Configure each selected client
- ✅ Test the server startup
- ✅ Offer to index your vault

## 🔧 Manual Installation

If you prefer to configure each client manually:

### Step 1: Set Up Environment

```bash
# Clone and enter the project
git clone https://github.com/yourusername/graph-rag-mcp-server
cd graph-rag-mcp-server

# Install dependencies with uv
uv sync

# Create .env file
cp .env.example .env

# Edit .env with your settings
# Add your GEMINI_API_KEY and vault path
```

### Step 2: Test the Server

```bash
# Test stdio mode (for Claude/Cursor/Raycast)
uv run graph-rag-mcp-stdio

# Test HTTP mode (alternative for Cursor)
uv run graph-rag-mcp-http
```

### Step 3: Configure Your Client

## 🖥️ Claude Desktop Configuration

### Automatic Setup
```bash
uv run install.py claude
```

### Manual Setup

1. **Find your config file:**
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`
   - Linux: `~/.config/Claude/claude_desktop_config.json`

2. **Add the server configuration:**

```json
{
  "mcpServers": {
    "graph-rag-obsidian": {
      "command": "uv",
      "args": ["run", "graph-rag-mcp-stdio"],
      "cwd": "/path/to/graph-rag-mcp-server",
      "env": {
        "GEMINI_API_KEY": "your-api-key-here",
        "OBSIDIAN_VAULT_PATH": "/path/to/your/vault"
      }
    }
  }
}
```

3. **Restart Claude Desktop**

4. **Verify:** Look for "graph-rag-obsidian" in the MCP servers list

## 🎨 Cursor Configuration

### Automatic Setup
```bash
uv run install.py cursor
```

### Manual Setup

1. **Option A: Stdio Mode (Recommended)**

Add to your Cursor MCP configuration:

```json
{
  "mcpServers": {
    "graph-rag-obsidian": {
      "command": "uv",
      "args": ["run", "graph-rag-mcp-stdio"],
      "cwd": "/path/to/graph-rag-mcp-server",
      "env": {
        "GEMINI_API_KEY": "your-api-key-here",
        "OBSIDIAN_VAULT_PATH": "/path/to/your/vault"
      }
    }
  }
}
```

2. **Option B: HTTP Mode**

First, start the HTTP server:
```bash
uv run graph-rag-mcp-http
# Server runs on http://localhost:8765
```

Then add to Cursor config:
```json
{
  "mcpServers": {
    "graph-rag-obsidian-http": {
      "type": "http",
      "url": "http://localhost:8765"
    }
  }
}
```

3. **Restart Cursor**

## 🔍 Raycast Configuration

### Automatic Setup
```bash
uv run install.py raycast
```

### Manual Setup

1. **Install Raycast MCP Extension:**
   ```
   raycast://extensions/raycast/mcp
   ```

2. **Open MCP Servers folder:**
   - Raycast → AI → Manage MCP Servers → Open Servers Folder

3. **Create/edit `mcp-config.json`:**

```json
{
  "mcpServers": {
    "graph-rag-obsidian": {
      "command": "uv",
      "args": ["run", "graph-rag-mcp-stdio"],
      "cwd": "/path/to/graph-rag-mcp-server",
      "env": {
        "GEMINI_API_KEY": "your-api-key-here",
        "OBSIDIAN_VAULT_PATH": "/path/to/your/vault"
      },
      "metadata": {
        "name": "Graph RAG Obsidian",
        "description": "AI-powered search for your vault",
        "icon": "🧠"
      }
    }
  }
}
```

4. **Restart Raycast**

## 📚 Initial Vault Indexing

After configuration, index your vault to enable search and Q&A:

```bash
# Index everything (unified ChromaDB store)
uv run scripts/reindex.py all

# Check indexing status
uv run scripts/reindex.py status

# Optional: Enrich notes with PARA taxonomy
uv run scripts/enrich_para_taxonomy.py enrich-all --apply
```

## 🎮 Using the MCP Server

Once configured, you can use these tools in your client:

### Available Tools

#### Search & Q&A
- `search_notes` - Vector search across your vault
- `answer_question` - AI-powered Q&A with citations
- `graph_neighbors` - Find related notes
- `get_subgraph` - Extract note relationships

#### Note Management
- `create_note` - Create new notes with auto-enrichment
- `read_note` - Read note content
- `update_note_properties` - Update frontmatter
- `list_notes` - Browse vault contents
- `archive_note` - Move notes to archive

#### Graph Navigation
- `get_backlinks` - Find notes linking to a target
- `get_notes_by_tag` - Find notes by tag

#### Maintenance
- `reindex_vault` - Reindex databases
- `enrich_notes` - Apply PARA taxonomy

### Example Usage in Claude

```
User: Search for notes about machine learning
Claude: [Uses search_notes tool] Found 5 relevant notes...

User: What are my current projects?
Claude: [Uses answer_question tool] Based on your vault, you have 3 active projects...

User: Create a new note for today's meeting
Claude: [Uses create_note tool] Created "Meeting Notes 2024-08-23" with enriched metadata...
```

## 🔍 Troubleshooting

### Common Issues

#### "uv not found"
```bash
# Make sure uv is in your PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Or use full path in configs
"command": "/Users/yourname/.cargo/bin/uv"
```

#### "Server not appearing in client"
- Restart the client application completely
- Check JSON syntax in config files (no trailing commas!)
- Verify the `cwd` path points to the project directory

#### "Permission denied" on macOS
```bash
# Grant terminal/app full disk access
System Preferences → Security & Privacy → Full Disk Access
```

#### "Gemini API errors"
- Verify your API key is valid
- Check you haven't exceeded quota
- Ensure the key is properly quoted in .env file

#### "ChromaDB persistence issues"
```bash
# Clear ChromaDB if corruption occurs
rm -rf .chroma_db
uv run scripts/reindex.py all --full
```

### Testing Individual Components

```bash
# Test unified store
uv run python -c "from src.unified_store import UnifiedStore; print('Unified Store OK')"

# Test MCP tools
uv run python -c "from src.mcp_server import mcp; print(f'Tools: {len(mcp.list_tools())}')"
```

## 📦 Environment Variables

Complete list of supported environment variables:

```bash
# Required
GEMINI_API_KEY=your-key-here

# Paths
OBSIDIAN_VAULT_PATH=/path/to/vault
OBSIDIAN_RAG_CHROMA_DIR=.chroma_db
OBSIDIAN_RAG_COLLECTION=vault_collection

# Chunking Strategy
OBSIDIAN_RAG_CHUNK_STRATEGY=semantic  # or "character"
OBSIDIAN_RAG_SEMANTIC_MIN_CHUNK_SIZE=100
OBSIDIAN_RAG_SEMANTIC_MAX_CHUNK_SIZE=3000
OBSIDIAN_RAG_SEMANTIC_MERGE_THRESHOLD=200

# Models
OBSIDIAN_RAG_EMBEDDING_MODEL=all-MiniLM-L6-v2
OBSIDIAN_RAG_GEMINI_MODEL=gemini-2.5-flash

# Server Settings
MCP_HOST=127.0.0.1
MCP_PORT=8765
MCP_LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
```

## 🚀 Advanced Usage

### Running Multiple Vaults

Create separate configs for each vault:

```json
{
  "mcpServers": {
    "vault-personal": {
      "command": "uv",
      "args": ["run", "graph-rag-mcp-stdio"],
      "env": {
        "OBSIDIAN_VAULT_PATH": "/path/to/personal/vault"
      }
    },
    "vault-work": {
      "command": "uv",
      "args": ["run", "graph-rag-mcp-stdio"],
      "env": {
        "OBSIDIAN_VAULT_PATH": "/path/to/work/vault"
      }
    }
  }
}
```

### Custom Embedding Models

Use any sentence-transformers model:

```bash
OBSIDIAN_RAG_EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
```

### Batch Processing

Process your entire vault:

```bash
# Full PARA enrichment
uv run scripts/enrich_para_taxonomy.py enrich-all --apply --batch-size 50

# Selective enrichment
uv run scripts/enrich_para_taxonomy.py enrich --folder "Projects" --apply
```

## 📚 Resources

- [MCP Protocol Documentation](https://modelcontextprotocol.io)
- [Claude Desktop MCP Guide](https://claude.ai/docs/mcp)
- [Cursor MCP Integration](https://cursor.sh/docs/mcp)
- [Raycast MCP Extension](https://raycast.com/extensions/mcp)
- [uv Documentation](https://docs.astral.sh/uv)

## 🤝 Getting Help

- **Issues**: [GitHub Issues](https://github.com/yourusername/graph-rag-mcp-server/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/graph-rag-mcp-server/discussions)
- **Documentation**: [README.md](README.md)

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

---

**Happy Knowledge Graphing! 🧠✨**