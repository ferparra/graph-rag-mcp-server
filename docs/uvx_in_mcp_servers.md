---
enrichment_model: gemini-2.5-flash
enrichment_version: '1.0'
key_concepts:
- uvx for MCP Server Hosting Isolated and Reproducible Server Execution Dependency
  and Version Pinning with uvx Running Servers from Git and with Plugins Comparison
  of uvx vs. uv tool install
last_enriched: '2025-09-05T15:03:59.330832'
para_category: graph-rag-mcp-server
para_confidence: 0.9
para_reasoning: The note is a "Developer Playbook" providing actionable steps for
  hosting and operating specific servers, and its file path explicitly places it within
  a directory named graph-rag-mcp-server. This indicates it's an operational guide
  or documentation for a specific software development project.
para_type: project
related_topics:
- uv (tool) Python virtual environments Server deployment strategies Multi-Component
  Protocol (MCP) CI/CD pipelines Isolated development environments uv tool install
tags:
- dev/ops/deployment
- dev/playbook
- org/astral
- para/project
- para/project/graph-rag-mcp-server
- tech/python/dependency-management
- tech/server/mcp
- tool/uvx
---

# Hosting MCP Servers with `uvx` (Astral uv) — Developer Playbook

A compact, MECE and production-ready guide to run, pin, ship, and operate MCP servers using `uvx` (aka `uv tool run`). Includes copy/paste commands, client wiring (Claude/Cursor), reproducibility, CI, and troubleshooting.

---

## 0) TL;DR (Copy/Paste)

```bash
# Run a published MCP server (SQLite example)
uvx mcp-server-sqlite --db-path ~/data/app.db

# Pin the exact server *and* Python
uvx --python 3.13 mcp-server-sqlite@1.5.0 --db-path ~/data/app.db

# If CLI name ≠ package name, specify the source package explicitly
uvx --from 'mcp-sqlite-server==1.5.0' mcp-server-sqlite --db-path ~/data/app.db

# Run a server straight from Git (branch/tag/commit)
uvx --from git+https://github.com/org/repo@v1.2.3 my-mcp-entrypoint --flag foo

# Add a plugin/extra alongside the tool at runtime
uvx --with my-mcp-plugin==0.3.2 my-mcp-entrypoint --flag foo

# Graph RAG MCP Server (from GitHub)
uvx --python 3.13 --from git+https://github.com/ferparra/graph-rag-mcp-server graph-rag-mcp-stdio
```

---

## 1) What `uvx` Does (and When to Use It)

- **One-shot, isolated runner.** Resolves deps, creates an isolated env, caches artifacts, runs the CLI. No global installs.
- **Perfect for MCP.** Launch servers directly from client configs (Claude/Cursor/etc.) without managing venvs.
- **Use `uv tool install`** only when you want a persistent PATH binary. Otherwise prefer `uvx` for hermetic runs.
- **Precision pinning.** Pin the command (`@x.y.z`), interpreter (`--python 3.13`), or the source (`--from 'pkg==x.y.z'`).

---

## 2) Make Your MCP Server “uvx-Runnable”

### 2.1 Expose a Console Script
Add a CLI entry point so `uvx` can address your server by name.

```toml
# pyproject.toml
[project]
name = "my-mcp-server"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
  "mcp>=1.2",   # official MCP Python SDK
]

[project.scripts]
my-mcp = "my_mcp_server.__main__:main"
```

### 2.2 Implement a Stdio MCP Server (No stdout logging)
MCP stdio uses **stdout** for JSON-RPC. Send logs to **stderr** or a file.

```python
# my_mcp_server/__main__.py
import sys
import logging
from mcp.server.fastmcp import FastMCP

# Log to stderr to avoid corrupting MCP stdio on stdout
logging.basicConfig(stream=sys.stderr, level=logging.INFO)

mcp = FastMCP(name="My MCP Server")

@mcp.tool()
async def ping() -> str:
    return "pong"

def main():
    # Never print to stdout here; MCP needs stdout for protocol messages.
    mcp.run()
```

### 2.3 Run It with `uvx`
```bash
uvx my-mcp
uvx --python 3.13 my-mcp@0.1.0
uvx --from 'my-mcp-server==0.1.0' my-mcp
```

---

## 3) Wire Into MCP Clients (Claude Desktop, Cursor, etc.)

All clients follow a common pattern: `"command": "uvx"`, `"args": ["<entrypoint>", ...]`, optional `"env"`.

### 3.1 Claude Desktop (`claude_desktop_config.json`)
```json
{
  "mcpServers": {
    "My Server": {
      "command": "uvx",
      "args": ["my-mcp", "--mode", "prod"],
      "env": { "API_KEY": "xxxxx" }
    },
    "SQLite": {
      "command": "uvx",
      "args": ["mcp-server-sqlite@1.5.0", "--db-path", "~/data/app.db"]
    }
  }
}
```
**Notes**
- Restart Claude Desktop after config changes.
- Logs (macOS): `~/Library/Logs/Claude/mcp.log` (adjust path per OS).

### 3.2 Cursor (`mcp.json`)
```json
{
  "mcpServers": {
    "My Server": {
      "command": "uvx",
      "args": ["my-mcp", "--flag", "value"],
      "env": { "API_TOKEN": "xxxxx" }
    }
  }
}
```

### 3.3 Env/Secrets
- Some clients do **not** inherit your shell env.
- Prefer explicit `"env"` entries in the client config or use the client’s secret manager if provided.

---

## 4) Reproducibility & Pinning

**Goals:** deterministic resolution across machines, fast cold-starts in CI, no “works on my machine.”

```bash
# Pin tool version by command alias (when command==package)
uvx my-mcp@0.1.0

# Pin when command ≠ package
uvx --from 'my-mcp-server==0.1.0' my-mcp

# Pin Python interpreter (install once on agents; see §5.2)
uvx --python 3.13 my-mcp@0.1.0

# Add optional plugins/extras at runtime
uvx --with my-mcp-plugin==0.3.2 my-mcp

# Install from Git (tag/branch/commit)
uvx --from git+https://github.com/org/repo@v1.2.3 my-mcp
```

**Best Practices**
- Prefer **exact pins** (`==` / `@x.y.z`) for servers and plugins.
- Pin Python via `--python` to avoid ABI drift.
- Document the **single source of truth command** in your runbook (see §9).

---

## 5) Ops: Caches, CI, and Tooling

### 5.1 uv Cache Management
```bash
# Show cache
uv cache dir

# Clean all cached artifacts
uv cache clean

# Prune aggressively in CI (skip interactive prompts)
uv cache prune --ci
```

### 5.2 Preinstall Python on Build Agents
```bash
uv python install 3.13
uvx --python 3.13 my-mcp@0.1.0  # now deterministic across agents
```

### 5.3 GitHub Actions (Example)
```yaml
name: mcp-ci
on: [push, pull_request]
jobs:
  run-mcp:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      # Setup uv (official action)
      - uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true

      # Preinstall a precise Python for consistent wheels
      - run: uv python install 3.13

      # Smoke-test the server’s CLI entrypoint
      - run: uvx --python 3.13 my-mcp@0.1.0 --help
```

---

## 6) Troubleshooting (Triage → Fix)

**Symptom → Likely Cause → Fix**

- **`command not found`**
  - The CLI name isn’t exported or doesn’t match the package.
  - **Fix:** Use `--from '<pkg spec>' <command>` or add `[project.scripts]` entry.

- **Version drift between machines**
  - Unpinned version or Python mismatch.
  - **Fix:** `my-mcp@x.y.z` and `--python 3.12`. Document the exact command.

- **Server exits immediately / protocol errors**
  - Accidental `print()` or logging to **stdout**.
  - **Fix:** Route logs to **stderr** or a file; keep stdout clean for MCP JSON-RPC.

- **Client can’t see env vars**
  - Clients may not inherit shell env.
  - **Fix:** Provide `"env": {...}` in client config or use the client’s secret store.

- **Slow cold start in CI**
  - Empty cache; Python version resolution each run.
  - **Fix:** Cache uv artifacts; preinstall Python (`uv python install 3.12`); pin deps.

---

## 7) Security & Supply Chain Hygiene

- **Pin everything**: tool, plugins, and Python version; use Git tags/commits for forks.
- **Audit transitive deps** periodically (e.g., `uv export requirements` then scan).
- **Principle of least privilege**: only expose required tools/resources via your MCP server.
- **Secrets**: inject via client’s secret manager or minimal `"env"`; avoid committing creds.

---

## 8) End-to-End Examples

### 8.1 Shipping Your Own Server via PyPI

**`pyproject.toml`**
```toml
[project]
name = "my-mcp-server"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["mcp>=1.2"]

[project.scripts]
my-mcp = "my_mcp_server.__main__:main"
```

**`__main__.py`**
```python
import sys, logging
from mcp.server.fastmcp import FastMCP
logging.basicConfig(stream=sys.stderr, level=logging.INFO)

mcp = FastMCP(name="My MCP Server")

@mcp.tool()
async def health() -> str:
    return "ok"

def main():
    mcp.run()
```

**Run**
```bash
uvx --python 3.13 my-mcp@0.1.0
```

**Claude config**
```json
{
  "mcpServers": {
    "My Server": {
      "command": "uvx",
      "args": ["my-mcp@0.1.0", "--mode", "prod"]
    }
  }
}
```

### 8.2 Using a Community Server (SQLite) with Pinning

**Claude config**
```json
{
  "mcpServers": {
    "sqlite": {
      "command": "uvx",
      "args": ["mcp-server-sqlite@1.5.0", "--db-path", "~/data/app.db"]
    }
  }
}
```

**Local smoke test**
```bash
uvx mcp-server-sqlite@1.5.0 --db-path ./test.db --help
```

---

## 9) Minimal Runbook (Drop-in Template)

**Purpose:** Keep a single source of truth for how your team invokes and debugs the server.

```md
# My MCP Server — Runbook

## Command of Record
uvx --python 3.13 my-mcp@0.1.0 --mode prod

## Client Wiring
- Claude Desktop: edit `claude_desktop_config.json`
- Cursor: edit `mcp.json`

## Logs
- Claude Desktop (macOS): ~/Library/Logs/Claude/mcp.log
- Server logs: stderr (redirect via supervisor if needed)

## Health Check
uvx --python 3.13 my-mcp@0.1.0 --help

## Common Issues
- Protocol errors: check stdout pollution (must be clean).
- Env vars missing: configure `"env"` in client JSON.
- Version mismatch: confirm pins match the Runbook command.

## CI
- Preinstall Python: `uv python install 3.12`
- Cache uv: `astral-sh/setup-uv@v3` with `enable-cache: true`

## Security
- Pins: command, plugins, Python version.
- Rotate secrets quarterly; validate no secrets in repo.
```

---

## 10) Quick Checklist

- [ ] CLI entry point defined in `[project.scripts]`
- [ ] No stdout logging; stderr/file only
- [ ] `uvx` command pinned: tool `@x.y.z` + `--python 3.13`
- [ ] Client config uses `"command":"uvx"` with correct `"args"`
- [ ] Secrets injected via client config or secret manager
- [ ] CI: preinstall Python, cache uv, smoke test entrypoint
- [ ] Runbook published with a **single** command of record

---
```diff
# Pro Tip
+ For local iteration on unreleased fixes, prefer a Git source:
  uvx --from git+https://github.com/you/my-mcp-server@main my-mcp
```

---

## A) Graph RAG MCP Server (This Repo) via `uvx`

Use `uvx` to run the server hermetically from source during development or from a tag/release in production.

- Console scripts: `graph-rag-mcp-stdio`, `graph-rag-mcp-http` (see `pyproject.toml: [project.scripts]`).
- Python: requires `>=3.13`. Prefer pinning with `--python 3.13` for consistency.

Local dev (from this repo):

```bash
# Stdio mode (Claude Desktop)
uvx --python 3.13 --from . graph-rag-mcp-stdio

# HTTP mode (Cursor/Raycast)
uv run graph-rag-mcp-http  # HTTP isn’t stdio-bound; uv run is fine here
```

Claude Desktop config (installed by `install.py`):

```json
{
  "mcpServers": {
    "graph-rag-obsidian": {
      "command": "uvx",
      "args": ["--python", "3.13", "--from", ".", "graph-rag-mcp-stdio"],
      "cwd": "/path/to/graph-rag-mcp-server",
      "env": {
        "GEMINI_API_KEY": "...",
        "OBSIDIAN_RAG_VAULTS": "/path/to/vault"
      }
    }
  }
}
```

From a published package or Git tag:

```bash
# From PyPI (if published)
uvx --python 3.13 graph-rag-mcp-stdio@0.1.0

# From Git (branch/tag/commit), no local checkout needed
uvx --python 3.13 --from git+https://github.com/yourusername/graph-rag-mcp-server@v0.1.0 graph-rag-mcp-stdio
```

Important: The server now avoids stdout logging in stdio mode. All logs go to stderr so MCP JSON-RPC remains clean.

If you want, specify your server’s shape (SQLite/filesystem/HTTP integrations), and we can snap out a tailored `pyproject.toml` + `__main__.py` + client configs matching your exact flags and env layout.