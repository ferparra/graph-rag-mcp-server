---
up: "[[Projects/Personal]]"
tags:
  - project/personal
  - resource/topic/ai
  - status/processing
---
# Executive summary
- A daemon that is being run while using the Obsidian Vault in the parent folder (but do architect the solution for a potential multi-vault setup)
- A sub-directory within the obsidian vault containing strongly-typed python code
- Uses astral `uv` cli for managing packages based on an `uv.lock` and a `pyproject.toml`
	- Using `uv sync` for syncing packages
	- Using `uv run` for running python scripts (e.g. `uv run mypythonscript.py`)
- Content managed with  `git`
- Using neo4j graph database with the following purpose:
	- Two-way tracking of all backlinks between files in the Obsidian Vault
	- Two-way tracking of relevant properties used as modifiers to enable graph meaning
- Using `chromaDB` as the vector DB
- neo4j and chromaDB need to work as synergistically as possible. Use the best features of each in conjunction.
- Expose the capability through an MCP Server that can:
	- Get a list of notes
	- Show the contents of a note
	- Read the file properties one one or many notes
	- Add and modify file properties of one or many notes
	- Respond to natural language questions
		- Retrieval is powered by a lighter model like `gemini-2.5-flash`
		- Answering is powered by whatever model is chosen by the consumer of the MCP Server
	- Archive notes and folders (by moving into archival parent folder)
	- Create and rename folders
	- Add content sections to a note

# Guiding principles
- Prioritise declarative programming than imperative programming
- Use as strongly-typed python code as possible
- Prioritise open-source tools with thriving communities and well-documented SDKs
- Focus is on single-user PKM systems

# Developer guideline
Below is a **developer-grade guideline** you can drop into a repo and start wiring up. It’s **local-first**, file-system based, scales to multi-vault, uses **ChromaDB** for vectors, **DSPy** for retrieval + program structure with **Gemini 2.5 Flash**, and exposes **Graph-RAG** over an **MCP server** (SSE/HTTP or stdio) installable via **Astral uv**.

I keep code **strongly-typed**, minimal, and production-lean. Where I reference non-obvious behaviors/APIs, I cite primary docs.
---
# **0) Environment & Project Layout**

```
# init project
uv init obsidian-rag
cd obsidian-rag

# core deps
uv add "chromadb>=0.5" "sentence-transformers>=3" "dspy-ai>=2" \
       "google-generativeai>=0.7" "pydantic>=2" "networkx>=3" \
       "mcp[cli]" "watchfiles>=0.22" "typer>=0.12" "uvicorn>=0.30" "starlette>=0.39"
```

Why these:

- **ChromaDB**: fast local vector DB, default local ST embedding works out-of-the-box .
- **Sentence-Transformers**: lets you bring your own embedder (e.g., all-MiniLM-L6-v2) ; Chroma’s SentenceTransformerEmbeddingFunction is supported in-tree .
- **DSPy**: declarative programs, retrieval modules, and optimizers; we’ll pair it with Gemini 2.5 Flash via Google’s SDK (or the OpenAI-compat endpoint) .
- **MCP**: official Python SDK w/ **FastMCP**, stdio + SSE/HTTP transports, and uv integration .
  

Recommended repo layout:

```
obsidian-rag/
  src/
    config.py
    fs_indexer.py
    chroma_store.py
    graph_store.py
    dspy_rag.py
    mcp_server.py
  scripts/
    reindex.py
  .env.example
```

---

# **1) Configuration (paths, models, knobs)**

```
# src/config.py
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional

class Settings(BaseModel):
    vaults: List[Path] = Field(default_factory=lambda: [Path.home()/ "Obsidian" / "MainVault"])
    chroma_dir: Path = Path(".chroma_db")
    collection: str = "obsidian_vault"
    embedding_model: str = "all-MiniLM-L6-v2"  # local default is fine [oai_citation:5‡Chroma Docs](https://docs.trychroma.com/guides/embeddings?utm_source=chatgpt.com)
    # Gemini 2.5 Flash
    gemini_model: str = "gemini-2.5-flash"
    gemini_api_key: Optional[str] = None
    # Chunking
    max_chars: int = 1800
    overlap: int = 200

settings = Settings()
```

Populate .env with your GEMINI_API_KEY if using the Google SDK directly.

---

# **2) Filesystem → ChromaDB (indexing Markdown/code/Excalidraw)**

```
# src/fs_indexer.py
from __future__ import annotations
import re, json
from pathlib import Path
from typing import Iterable, Dict, List, Tuple
from dataclasses import dataclass

WIKILINK = re.compile(r"\[\[([^\]]+)\]\]")
FRONTMATTER = re.compile(r"^---\n.*?\n---\n", re.DOTALL)

@dataclass
class NoteDoc:
    id: str
    text: str
    path: Path
    title: str
    tags: List[str]
    links: List[str]
    meta: Dict

def discover_files(vaults: Iterable[Path]) -> Iterable[Path]:
    for root in vaults:
        for p in root.rglob("*"):
            if p.suffix.lower() in {".md", ".markdown", ".txt", ".excalidraw"}:
                yield p

def load_text(path: Path) -> str:
    if path.suffix.lower() == ".excalidraw":
        # index textual labels only
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            labels = []
            for el in data.get("elements", []):
                txt = el.get("text")
                if txt: labels.append(txt)
            return "\n".join(labels) or path.stem
        except Exception:
            return path.stem
    return path.read_text(encoding="utf-8", errors="ignore")

def parse_note(path: Path) -> NoteDoc:
    raw = load_text(path)
    front = FRONTMATTER.findall(raw)
    body = FRONTMATTER.sub("", raw) if front else raw
    links = WIKILINK.findall(body)
    tags = sorted({t.strip("#") for t in re.findall(r"(#\w[\w/-]+)", body)})
    title = path.stem
    meta = {}  # extend by parsing frontmatter yaml if present
    # normalize id as posix path (stable across vaults)
    nid = str(path).replace("\\", "/")
    return NoteDoc(id=nid, text=body.strip(), path=path, title=title, tags=tags, links=links, meta=meta)

def chunk(text: str, max_chars: int, overlap: int) -> List[str]:
    chunks, i, n = [], 0, len(text)
    while i < n:
        j = min(i + max_chars, n)
        chunks.append(text[i:j])
        i = j - overlap
        if i < 0: i = 0
        if i >= n: break
    return chunks
```

Chroma store wrapper:

```
# src/chroma_store.py
from __future__ import annotations
from typing import List, Dict
from chromadb import PersistentClient
from chromadb.api.models.Collection import Collection
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pydantic import BaseModel
from pathlib import Path
from .config import settings
from .fs_indexer import discover_files, parse_note, chunk, NoteDoc

class ChromaStore(BaseModel):
    client_dir: Path
    collection_name: str
    embed_model: str

    def _client(self) -> PersistentClient:
        return PersistentClient(path=str(self.client_dir))

    def _collection(self) -> Collection:
        ef = SentenceTransformerEmbeddingFunction(model_name=self.embed_model)
        client = self._client()
        return client.get_or_create_collection(self.collection_name, embedding_function=ef)

    def reindex(self, vaults: List[Path]) -> int:
        col = self._collection()
        # wipe: comment if you prefer incremental
        # self._client().delete_collection(self.collection_name); col = self._collection()
        ids, docs, metas = [], [], []
        for path in discover_files(vaults):
            nd = parse_note(path)
            if not nd.text: continue
            for k, ch in enumerate(chunk(nd.text, settings.max_chars, settings.overlap)):
                ids.append(f"{nd.id}#chunk={k}")
                docs.append(ch)
                metas.append({"path": str(nd.path), "title": nd.title, "tags": nd.tags})
            if len(ids) >= 512:
                col.add(ids=ids, documents=docs, metadatas=metas); ids, docs, metas = [], [], []
        if ids:
            col.add(ids=ids, documents=docs, metadatas=metas)
        return col.count()

    def query(self, q: str, k: int = 6, where: Dict | None = None) -> List[Dict]:
        col = self._collection()
        res = col.query(query_texts=[q], n_results=k, where=where)
        hits = []
        for i in range(len(res["ids"][0])):
            hits.append({
                "id": res["ids"][0][i],
                "text": res["documents"][0][i],
                "meta": res["metadatas"][0][i],
                "dist": res["distances"][0][i] if "distances" in res else None
            })
        return hits
```

> **Notes**

- > Chroma’s default/local embedding function is MiniLM; swapping to a bigger local ST model is trivial  .
- > Keep a **stable string id** per chunk (path#chunk=k) so you can back-map to files.
  

Batch reindex script:

```
# scripts/reindex.py
import typer
from src.config import settings
from src.chroma_store import ChromaStore

app = typer.Typer()

@app.command()
def all():
    store = ChromaStore(client_dir=settings.chroma_dir,
                        collection_name=settings.collection,
                        embed_model=settings.embedding_model)
    n = store.reindex(settings.vaults)
    typer.echo(f"Indexed {n} chunks into {settings.collection} @ {settings.chroma_dir}")

if __name__ == "__main__":
    app()
```

Run: uv run scripts/reindex.py all

---

# **3) Graph-RAG (lightweight): build a vault graph from links/tags**

```
# src/graph_store.py
from __future__ import annotations
import networkx as nx
from pathlib import Path
from typing import Iterable, List, Dict, Set
from .fs_indexer import discover_files, parse_note, NoteDoc

class GraphStore:
    def __init__(self):
        self.G = nx.Graph()

    def build(self, vaults: Iterable[Path]) -> int:
        self.G.clear()
        for p in discover_files(vaults):
            nd: NoteDoc = parse_note(p)
            nid = nd.id
            self.G.add_node(nid, title=nd.title, path=str(nd.path), tags=nd.tags)
            for link in nd.links:
                # naive resolution: match by stem
                # you can maintain a title->id index for precision
                self.G.add_edge(nid, link)
            for tag in nd.tags:
                self.G.add_edge(nid, f"tag::{tag}")
        return self.G.number_of_nodes()

    def neighbors(self, nid: str, depth: int = 1) -> List[str]:
        seen: Set[str] = {nid}
        frontier = {nid}
        for _ in range(depth):
            nxt = set()
            for u in frontier:
                nxt.update(self.G.neighbors(u))
            frontier = nxt - seen
            seen |= frontier
        return [n for n in seen if n != nid]

    def subgraph_for_seeds(self, seeds: List[str], depth: int = 1) -> Dict:
        nodes = set(seeds)
        for s in seeds:
            nodes.update(self.neighbors(s, depth))
        H = self.G.subgraph(nodes).copy()
        return {
            "nodes": [{"id": n, **self.G.nodes[n]} for n in H.nodes],
            "edges": [{"u": u, "v": v} for u, v in H.edges]
        }
```

This **graph layer** lets you (a) expand context around retrieved chunks and (b) expose neighbors/subgraphs via MCP.

---

# **4) DSPy retrieval program with** 

# **Gemini 2.5 Flash**


There are two clean paths
  
**A. Native Google SDK** (robust and explicit; recommended).
**B. OpenAI-compat endpoint** that some folks use with DSPy’s built-in LM shim .

Below is **A**: DSPy handles retrieval and program structure; Gemini handles generation.

```
# src/dspy_rag.py
from __future__ import annotations
import os
import dspy
import google.generativeai as genai
from typing import List, Dict
from pydantic import BaseModel
from .config import settings
from .chroma_store import ChromaStore

# --- configure Gemini
if settings.gemini_api_key:
    genai.configure(api_key=settings.gemini_api_key)

# DSPy signature: inputs/outputs are typed
class VaultQA(dspy.Signature):
    """Answer user questions grounded STRICTLY in provided vault snippets."""
    question = dspy.InputField()
    context = dspy.InputField(desc="relevant snippets from the vault")
    answer = dspy.OutputField(desc="concise answer with inline citations like [#] per snippet order")

class ChromaRetriever:
    def __init__(self, store: ChromaStore, k: int = 6):
        self.store = store; self.k = k
    def retrieve(self, query: str) -> List[Dict]:
        return self.store.query(query, k=self.k)

class RAGProgram(dspy.Module):
    def __init__(self, retriever: ChromaRetriever, model: str):
        super().__init__()
        self.retriever = retriever
        self.model = model
        self.predict = dspy.Predict(VaultQA)

    def _call_gemini(self, prompt: str) -> str:
        mdl = genai.GenerativeModel(self.model)
        resp = mdl.generate_content(prompt, generation_config=genai.types.GenerationConfig(
            temperature=0.1, max_output_tokens=900))
        return resp.text or ""

    def forward(self, question: str) -> dspy.Prediction:
        hits = self.retriever.retrieve(question)
        ctx_parts = []
        for i, h in enumerate(hits, start=1):
            ctx_parts.append(f"[{i}] {h['meta'].get('title')} ({h['meta'].get('path')})\n{h['text']}\n")
        ctx = "\n".join(ctx_parts) if ctx_parts else "NO CONTEXT FOUND"
        # Let DSPy do IO structuring; use Gemini for text
        prompt = f"""You are a strict RAG assistant. Use ONLY these snippets to answer.
Snippets:
{ctx}

Question: {question}
Rules: Answer concisely. If insufficient evidence, say so. Add bracketed citations [#] to the snippets used."""
        out = self._call_gemini(prompt)
        return self.predict(question=question, context=ctx, answer=out)

def build_rag() -> RAGProgram:
    store = ChromaStore(client_dir=settings.chroma_dir,
                        collection_name=settings.collection,
                        embed_model=settings.embedding_model)
    retriever = ChromaRetriever(store, k=6)
    return RAGProgram(retriever, model=settings.gemini_model)
```

> If you prefer **B (OpenAI-compat)**, the DSPy article shows configuring DSPy’s LM with the Gemini OpenAI-compatible base and model name .

---

# **5) Expose retrieval + Graph-RAG via** 

# **MCP**

#  **(stdio or SSE/HTTP)**


We’ll use **FastMCP** for typed tools and then either:

- run **stdio** (ideal for Claude Desktop), or
- mount **SSE/HTTP** under Starlette/uvicorn (works with tools that expect HTTP; note SSE is being superseded by _Streamable HTTP_ in the SDK) .

```
# src/mcp_server.py
from __future__ import annotations
from typing import List, Optional, Dict
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession
from pydantic import BaseModel
from .dspy_rag import build_rag
from .graph_store import GraphStore
from .config import settings

class AppState(BaseModel):
    rag = build_rag()
    graph = GraphStore()

mcp = FastMCP("Obsidian-RAG")

# --- lifecycle: build graph once on startup
@mcp.lifespan
async def lifespan(server: FastMCP):
    state = AppState()
    state.graph.build(settings.vaults)
    yield state

# ---- TOOLS ----

@mcp.tool()
def search_notes(query: str, k: int = 6, where: Optional[Dict]=None,
                 ctx: Context[ServerSession, AppState] = None) -> List[Dict]:
    """Vector search across vault chunks (ChromaDB)."""
    return ctx.request_context.lifespan_context.rag.retriever.store.query(query, k=k, where=where)

@mcp.tool()
def answer_question(question: str,
                    ctx: Context[ServerSession, AppState] = None) -> Dict:
    """RAG answer using Gemini 2.5 Flash grounded in vault snippets."""
    prog = ctx.request_context.lifespan_context.rag
    pred = prog(question=question)
    return {"answer": pred.answer, "used_context": pred.context}

@mcp.tool()
def graph_neighbors(note_id_or_title: str, depth: int = 1,
                    ctx: Context[ServerSession, AppState] = None) -> Dict:
    """Return neighbors around a note (or tag::<name>) up to depth."""
    gs = ctx.request_context.lifespan_context.graph
    # naive: treat input as exact node id; in practice map title->id index
    return {"subgraph": gs.subgraph_for_seeds([note_id_or_title], depth)}

# --- entrypoints ---

def run_stdio():
    # perfect for Claude Desktop “command + args” config
    mcp.run_stdio()

def app_sse():
    # mountable SSE app (SDK shows SSE -> Streamable HTTP migration) [oai_citation:11‡GitHub](https://github.com/modelcontextprotocol/python-sdk)
    return mcp.sse_app()

def app_streamable_http():
    # preferred modern transport (mount under Starlette)
    return mcp.streamable_http_app()
```

Minimal **stdio** runner:

```
uv run python -c "from src.mcp_server import run_stdio; run_stdio()"
```

**Claude Desktop** config (example) points to the above command; the “Obsidian Intelligent MCP Server” shows this pattern and envs like OBSIDIAN_VAULT_PATH, MCP_LOG_LEVEL, etc., for reference .


Mount **Streamable HTTP** (or SSE) under uvicorn:

```
# main_http.py
from starlette.applications import Starlette
from starlette.routing import Mount
from src.mcp_server import app_streamable_http
app = Starlette(routes=[Mount("/", app=app_streamable_http())])
```

```
uv run uvicorn main_http:app --reload --port 8765
```

> The MCP SDK docs show **uv-native install/run**, **FastMCP quickstart**, and how to mount **SSE** and **Streamable HTTP** servers under Starlette/uvicorn .

---

# **6) Hot-reload indexing (optional, nice in big vaults)**

```
# Watch for file changes and incrementally upsert to Chroma
# scripts/reindex_watch.py
from watchfiles import awatch
import asyncio
from pathlib import Path
from src.config import settings
from src.chroma_store import ChromaStore
from src.fs_indexer import parse_note, chunk

async def main():
    store = ChromaStore(client_dir=settings.chroma_dir,
                        collection_name=settings.collection,
                        embed_model=settings.embedding_model)
    print("Watching for changes...")
    async for changes in awatch(*settings.vaults):
        ids, docs, metas = [], [], []
        for _, path_str in changes:
            p = Path(path_str)
            if p.suffix.lower() not in {".md",".markdown",".txt",".excalidraw"}: continue
            nd = parse_note(p)
            for k, ch in enumerate(chunk(nd.text, settings.max_chars, settings.overlap)):
                ids.append(f"{nd.id}#chunk={k}")
                docs.append(ch)
                metas.append({"path": str(nd.path), "title": nd.title, "tags": nd.tags})
        if ids:
            store._collection().upsert(ids=ids, documents=docs, metadatas=metas)
            print(f"Upserted {len(ids)} chunks")

if __name__ == "__main__":
    asyncio.run(main())
```

Run: uv run python scripts/reindex_watch.py

---

# **7) Usage flow (end-to-end)**

1. **Index** your vault(s):
    
    uv run scripts/reindex.py all
    
2. **Run MCP (stdio)** for Claude/Cursor:
    
    uv run python -c "from src.mcp_server import run_stdio; run_stdio()"
    
    Configure in Claude Desktop (command + args). The MCP Python SDK README shows uv run mcp install server.py patterns and integration notes .
    
3. **HTTP transport** (optional):
    
    uv run uvicorn main_http:app --port 8765
    
4. Ask Claude:
    
    “Use the obsidian-rag MCP server’s answer_question tool to answer: _What did I decide about RAG chunk sizes last week?_”
    

---

# **8) Scaling, Multi-vault, and Storage**

- **Multi-vault**: add all vault roots to Settings.vaults. Keep a **single collection** and include meta.vault=<name> if you want per-vault filters, or one collection per vault (simpler operationally, more clients).
    
- **Large scale**: Chroma handles large local collections. For distributed/remote, swap to **Qdrant/Weaviate** and keep the exact same retriever shape (Chroma wrapper is the only file to change). Qdrant is widely used for massive semantic indexes .
    
- **Embeddings**: MiniLM is fast/local; upgrade to all-mpnet-base-v2 or gtr-t5-large on beefier hardware . Keep chunk sizes ~1.5–2k chars with 10–15% overlap as a good vault default.
    
- **Sync**: iCloud/Syncthing/IPFS handle files; rebuild/index on each node on change. The vector index lives under .chroma_db/—don’t sync that unless you coordinate exact Chroma versions.
    

---

# **9) Security & Failure Modes**

- **Local-only** by default (no external calls except Gemini if you configure it). The Obsidian-focused MCP servers emphasize local processing & vault scoping .
    
- **Path traversal**: enforce a **vault allow-list** and reject paths outside it when you later add write tools.
    
- **SSE vs Streamable HTTP**: the SDK notes SSE is being superseded; prefer **Streamable HTTP** for new HTTP integrations .
    
- **Graceful degenerate mode**: If no snippets are retrieved, respond with “insufficient evidence” (as shown in prompt logic).
    

---

## **References (key, non-obvious claims/APIs)**

- **MCP Python SDK**: FastMCP quickstart, uv usage, stdio + SSE/Streamable HTTP mounting examples .
    
- **Obsidian MCP (reference design)**: vault tools & local embeddings; shows how Claude config points to a Python command and env vars .
    
- **Chroma default ST embeddings & usage** ; ST embedding function in-tree .
    
- **DSPy + Gemini 2.5 Flash**: patterns for wiring Gemini to DSPy and OpenAI-compat alternative .
    
- **Agentic RAG tutorials** (retriever tools, decision to retrieve) for broader design ideas .
    

---

## **Quick sanity checks you can run**

```
# 1) index a tiny test vault
echo "# Test" > ~/Obsidian/MainVault/Test.md
uv run scripts/reindex.py all

# 2) ask the RAG program directly (bypass MCP)
uv run python - <<'PY'
from src.dspy_rag import build_rag
prog = build_rag()
print(prog(question="What notes mention 'Test'?").answer)
PY

# 3) run MCP (stdio) and connect from Claude Desktop
uv run python -c "from src.mcp_server import run_stdio; run_stdio()"
```

If anything errors, the probable culprits are: missing GEMINI_API_KEY, a typo’d vault path, or a Starlette/uvicorn import mismatch on older Python.