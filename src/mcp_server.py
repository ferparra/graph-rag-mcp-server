from __future__ import annotations
import sys
import logging
import frontmatter
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastmcp import FastMCP
from pydantic import BaseModel
from .dspy_rag import VaultSearcher
from .unified_store import UnifiedStore
from .fs_indexer import parse_note, is_protected_test_content
from .config import settings

# Configure logging to stderr to avoid corrupting MCP stdio on stdout
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger("graph_rag_mcp")

class AppState:
    def __init__(self):
        # Initialize unified store combining vector search and graph capabilities
        self.unified_store = UnifiedStore(
            client_dir=settings.chroma_dir,
            collection_name=settings.collection,
            embed_model=settings.embedding_model
        )
        logger.info("Connected to unified ChromaDB store: %s", settings.chroma_dir)
        
        # Initialize searcher with unified store
        self.searcher = VaultSearcher(unified_store=self.unified_store)

app_state = AppState()
mcp = FastMCP("Graph-RAG Obsidian Server")

class SearchResult(BaseModel):
    hits: List[Dict]
    total_results: int
    query: str

class AnswerResult(BaseModel):
    question: str
    answer: str
    context: str
    success: bool

class NoteInfo(BaseModel):
    id: str
    title: str
    path: str
    tags: List[str]
    links: List[str]
    content: str
    frontmatter: Dict

class GraphResult(BaseModel):
    nodes: List[Dict]
    edges: List[Dict]
    stats: Optional[Dict] = None

@mcp.tool()
def search_notes(
    query: str,
    k: int = 6,
    vault_filter: Optional[str] = None,
    tag_filter: Optional[str] = None
) -> SearchResult:
    """Vector search across vault chunks using ChromaDB."""
    where = {}
    if vault_filter:
        where["vault"] = {"$eq": vault_filter}
    if tag_filter:
        where["tags"] = {"$contains": tag_filter}
    
    where_clause = where if where else None
    hits = app_state.searcher.search(query, k=k, where=where_clause)
    
    return SearchResult(
        hits=hits,
        total_results=len(hits),
        query=query
    )

@mcp.tool()
def answer_question(
    question: str,
    vault_filter: Optional[str] = None,
    tag_filter: Optional[str] = None
) -> AnswerResult:
    """RAG-powered Q&A using Gemini 2.5 Flash grounded in vault snippets."""
    where = {}
    if vault_filter:
        where["vault"] = {"$eq": vault_filter}
    if tag_filter:
        where["tags"] = {"$contains": tag_filter}
    
    where_clause = where if where else None
    result = app_state.searcher.ask(question, where=where_clause)
    
    return AnswerResult(**result)

@mcp.tool()
def graph_neighbors(
    note_id_or_title: str,
    depth: int = 1,
    relationship_types: Optional[List[str]] = None
) -> GraphResult:
    """Get neighboring notes in the graph up to specified depth."""
    neighbors = app_state.unified_store.get_neighbors(
        note_id_or_title, 
        depth=depth, 
        relationship_types=relationship_types
    )
    return GraphResult(
        nodes=neighbors,
        edges=[],
        stats={"neighbor_count": len(neighbors)}
    )

@mcp.tool()
def get_subgraph(
    seed_notes: List[str],
    depth: int = 1
) -> GraphResult:
    """Get a subgraph containing seed notes and their neighbors."""
    subgraph = app_state.unified_store.get_subgraph(seed_notes, depth)
    return GraphResult(**subgraph)

@mcp.tool()
def list_notes(
    limit: Optional[int] = 50,
    vault_filter: Optional[str] = None
) -> List[Dict]:
    """List all notes in the vault with metadata."""
    notes = app_state.unified_store.get_all_notes(limit=limit)
    
    if vault_filter:
        notes = [n for n in notes if n.get("meta", {}).get("vault") == vault_filter]
    
    return notes

def _load_note(note_path: str) -> NoteInfo:
    """Helper function to load a note and return NoteInfo."""
    path = Path(note_path)
    
    if not path.exists():
        for vault in settings.vaults:
            potential_path = vault / note_path
            if potential_path.exists():
                path = potential_path
                break
        else:
            raise FileNotFoundError(f"Note not found: {note_path}")
    
    note = parse_note(path)
    
    return NoteInfo(
        id=note.id,
        title=note.title,
        path=str(note.path),
        tags=note.tags,
        links=note.links,
        content=note.text,
        frontmatter=note.frontmatter
    )

@mcp.tool()
def read_note(note_path: str) -> NoteInfo:
    """Read the full content of a note by path."""
    return _load_note(note_path)

@mcp.tool()
def get_note_properties(note_path: str) -> Dict:
    """Get frontmatter properties of a note."""
    note_info = _load_note(note_path)
    return note_info.frontmatter

@mcp.tool()
def update_note_properties(
    note_path: str,
    properties: Dict[str, Any],
    merge: bool = True
) -> Dict:
    """Update frontmatter properties of a note."""
    path = Path(note_path)
    
    if not path.exists():
        for vault in settings.vaults:
            potential_path = vault / note_path
            if potential_path.exists():
                path = potential_path
                break
        else:
            raise FileNotFoundError(f"Note not found: {note_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        post = frontmatter.load(f)
    
    if merge:
        post.metadata.update(properties)
    else:
        post.metadata = properties
    
    with open(path, 'w', encoding='utf-8') as f:
        rendered = frontmatter.dumps(post)
        if isinstance(rendered, bytes):
            rendered = rendered.decode('utf-8')
        f.write(rendered)
    
    note = parse_note(path)
    app_state.unified_store.upsert_note(note)
    
    return post.metadata

@mcp.tool()
def archive_note(
    note_path: str,
    archive_folder: Optional[str] = None
) -> str:
    """Move a note to the archive folder."""
    archive_name = archive_folder or settings.archive_folder
    path = Path(note_path)
    
    if not path.exists():
        for vault in settings.vaults:
            potential_path = vault / note_path
            if potential_path.exists():
                path = potential_path
                break
        else:
            raise FileNotFoundError(f"Note not found: {note_path}")
    
    vault_root = None
    for vault in settings.vaults:
        try:
            path.relative_to(vault)
            vault_root = vault
            break
        except ValueError:
            continue
    
    if not vault_root:
        raise ValueError(f"Note {note_path} not in any configured vault")
    
    archive_dir = vault_root / archive_name
    archive_dir.mkdir(exist_ok=True)
    
    new_path = archive_dir / path.name
    path.rename(new_path)
    
    app_state.unified_store.delete_note(str(path.relative_to(vault_root)))
    
    return str(new_path)

@mcp.tool()
def create_note(
    title: str,
    content: str = "",
    folder: Optional[str] = None,
    tags: Optional[List[str]] = None,
    para_type: Optional[str] = None,
    enrich: bool = True
) -> Dict[str, Any]:
    """Create a new Obsidian note with enriched frontmatter.
    
    Args:
        title: Note title (will be used as filename)
        content: Initial note content (markdown)
        folder: Folder path within vault (e.g., "Projects", "00 Inbox")
        tags: Initial tags to add
        para_type: PARA type hint (project/area/resource/archive)
        enrich: Whether to apply AI enrichment for PARA classification
    
    Returns:
        Dict with created note path and metadata
    """
    import frontmatter
    from datetime import datetime
    
    # Sanitize title for filename
    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
    filename = f"{safe_title}.md"
    
    # Determine folder
    vault_root = settings.vaults[0]
    if folder:
        note_folder = vault_root / folder
        note_folder.mkdir(parents=True, exist_ok=True)
    else:
        # Default to inbox if it exists, otherwise root
        inbox = vault_root / "00 Inbox"
        if inbox.exists():
            note_folder = inbox
        else:
            note_folder = vault_root
    
    note_path = note_folder / filename
    
    # Check if file already exists
    if note_path.exists():
        # Add timestamp to make unique
        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename: str = f"{safe_title}_{timestamp}.md"
        note_path: Path = note_folder / filename
    
    # Prepare single, valid frontmatter by merging any YAML blocks from content
    # and tool-provided defaults/overrides into one block.
    import re
    import yaml

    now_iso = datetime.now().isoformat()

    # Helper: normalize tags to a list[str]
    def _normalize_tags(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value.lstrip('#')]
        if isinstance(value, list):
            out: list[str] = []
            for t in value:
                if isinstance(t, str):
                    out.append(t.lstrip('#'))
            return out
        return []

    # Helper: merge two tag lists with order preserved and no duplicates
    def _merge_tags(a: list[str], b: list[str]) -> list[str]:
        seen = set()
        merged: list[str] = []
        for t in (a + b):
            if t not in seen:
                seen.add(t)
                merged.append(t)
        return merged

    # Start by parsing any frontmatter already present in the content
    base_post = frontmatter.loads(content)
    merged_meta: Dict[str, Any] = dict(base_post.metadata or {})
    body: str = base_post.content or ""

    # Additionally, some content may (incorrectly) contain another YAML block
    # immediately after the first one. Merge any additional leading YAML blocks.
    yaml_block_pattern = re.compile(r"^\s*---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
    while True:
        m = yaml_block_pattern.match(body)
        if not m:
            break
        block_text = m.group(1)
        try:
            loaded = yaml.safe_load(block_text)
        except Exception:
            # Not valid YAML; treat as content
            break
        if isinstance(loaded, dict) and loaded:
            # Merge this block and remove it from the body
            for k, v in loaded.items():
                if k == "tags":
                    existing = _normalize_tags(merged_meta.get("tags"))
                    merged_meta["tags"] = _merge_tags(existing, _normalize_tags(v))
                else:
                    # Prefer earlier keys unless we explicitly override later
                    if k not in merged_meta:
                        merged_meta[k] = v
            body = body[m.end():]
        else:
            # Empty or non-dict YAML (likely a horizontal rule usage) — stop
            break

    # Defaults: created/modified timestamps
    if "created" not in merged_meta:
        merged_meta["created"] = now_iso
    # Always set modified to now for a newly created note
    merged_meta["modified"] = now_iso

    # Merge user-provided tags and PARA info
    if tags:
        merged_meta["tags"] = _merge_tags(_normalize_tags(merged_meta.get("tags")), _normalize_tags(tags))
    # PARA type + tag
    if para_type and para_type in ["project", "area", "resource", "archive"]:
        merged_meta["para_type"] = para_type
        merged_meta["tags"] = _merge_tags(_normalize_tags(merged_meta.get("tags")), [f"para/{para_type}"])

    # Create the note with a single consolidated frontmatter block
    post = frontmatter.Post(body, handler=None, **merged_meta)
    
    # Write the initial note
    with open(note_path, 'w', encoding='utf-8') as f:
        rendered = frontmatter.dumps(post)
        if isinstance(rendered, bytes):
            rendered = rendered.decode('utf-8')
        f.write(rendered)
    
    # Index the note
    note = parse_note(note_path)
    app_state.unified_store.upsert_note(note)
    
    # Apply enrichment if requested
    enriched_metadata = {}
    if enrich and content.strip():  # Only enrich if there's content
        try:
            # Import enrichment module dynamically since it's in scripts directory
            import importlib.util
            script_path = Path(__file__).parent.parent / "scripts" / "enrich_para_taxonomy.py"
            spec = importlib.util.spec_from_file_location("enrich_para_taxonomy", script_path)
            ParaTaxonomyEnricher = None
            if spec and spec.loader:
                enrich_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(enrich_module)
                ParaTaxonomyEnricher = enrich_module.ParaTaxonomyEnricher
            
            # Initialize enricher
            if ParaTaxonomyEnricher is None:
                raise ImportError("Could not import ParaTaxonomyEnricher")
            enricher = ParaTaxonomyEnricher()
            
            # Enrich the newly created note
            result = enricher.enrich_note_properties(str(note_path), dry_run=False)
            
            if result:
                enriched_metadata = result.get('enriched_properties', {})
                
                # Re-index after enrichment
                updated_note = parse_note(note_path)
                app_state.unified_store.upsert_note(updated_note)
        except Exception as e:
            # Enrichment failed, but note was still created
            enriched_metadata: dict[str, str] = {"enrichment_error": str(e)}
    
    # Return created note info
    final_metadata = dict(merged_meta)
    if enriched_metadata:
        final_metadata.update(enriched_metadata)
    
    return {
        "path": str(note_path),
        "title": title,
        "folder": str(note_folder.relative_to(vault_root)),
        "metadata": final_metadata,
        "enriched": enrich and bool(enriched_metadata),
        "message": f"Note created: {note_path.name}"
    }

@mcp.tool()
def create_folder(folder_path: str, vault_name: Optional[str] = None) -> str:
    """Create a new folder in the vault."""
    if vault_name:
        vault_root = None
        for vault in settings.vaults:
            if vault.name == vault_name:
                vault_root = vault
                break
        if not vault_root:
            raise ValueError(f"Vault {vault_name} not found")
    else:
        vault_root = settings.vaults[0]
    
    folder = vault_root / folder_path
    folder.mkdir(parents=True, exist_ok=True)
    
    return str(folder)

@mcp.tool()
def add_content_to_note(
    note_path: str,
    content: str,
    position: str = "end"
) -> str:
    """Add content to an existing note."""
    path = Path(note_path)
    
    if not path.exists():
        for vault in settings.vaults:
            potential_path = vault / note_path
            if potential_path.exists():
                path = potential_path
                break
        else:
            raise FileNotFoundError(f"Note not found: {note_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        post = frontmatter.load(f)
    
    if position == "start":
        post.content = content + "\n\n" + post.content
    else:
        post.content = post.content + "\n\n" + content
    
    with open(path, 'w', encoding='utf-8') as f:
        rendered = frontmatter.dumps(post)
        if isinstance(rendered, bytes):
            rendered = rendered.decode('utf-8')
        f.write(rendered)
    
    note = parse_note(path)
    app_state.unified_store.upsert_note(note)
    
    return f"Content added to {note_path}"

@mcp.tool()
def update_note_section(
    note_path: str,
    section_heading: str,
    new_content: str,
    heading_level: Optional[int] = None
) -> Dict[str, Any]:
    """Replace the content of a specific Markdown section in-place.

    This updates only the body of a section identified by its heading, keeping the
    heading itself intact. The rest of the note remains unchanged. If the section
    is not found, an error is raised. Uses ATX-style headings ("#", "##", etc.).

    Args:
        note_path: Path to the note (absolute or vault-relative).
        section_heading: The visible heading text to match (case-insensitive).
        new_content: Replacement body for the section (Markdown). Inserted in-place.
        heading_level: Optional exact heading level to match (1-6). If omitted, the
            first matching heading text is used regardless of level.

    Returns:
        Dict with path, section name, and a short message.
    """
    import re

    path = Path(note_path)
    if not path.exists():
        for vault in settings.vaults:
            potential_path = vault / note_path
            if potential_path.exists():
                path = potential_path
                break
        else:
            raise FileNotFoundError(f"Note not found: {note_path}")

    # Load note preserving frontmatter
    with open(path, 'r', encoding='utf-8') as f:
        post = frontmatter.load(f)

    content = post.content
    lines = content.splitlines(keepends=True)

    # Normalize function for comparing heading text
    def norm(s: str) -> str:
        return " ".join(s.strip().split()).lower()

    target_text = norm(section_heading)

    # Simple ATX heading parser, ignoring fenced code blocks
    def parse_heading(line: str) -> Optional[tuple[int, str]]:
        s = line.strip()
        if not s.startswith('#'):
            return None
        # Count leading '#'
        i = 0
        while i < len(s) and s[i] == '#':
            i += 1
        if i == 0 or i > 6:
            return None
        title = s[i:].strip()
        # Strip optional trailing hashes
        title = re.sub(r"\s*#+\s*$", "", title).strip()
        if not title:
            return None
        return i, title

    # Track fenced code blocks to avoid false positive headings
    in_fence = False
    fence_re = re.compile(r"^\s*```")

    headings: list[tuple[int, int, int, str]] = []  # (line_index, level, char_index_start, title)
    char_index = 0
    for idx, line in enumerate(lines):
        if fence_re.match(line):
            in_fence = not in_fence
        if not in_fence:
            parsed = parse_heading(line)
            if parsed:
                lvl, title = parsed
                headings.append((idx, lvl, char_index, title))
        char_index += len(line)

    # Find target heading index in lines
    target_idx: Optional[int] = None
    target_lvl: Optional[int] = None

    for idx, lvl, _start_char, title in headings:
        if norm(title) == target_text and (heading_level is None or lvl == heading_level):
            target_idx = idx
            target_lvl = lvl
            break

    if target_idx is None or target_lvl is None:
        raise ValueError(f"Section heading not found: '{section_heading}'")

    # Determine the end of the section: next heading with level <= target level, or EOF
    end_idx = len(lines)
    for idx, lvl, _start_char, _title in headings:
        if idx <= target_idx:
            continue
        if lvl <= target_lvl:
            end_idx = idx
            break

    # Build new content: keep heading line, replace body between heading and end_idx
    prefix = ''.join(lines[: target_idx + 1])
    suffix = ''.join(lines[end_idx:])

    body = new_content.rstrip('\n')
    # Ensure one blank line after the heading before body (common Markdown style)
    if body:
        replacement = prefix + "\n" + body + "\n" + suffix
    else:
        # Empty body: keep a single blank line after heading for readability
        replacement = prefix + "\n" + suffix

    post.content = replacement

    # Persist changes
    with open(path, 'w', encoding='utf-8') as f:
        rendered = frontmatter.dumps(post)
        if isinstance(rendered, bytes):
            rendered = rendered.decode('utf-8')
        f.write(rendered)

    # Re-index updated note
    note = parse_note(path)
    app_state.unified_store.upsert_note(note)

    return {
        "path": str(path),
        "section": section_heading,
        "message": "Section content updated in-place"
    }

@mcp.tool()
def get_backlinks(note_id_or_path: str) -> List[Dict]:
    """Get all notes that link to the specified note."""
    return app_state.unified_store.get_backlinks(note_id_or_path)

@mcp.tool()
def get_notes_by_tag(tag: str) -> List[Dict]:
    """Get all notes that have the specified tag."""
    return app_state.unified_store.get_notes_by_tag(tag)

class ReindexResult(BaseModel):
    """Result of reindexing operation."""
    operation: str
    notes_indexed: int
    success: bool
    message: str

@mcp.tool()
def reindex_vault(
    target: str = "all",
    full_reindex: bool = False
) -> ReindexResult:
    """
    Reindex the vault with unified store.
    
    Args:
        target: What to reindex - "all" (unified store supports all data)
        full_reindex: If True, completely rebuild the database (default: False)
    
    Returns:
        ReindexResult with operation details
    """
    try:
        # Use unified store's reindex method
        notes_count = app_state.unified_store.reindex(
            vaults=settings.vaults,
            full_reindex=full_reindex
        )
        
        return ReindexResult(
            operation=f"reindex_{target}",
            notes_indexed=notes_count,
            success=True,
            message=f"Successfully reindexed {notes_count} chunks from vault"
        )
    
    except Exception as e:
        return ReindexResult(
            operation=f"reindex_{target}",
            notes_indexed=0,
            success=False,
            message=f"Reindex failed: {str(e)}"
        )

class EnrichmentResult(BaseModel):
    """Result of note enrichment."""
    processed_notes: int
    successful: int
    failed: int
    para_distribution: Dict[str, int]
    message: str

@mcp.tool()
def enrich_notes(
    note_paths: Optional[List[str]] = None,
    limit: Optional[int] = None,
    dry_run: bool = False
) -> EnrichmentResult:
    """
    Enrich notes with PARA taxonomy and semantic relationships.
    
    Args:
        note_paths: Specific note paths to enrich. If None, enriches all notes.
        limit: Maximum number of notes to process (if note_paths is None)
        dry_run: If True, analyze but don't save changes (default: False)
    
    Returns:
        EnrichmentResult with enrichment statistics
    """
    try:
        # Import enrichment module dynamically
        import importlib.util
        script_path = Path(__file__).parent.parent / "scripts" / "enrich_para_taxonomy.py"
        spec = importlib.util.spec_from_file_location("enrich_para_taxonomy", script_path)
        ParaTaxonomyEnricher = None
        if spec and spec.loader:
            enrich_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(enrich_module)
            ParaTaxonomyEnricher = enrich_module.ParaTaxonomyEnricher
        
        # Initialize enricher
        if ParaTaxonomyEnricher is None:
            raise ImportError("Could not import ParaTaxonomyEnricher")
        enricher = ParaTaxonomyEnricher()
        
        # Determine which notes to process
        if note_paths:
            # Filter out protected test corpus paths
            paths_to_process: list[str] = [
                str(p) for p in (Path(p) for p in note_paths)
                if not is_protected_test_content(Path(p))
            ]
        else:
            # Get all notes from vault
            from .fs_indexer import discover_files
            all_paths = [
                p for p in discover_files(settings.vaults, settings.supported_extensions)
                if not is_protected_test_content(p)
            ]
            
            # Apply limit if specified
            if limit:
                paths_to_process = [str(p) for p in all_paths[:limit]]
            else:
                paths_to_process = [str(p) for p in all_paths]
        
        # Process notes
        processed = 0
        successful = 0
        failed = 0
        para_distribution = {}
        
        for note_path in paths_to_process:
            try:
                result = enricher.enrich_note_properties(note_path, dry_run=dry_run)
                if result:
                    processed += 1
                    successful += 1
                    
                    # Track PARA distribution
                    para_type = result.get('para_type', 'unknown')
                    para_distribution[para_type] = para_distribution.get(para_type, 0) + 1
                else:
                    failed += 1
            except Exception:
                failed += 1
        
        mode = "Dry run - no changes applied" if dry_run else "Changes applied"
        
        return EnrichmentResult(
            processed_notes=processed,
            successful=successful,
            failed=failed,
            para_distribution=para_distribution,
            message=f"Enrichment complete. {mode}. Processed {processed} notes: {successful} successful, {failed} failed."
        )
    
    except Exception as e:
        return EnrichmentResult(
            processed_notes=0,
            successful=0,
            failed=0,
            para_distribution={},
            message=f"Enrichment failed: {str(e)}"
        )

def run_stdio():
    """Run MCP server via stdio for Claude Desktop integration."""
    mcp.run()

def run_http(host: Optional[str] = None, port: Optional[int] = None):
    """Run MCP server via HTTP for Cursor and other HTTP clients."""
    import uvicorn
    from .config import settings
    
    # Use provided values or fall back to settings
    host = host or settings.mcp_host
    port = port or settings.mcp_port
    
    logger.info("Starting Graph RAG MCP Server (HTTP) on %s:%s", host, port)
    logger.info("Vault paths: %s", [str(p) for p in settings.vaults])
    logger.info("Unified Store (ChromaDB): %s", settings.chroma_dir)
    
    # Run FastMCP with HTTP transport
    uvicorn.run(
        "src.mcp_server:mcp.get_app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    run_stdio()
