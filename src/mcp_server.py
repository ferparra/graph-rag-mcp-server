from __future__ import annotations
import frontmatter
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastmcp import FastMCP
from pydantic import BaseModel
from .dspy_rag import VaultSearcher
from .graph_store import RDFGraphStore
from .chroma_store import ChromaStore
from .fs_indexer import parse_note
from .config import settings

class AppState:
    def __init__(self):
        # Initialize RDF graph store with SQLite backend first
        self.graph_store = RDFGraphStore(
            db_path=settings.rdf_db_path,
            store_identifier=settings.rdf_store_identifier
        )
        print(f"Connected to RDF graph store: {settings.rdf_db_path}")
        
        self.chroma_store = ChromaStore(
            client_dir=settings.chroma_dir,
            collection_name=settings.collection,
            embed_model=settings.embedding_model
        )
        
        # Initialize searcher with graph store for semantic retrieval
        self.searcher = VaultSearcher(graph_store=self.graph_store)

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
    neighbors = app_state.graph_store.get_neighbors(
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
    subgraph = app_state.graph_store.get_subgraph(seed_notes, depth)
    return GraphResult(**subgraph)

@mcp.tool()
def list_notes(
    limit: Optional[int] = 50,
    vault_filter: Optional[str] = None
) -> List[Dict]:
    """List all notes in the vault with metadata."""
    notes = app_state.chroma_store.get_all_notes(limit=limit)
    
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
        frontmatter.dump(post, f)
    
    note = parse_note(path)
    app_state.chroma_store.upsert_note(note)
    
    app_state.graph_store.upsert_note(note)
    
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
    
    app_state.chroma_store.delete_note(str(path.relative_to(vault_root)))
    
    app_state.graph_store.delete_note(str(path.relative_to(vault_root)))
    
    return str(new_path)

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
        frontmatter.dump(post, f)
    
    note = parse_note(path)
    app_state.chroma_store.upsert_note(note)
    
    app_state.graph_store.upsert_note(note)
    
    return f"Content added to {note_path}"

@mcp.tool()
def get_backlinks(note_id_or_path: str) -> List[Dict]:
    """Get all notes that link to the specified note."""
    return app_state.graph_store.get_backlinks(note_id_or_path)

@mcp.tool()
def get_notes_by_tag(tag: str) -> List[Dict]:
    """Get all notes that have the specified tag."""
    return app_state.graph_store.get_notes_by_tag(tag)

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
    Reindex the vault databases.
    
    Args:
        target: What to reindex - "all", "chroma", or "rdf" (default: "all")
        full_reindex: If True, completely rebuild the database (default: False)
    
    Returns:
        ReindexResult with operation details
    """
    try:
        notes_count = 0
        
        if target in ["all", "chroma"]:
            if full_reindex:
                # Clear and rebuild ChromaDB
                app_state.chroma_store._collection().delete(where={})
            
            # Reindex ChromaDB
            from .fs_indexer import discover_files
            for path in discover_files(settings.vaults, settings.supported_extensions):
                try:
                    note = parse_note(path)
                    app_state.chroma_store.upsert_note(note)
                    notes_count += 1
                except Exception:
                    continue
        
        if target in ["all", "rdf"]:
            # Reindex RDF graph
            if full_reindex:
                app_state.graph_store.clear_graph()
            notes_count = app_state.graph_store.build_from_notes(settings.vaults)
        
        return ReindexResult(
            operation=f"reindex_{target}",
            notes_indexed=notes_count,
            success=True,
            message=f"Successfully reindexed {notes_count} notes"
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
        # Import enrichment module
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        from scripts.enrich_para_taxonomy import ParaTaxonomyEnricher
        
        # Initialize enricher
        enricher = ParaTaxonomyEnricher()
        
        # Determine which notes to process
        if note_paths:
            paths_to_process = note_paths
        else:
            # Get all notes from vault
            from .fs_indexer import discover_files
            all_paths = list(discover_files(settings.vaults, settings.supported_extensions))
            
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

if __name__ == "__main__":
    run_stdio()
