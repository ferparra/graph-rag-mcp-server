#!/usr/bin/env python3
"""Graph RAG MCP Server Indexing Tools."""
from __future__ import annotations

import sys
from pathlib import Path

import typer

app = typer.Typer(help="Graph RAG MCP Server Indexing Tools")

def _get_modules():      
    """Dynamically import src modules."""
    project_root: Path = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from src.chroma_store import ChromaStore
    from src.config import settings
    from src.graph_store import RDFGraphStore
    
    return ChromaStore, settings, RDFGraphStore

@app.command()
def chroma(
    full_reindex: bool = typer.Option(False, "--full", help="Delete and rebuild entire ChromaDB collection"),
    vault_path: str = typer.Option(None, "--vault", help="Specific vault path to index")
) -> None:
    """Index vault content into ChromaDB for vector search."""
    typer.echo("🔍 Starting ChromaDB indexing...")
    
    ChromaStore, settings, _ = _get_modules()
    
    vaults = [Path(vault_path)] if vault_path else settings.vaults
    
    store = ChromaStore(
        client_dir=settings.chroma_dir,
        collection_name=settings.collection,
        embed_model=settings.embedding_model
    )
    
    try:
        chunk_count = store.reindex(vaults, full_reindex=full_reindex)
        typer.echo(f"✅ Indexed {chunk_count} chunks into ChromaDB collection '{settings.collection}'")
        typer.echo(f"📁 ChromaDB location: {settings.chroma_dir}")
    except Exception as e:
        typer.echo(f"❌ ChromaDB indexing failed: {e}", err=True)
        raise typer.Exit(1)

@app.command()
def rdf(
    vault_path: str = typer.Option(None, "--vault", help="Specific vault path to index")
) -> None:
    """Index vault content into RDF graph with SQLite storage."""
    typer.echo("🕸️ Starting RDF graph indexing...")
    
    _, settings, RDFGraphStore = _get_modules()
    
    vaults = [Path(vault_path)] if vault_path else settings.vaults
    
    try:
        graph_store = RDFGraphStore(
            db_path=settings.rdf_db_path,
            store_identifier=settings.rdf_store_identifier
        )
        
        node_count = graph_store.build_from_notes(vaults)
        
        typer.echo(f"✅ Indexed {node_count} nodes into RDF graph")
        typer.echo(f"🗃️ RDF SQLite database: {settings.rdf_db_path}")
        
        graph_store.close()
        
    except Exception as e:
        typer.echo(f"❌ RDF indexing failed: {e}", err=True)
        raise typer.Exit(1)

@app.command()
def all(
    full_reindex: bool = typer.Option(False, "--full", help="Delete and rebuild entire ChromaDB collection"),
    vault_path: str = typer.Option(None, "--vault", help="Specific vault path to index")
) -> None:
    """Index vault content into both ChromaDB and RDF graph store."""
    typer.echo("🚀 Starting full vault indexing...")
    
    ChromaStore, settings, RDFGraphStore = _get_modules()
    
    vaults = [Path(vault_path)] if vault_path else settings.vaults
    
    typer.echo(f"📚 Indexing vaults: {[str(v) for v in vaults]}")
    
    chroma_success = False
    graph_success = False
    
    # Index ChromaDB
    try:
        store = ChromaStore(
            client_dir=settings.chroma_dir,
            collection_name=settings.collection,
            embed_model=settings.embedding_model
        )
        
        chunk_count = store.reindex(vaults, full_reindex=full_reindex)
        typer.echo(f"✅ ChromaDB: Indexed {chunk_count} chunks")
        chroma_success = True
        
    except Exception as e:
        typer.echo(f"❌ ChromaDB indexing failed: {e}", err=True)
    
    # Index RDF graph  
    try:
        graph_store = RDFGraphStore(
            db_path=settings.rdf_db_path,
            store_identifier=settings.rdf_store_identifier
        )
        
        node_count = graph_store.build_from_notes(vaults)
        typer.echo(f"✅ RDF Graph: Indexed {node_count} nodes")
        graph_store.close()
        graph_success = True
        
    except Exception as e:
        typer.echo(f"❌ RDF graph indexing failed: {e}", err=True)
    
    if chroma_success and graph_success:
        typer.echo("🎉 Full indexing completed successfully!")
    elif chroma_success:
        typer.echo("⚠️ Partial success: ChromaDB indexed, RDF graph indexing failed")
    elif graph_success:
        typer.echo("⚠️ Partial success: RDF graph indexed, ChromaDB indexing failed")
    else:
        typer.echo("❌ Indexing failed")
        raise typer.Exit(1)

@app.command()
def status() -> None:
    """Show indexing status and statistics."""
    typer.echo("📊 Graph RAG MCP Server Status")
    typer.echo("=" * 40)
    
    ChromaStore, settings, RDFGraphStore = _get_modules()
    
    typer.echo(f"📁 Configured vaults: {len(settings.vaults)}")
    for vault in settings.vaults:
        typer.echo(f"   • {vault}")
    
    try:
        store = ChromaStore(
            client_dir=settings.chroma_dir,
            collection_name=settings.collection,
            embed_model=settings.embedding_model
        )
        chunk_count = store.count()
        typer.echo(f"🔍 ChromaDB: {chunk_count} chunks indexed")
        
        notes = store.get_all_notes(limit=None)
        typer.echo(f"📝 Unique notes: {len(notes)}")
        
    except Exception as e:
        typer.echo(f"❌ ChromaDB status error: {e}")
    
    try:
        graph_store = RDFGraphStore(
            db_path=settings.rdf_db_path,
            store_identifier=settings.rdf_store_identifier
        )
        stats = graph_store.get_stats()
        typer.echo(f"✅ RDF Graph: {stats['notes']} notes, {stats['links']} links, {stats['tags']} tags")
        typer.echo(f"🗃️ RDF Database: {settings.rdf_db_path}")
        graph_store.close()
        
    except Exception as e:
        typer.echo(f"❌ RDF Graph: {e}")
    
    typer.echo(f"🤖 Embedding model: {settings.embedding_model}")
    typer.echo(f"🧠 Gemini model: {settings.gemini_model}")
    typer.echo(f"🔑 Gemini API key: {'✅ Set' if settings.gemini_api_key else '❌ Not set'}")

if __name__ == "__main__":
    app()