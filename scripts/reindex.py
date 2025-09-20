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
    
    from src.unified_store import UnifiedStore
    from src.config import settings
    
    return UnifiedStore, settings


@app.command()
def unified(
    full_reindex: bool = typer.Option(False, "--full", help="Delete and rebuild entire unified store"),
    vault_path: str = typer.Option(None, "--vault", help="Specific vault path to index")
) -> None:
    """Index vault content into unified ChromaDB store with graph capabilities."""
    typer.echo("🚀 Starting unified store indexing...")
    
    UnifiedStore, settings = _get_modules()
    
    vaults = [Path(vault_path).expanduser()] if vault_path else settings.vaults
    
    try:
        unified_store = UnifiedStore(
            client_dir=settings.chroma_dir,
            collection_name=settings.collection,
            embed_model=settings.embedding_model
        )
        
        chunk_count = unified_store.reindex(vaults, full_reindex=full_reindex)
        
        typer.echo(f"✅ Indexed {chunk_count} chunks into unified store")
        typer.echo(f"📁 Store location: {settings.chroma_dir}")
        
    except Exception as e:
        typer.echo(f"❌ Unified store indexing failed: {e}", err=True)
        raise typer.Exit(1)

@app.command()
def all(
    full_reindex: bool = typer.Option(False, "--full", help="Delete and rebuild entire ChromaDB collection"),
    vault_path: str = typer.Option(None, "--vault", help="Specific vault path to index")
) -> None:
    """Index vault content into unified store (same as 'unified' command)."""
    typer.echo("🚀 Starting unified store indexing...")
    
    UnifiedStore, settings = _get_modules()
    
    vaults = [Path(vault_path).expanduser()] if vault_path else settings.vaults
    
    typer.echo(f"📚 Indexing vaults: {[str(v) for v in vaults]}")
    
    try:
        unified_store = UnifiedStore(
            client_dir=settings.chroma_dir,
            collection_name=settings.collection,
            embed_model=settings.embedding_model
        )
        
        chunk_count = unified_store.reindex(vaults, full_reindex=full_reindex)
        typer.echo(f"✅ Indexed {chunk_count} chunks into unified store")
        typer.echo(f"📁 Store location: {settings.chroma_dir}")
        typer.echo("🎉 Indexing completed successfully!")
        
    except Exception as e:
        typer.echo(f"❌ Indexing failed: {e}", err=True)
        raise typer.Exit(1)

@app.command()
def status() -> None:
    """Show indexing status and statistics."""
    typer.echo("📊 Graph RAG MCP Server Status")
    typer.echo("=" * 40)
    
    UnifiedStore, settings = _get_modules()
    
    typer.echo(f"📁 Configured vaults: {len(settings.vaults)}")
    for vault in settings.vaults:
        typer.echo(f"   • {vault}")
    
    try:
        unified_store = UnifiedStore(
            client_dir=settings.chroma_dir,
            collection_name=settings.collection,
            embed_model=settings.embedding_model
        )
        
        chunk_count = unified_store.count()
        stats = unified_store.get_stats()
        
        typer.echo(f"🔍 Unified Store: {chunk_count} chunks indexed")
        typer.echo(f"📝 Notes: {stats['notes']}")
        typer.echo(f"🏷️ Tags: {stats['tags']}")
        typer.echo(f"🔗 Links: {stats['links']}")
        typer.echo(f"📁 Store location: {settings.chroma_dir}")
        
    except Exception as e:
        typer.echo(f"❌ Unified store status error: {e}")
    
    typer.echo(f"🤖 Embedding model: {settings.embedding_model}")
    typer.echo(f"🧠 Gemini model: {settings.gemini_model}")
    typer.echo(f"🔑 Gemini API key: {'✅ Set' if settings.gemini_api_key else '❌ Not set'}")

if __name__ == "__main__":
    app()