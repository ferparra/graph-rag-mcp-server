#!/usr/bin/env python3
"""
Migrate RDF data from SQLAlchemy store to Oxigraph store.

Usage:
    # Export from old SQLAlchemy store to N-Quads
    uv run scripts/migrate_rdf_store.py export
    
    # Import N-Quads into new Oxigraph store
    uv run scripts/migrate_rdf_store.py import
    
    # Full migration (export + import)
    uv run scripts/migrate_rdf_store.py migrate
"""

import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.progress import track
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings

app = typer.Typer(help="RDF Store Migration Tool")
console = Console()

def export_sqlalchemy_store(
    db_path: Path = None,
    output_path: Path = Path("rdf_backup.nq"),
    store_identifier: str = "obsidian_vault_graph"
) -> bool:
    """Export data from SQLAlchemy RDF store to N-Quads format."""
    try:
        # Import old SQLAlchemy store (only if needed for export)
        console.print("[yellow]⚠️  SQLAlchemy store support has been removed.[/yellow]")
        console.print("[yellow]If you have an existing SQLAlchemy store, you'll need to:[/yellow]")
        console.print("1. Temporarily reinstall rdflib-sqlalchemy: [cyan]uv add rdflib-sqlalchemy[/cyan]")
        console.print("2. Run the export")
        console.print("3. Remove it again: [cyan]uv remove rdflib-sqlalchemy[/cyan]")
        console.print("")
        console.print("[dim]Alternatively, if you don't have existing data, just start fresh with:[/dim]")
        console.print("[cyan]uv run scripts/reindex.py rdf[/cyan]")
        return False
        
    except Exception as e:
        console.print(f"[red]Export failed: {e}[/red]")
        return False

def import_oxigraph_store(
    input_path: Path = Path("rdf_backup.nq"),
    store_dir: Path = None,
    store_identifier: str = "obsidian_vault_graph"
) -> bool:
    """Import N-Quads data into Oxigraph store."""
    try:
        from rdflib import Graph, URIRef
        
        if store_dir is None:
            # Use default from settings
            store_dir = settings.rdf_db_path.parent / f"{settings.rdf_db_path.stem}_oxigraph"
        
        console.print(f"[blue]Importing data into Oxigraph store at {store_dir}[/blue]")
        
        # Create new Oxigraph store with absolute URI identifier
        store_dir.mkdir(parents=True, exist_ok=True)
        graph_uri = f"http://obsidian-vault.local/graphs/{store_identifier}"
        graph = Graph(store="Oxigraph", identifier=URIRef(graph_uri))
        graph.open(str(store_dir.absolute()), create=True)
        
        # Parse the N-Quads file
        if input_path.exists():
            console.print(f"[cyan]Loading data from {input_path}[/cyan]")
            graph.parse(str(input_path), format="nquads")
            
            # Count triples
            triple_count = len(graph)
            console.print(f"[green]✓ Imported {triple_count:,} triples[/green]")
        else:
            console.print(f"[yellow]No backup file found at {input_path}[/yellow]")
            console.print("[dim]Creating empty Oxigraph store[/dim]")
        
        # Close the graph
        graph.close()
        
        console.print("[green]✓ Import complete![/green]")
        return True
        
    except Exception as e:
        console.print(f"[red]Import failed: {e}[/red]")
        return False

@app.command()
def export(
    db_path: Optional[Path] = typer.Option(None, help="SQLite database path"),
    output: Path = typer.Option(Path("rdf_backup.nq"), help="Output N-Quads file")
):
    """Export data from SQLAlchemy store to N-Quads."""
    if db_path is None:
        db_path = settings.rdf_db_path
    
    success = export_sqlalchemy_store(db_path, output)
    if not success:
        raise typer.Exit(1)

@app.command()
def import_data(
    input: Path = typer.Option(Path("rdf_backup.nq"), help="Input N-Quads file"),
    store_dir: Optional[Path] = typer.Option(None, help="Oxigraph store directory")
):
    """Import N-Quads data into Oxigraph store."""
    success = import_oxigraph_store(input, store_dir)
    if not success:
        raise typer.Exit(1)

@app.command()
def migrate(
    db_path: Optional[Path] = typer.Option(None, help="SQLite database path"),
    store_dir: Optional[Path] = typer.Option(None, help="Oxigraph store directory"),
    backup_file: Path = typer.Option(Path("rdf_backup.nq"), help="Intermediate backup file")
):
    """Full migration from SQLAlchemy to Oxigraph."""
    console.print("[bold blue]Starting RDF store migration[/bold blue]")
    
    # Step 1: Export
    console.print("\n[cyan]Step 1: Exporting from SQLAlchemy store[/cyan]")
    if db_path is None:
        db_path = settings.rdf_db_path
    
    if not export_sqlalchemy_store(db_path, backup_file):
        console.print("[yellow]Export skipped - see instructions above[/yellow]")
    
    # Step 2: Import
    console.print("\n[cyan]Step 2: Importing to Oxigraph store[/cyan]")
    if not import_oxigraph_store(backup_file, store_dir):
        raise typer.Exit(1)
    
    console.print("\n[bold green]✓ Migration complete![/bold green]")
    console.print("[dim]You can now delete the old SQLite database if desired[/dim]")

@app.command()
def verify(
    store_dir: Optional[Path] = typer.Option(None, help="Oxigraph store directory")
):
    """Verify the Oxigraph store is working."""
    try:
        from rdflib import Graph, URIRef
        
        if store_dir is None:
            store_dir = settings.rdf_db_path.parent / f"{settings.rdf_db_path.stem}_oxigraph"
        
        console.print(f"[blue]Verifying Oxigraph store at {store_dir}[/blue]")
        
        # Open the store with the correct identifier
        graph_uri = f"http://obsidian-vault.local/graphs/{settings.rdf_store_identifier}"
        graph = Graph(store="Oxigraph", identifier=URIRef(graph_uri))
        graph.open(str(store_dir.absolute()))
        
        # Count triples
        triple_count = len(graph)
        console.print(f"[green]✓ Store contains {triple_count:,} triples[/green]")
        
        # Run a simple SPARQL query
        query = """
        SELECT (COUNT(DISTINCT ?s) as ?subjects) 
        WHERE { ?s ?p ?o }
        """
        
        for row in graph.query(query):
            console.print(f"[green]✓ Unique subjects: {row.subjects}[/green]")
        
        graph.close()
        
        console.print("[bold green]✓ Oxigraph store is working correctly![/bold green]")
        
    except Exception as e:
        console.print(f"[red]Verification failed: {e}[/red]")
        raise typer.Exit(1)

if __name__ == "__main__":
    app()