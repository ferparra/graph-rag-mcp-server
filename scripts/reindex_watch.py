#!/usr/bin/env python3
import sys
import asyncio
from pathlib import Path
from watchfiles import awatch
import typer

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import settings
from src.chroma_store import ChromaStore
from src.graph_store import RDFGraphStore
from src.fs_indexer import parse_note

app = typer.Typer(help="Real-time vault indexing watcher")

class VaultWatcher:
    def __init__(self):
        self.chroma_store = ChromaStore(
            client_dir=settings.chroma_dir,
            collection_name=settings.collection,
            embed_model=settings.embedding_model
        )
        
        self.graph_store = RDFGraphStore(
            db_path=settings.rdf_db_path,
            store_identifier=settings.rdf_store_identifier
        )
        print(f"✅ Connected to RDF graph store: {settings.rdf_db_path}")
    
    async def process_file_change(self, change_type: str, file_path: Path):
        """Process a single file change event."""
        try:
            vault_root = None
            for vault in settings.vaults:
                try:
                    file_path.relative_to(vault)
                    vault_root = vault
                    break
                except ValueError:
                    continue
            
            if not vault_root:
                return
            
            note_id = str(file_path.relative_to(vault_root)).replace("\\", "/")
            
            if change_type in ["added", "modified"]:
                if file_path.exists():
                    note = parse_note(file_path)
                    
                    chunks_updated = self.chroma_store.upsert_note(note)
                    print(f"📝 Updated {file_path.name}: {chunks_updated} chunks in ChromaDB")
                    
                    self.graph_store.upsert_note(note)
                    print(f"🕸️ Updated RDF graph relationships for {file_path.name}")
                    
            elif change_type == "deleted":
                chunks_deleted = self.chroma_store.delete_note(note_id)
                print(f"🗑️ Deleted {file_path.name}: {chunks_deleted} chunks from ChromaDB")
                
                self.graph_store.delete_note(note_id)
                print(f"🕸️ Removed RDF graph relationships for {file_path.name}")
                    
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
    
    async def watch_vaults(self, debounce_ms: int = 1000):
        """Watch all configured vaults for changes."""
        print(f"👀 Watching {len(settings.vaults)} vaults for changes...")
        for vault in settings.vaults:
            print(f"   📁 {vault}")
        
        print(f"⏱️ Debounce delay: {debounce_ms}ms")
        print("🔄 Press Ctrl+C to stop watching\n")
        
        pending_changes = {}
        
        async for changes in awatch(*settings.vaults, debounce=debounce_ms):
            for change_type, file_path_str in changes:
                file_path = Path(file_path_str)
                
                if file_path.suffix.lower() not in {ext.lower() for ext in settings.supported_extensions}:
                    continue
                
                if any(part.startswith('.') for part in file_path.parts):
                    continue
                
                change_key = str(file_path)
                pending_changes[change_key] = (change_type.name.lower(), file_path)
            
            if pending_changes:
                print(f"🔄 Processing {len(pending_changes)} file changes...")
                
                for change_type, file_path in pending_changes.values():
                    await self.process_file_change(change_type, file_path)
                
                pending_changes.clear()
                print("✅ Batch update completed\n")

@app.command()
def start(
    debounce_ms: int = typer.Option(1000, "--debounce", help="Debounce delay in milliseconds")
):
    """Start watching vaults for real-time indexing updates."""
    watcher = None
    try:
        watcher = VaultWatcher()
        asyncio.run(watcher.watch_vaults(debounce_ms=debounce_ms))
    except KeyboardInterrupt:
        print("\n👋 Stopping vault watcher...")
        if watcher and hasattr(watcher, 'graph_store') and hasattr(watcher.graph_store, 'close'):
            watcher.graph_store.close()
        print("✅ Vault watcher stopped")
    except Exception as e:
        print(f"❌ Watcher error: {e}")
        raise typer.Exit(1)

@app.command()
def test():
    """Test file change detection without making changes."""
    print("🧪 Testing file change detection...")
    print("📝 Create, modify, or delete a file in your vault to see detection in action")
    print("🔄 Press Ctrl+C to stop test\n")
    
    async def test_watch():
        async for changes in awatch(*settings.vaults):
            for change_type, file_path_str in changes:
                file_path = Path(file_path_str)
                print(f"🔍 Detected: {change_type.name} -> {file_path}")
                
                if file_path.suffix.lower() in {ext.lower() for ext in settings.supported_extensions}:
                    print(f"   ✅ Would process this file")
                else:
                    print(f"   ⏭️ Would skip (unsupported extension)")
    
    try:
        asyncio.run(test_watch())
    except KeyboardInterrupt:
        print("\n✅ Test completed")

if __name__ == "__main__":
    app()
