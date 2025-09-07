#!/usr/bin/env python3
import asyncio
import sys
from pathlib import Path

import typer
from watchfiles import awatch

app = typer.Typer(help="Real-time vault indexing watcher")

def _get_modules():
    """Dynamically import src modules."""
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from src.config import settings
    from src.unified_store import UnifiedStore
    from src.fs_indexer import parse_note
    
    return settings, UnifiedStore, parse_note

class VaultWatcher:
    def __init__(self):
        settings, UnifiedStore, parse_note = _get_modules()
        
        self.settings = settings
        self.parse_note = parse_note
        
        self.unified_store = UnifiedStore(
            client_dir=settings.chroma_dir,
            collection_name=settings.collection,
            embed_model=settings.embedding_model
        )
    
    async def process_file_change(self, change_type: str, file_path: Path):
        """Process a single file change event."""
        try:
            vault_root = None
            for vault in self.settings.vaults:
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
                    note = self.parse_note(file_path)
                    
                    chunks_updated = self.unified_store.upsert_note(note)
                    print(f"📝 Updated {file_path.name}: {chunks_updated} chunks in unified store")
                    
            elif change_type == "deleted":
                chunks_deleted = self.unified_store.delete_note(note_id)
                print(f"🗑️ Deleted {file_path.name}: {chunks_deleted} chunks from unified store")
                    
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
    
    async def watch_vaults(self, debounce_ms: int = 1000):
        """Watch all configured vaults for changes."""
        print(f"👀 Watching {len(self.settings.vaults)} vaults for changes...")
        for vault in self.settings.vaults:
            print(f"   📁 {vault}")
        
        print(f"⏱️ Debounce delay: {debounce_ms}ms")
        print("🔄 Press Ctrl+C to stop watching\n")
        
        pending_changes = {}
        
        async for changes in awatch(*self.settings.vaults, debounce=debounce_ms):
            for change_type, file_path_str in changes:
                file_path = Path(file_path_str)
                
                if file_path.suffix.lower() not in {ext.lower() for ext in self.settings.supported_extensions}:
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
    
    settings, _, _, _ = _get_modules()
    
    async def test_watch():
        async for changes in awatch(*settings.vaults):
            for change_type, file_path_str in changes:
                file_path = Path(file_path_str)
                print(f"🔍 Detected: {change_type.name} -> {file_path}")
                
                if file_path.suffix.lower() in {ext.lower() for ext in settings.supported_extensions}:
                    print("   ✅ Would process this file")
                else:
                    print("   ⏭️ Would skip (unsupported extension)")
    
    try:
        asyncio.run(test_watch())
    except KeyboardInterrupt:
        print("\n✅ Test completed")

if __name__ == "__main__":
    app()