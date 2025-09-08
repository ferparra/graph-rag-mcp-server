from __future__ import annotations
import re
import json
import frontmatter
from pathlib import Path
from typing import Iterable, Dict, List, Tuple, Any
from pydantic import BaseModel

WIKILINK_PATTERN = re.compile(r"\[\[([^\]]+)\]\]")
TAG_PATTERN = re.compile(r"(#\w[\w/-]+)")

class NoteDoc(BaseModel):
    """A document representing a parsed note with metadata."""
    id: str
    text: str
    path: Path
    title: str
    tags: List[str]
    links: List[str]
    meta: Dict[str, Any]
    frontmatter: Dict[str, Any]

def discover_files(vaults: Iterable[Path], supported_extensions: List[str]) -> Iterable[Path]:
    for root in vaults:
        for p in root.rglob("*"):
            if p.suffix.lower() in {ext.lower() for ext in supported_extensions}:
                if not any(part.startswith('.') for part in p.parts):
                    yield p

def is_protected_test_content(path: Path) -> bool:
    """Return True if the path is within a test environment folder.

    We treat any path that has a path segment that starts with test-related prefixes
    as protected to avoid accidental modification of test content.
    """
    try:
        # Protect temporary test directories and fixture content
        test_prefixes = {
            'test_vault_', 'eval_', 'temp_', 'standard_test_', 'planets_only_'
        }
        test_paths = {'fixtures', 'content'}
        
        return any(
            any(part.startswith(prefix) for prefix in test_prefixes) or part in test_paths
            for part in path.parts
        )
    except Exception:
        return False

def load_text(path: Path) -> Tuple[str, Dict]:
    if path.suffix.lower() == ".excalidraw":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            labels = []
            for el in data.get("elements", []):
                txt = el.get("text")
                if txt:
                    labels.append(txt)
            return "\n".join(labels) or path.stem, {}
        except Exception:
            return path.stem, {}
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            post = frontmatter.load(f)
            return post.content, post.metadata
    except Exception:
        try:
            raw_text = path.read_text(encoding="utf-8", errors="ignore")
            return raw_text, {}
        except Exception:
            return path.stem, {}

def parse_note(path: Path) -> NoteDoc:
    raw_text, frontmatter_data = load_text(path)
    
    links = WIKILINK_PATTERN.findall(raw_text)
    tags_from_content = [t.strip("#") for t in TAG_PATTERN.findall(raw_text)]
    tags_from_frontmatter = frontmatter_data.get("tags", [])
    
    if isinstance(tags_from_frontmatter, str):
        tags_from_frontmatter = [tags_from_frontmatter]
    elif not isinstance(tags_from_frontmatter, list):
        tags_from_frontmatter = []
    
    all_tags = sorted(set(tags_from_content + tags_from_frontmatter))
    
    title = frontmatter_data.get("title", path.stem)
    
    nid = str(path.relative_to(path.parents[len(path.parents)-1])).replace("\\", "/")
    
    # Convert tags list to comma-separated string for ChromaDB compatibility
    meta = {
        "path": str(path),
        "title": title,
        "tags": ", ".join(all_tags) if all_tags else "",
        "vault": str(path.parents[len(path.parents)-1])
    }
    
    # Add frontmatter fields to metadata, ensuring ChromaDB compatibility
    for key, value in frontmatter_data.items():
        if key not in ["title", "tags"]:  # Skip already handled fields
            if isinstance(value, list):
                # Convert lists to comma-separated strings
                meta[key] = ", ".join(str(v) for v in value)
            elif isinstance(value, (dict, set)):
                # Convert complex types to string representation
                meta[key] = str(value)
            elif value is None:
                meta[key] = ""
            else:
                # Primitive types are fine
                meta[key] = value
    
    return NoteDoc(
        id=nid,
        text=raw_text.strip(),
        path=path,
        title=title,
        tags=all_tags,
        links=links,
        meta=meta,
        frontmatter=frontmatter_data
    )

def chunk_text(text: str, max_chars: int, overlap: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + max_chars, len(text))
        
        if end < len(text):
            last_space = text.rfind(' ', start, end)
            last_newline = text.rfind('\n', start, end)
            
            best_break = max(last_space, last_newline)
            if best_break > start:
                end = best_break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        if end >= len(text):
            break
            
        start = max(start + 1, end - overlap)
    
    return chunks

def extract_backlinks(note: NoteDoc, all_notes: Dict[str, NoteDoc]) -> List[str]:
    backlinks = []
    note_title = note.title.lower() if note.title else ""
    note_stem = note.path.stem.lower()
    
    for other_note in all_notes.values():
        if other_note.id == note.id:
            continue
            
        for link in other_note.links:
            link_lower = link.lower()
            if link_lower == note_title or link_lower == note_stem:
                backlinks.append(other_note.id)
                break
    
    return backlinks

def resolve_links(note: NoteDoc, all_notes: Dict[str, NoteDoc]) -> List[str]:
    resolved = []
    
    for link in note.links:
        link_lower: str = link.lower()
        
        for other_note in all_notes.values():
            other_title: str = other_note.title.lower() if other_note.title else ""
            if (other_title == link_lower or 
                other_note.path.stem.lower() == link_lower):
                resolved.append(other_note.id)
                break
    
    return resolved
