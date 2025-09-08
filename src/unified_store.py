from __future__ import annotations
from typing import List, Dict, Optional, Any
from pathlib import Path
import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.types import EmbeddingFunction, Embeddable
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
    DefaultEmbeddingFunction,
)
from pydantic import BaseModel, Field, ConfigDict
# Support both package and module execution contexts
try:
    from config import settings
    from fs_indexer import discover_files, parse_note, chunk_text, NoteDoc, resolve_links
    from semantic_chunker import SemanticChunker
except ImportError:  # When imported as part of a package
    from .config import settings
    from .fs_indexer import discover_files, parse_note, chunk_text, NoteDoc, resolve_links
    from .semantic_chunker import SemanticChunker
from typing import cast


class UnifiedStore(BaseModel):
    """Unified ChromaDB store with graph relationship capabilities."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    client_dir: Path
    collection_name: str
    embed_model: str
    semantic_chunker: SemanticChunker = Field(
        default_factory=lambda: SemanticChunker(
            min_chunk_size=settings.semantic_min_chunk_size,
            max_chunk_size=settings.semantic_max_chunk_size,
            merge_threshold=settings.semantic_merge_threshold,
            include_context=settings.semantic_include_context,
        ),
        exclude=True,
    )

    def _client(self) -> ClientAPI:
        return chromadb.PersistentClient(path=str(self.client_dir))

    def _clean_metadata_for_chroma(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Clean metadata to ensure all values are ChromaDB-compatible (str, int, float, bool, None)."""
        cleaned = {}
        for key, value in metadata.items():
            if isinstance(value, (list, tuple)):
                # Convert lists to comma-separated strings
                cleaned[key] = ",".join(str(v) for v in value) if value else ""
            elif isinstance(value, dict):
                # Convert dicts to JSON strings
                import json
                cleaned[key] = json.dumps(value)
            elif value is None or isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            else:
                # Convert other types to strings
                cleaned[key] = str(value)
        return cleaned

    def _collection(self) -> Collection:
        # Prefer SentenceTransformer embeddings; fall back to a lightweight default in offline/test environments
        try:
            ef_impl: SentenceTransformerEmbeddingFunction = SentenceTransformerEmbeddingFunction(
                model_name=self.embed_model
            )
            ef = cast(EmbeddingFunction[Embeddable], ef_impl)
        except Exception:
            # Network-restricted or model missing; use a simple default embedding function
            ef = cast(EmbeddingFunction[Embeddable], DefaultEmbeddingFunction())
        client: ClientAPI = self._client()
        return client.get_or_create_collection(self.collection_name, embedding_function=ef)

    def _parse_delimited_string(self, value: Optional[str], delimiter: str = ",") -> List[str]:
        """Parse comma-separated string into list, handling empty strings."""
        if not value or str(value).strip() == "":
            return []
        return [item.strip() for item in str(value).split(delimiter) if item.strip()]
    
    def _resolve_note_id(self, note_id_or_title: str) -> str:
        """Resolve a title or partial path to a full note ID."""
        # If it's already a full path, return as-is
        if note_id_or_title.startswith('/') or '/' in note_id_or_title:
            return note_id_or_title
        
        # Try to find by title
        try:
            col = self._collection()
            results = col.get(include=['metadatas'])
            
            metadatas = results.get('metadatas')
            if metadatas:
                for metadata in metadatas:
                    title = metadata.get('title', '')
                    note_id_raw = metadata.get('note_id', '')
                    note_id = str(note_id_raw) if note_id_raw is not None else ''
                    # Match by title or filename
                    if (title == note_id_or_title or 
                        (note_id and note_id_or_title in note_id) or
                        (note_id and note_id.endswith(f'/{note_id_or_title}.md')) or
                        (note_id and note_id.endswith(f'{note_id_or_title}.md'))):
                        return note_id
        except Exception:
            pass
        
        # Return as-is if no match found
        return note_id_or_title

    def _build_graph_metadata(self, note: NoteDoc, all_notes: Optional[Dict[str, NoteDoc]] = None) -> Dict[str, Any]:
        """Build comprehensive graph metadata for a note."""
        metadata = note.meta.copy()
        
        # Add note-level graph relationships
        metadata.update({
            "note_id": note.id,
            "title": note.title,
            "path": str(note.path),
            "vault": note.meta.get("vault", ""),
            "text_length": len(note.text),
            "tags": ",".join(note.tags) if note.tags else "",
        })
        
        # Add link relationships if all_notes provided
        if all_notes:
            resolved_links = resolve_links(note, all_notes)
            metadata["links_to"] = ",".join(resolved_links) if resolved_links else ""
            
            # Find backlinks (notes that link to this note)
            backlinks = []
            for other_note_id, other_note in all_notes.items():
                if other_note_id != note.id:
                    other_resolved_links = resolve_links(other_note, all_notes)
                    if note.id in other_resolved_links:
                        backlinks.append(other_note_id)
            metadata["backlinks_from"] = ",".join(backlinks) if backlinks else ""
        else:
            metadata["links_to"] = ""
            metadata["backlinks_from"] = ""
        
        return metadata

    def _build_chunk_metadata(self, chunk, note_metadata: Dict[str, Any], 
                             all_chunks: Optional[List] = None, chunk_index: int = 0) -> Dict[str, Any]:
        """Build comprehensive metadata for a semantic chunk including relationships."""
        chunk_meta = note_metadata.copy()
        
        # Basic chunk information
        chunk_meta.update({
            "chunk_id": chunk.id,
            "chunk_type": chunk.chunk_type.value,
            "chunk_position": chunk.position,
            "header_text": chunk.header_text or "",
            "header_level": chunk.header_level or 0,
            "parent_headers": ",".join(chunk.parent_headers) if chunk.parent_headers else "",
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "contains_tags": ",".join(chunk.contains_tags) if chunk.contains_tags else "",
            "contains_links": ",".join(chunk.contains_links) if chunk.contains_links else "",
            "importance_score": chunk.importance_score,
            "semantic_chunk": True
        })
        
        # Add chunk relationships if all_chunks provided
        if all_chunks is not None:
            # Sequential relationships
            sequential_next = ""
            sequential_prev = ""
            if chunk_index > 0:
                sequential_prev = all_chunks[chunk_index - 1].id
            if chunk_index < len(all_chunks) - 1:
                sequential_next = all_chunks[chunk_index + 1].id
            
            chunk_meta["sequential_next"] = sequential_next
            chunk_meta["sequential_prev"] = sequential_prev
            
            # Hierarchical relationships
            parent_chunk = ""
            child_chunks = []
            sibling_chunks = []
            
            if chunk.header_level:
                # Find parent (higher level header that comes before this chunk)
                for i in range(chunk_index - 1, -1, -1):
                    other_chunk = all_chunks[i]
                    if (other_chunk.header_level and 
                        other_chunk.header_level < chunk.header_level):
                        parent_chunk = other_chunk.id
                        break
                
                # Find children (lower level headers that come after this chunk)
                for i in range(chunk_index + 1, len(all_chunks)):
                    other_chunk = all_chunks[i]
                    if other_chunk.header_level:
                        if other_chunk.header_level <= chunk.header_level:
                            break  # End of this section
                        elif other_chunk.header_level == chunk.header_level + 1:
                            child_chunks.append(other_chunk.id)
                
                # Find siblings (same level headers under same parent)
                current_parent = parent_chunk
                for other_chunk in all_chunks:
                    if (other_chunk.id != chunk.id and
                        other_chunk.header_level == chunk.header_level):
                        # Check if they have the same parent
                        other_parent = ""
                        other_index = all_chunks.index(other_chunk)
                        for i in range(other_index - 1, -1, -1):
                            check_chunk = all_chunks[i]
                            if (check_chunk.header_level and 
                                check_chunk.header_level < other_chunk.header_level):
                                other_parent = check_chunk.id
                                break
                        
                        if other_parent == current_parent:
                            sibling_chunks.append(other_chunk.id)
            
            chunk_meta["parent_chunk"] = parent_chunk
            chunk_meta["child_chunks"] = ",".join(child_chunks)
            chunk_meta["sibling_chunks"] = ",".join(sibling_chunks)
            
            # All chunks from the same note
            note_chunks = [c.id for c in all_chunks if c.id != chunk.id]
            chunk_meta["note_chunks"] = ",".join(note_chunks)
        
        return chunk_meta

    def reindex(self, vaults: List[Path], full_reindex: bool = False) -> int:
        """Reindex with enhanced graph metadata."""
        col = self._collection()
        
        if full_reindex:
            try:
                self._client().delete_collection(self.collection_name)
                col = self._collection()
            except Exception:
                pass
        
        # First pass: collect all notes for relationship resolution
        all_notes = {}
        for path in discover_files(vaults, settings.supported_extensions):
            try:
                note = parse_note(path)
                all_notes[note.id] = note
            except Exception as e:
                print(f"Error parsing {path}: {e}")
                continue
        
        # Second pass: index with relationships
        ids, docs, metas = [], [], []
        total_chunks = 0
        
        for note in all_notes.values():
            try:
                if not note.text:
                    continue
                
                note_metadata = self._build_graph_metadata(note, all_notes)
                
                if settings.chunk_strategy == "semantic":
                    semantic_chunks = self.semantic_chunker.chunk_note(note)
                    
                    for i, chunk in enumerate(semantic_chunks):
                        ids.append(chunk.id)
                        docs.append(chunk.content)
                        
                        chunk_meta = self._build_chunk_metadata(
                            chunk, note_metadata, semantic_chunks, i
                        )
                        metas.append(self._clean_metadata_for_chroma(chunk_meta))
                else:
                    # Character chunking with basic metadata
                    chunks = chunk_text(note.text, settings.max_chars, settings.overlap)
                    
                    for k, chunk in enumerate(chunks):
                        chunk_id = f"{note.id}#chunk={k}"
                        ids.append(chunk_id)
                        docs.append(chunk)
                        
                        chunk_meta = note_metadata.copy()
                        chunk_meta.update({
                            "chunk_index": k,
                            "total_chunks": len(chunks),
                            "chunk_id": chunk_id,
                            "semantic_chunk": False,
                            "sequential_next": f"{note.id}#chunk={k+1}" if k < len(chunks) - 1 else "",
                            "sequential_prev": f"{note.id}#chunk={k-1}" if k > 0 else "",
                        })
                        metas.append(self._clean_metadata_for_chroma(chunk_meta))
                
                # Batch upsert
                if len(ids) >= 512:
                    col.upsert(ids=ids, documents=docs, metadatas=metas)
                    total_chunks += len(ids)
                    ids, docs, metas = [], [], []
                    
            except Exception as e:
                print(f"Error processing {note.id}: {e}")
                continue
        
        if ids:
            col.upsert(ids=ids, documents=docs, metadatas=metas)
            total_chunks += len(ids)
        
        return total_chunks

    def upsert_note(self, note: NoteDoc, all_notes: Optional[Dict[str, NoteDoc]] = None) -> int:
        """Add or update a note with graph relationships."""
        col = self._collection()
        
        # Delete existing chunks for this note
        existing_ids = self._get_note_chunk_ids(note.id)
        if existing_ids:
            col.delete(ids=existing_ids)
        
        if not note.text:
            return 0
        
        note_metadata = self._build_graph_metadata(note, all_notes)
        ids, docs, metas = [], [], []
        
        if settings.chunk_strategy == "semantic":
            semantic_chunks = self.semantic_chunker.chunk_note(note)
            
            for i, chunk in enumerate(semantic_chunks):
                ids.append(chunk.id)
                docs.append(chunk.content)
                
                chunk_meta = self._build_chunk_metadata(
                    chunk, note_metadata, semantic_chunks, i
                )
                metas.append(self._clean_metadata_for_chroma(chunk_meta))
        else:
            chunks = chunk_text(note.text, settings.max_chars, settings.overlap)
            
            for k, chunk in enumerate(chunks):
                chunk_id = f"{note.id}#chunk={k}"
                ids.append(chunk_id)
                docs.append(chunk)
                
                chunk_meta = note_metadata.copy()
                chunk_meta.update({
                    "chunk_index": k,
                    "total_chunks": len(chunks),
                    "chunk_id": chunk_id,
                    "semantic_chunk": False,
                    "sequential_next": f"{note.id}#chunk={k+1}" if k < len(chunks) - 1 else "",
                    "sequential_prev": f"{note.id}#chunk={k-1}" if k > 0 else "",
                })
                metas.append(self._clean_metadata_for_chroma(chunk_meta))
        
        if ids:
            col.upsert(ids=ids, documents=docs, metadatas=metas)
        
        return len(ids)

    def delete_note(self, note_id: str) -> int:
        """Remove a note and all its chunks."""
        existing_ids = self._get_note_chunk_ids(note_id)
        if existing_ids:
            col = self._collection()
            col.delete(ids=existing_ids)
            return len(existing_ids)
        return 0

    def _get_note_chunk_ids(self, note_id: str) -> List[str]:
        """Get all chunk IDs for a note."""
        try:
            col = self._collection()
            results = col.get(where={"note_id": {"$eq": note_id}})
            return results.get('ids', [])
        except Exception:
            return []

    def query(self, q: str, k: int = 6, where: Optional[Dict] = None) -> List[Dict]:
        """Vector similarity search with metadata."""
        col = self._collection()
        
        try:
            res = col.query(
                query_texts=[q], 
                n_results=min(k, col.count()), 
                where=where
            )
            
            hits = []
            if res["ids"] and res["ids"][0]:
                for i in range(len(res["ids"][0])):
                    hit: Dict[str, Any] = {
                        "id": res["ids"][0][i],
                        "text": res["documents"][0][i] if res["documents"] and res["documents"][0] else "",
                        "meta": res["metadatas"][0][i] if res["metadatas"] and res["metadatas"][0] else {},
                    }
                    
                    if "distances" in res and res["distances"] and res["distances"][0]:
                        distance = res["distances"][0][i]
                        if isinstance(distance, (int, float)):
                            hit["distance"] = float(distance)
                    
                    hits.append(hit)
            
            return hits
            
        except Exception as e:
            print(f"Query error: {e}")
            return []

    # Graph operations using ChromaDB metadata queries
    
    def get_neighbors(self, note_id_or_title: str, depth: int = 1, relationship_types: Optional[List[str]] = None) -> List[Dict]:
        """Get neighboring notes through graph relationships."""
        if relationship_types is None:
            relationship_types = ["links_to", "backlinks_from", "tags"]
        
        # Try to resolve title to note_id if needed
        note_id = self._resolve_note_id(note_id_or_title)
        
        visited = set()
        current_level = {note_id}
        all_neighbors = []
        
        for _ in range(depth):
            next_level = set()
            
            for current_note in current_level:
                if current_note in visited:
                    continue
                visited.add(current_note)
                
                # Get chunks for this note to extract relationships
                try:
                    col = self._collection()
                    results = col.get(
                        where={"note_id": {"$eq": current_note}},
                        limit=1  # Just need one chunk to get note metadata
                    )
                    
                    if not results['metadatas']:
                        continue
                    
                    metadata = results['metadatas'][0]
                    
                    # Extract connected notes from metadata
                    connected_notes = set()
                    
                    if "links_to" in relationship_types:
                        links_value = metadata.get("links_to", "")
                        links = self._parse_delimited_string(str(links_value) if links_value is not None else "")
                        connected_notes.update(links)
                    
                    if "backlinks_from" in relationship_types:
                        backlinks_value = metadata.get("backlinks_from", "")
                        backlinks = self._parse_delimited_string(str(backlinks_value) if backlinks_value is not None else "")
                        connected_notes.update(backlinks)
                    
                    if "tags" in relationship_types:
                        # Find other notes with same tags
                        tags_value = metadata.get("tags", "")
                        note_tags = self._parse_delimited_string(str(tags_value) if tags_value is not None else "")
                        for tag in note_tags:
                            tag_results = col.get(
                                where={"tags": {"$eq": tag}},
                                include=['metadatas']
                            )
                            metadatas = tag_results.get('metadatas')
                            if metadatas:
                                for tag_meta in metadatas:
                                    tag_note_id = tag_meta.get('note_id')
                                    if tag_note_id and tag_note_id != current_note:
                                        connected_notes.add(str(tag_note_id))
                    
                    # Add to neighbors list and next level
                    for connected_note in connected_notes:
                        if connected_note not in visited:
                            next_level.add(connected_note)
                            
                            # Get title and path for this neighbor
                            try:
                                neighbor_results = col.get(
                                    where={"note_id": {"$eq": connected_note}},
                                    limit=1,
                                    include=['metadatas']
                                )
                                title = ""
                                path = ""
                                neighbor_metadatas = neighbor_results.get('metadatas')
                                if neighbor_metadatas and len(neighbor_metadatas) > 0:
                                    neighbor_meta = neighbor_metadatas[0]
                                    title = neighbor_meta.get('title', '')
                                    path = neighbor_meta.get('path', '')
                            except Exception:
                                title = ""
                                path = ""
                            
                            all_neighbors.append({
                                "id": connected_note,
                                "title": title,
                                "path": path,
                                "type": "Note"
                            })
                
                except Exception as e:
                    print(f"Error getting neighbors for {current_note}: {e}")
                    continue
            
            current_level = next_level
            if not current_level:
                break
        
        return all_neighbors

    def get_backlinks(self, note_id_or_title: str) -> List[Dict]:
        """Get all notes that link to the specified note."""
        # Resolve title to note_id if needed
        note_id = self._resolve_note_id(note_id_or_title)
        
        try:
            col = self._collection()
            
            # Find chunks where links_to contains this note_id
            # Note: ChromaDB doesn't have substring search, so we need to get all and filter
            results = col.get(include=['metadatas'])
            
            backlink_notes = set()
            metadatas = results.get('metadatas')
            if metadatas:
                for metadata in metadatas:
                    links_to = metadata.get('links_to', '')
                    linked_notes = []
                    if links_to:
                        linked_notes = self._parse_delimited_string(str(links_to) if links_to is not None else "")
                    if note_id in linked_notes:
                        source_note_id = metadata.get('note_id')
                        if source_note_id:
                            backlink_notes.add(source_note_id)
            
            # Get details for backlink notes
            backlinks = []
            for backlink_note_id in backlink_notes:
                try:
                    note_results = col.get(
                        where={"note_id": {"$eq": backlink_note_id}},
                        limit=1,
                        include=['metadatas']
                    )
                    if note_results['metadatas']:
                        metadata = note_results['metadatas'][0]
                        backlinks.append({
                            "id": backlink_note_id,
                            "title": metadata.get('title', ''),
                            "path": metadata.get('path', '')
                        })
                except Exception:
                    continue
            
            return backlinks
            
        except Exception as e:
            print(f"Error getting backlinks for {note_id}: {e}")
            return []

    def get_notes_by_tag(self, tag: str) -> List[Dict]:
        """Get all notes that have the specified tag."""
        try:
            col = self._collection()
            
            # Find chunks where tags contains this tag
            results = col.get(include=['metadatas'])
            
            tagged_notes = set()
            metadatas = results.get('metadatas')
            if metadatas:
                for metadata in metadatas:
                    tags = metadata.get('tags', '')
                    if tags:
                        note_tags = self._parse_delimited_string(str(tags) if tags is not None else "")
                    if tag in note_tags:
                        note_id = metadata.get('note_id')
                        if note_id:
                            tagged_notes.add(note_id)
            
            # Get details for tagged notes
            notes = []
            for note_id in tagged_notes:
                try:
                    note_results = col.get(
                        where={"note_id": {"$eq": note_id}},
                        limit=1,
                        include=['metadatas']
                    )
                    if note_results['metadatas']:
                        metadata = note_results['metadatas'][0]
                        notes.append({
                            "id": note_id,
                            "title": metadata.get('title', ''),
                            "path": metadata.get('path', '')
                        })
                except Exception:
                    continue
            
            return notes
            
        except Exception as e:
            print(f"Error getting notes by tag {tag}: {e}")
            return []

    def fuzzy_tag_search(self, entities: List[str], k: int = 6, where: Optional[Dict] = None) -> List[Dict]:
        """Enhanced tag search with fuzzy matching and hierarchical support."""
        try:
            col = self._collection()
            results = col.get(include=['metadatas', 'documents'])
            
            if not results.get('metadatas'):
                return []
            
            # Score chunks based on tag relevance
            chunk_scores = []
            
            metadatas = results.get('metadatas')
            if not metadatas:
                return []
                
            for i, metadata in enumerate(metadatas):
                tags_value = metadata.get('tags', '')
                if not tags_value:
                    continue
                
                note_tags = self._parse_delimited_string(str(tags_value) if tags_value is not None else "")
                if not note_tags:
                    continue
                
                # Calculate relevance score for this chunk
                score = self._calculate_tag_relevance_score(entities, note_tags)
                
                if score > 0:
                    chunk_id = metadata.get('chunk_id', f'chunk_{i}')
                    documents = results.get('documents')
                    document = (documents[i] if documents and len(documents) > i else "")
                    
                    chunk_scores.append({
                        'id': chunk_id,
                        'text': document,
                        'meta': metadata,
                        'tag_score': score,
                        'distance': 1.0 - score  # Convert to distance-like metric
                    })
            
            # Sort by relevance score and apply additional filters
            chunk_scores.sort(key=lambda x: x['tag_score'], reverse=True)
            
            # Apply where clause filters if provided
            if where:
                filtered_chunks = []
                for chunk in chunk_scores:
                    if self._matches_where_clause(chunk['meta'], where):
                        filtered_chunks.append(chunk)
                chunk_scores = filtered_chunks
            
            return chunk_scores[:k]
            
        except Exception as e:
            print(f"Error in fuzzy tag search: {e}")
            return []
    
    def _calculate_tag_relevance_score(self, query_entities: List[str], note_tags: List[str]) -> float:
        """Calculate relevance score between query entities and note tags."""
        if not query_entities or not note_tags:
            return 0.0
        
        total_score = 0.0
        max_possible_score = len(query_entities)
        
        for entity in query_entities:
            entity_lower = entity.lower()
            best_match_score = 0.0
            
            for tag in note_tags:
                tag_lower = tag.lower()
                
                # Exact match (highest score)
                if entity_lower == tag_lower:
                    best_match_score = max(best_match_score, 1.0)
                    continue
                
                # Hierarchical tag matching (e.g., "health" matches "para/area/health")
                if '/' in tag_lower:
                    tag_parts = tag_lower.split('/')
                    if entity_lower in tag_parts:
                        best_match_score = max(best_match_score, 0.9)
                        continue
                    
                    # Partial hierarchical match
                    for part in tag_parts:
                        if entity_lower in part or part in entity_lower:
                            best_match_score = max(best_match_score, 0.7)
                
                # Substring matching
                if entity_lower in tag_lower:
                    best_match_score = max(best_match_score, 0.8)
                elif tag_lower in entity_lower:
                    best_match_score = max(best_match_score, 0.6)
                
                # Fuzzy matching for common variations
                score = self._fuzzy_string_match(entity_lower, tag_lower)
                best_match_score = max(best_match_score, score)
            
            total_score += best_match_score
        
        return total_score / max_possible_score
    
    def _fuzzy_string_match(self, s1: str, s2: str) -> float:
        """Simple fuzzy string matching."""
        if not s1 or not s2:
            return 0.0
        
        # Jaccard similarity on character bigrams
        def get_bigrams(s):
            return set(s[i:i+2] for i in range(len(s)-1))
        
        bigrams1 = get_bigrams(s1)
        bigrams2 = get_bigrams(s2)
        
        if not bigrams1 and not bigrams2:
            return 1.0 if s1 == s2 else 0.0
        
        if not bigrams1 or not bigrams2:
            return 0.0
        
        intersection = len(bigrams1 & bigrams2)
        union = len(bigrams1 | bigrams2)
        
        jaccard = intersection / union if union > 0 else 0.0
        
        # Boost score for similar length strings
        length_similarity = 1.0 - abs(len(s1) - len(s2)) / max(len(s1), len(s2))
        
        return (jaccard * 0.7 + length_similarity * 0.3) * 0.5  # Scale down fuzzy matches
    
    def _matches_where_clause(self, metadata: Dict, where: Dict) -> bool:
        """Check if metadata matches a where clause."""
        for key, condition in where.items():
            if isinstance(condition, dict):
                for op, value in condition.items():
                    meta_value = metadata.get(key, '')
                    
                    if op == "$eq":
                        if str(meta_value) != str(value):
                            return False
                    elif op == "$ne":
                        if str(meta_value) == str(value):
                            return False
                    elif op == "$contains":
                        if str(value) not in str(meta_value):
                            return False
                    elif op == "$in":
                        if str(meta_value) not in [str(v) for v in value]:
                            return False
            else:
                # Direct equality check
                if str(metadata.get(key, '')) != str(condition):
                    return False
        
        return True

    def get_tag_hierarchy(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """Get hierarchical tag structure with counts."""
        try:
            col = self._collection()
            results = col.get(include=['metadatas'])
            
            tag_hierarchy = {}
            flat_tags = {}
            
            metadatas = results.get('metadatas')
            if metadatas:
                for metadata in metadatas:
                    tags_value = metadata.get('tags', '')
                    if tags_value:
                        note_tags = self._parse_delimited_string(str(tags_value) if tags_value is not None else "")
                    
                    for tag in note_tags:
                        # Count flat tags
                        flat_tags[tag] = flat_tags.get(tag, 0) + 1
                        
                        # Build hierarchy for hierarchical tags
                        if '/' in tag:
                            parts = tag.split('/')
                            current_level = tag_hierarchy
                            
                            for i, part in enumerate(parts):
                                if part not in current_level:
                                    current_level[part] = {
                                        'count': 0,
                                        'children': {},
                                        'full_path': '/'.join(parts[:i+1])
                                    }
                                current_level[part]['count'] += 1
                                current_level = current_level[part]['children']
            
            # Sort and apply limit
            sorted_flat = sorted(flat_tags.items(), key=lambda x: x[1], reverse=True)
            if limit:
                sorted_flat = sorted_flat[:limit]
            
            return {
                'flat_tags': dict(sorted_flat),
                'hierarchy': tag_hierarchy,
                'total_unique_tags': len(flat_tags)
            }
            
        except Exception as e:
            print(f"Error getting tag hierarchy: {e}")
            return {'flat_tags': {}, 'hierarchy': {}, 'total_unique_tags': 0}

    def get_all_tags(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return all tags with frequency counts."""
        try:
            col = self._collection()
            results = col.get(include=['metadatas'])
            
            tag_counts = {}
            metadatas = results.get('metadatas')
            if metadatas:
                for metadata in metadatas:
                    tags = metadata.get('tags', '')
                    if tags:
                        note_tags = self._parse_delimited_string(str(tags) if tags is not None else "")
                    for tag in note_tags:
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            # Sort by count and apply limit
            sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
            if limit:
                sorted_tags = sorted_tags[:limit]
            
            return [{"tag": tag, "count": count} for tag, count in sorted_tags]
            
        except Exception as e:
            print(f"Error getting all tags: {e}")
            return []

    def get_chunk_neighbors(self, chunk_id: str, include_sequential: bool = True, 
                           include_hierarchical: bool = True) -> List[Dict]:
        """Get neighboring chunks (sequential and hierarchical)."""
        try:
            col = self._collection()
            
            # Get the chunk metadata
            results = col.get(
                where={"chunk_id": {"$eq": chunk_id}},
                include=['metadatas']
            )
            
            if not results['metadatas']:
                return []
            
            metadata = results['metadatas'][0]
            neighbors = []
            
            if include_sequential:
                # Add sequential neighbors
                sequential_next = metadata.get('sequential_next', '')
                if sequential_next:
                    neighbors.append({
                        "chunk_id": sequential_next,
                        "relationship": "sequential_next"
                    })
                
                sequential_prev = metadata.get('sequential_prev', '')
                if sequential_prev:
                    neighbors.append({
                        "chunk_id": sequential_prev,
                        "relationship": "sequential_prev"
                    })
            
            if include_hierarchical:
                # Add hierarchical neighbors
                parent_chunk = metadata.get('parent_chunk', '')
                if parent_chunk:
                    neighbors.append({
                        "chunk_id": parent_chunk,
                        "relationship": "parent"
                    })
                
                child_chunks_value = metadata.get('child_chunks', '')
                child_chunks = self._parse_delimited_string(str(child_chunks_value) if child_chunks_value is not None else "")
                for child_chunk in child_chunks:
                    neighbors.append({
                        "chunk_id": child_chunk,
                        "relationship": "child"
                    })
                
                sibling_chunks_value = metadata.get('sibling_chunks', '')
                sibling_chunks = self._parse_delimited_string(str(sibling_chunks_value) if sibling_chunks_value is not None else "")
                for sibling_chunk in sibling_chunks:
                    neighbors.append({
                        "chunk_id": sibling_chunk,
                        "relationship": "sibling"
                    })
            
            # Enrich with chunk details
            for neighbor in neighbors:
                try:
                    neighbor_results = col.get(
                        where={"chunk_id": {"$eq": neighbor["chunk_id"]}},
                        include=['metadatas']
                    )
                    if neighbor_results['metadatas']:
                        neighbor_meta = neighbor_results['metadatas'][0]
                        chunk_type = neighbor_meta.get('chunk_type', '')
                        importance_score = neighbor_meta.get('importance_score', 0.5)
                        header = neighbor_meta.get('header_text', '')
                        
                        neighbor["chunk_type"] = str(chunk_type) if chunk_type is not None else ""
                        neighbor["importance_score"] = float(importance_score) if isinstance(importance_score, (int, float)) else 0.5
                        neighbor["header"] = str(header) if header is not None else ""
                except Exception:
                    continue
            
            return neighbors
            
        except Exception as e:
            print(f"Error getting chunk neighbors for {chunk_id}: {e}")
            return []

    def get_subgraph(self, seed_notes: List[str], depth: int = 1) -> Dict:
        """Extract a subgraph around the specified seed notes."""
        all_connected = set(seed_notes)
        
        # Collect all connected notes
        for seed in seed_notes:
            neighbors = self.get_neighbors(seed, depth)
            all_connected.update([n["id"] for n in neighbors])
        
        # Get node details
        nodes = []
        edges = []
        
        for note_id in all_connected:
            try:
                col = self._collection()
                results = col.get(
                    where={"note_id": {"$eq": note_id}},
                    limit=1,
                    include=['metadatas']
                )
                
                if results['metadatas']:
                    metadata = results['metadatas'][0]
                    nodes.append({
                        "id": note_id,
                        "title": metadata.get('title', ''),
                        "path": metadata.get('path', ''),
                        "vault": metadata.get('vault', ''),
                        "text_length": metadata.get('text_length', 0)
                    })
                    
                    # Add edges
                    links_to_value = metadata.get('links_to', '')
                    links_to = self._parse_delimited_string(str(links_to_value) if links_to_value is not None else "")
                    for target in links_to:
                        if target in all_connected:
                            edges.append({
                                "source": note_id,
                                "target": target,
                                "relationship": "links_to"
                            })
            except Exception:
                continue
        
        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "note_count": len(nodes),
                "link_count": len([e for e in edges if e["relationship"] == "links_to"]),
                "tag_count": 0  # Would need additional computation
            }
        }

    def get_all_notes(self, limit: Optional[int] = None) -> List[Dict]:
        """Get all notes with their metadata."""
        try:
            col = self._collection()
            results = col.get(limit=limit, include=['metadatas'])
            
            notes_map = {}
            metadatas = results.get('metadatas')
            if metadatas:
                for metadata in metadatas:
                    note_id = metadata.get('note_id')
                    if note_id and str(note_id) not in notes_map:
                        # Clean metadata for note representation
                        note_meta = {}
                        for key, value in metadata.items():
                            if not key.startswith('chunk_') and key not in ['semantic_chunk', 'importance_score']:
                                note_meta[key] = value
                        
                        notes_map[str(note_id)] = {
                            "id": str(note_id),
                            "meta": note_meta,
                            "chunk_count": 0
                        }
                    
                    if note_id:
                        notes_map[str(note_id)]["chunk_count"] += 1
            
            return list(notes_map.values())
            
        except Exception as e:
            print(f"Error getting all notes: {e}")
            return []

    def count(self) -> int:
        """Get total number of chunks."""
        try:
            col = self._collection()
            return col.count()
        except Exception:
            return 0

    def get_stats(self) -> Dict:
        """Get statistics about the unified store."""
        try:
            col = self._collection()
            results = col.get(include=['metadatas'])
            
            notes = set()
            tags = set()
            links = 0
            
            metadatas = results.get('metadatas')
            if metadatas:
                for metadata in metadatas:
                    note_id = metadata.get('note_id')
                    if note_id:
                        notes.add(str(note_id))
                    
                    tags_value = metadata.get('tags', '')
                    note_tags = self._parse_delimited_string(str(tags_value) if tags_value is not None else "")
                    tags.update(note_tags)
                    
                    links_value = metadata.get('links_to', '')
                    links_to = self._parse_delimited_string(str(links_value) if links_value is not None else "")
                links += len(links_to)
            
            return {
                "notes": len(notes),
                "tags": len(tags),
                "links": links,
                "total_chunks": len(results.get('ids') or [])
            }
            
        except Exception:
            return {"notes": 0, "tags": 0, "links": 0, "total_chunks": 0}
