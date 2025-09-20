from __future__ import annotations
from typing import Dict, List, Optional, Any, cast, Tuple
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
import logging
# Support both package and module execution contexts
try:
    from config import settings
    from fs_indexer import discover_files, parse_note, chunk_text, NoteDoc, resolve_links
    from semantic_chunker import SemanticChunker
    from resilience import retry_with_backoff, CircuitBreaker
except ImportError:  # When imported as part of a package
    from .config import settings
    from .fs_indexer import discover_files, parse_note, chunk_text, NoteDoc, resolve_links
    from .semantic_chunker import SemanticChunker
    from .resilience import retry_with_backoff, CircuitBreaker

logger = logging.getLogger(__name__)


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

    def __init__(self, **data):
        """Initialize with circuit breaker for ChromaDB connection."""
        super().__init__(**data)
        self._client_cache: Optional[ClientAPI] = None
        self._collection_cache: Optional[Collection] = None
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=30.0
        )
        self._retry_counter: int = 0
        self._warnings: List[str] = []

    @retry_with_backoff(
        max_attempts=3,
        initial_delay=1.0,
        exceptions=(Exception,)
    )
    def _client(self) -> ClientAPI:
        """Get ChromaDB client with connection resilience."""
        if self._client_cache is not None:
            # Validate cached client is still healthy
            try:
                # Simple health check - list collections
                self._client_cache.list_collections()
                return self._client_cache
            except Exception:
                logger.warning("Cached ChromaDB client is unhealthy, reconnecting...")
                self._client_cache = None
        
        def create_client():
            return chromadb.PersistentClient(path=str(self.client_dir))
        
        try:
            client = self._circuit_breaker.call(create_client)
            self._client_cache = client
            logger.info(f"Connected to ChromaDB at {self.client_dir}")
            return client
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            raise

    def _clean_metadata_for_chroma(self, metadata: Dict[str, Any], max_value_length: int = 50000) -> Dict[str, Any]:
        """
        Clean metadata to ensure all values are ChromaDB-compatible (str, int, float, bool, None).
        Includes comprehensive safety checks for serialization.
        """
        import json
        
        cleaned = {}
        seen_objects = set()  # Track objects to prevent circular references
        
        def clean_value(value: Any, depth: int = 0) -> Any:
            """Recursively clean values with depth protection."""
            if depth > 5:  # Prevent deep recursion
                logger.warning("Max recursion depth reached in metadata cleaning")
                return str(value)[:100] if value else None
            
            # Check for circular references
            if id(value) in seen_objects:
                return "[Circular Reference]"
            
            if isinstance(value, (dict, list, tuple)) and id(value) not in seen_objects:
                seen_objects.add(id(value))
            
            try:
                if value is None:
                    return None
                elif isinstance(value, bool):  # Check bool before int (bool is subclass of int)
                    return value
                elif isinstance(value, (int, float)):
                    # Ensure numbers are within safe ranges
                    if isinstance(value, float) and (value != value or value == float('inf') or value == float('-inf')):
                        return None  # Handle NaN and infinity
                    return value
                elif isinstance(value, str):
                    # Truncate extremely long strings
                    if len(value) > max_value_length:
                        logger.warning(f"Truncating long string value of length {len(value)}")
                        return value[:max_value_length] + "...[truncated]"
                    return value
                elif isinstance(value, (list, tuple)):
                    # Convert lists to comma-separated strings, with safety checks
                    if len(value) > 100:  # Limit list size
                        logger.warning(f"Large list of {len(value)} items, truncating to 100")
                        value = value[:100]
                    cleaned_items = [str(clean_value(v, depth + 1)) for v in value]
                    result = ",".join(cleaned_items) if cleaned_items else ""
                    if len(result) > max_value_length:
                        result = result[:max_value_length] + "...[truncated]"
                    return result
                elif isinstance(value, dict):
                    # Convert dicts to JSON strings with safety
                    if len(value) > 50:  # Limit dict size
                        logger.warning(f"Large dict with {len(value)} keys, truncating")
                        value = dict(list(value.items())[:50])
                    cleaned_dict = {k: clean_value(v, depth + 1) for k, v in value.items()}
                    result = json.dumps(cleaned_dict, default=str)
                    if len(result) > max_value_length:
                        result = result[:max_value_length] + "...[truncated]"
                    return result
                else:
                    # Convert other types to strings safely
                    result = str(value)
                    if len(result) > max_value_length:
                        result = result[:max_value_length] + "...[truncated]"
                    return result
            except Exception as e:
                logger.error(f"Error cleaning metadata value: {e}")
                return "[Serialization Error]"
        
        # Clean each metadata field
        for key, value in metadata.items():
            # Ensure keys are also safe
            if not isinstance(key, str):
                key = str(key)
            if len(key) > 100:  # ChromaDB has limits on key length
                logger.warning(f"Truncating long metadata key: {key}")
                key = key[:100]
            
            cleaned[key] = clean_value(value)
        
        return cleaned

    @retry_with_backoff(
        max_attempts=3,
        initial_delay=1.0,
        exceptions=(Exception,)
    )
    def _collection(self) -> Collection:
        """Get ChromaDB collection with connection resilience and caching."""
        # Check if we have a healthy cached collection
        if self._collection_cache is not None:
            try:
                # Validate collection is still accessible
                self._collection_cache.count()
                return self._collection_cache
            except Exception:
                logger.warning("Cached collection is unhealthy, recreating...")
                self._collection_cache = None
        
        # Prefer SentenceTransformer embeddings; fall back to a lightweight default in offline/test environments
        try:
            ef_impl: SentenceTransformerEmbeddingFunction = SentenceTransformerEmbeddingFunction(
                model_name=self.embed_model
            )
            primary_embedding = cast(EmbeddingFunction[Embeddable], ef_impl)
            logger.debug(f"Using SentenceTransformer embedding: {self.embed_model}")
        except Exception as e:
            logger.warning(f"Failed to load SentenceTransformer {self.embed_model}: {e}. Using default embeddings.")
            primary_embedding = cast(EmbeddingFunction[Embeddable], DefaultEmbeddingFunction())

        fallback_embedding = cast(EmbeddingFunction[Embeddable], DefaultEmbeddingFunction())
        ef = cast(
            EmbeddingFunction[Embeddable],
            self._SafeEmbeddingFunction(self, primary_embedding, fallback_embedding)
        )

        client: ClientAPI = self._client()
        self._collection_cache = client.get_or_create_collection(self.collection_name, embedding_function=ef)
        logger.info(f"Connected to collection '{self.collection_name}' with {self._collection_cache.count()} documents")
        return self._collection_cache

    class _SafeEmbeddingFunction:
        """Embedding function with retry and fallback support."""

        def __init__(
            self,
            store: "UnifiedStore",
            primary: EmbeddingFunction[Embeddable],
            fallback: EmbeddingFunction[Embeddable],
        ) -> None:
            self._store = store
            self._primary = primary
            self._fallback = fallback

        def __call__(self, input: List[str]) -> List[List[float]]:
            texts = input
            def _normalize(raw_embeddings: Any) -> List[List[float]]:
                normalized: List[List[float]] = []
                if raw_embeddings is None:
                    return normalized
                for vector in raw_embeddings:
                    if vector is None:
                        normalized.append([])
                        continue
                    normalized.append([float(element) for element in list(vector)])
                return normalized

            @retry_with_backoff(
                max_attempts=3,
                initial_delay=0.75,
                exceptions=(Exception,),
            )
            def _call_primary() -> List[List[float]]:
                return _normalize(self._primary(texts))

            try:
                return _call_primary()
            except Exception as exc:
                if not settings.enable_embed_fallback:
                    raise
                logger.warning("Primary embedding function failed, using fallback: %s", exc)
                self._store._record_warning("fallback_embedding")
                return _normalize(self._fallback(texts))

        def name(self) -> str:
            if hasattr(self._primary, "name"):
                try:
                    return str(self._primary.name())
                except Exception:
                    pass
            return "default"

        def is_legacy(self) -> bool:  # pragma: no cover - compatibility hook
            if hasattr(self._primary, "is_legacy"):
                try:
                    return bool(self._primary.is_legacy())
                except Exception:
                    pass
            return False

    def _record_warning(self, warning: str) -> None:
        self._warnings.append(warning)

    def reset_operation_metrics(self) -> None:
        self._retry_counter = 0
        self._warnings = []

    def consume_operation_metrics(self) -> Tuple[int, List[str]]:
        retries = self._retry_counter
        warnings = list(self._warnings)
        self._retry_counter = 0
        self._warnings = []
        return retries, warnings

    def get_circuit_breaker_state(self) -> str:
        return self._circuit_breaker.state.value

    def _safe_collection_call(self, method_name: str, *args, collection: Optional[Collection] = None, **kwargs):
        retry_tracker = {"count": 0}

        def _on_retry(_: Exception, __: int) -> None:
            retry_tracker["count"] += 1

        @retry_with_backoff(
            max_attempts=3,
            initial_delay=0.75,
            exceptions=(Exception,),
            on_retry=_on_retry,
        )
        def _runner():
            target_collection = collection or self._collection()
            method = getattr(target_collection, method_name)
            return self._circuit_breaker.call(method, *args, **kwargs)

        result = _runner()
        self._retry_counter += retry_tracker["count"]
        return result

    def _safe_get(self, *, collection: Optional[Collection] = None, **kwargs):
        return self._safe_collection_call('get', collection=collection, **kwargs)

    def _safe_query(self, *, collection: Optional[Collection] = None, **kwargs):
        return self._safe_collection_call('query', collection=collection, **kwargs)

    def _safe_upsert(self, *, collection: Optional[Collection] = None, **kwargs):
        return self._safe_collection_call('upsert', collection=collection, **kwargs)

    @staticmethod
    def _flatten_chroma_field(data: Any) -> List[Any]:
        """Flatten Chroma responses that occasionally nest values in lists."""
        if not isinstance(data, list):
            return list(data) if isinstance(data, tuple) else ([] if data is None else [data])
        if data and isinstance(data[0], list):
            flattened: List[Any] = []
            for row in data:
                if isinstance(row, list):
                    flattened.extend(row)
                elif row is not None:
                    flattened.append(row)
            return flattened
        return data

    @classmethod
    def _normalize_get_rows(cls, results: Dict[str, Any], include_docs: bool = True) -> List[Dict[str, Any]]:
        """Normalize `collection.get` responses into predictable records."""
        ids = cls._flatten_chroma_field(results.get("ids") or [])
        metadatas = cls._flatten_chroma_field(results.get("metadatas") or [])
        documents = cls._flatten_chroma_field(results.get("documents") or []) if include_docs else []

        normalized: List[Dict[str, Any]] = []
        for idx, raw_id in enumerate(ids):
            chunk_id = str(raw_id) if raw_id is not None else ""
            meta_candidate = metadatas[idx] if idx < len(metadatas) else {}
            metadata = meta_candidate if isinstance(meta_candidate, dict) else {}

            document = ""
            if include_docs and idx < len(documents):
                doc_candidate = documents[idx]
                if isinstance(doc_candidate, str):
                    document = doc_candidate
                elif doc_candidate is None:
                    document = ""
                else:
                    document = str(doc_candidate)

            normalized.append({
                "id": chunk_id,
                "meta": metadata,
                "document": document,
                "text": document,
            })

        return normalized

    def fetch_chunks(self, chunk_ids: List[str], include_docs: bool = True) -> Dict[str, Dict[str, Any]]:
        """Batch hydrate chunk metadata/documents by their IDs."""
        if not chunk_ids:
            return {}

        ordered_unique_ids: List[str] = []
        seen: set[str] = set()
        for chunk_id in chunk_ids:
            if not chunk_id:
                continue
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            ordered_unique_ids.append(chunk_id)

        if not ordered_unique_ids:
            return {}

        include_fields = ['metadatas']
        if include_docs:
            include_fields.append('documents')

        try:
            results = self._safe_get(ids=ordered_unique_ids, include=include_fields)
        except Exception as exc:
            logger.error("Failed to fetch chunks %s: %s", ordered_unique_ids[:5], exc)
            return {}

        hydrated_rows = self._normalize_get_rows(results, include_docs=include_docs)
        return {row["id"]: row for row in hydrated_rows if row.get("id")}

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
            results = self._safe_get(include=['metadatas'])
        
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
            except Exception:
                logger.exception("Error parsing %s", path)
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
                    self._safe_upsert(ids=ids, documents=docs, metadatas=metas)
                    total_chunks += len(ids)
                    ids, docs, metas = [], [], []
                    
            except Exception:
                logger.exception("Error processing note %s", note.id)
                continue
        
        if ids:
            self._safe_upsert(ids=ids, documents=docs, metadatas=metas)
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
            self._safe_upsert(ids=ids, documents=docs, metadatas=metas)

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
            results = self._safe_get(where={"note_id": {"$eq": note_id}})
            return results.get('ids', [])
        except Exception:
            return []

    def query(self, q: str, k: int = 6, where: Optional[Dict] = None) -> List[Dict]:
        """Vector similarity search with metadata."""
        try:
            collection = self._collection()
            res = self._safe_query(
                query_texts=[q],
                n_results=min(k, collection.count()),
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
            
        except Exception:
            logger.exception("Vector query failed for '%s'", q)
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
                    collection = self._collection()
                    results = self._safe_get(
                        collection=collection,
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
                            tag_results = self._safe_get(
                                collection=collection,
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
                                neighbor_results = self._safe_get(
                                    collection=collection,
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
                
                except Exception:
                    logger.exception("Error retrieving neighbors for %s", current_note)
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
            collection = self._collection()
            # Find chunks where links_to contains this note_id
            # Note: ChromaDB doesn't have substring search, so we need to get all and filter
            results = self._safe_get(collection=collection, include=['metadatas'])
            
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
                    note_results = self._safe_get(
                        collection=collection,
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
            
        except Exception:
            logger.exception("Error getting backlinks for %s", note_id)
            return []

    def get_notes_by_tag(self, tag: str) -> List[Dict]:
        """Get all notes that have the specified tag."""
        try:
            collection = self._collection()
            # Find chunks where tags contains this tag
            results = self._safe_get(collection=collection, include=['metadatas'])
            
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
                    note_results = self._safe_get(
                        collection=collection,
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
            
        except Exception:
            logger.exception("Error getting notes by tag %s", tag)
            return []

    def fuzzy_tag_search(self, entities: List[str], k: int = 6, where: Optional[Dict] = None) -> List[Dict]:
        """Enhanced tag search with fuzzy matching and hierarchical support."""
        try:
            collection = self._collection()
            results = self._safe_get(collection=collection, include=['metadatas', 'documents'])
            
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
            
        except Exception:
            logger.exception("Error executing fuzzy tag search for %s", entities)
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
            collection = self._collection()
            results = self._safe_get(collection=collection, include=['metadatas'])
            
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
            
        except Exception:
            logger.exception("Error building tag hierarchy")
            return {'flat_tags': {}, 'hierarchy': {}, 'total_unique_tags': 0}

    def get_all_tags(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return all tags with frequency counts."""
        try:
            collection = self._collection()
            results = self._safe_get(collection=collection, include=['metadatas'])
            
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
            
        except Exception:
            logger.exception("Error retrieving all tags")
            return []

    def get_chunk_neighbors(self, chunk_id: str, include_sequential: bool = True, 
                           include_hierarchical: bool = True) -> List[Dict]:
        """Get neighboring chunks (sequential and hierarchical)."""
        try:
            collection = self._collection()
            # Get the chunk metadata
            results = self._safe_get(
                collection=collection,
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
            
            # Enrich with chunk details using batch hydration
            neighbor_ids: List[str] = []
            for neighbor in neighbors:
                raw_neighbor_id = neighbor.get("chunk_id")
                if isinstance(raw_neighbor_id, str) and raw_neighbor_id:
                    neighbor_ids.append(raw_neighbor_id)
            if neighbor_ids:
                hydrated = self.fetch_chunks(neighbor_ids, include_docs=False)
                for neighbor in neighbors:
                    raw_chunk_id = neighbor.get("chunk_id")
                    if not isinstance(raw_chunk_id, str) or not raw_chunk_id:
                        continue
                    chunk_id = raw_chunk_id
                    row = hydrated.get(chunk_id)
                    if not row:
                        continue
                    meta = row.get("meta") or {}
                    chunk_type = meta.get('chunk_type', '')
                    importance_score = meta.get('importance_score', 0.5)
                    header = meta.get('header_text', '')

                    neighbor["chunk_type"] = str(chunk_type) if chunk_type is not None else ""
                    if isinstance(importance_score, (int, float)):
                        neighbor["importance_score"] = float(importance_score)
                    neighbor["header"] = str(header) if header is not None else ""
                    neighbor["meta"] = meta

            return neighbors

        except Exception:
            logger.exception("Error retrieving chunk neighbors for %s", chunk_id)
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
                collection = self._collection()
                results = self._safe_get(
                    collection=collection,
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
            collection = self._collection()
            results = self._safe_get(collection=collection, limit=limit, include=['metadatas'])
            
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
            
        except Exception:
            logger.exception("Error retrieving all notes")
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
            collection = self._collection()
            results = self._safe_get(collection=collection, include=['metadatas'])
            
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
