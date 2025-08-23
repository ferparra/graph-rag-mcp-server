from __future__ import annotations
from typing import List, Dict, Optional, Any, cast
import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.types import EmbeddingFunction, Embeddable
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pydantic import BaseModel, Field, ConfigDict
from pathlib import Path
from .config import settings
from .fs_indexer import discover_files, parse_note, chunk_text, NoteDoc
from .semantic_chunker import SemanticChunker

class ChromaStore(BaseModel):
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
        ef_impl: SentenceTransformerEmbeddingFunction = SentenceTransformerEmbeddingFunction(model_name=self.embed_model)
        # Cast to the expected union type - this is safe because List[str] ⊆ Union[List[str], List[ndarray]]
        ef = cast(EmbeddingFunction[Embeddable], ef_impl)
        client: ClientAPI = self._client()
        return client.get_or_create_collection(self.collection_name, embedding_function=ef)

    def reindex(self, vaults: List[Path], full_reindex: bool = False) -> int:
        col = self._collection()
        
        if full_reindex:
            try:
                self._client().delete_collection(self.collection_name)
                col: Collection = self._collection()
            except Exception:
                pass
        
        ids, docs, metas = [], [], []
        total_chunks = 0
        
        for path in discover_files(vaults, settings.supported_extensions):
            try:
                nd = parse_note(path)
                if not nd.text:
                    continue
                
                # Use semantic or character chunking based on configuration
                if settings.chunk_strategy == "semantic":
                    semantic_chunks = self.semantic_chunker.chunk_note(nd)
                    
                    for chunk in semantic_chunks:
                        ids.append(chunk.id)
                        docs.append(chunk.content)
                        
                        chunk_meta = nd.meta.copy()
                        chunk_meta.update({
                            "chunk_type": chunk.chunk_type.value,
                            "chunk_position": chunk.position,
                            "header_text": chunk.header_text,
                            "header_level": chunk.header_level,
                            "parent_headers": chunk.parent_headers,
                            "start_line": chunk.start_line,
                            "end_line": chunk.end_line,
                            "contains_tags": chunk.contains_tags,
                            "contains_links": chunk.contains_links,
                            "importance_score": chunk.importance_score,
                            "chunk_id": chunk.id,
                            "semantic_chunk": True
                        })
                        metas.append(self._clean_metadata_for_chroma(chunk_meta))
                else:
                    # Fall back to character chunking
                    chunks: list[str] = chunk_text(nd.text, settings.max_chars, settings.overlap)
                    
                    for k, chunk in enumerate(chunks):
                        chunk_id = f"{nd.id}#chunk={k}"
                        ids.append(chunk_id)
                        docs.append(chunk)
                        
                        chunk_meta = nd.meta.copy()
                        chunk_meta.update({
                            "chunk_index": k,
                            "total_chunks": len(chunks),
                            "chunk_id": chunk_id,
                            "semantic_chunk": False
                        })
                        metas.append(self._clean_metadata_for_chroma(chunk_meta))
                
                if len(ids) >= 512:
                    col.upsert(ids=ids, documents=docs, metadatas=metas)
                    total_chunks += len(ids)
                    ids, docs, metas = [], [], []
                    
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue
        
        if ids:
            col.upsert(ids=ids, documents=docs, metadatas=metas)
            total_chunks += len(ids)
        
        return total_chunks

    def upsert_note(self, note: NoteDoc) -> int:
        col: Collection = self._collection()
        
        existing_ids: list[str] = self._get_note_chunk_ids(note.id)
        if existing_ids:
            col.delete(ids=existing_ids)
        
        if not note.text:
            return 0
        
        ids, docs, metas = [], [], []
        
        # Use semantic or character chunking based on configuration
        if settings.chunk_strategy == "semantic":
            semantic_chunks = self.semantic_chunker.chunk_note(note)
            
            for chunk in semantic_chunks:
                ids.append(chunk.id)
                docs.append(chunk.content)
                
                # Prepare chunk metadata - ensure all values are ChromaDB-compatible
                chunk_meta = {}
                for key, value in note.meta.items():
                    if isinstance(value, list):
                        chunk_meta[key] = ", ".join(str(v) for v in value)
                    elif value is None:
                        chunk_meta[key] = ""
                    else:
                        chunk_meta[key] = value
                
                chunk_meta.update({
                    "chunk_type": chunk.chunk_type.value,
                    "chunk_position": chunk.position,
                    "header_text": chunk.header_text or "",
                    "header_level": chunk.header_level or 0,
                    "parent_headers": ", ".join(chunk.parent_headers) if chunk.parent_headers else "",
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "contains_tags": ", ".join(chunk.contains_tags) if chunk.contains_tags else "",
                    "contains_links": ", ".join(chunk.contains_links) if chunk.contains_links else "",
                    "importance_score": chunk.importance_score,
                    "chunk_id": chunk.id,
                    "semantic_chunk": True
                })
                metas.append(chunk_meta)
        else:
            # Fall back to character chunking
            chunks: list[str] = chunk_text(note.text, settings.max_chars, settings.overlap)
            
            for k, chunk in enumerate(chunks):
                chunk_id: str = f"{note.id}#chunk={k}"
                ids.append(chunk_id)
                docs.append(chunk)
                
                # Prepare chunk metadata - ensure all values are ChromaDB-compatible
                chunk_meta = {}
                for key, value in note.meta.items():
                    if isinstance(value, list):
                        chunk_meta[key] = ", ".join(str(v) for v in value)
                    elif value is None:
                        chunk_meta[key] = ""
                    else:
                        chunk_meta[key] = value
                
                chunk_meta.update({
                    "chunk_index": k,
                    "total_chunks": len(chunks),
                    "chunk_id": chunk_id,
                    "semantic_chunk": False
                })
                metas.append(chunk_meta)
        
        if ids:
            col.upsert(ids=ids, documents=docs, metadatas=metas)
        
        return len(ids)

    def delete_note(self, note_id: str) -> int:
        existing_ids: list[str] = self._get_note_chunk_ids(note_id)
        if existing_ids:
            col: Collection = self._collection()
            col.delete(ids=existing_ids)
            return len(existing_ids)
        return 0

    def _get_note_chunk_ids(self, note_id: str) -> List[str]:
        try:
            col: Collection = self._collection()
            results = col.get(where={"$and": [{"chunk_id": {"$contains": note_id}}]})
            return results['ids'] if results else []
        except Exception:
            return []

    def query(self, q: str, k: int = 6, where: Optional[Dict] = None) -> List[Dict]:
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
                        hit["distance"] = res["distances"][0][i]
                    
                    hits.append(hit)
            
            return hits
            
        except Exception as e:
            print(f"Query error: {e}")
            return []

    def get_all_notes(self, limit: Optional[int] = None) -> List[Dict]:
        col: Collection = self._collection()
        
        try:
            results = col.get(limit=limit)
            
            notes_map = {}
            for i, chunk_id in enumerate(results['ids']):
                # Extract note ID from chunk ID (format: path/chunk_N)
                note_id: str = chunk_id.rsplit('/chunk_', 1)[0] if '/chunk_' in chunk_id else chunk_id
                
                if note_id not in notes_map:
                    if results['metadatas'] and i < len(results['metadatas']):
                        meta: dict[str, bool | float | int | str | None] = dict(results['metadatas'][i])
                        meta.pop('chunk_index', None)
                        meta.pop('total_chunks', None)
                        meta.pop('chunk_id', None)
                    else:
                        meta = {}
                    
                    notes_map[note_id] = {
                        "id": note_id,
                        "meta": meta,
                        "chunk_count": 0
                    }
                
                notes_map[note_id]["chunk_count"] += 1
            
            return list(notes_map.values())
            
        except Exception as e:
            print(f"Error getting all notes: {e}")
            return []

    def count(self) -> int:
        try:
            col = self._collection()
            return col.count()
        except Exception:
            return 0
