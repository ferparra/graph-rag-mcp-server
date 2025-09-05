from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Dict, Optional, Any
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDFS
from .fs_indexer import discover_files, parse_note, NoteDoc, resolve_links
from .semantic_chunker import SemanticChunker
from .config import settings

# Define our vault ontology namespaces
VAULT = Namespace("http://obsidian-vault.local/ontology#")
NOTES = Namespace("http://obsidian-vault.local/notes/")
TAGS = Namespace("http://obsidian-vault.local/tags/")
CHUNKS = Namespace("http://obsidian-vault.local/chunks/")

# Define commonly used RDF/XSD URIs as explicit constants for type safety
RDF_TYPE = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
XSD_INTEGER = URIRef("http://www.w3.org/2001/XMLSchema#integer") 
XSD_DECIMAL = URIRef("http://www.w3.org/2001/XMLSchema#decimal")
XSD_BOOLEAN = URIRef("http://www.w3.org/2001/XMLSchema#boolean")

def get_query_attr(row: Any, attr: str) -> Any:
    """Safely get attribute from SPARQL query result row.
    
    RDFLib ResultRow objects support direct attribute access for named variables
    in SPARQL queries (e.g., row.subject, row.predicate).
    """
    if hasattr(row, attr):
        return getattr(row, attr)
    elif hasattr(row, 'get'):
        # ResultRow has a get method for safe access
        return row.get(attr)
    else:
        # Fallback - shouldn't normally be needed
        return None

class RDFGraphStore:
    def __init__(self, db_path: Path, store_identifier: str = "obsidian_vault_graph"):
        self.db_path = db_path
        self.store_identifier = store_identifier
        
        # Create Oxigraph-backed RDF store
        # Note: Oxigraph uses a directory, not a file
        store_dir = db_path.parent / f"{db_path.stem}_oxigraph"
        store_dir.mkdir(parents=True, exist_ok=True)
        
        # Create the graph with the Oxigraph store
        # Oxigraph requires absolute URIs for graph identifiers
        graph_uri = f"http://obsidian-vault.local/graphs/{store_identifier}"
        self.graph = Graph(store="Oxigraph", identifier=URIRef(graph_uri))
        
        # Open the store directory
        self.graph.open(str(store_dir.absolute()), create=True)
        
        # Bind namespaces for cleaner SPARQL queries
        self.graph.bind("vault", VAULT)
        self.graph.bind("notes", NOTES)
        self.graph.bind("tags", TAGS)
        self.graph.bind("chunks", CHUNKS)
        self.graph.bind("rdfs", RDFS)
        
        # Initialize semantic chunker
        self.semantic_chunker = SemanticChunker(
            min_chunk_size=settings.semantic_min_chunk_size,
            max_chunk_size=settings.semantic_max_chunk_size,
            merge_threshold=settings.semantic_merge_threshold,
            include_context=settings.semantic_include_context
        )
    
    def close(self):
        """Close the database connection."""
        try:
            self.graph.close()
        except Exception:
            pass
    
    def clear_graph(self):
        """Remove all triples from the graph."""
        self.graph.remove((None, None, None))
    
    def _safe_uri_part(self, text: str) -> str:
        """Convert text to a safe URI component."""
        import urllib.parse
        # First, encode special characters
        safe_text = urllib.parse.quote(text, safe='')
        # Make it more readable by allowing some characters
        safe_text = safe_text.replace('%2F', '_')  # / -> _
        safe_text = safe_text.replace('%20', '_')  # space -> _
        safe_text = safe_text.replace('%23', '_')  # # -> _
        safe_text = safe_text.replace('%0A', '_')  # newline -> _
        safe_text = safe_text.replace('%0D', '_')  # carriage return -> _
        return safe_text
    
    def _decode_uri_part(self, uri_str: str, prefix: str = "") -> str:
        """Decode a URI component back to original text."""
        import urllib.parse
        # Remove the namespace prefix if present
        if prefix:
            uri_str = uri_str.replace(str(prefix), "")
        # Decode the URI component
        decoded = urllib.parse.unquote(uri_str)
        return decoded
    
    def _note_uri(self, note_id: str) -> URIRef:
        """Generate URI for a note."""
        return NOTES[self._safe_uri_part(note_id)]
    
    def _tag_uri(self, tag: str) -> URIRef:
        """Generate URI for a tag."""
        return TAGS[self._safe_uri_part(tag)]
    
    def _chunk_uri(self, chunk_id: str) -> URIRef:
        """Generate URI for a chunk."""
        return CHUNKS[self._safe_uri_part(chunk_id)]
    
    def upsert_note(self, note: NoteDoc, all_notes: Optional[Dict[str, NoteDoc]] = None):
        """Add or update a note in the RDF graph."""
        note_uri = self._note_uri(note.id)
        
        # Remove existing triples for this note
        self.graph.remove((note_uri, None, None))
        
        # Add note type and basic properties
        self.graph.add((note_uri, RDF_TYPE, VAULT.Note))
        self.graph.add((note_uri, VAULT.hasTitle, Literal(note.title)))
        self.graph.add((note_uri, VAULT.hasPath, Literal(str(note.path))))
        self.graph.add((note_uri, VAULT.hasVault, Literal(note.meta.get("vault", ""))))
        self.graph.add((note_uri, VAULT.hasTextLength, Literal(len(note.text), datatype=XSD_INTEGER)))
        
        # Add frontmatter properties
        for key, value in note.frontmatter.items():
            if isinstance(value, str):
                self.graph.add((note_uri, VAULT[f"fm_{key}"], Literal(value)))
            elif isinstance(value, (int, float)):
                self.graph.add((note_uri, VAULT[f"fm_{key}"], Literal(value, datatype=XSD_DECIMAL)))
            elif isinstance(value, bool):
                self.graph.add((note_uri, VAULT[f"fm_{key}"], Literal(value, datatype=XSD_BOOLEAN)))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        self.graph.add((note_uri, VAULT[f"fm_{key}"], Literal(item)))
        
        # Add tag relationships
        for tag in note.tags:
            tag_uri = self._tag_uri(tag)
            self.graph.add((tag_uri, RDF_TYPE, VAULT.Tag))
            self.graph.add((tag_uri, VAULT.hasName, Literal(tag)))
            self.graph.add((note_uri, VAULT.hasTag, tag_uri))
        
        # Add link relationships if all_notes provided
        if all_notes:
            resolved_links = resolve_links(note, all_notes)
            for linked_note_id in resolved_links:
                linked_note_uri = self._note_uri(linked_note_id)
                self.graph.add((note_uri, VAULT.linksTo, linked_note_uri))
        
        # Add semantic chunks if using semantic chunking
        if settings.chunk_strategy == "semantic":
            self._add_semantic_chunks(note, note_uri)
    
    def _add_semantic_chunks(self, note: NoteDoc, note_uri: URIRef):
        """Add semantic chunks for a note to the RDF graph."""
        semantic_chunks = self.semantic_chunker.chunk_note(note)
        
        for chunk in semantic_chunks:
            chunk_uri = self._chunk_uri(chunk.id)
            
            # Add chunk type and basic properties
            self.graph.add((chunk_uri, RDF_TYPE, VAULT.Chunk))
            self.graph.add((chunk_uri, VAULT.belongsToNote, note_uri))
            self.graph.add((chunk_uri, VAULT.chunkType, Literal(chunk.chunk_type.value)))
            self.graph.add((chunk_uri, VAULT.chunkPosition, Literal(chunk.position, datatype=XSD_INTEGER)))
            self.graph.add((chunk_uri, VAULT.importanceScore, Literal(chunk.importance_score, datatype=XSD_DECIMAL)))
            self.graph.add((chunk_uri, VAULT.startLine, Literal(chunk.start_line, datatype=XSD_INTEGER)))
            self.graph.add((chunk_uri, VAULT.endLine, Literal(chunk.end_line, datatype=XSD_INTEGER)))
            
            # Add header information if present
            if chunk.header_text:
                self.graph.add((chunk_uri, VAULT.hasHeader, Literal(chunk.header_text)))
                if chunk.header_level:
                    self.graph.add((chunk_uri, VAULT.headerLevel, Literal(chunk.header_level, datatype=XSD_INTEGER)))
            
            # Add parent headers as context
            for i, parent_header in enumerate(chunk.parent_headers):
                self.graph.add((chunk_uri, VAULT.hasParentHeader, Literal(parent_header)))
                self.graph.add((chunk_uri, VAULT[f"parentHeader_{i}"], Literal(parent_header)))
            
            # Add tags contained in chunk
            for tag in chunk.contains_tags:
                tag_uri = self._tag_uri(tag)
                self.graph.add((chunk_uri, VAULT.containsTag, tag_uri))
            
            # Add links contained in chunk
            for link in chunk.contains_links:
                self.graph.add((chunk_uri, VAULT.containsLink, Literal(link)))
        
        # Add sequential relationships between chunks
        for i in range(len(semantic_chunks) - 1):
            current_chunk_uri = self._chunk_uri(semantic_chunks[i].id)
            next_chunk_uri = self._chunk_uri(semantic_chunks[i + 1].id)
            self.graph.add((current_chunk_uri, VAULT.followedBy, next_chunk_uri))
            self.graph.add((next_chunk_uri, VAULT.precededBy, current_chunk_uri))
        
        # Add hierarchical relationships for sections
        for chunk in semantic_chunks:
            if chunk.chunk_type.value in ['section', 'paragraph'] and chunk.header_level:
                # Find parent sections (higher level headers)
                for other_chunk in semantic_chunks:
                    if (other_chunk.position < chunk.position and 
                        other_chunk.header_level and 
                        other_chunk.header_level < chunk.header_level):
                        parent_chunk_uri = self._chunk_uri(other_chunk.id)
                        child_chunk_uri = self._chunk_uri(chunk.id)
                        self.graph.add((child_chunk_uri, VAULT.hasParentSection, parent_chunk_uri))
                        self.graph.add((parent_chunk_uri, VAULT.hasChildSection, child_chunk_uri))
                        break  # Only connect to immediate parent
    
    def delete_note(self, note_id: str):
        """Remove a note and all its relationships from the graph."""
        note_uri = self._note_uri(note_id)
        
        # Remove chunks associated with this note
        if settings.chunk_strategy == "semantic":
            # Find and remove all chunks belonging to this note
            chunks_query = f"""
            SELECT ?chunk WHERE {{
                ?chunk vault:belongsToNote <{note_uri}> .
            }}
            """
            
            for row in self.graph.query(chunks_query):
                chunk_uri = get_query_attr(row, 'chunk')
                # Remove all triples where chunk is subject or object
                self.graph.remove((chunk_uri, None, None))
                self.graph.remove((None, None, chunk_uri))
        
        # Remove all triples where this note is subject or object
        self.graph.remove((note_uri, None, None))
        self.graph.remove((None, None, note_uri))
    
    def build_from_notes(self, vaults: Iterable[Path]) -> int:
        """Build the complete graph from vault files."""
        self.clear_graph()
        
        # First pass: collect all notes
        all_notes = {}
        for path in discover_files(vaults, settings.supported_extensions):
            try:
                note = parse_note(path)
                all_notes[note.id] = note
            except Exception as e:
                print(f"Error parsing {path}: {e}")
                continue
        
        # Second pass: add notes with relationships
        for note in all_notes.values():
            try:
                self.upsert_note(note, all_notes)
            except Exception as e:
                print(f"Error indexing {note.id}: {e}")
                continue
        
        return len(all_notes)
    
    def get_neighbors(self, note_id: str, depth: int = 1, relationship_types: Optional[List[str]] = None) -> List[Dict]:
        """Get neighboring notes up to specified depth."""
        note_uri = self._note_uri(note_id)
        
        if relationship_types is None:
            relationship_types = ["linksTo", "hasTag"]
        
        # Build SPARQL query for neighbors
        predicates = " | ".join([f"vault:{rel}" for rel in relationship_types])
        
        query = f"""
        PREFIX vault: <{VAULT}>
        PREFIX notes: <{NOTES}>
        
        SELECT DISTINCT ?connected ?title ?path ?type
        WHERE {{
            ?start vault:hasPath ?startPath .
            ?start ({predicates})+ ?connected .
            ?connected vault:hasTitle ?title .
            OPTIONAL {{ ?connected vault:hasPath ?path }}
            OPTIONAL {{ ?connected a ?type }}
            FILTER(?start = <{note_uri}>)
            FILTER(?connected != ?start)
        }}
        ORDER BY ?title
        """
        
        results = []
        for row in self.graph.query(query):
            connected = get_query_attr(row, 'connected')
            title = get_query_attr(row, 'title')
            path = get_query_attr(row, 'path')
            node_type = get_query_attr(row, 'type')
            
            # Decode the URI back to original ID
            if str(connected).startswith(str(NOTES)):
                decoded_id = self._decode_uri_part(str(connected), NOTES)
            elif str(connected).startswith(str(TAGS)):
                decoded_id = self._decode_uri_part(str(connected), TAGS)
            else:
                decoded_id = str(connected)
            
            result = {
                "id": decoded_id,
                "title": str(title) if title else "",
                "path": str(path) if path else "",
                "type": str(node_type) if node_type else ""
            }
            results.append(result)
        
        return results
    
    def get_backlinks(self, note_id: str) -> List[Dict]:
        """Get all notes that link to the specified note."""
        note_uri = self._note_uri(note_id)
        
        query = f"""
        PREFIX vault: <{VAULT}>
        
        SELECT ?source ?title ?path
        WHERE {{
            ?source vault:linksTo <{note_uri}> .
            ?source vault:hasTitle ?title .
            ?source vault:hasPath ?path .
        }}
        ORDER BY ?title
        """
        
        results = []
        for row in self.graph.query(query):
            source = get_query_attr(row, 'source')
            title = get_query_attr(row, 'title')
            path = get_query_attr(row, 'path')
            
            result = {
                "id": self._decode_uri_part(str(source), NOTES),
                "title": str(title),
                "path": str(path)
            }
            results.append(result)
        
        return results

    def get_all_tags(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return all tags with optional frequency counts.

        Returns a list of dicts like {"tag": str, "count": int} ordered by count desc.
        """
        limit_clause = f"LIMIT {int(limit)}" if isinstance(limit, int) and limit > 0 else ""
        query = f"""
        PREFIX vault: <{VAULT}>
        
        SELECT ?tagName (COUNT(?note) AS ?count)
        WHERE {{
            ?tag a vault:Tag .
            ?tag vault:hasName ?tagName .
            OPTIONAL {{ ?note vault:hasTag ?tag }}
        }}
        GROUP BY ?tagName
        ORDER BY DESC(?count) ?tagName
        {limit_clause}
        """
        results: List[Dict[str, Any]] = []
        for row in self.graph.query(query):
            tag_name = get_query_attr(row, 'tagName')
            count = get_query_attr(row, 'count')
            results.append({
                "tag": str(tag_name) if tag_name is not None else "",
                "count": int(count) if count is not None else 0,
            })
        return results
    
    def get_tags_for_note(self, note_id: str) -> List[str]:
        """Get all tags for a specific note."""
        note_uri = self._note_uri(note_id)
        
        query = f"""
        PREFIX vault: <{VAULT}>
        
        SELECT ?tagName
        WHERE {{
            <{note_uri}> vault:hasTag ?tag .
            ?tag vault:hasName ?tagName .
        }}
        ORDER BY ?tagName
        """
        
        return [str(get_query_attr(row, 'tagName')) for row in self.graph.query(query)]
    
    def get_notes_by_tag(self, tag: str) -> List[Dict]:
        """Get all notes that have the specified tag."""
        tag_uri = self._tag_uri(tag)
        
        query = f"""
        PREFIX vault: <{VAULT}>
        
        SELECT ?note ?title ?path
        WHERE {{
            ?note vault:hasTag <{tag_uri}> .
            ?note vault:hasTitle ?title .
            ?note vault:hasPath ?path .
        }}
        ORDER BY ?title
        """
        
        results = []
        for row in self.graph.query(query):
            note = get_query_attr(row, 'note')
            title = get_query_attr(row, 'title')
            path = get_query_attr(row, 'path')
            
            result = {
                "id": self._decode_uri_part(str(note), NOTES),
                "title": str(title),
                "path": str(path)
            }
            results.append(result)
        
        return results
    
    def get_subgraph(self, seed_notes: List[str], depth: int = 1) -> Dict:
        """Extract a subgraph around the specified seed notes."""
        # Collect all connected notes
        all_connected = set(seed_notes)
        
        for seed in seed_notes:
            neighbors = self.get_neighbors(seed, depth)
            all_connected.update([n["id"] for n in neighbors])
        
        # Build URIs for connected notes
        note_uris = [f"<{self._note_uri(note_id)}>" for note_id in all_connected]
        uris_filter = " ".join(note_uris)
        
        # Get nodes
        nodes_query = f"""
        PREFIX vault: <{VAULT}>
        
        SELECT ?note ?title ?path ?vault ?textLength
        WHERE {{
            ?note vault:hasTitle ?title .
            ?note vault:hasPath ?path .
            OPTIONAL {{ ?note vault:hasVault ?vault }}
            OPTIONAL {{ ?note vault:hasTextLength ?textLength }}
            FILTER(?note IN ({uris_filter}))
        }}
        ORDER BY ?title
        """
        
        nodes = []
        for row in self.graph.query(nodes_query):
            note_val = get_query_attr(row, 'note')
            title_val = get_query_attr(row, 'title')
            path_val = get_query_attr(row, 'path')
            vault_val = get_query_attr(row, 'vault')
            text_len_val = get_query_attr(row, 'textLength')

            node = {
                "id": self._decode_uri_part(str(note_val), NOTES) if note_val is not None else "",
                "title": str(title_val) if title_val is not None else "",
                "path": str(path_val) if path_val is not None else "",
                "vault": str(vault_val) if vault_val is not None else "",
                "text_length": int(text_len_val) if text_len_val is not None else 0
            }
            nodes.append(node)
        
        # Get edges (links)
        edges_query = f"""
        PREFIX vault: <{VAULT}>
        
        SELECT ?source ?target ?relationship
        WHERE {{
            ?source ?rel ?target .
            ?source vault:hasTitle ?sourceTitle .
            ?target vault:hasTitle ?targetTitle .
            FILTER(?source IN ({uris_filter}))
            FILTER(?target IN ({uris_filter}))
            FILTER(?rel IN (vault:linksTo, vault:hasTag))
            BIND(IF(?rel = vault:linksTo, "links_to", "has_tag") AS ?relationship)
        }}
        """
        
        edges = []
        for row in self.graph.query(edges_query):
            source_val = get_query_attr(row, 'source')
            target_val = get_query_attr(row, 'target')
            rel_val = get_query_attr(row, 'relationship')

            target_str = str(target_val) if target_val is not None else ""
            if target_str.startswith(str(NOTES)):
                decoded_target = self._decode_uri_part(target_str, NOTES)
            elif target_str.startswith(str(TAGS)):
                decoded_target = self._decode_uri_part(target_str, TAGS)
            else:
                decoded_target = target_str

            edge = {
                "source": self._decode_uri_part(str(source_val), NOTES) if source_val is not None else "",
                "target": decoded_target,
                "relationship": str(rel_val) if rel_val is not None else ""
            }
            edges.append(edge)
        
        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "note_count": len([n for n in nodes if n.get("text_length", 0) > 0]),
                "link_count": len([e for e in edges if e["relationship"] == "links_to"]),
                "tag_count": len([e for e in edges if e["relationship"] == "has_tag"])
            }
        }
    
    def get_chunks_by_importance(self, min_score: float = 0.7, limit: int = 20) -> List[Dict]:
        """Get chunks ordered by importance score."""
        if settings.chunk_strategy != "semantic":
            return []
        
        query = f"""
        PREFIX vault: <{VAULT}>
        PREFIX chunks: <{CHUNKS}>
        
        SELECT ?chunk ?note ?chunkType ?importance ?header ?headerLevel ?content
        WHERE {{
            ?chunk a vault:Chunk .
            ?chunk vault:belongsToNote ?note .
            ?chunk vault:chunkType ?chunkType .
            ?chunk vault:importanceScore ?importance .
            OPTIONAL {{ ?chunk vault:hasHeader ?header }}
            OPTIONAL {{ ?chunk vault:headerLevel ?headerLevel }}
            FILTER(?importance >= {min_score})
        }}
        ORDER BY DESC(?importance)
        LIMIT {limit}
        """
        
        results = []
        for row in self.graph.query(query):
            chunk = get_query_attr(row, 'chunk')
            note = get_query_attr(row, 'note')
            chunk_type = get_query_attr(row, 'chunkType')
            importance = get_query_attr(row, 'importance')
            header = get_query_attr(row, 'header')
            header_level = get_query_attr(row, 'headerLevel')
            
            result = {
                "chunk_id": self._decode_uri_part(str(chunk), CHUNKS),
                "note_id": self._decode_uri_part(str(note), NOTES),
                "chunk_type": str(chunk_type),
                "importance_score": float(importance),
                "header": str(header) if header else None,
                "header_level": int(header_level) if header_level else None
            }
            results.append(result)
        
        return results
    
    def get_chunk_neighbors(self, chunk_id: str, include_sequential: bool = True, 
                           include_hierarchical: bool = True) -> List[Dict]:
        """Get neighboring chunks (sequential and hierarchical)."""
        if settings.chunk_strategy != "semantic":
            return []
        
        chunk_uri = self._chunk_uri(chunk_id)
        
        relations = []
        if include_sequential:
            relations.extend(["vault:followedBy", "vault:precededBy"])
        if include_hierarchical:
            relations.extend(["vault:hasParentSection", "vault:hasChildSection"])
        
        if not relations:
            return []
        
        relations_filter = " | ".join(relations)
        
        query = f"""
        PREFIX vault: <{VAULT}>
        PREFIX chunks: <{CHUNKS}>
        
        SELECT ?neighbor ?chunkType ?importance ?header ?relationship
        WHERE {{
            <{chunk_uri}> ({relations_filter}) ?neighbor .
            ?neighbor vault:chunkType ?chunkType .
            ?neighbor vault:importanceScore ?importance .
            OPTIONAL {{ ?neighbor vault:hasHeader ?header }}
            
            # Determine relationship type
            OPTIONAL {{ <{chunk_uri}> vault:followedBy ?neighbor . BIND("followed_by" AS ?rel1) }}
            OPTIONAL {{ <{chunk_uri}> vault:precededBy ?neighbor . BIND("preceded_by" AS ?rel2) }}
            OPTIONAL {{ <{chunk_uri}> vault:hasParentSection ?neighbor . BIND("parent_section" AS ?rel3) }}
            OPTIONAL {{ <{chunk_uri}> vault:hasChildSection ?neighbor . BIND("child_section" AS ?rel4) }}
            
            BIND(COALESCE(?rel1, ?rel2, ?rel3, ?rel4, "unknown") AS ?relationship)
        }}
        ORDER BY ?importance DESC
        """
        
        results = []
        for row in self.graph.query(query):
            neighbor = get_query_attr(row, 'neighbor')
            chunk_type = get_query_attr(row, 'chunkType')
            importance = get_query_attr(row, 'importance')
            header = get_query_attr(row, 'header')
            relationship = get_query_attr(row, 'relationship')
            
            result = {
                "chunk_id": self._decode_uri_part(str(neighbor), CHUNKS),
                "chunk_type": str(chunk_type),
                "importance_score": float(importance),
                "header": str(header) if header else None,
                "relationship": str(relationship)
            }
            results.append(result)
        
        return results
    
    def get_chunks_by_header_hierarchy(self, note_id: str, header_level: int = 1) -> List[Dict]:
        """Get all chunks under headers of specified level in a note."""
        if settings.chunk_strategy != "semantic":
            return []
        
        note_uri = self._note_uri(note_id)
        
        query = f"""
        PREFIX vault: <{VAULT}>
        PREFIX chunks: <{CHUNKS}>
        
        SELECT ?chunk ?chunkType ?header ?position ?importance
        WHERE {{
            ?chunk vault:belongsToNote <{note_uri}> .
            ?chunk vault:chunkType ?chunkType .
            ?chunk vault:chunkPosition ?position .
            ?chunk vault:importanceScore ?importance .
            OPTIONAL {{ ?chunk vault:hasHeader ?header }}
            OPTIONAL {{ ?chunk vault:headerLevel ?level }}
            FILTER(!BOUND(?level) || ?level >= {header_level})
        }}
        ORDER BY ?position
        """
        
        results = []
        for row in self.graph.query(query):
            chunk = get_query_attr(row, 'chunk')
            chunk_type = get_query_attr(row, 'chunkType')
            header = get_query_attr(row, 'header')
            position = get_query_attr(row, 'position')
            importance = get_query_attr(row, 'importance')
            
            result = {
                "chunk_id": self._decode_uri_part(str(chunk), CHUNKS),
                "chunk_type": str(chunk_type),
                "header": str(header) if header else None,
                "position": int(position),
                "importance_score": float(importance)
            }
            results.append(result)
        
        return results
    
    def count_triples(self) -> int:
        """Get the total number of triples in the graph."""
        return len(self.graph)
    
    def get_stats(self) -> Dict:
        """Get statistics about the graph."""
        stats_query = """
        PREFIX vault: <http://obsidian-vault.local/ontology#>
        
        SELECT 
            (COUNT(DISTINCT ?note) AS ?noteCount)
            (COUNT(DISTINCT ?tag) AS ?tagCount)
            (COUNT(?linkRel) AS ?linkCount)
        WHERE {
            ?note a vault:Note .
            OPTIONAL { ?tag a vault:Tag }
            OPTIONAL { ?n1 vault:linksTo ?n2 . BIND(?n1 AS ?linkRel) }
        }
        """
        
        for row in self.graph.query(stats_query):
            note_count = get_query_attr(row, 'noteCount')
            tag_count = get_query_attr(row, 'tagCount')
            link_count = get_query_attr(row, 'linkCount')
            
            return {
                "notes": int(note_count) if note_count else 0,
                "tags": int(tag_count) if tag_count else 0,
                "links": int(link_count) if link_count else 0,
                "total_triples": self.count_triples()
            }
        
        return {"notes": 0, "tags": 0, "links": 0, "total_triples": 0}
