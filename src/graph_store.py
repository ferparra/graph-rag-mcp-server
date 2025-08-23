from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Dict, Optional, Any
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, XSD
from rdflib.query import ResultRow
from rdflib_sqlalchemy.store import SQLAlchemy
from .fs_indexer import discover_files, parse_note, NoteDoc, resolve_links
from .semantic_chunker import SemanticChunker, SemanticChunk
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
        
        # Create SQLite-backed RDF store
        self.store = SQLAlchemy(identifier=store_identifier)
        
        # Create the graph with the store
        self.graph = Graph(store=self.store)
        
        # Open the database connection
        db_url = f"sqlite:///{db_path}"
        self.graph.open(db_url, create=True)
        
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
    
    def _note_uri(self, note_id: str) -> URIRef:
        """Generate URI for a note."""
        return NOTES[note_id.replace("/", "_").replace(" ", "_")]
    
    def _tag_uri(self, tag: str) -> URIRef:
        """Generate URI for a tag."""
        return TAGS[tag.replace("/", "_").replace(" ", "_")]
    
    def _chunk_uri(self, chunk_id: str) -> URIRef:
        """Generate URI for a chunk."""
        return CHUNKS[chunk_id.replace("/", "_").replace(" ", "_").replace("#", "_")]
    
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
            
            result = {
                "id": str(connected).replace(str(NOTES), "").replace("_", "/"),
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
                "id": str(source).replace(str(NOTES), "").replace("_", "/"),
                "title": str(title),
                "path": str(path)
            }
            results.append(result)
        
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
                "id": str(note).replace(str(NOTES), "").replace("_", "/"),
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
            node = {
                # pyrefly: ignore  # missing-attribute
                "id": str(row.note).replace(str(NOTES), "").replace("_", "/"),
                # pyrefly: ignore  # missing-attribute
                "title": str(row.title),
                # pyrefly: ignore  # missing-attribute
                "path": str(row.path),
                # pyrefly: ignore  # missing-attribute
                "vault": str(row.vault) if row.vault else "",
                # pyrefly: ignore  # no-matching-overload, missing-attribute
                "text_length": int(row.textLength) if row.textLength else 0
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
            edge = {
                # pyrefly: ignore  # missing-attribute
                "source": str(row.source).replace(str(NOTES), "").replace("_", "/"),
                # pyrefly: ignore  # missing-attribute
                "target": str(row.target).replace(str(NOTES), "").replace(str(TAGS), "").replace("_", "/"),
                # pyrefly: ignore  # missing-attribute
                "relationship": str(row.relationship)
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
                "chunk_id": str(chunk).replace(str(CHUNKS), "").replace("_", "/"),
                "note_id": str(note).replace(str(NOTES), "").replace("_", "/"),
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
                "chunk_id": str(neighbor).replace(str(CHUNKS), "").replace("_", "/"),
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
                "chunk_id": str(chunk).replace(str(CHUNKS), "").replace("_", "/"),
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
