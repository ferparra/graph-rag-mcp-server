#!/usr/bin/env python3
"""
Mock objects and utilities for testing Graph RAG MCP Server.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
from unittest.mock import MagicMock


class MockChromaCollection:
    """Mock ChromaDB collection for testing"""
    
    def __init__(self):
        self.name = "test_collection"
        self._data = {
            'ids': [],
            'documents': [],
            'metadatas': [],
            'embeddings': []
        }
    
    def count(self) -> int:
        return len(self._data['ids'])
    
    def get(self, ids=None, where=None, limit=None, include=None):
        """Mock get operation"""
        if not include:
            include = ['ids', 'documents', 'metadatas']
        
        result = {}
        for field in include:
            if field in self._data:
                data = self._data[field]
                if limit:
                    data = data[:limit]
                result[field] = data
        
        return result
    
    def query(self, query_texts, n_results=10, where=None, include=None):
        """Mock query operation"""
        if not include:
            include = ['ids', 'documents', 'metadatas', 'distances']
        
        # Return mock results
        result = {}
        for field in include:
            if field == 'distances':
                result[field] = [[0.1, 0.2, 0.3][:n_results]]
            elif field in self._data:
                data = self._data[field][:n_results]
                result[field] = [data]  # Wrap in list for query format
            else:
                result[field] = [[]]
        
        return result
    
    def upsert(self, ids, documents, metadatas=None):
        """Mock upsert operation"""
        self._data['ids'].extend(ids)
        self._data['documents'].extend(documents)
        if metadatas:
            self._data['metadatas'].extend(metadatas)
    
    def delete(self, ids):
        """Mock delete operation"""
        for id_to_delete in ids:
            if id_to_delete in self._data['ids']:
                idx = self._data['ids'].index(id_to_delete)
                for field in self._data:
                    if idx < len(self._data[field]):
                        del self._data[field][idx]


class MockUnifiedStore:
    """Mock UnifiedStore for testing"""
    
    def __init__(self, client_dir: Path, collection_name: str, embed_model: str):
        self.client_dir = client_dir
        self.collection_name = collection_name
        self.embed_model = embed_model
        self._collection_mock = MockChromaCollection()
    
    def _collection(self):
        return self._collection_mock
    
    def query(self, query: str, k: int = 6, where: Optional[Dict] = None) -> List[Dict]:
        """Mock query with sample results"""
        return [
            {
                'id': 'earth_chunk_1',
                'text': 'Earth is the third planet from the Sun.',
                'meta': {
                    'title': 'Earth',
                    'path': 'planets/Earth.md',
                    'chunk_type': 'section',
                    'importance_score': 0.8
                },
                'distance': 0.1
            },
            {
                'id': 'mars_chunk_1', 
                'text': 'Mars is the fourth planet from the Sun.',
                'meta': {
                    'title': 'Mars',
                    'path': 'planets/Mars.md',
                    'chunk_type': 'section',
                    'importance_score': 0.7
                },
                'distance': 0.2
            }
        ][:k]
    
    def get_neighbors(self, note_id: str, depth: int = 1, relationship_types: Optional[List[str]] = None) -> List[Dict]:
        """Mock neighbor discovery"""
        if note_id == "Earth":
            return [
                {'id': 'Mars', 'title': 'Mars', 'path': 'planets/Mars.md', 'type': 'Note'},
                {'id': 'Link Map', 'title': 'Link Map', 'path': 'links/Link Map.md', 'type': 'Note'}
            ]
        elif note_id == "Mars":
            return [
                {'id': 'Earth', 'title': 'Earth', 'path': 'planets/Earth.md', 'type': 'Note'},
                {'id': 'Link Map', 'title': 'Link Map', 'path': 'links/Link Map.md', 'type': 'Note'}
            ]
        return []
    
    def get_backlinks(self, note_id: str) -> List[Dict]:
        """Mock backlink discovery"""
        if note_id in ["Earth", "Mars"]:
            return [
                {'id': 'Link Map', 'title': 'Link Map', 'path': 'links/Link Map.md'}
            ]
        return []
    
    def get_notes_by_tag(self, tag: str) -> List[Dict]:
        """Mock tag-based note discovery"""
        if tag == "topic/planets":
            return [
                {'id': 'Earth', 'title': 'Earth', 'path': 'planets/Earth.md'},
                {'id': 'Mars', 'title': 'Mars', 'path': 'planets/Mars.md'}
            ]
        return []
    
    def fuzzy_tag_search(self, entities: List[str], k: int = 6, where: Optional[Dict] = None) -> List[Dict]:
        """Mock fuzzy tag search"""
        mock_results = []
        for entity in entities:
            if entity.lower() in ['planets', 'planet']:
                mock_results.extend([
                    {
                        'id': 'earth_chunk_1',
                        'text': 'Earth is a planet with verification code.',
                        'meta': {'title': 'Earth', 'tags': 'topic/planets,test/suite'},
                        'tag_score': 0.9
                    },
                    {
                        'id': 'mars_chunk_1',
                        'text': 'Mars is a planet in our solar system.',
                        'meta': {'title': 'Mars', 'tags': 'topic/planets,test/suite'},
                        'tag_score': 0.9
                    }
                ])
        return mock_results[:k]
    
    def get_chunk_neighbors(self, chunk_id: str, include_sequential: bool = True, 
                           include_hierarchical: bool = True) -> List[Dict]:
        """Mock chunk neighbor discovery"""
        return [
            {
                'chunk_id': f'{chunk_id}_neighbor',
                'relationship': 'sequential_next',
                'chunk_type': 'paragraph',
                'importance_score': 0.6
            }
        ]


class MockVaultSearcher:
    """Mock VaultSearcher for testing"""
    
    def __init__(self, unified_store):
        self.unified_store = unified_store
    
    def search(self, query: str, k: int = 6, where: Optional[Dict] = None) -> List[Dict]:
        """Mock search operation"""
        return self.unified_store.query(query, k, where)
    
    def ask(self, question: str, where: Optional[Dict] = None) -> Dict:
        """Mock Q&A operation"""
        return {
            'question': question,
            'answer': f'Mock answer for: {question}',
            'context': 'Mock context from search results',
            'success': True
        }


def create_mock_mcp_server():
    """Create a mock MCP server with all required tools"""
    server = MagicMock()
    
    # Mock unified store and searcher
    unified_store = MockUnifiedStore(
        client_dir=Path("/tmp/test"),
        collection_name="test",
        embed_model="test-model"
    )
    searcher = MockVaultSearcher(unified_store)
    
    # Mock app state
    app_state = MagicMock()
    app_state.unified_store = unified_store
    app_state.searcher = searcher
    
    server.app_state = app_state
    
    # Mock tool functions
    def mock_search_notes(query, k=6, vault_filter=None, tag_filter=None):
        hits = unified_store.query(query, k)
        return type('SearchResult', (), {
            'hits': hits,
            'total_results': len(hits),
            'query': query
        })()
    
    def mock_smart_search(query, k=6, vault_filter=None):
        hits = unified_store.query(query, k)
        strategy = "vector"  # Default strategy for mock
        return type('SmartSearchResult', (), {
            'query': query,
            'strategy_used': strategy,
            'hits': hits,
            'total_results': len(hits),
            'explanation': f'Mock {strategy} search',
            'related_chunks': None
        })()
    
    def mock_graph_neighbors(note_id_or_title, depth=1, relationship_types=None):
        nodes = unified_store.get_neighbors(note_id_or_title, depth, relationship_types)
        return type('GraphResult', (), {
            'nodes': nodes,
            'edges': [],
            'stats': {'neighbor_count': len(nodes)}
        })()
    
    # Attach mock functions
    server.search_notes = mock_search_notes
    server.smart_search = mock_search_notes  # Fallback to basic search
    server.graph_neighbors = mock_graph_neighbors
    server.get_backlinks = lambda note_id_or_path: unified_store.get_backlinks(note_id_or_path)
    server.get_notes_by_tag = lambda tag: unified_store.get_notes_by_tag(tag)
    
    return server


def create_sample_test_content() -> Dict[str, str]:
    """Create sample markdown content for testing"""
    return {
        'Earth.md': '''---
title: Earth
tags: [topic/planets, test/suite]
---

# Earth

Earth is the third planet from the Sun and the only known planet to harbor life.

## Verification Code

The verification code for Earth is: CONST_TOKEN_ALPHA_earth_93e8a4

## Links

- [[Mars]]
- [[Link Map]]
''',
        'Mars.md': '''---
title: Mars
tags: [topic/planets, test/suite]
---

# Mars

Mars is the fourth planet from the Sun, often called the "Red Planet".

## Verification Code

The verification code for Mars is: CONST_TOKEN_ALPHA_mars_77c2d1

## Links

- [[Earth]]
- [[Link Map]]
''',
        'Link Map.md': '''---
title: Link Map
tags: [links, test/suite]
---

# Link Map

This note contains links to various planets in our test dataset.

## Planets

- [[Earth]] - The blue planet
- [[Mars]] - The red planet

Both planets have verification codes for testing.
'''
    }


class MockSettings:
    """Mock settings object for testing"""
    
    def __init__(self, vaults: List[Path] = None):
        self.vaults = vaults or [Path("/tmp/test_vault")]
        self.chroma_dir = Path("/tmp/test_chroma")
        self.collection = "test_collection"
        self.embedding_model = "all-MiniLM-L6-v2"
        self.max_chars = 1000
        self.overlap = 100
        self.chunk_strategy = "semantic"
        self.supported_extensions = [".md", ".txt"]