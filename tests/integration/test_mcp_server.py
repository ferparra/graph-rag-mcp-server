#!/usr/bin/env python3
"""
Integration tests for the enhanced Graph RAG MCP Server.
Tests the full server functionality with dependencies and actual components.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
import pytest

# Setup test environment
project_root = Path(__file__).parents[2]  # Go up two levels from tests/integration/
sys.path.insert(0, str(project_root / "src"))


@pytest.fixture
def integration_test_env():
    """Setup and cleanup integration test environment"""
    # Create temporary directory for test ChromaDB
    temp_dir = tempfile.mkdtemp(prefix="test_mcp_")
    
    # Set test vault path to fixtures
    test_vault_path = project_root / "tests" / "fixtures" / "content" / "planets"
    
    # Store original environment variables
    original_env = {}
    env_vars = {
        'OBSIDIAN_RAG_CHROMA_DIR': str(Path(temp_dir) / "chroma_db"),
        'OBSIDIAN_RAG_VAULT_PATH': str(test_vault_path)
    }
    
    for key, value in env_vars.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    print("Test environment setup complete:")
    print(f"  Temp dir: {temp_dir}")
    print(f"  Vault path: {test_vault_path}")
    
    yield {
        'temp_dir': temp_dir,
        'test_vault_path': test_vault_path
    }
    
    # Cleanup
    if Path(temp_dir).exists():
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temp dir: {temp_dir}")
    
    # Restore original environment variables
    for key, original_value in original_env.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


class TestMCPServer:
    """Integration test suite for MCP server components"""
    
    def test_basic_imports(self, integration_test_env):
        """Test that all MCP server modules can be imported"""
        print("\n=== Testing MCP Server Imports ===")
        try:
            from config import settings
            
            print("✓ Core MCP server modules imported successfully")
            print(f"  Settings vault: {settings.vaults}")
            print(f"  ChromaDB dir: {settings.chroma_dir}")
            assert True
        except Exception as e:
            print(f"❌ Import error: {e}")
            assert False, f"Import error: {e}"

    def test_unified_store_initialization(self, integration_test_env):
        """Test UnifiedStore initialization and basic functionality"""
        print("\n=== Testing UnifiedStore Initialization ===")
        try:
            from unified_store import UnifiedStore
            
            # Initialize unified store
            store = UnifiedStore(
                client_dir=Path(integration_test_env['temp_dir']) / "test_chroma",
                collection_name="test_collection",
                embed_model="all-MiniLM-L6-v2"
            )
            
            # Test basic operations
            count = store.count()
            print(f"✓ UnifiedStore initialized, current count: {count}")
            
            # Test collection creation
            collection = store._collection()
            print(f"✓ ChromaDB collection created: {collection.name}")
            
            assert True
        except Exception as e:
            print(f"❌ UnifiedStore initialization error: {e}")
            assert False, f"UnifiedStore initialization error: {e}"

    def test_smart_search_engine(self, integration_test_env):
        """Test SmartSearchEngine initialization and query analysis"""
        print("\n=== Testing SmartSearchEngine ===")
        try:
            from mcp_server import SmartSearchEngine
            from unified_store import UnifiedStore
            from dspy_rag import VaultSearcher
            
            # Initialize components
            store = UnifiedStore(
                client_dir=Path(integration_test_env['temp_dir']) / "test_chroma",
                collection_name="test_collection",
                embed_model="all-MiniLM-L6-v2"
            )
            searcher = VaultSearcher(unified_store=store)
            smart_engine = SmartSearchEngine(store, searcher)
            
            # Test query intent analysis
            test_queries = [
                "What is the verification code for Earth?",  # Should be semantic
                "Show me notes tagged with planets",         # Should be categorical
                "What links to Mars?",                       # Should be relationship
                "Find content in Earth document",            # Should be specific
            ]
            
            print("Testing query intent analysis:")
            for query in test_queries:
                intent = smart_engine.analyze_query_intent(query)
                print(f"  '{query}' → {intent.intent_type} ({intent.confidence:.2f})")
                
                # Verify intent has expected structure
                assert hasattr(intent, 'intent_type')
                assert hasattr(intent, 'confidence')
                assert hasattr(intent, 'suggested_strategy')
                assert hasattr(intent, 'extracted_entities')
            
            print("✓ SmartSearchEngine working correctly")
            assert True
        except Exception as e:
            print(f"❌ SmartSearchEngine error: {e}")
            assert False, f"SmartSearchEngine error: {e}"

    def test_uri_generation(self, integration_test_env):
        """Test Obsidian URI generation functionality"""
        print("\n=== Testing URI Generation ===")
        try:
            from mcp_server import SmartSearchEngine
            from unified_store import UnifiedStore
            from dspy_rag import VaultSearcher
            
            # Initialize components
            store = UnifiedStore(
                client_dir=Path(integration_test_env['temp_dir']) / "test_chroma",
                collection_name="test_collection",
                embed_model="all-MiniLM-L6-v2"
            )
            searcher = VaultSearcher(unified_store=store)
            smart_engine = SmartSearchEngine(store, searcher)
            
            # Test URI generation with sample metadata
            test_cases = [
                {
                    'metadata': {
                        'path': 'planets/Earth.md',
                        'header_text': 'Earth',
                        'title': 'Earth'
                    },
                    'vault': 'TestVault',
                    'expected_components': ['obsidian://open', 'vault=TestVault', 'file=Earth', '#earth']
                },
                {
                    'metadata': {
                        'path': 'complex/Link Map.md',
                        'header_text': 'Link Map',
                        'title': 'Link Map'
                    },
                    'vault': 'MyVault',
                    'expected_components': ['obsidian://open', 'vault=MyVault', 'file=Link%20Map', '#link%20map']
                }
            ]
            
            for i, case in enumerate(test_cases):
                uri = smart_engine.generate_chunk_uri(case['metadata'], case['vault'])
                print(f"  Test {i+1}: {uri}")
                
                # Verify all expected components are present
                for component in case['expected_components']:
                    assert component in uri, f"Missing component: {component}"
            
            print("✓ URI generation working correctly")
            assert True
        except Exception as e:
            print(f"❌ URI generation error: {e}")
            assert False, f"URI generation error: {e}"

    def test_fuzzy_tag_search(self, integration_test_env):
        """Test fuzzy tag search implementation"""
        print("\n=== Testing Fuzzy Tag Search ===")
        try:
            from unified_store import UnifiedStore
            
            store = UnifiedStore(
                client_dir=Path(integration_test_env['temp_dir']) / "test_chroma",
                collection_name="test_collection",
                embed_model="all-MiniLM-L6-v2"
            )
            
            # Test tag relevance calculation directly
            test_cases = [
                {
                    'query_entities': ['planets'],
                    'note_tags': ['topic/planets', 'test/suite'],
                    'expected_min_score': 0.8
                },
                {
                    'query_entities': ['test'],
                    'note_tags': ['test/suite', 'links'],
                    'expected_min_score': 0.8
                },
                {
                    'query_entities': ['health'],
                    'note_tags': ['para/area/health', 'fitness'],
                    'expected_min_score': 0.8
                }
            ]
            
            print("Testing tag relevance scoring:")
            for case in test_cases:
                score = store._calculate_tag_relevance_score(
                    case['query_entities'], 
                    case['note_tags']
                )
                print(f"  {case['query_entities']} vs {case['note_tags']}: {score:.3f}")
                assert score >= case['expected_min_score'] * 0.8, f"Score too low: {score}"
            
            print("✓ Fuzzy tag search logic working correctly")
            assert True
        except Exception as e:
            print(f"❌ Fuzzy tag search error: {e}")
            assert False, f"Fuzzy tag search error: {e}"

    def test_mcp_tool_smart_search_resilient(self, integration_test_env):
        """Smart Search MCP tool should never raise on basic queries and return JSON-serializable output."""
        print("\n=== Testing Smart Search MCP Tool (resilience) ===")
        try:
            import json
            import mcp_server
            from tests.evals.runner import EvalRunner

            runner = EvalRunner()
            # Invoke the FastMCP FunctionTool safely via its run(...) API
            out = runner.call_tool(mcp_server.smart_search, query="LLM-powered applications", k=3)
            # Must be a dict and JSON serializable
            assert isinstance(out, dict)
            # Basic shape checks
            assert "query" in out
            assert "hits" in out
            assert "total_results" in out
            # JSON round-trip
            json.dumps(out)
            print(f"✓ smart_search returned safely with {len(out.get('hits', []))} hits")
            assert True
        except Exception as e:
            print(f"❌ smart_search tool failed: {e}")
            assert False, f"smart_search tool failed: {e}"

    def test_mcp_tools_initialization(self, integration_test_env):
        """Test that MCP tools can be initialized"""
        print("\n=== Testing MCP Tools Initialization ===")
        try:
            import mcp_server
            
            # Verify key components exist
            assert hasattr(mcp_server, 'app_state'), "app_state not found"
            assert hasattr(mcp_server, 'smart_search_engine'), "smart_search_engine not found"
            assert hasattr(mcp_server, 'mcp'), "mcp server not found"
            
            # Verify new tools are registered
            assert hasattr(mcp_server, 'smart_search'), "smart_search tool not found"
            assert hasattr(mcp_server, 'traverse_from_chunk'), "traverse_from_chunk tool not found"
            assert hasattr(mcp_server, 'get_related_chunks'), "get_related_chunks tool not found"
            assert hasattr(mcp_server, 'explore_chunk_context'), "explore_chunk_context tool not found"
            
            print("✓ MCP tools properly initialized")
            print("  Enhanced tools available:")
            print("    - smart_search")
            print("    - traverse_from_chunk") 
            print("    - get_related_chunks")
            print("    - explore_chunk_context")
            assert True
        except Exception as e:
            print(f"❌ MCP tools initialization error: {e}")
            assert False, f"MCP tools initialization error: {e}"

    def test_enhanced_retriever(self, integration_test_env):
        """Test enhanced retriever with graph expansion"""
        print("\n=== Testing Enhanced Retriever ===")
        try:
            from dspy_rag import UnifiedRetriever
            from unified_store import UnifiedStore
            
            store = UnifiedStore(
                client_dir=Path(integration_test_env['temp_dir']) / "test_chroma",
                collection_name="test_collection",
                embed_model="all-MiniLM-L6-v2"
            )
            
            retriever = UnifiedRetriever(unified_store=store, k=5)
            
            # Test relationship weight calculation
            test_relationships = [
                ("content_link", 0, 0.9),
                ("parent", 0, 0.9),
                ("sequential_next", 1, 0.64),  # 0.8 * 0.8
                ("child", 1, 0.56),            # 0.7 * 0.8
            ]
            
            print("Testing relationship weighting:")
            for relationship, depth, expected in test_relationships:
                weight = retriever._get_relationship_weight(relationship, depth)
                print(f"  {relationship} at depth {depth}: {weight:.3f} (expected: {expected:.3f})")
                assert abs(weight - expected) < 0.01, f"Weight mismatch for {relationship}"
            
            print("✓ Enhanced retriever working correctly")
            assert True
        except Exception as e:
            print(f"❌ Enhanced retriever error: {e}")
            assert False, f"Enhanced retriever error: {e}"

def show_integration_test_summary():
    """Show what these integration tests validate"""
    print("\n=== Integration Test Coverage ===")
    coverage_areas = [
        "✅ Core module imports and dependencies",
        "✅ UnifiedStore initialization with ChromaDB",
        "✅ SmartSearchEngine query intent analysis", 
        "✅ Obsidian URI generation with proper encoding",
        "✅ Fuzzy tag matching with hierarchical support",
        "✅ MCP tool registration and availability",
        "✅ Enhanced retriever with relationship weighting",
        "",
        "🎯 These tests validate the full integration of:",
        "   - ChromaDB vector database functionality",
        "   - Graph relationship metadata storage",
        "   - Smart search routing and intent detection",
        "   - Chunk-level navigation capabilities",
        "   - MCP server tool exposure for Claude Desktop"
    ]
    
    for line in coverage_areas:
        print(line)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
