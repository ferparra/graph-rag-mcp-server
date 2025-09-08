#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures for Graph RAG MCP Server tests.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
import pytest
from typing import Generator, Dict, Any, Optional

# Add src to path for all tests
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

# Import test utilities
from tests.fixtures.factories import TestContentFactory, TestEnvironmentManager  # noqa: E402  # type: ignore[reportMissingImports]
from tests.fixtures.mocks import MockUnifiedStore, MockVaultSearcher, MockSettings  # noqa: E402  # type: ignore[reportMissingImports]


@pytest.fixture(scope="session")
def project_root_path() -> Path:
    """Get the project root path"""
    return project_root


@pytest.fixture(scope="session")
def test_content_dir() -> Path:
    """Get the test content directory"""
    return project_root / "tests" / "fixtures" / "content"


@pytest.fixture
def temp_vault() -> Generator[Path, None, None]:
    """Create a temporary vault with planet test content"""
    with TestEnvironmentManager() as env:
        vault_dir = env.create_test_environment(TestContentFactory.create_planet_notes())
        yield vault_dir


@pytest.fixture
def extended_vault() -> Generator[Path, None, None]:
    """Create a temporary vault with full test content (planets + health + projects)"""
    with TestEnvironmentManager() as env:
        all_notes = (
            TestContentFactory.create_planet_notes() +
            TestContentFactory.create_health_fitness_notes() +
            TestContentFactory.create_project_notes()
        )
        vault_dir = env.create_test_environment(all_notes)
        yield vault_dir


@pytest.fixture
def planets_only_vault() -> Generator[Path, None, None]:
    """Create a temporary vault with only planet notes"""
    vault_dir = TestContentFactory.create_planets_only_vault()
    yield vault_dir
    # Cleanup
    if vault_dir.exists():
        shutil.rmtree(vault_dir)


@pytest.fixture
def mock_settings(temp_vault: Path) -> MockSettings:
    """Create mock settings for testing"""
    return MockSettings(vaults=[temp_vault])


@pytest.fixture
def mock_unified_store() -> MockUnifiedStore:
    """Create a mock UnifiedStore for testing"""
    temp_dir = Path(tempfile.mkdtemp(prefix="test_chroma_"))
    store = MockUnifiedStore(
        client_dir=temp_dir,
        collection_name="test_collection",
        embed_model="all-MiniLM-L6-v2"
    )
    yield store
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def mock_vault_searcher(mock_unified_store: MockUnifiedStore) -> MockVaultSearcher:
    """Create a mock VaultSearcher for testing"""
    return MockVaultSearcher(mock_unified_store)


@pytest.fixture
def test_environment_vars(temp_vault: Path):
    """Set up test environment variables"""
    original_env = {}
    
    # Store original values
    test_vars = {
        'OBSIDIAN_RAG_VAULT_PATH': str(temp_vault),
        'OBSIDIAN_RAG_CHROMA_DIR': str(temp_vault / '.test_chroma'),
        'OBSIDIAN_RAG_COLLECTION': 'test_collection',
    }
    
    for key, value in test_vars.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    yield test_vars
    
    # Restore original values
    for key, original_value in original_env.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


@pytest.fixture
def sample_search_queries() -> Dict[str, Any]:
    """Sample search queries for testing"""
    return {
        'semantic': [
            "What is the verification code for Earth?",
            "Tell me about planetary characteristics",
            "Explain Mars features"
        ],
        'categorical': [
            "Show me notes tagged with planets",
            "Find content tagged test/suite",
            "#planets related content"
        ],
        'relationship': [
            "What links to Mars?",
            "Show connections to Earth",
            "Graph neighbors of planets"
        ],
        'specific': [
            "Find content in Earth document",
            "Search Mars file specifically",
            "Get exact content from Link Map"
        ]
    }


@pytest.fixture
def sample_test_results() -> Dict[str, Any]:
    """Sample test results for metrics testing"""
    return {
        'search_results': [
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
        ],
        'graph_results': {
            'nodes': [
                {'id': 'Earth', 'title': 'Earth', 'path': 'planets/Earth.md'},
                {'id': 'Mars', 'title': 'Mars', 'path': 'planets/Mars.md'},
                {'id': 'Link Map', 'title': 'Link Map', 'path': 'links/Link Map.md'}
            ],
            'edges': [
                {'source': 'Earth', 'target': 'Mars', 'relationship': 'links_to'},
                {'source': 'Mars', 'target': 'Earth', 'relationship': 'links_to'}
            ]
        }
    }


@pytest.fixture(autouse=True)
def cleanup_temp_dirs():
    """Automatically cleanup temporary directories after each test"""
    # This runs after each test to ensure cleanup
    yield
    # Cleanup any remaining temp directories
    temp_dirs = [p for p in Path.cwd().iterdir() if p.is_dir() and p.name.startswith(('test_', 'eval_', 'temp_'))]
    for temp_dir in temp_dirs:
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass  # Ignore cleanup errors


# Test markers for categorizing tests
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line("markers", "unit: Unit tests (no external dependencies)")
    config.addinivalue_line("markers", "integration: Integration tests (with dependencies)")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "eval: Evaluation tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "requires_chroma: Tests that require ChromaDB")
    config.addinivalue_line("markers", "requires_gemini: Tests that require Gemini API")


# Custom test collection rules
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location"""
    for item in items:
        # Add markers based on test file location
        test_path = str(item.fspath)
        
        if "/unit/" in test_path:
            item.add_marker(pytest.mark.unit)
        elif "/integration/" in test_path:
            item.add_marker(pytest.mark.integration)
        elif "/e2e/" in test_path:
            item.add_marker(pytest.mark.e2e)
        elif "/evals/" in test_path:
            item.add_marker(pytest.mark.eval)
        elif "/benchmarks/" in test_path:
            item.add_marker(pytest.mark.slow)


# Custom assertion helpers
class TestAssertions:
    """Custom assertion helpers for MCP server testing"""
    
    @staticmethod
    def assert_search_result_valid(result, min_results: int = 1):
        """Assert that a search result is valid"""
        assert hasattr(result, 'hits'), "Search result missing 'hits' attribute"
        assert hasattr(result, 'total_results'), "Search result missing 'total_results' attribute"
        assert hasattr(result, 'query'), "Search result missing 'query' attribute"
        assert len(result.hits) >= min_results, f"Expected at least {min_results} results, got {len(result.hits)}"
        
        # Validate hit structure
        for hit in result.hits:
            assert 'id' in hit, "Hit missing 'id' field"
            assert 'text' in hit, "Hit missing 'text' field"
            assert 'meta' in hit, "Hit missing 'meta' field"
    
    @staticmethod
    def assert_graph_result_valid(result, min_nodes: int = 1):
        """Assert that a graph result is valid"""
        assert hasattr(result, 'nodes'), "Graph result missing 'nodes' attribute"
        assert hasattr(result, 'edges'), "Graph result missing 'edges' attribute"
        assert len(result.nodes) >= min_nodes, f"Expected at least {min_nodes} nodes, got {len(result.nodes)}"
        
        # Validate node structure
        for node in result.nodes:
            assert 'id' in node, "Node missing 'id' field"
            assert 'title' in node or 'path' in node, "Node missing 'title' or 'path' field"
    
    @staticmethod
    def assert_qa_result_valid(result, expect_success: bool = True):
        """Assert that a Q&A result is valid"""
        assert 'question' in result, "Q&A result missing 'question' field"
        assert 'answer' in result, "Q&A result missing 'answer' field"
        assert 'success' in result, "Q&A result missing 'success' field"
        
        if expect_success:
            assert result['success'], "Expected successful Q&A result"
            assert result['answer'].strip(), "Expected non-empty answer"
    
    @staticmethod
    def assert_uri_valid(uri: str, vault_name: Optional[str] = None):
        """Assert that an Obsidian URI is valid"""
        assert uri.startswith("obsidian://open"), "URI should start with obsidian://open"
        assert "vault=" in uri, "URI should contain vault parameter"
        assert "file=" in uri, "URI should contain file parameter"
        
        if vault_name:
            assert f"vault={vault_name}" in uri, f"URI should contain vault={vault_name}"


@pytest.fixture
def test_assertions() -> TestAssertions:
    """Provide test assertion helpers"""
    return TestAssertions()