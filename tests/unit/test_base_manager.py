"""Unit tests for base manager functionality."""

import pytest
from unittest.mock import Mock
from pathlib import Path
import tempfile
import json

from src.base_manager import BaseManager, BaseQueryResult
from src.base_parser import BaseFile, BaseSource, BaseView, BaseColumn, ViewType, SortConfig, SortDirection


class TestBaseManager:
    """Test BaseManager functionality."""
    
    @pytest.fixture
    def mock_unified_store(self):
        """Create a mock unified store."""
        store = Mock()
        store._collection = Mock(return_value=Mock())
        return store
    
    @pytest.fixture
    def base_manager(self, mock_unified_store):
        """Create a BaseManager instance with mock store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = BaseManager(
                unified_store=mock_unified_store,
                vault_path=Path(tmpdir)
            )
            yield manager
    
    def test_discover_bases(self, base_manager):
        """Test discovering .base files in vault."""
        # Create a test base file
        base_content = {
            "id": "test-base",
            "name": "Test Base",
            "version": 1,
            "source": {"folders": ["Notes"], "filters": []},
            "views": [{
                "id": "view1",
                "name": "View 1",
                "type": "table",
                "columns": [{"id": "col1", "header": "Column 1", "source": "title"}]
            }]
        }
        
        base_path = base_manager.vault_path / "test.base"
        base_path.write_text(json.dumps(base_content))
        
        # Discover bases
        bases = base_manager.discover_bases()
        
        assert len(bases) == 1
        assert bases[0].id == "test-base"
        assert bases[0].name == "Test Base"
        assert bases[0].view_count == 1
    
    def test_get_base(self, base_manager):
        """Test getting a base by ID."""
        # Create a test base file
        base_content = {
            "id": "test-base",
            "name": "Test Base",
            "version": 1,
            "source": {"folders": ["Notes"], "filters": []},
            "views": [{
                "id": "view1",
                "name": "View 1",
                "type": "table",
                "columns": [{"id": "col1", "header": "Column 1", "source": "title"}]
            }]
        }
        
        base_path = base_manager.vault_path / "test.base"
        base_path.write_text(json.dumps(base_content))
        
        # Get base by ID
        base = base_manager.get_base("test-base")
        
        assert base is not None
        assert base.id == "test-base"
        assert base.name == "Test Base"
    
    def test_convert_filter_to_chroma(self, base_manager):
        """Test converting base filters to ChromaDB format."""
        from src.base_parser import BaseFilter, FilterOperator
        
        # EQ filter
        filter1 = BaseFilter(property="status", op=FilterOperator.EQ, value="active")
        result1 = base_manager._convert_filter_to_chroma(filter1)
        assert result1 == {"status": {"$eq": "active"}}
        
        # IN filter
        filter2 = BaseFilter(property="tags", op=FilterOperator.IN, value=["work", "urgent"])
        result2 = base_manager._convert_filter_to_chroma(filter2)
        assert result2 == {"tags": {"$in": ["work", "urgent"]}}
        
        # EXISTS filter
        filter3 = BaseFilter(property="title", op=FilterOperator.EXISTS)
        result3 = base_manager._convert_filter_to_chroma(filter3)
        assert result3 == {"title": {"$ne": None}}
        
        # CONTAINS filter (requires post-processing)
        filter4 = BaseFilter(property="content", op=FilterOperator.CONTAINS, value="test")
        result4 = base_manager._convert_filter_to_chroma(filter4)
        assert result4 is None  # Should return None for post-processing
    
    def test_sort_results(self, base_manager):
        """Test sorting results."""
        results = [
            {"title": "C Note", "priority": 1, "date": "2024-01-03"},
            {"title": "A Note", "priority": 3, "date": "2024-01-01"},
            {"title": "B Note", "priority": 2, "date": "2024-01-02"},
        ]
        
        # Sort by title ascending
        sort_configs = [SortConfig(by="title", dir=SortDirection.ASC)]
        sorted_results = base_manager._sort_results(results, sort_configs)
        
        assert sorted_results[0]["title"] == "A Note"
        assert sorted_results[1]["title"] == "B Note"
        assert sorted_results[2]["title"] == "C Note"
        
        # Sort by priority descending
        sort_configs = [SortConfig(by="priority", dir=SortDirection.DESC)]
        sorted_results = base_manager._sort_results(results, sort_configs)
        
        assert sorted_results[0]["priority"] == 3
        assert sorted_results[1]["priority"] == 2
        assert sorted_results[2]["priority"] == 1
    
    def test_group_results(self, base_manager):
        """Test grouping results."""
        from src.base_parser import GroupConfig
        
        results = [
            {"title": "Note 1", "status": "active"},
            {"title": "Note 2", "status": "done"},
            {"title": "Note 3", "status": "active"},
            {"title": "Note 4", "status": "blocked"},
        ]
        
        group_config = GroupConfig(by="status")
        groups = base_manager._group_results(results, group_config)
        
        assert len(groups) == 3
        assert len(groups["active"]) == 2
        assert len(groups["done"]) == 1
        assert len(groups["blocked"]) == 1
        
        # Test with custom order
        group_config = GroupConfig(by="status", order=["blocked", "active", "done"])
        groups = base_manager._group_results(results, group_config)
        
        # Check order is preserved
        keys = list(groups.keys())
        assert keys[0] == "blocked"
        assert keys[1] == "active"
        assert keys[2] == "done"
    
    def test_format_table_view(self, base_manager):
        """Test formatting results as table view."""
        # Create a mock query result
        query_result = BaseQueryResult(
            base_id="test-base",
            view_id="table-view",
            total_count=2,
            filtered_count=2,
            results=[
                {"title": "Note 1", "tags": "work", "priority": 1},
                {"title": "Note 2", "tags": "personal", "priority": 2}
            ]
        )
        
        # Create a base with table view
        base = BaseFile(
            id="test-base",
            name="Test Base",
            version=1,
            source=BaseSource(folders=["/"], filters=[]),
            views=[
                BaseView(
                    id="table-view",
                    name="Table View",
                    type=ViewType.TABLE,
                    columns=[
                        BaseColumn(id="title", header="Title", source="title"),
                        BaseColumn(id="tags", header="Tags", source="tags"),
                        BaseColumn(id="priority", header="Priority", source="priority")
                    ]
                )
            ]
        )
        
        # Mock get_base to return our test base
        base_manager.get_base = Mock(return_value=base)
        
        # Format as table view
        view_data = base_manager.format_view_data(query_result, "test-base", "table-view")
        
        assert view_data.view_type == "table"
        assert len(view_data.columns) == 3
        assert len(view_data.rows) == 2
        assert view_data.rows[0]["title"] == "Note 1"
        assert view_data.rows[1]["tags"] == "personal"
    
    def test_format_card_view(self, base_manager):
        """Test formatting results as card view."""
        from src.base_parser import CardConfig
        
        # Create a mock query result
        query_result = BaseQueryResult(
            base_id="test-base",
            view_id="card-view",
            total_count=1,
            filtered_count=1,
            results=[
                {
                    "title": "Project Alpha",
                    "owner": "John Doe",
                    "status": "active",
                    "priority": "high",
                    "nextAction": "Review proposal"
                }
            ]
        )
        
        # Create a base with card view
        base = BaseFile(
            id="test-base",
            name="Test Base",
            version=1,
            source=BaseSource(folders=["/"], filters=[]),
            views=[
                BaseView(
                    id="card-view",
                    name="Card View",
                    type=ViewType.CARD,
                    card=CardConfig(
                        title="title",
                        subtitle="owner",
                        badges=["status", "priority"],
                        footer="nextAction"
                    )
                )
            ]
        )
        
        # Mock get_base to return our test base
        base_manager.get_base = Mock(return_value=base)
        
        # Format as card view
        view_data = base_manager.format_view_data(query_result, "test-base", "card-view")
        
        assert view_data.view_type == "card"
        assert len(view_data.cards) == 1
        assert view_data.cards[0]["title"] == "Project Alpha"
        assert view_data.cards[0]["subtitle"] == "John Doe"
        assert view_data.cards[0]["badges"] == ["active", "high"]
        assert view_data.cards[0]["footer"] == "Review proposal"