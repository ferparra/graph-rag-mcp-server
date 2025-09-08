"""Unit tests for base file parser and models."""

import pytest
import json
from src.base_parser import (
    BaseParser, BaseFile, BaseFilter, BaseSource, BaseView, BaseColumn,
    FilterOperator, ViewType, ColumnFormat, ComputedField
)


class TestBaseParser:
    """Test BaseParser functionality."""
    
    def test_parse_minimal_json(self):
        """Test parsing minimal valid JSON base file."""
        json_content = """
        {
            "$schema": "vault://schemas/obsidian/bases-2025-09.schema.json",
            "id": "test-base",
            "name": "Test Base",
            "version": 1,
            "description": "Test description",
            "source": {
                "folders": ["Notes"],
                "includeSubfolders": true,
                "filters": []
            },
            "views": [
                {
                    "id": "table-main",
                    "name": "Main Table",
                    "type": "table",
                    "columns": [
                        {"id": "title", "header": "Title", "source": "title"}
                    ]
                }
            ]
        }
        """
        
        base = BaseParser.parse_json(json_content)
        assert base.id == "test-base"
        assert base.name == "Test Base"
        assert base.version == 1
        assert len(base.views) == 1
        assert base.views[0].type == ViewType.TABLE
    
    def test_parse_yaml(self):
        """Test parsing YAML base file."""
        yaml_content = """
        $schema: vault://schemas/obsidian/bases-2025-09.schema.json
        id: test-base
        name: Test Base
        version: 1
        description: Test description
        source:
          folders: ["Notes"]
          includeSubfolders: true
          filters: []
        views:
          - id: table-main
            name: Main Table
            type: table
            columns:
              - id: title
                header: Title
                source: title
        """
        
        base = BaseParser.parse_yaml(yaml_content)
        assert base.id == "test-base"
        assert base.name == "Test Base"
    
    def test_validate_valid_content(self):
        """Test validation of valid base content."""
        json_content = """
        {
            "id": "valid-base",
            "name": "Valid Base",
            "version": 1,
            "source": {"folders": ["/"], "filters": []},
            "views": [
                {
                    "id": "view1",
                    "name": "View 1",
                    "type": "table",
                    "columns": [
                        {"id": "col1", "header": "Column 1", "source": "prop1"}
                    ]
                }
            ]
        }
        """
        
        is_valid, error = BaseParser.validate(json_content, "json")
        assert is_valid
        assert error is None
    
    def test_validate_invalid_content(self):
        """Test validation of invalid base content."""
        # Missing required fields
        json_content = """
        {
            "id": "invalid-base",
            "name": "Invalid Base"
        }
        """
        
        is_valid, error = BaseParser.validate(json_content, "json")
        assert not is_valid
        assert error is not None
        assert "version" in error or "source" in error or "views" in error
    
    def test_to_json(self):
        """Test converting BaseFile to JSON."""
        base = BaseFile(
            id="test-base",
            name="Test Base",
            version=1,
            source=BaseSource(folders=["Notes"], filters=[]),
            views=[
                BaseView(
                    id="view1",
                    name="View 1",
                    type=ViewType.TABLE,
                    columns=[
                        BaseColumn(id="col1", header="Column 1", source="title")
                    ]
                )
            ]
        )
        
        json_str = BaseParser.to_json(base)
        parsed = json.loads(json_str)
        
        assert parsed["id"] == "test-base"
        assert parsed["name"] == "Test Base"
        assert parsed["version"] == 1


class TestBaseModels:
    """Test base file models and validation."""
    
    def test_filter_validation(self):
        """Test filter model validation."""
        # Valid filter
        filter1 = BaseFilter(property="status", op=FilterOperator.EQ, value="active")
        assert filter1.property == "status"
        assert filter1.op == FilterOperator.EQ
        assert filter1.value == "active"
        
        # Filter with IN operator should convert single value to list
        filter2 = BaseFilter(property="tags", op=FilterOperator.IN, value="work")
        assert isinstance(filter2.value, list)
        
        # EXISTS operator shouldn't require value
        filter3 = BaseFilter(property="title", op=FilterOperator.EXISTS)
        assert filter3.value is None
    
    def test_column_validation(self):
        """Test column model validation."""
        # Valid column
        col = BaseColumn(
            id="test-col",
            header="Test Column",
            source="test_property"
        )
        assert col.id == "test-col"
        
        # Column with invalid ID pattern should fail
        with pytest.raises(ValueError):
            BaseColumn(
                id="Test Col",  # Invalid: contains space and uppercase
                header="Test",
                source="test"
            )
        
        # Progress column requires min/max
        with pytest.raises(ValueError):
            BaseColumn(
                id="progress",
                header="Progress",
                source="completion",
                format=ColumnFormat.PROGRESS
                # Missing min/max
            )
    
    def test_view_validation(self):
        """Test view model validation."""
        # Table view requires columns
        with pytest.raises(ValueError):
            BaseView(
                id="table1",
                name="Table",
                type=ViewType.TABLE
                # Missing columns
            )
        
        # Valid table view
        view = BaseView(
            id="table1",
            name="Table",
            type=ViewType.TABLE,
            columns=[
                BaseColumn(id="col1", header="Column 1", source="prop1")
            ]
        )
        assert view.type == ViewType.TABLE
        assert len(view.columns) == 1
    
    def test_computed_field_validation(self):
        """Test computed field validation."""
        # Valid computed field
        cf = ComputedField(
            id="health-score",
            expr="clamp(priority * 10, 0, 100)",
            type="number"
        )
        assert cf.id == "health-score"
        
        # Invalid expression (unbalanced parentheses)
        with pytest.raises(ValueError):
            ComputedField(
                id="bad-expr",
                expr="clamp(x, 0, 100",  # Missing closing paren
                type="number"
            )
    
    def test_base_file_unique_ids(self):
        """Test that BaseFile enforces unique IDs."""
        # Duplicate IDs should fail
        with pytest.raises(ValueError) as exc_info:
            BaseFile(
                id="test",
                name="Test",
                version=1,
                source=BaseSource(folders=["/"], filters=[]),
                views=[
                    BaseView(
                        id="view1",
                        name="View 1",
                        type=ViewType.TABLE,
                        columns=[
                            BaseColumn(id="col1", header="Col 1", source="p1"),
                            BaseColumn(id="col1", header="Col 2", source="p2")  # Duplicate ID
                        ]
                    )
                ]
            )
        assert "Duplicate IDs" in str(exc_info.value)
    
    def test_path_normalization(self):
        """Test that paths are normalized to forward slashes."""
        source = BaseSource(
            folders=["Notes\\Books", "Work\\Projects"],  # Windows-style paths
            includeSubfolders=True,
            filters=[]
        )
        
        # Paths should be normalized to forward slashes
        assert source.folders[0] == "Notes/Books"
        assert source.folders[1] == "Work/Projects"
    
    def test_path_traversal_prevention(self):
        """Test that path traversal attempts are prevented."""
        with pytest.raises(ValueError) as exc_info:
            BaseSource(
                folders=["../../../etc"],  # Path traversal attempt
                filters=[]
            )
        assert "Path traversal not allowed" in str(exc_info.value)