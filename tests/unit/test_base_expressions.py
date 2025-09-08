"""Unit tests for base expression evaluator."""

import pytest
from datetime import datetime
from src.base_expressions import ExpressionEvaluator, ExpressionContext, evaluate_computed_fields, CircularReferenceError
from src.base_parser import BaseFile, BaseSource, BaseView, ViewType, ComputedField, BaseColumn


class TestExpressionEvaluator:
    """Test expression evaluator functionality."""
    
    @pytest.fixture
    def evaluator(self):
        """Create an expression evaluator instance."""
        return ExpressionEvaluator()
    
    @pytest.fixture
    def context(self):
        """Create a test context with sample note data."""
        return ExpressionContext(
            note_data={
                "title": "Test Note",
                "status": "active",
                "priority": 5,
                "tags": ["work", "urgent"],
                "created": "2024-01-01",
                "file": {
                    "name": "test.md",
                    "mtime": datetime(2024, 1, 15),
                    "size": 1024
                }
            },
            computed_values={}
        )
    
    def test_evaluate_literals(self, evaluator, context):
        """Test evaluation of literal values."""
        assert evaluator.evaluate("42", context) == 42
        assert evaluator.evaluate("3.14", context) == 3.14
        assert evaluator.evaluate('"hello"', context) == "hello"
        assert evaluator.evaluate("true", context) is True
        assert evaluator.evaluate("false", context) is False
        assert evaluator.evaluate("null", context) is None
    
    def test_evaluate_property_access(self, evaluator, context):
        """Test property access."""
        assert evaluator.evaluate("title", context) == "Test Note"
        assert evaluator.evaluate("status", context) == "active"
        assert evaluator.evaluate("priority", context) == 5
        
        # Dotted notation
        assert evaluator.evaluate("file.name", context) == "test.md"
        assert evaluator.evaluate("file.size", context) == 1024
    
    def test_evaluate_arithmetic(self, evaluator, context):
        """Test arithmetic operations."""
        assert evaluator.evaluate("2 + 3", context) == 5
        assert evaluator.evaluate("10 - 4", context) == 6
        assert evaluator.evaluate("3 * 4", context) == 12
        assert evaluator.evaluate("15 / 3", context) == 5.0
        assert evaluator.evaluate("10 % 3", context) == 1
        
        # With property values
        assert evaluator.evaluate("priority * 2", context) == 10
        assert evaluator.evaluate("priority + 10", context) == 15
    
    def test_evaluate_comparison(self, evaluator, context):
        """Test comparison operations."""
        assert evaluator.evaluate("5 > 3", context) is True
        assert evaluator.evaluate("2 < 1", context) is False
        assert evaluator.evaluate("5 == 5", context) is True
        assert evaluator.evaluate("4 != 4", context) is False
        assert evaluator.evaluate("3 >= 3", context) is True
        assert evaluator.evaluate("4 <= 3", context) is False
        
        # With properties
        assert evaluator.evaluate('status == "active"', context) is True
        assert evaluator.evaluate("priority > 3", context) is True
    
    def test_evaluate_logical(self, evaluator, context):
        """Test logical operations."""
        assert evaluator.evaluate("true && true", context) is True
        assert evaluator.evaluate("true && false", context) is False
        assert evaluator.evaluate("false || true", context) is True
        assert evaluator.evaluate("false || false", context) is False
        assert evaluator.evaluate("!true", context) is False
        assert evaluator.evaluate("!false", context) is True
        
        # With expressions
        assert evaluator.evaluate("priority > 3 && status == 'active'", context) is True
    
    def test_evaluate_ternary(self, evaluator, context):
        """Test ternary operator."""
        assert evaluator.evaluate("true ? 1 : 2", context) == 1
        assert evaluator.evaluate("false ? 1 : 2", context) == 2
        
        # With conditions
        assert evaluator.evaluate('status == "active" ? "Active" : "Inactive"', context) == "Active"
        assert evaluator.evaluate('priority > 10 ? "High" : "Normal"', context) == "Normal"
    
    def test_function_coalesce(self, evaluator, context):
        """Test coalesce function."""
        context.note_data["nullable"] = None
        
        assert evaluator.evaluate("coalesce(nullable, 'default')", context) == "default"
        assert evaluator.evaluate("coalesce(title, 'fallback')", context) == "Test Note"
        assert evaluator.evaluate("coalesce(null, null, 'third')", context) == "third"
    
    def test_function_clamp(self, evaluator, context):
        """Test clamp function."""
        assert evaluator.evaluate("clamp(5, 0, 10)", context) == 5
        assert evaluator.evaluate("clamp(-5, 0, 10)", context) == 0
        assert evaluator.evaluate("clamp(15, 0, 10)", context) == 10
        
        # With property
        assert evaluator.evaluate("clamp(priority, 1, 10)", context) == 5
    
    def test_function_string_operations(self, evaluator, context):
        """Test string operation functions."""
        assert evaluator.evaluate('lower("HELLO")', context) == "hello"
        assert evaluator.evaluate('upper("hello")', context) == "HELLO"
        assert evaluator.evaluate('trim("  hello  ")', context) == "hello"
        assert evaluator.evaluate('concat("Hello", " ", "World")', context) == "Hello World"
        assert evaluator.evaluate('contains("hello world", "world")', context) is True
        assert evaluator.evaluate('contains("hello", "world")', context) is False
        
        # With properties
        assert evaluator.evaluate("lower(title)", context) == "test note"
    
    def test_function_len(self, evaluator, context):
        """Test len function."""
        assert evaluator.evaluate('len("hello")', context) == 5
        assert evaluator.evaluate("len(tags)", context) == 2
        assert evaluator.evaluate("len(title)", context) == 9
    
    def test_function_regex_match(self, evaluator, context):
        """Test regex matching."""
        assert evaluator.evaluate('regexMatch("test123", "\\d+")', context) is True
        assert evaluator.evaluate('regexMatch("hello", "\\d+")', context) is False
        assert evaluator.evaluate('regexMatch(title, "Test.*")', context) is True
    
    def test_computed_reference(self, evaluator, context):
        """Test referencing computed fields."""
        context.computed_values["score"] = 100
        
        assert evaluator.evaluate("@score", context) == 100
        assert evaluator.evaluate("@score / 2", context) == 50
    
    def test_complex_expressions(self, evaluator, context):
        """Test complex nested expressions."""
        # Complex arithmetic
        result = evaluator.evaluate("(priority * 2) + (10 - 3)", context)
        assert result == 17
        
        # Nested ternary
        expr = 'priority > 3 ? (status == "active" ? "High Active" : "High Inactive") : "Low"'
        result = evaluator.evaluate(expr, context)
        assert result == "High Active"
        
        # Function with expression arguments
        result = evaluator.evaluate("clamp(priority * 2, 0, priority + 5)", context)
        assert result == 10  # priority * 2 = 10, priority + 5 = 10, so result is 10


class TestComputedFields:
    """Test computed field evaluation."""
    
    def test_evaluate_computed_fields(self):
        """Test evaluating all computed fields for a note."""
        # Create a base with computed fields
        base = BaseFile(
            id="test-base",
            name="Test Base",
            version=1,
            source=BaseSource(folders=["/"], filters=[]),
            computed=[
                ComputedField(
                    id="priority-score",
                    expr="priority * 10",
                    type="number"
                ),
                ComputedField(
                    id="is-urgent",
                    expr='contains(tags, "urgent")',
                    type="boolean"
                ),
                ComputedField(
                    id="display-status",
                    expr='upper(status)',
                    type="string"
                )
            ],
            views=[
                BaseView(
                    id="view1",
                    name="View 1",
                    type=ViewType.TABLE,
                    columns=[
                        BaseColumn(id="test", header="Test", source="title")
                    ]
                )
            ]
        )
        
        # Note data
        note_data = {
            "title": "Test Note",
            "status": "active",
            "priority": 5,
            "tags": "work,urgent"
        }
        
        # Evaluate computed fields
        computed = evaluate_computed_fields(base, note_data)
        
        assert computed["priority-score"] == 50
        assert computed["is-urgent"] is True
        assert computed["display-status"] == "ACTIVE"
    
    def test_computed_field_references(self):
        """Test computed fields referencing other computed fields."""
        base = BaseFile(
            id="test-base",
            name="Test Base",
            version=1,
            source=BaseSource(folders=["/"], filters=[]),
            computed=[
                ComputedField(
                    id="base-score",
                    expr="priority * 10",
                    type="number"
                ),
                ComputedField(
                    id="final-score",
                    expr="@base-score + 5",
                    type="number"
                )
            ],
            views=[
                BaseView(
                    id="view1",
                    name="View 1",
                    type=ViewType.TABLE,
                    columns=[
                        BaseColumn(id="test", header="Test", source="title")
                    ]
                )
            ]
        )
        
        note_data = {"priority": 3}
        computed = evaluate_computed_fields(base, note_data)
        
        assert computed["base-score"] == 30
        assert computed["final-score"] == 35
    
    def test_circular_reference_detection(self):
        """Test that circular references are detected."""
        base = BaseFile(
            id="test-base",
            name="Test Base",
            version=1,
            source=BaseSource(folders=["/"], filters=[]),
            computed=[
                ComputedField(
                    id="field-a",
                    expr="@field-b + 1",
                    type="number"
                ),
                ComputedField(
                    id="field-b",
                    expr="@field-a + 1",  # Circular reference!
                    type="number"
                )
            ],
            views=[
                BaseView(
                    id="view1",
                    name="View 1",
                    type=ViewType.TABLE,
                    columns=[
                        BaseColumn(id="test", header="Test", source="title")
                    ]
                )
            ]
        )
        
        note_data = {}
        
        # Should raise CircularReferenceError
        with pytest.raises((CircularReferenceError, ValueError)) as exc_info:
            evaluate_computed_fields(base, note_data)
        
        assert "Circular" in str(exc_info.value) or "Failed to evaluate" in str(exc_info.value)