# Graph RAG MCP Server Test Suite

Comprehensive testing framework for the Graph RAG MCP Server with organized test categories, evaluation metrics, and scalable test management.

## Test Organization

```
tests/
├── unit/                     # Unit tests (no external dependencies)
├── integration/              # Integration tests (with dependencies)
├── e2e/                      # End-to-end tests
├── evals/                    # Evaluation frameworks
├── fixtures/                 # Test data and content
├── benchmarks/              # Performance tests
├── conftest.py              # Shared pytest fixtures
└── README.md               # This documentation
```

## Quick Start

### Running Tests

Recommended options depending on your environment:

```bash
# Using uv (recommended)
uv sync --extra test              # install test dependencies
uv run pytest -q                  # run the full suite

# Using the local virtualenv created for this repo
PYTHONPATH=. .venv/bin/pytest -q  # ensures local 'tests' package is importable

# Run specific categories
PYTHONPATH=. .venv/bin/pytest tests/unit -q           # Unit tests only
PYTHONPATH=. .venv/bin/pytest tests/integration -q    # Integration tests only
PYTHONPATH=. .venv/bin/pytest tests/evals -q          # Evaluation tests only

# Run with markers
PYTHONPATH=. .venv/bin/pytest -m unit           # Unit tests
PYTHONPATH=. .venv/bin/pytest -m integration    # Integration tests
PYTHONPATH=. .venv/bin/pytest -m "not slow"     # Skip slow tests

# Run specific test files
PYTHONPATH=. .venv/bin/pytest tests/unit/test_query_intent.py -q
PYTHONPATH=. .venv/bin/pytest tests/integration/test_mcp_server.py -q

# Coverage (configured in pytest.ini)
PYTHONPATH=. .venv/bin/pytest --cov -q
```

Notes:
- If you encounter `ModuleNotFoundError: No module named 'tests'`, prepend `PYTHONPATH=.` as shown above.
- The integration tests run fully offline. Embeddings fall back to a built‑in default and DSPy caches are written to `.cache/` in the repo.

### Running Evaluations

```bash
# Run the evaluation framework
python tests/evals/runner.py

# Run specific evaluation suites
python -m tests.evals.suites.core_mcp

# View evaluation reports
ls tests/evals/reports/
```

## Test Categories

### Unit Tests (`tests/unit/`)

Fast, isolated tests with no external dependencies. Test individual components and algorithms.

**Files:**
- `test_query_intent.py` - Query intent detection logic
- `test_fuzzy_matching.py` - Fuzzy string and tag matching algorithms
- `test_relationship_weighting.py` - Graph relationship weighting logic
- `test_uri_generation.py` - Obsidian URI generation

**Usage:**
```bash
pytest tests/unit/ -v
```

### Integration Tests (`tests/integration/`)

Tests that verify component interaction with real dependencies (ChromaDB, file system).

**Files:**
- `test_mcp_server.py` - Full MCP server integration testing

**Usage:**
```bash
PYTHONPATH=. .venv/bin/pytest tests/integration -v
```

### End-to-End Tests (`tests/e2e/`)

Complete workflow tests simulating real user scenarios.

**Usage:**
```bash
PYTHONPATH=. .venv/bin/pytest tests/e2e -v
```

### Evaluation Framework (`tests/evals/`)

Comprehensive evaluation system with metrics collection and reporting.

**Components:**
- `runner.py` - Main evaluation runner
- `metrics.py` - Performance and quality metrics
- `suites/core_mcp.py` - Core MCP functionality evaluation

**Features:**
- Automated metrics collection
- Performance timing
- Quality scoring
- JSON report generation

### Benchmarks (`tests/benchmarks/`)

Performance and stress tests for scalability validation.

**Usage:**
```bash
PYTHONPATH=. .venv/bin/pytest tests/benchmarks -v --benchmark-only
```

## Test Fixtures and Data

### Shared Fixtures (`tests/conftest.py`)

- `temp_vault` - Temporary vault with planet test content
- `extended_vault` - Full test vault (planets + health + projects)
- `mock_unified_store` - Mock ChromaDB store
- `mock_vault_searcher` - Mock search functionality
- `test_assertions` - Custom assertion helpers

### Test Data (`tests/fixtures/`)

- `content/` - Test markdown files and vaults
- `mocks.py` - Mock objects for testing
- `factories.py` - Test data generation utilities

### Test Content

**Planet Notes:** Earth, Mars, Link Map with verification codes and cross-links
**Health & Fitness:** Notes with hierarchical tags for fuzzy matching tests
**Project Notes:** PARA taxonomy examples for organizational testing

## Test Markers

Use pytest markers to categorize and run specific test types:

```bash
# Available markers
pytest --markers

# Common marker combinations
pytest -m "unit and not slow"
pytest -m "integration and requires_chroma"
pytest -m "eval or benchmark"
```

**Standard Markers:**
- `unit` - Unit tests (no external dependencies)
- `integration` - Integration tests (with dependencies)
- `e2e` - End-to-end tests
- `eval` - Evaluation tests
- `slow` - Tests that take >5 seconds
- `requires_chroma` - Tests needing ChromaDB
- `requires_gemini` - Tests needing Gemini API key
- `requires_vault` - Tests needing test vault content

## Coverage Reports

```bash
# Generate coverage reports (threshold defined in pytest.ini)
PYTHONPATH=. .venv/bin/pytest --cov=src \
  --cov-report=html:tests/coverage_html \
  --cov-report=xml:tests/coverage.xml \
  --cov-report=term

# View HTML coverage report
open tests/coverage_html/index.html

# Coverage requirements
# - Minimum 80% coverage (configured in pytest.ini)
# - Excludes scripts/ and site-packages
```

## Writing New Tests

### Unit Test Example

```python
#!/usr/bin/env python3
"""Unit test example"""

def test_my_function():
    """Test a specific function"""
    result = my_function("input")
    assert result == "expected_output"
    
def test_edge_case():
    """Test edge cases"""
    with pytest.raises(ValueError):
        my_function(None)
```

### Integration Test Example

```python
#!/usr/bin/env python3
"""Integration test example"""

def test_full_workflow(temp_vault, mock_unified_store):
    """Test complete workflow"""
    # Setup
    server = setup_mcp_server(temp_vault)
    
    # Test
    result = server.search_notes("test query")
    
    # Validate
    assert result.total_results > 0
```

### Using Test Fixtures

```python
def test_with_fixtures(temp_vault, test_assertions):
    """Example using shared fixtures"""
    # temp_vault provides test content
    # test_assertions provides validation helpers
    
    result = search_in_vault(temp_vault, "Earth")
    test_assertions.assert_search_result_valid(result)
```

## Evaluation Framework

### Creating New Evaluation Suites

1. Create new suite in `tests/evals/suites/`
2. Implement required methods: `setup()`, `run_evals()`, `cleanup()`
3. Add to runner in `tests/evals/runner.py`

```python
class MyEvalSuite:
    def __init__(self, test_dir: Path, eval_runner, **kwargs):
        self.test_dir = test_dir
        self.eval_runner = eval_runner
        self.metrics = EvalMetrics()
    
    def setup(self) -> None:
        """Setup evaluation environment"""
        pass
    
    def run_evals(self) -> Dict[str, Any]:
        """Run all evaluations"""
        return {'success': True, 'metrics': {}}
    
    def cleanup(self) -> None:
        """Clean up after evaluations"""
        pass
```

### Metrics Collection

The evaluation framework automatically collects:
- Response times
- Result quality scores
- Success rates
- Graph connectivity metrics
- Overall performance scores

Results are saved as JSON reports in `tests/evals/reports/`.

## Best Practices

### Test Structure

1. **Arrange** - Set up test data and environment
2. **Act** - Execute the code being tested
3. **Assert** - Verify the results

### Naming Conventions

- Test files: `test_*.py`
- Test functions: `test_*`
- Test classes: `Test*`
- Descriptive names explaining what is tested

### Test Independence

- Each test should be independent
- Use fixtures for shared setup
- Clean up temporary resources
- Don't rely on test execution order

### Performance Considerations

- Mark slow tests with `@pytest.mark.slow`
- Use mocks for external dependencies in unit tests
- Profile tests with `--benchmark-only` for performance tests

## Continuous Integration

Configure CI to run different test categories:

```yaml
# Example CI configuration
- name: Unit Tests
  run: pytest tests/unit/ -m unit

- name: Integration Tests  
  run: pytest tests/integration/ -m integration

- name: Evaluations
  run: python tests/evals/runner.py
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure project root is on `PYTHONPATH` (use `PYTHONPATH=.`)
   - Integration tests add `src/` automatically; no extra setup needed

2. **ChromaDB Connection Issues**
   - Use mock stores for unit tests
   - Ensure proper cleanup in integration tests

3. **Test Data Issues**
   - Verify fixtures are properly created
   - Check test content paths

4. **Slow Tests**
   - Use `pytest -x` to stop on first failure
   - Run specific test categories: `pytest tests/unit/`

5. **No Network Environments**
   - Embedding downloads are not required: tests fall back to Chroma’s `DefaultEmbeddingFunction`
   - DSPy disk cache is redirected to `.cache/` (override with `DSPY_CACHEDIR`)

### Debug Mode

```bash
# Run with detailed output
pytest -v -s

# Run single test with debugging
pytest tests/unit/test_query_intent.py::test_query_intent_detection -v -s

# Drop into debugger on failure
pytest --pdb
```

## Test Metrics and Reporting

The test suite provides comprehensive metrics:

- **Code Coverage**: Line and branch coverage reports
- **Test Performance**: Execution time tracking
- **Quality Metrics**: Success rates and accuracy scores
- **Evaluation Reports**: Detailed JSON reports with metrics

View reports in:
- `tests/coverage_html/` - HTML coverage reports
- `tests/evals/reports/` - Evaluation JSON reports
- `tests/benchmark_results/` - Performance benchmark data
