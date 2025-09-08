#!/usr/bin/env python3
"""
Example usage of the new test organization structure.
Demonstrates how to run different test categories.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Example of using the organized test structure"""
    
    print("🧪 Graph RAG MCP Server - Test Organization Example")
    print("=" * 60)
    
    examples = [
        ("Unit Tests", "pytest tests/unit/ -v"),
        ("Integration Tests", "pytest tests/integration/ -v"),
        ("Evaluation Framework", "python tests/evals/runner.py"),
        ("Fast Tests Only", "pytest -m 'unit or (integration and not slow)' -v"),
        ("Coverage Report", "pytest --cov=src --cov-report=html"),
        ("Test Runner Script", "python scripts/run_tests.py --help"),
    ]
    
    print("Available test commands:")
    print()
    
    for name, command in examples:
        print(f"📋 {name}")
        print(f"   {command}")
        print()
    
    print("Test Structure Overview:")
    print("""
tests/
├── unit/                     # Fast, isolated tests
│   ├── test_query_intent.py
│   ├── test_fuzzy_matching.py
│   ├── test_relationship_weighting.py
│   └── test_uri_generation.py
├── integration/              # Component interaction tests
│   └── test_mcp_server.py
├── evals/                    # Evaluation framework
│   ├── runner.py
│   ├── metrics.py
│   └── suites/core_mcp.py
└── fixtures/                 # Test data and utilities
    ├── content/
    ├── mocks.py
    └── factories.py
""")
    
    print("Benefits of the new organization:")
    benefits = [
        "✅ Scalable test organization by category",
        "✅ Isolated unit tests with no dependencies",
        "✅ Comprehensive evaluation framework",
        "✅ Reusable test fixtures and utilities",
        "✅ Performance metrics and reporting",
        "✅ Easy CI/CD integration",
        "✅ Clear separation of concerns"
    ]
    
    for benefit in benefits:
        print(f"   {benefit}")
    
    print(f"\n📁 All test files organized in: {project_root}/tests/")
    print(f"📊 Test reports will be saved in: {project_root}/tests/evals/reports/")
    print("🧹 Old test files removed from root directory")


if __name__ == "__main__":
    main()