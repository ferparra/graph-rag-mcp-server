#!/usr/bin/env python3
"""
Test runner utility for Graph RAG MCP Server.
Provides convenient commands for running different test categories.
"""

import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Optional

# Project root
ROOT = Path(__file__).parent.parent


def run_command(cmd: List[str], description: str) -> int:
    """Run a command and return exit code"""
    print(f"🔄 {description}")
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, cwd=ROOT, check=False)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
        else:
            print(f"❌ {description} failed with exit code {result.returncode}")
        return result.returncode
    except Exception as e:
        print(f"❌ {description} failed with error: {e}")
        return 1


def run_unit_tests(verbose: bool = False, coverage: bool = False) -> int:
    """Run unit tests"""
    cmd = ["python", "-m", "pytest", "tests/unit/", "-m", "unit"]
    if verbose:
        cmd.append("-v")
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=term-missing"])
    
    return run_command(cmd, "Running unit tests")


def run_integration_tests(verbose: bool = False) -> int:
    """Run integration tests"""
    cmd = ["python", "-m", "pytest", "tests/integration/", "-m", "integration"]
    if verbose:
        cmd.append("-v")
    
    return run_command(cmd, "Running integration tests")


def run_e2e_tests(verbose: bool = False) -> int:
    """Run end-to-end tests"""
    cmd = ["python", "-m", "pytest", "tests/e2e/", "-m", "e2e"]
    if verbose:
        cmd.append("-v")
    
    return run_command(cmd, "Running end-to-end tests")


def run_evaluations() -> int:
    """Run evaluation framework"""
    cmd = ["python", "tests/evals/runner.py"]
    return run_command(cmd, "Running evaluation framework")


def run_benchmarks(verbose: bool = False) -> int:
    """Run performance benchmarks"""
    cmd = ["python", "-m", "pytest", "tests/benchmarks/", "--benchmark-only"]
    if verbose:
        cmd.append("-v")
    
    return run_command(cmd, "Running performance benchmarks")


def run_all_tests(verbose: bool = False, coverage: bool = False, skip_slow: bool = False) -> int:
    """Run all test categories"""
    cmd = ["python", "-m", "pytest"]
    
    if skip_slow:
        cmd.extend(["-m", "not slow"])
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend([
            "--cov=src", 
            "--cov-report=term-missing",
            "--cov-report=html:tests/coverage_html"
        ])
    
    return run_command(cmd, "Running all tests")


def run_fast_tests(verbose: bool = False) -> int:
    """Run only fast tests (unit + quick integration)"""
    cmd = ["python", "-m", "pytest", "-m", "unit or (integration and not slow)"]
    if verbose:
        cmd.append("-v")
    
    return run_command(cmd, "Running fast tests")


def check_test_coverage() -> int:
    """Generate detailed coverage report"""
    cmd = [
        "python", "-m", "pytest", 
        "--cov=src", 
        "--cov-report=html:tests/coverage_html",
        "--cov-report=term-missing",
        "--cov-fail-under=80"
    ]
    
    result = run_command(cmd, "Generating coverage report")
    
    if result == 0:
        print(f"📊 Coverage report generated: {ROOT}/tests/coverage_html/index.html")
    
    return result


def lint_tests() -> int:
    """Run linting on test files"""
    cmd = ["python", "-m", "pytest", "--flake8", "tests/"]
    return run_command(cmd, "Linting test files")


def typecheck_tests() -> int:
    """Run type checking on test files"""
    cmd = ["python", "-m", "pytest", "--mypy", "tests/"]
    return run_command(cmd, "Type checking test files")


def clean_test_artifacts() -> int:
    """Clean up test artifacts and temporary files"""
    import shutil
    
    artifacts = [
        ROOT / "tests" / "coverage_html",
        ROOT / "tests" / "coverage.xml", 
        ROOT / ".coverage",
        ROOT / ".pytest_cache",
        ROOT / "tests" / ".pytest_cache"
    ]
    
    cleaned = 0
    for artifact in artifacts:
        if artifact.exists():
            if artifact.is_dir():
                shutil.rmtree(artifact)
            else:
                artifact.unlink()
            cleaned += 1
            print(f"🗑️  Removed: {artifact}")
    
    # Clean up temporary test directories
    temp_dirs = [p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith(('test_', 'eval_', 'temp_'))]
    for temp_dir in temp_dirs:
        try:
            shutil.rmtree(temp_dir)
            cleaned += 1
            print(f"🗑️  Removed temp dir: {temp_dir}")
        except Exception:
            pass
    
    print(f"✅ Cleaned {cleaned} test artifacts")
    return 0


def list_tests() -> int:
    """List all available tests"""
    cmd = ["python", "-m", "pytest", "--collect-only", "-q"]
    return run_command(cmd, "Listing all tests")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Test runner for Graph RAG MCP Server")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--skip-slow", action="store_true", help="Skip slow tests")
    
    subparsers = parser.add_subparsers(dest="command", help="Test commands")
    
    # Test category commands
    unit_parser = subparsers.add_parser("unit", help="Run unit tests")
    integration_parser = subparsers.add_parser("integration", help="Run integration tests")
    e2e_parser = subparsers.add_parser("e2e", help="Run end-to-end tests")
    evals_parser = subparsers.add_parser("evals", help="Run evaluation framework")
    benchmarks_parser = subparsers.add_parser("benchmarks", help="Run performance benchmarks")
    all_parser = subparsers.add_parser("all", help="Run all tests")
    fast_parser = subparsers.add_parser("fast", help="Run only fast tests")
    
    # Utility commands
    subparsers.add_parser("coverage", help="Generate detailed coverage report")
    subparsers.add_parser("lint", help="Lint test files")
    subparsers.add_parser("typecheck", help="Type check test files")
    subparsers.add_parser("clean", help="Clean test artifacts")
    subparsers.add_parser("list", help="List all tests")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Route to appropriate function
    command_map = {
        "unit": lambda: run_unit_tests(args.verbose, args.coverage),
        "integration": lambda: run_integration_tests(args.verbose),
        "e2e": lambda: run_e2e_tests(args.verbose),
        "evals": run_evaluations,
        "benchmarks": lambda: run_benchmarks(args.verbose),
        "all": lambda: run_all_tests(args.verbose, args.coverage, args.skip_slow),
        "fast": lambda: run_fast_tests(args.verbose),
        "coverage": check_test_coverage,
        "lint": lint_tests,
        "typecheck": typecheck_tests,
        "clean": clean_test_artifacts,
        "list": list_tests,
    }
    
    if args.command in command_map:
        return command_map[args.command]()
    else:
        print(f"❌ Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())