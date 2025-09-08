#!/usr/bin/env python3
"""
Evaluation runner for Graph RAG MCP Server.
Provides the core framework for running evaluation suites and collecting results.
"""

from __future__ import annotations
import os
import sys
import importlib
import json
import tempfile
import shutil
from pathlib import Path
from typing import Any, Coroutine, Dict, List, Optional, Protocol, cast
from datetime import datetime
import asyncio

# Ensure repo root in sys.path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class AttrDict(dict):
    """Dictionary with attribute-style access"""
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(item) from e
    def __setattr__(self, key, value):
        self[key] = value


def _to_attrdict(obj: Any) -> Any:
    """Convert nested dicts/lists to AttrDict objects"""
    if isinstance(obj, dict):
        return AttrDict({k: _to_attrdict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_attrdict(v) for v in obj]
    return obj


class EvalSuite(Protocol):
    """Protocol for evaluation suites"""
    
    def setup(self) -> None:
        """Setup the evaluation environment"""
        ...
    
    def run_evals(self) -> Dict[str, Any]:
        """Run all evaluations in this suite"""
        ...
    
    def cleanup(self) -> None:
        """Clean up after evaluations"""
        ...


class EvalResult:
    """Container for evaluation results"""
    
    def __init__(self, suite_name: str, results: Dict[str, Any]):
        self.suite_name = suite_name
        self.results = results
        self.timestamp = datetime.now().isoformat()
        self.success = results.get('success', False)
        self.metrics = results.get('metrics', {})
        self.details = results.get('details', {})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'suite_name': self.suite_name,
            'timestamp': self.timestamp,
            'success': self.success,
            'metrics': self.metrics,
            'details': self.details,
            'results': self.results
        }


class EvalRunner:
    """Main evaluation runner"""
    
    def __init__(self, test_dir: Optional[Path] = None, output_dir: Optional[Path] = None):
        self.test_dir = test_dir
        self.output_dir = output_dir or (ROOT / "tests" / "evals" / "reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dirs: List[Path] = []
        
    def call_tool(self, tool_obj, /, **kwargs):
        """Invoke a FastMCP FunctionTool (or plain callable) and normalize output."""
        run_attr = getattr(tool_obj, "run", None)
        if callable(run_attr):
            coro_or_res = run_attr(dict(kwargs))
            try:
                # Await if coroutine
                if asyncio.iscoroutine(coro_or_res):
                    res = asyncio.run(cast(Coroutine[Any, Any, Any], coro_or_res))
                else:
                    res = coro_or_res
            except RuntimeError:
                # If we're already in an event loop, create a new loop
                loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(loop)
                    res = loop.run_until_complete(cast(Coroutine[Any, Any, Any], coro_or_res))
                finally:
                    loop.close()
                    asyncio.set_event_loop(None)

            # Parse ToolResult -> content
            contents = getattr(res, "content", None)
            if contents and len(contents) > 0:
                text = getattr(contents[0], "text", None)
                if text is not None:
                    # Try JSON first
                    try:
                        data = json.loads(text)
                        return _to_attrdict(data)
                    except Exception:
                        # Fallback to raw text
                        return text
            return res
        # Plain callable fallback
        return tool_obj(**kwargs)

    def reset_app_state_to_test_dir(self, test_dir: Path):
        """Point settings.vaults to test_dir and reinit global app_state."""
        # Set env so downstream components also see it if they read afresh
        os.environ["OBSIDIAN_RAG_VAULTS"] = str(test_dir)

        from src import config as cfg
        # Override settings in-place
        cfg.settings.vaults = [test_dir]

        # (Re)import mcp_server and reset app_state, ensuring existing stores are closed first
        # If already imported, close open resources to avoid database conflicts
        if "src.mcp_server" in sys.modules:
            mcp_server = sys.modules["src.mcp_server"]
            try:
                # Best-effort close of existing unified store
                if hasattr(mcp_server, "app_state") and hasattr(mcp_server.app_state, "unified_store"):
                    # ChromaDB handles connection cleanup automatically
                    pass
            except Exception:
                pass
            mcp_server = importlib.reload(mcp_server)
        else:
            import src.mcp_server as mcp_server

        # Now that module-level app_state has reinitialized, return module
        return mcp_server

    def reindex_all(self, mcp_server_module) -> None:
        """Reindex all content in the test environment"""
        res = self.call_tool(mcp_server_module.reindex_vault, target="all", full_reindex=True)
        if not getattr(res, "success", False):
            raise RuntimeError(f"Reindex failed: {res}")

    def setup_test_environment(self, test_content_source: Optional[Path] = None) -> Path:
        """Setup a temporary test environment with content"""
        from scripts.build_test_content import write_baseline, BASELINE_DIR
        
        # Ensure baseline content exists
        write_baseline()
        
        # Use provided source or default baseline
        source_dir = test_content_source or BASELINE_DIR
        
        # Create temporary working directory
        temp_dir = Path(tempfile.mkdtemp(prefix="eval_mcp_", dir=str(ROOT)))
        self.temp_dirs.append(temp_dir)
        
        work_dir = temp_dir / "eval_content"
        shutil.copytree(source_dir, work_dir)
        
        return work_dir

    def run_suite(self, suite_name: str, suite_class: type, **kwargs) -> EvalResult:
        """Run a single evaluation suite"""
        print(f"\n🧪 Running {suite_name} evaluation suite")
        print("=" * 60)
        
        try:
            # Setup test environment
            test_dir = self.setup_test_environment(kwargs.get('test_content_source'))
            
            # Initialize suite
            suite = suite_class(test_dir=test_dir, eval_runner=self, **kwargs)
            
            # Run the evaluation
            suite.setup()
            results = suite.run_evals()
            suite.cleanup()
            
            # Create result object
            eval_result = EvalResult(suite_name, results)
            
            # Save results
            self.save_results(eval_result)
            
            return eval_result
            
        except Exception as e:
            error_results = {
                'success': False,
                'error': str(e),
                'metrics': {},
                'details': {}
            }
            return EvalResult(suite_name, error_results)

    def run_multiple_suites(self, suites: Dict[str, type], **kwargs) -> List[EvalResult]:
        """Run multiple evaluation suites"""
        results = []
        
        print(f"🚀 Running {len(suites)} evaluation suites")
        print("=" * 80)
        
        for suite_name, suite_class in suites.items():
            result = self.run_suite(suite_name, suite_class, **kwargs)
            results.append(result)
            
            # Print summary
            status = "✅ PASSED" if result.success else "❌ FAILED"
            print(f"\n{suite_name}: {status}")
            if result.metrics:
                for metric, value in result.metrics.items():
                    print(f"  {metric}: {value}")
        
        # Overall summary
        print("\n" + "=" * 80)
        passed = sum(1 for r in results if r.success)
        total = len(results)
        
        if passed == total:
            print(f"🎉 ALL EVALUATIONS PASSED ({passed}/{total})")
        else:
            print(f"❌ {total - passed} evaluations failed ({passed}/{total} passed)")
        
        return results

    def save_results(self, result: EvalResult) -> None:
        """Save evaluation results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{result.suite_name}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        print(f"📊 Results saved to: {filepath}")

    def cleanup(self) -> None:
        """Clean up temporary directories"""
        for temp_dir in self.temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        self.temp_dirs.clear()

    def assert_true(self, condition: bool, message: str) -> None:
        """Assertion helper for evaluations"""
        if not condition:
            raise AssertionError(message)


def main():
    """Main entry point for running evaluations"""
    # Import available evaluation suites
    from suites.core_mcp import CoreMCPEvals
    
    # Define available suites
    available_suites = {
        'core_mcp': CoreMCPEvals,
    }
    
    # Run evaluations
    runner = EvalRunner()
    
    try:
        results = runner.run_multiple_suites(available_suites)
        success = all(r.success for r in results)
        sys.exit(0 if success else 1)
    finally:
        runner.cleanup()


if __name__ == "__main__":
    main()