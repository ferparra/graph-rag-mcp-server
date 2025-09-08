#!/usr/bin/env python3
"""
Core MCP evaluation suite.
Adapted from the original run_mcp_evals.py script with enhanced metrics.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Any, List
from tests.evals.metrics import EvalMetrics, PerformanceTimer

# Ensure repo root in sys.path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


class CoreMCPEvals:
    """Core MCP functionality evaluation suite"""
    
    def __init__(self, test_dir: Path, eval_runner, **kwargs):
        self.test_dir = test_dir
        self.eval_runner = eval_runner
        self.mcp_server = None
        self.metrics = EvalMetrics()
    
    def setup(self) -> None:
        """Setup evaluation environment"""
        print("Setting up Core MCP evaluation environment...")
        
        # Reset app state to test directory
        self.mcp_server = self.eval_runner.reset_app_state_to_test_dir(self.test_dir)
        
        # Reindex all content
        self.eval_runner.reindex_all(self.mcp_server)
        
        # Start timing
        self.metrics.start_timing()
        
        print(f"✓ Core MCP environment ready with test content: {self.test_dir}")
    
    def test_search_notes(self) -> Dict[str, Any]:
        """Test basic note search functionality"""
        tests = [
            {
                "query": "earth verification code",
                "expect_contains": "Earth",
            },
            {
                "query": "mars verification code", 
                "expect_contains": "Mars",
            },
            {
                "query": "CONST_TOKEN_ALPHA_earth_93e8a4",
                "expect_contains": "Earth",
            },
            {
                "query": "CONST_TOKEN_ALPHA_mars_77c2d1",
                "expect_contains": "Mars",
            },
        ]
        
        results = []
        for test_case in tests:
            with PerformanceTimer() as timer:
                res = self.eval_runner.call_tool(
                    self.mcp_server.search_notes, 
                    query=test_case["query"], 
                    k=3
                )
            
            # Record metrics
            search_metrics = self.metrics.record_search(
                query=test_case["query"],
                results=res.hits,
                response_time_ms=timer.duration_ms(),
                strategy_used="vector"
            )
            
            # Validate results
            success = False
            if res.total_results > 0:
                top_texts = [h.get("meta", {}).get("title", "") for h in res.hits]
                success = any(test_case["expect_contains"].lower() in x.lower() for x in top_texts)
            
            results.append({
                'query': test_case["query"],
                'expected': test_case["expect_contains"],
                'found_results': res.total_results,
                'success': success,
                'response_time_ms': timer.duration_ms()
            })
            
            if not success:
                print(f"❌ Search failed for query: {test_case['query']}")
            else:
                print(f"✓ Search successful for query: {test_case['query']}")
        
        passed = sum(1 for r in results if r['success'])
        return {
            'test_name': 'search_notes',
            'passed': passed,
            'total': len(tests),
            'success': passed == len(tests),
            'details': results
        }
    
    def test_smart_search(self) -> Dict[str, Any]:
        """Test enhanced smart search functionality"""
        tests = [
            {
                "query": "Show me notes tagged with planets",
                "expected_strategy": "tag",
            },
            {
                "query": "What links to Mars?", 
                "expected_strategy": "graph",
            },
            {
                "query": "earth verification details",
                "expected_strategy": "vector",
            },
        ]
        
        results = []
        for test_case in tests:
            try:
                with PerformanceTimer() as timer:
                    res = self.eval_runner.call_tool(
                        self.mcp_server.smart_search,
                        query=test_case["query"],
                        k=3
                    )
                
                # Record metrics
                search_metrics = self.metrics.record_search(
                    query=test_case["query"],
                    results=res.hits,
                    response_time_ms=timer.duration_ms(),
                    strategy_used=res.strategy_used
                )
                
                # Validate strategy selection
                strategy_match = res.strategy_used == test_case["expected_strategy"]
                has_results = res.total_results > 0
                
                success = strategy_match and has_results
                
                results.append({
                    'query': test_case["query"],
                    'expected_strategy': test_case["expected_strategy"],
                    'actual_strategy': res.strategy_used,
                    'found_results': res.total_results,
                    'success': success,
                    'response_time_ms': timer.duration_ms()
                })
                
                if success:
                    print(f"✓ Smart search successful: {test_case['query']} → {res.strategy_used}")
                else:
                    print(f"❌ Smart search failed: {test_case['query']} (expected {test_case['expected_strategy']}, got {res.strategy_used})")
                    
            except Exception as e:
                print(f"❌ Smart search error for '{test_case['query']}': {e}")
                results.append({
                    'query': test_case["query"],
                    'success': False,
                    'error': str(e)
                })
        
        passed = sum(1 for r in results if r.get('success', False))
        return {
            'test_name': 'smart_search',
            'passed': passed,
            'total': len(tests),
            'success': passed == len(tests),
            'details': results
        }
    
    def test_graph_neighbors(self) -> Dict[str, Any]:
        """Test graph neighbor discovery"""
        tests = [
            {
                "note": "Earth",
                "expect_neighbor": "Mars",
                "depth": 1
            },
            {
                "note": "Mars", 
                "expect_neighbor": "Earth",
                "depth": 1
            }
        ]
        
        results = []
        for test_case in tests:
            with PerformanceTimer() as timer:
                res = self.eval_runner.call_tool(
                    self.mcp_server.graph_neighbors,
                    note_id_or_title=test_case["note"],
                    depth=test_case["depth"]
                )
            
            # Record metrics
            graph_metrics = self.metrics.record_graph_operation(
                operation="graph_neighbors",
                result={'nodes': res.nodes, 'edges': []},
                response_time_ms=timer.duration_ms()
            )
            
            # Validate neighbors
            neighbor_names = [n.get("title") or n.get("id") for n in res.nodes]
            success = any(test_case["expect_neighbor"] in (name or "") for name in neighbor_names)
            
            results.append({
                'note': test_case["note"],
                'expected_neighbor': test_case["expect_neighbor"],
                'found_neighbors': len(res.nodes),
                'neighbor_names': neighbor_names,
                'success': success,
                'response_time_ms': timer.duration_ms()
            })
            
            if success:
                print(f"✓ Graph neighbors found for {test_case['note']}")
            else:
                print(f"❌ Expected neighbor {test_case['expect_neighbor']} not found for {test_case['note']}")
        
        passed = sum(1 for r in results if r['success'])
        return {
            'test_name': 'graph_neighbors',
            'passed': passed,
            'total': len(tests),
            'success': passed == len(tests),
            'details': results
        }
    
    def test_get_subgraph(self) -> Dict[str, Any]:
        """Test subgraph extraction"""
        with PerformanceTimer() as timer:
            res = self.eval_runner.call_tool(
                self.mcp_server.get_subgraph,
                seed_notes=["Earth"],
                depth=1
            )
        
        # Record metrics
        graph_metrics = self.metrics.record_graph_operation(
            operation="get_subgraph",
            result={'nodes': res.nodes, 'edges': res.edges},
            response_time_ms=timer.duration_ms()
        )
        
        success = len(res.nodes) >= 2
        
        return {
            'test_name': 'get_subgraph',
            'passed': 1 if success else 0,
            'total': 1,
            'success': success,
            'details': {
                'seed_notes': ["Earth"],
                'nodes_found': len(res.nodes),
                'edges_found': len(res.edges),
                'response_time_ms': timer.duration_ms()
            }
        }
    
    def test_backlinks_and_tags(self) -> Dict[str, Any]:
        """Test backlink and tag functionality"""
        results = []
        
        # Test backlinks
        for planet in ["Earth", "Mars"]:
            with PerformanceTimer() as timer:
                backlinks = self.eval_runner.call_tool(
                    self.mcp_server.get_backlinks,
                    note_id_or_path=planet
                )
            
            backlink_titles = [b.get("title") or b.get("id") or b for b in backlinks]
            has_link_map = any("Link Map" in (title or "") for title in backlink_titles)
            
            results.append({
                'operation': 'backlinks',
                'note': planet,
                'found_backlinks': len(backlinks),
                'has_link_map': has_link_map,
                'success': has_link_map,
                'response_time_ms': timer.duration_ms()
            })
        
        # Test tag search
        with PerformanceTimer() as timer:
            tagged_notes = self.eval_runner.call_tool(
                self.mcp_server.get_notes_by_tag,
                tag="topic/planets"
            )
        
        note_titles = [n.get("title") or n.get("id") for n in tagged_notes]
        has_earth = any("Earth" in (title or "") for title in note_titles)
        has_mars = any("Mars" in (title or "") for title in note_titles)
        
        results.append({
            'operation': 'tag_search',
            'tag': 'topic/planets',
            'found_notes': len(tagged_notes),
            'has_earth': has_earth,
            'has_mars': has_mars,
            'success': has_earth and has_mars,
            'response_time_ms': timer.duration_ms()
        })
        
        passed = sum(1 for r in results if r['success'])
        return {
            'test_name': 'backlinks_and_tags',
            'passed': passed,
            'total': len(results),
            'success': passed == len(results),
            'details': results
        }
    
    def test_read_and_properties(self) -> Dict[str, Any]:
        """Test note reading and property operations"""
        results = []
        
        # Test note reading
        earth_rel = Path("eval_content/planets/Earth.md")
        with PerformanceTimer() as timer:
            earth_note = self.eval_runner.call_tool(
                self.mcp_server.read_note,
                note_path=str(earth_rel)
            )
        
        title_correct = "Earth" in earth_note.title
        content_has_token = "93E8A4" in earth_note.content
        
        results.append({
            'operation': 'read_note',
            'note': 'Earth.md',
            'title_correct': title_correct,
            'content_has_token': content_has_token,
            'success': title_correct and content_has_token,
            'response_time_ms': timer.duration_ms()
        })
        
        # Test frontmatter operations
        with PerformanceTimer() as timer:
            frontmatter = self.eval_runner.call_tool(
                self.mcp_server.get_note_properties,
                note_path=str(earth_rel)
            )
        
        has_tags = "tags" in frontmatter
        
        results.append({
            'operation': 'get_properties',
            'note': 'Earth.md',
            'has_tags': has_tags,
            'success': has_tags,
            'response_time_ms': timer.duration_ms()
        })
        
        # Test property updates
        with PerformanceTimer() as timer:
            updated = self.eval_runner.call_tool(
                self.mcp_server.update_note_properties,
                note_path=str(earth_rel),
                properties={"test_flag": True},
                merge=True
            )
        
        test_flag_set = updated.get("test_flag") is True
        
        results.append({
            'operation': 'update_properties',
            'note': 'Earth.md',
            'test_flag_set': test_flag_set,
            'success': test_flag_set,
            'response_time_ms': timer.duration_ms()
        })
        
        passed = sum(1 for r in results if r['success'])
        return {
            'test_name': 'read_and_properties',
            'passed': passed,
            'total': len(results),
            'success': passed == len(results),
            'details': results
        }
    
    def test_create_add_archive(self) -> Dict[str, Any]:
        """Test note creation, content addition, and archiving"""
        results = []
        
        # Test note creation
        with PerformanceTimer() as timer:
            created = self.eval_runner.call_tool(
                self.mcp_server.create_note,
                title="Transient Test Note",
                content="Ephemeral content for CRUD eval. #test/suite",
                folder="scratch",
                tags=["test/suite"],
                enrich=False,
            )
        
        created_path = Path(created["path"])
        note_exists = created_path.exists()
        
        results.append({
            'operation': 'create_note',
            'note_created': note_exists,
            'success': note_exists,
            'response_time_ms': timer.duration_ms()
        })
        
        if note_exists:
            # Test content addition
            with PerformanceTimer() as timer:
                self.eval_runner.call_tool(
                    self.mcp_server.add_content_to_note,
                    note_path=str(created_path),
                    content="\nAdded line."
                )
                
                readback = self.eval_runner.call_tool(
                    self.mcp_server.read_note,
                    note_path=str(created_path)
                )
            
            content_added = "Added line." in readback.content
            
            results.append({
                'operation': 'add_content',
                'content_added': content_added,
                'success': content_added,
                'response_time_ms': timer.duration_ms()
            })
            
            # Test archiving
            with PerformanceTimer() as timer:
                archived_path = self.eval_runner.call_tool(
                    self.mcp_server.archive_note,
                    note_path=str(created_path)
                )
            
            arch_path = Path(archived_path)
            archived_exists = arch_path.exists()
            
            results.append({
                'operation': 'archive_note',
                'archived_exists': archived_exists,
                'success': archived_exists,
                'response_time_ms': timer.duration_ms()
            })
        
        passed = sum(1 for r in results if r['success'])
        return {
            'test_name': 'create_add_archive',
            'passed': passed,
            'total': len(results),
            'success': passed == len(results),
            'details': results
        }
    
    def test_list_notes(self) -> Dict[str, Any]:
        """Test note listing functionality"""
        with PerformanceTimer() as timer:
            notes_list = self.eval_runner.call_tool(
                self.mcp_server.list_notes,
                limit=1000
            )
        
        has_notes = len(notes_list) > 0
        
        return {
            'test_name': 'list_notes',
            'passed': 1 if has_notes else 0,
            'total': 1,
            'success': has_notes,
            'details': {
                'notes_found': len(notes_list),
                'success': has_notes,
                'response_time_ms': timer.duration_ms()
            }
        }
    
    def run_evals(self) -> Dict[str, Any]:
        """Run all core MCP evaluations"""
        print("Running Core MCP evaluation suite...")
        
        # Run all test methods
        test_results = [
            self.test_search_notes(),
            self.test_smart_search(),
            self.test_graph_neighbors(),
            self.test_get_subgraph(),
            self.test_backlinks_and_tags(),
            self.test_read_and_properties(),
            self.test_create_add_archive(),
            self.test_list_notes(),
        ]
        
        # Calculate overall results
        total_passed = sum(r['passed'] for r in test_results)
        total_tests = sum(r['total'] for r in test_results)
        overall_success = all(r['success'] for r in test_results)
        
        # End timing
        self.metrics.end_timing()
        
        # Generate metrics summary
        metrics_summary = self.metrics.overall_summary()
        performance_score = self.metrics.performance_score()
        
        return {
            'success': overall_success,
            'summary': {
                'total_passed': total_passed,
                'total_tests': total_tests,
                'success_rate': total_passed / total_tests if total_tests > 0 else 0,
                'performance_score': performance_score
            },
            'metrics': metrics_summary,
            'test_results': test_results
        }
    
    def cleanup(self) -> None:
        """Clean up evaluation environment"""
        print("Cleaning up Core MCP evaluation environment...")
        # Cleanup is handled by the eval runner