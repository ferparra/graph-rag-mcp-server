#!/usr/bin/env python3
"""
Test data factories for Graph RAG MCP Server.
Provides factory functions for creating test data and fixtures.
"""

import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
import frontmatter


class TestNote(BaseModel):
    """Test note data structure"""
    title: str
    content: str
    tags: List[str]
    links: List[str]
    path: Optional[Path] = Field(default=None)
    frontmatter: Optional[Dict[str, Any]] = Field(default=None)


class TestContentFactory:
    """Factory for creating test content and fixtures"""
    
    @staticmethod
    def create_planet_notes() -> List[TestNote]:
        """Create the standard planet test notes (Earth, Mars, Link Map)"""
        return [
            TestNote(
                title="Earth",
                content="""# Earth

Earth is the third planet from the Sun and the only known planet to harbor life.

## Verification Code

The verification code for Earth is: CONST_TOKEN_ALPHA_earth_93e8a4

## Characteristics

- Third planet from the Sun
- Has liquid water
- Supports life
- One natural satellite (Moon)

## Links

- [[Mars]]
- [[Link Map]]
""",
                tags=["topic/planets", "test/suite"],
                links=["Mars", "Link Map"]
            ),
            TestNote(
                title="Mars",
                content="""# Mars

Mars is the fourth planet from the Sun, often called the "Red Planet".

## Verification Code

The verification code for Mars is: CONST_TOKEN_ALPHA_mars_77c2d1

## Characteristics

- Fourth planet from the Sun
- Red appearance due to iron oxide
- Two small moons (Phobos and Deimos)
- Polar ice caps

## Links

- [[Earth]]
- [[Link Map]]
""",
                tags=["topic/planets", "test/suite"],
                links=["Earth", "Link Map"]
            ),
            TestNote(
                title="Link Map",
                content="""# Link Map

This note contains links to various planets in our test dataset.

## Planets

- [[Earth]] - The blue planet with life
- [[Mars]] - The red planet with ice caps

Both planets have verification codes for testing purposes.

## Test Information

This map is used to verify graph relationships and backlink functionality.
""",
                tags=["links", "test/suite"],
                links=["Earth", "Mars"]
            )
        ]
    
    @staticmethod
    def create_health_fitness_notes() -> List[TestNote]:
        """Create health and fitness test notes for tag matching tests"""
        return [
            TestNote(
                title="Health Goals",
                content="""# Health Goals

Setting and achieving health goals is important for overall wellbeing.

## Current Goals

- Exercise 30 minutes daily
- Drink 8 glasses of water
- Get 8 hours of sleep
- Eat 5 servings of fruits and vegetables

## Progress Tracking

Track daily habits and measure progress monthly.
""",
                tags=["para/area/health", "goals", "lifestyle"],
                links=["Fitness Routine", "Nutrition Plan"]
            ),
            TestNote(
                title="Fitness Routine",
                content="""# Fitness Routine

A structured approach to physical fitness.

## Weekly Schedule

- Monday: Cardio
- Tuesday: Strength training
- Wednesday: Yoga
- Thursday: Cardio
- Friday: Strength training
- Weekend: Rest or light activity

## Equipment Needed

- Dumbbells
- Yoga mat
- Running shoes
""",
                tags=["lifestyle/fitness", "exercise", "routine"],
                links=["Health Goals"]
            ),
            TestNote(
                title="Nutrition Plan",
                content="""# Nutrition Plan

Balanced nutrition for optimal health.

## Daily Requirements

- Protein: 1g per kg body weight
- Carbohydrates: 45-65% of calories
- Fats: 20-35% of calories
- Fiber: 25-35g daily

## Meal Planning

Plan meals weekly and prep ingredients in advance.
""",
                tags=["para/area/health", "nutrition", "planning"],
                links=["Health Goals"]
            )
        ]
    
    @staticmethod
    def create_project_notes() -> List[TestNote]:
        """Create project management test notes for PARA testing"""
        return [
            TestNote(
                title="MCP Server Enhancement",
                content="""# MCP Server Enhancement

Project to enhance the Graph RAG MCP Server with smart search capabilities.

## Objectives

- Implement query intent detection
- Add fuzzy tag matching
- Enhance graph traversal
- Improve chunk-level navigation

## Timeline

- Phase 1: Core enhancements (2 weeks)
- Phase 2: Testing and evaluation (1 week)
- Phase 3: Documentation (3 days)

## Status

Currently in Phase 2 - comprehensive testing underway.
""",
                tags=["para/project", "development", "mcp"],
                links=["Technical Specifications", "Test Plan"]
            ),
            TestNote(
                title="Technical Specifications",
                content="""# Technical Specifications

Detailed technical specifications for the MCP server enhancement.

## Architecture

- Unified ChromaDB store
- Smart search engine
- Enhanced retriever
- Chunk-level graph relationships

## Implementation Details

- Query intent detection using regex patterns
- Fuzzy string matching with Jaccard similarity
- Relationship weighting by type and depth
- Obsidian URI generation for chunks
""",
                tags=["para/resource", "technical", "documentation"],
                links=["MCP Server Enhancement"]
            )
        ]
    
    @staticmethod
    def write_note_to_file(note: TestNote, base_dir: Path) -> Path:
        """Write a TestNote to a markdown file"""
        # Create frontmatter
        fm_data = {
            'title': note.title,
            'tags': note.tags
        }
        if note.frontmatter:
            fm_data.update(note.frontmatter)
        
        # Create post with frontmatter
        post = frontmatter.Post(note.content)
        post.metadata.update(fm_data)
        
        # Determine file path
        if note.path:
            file_path = base_dir / note.path
        else:
            # Generate path from title
            safe_title = "".join(c for c in note.title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            file_path = base_dir / f"{safe_title}.md"
        
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(frontmatter.dumps(post))
        
        return file_path
    
    @staticmethod
    def create_test_vault(notes: List[TestNote], base_name: str = "test_vault") -> Path:
        """Create a temporary test vault with the given notes"""
        temp_dir = Path(tempfile.mkdtemp(prefix=f"{base_name}_"))
        
        for note in notes:
            TestContentFactory.write_note_to_file(note, temp_dir)
        
        return temp_dir
    
    @staticmethod
    def create_standard_test_vault() -> Path:
        """Create the standard test vault with planets + health + projects"""
        all_notes = (
            TestContentFactory.create_planet_notes() +
            TestContentFactory.create_health_fitness_notes() +
            TestContentFactory.create_project_notes()
        )
        
        return TestContentFactory.create_test_vault(all_notes, "standard_test")
    
    @staticmethod
    def create_planets_only_vault() -> Path:
        """Create a test vault with only planet notes"""
        planet_notes = TestContentFactory.create_planet_notes()
        return TestContentFactory.create_test_vault(planet_notes, "planets_only")


class MockDataFactory:
    """Factory for creating mock data objects"""
    
    @staticmethod
    def create_search_result(query: str, num_results: int = 3) -> Dict[str, Any]:
        """Create a mock search result"""
        hits = []
        for i in range(num_results):
            hits.append({
                'id': f'chunk_{i}',
                'text': f'Mock content for query "{query}" - result {i+1}',
                'meta': {
                    'title': f'Test Note {i+1}',
                    'path': f'test/note_{i+1}.md',
                    'chunk_type': 'section',
                    'importance_score': 0.8 - (i * 0.1)
                },
                'distance': 0.1 + (i * 0.1)
            })
        
        return {
            'hits': hits,
            'total_results': num_results,
            'query': query
        }
    
    @staticmethod
    def create_graph_result(node_count: int = 3, edge_count: int = 2) -> Dict[str, Any]:
        """Create a mock graph result"""
        nodes = []
        for i in range(node_count):
            nodes.append({
                'id': f'node_{i}',
                'title': f'Test Node {i+1}',
                'path': f'test/node_{i+1}.md',
                'type': 'Note'
            })
        
        edges = []
        for i in range(min(edge_count, node_count - 1)):
            edges.append({
                'source': f'node_{i}',
                'target': f'node_{i+1}',
                'relationship': 'links_to'
            })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'stats': {
                'node_count': node_count,
                'edge_count': edge_count
            }
        }
    
    @staticmethod
    def create_qa_result(question: str, success: bool = True) -> Dict[str, Any]:
        """Create a mock Q&A result"""
        if success:
            return {
                'question': question,
                'answer': f'Mock answer for: {question}',
                'context': 'Mock context from search results',
                'success': True
            }
        else:
            return {
                'question': question,
                'answer': 'No answer could be generated',
                'context': 'No relevant context found',
                'success': False
            }


class TestEnvironmentManager:
    """Manages test environments and cleanup"""
    
    def __init__(self):
        self.temp_dirs: List[Path] = []
    
    def create_test_environment(self, notes: Optional[List[TestNote]] = None) -> Path:
        """Create a test environment with optional custom notes"""
        if notes is None:
            notes = TestContentFactory.create_planet_notes()
        
        test_dir = TestContentFactory.create_test_vault(notes)
        self.temp_dirs.append(test_dir)
        return test_dir
    
    def cleanup(self):
        """Clean up all created test environments"""
        for temp_dir in self.temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        self.temp_dirs.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


# Convenience functions for common test scenarios

def with_planet_vault():
    """Decorator/context manager for tests that need a planet test vault"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with TestEnvironmentManager() as env:
                vault_dir = env.create_test_environment(TestContentFactory.create_planet_notes())
                return func(vault_dir, *args, **kwargs)
        return wrapper
    return decorator


def with_standard_vault():
    """Decorator/context manager for tests that need a full test vault"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with TestEnvironmentManager() as env:
                all_notes = (
                    TestContentFactory.create_planet_notes() +
                    TestContentFactory.create_health_fitness_notes() +
                    TestContentFactory.create_project_notes()
                )
                vault_dir = env.create_test_environment(all_notes)
                return func(vault_dir, *args, **kwargs)
        return wrapper
    return decorator