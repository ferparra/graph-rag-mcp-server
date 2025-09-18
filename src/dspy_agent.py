"""
DSPy ReAct Agent for Complex Multi-hop Graph Queries

This module implements a sophisticated ReAct agent that can perform multi-hop reasoning
through the knowledge graph to answer complex questions requiring multiple steps.
"""

from __future__ import annotations
import dspy
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import logging
import re
import inspect

# Support both package and module execution contexts
try:
    from unified_store import UnifiedStore
except ImportError:  # When imported as part of a package
    from .unified_store import UnifiedStore

logger = logging.getLogger(__name__)


class ComplexQA(dspy.Signature):
    """Answer a complex question with a comprehensive answer."""
    complex_question = dspy.InputField(desc="The multi-hop or complex question to answer")
    comprehensive_answer = dspy.OutputField(desc="A thorough, well-supported answer")


class VaultTools:
    """Tools for the ReAct agent to interact with the vault."""
    
    def __init__(self, unified_store: UnifiedStore):
        self.unified_store = unified_store
        self._search_cache: Dict[str, List[Dict]] = {}
        self._visited_nodes: Set[str] = set()
    
    def search_semantic(self, query: str, k: int = 5) -> List[str]:
        """Tool: Semantic search over vault content."""
        try:
            # Check cache first
            cache_key: str = f"semantic:{query}:{k}"
            if cache_key in self._search_cache:
                results = self._search_cache[cache_key]
            else:
                results = self.unified_store.query(query, k=k)
                self._search_cache[cache_key] = results
            
            # Format results for agent
            formatted_results = []
            for i, result in enumerate(results):
                title = result.get('meta', {}).get('title', 'Unknown')
                text = result.get('text', '')[:200]  # Truncate for readability
                chunk_id = result.get('id', '')
                
                formatted_results.append(
                    f"[{i+1}] {title} (chunk: {chunk_id})\n{text}..."
                )
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return [f"Error in semantic search: {str(e)}"]
    
    def traverse_graph(self, start_node: str, relationship_type: str = "all", depth: int = 2) -> List[str]:
        """Tool: Traverse graph relationships from a starting point."""
        try:
            # Try to find the starting node
            start_results = self.unified_store.query(start_node, k=1)
            if not start_results:
                return [f"Could not find starting node: {start_node}"]
            
            start_chunk_id = start_results[0].get('id', '')
            if not start_chunk_id:
                return [f"No chunk ID found for: {start_node}"]
            
            # Track visited nodes to avoid cycles
            if start_chunk_id in self._visited_nodes:
                return [f"Already explored {start_node} - avoiding cycle"]
            
            self._visited_nodes.add(start_chunk_id)
            
            # Get neighbors
            neighbors = self.unified_store.get_chunk_neighbors(
                start_chunk_id,
                include_sequential=(relationship_type in ["all", "sequential"]),
                include_hierarchical=(relationship_type in ["all", "hierarchical"])
            )
            
            # Format traversal results
            traversal_results = []
            traversal_results.append(f"Starting from: {start_node}")
            
            for neighbor in neighbors[:8]:  # Limit to prevent overwhelming
                chunk_id = neighbor.get("chunk_id", "")
                relationship = neighbor.get("relationship", "unknown")
                
                # Get chunk content
                try:
                    hydrated = self.unified_store.fetch_chunks([chunk_id], include_docs=True)
                    row = hydrated.get(chunk_id)
                    if row:
                        meta = row.get('meta') or {}
                        title = meta.get('title', 'Unknown')
                        header = meta.get('header_text', '')
                        doc = row.get('document') or ''
                        preview = doc[:150] if isinstance(doc, str) and doc else 'No content'

                        result_line: str = f"→ [{relationship}] {title}"
                        if header:
                            result_line += f" > {header}"
                        result_line += f"\n  {preview}..."

                        traversal_results.append(result_line)
                except Exception as e:
                    traversal_results.append(f"→ [{relationship}] Error accessing chunk {chunk_id}: {e}")
            
            if not neighbors:
                traversal_results.append("No graph connections found from this node")
            
            return traversal_results
            
        except Exception as e:
            logger.error(f"Graph traversal failed: {e}")
            return [f"Error in graph traversal: {str(e)}"]
    
    def extract_properties(self, note_id_or_title: str, property_type: str = "general") -> List[str]:
        """Tool: Extract specific properties from a note."""
        try:
            # Search for the note
            results = self.unified_store.query(note_id_or_title, k=3)
            if not results:
                return [f"Could not find note: {note_id_or_title}"]
            
            # Extract properties from the most relevant result
            best_result = results[0]
            text: str = best_result.get('text', '')
            meta: dict[str, Any] = best_result.get('meta', {})
            
            properties = []
            
            if property_type in ["general", "metadata"]:
                # Extract metadata properties
                title = meta.get('title', 'Unknown')
                tags = meta.get('tags', '').split(',') if meta.get('tags') else []
                
                properties.append(f"Title: {title}")
                if tags and tags[0]:
                    properties.append(f"Tags: {', '.join(tags)}")
                
                # Extract creation/modification info if available
                if meta.get('creation_date'):
                    properties.append(f"Created: {meta['creation_date']}")
            
            if property_type in ["general", "links"]:
                # Extract links
                links = meta.get('contains_links', '').split(',') if meta.get('contains_links') else []
                if links and links[0]:
                    properties.append(f"Links to: {', '.join(links[:5])}")  # Limit to 5
            
            if property_type in ["general", "content"]:
                # Extract key content patterns
                
                # Headers
                headers = re.findall(r'^#+\s+(.+)$', text, re.MULTILINE)
                if headers:
                    properties.append(f"Main sections: {', '.join(headers[:3])}")
                
                # Lists
                list_items = re.findall(r'^\s*[-*+]\s+(.+)$', text, re.MULTILINE)
                if list_items:
                    properties.append(f"Key points: {len(list_items)} items listed")
                
                # Dates mentioned
                dates = re.findall(r'\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{4}\b', text)
                if dates:
                    properties.append(f"Dates mentioned: {', '.join(dates[:3])}")
            
            if not properties:
                properties.append(f"Limited information available for {note_id_or_title}")
            
            return properties
            
        except Exception as e:
            logger.error(f"Property extraction failed: {e}")
            return [f"Error extracting properties: {str(e)}"]
    
    def expand_context(self, current_findings: str, focus_area: str) -> List[str]:
        """Tool: Expand context around current findings."""
        try:
            # Create a focused search query from current findings and focus area
            search_query: str = f"{focus_area} {current_findings}"
            
            # Semantic search for related content
            results = self.unified_store.query(search_query, k=4)
            
            expansions = []
            for result in results:
                title = result.get('meta', {}).get('title', 'Unknown')
                text = result.get('text', '')
                
                # Extract relevant snippets
                focus_words: list[str] = focus_area.lower().split()
                relevant_sentences = []
                
                for sentence in text.split('.'):
                    if any(word in sentence.lower() for word in focus_words):
                        relevant_sentences.append(sentence.strip())
                
                if relevant_sentences:
                    snippet = '. '.join(relevant_sentences[:2])
                    expansions.append(f"From {title}: {snippet}")
            
            if not expansions:
                expansions.append(f"No additional context found for: {focus_area}")
            
            return expansions
            
        except Exception as e:
            logger.error(f"Context expansion failed: {e}")
            return [f"Error expanding context: {str(e)}"]
    
    def find_connections(self, concept_a: str, concept_b: str) -> List[str]:
        """Tool: Find connections between two concepts."""
        try:
            # Search for content mentioning both concepts
            combined_query = f"{concept_a} {concept_b}"
            results = self.unified_store.query(combined_query, k=5)
            
            connections = []
            for result in results:
                title: str = result.get('meta', {}).get('title', 'Unknown')
                text: str = result.get('text', '')
                
                # Check if both concepts appear in the text
                text_lower: str = text.lower()
                if concept_a.lower() in text_lower and concept_b.lower() in text_lower:
                    # Extract sentences containing both concepts
                    sentences: list[str] = text.split('.')
                    relevant_sentences = []
                    
                    for sentence in sentences:
                        if (concept_a.lower() in sentence.lower() and 
                            concept_b.lower() in sentence.lower()):
                            relevant_sentences.append(sentence.strip())
                    
                    if relevant_sentences:
                        connection_text = '. '.join(relevant_sentences[:2])
                        connections.append(f"In {title}: {connection_text}")
            
            if not connections:
                connections.append(f"No direct connections found between {concept_a} and {concept_b}")
            
            return connections
            
        except Exception as e:
            logger.error(f"Connection finding failed: {e}")
            return [f"Error finding connections: {str(e)}"]
    
    def reset_exploration(self) -> None:
        """Reset exploration state for new query."""
        self._visited_nodes.clear()
        self._search_cache.clear()


class GraphReasoningAgent(dspy.Module):
    """ReAct agent for complex multi-hop reasoning through the knowledge graph."""
    
    def __init__(self, unified_store: UnifiedStore, state_dir: Optional[Path] = None):
        super().__init__()
        self.unified_store = unified_store
        self.tools = VaultTools(unified_store)
        
        # Initialize or load optimized agent
        if state_dir:
            agent_state = state_dir / "graph_reasoning_agent.json"
            if agent_state.exists():
                try:
                    self.react_agent = dspy.load(str(agent_state))
                    logger.info("Loaded optimized ReAct agent")
                except Exception as e:
                    logger.warning(f"Failed to load optimized agent: {e}")
                    self.react_agent = self._create_default_agent()
            else:
                self.react_agent = self._create_default_agent()
        else:
            self.react_agent = self._create_default_agent()
        
        # Tools registry for the agent
        self.tool_registry = {
            "search_semantic": self.tools.search_semantic,
            "traverse_graph": self.tools.traverse_graph,
            "extract_properties": self.tools.extract_properties,
            "expand_context": self.tools.expand_context,
            "find_connections": self.tools.find_connections
        }
    
    class _FallbackAgent(dspy.Module):
        def __init__(self, simple_answer_callable):
            super().__init__()
            self._simple_answer = simple_answer_callable
        def forward(self, complex_question: str) -> dspy.Prediction:
            return self._simple_answer(complex_question)

    def _create_default_agent(self) -> dspy.Module:
        """Create default ReAct agent with tools, or a fallback module."""
        try:
            return dspy.ReAct(
                signature=ComplexQA,
                tools=[
                    self.tools.search_semantic,
                    self.tools.traverse_graph,
                    self.tools.extract_properties,
                    self.tools.expand_context,
                    self.tools.find_connections
                ],
                max_iters=6
            )
        except Exception as e:
            logger.error(f"Failed to create ReAct agent: {e}")
            return GraphReasoningAgent._FallbackAgent(self._simple_answer)
    
    async def forward(self, question: str) -> dspy.Prediction:
        """Execute multi-hop reasoning for complex questions."""
        
        # Reset exploration state for new query
        self.tools.reset_exploration()
        
        # Classify question complexity
        complexity: str = self._assess_question_complexity(question)
        
        if complexity == "simple":
            # Use direct search for simple questions
            return self._simple_answer(question)
        
        try:
            # Use ReAct agent for complex reasoning
            logger.info(f"Using ReAct agent for complex question: {question}")
            maybe_coro = self.react_agent.forward(complex_question=question)
            raw_pred = await maybe_coro if inspect.isawaitable(maybe_coro) else maybe_coro
            
            # Normalize to a dspy.Prediction
            if isinstance(raw_pred, dspy.Prediction):
                pred = raw_pred
            else:
                answer = (getattr(raw_pred, 'comprehensive_answer', None) or
                          getattr(raw_pred, 'answer', None) or 
                          getattr(raw_pred, 'response', None) or
                          "The agent was unable to generate a complete answer.")
                pred = dspy.Prediction(comprehensive_answer=answer)
            
            # Add reasoning trace if available
            if hasattr(raw_pred, 'rationale') and not hasattr(pred, 'reasoning_trace'):
                pred.reasoning_trace = raw_pred.rationale
            
            pred.method = "react_agent"
            pred.complexity = complexity
            
            return pred
            
        except Exception as e:
            logger.error(f"ReAct agent failed: {e}")
            # Fallback to simple answer
            fallback: dspy.Prediction = self._simple_answer(question)
            fallback.error = f"ReAct agent error: {str(e)}"
            fallback.method = "fallback"
            return fallback
    
    def _assess_question_complexity(self, question: str) -> str:
        """Assess whether a question requires multi-hop reasoning."""
        
        # Keywords indicating complex reasoning
        complex_indicators = [
            "how", "why", "compare", "relationship", "connect", "between",
            "influence", "impact", "lead to", "result in", "because",
            "analyze", "explain the difference", "what's the connection",
            "trace", "follow", "sequence", "evolution", "development"
        ]
        
        # Keywords indicating simple lookup
        simple_indicators = [
            "what is", "define", "list", "show me", "find", "when was",
            "who is", "where is"
        ]
        
        question_lower: str = question.lower()
        
        # Count indicators
        complex_count: int = sum(1 for indicator in complex_indicators 
                           if indicator in question_lower)
        simple_count: int = sum(1 for indicator in simple_indicators 
                          if indicator in question_lower)
        
        # Additional complexity checks
        has_multiple_entities: bool = len(re.findall(r'\b[A-Z][a-z]+\b', question)) > 2
        has_multiple_questions: bool = '?' in question[:-1]  # Multiple question marks
        
        if complex_count > simple_count or has_multiple_entities or has_multiple_questions:
            return "complex"
        else:
            return "simple"
    
    def _simple_answer(self, question: str) -> dspy.Prediction:
        """Provide simple answer using direct search."""
        try:
            results: list[str] = self.tools.search_semantic(question, k=3)
            
            # Format simple answer
            if results and not results[0].startswith("Error"):
                answer: str = f"Based on the vault content:\n\n{chr(10).join(results)}"
            else:
                answer = "I couldn't find relevant information in the vault to answer this question."
            
            return dspy.Prediction(
                comprehensive_answer=answer,
                method="simple_search",
                complexity="simple"
            )
            
        except Exception as e:
            logger.error(f"Simple answer failed: {e}")
            return dspy.Prediction(
                comprehensive_answer=f"Error processing question: {str(e)}",
                method="error",
                complexity="unknown",
                error=str(e)
            )
    
    def save_optimized_state(self, state_dir: Path) -> None:
        """Save optimized agent state."""
        try:
            if self.react_agent:
                agent_state = state_dir / "graph_reasoning_agent.json"
                self.react_agent.save(str(agent_state), save_program=False)
                logger.info(f"Saved optimized ReAct agent to {agent_state}")
        except Exception as e:
            logger.error(f"Failed to save agent state: {e}")


class ComplexQueryHandler:
    """High-level handler for complex multi-hop queries."""
    
    def __init__(self, unified_store: UnifiedStore, state_dir: Optional[Path] = None):
        self.unified_store = unified_store
        self.state_dir = state_dir
        self.agent = GraphReasoningAgent(unified_store, state_dir)
        
        # Query patterns that benefit from ReAct reasoning
        self.complex_patterns = [
            r"how.*(?:relate|connect|influence|lead|cause)",
            r"what.*(?:relationship|connection|difference|similarity)",
            r"explain.*(?:why|how|relationship|connection)",
            r"compare.*(?:and|with|to|between)",
            r"trace.*(?:development|evolution|history|progression)",
            r"analyze.*(?:impact|effect|influence|relationship)"
        ]
    
    def should_use_agent(self, question: str) -> bool:
        """Determine if question should use ReAct agent."""
        question_lower: str = question.lower()
        
        # Check for complex patterns
        for pattern in self.complex_patterns:
            if re.search(pattern, question_lower):
                return True
        
        # Check for multiple entities or concepts
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', question)
        if len(entities) > 2:
            return True
        
        # Check for multi-part questions
        if question.count('?') > 1 or ' and ' in question_lower:
            return True
        
        return False
    
    async def handle_query(self, question: str) -> Dict[str, Any]:
        """Handle complex query with appropriate method."""
        
        use_agent: bool = self.should_use_agent(question)
        
        try:
            if use_agent:
                logger.info("Using ReAct agent for complex query")
                prediction: dspy.Prediction = await self.agent.forward(question)
            else:
                logger.info("Using simple search for straightforward query")
                prediction: dspy.Prediction = self.agent._simple_answer(question)
            
            return {
                "question": question,
                "answer": getattr(prediction, 'comprehensive_answer', 'No answer generated'),
                "method": getattr(prediction, 'method', 'unknown'),
                "complexity": getattr(prediction, 'complexity', 'unknown'),
                "reasoning_trace": getattr(prediction, 'reasoning_trace', None),
                "success": True,
                "used_agent": use_agent
            }
            
        except Exception as e:
            logger.error(f"Complex query handling failed: {e}")
            return {
                "question": question,
                "answer": f"Error processing complex query: {str(e)}",
                "method": "error",
                "success": False,
                "used_agent": use_agent,
                "error": str(e)
            }